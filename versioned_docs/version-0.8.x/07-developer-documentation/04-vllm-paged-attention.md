---
title: vLLM  分页注意力
---


* 目前，vLLM 使用自己的多头查询注意力内核 (multi-head query attention kernel) 实现（位于`csrc/attention/attention_kernels.cu`）。该内核旨在兼容 vLLM 的分页键值缓存，其中键和值缓存存储在不同的块中（注意，这里的块概念与 GPU 线程块不同。因此，在后续文档中，我们将把 vLLM 分页注意力块称为「块 (block)」，而把 GPU 线程块称为「线程块(thread block)」）。

* 为了实现高性能，该内核依赖于专门设计的内存布局和访问方法，特别是在线程从全局内存读取数据到共享内存时。本文档的目的是逐步提供内核实现的高层次解释，以帮助那些希望了解 vLLM 多头查询注意力内核的人。在阅读完本文档后，用户能够更好地理解，并更容易跟进实际的实现。

* 请注意，本文件可能不会涵盖所有细节，例如如何计算相应数据的正确索引或点乘实现。不过，在阅读完本文档并熟悉高层次的逻辑流程后，您应该能够更容易阅读实际代码并理解细节。


## 输入

* 内核函数接收当前线程的参数列表来执行其分配的工作。其中最重要的 3 个参数是输入指针 `q`、 `k_cache` 和 `v_cache` ，它们分别指向全局内存上需要读取和处理的查询、键和值数据。输出指针 `out` 指向应将结果写入的全局内存。这 4 个指针实际上指向的是多维数组，但每个线程只访问分配给它的那部分数据。为了简化说明，我们在这里省略了所有其他运行时参数。

```plain
    template<
    typename scalar_t,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS,
    int PARTITION_SIZE = 0>
    __device__ void paged_attention_kernel(
    ... // Other side args. // 其他副参数
    const scalar_t* __restrict__ out,       // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
    ... // Other side args.
    )
```
* 在编译时确定的函数签名上方还有一个模板参数列表。 `scalar_t` 表示查询、键和值数据元素的数据类型，例如 FP16。 `HEAD_SIZE` 表示每个头中的元素数量。 `BLOCK_SIZE` 是指每个块中的 token 数量。 `NUM_THREADS` 表示每个线程块中的线程数。  `PARTITION_SIZE` 表示张量并行 GPU 的数量（为简单起见，我们假设这个值为 0 且禁用张量并行）。

* 我们需要借助这些参数进行一系列的准备工作。包括计算当前的头索引、块索引和其他必要的变量。不过，目前我们可以跳过这些准备工作，直接进行实际的计算。一旦我们掌握了整个流程，就会更容易理解这些准备工作。


## 概念

* 在我们深入讨论计算流程之前，我想先介绍一下后续部分需要理解的一些概念。如果您遇到任何难以理解的术语，可以选择跳过本节，稍后再回来查看。

* **序列** **(****Sequence****)**：序列代表了客户端的请求。例如，由 `q` 指向的数据具有 `[num_seqs，num_heads，head_size]` 的形状。这表示 `q` 指向总共 `num_seqs` 个查询序列数据。由于该内核是 1 个单查询注意力内核，每个序列仅包含一个查询 token 。因此，`num_seqs` 等于批次中处理的 token 总数。

* **上下文****(Context)：**上下文由序列中生成的 token 组成。例如，`["What" ,  "is" ,  "your"]` 是上下文 token，输入查询 token 是 `"name"`。该模型可能会生成 token `?`。

* **向量 (****Vec****)：**Vec 是指一起获取和计算的一组元素。对于查询和键数据，确定 vec 大小 （`VEC_SIZE`），以便每个线程组可以一次性获取和计算 16 字节的数据。对于值数据，确定 vec 大小 (`V_VEC_SIZE`)，以便每个线程可以一次性获取和计算 16 字节的数据。例如，如果 `scalar_t` 为 FP16（2 字节） 且 `THREAD_GROUP_SIZE` 为 2，则 `VEC_SIZE` 将为 4，而 `V_VEC_SIZE` 将为 8。

* **线程组** **(****Thread group****)：**线程组是由一定数量的线程 (`THREAD_GROUP_SIZE`) 组成的小组，它们一次获取和计算 1 个查询标记和 1 个键标记。每个线程仅处理 token 数据的一部分。一个线程组处理的元素总数称为 `x`。例如，如果线程组包含 2 个线程，头大小为 8，则线程 0 处理索引为 0、2、4、6 处的查询和键元素，而线程 1 处理索引为 1、3、5、7。

* **块****(Block)：**vLLM 中的键和值缓存数据被分分割成块。每个块在一个头中存储固定数量 (`BLOCK_SIZE`) 的 token 的数据。每个块可能仅包含整个上下文 token 的一部分。例如，如果块大小为 16，头大小为 128，那么对于 1 个头，1 个块可以存储 16 * 128 = 2048 个元素。

* **Warp**：Warp 是一组 32 个线程 (`WARP_SIZE`)，它们在流多处理器 (SM) 上同时执行。在这个内核中，每个 warp 同时处理一个查询 token 与一个完整块的键 token 之间的计算（它可以在多次迭代中处理多个块）。例如，如果有 4 个 warp 和 6 个块用于一个上下文，那么分配方式将是：warp 0 处理第 0 和第 4 块，warp 1 处理第 1 和第 5 块，warp 2 处理第 2 个块，warp 3 处理第 3 个块。

* **线程块** **(****Thread block****)：**线程块是一组可以访问同一共享内存的线程 (`NUM_THREADS`)。每个线程块包含多个 warp (NUM_WARPS)，在这个内核中，每个线程块处理一个查询 token 和整个上下文的键 token 之间的计算。

* **网格****(Grid)：**网格是线程块的集合，并定义了集合的形状。在此内核中，形状为 `(num_heads, num_seqs, max_num_partitions)`。因此，每个线程块只处理 1 个头、1 个序列、1 个分区的计算。


## 询问 (Query)

* 本节将介绍查询数据如何存储在内存中以及如何被每个线程获取。如上所述，每个线程组获取一个查询 token 数据，而每个线程本身仅处理一个查询 token 数据的一部分。在每个 warp 中，每个线程组将获取相同的查询 token 数据，但会将其与不同的键 token 数据相乘。

```plain
    const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
```
![图片](/img/docs/07-04/query.png)
1 个头中的 1 个 token 的查询数据


* 每个线程定义自己的 `q_ptr`，它指向全局内存中分配的查询 token 数据。例如，如果 `VEC_SIZE` 为 4 且 `HEAD_SIZE` 为 128，则 `q_ptr` 指向包含总共 128 个元素的数据，这些元素被分为 128 / 4 = 32 个向量。

![图片](/img/docs/07-04/q_vecs.png)
1 个线程组中的 `q_vecs`

```plain
    __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
```


* 接下来，我们需要将`q_ptr`指向的全局内存数据读取到共享内存中作为 `q_vecs`。需要注意的是，每个 vec 都被分配到不同的行。例如，如果 `THREAD_GROUP_SIZE` 为 2，则线程 0 将处理第 0 行 vec，而线程 1 处理第 1 行 vec。通过这种方式读取查询数据，相邻线程（如线程 0 和线程 1）可以读取相邻内存，实现内存合并以提高性能。


## 键 (Key)

* 与「查询 (Query)」部分类似，本部分将介绍内存布局和键的分配。虽然每个线程组在一个内核运行时仅处理一个查询 token，但它可以在多次迭代中处理多个键 token。同时，每个 warp 将在多次迭代中处理多个键 token 块，确保内核运行结束后所有上下文 token 都被整个线程组处理。在本文中，「处理 (handle)」是指执行查询数据和键数据之间的点乘运算。

```plain
    const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                        + kv_head_idx * kv_head_stride
                        + physical_block_offset * x;
```
* 与 `q_ptr` 不同，每个线程中的 `k_ptr` 将在不同迭代中指向不同的键 token 。如上所示，`k_ptr` 基于 `k_cache` 的键 token 数据指向分配块、分配头和分配 token 处。

![图片](/img/docs/07-04/key.png)
1 个头中的所有上下文 token 的键数据


* 上图的图表展示了键数据的内存布局。假设 `BLOCK_SIZE` 为16，`HEAD_SIZE` 为128，`x` 为8，`THREAD_GROUP_SIZE` 为 2，总共有 4 个 warp。每个矩形代表一个头部的一个键 token 的所有元素，这些元素将由 1 个线程组处理。左半部分显示了 warp 0 的总共 16 个块的键 token 数据，而右半部分表示其他 warp 或迭代的剩余键 token 数据。每个矩形内部共有 32 个 vec（一个 token 的 128 个元素），将分别由 2 个线程（一个线程组） 处理。

![图片](/img/docs/07-04/k_vecs.png)
1 个线程组中的 `k_vecs`

```plain
    K_vec k_vecs[NUM_VECS_PER_THREAD]
```
* 接下来，我们需要从 `k_ptr` 读取键 token 数据并将它们存储在寄存器内存中作为 `k_vecs`。我们使用寄存器内存来存储 `k_vecs` ，因为它只能被 1 个线程访问 1 次，而 `q_vecs` 将被多个线程多次访问。每个 `k_vecs` 将包含多个向量以供后续计算。每个 vec 将在每次内部迭代时设置。 vec 的分配允许 warp 中的相邻线程一起读取相邻内存，这再次促进了内存合并。例如，线程 0 将读取 vec 0，而线程 1 将读取 vec 1。在下一个内部循环中，线程 0 将读取 vec 2，而线程 1 将读取 vec 3，依此类推。

* 您可能对整体流程仍然有点困惑。别担心，请继续阅读下一节「QK」部分。它将以更清晰、更高级的方式说明查询和键计算流程。

## QK

* 如下面的伪代码所示，在整个 for 循环块之前，我们获取一个 token 的查询数据并将其存储在 `q_vecs` 中。然后，在外部 for 循环中，我们迭代指向不同 token 的不同 `k_ptrs` ，并在内部 for 循环中准备 `k_vecs` 。最后，我们在 `q_vecs` 和每个 `k_vecs` 之间执行点乘运算。

```plain
    q_vecs = ...
    for ... {
       k_ptr = ...
       for ... {
          k_vecs[i] = ...
       }
       ...
       float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
    }
```
* 如前所述，对于每个线程，它一次仅获取部分查询和键 token 数据。然而，在`Qk_dot<>::dot` 中会发生跨线程组的归约操作。因此，这里返回的 `qk` 不仅仅是部分查询和键 token 之间点乘的结果，实际上是整个查询和键 token 数据之间的完整结果。

* 例如，如果 `HEAD_SIZE` 的值为 128，`THREAD_GROUP_SIZE` 为 2，则每个线程的 `k_vecs` 将包含总共 64 个元素。然而，返回的 qk 实际上是 128 个查询元素和 128 个键元素之间点乘的结果。如果你想了解更多关于点乘和归约的细节，可以参考 `Qk_dot<>::dot` 的实现。不过，为了简单起见，我们不会在本文档中介绍这些内容。

## Softmax

* 接下来，我们需要计算所有 `qk` 的归一化 softmax，如上所示，其中每个 $$x$$ 代表一个`qk`。为此，我们必须获得所有 `qk` 的 `qk_max`(m(x)) 和 `exp_sum` ($$\ell(x)$$) 的归约值。归约操作需要在整个线程块上执行，包括查询 token 和所有上下文键 token 之间的结果。

$$\begin{gather*}      m(x):=\max _i \quad x_i \\ \quad f(x):=\left[\begin{array}{lll}e^{x_1-m(x)} & \ldots & e^{x_B-m(x)}\end{array}\right]\\ \quad \ell(x):=\sum_i f(x)_i \\      \quad \operatorname{softmax}(x):=\frac{f(x)}{\ell(x)}      \end{gather*}$$

### `qk_max` 和 `logits`

* 得到 `qk` 结果后，我们可以用 `qk` 设置临时 `logits` 结果 （最后，`logits` 应该存储归一化的 softmax 结果）。同时，我们还可以比较并收集当前线程组计算的所有 `qk` 的 `qk_max`。

```plain
    if (thread_group_offset == 0) {
       const bool mask = token_idx >= context_len;
       logits[token_idx - start_token_idx] = mask ? 0.f : qk;
       qk_max = mask ? qk_max : fmaxf(qk_max, qk);
    }
```
* 请注意，这里的 `logits` 存储在共享内存上，因此每个线程组将为其分配的上下文 token 设置字段。总的来说，logits 的大小应该是上下文 token 的数量。

```python
    for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
        qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
    }


    if (lane == 0) {
       red_smem[warp_idx] = qk_max;
    }
```
* 然后我们需要获得每个 warp 上减少的 `qk_max` 。主要思想是让 warp 中的线程相互通信并获得最终的 `qk` 最大值 。

```plain
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
        qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
    }
    qk_max = VLLM_SHFL_SYNC(qk_max, 0);
```
* 最后，我们可以通过比较该线程块中所有 warp 的 `qk_max` ，来得到整个线程块的归约 `qk_max` ，然后将最终结果广播到每个线程。


### `exp_sum`

* 与 `qk_max` 类似，我们也需要从整个线程块中获取归约后的总和值。

```plain
    for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
        float val = __expf(logits[i] - qk_max);
        logits[i] = val;
        exp_sum += val;
    }
    ...
    exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);
```
* 首先，对每个线程组的所有 exp 值求和，同时将 `logits` 的每个条目从 `qk` 转换为 `exp(qk - qk_max)`。请注意，这里的 `qk_max` 已经是整个线程块的最大 `qk` 。然后我们可以像对 `qk_max` 一样在整个线程块上归约 `exp_sum` 。

```plain
    const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
    for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
       logits[i] *= inv_sum;
    }
```
* 最后，通过归约后的 `qk_max` 和 `exp_sum`，我们可以获得最终的归一化 softmax 结果，即 `logits`。这个 `logits` 变量将在后续步骤中用于与值数据进行点乘运算。现在，它应该存储所有分配的上下文 token 的 `qk` 的归一化 softmax 结果。


## 值

![图片](/img/docs/07-04/value.png)

1 个头中全部上下文 token 的值数据

![图片](/img/docs/07-04/logits_vec.png)

一个线程中的 `logits_vec`

![图片](/img/docs/07-04/v_vec.png)

一个线程中的 `v_vec` 列表


* 现在我们需要检索值数据并与 `logits` 进行点乘运算。与查询和键不同，值数据没有线程组的概念。如图所示，与键 token 的内存布局不同，同一列的元素对应于相同的值 token。对于 1 个值数据块，有 `HEAD_SIZE` 行，有 `BLOCK_SIZE` 列，它们被分割成多个 `v_vec`。

* 每个线程一次始终从相同的 `V_VEC_SIZE` 个 token 中获取 `V_VEC_SIZE` 个元素。因此，单个线程通过多次内部迭代，从不同的行和相同的列中检索多个 `v_vec`。对于每个 `v_vec`，它需要与相应的 `logits_vec` 进行点乘， `logits_vec` 也是来自 `logits` 的 `V_VEC_SIZE` 个元素。总体而言，通过多次内部迭代，每个 warp 将处理一个块的值 token；而通过多次外部迭代，整个上下文的值 token 将被处理。

```plain
    float accs[NUM_ROWS_PER_THREAD];
    for ... { // Iteration over different blocks. // 在不同块上迭代
        logits_vec = ...
        for ... { // Iteration over different rows. // 在不同行上迭代
            v_vec = ...
            ...
            accs[i] += dot(logits_vec, v_vec);
        }
    }
```
* 如上所示的伪代码中，在外部循环中，`logits_vec` 类似于 `k_ptr`，遍历不同的块并从 `logits` 中读取 `V_VEC_SIZE` 个元素。在内部循环中，每个线程从相同的 token 中读取 `V_VEC_SIZE` 个元素作为 `v_vec` 并进行点乘运算。需要注意的是，在每次内部迭代中，线程为相同的 token 获取不同头位置的元素。点乘结果随后被累加到 `accs` 中。因此，`accs` 的每个条目映射到当前线程分配的头位置。

* 例如，如果 `BLOCK_SIZE` 为 16，`V_VEC_SIZE` 为 8，则每个线程一次为 8 个 token 获取 8 个值元素。每个元素都来自同一头位置的不同 token。如果 `HEAD_SIZE` 为 128 且  `WARP_SIZE` 为 32，则对于每次内部循环，1 个 warp 需要获取 `WARP_SIZE * V_VEC_SIZE = 256` 个元素。这意味着 1 个 warp 总共需要 128 * 16 / 256 = 8 次内部迭代来处理整个块的值 token。并且每个线程中的每个 accs 包含 8 个元素，这些元素累积在 8 个不同的头位置。对于线程 0，`accs` 变量将有 8 个元素，它们是从所有分配的 8 个 token 中累积的值头的第 0 个、第 32 个……第 224 个元素。

## LV

* 现在，我们需要在每个 warp 内对 `accs` 进行归约。这一过程允许每个线程为一个块中所有 token 的指定头位置累积 `accs` 。

```plain
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
       float acc = accs[i];
       for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
          acc += VLLM_SHFL_XOR_SYNC(acc, mask);
       }
       accs[i] = acc;
    }
```
* 接下来，我们对所有 warp 执行 `accs` 归约，允许每个线程为所有上下文 token 的指定头位置积累 `accs` 。请注意，每个线程中的每个 `accs` 仅存储所有上下文 token 的整个头的部分元素的累加。然而，总体而言，所有输出结果都已经计算完毕，只是存储在不同的线程寄存器内存中。

```plain
    float* out_smem = reinterpret_cast<float*>(shared_mem);
    for (int i = NUM_WARPS; i > 1; i /= 2) {
        // Upper warps write to shared memory.
        // 上面的 warp 写入共享内存
        ...
            float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
            for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
                    ...
            dst[row_idx] = accs[i];
        }


        // Lower warps update the output.
        // 底下的 warp 更新输出
            const float* src = &out_smem[warp_idx * HEAD_SIZE];
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
                    ...
            accs[i] += src[row_idx];
        }


            // Write out the accs.
            // 写出 accs
    }
```


## 输出

* 我们可以将从局部寄存器内存中计算出的所有结果写入最终的输出全局内存。

```plain
    scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                    + head_idx * max_num_partitions * HEAD_SIZE
                    + partition_idx * HEAD_SIZE;
```
* 首先，我们需要定义 `out_ptr` 变量，它指向分配序列和分配头的起始地址。

```plain
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
    if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
    }
    }
```
* 最后，我们需要遍历不同分配的头位置，并根据 `out_ptr` 写出相应的累积结果。
