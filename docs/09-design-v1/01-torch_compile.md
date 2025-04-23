---

title: vLLM 的 `torch.compile` 集成

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


在 vLLM 的 V1 架构中，`torch.compile` 默认启用且是框架的关键组成部分。本文档通过一个简单示例展示如何理解 `torch.compile` 的使用方式。


在示例中，我们将使用 v1 架构运行一个常见的 Llama 模型，并开启调试级别日志以显示所有细节。使用的命令是 `VLLM_USE_V1=1 VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-1B`。


## 编译缓存

在详细日志中可见：

```plain
INFO 03-07 03:06:55 [backends.py:409] Using cache directory: ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0 for vLLM's torch.compile
```


vLLM 会考虑所有相关因素，并决定存储编译产物的目录。这意味着您可以直接复制整个 `~/.cache/vllm/torch_compile_cache` 目录到部署环境中，以节省大量编译时间，从而加速 vLLM 实例的启动。


考虑的因素包括：

* 所有相关配置（参见 [config.py](https://github.com/vllm-project/vllm/blob/main/vllm/config.py) 中的 `compute_hash` 函数）

* PyTorch 配置（参见 [compiler_interface.py](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/compiler_interface.py) 中的 `compute_hash` 函数）

* 模型的 forward 函数及其调用的相关函数（见下文）


综合考虑这些因素后，通常可以确保缓存是安全可用的，不会导致意外行为。因此，缓存默认启用。如果您需要调试编译过程，或怀疑缓存导致问题，可以通过设置环境变量 `VLLM_DISABLE_COMPILE_CACHE=1` 禁用缓存。


vLLM 的 `torch.compile` 集成的一个独特之处在于，我们确保所有编译在服务任何请求之前完成。不会有请求触发新的编译。否则，引擎会被该请求阻塞，响应时间会出现意外波动。


## Python 代码编译

在详细日志中，我们可以看到：

```plain
DEBUG 03-07 03:06:52 [decorators.py:203] Start compiling function <code object forward at 0x7f08acf40c90, file "xxx/vllm/model_executor/models/llama.py", line 339>


DEBUG 03-07 03:06:54 [backends.py:370] Traced files (to be considered for compilation cache):
DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/_dynamo/polyfills/builtins.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/container.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/module.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/attention/layer.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/communication_op.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/parallel_state.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/custom_op.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/activation.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/layernorm.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/linear.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/rotary_embedding.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/vocab_parallel_embedding.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/models/llama.py


DEBUG 03-07 03:07:07 [backends.py:462] Computation graph saved to ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py
DEBUG 03-07 03:07:07 [wrapper.py:105] Dynamo transformed code saved to ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py
```


这部分是关于 Python 代码编译，即 Dynamo 的图捕获。它尝试追踪代码位于 `xxx/vllm/model_executor/models/llama.py:339` 的函数，即我们编译的模型的 `forward` 函数。在前向传递过程中，Dynamo 还会内联调用其他函数，如日志所示，包括来自 `xxx/torch/nn/modules/module.py` 的一些 PyTorch 函数（由 PyTorch `nn.Module` 使用，因为模块属性访问会触发函数调用），以及来自 vLLM 的一些通信、注意力、激活函数。所有追踪的文件都会被用于决定缓存目录。这样，上述文件的任何代码更改都会触发编译缓存失效，从而重新编译。


Dynamo 编译的结果是一个新函数，存储在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py` 中。通常，此函数从模块中解包张量，然后将其传递给追踪的计算图。计算图存储在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py` 中。

## 计算图处理

计算图包含每个张量的形状注释。输入包括输入 ID、位置 ID、模型权重和缓冲区，输出是最终的隐藏状态。注意，lm 头投影和采样操作不在图中考虑。


计算图的大多数输入具有静态形状，因为它们是模型权重和缓冲区，在模型生命周期内不会改变。只有输入 ID 和位置 ID 具有符号形状，即形状可能随批次变化。但它们会共享相同的符号形状。也就是说，计算图中唯一变化的尺寸是批次大小（当前前向传递中处理的 token 数量）。


注意力操作很复杂，需要与 kv 缓存交互，形状复杂。幸运的是，注意力操作的输出形状与注意力操作的输入查询形状相同。因此，我们将整个注意力操作封装到一个 PyTorch 自定义操作 `torch.ops.vllm.unified_attention_with_output` 中，这样 Dynamo 就不会尝试检查任何内部操作。这样，尽管注意力操作很复杂，但从 Dynamo 的角度来看，我们仍然可以捕获完整的模型计算图。


计算图进一步被 `splitting_ops`（通常是注意力操作）分割成多个部分。因此，在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py` 文件中，我们可以看到许多子模块，每个子模块是分割后的一块图：

* 注意力操作本身是一个子模块。

* 从一次注意力操作到下一次注意力操作的计算图部分是一个子模块。


每个子模块可以通过其索引标识，并单独处理。

## 计算图编译

在详细日志中，我们还可以看到：

```plain
DEBUG 03-07 03:52:37 [backends.py:134] store the 0-th graph for shape None from inductor via handle ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
DEBUG 03-07 03:52:39 [backends.py:134] store the 1-th graph for shape None from inductor via handle ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
...
DEBUG 03-07 03:52:45 [backends.py:134] store the 15-th graph for shape None from inductor via handle ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
DEBUG 03-07 03:52:45 [backends.py:134] store the 16-th graph for shape None from inductor via handle ('fvj3ccoi7m34f3dnr4itmu55mmun44l5xymwhrjlwisylsk7q6jy', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/tf/ctfftkglj7b4lcttq5cymx6cew372uoauupqn6ldsvpiucavqcjc.py')
```


这意味着第一块计算图（符号形状为 `None`）由 Inductor 编译（键为 `fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw`）。编译后的内核存储在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py` 中。您可以打开该文件查看 Inductor 最终运行的代码。


另一个细节是：您可以看到第 1 个图和第 15 个图具有相同的键，而第 0 个图和第 16 个图不同。这是预期的，因为我们通过注意力操作分割图，得到 3 个独特的子图：

* 注意力操作前的第一层

* 中间层，从一次注意力操作到下一次注意力操作

* 注意力操作后的最后一层


如果我们已经有缓存目录（例如第二次运行相同的代码），我们将看到以下日志：

```plain
DEBUG 03-07 04:00:45 [backends.py:86] Directly load the 0-th graph for shape None from inductor via handle ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
```


这次，Inductor 编译完全被绕过，我们将从磁盘加载上次的编译产物。


上述示例仅使用 Inductor 编译通用形状（即符号形状）。我们还可以使用 Inductor 编译某些特定形状，例如：

`VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.2-1B --compilation_config "{'compile_sizes': [1, 2, 4, 8]}"`


然后它还会为批次大小 `1, 2, 4, 8` 编译特定内核。此时，计算图中的所有形状都是静态已知的，我们将启用自动调优以追求最大性能。第一次运行时可能很慢，但下次运行时，我们可以直接绕过调优并运行调优后的内核。


当所有形状已知时，`torch.compile` 可以比较不同配置，并通常找到更好的配置来运行内核。例如，我们可以看到以下日志：

```plain
AUTOTUNE mm(8x2048, 2048x3072)
  triton_mm_4 0.0130 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
  triton_mm_8 0.0134 ms 97.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
  triton_mm_12 0.0148 ms 87.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=4, num_warps=4
  mm 0.0160 ms 81.6% 
  triton_mm_16 0.0165 ms 78.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=8
  triton_mm_3 0.0199 ms 65.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
  triton_mm_1 0.0203 ms 64.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=2, num_warps=2
  triton_mm_7 0.0203 ms 64.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
  triton_mm_2 0.0208 ms 62.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
  triton_mm_11 0.0215 ms 60.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
SingleProcess AUTOTUNE benchmarking takes 2.0428 seconds and 7.5727 seconds precompiling
```


这意味着，对于形状为 `8x2048x3072` 的矩阵乘法，`torch.compile` 尝试了多种配置的 triton 模板，比默认代码（分派到 cublas 库）快得多。


遗憾的是，由于自动调优耗时较长（从几秒到几分钟，取决于模型大小和批次大小），尽管可以缓存供后续使用，但为了用户体验，我们默认关闭此功能。如果您需要最高性能，建议通过编译特定形状来尝试。


## Cudagraph 捕获

vLLM 的 V1 架构使用分段 cudagraph。如前所述，完整的计算图被分割，我们仅捕获注意力操作之间的图块（包括任何注意力操作前的第一个图和所有注意力操作后的最后一个图）。这是基于一个常见观察：注意力操作之间的计算通常是 token 级别的，易于 cudagraph 处理；而注意力操作本身难以兼容 cudagraph。因此，通过在 eager 模式下运行注意力操作，其余操作在 cudagraph 中运行，我们保持了注意力操作的灵活性。


分段 cudagraph 还具有细粒度的内存管理。目的是仅将注意力内核排除在 cudagraph 之外，同时将所有其他模块和内存分配操作保留在 cudagraph 中。这就是为什么 V1 中的注意力操作的输出张量是其输入的一部分。


cudagraph 由编译器后端捕获和管理，并在批次大小匹配时重放。模型的调用者（模型运行器）只需确保正确管理输入缓冲区。所有中间缓冲区由编译器后端自动管理。


默认情况下，vLLM 会尝试确定一组尺寸来捕获 cudagraph。您也可以通过配置 `cudagraph_capture_sizes` 覆盖：

`VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.2-1B --compilation_config "{'cudagraph_capture_sizes': [1, 2, 4, 8]}"`


然后它将仅为指定尺寸捕获 cudagraph。这对于精细控制 cudagraph 捕获非常有用。

