---
title: 引擎参数
---


您可以在下面找到 vLLM 每个引擎参数的解释：

```plain
usage: vllm serve [-h] [--model MODEL] [--tokenizer TOKENIZER]
                  [--skip-tokenizer-init] [--revision REVISION]
                  [--code-revision CODE_REVISION]
                  [--tokenizer-revision TOKENIZER_REVISION]
                  [--tokenizer-mode {auto,slow,mistral}] [--trust-remote-code]
                  [--download-dir DOWNLOAD_DIR]
                  [--load-format {auto,pt,safetensors,npcache,dummy,tensorizer,sharded_state,gguf,bitsandbytes,mistral}]
                  [--config-format {auto,hf,mistral}]
                  [--dtype {auto,half,float16,bfloat16,float,float32}]
                  [--kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}]
                  [--quantization-param-path QUANTIZATION_PARAM_PATH]
                  [--max-model-len MAX_MODEL_LEN]
                  [--guided-decoding-backend {outlines,lm-format-enforcer}]
                  [--distributed-executor-backend {ray,mp}] [--worker-use-ray]
                  [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE]
                  [--tensor-parallel-size TENSOR_PARALLEL_SIZE]
                  [--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS]
                  [--ray-workers-use-nsight] [--block-size {8,16,32}]
                  [--enable-prefix-caching] [--disable-sliding-window]
                  [--use-v2-block-manager]
                  [--num-lookahead-slots NUM_LOOKAHEAD_SLOTS] [--seed SEED]
                  [--swap-space SWAP_SPACE] [--cpu-offload-gb CPU_OFFLOAD_GB]
                  [--gpu-memory-utilization GPU_MEMORY_UTILIZATION]
                  [--num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE]
                  [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS]
                  [--max-num-seqs MAX_NUM_SEQS] [--max-logprobs MAX_LOGPROBS]
                  [--disable-log-stats]
                  [--quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,experts_int8,neuron_quant,None}]
                  [--rope-scaling ROPE_SCALING] [--rope-theta ROPE_THETA]
                  [--enforce-eager]
                  [--max-context-len-to-capture MAX_CONTEXT_LEN_TO_CAPTURE]
                  [--max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE]
                  [--disable-custom-all-reduce]
                  [--tokenizer-pool-size TOKENIZER_POOL_SIZE]
                  [--tokenizer-pool-type TOKENIZER_POOL_TYPE]
                  [--tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG]
                  [--limit-mm-per-prompt LIMIT_MM_PER_PROMPT] [--enable-lora]
                  [--max-loras MAX_LORAS] [--max-lora-rank MAX_LORA_RANK]
                  [--lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE]
                  [--lora-dtype {auto,float16,bfloat16,float32}]
                  [--long-lora-scaling-factors LONG_LORA_SCALING_FACTORS]
                  [--max-cpu-loras MAX_CPU_LORAS] [--fully-sharded-loras]
                  [--enable-prompt-adapter]
                  [--max-prompt-adapters MAX_PROMPT_ADAPTERS]
                  [--max-prompt-adapter-token MAX_PROMPT_ADAPTER_TOKEN]
                  [--device {auto,cuda,neuron,cpu,openvino,tpu,xpu}]
                  [--num-scheduler-steps NUM_SCHEDULER_STEPS]
                  [--scheduler-delay-factor SCHEDULER_DELAY_FACTOR]
                  [--enable-chunked-prefill [ENABLE_CHUNKED_PREFILL]]
                  [--speculative-model SPECULATIVE_MODEL]
                  [--speculative-model-quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,experts_int8,neuron_quant,None}]
                  [--num-speculative-tokens NUM_SPECULATIVE_TOKENS]
                  [--speculative-draft-tensor-parallel-size SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE]
                  [--speculative-max-model-len SPECULATIVE_MAX_MODEL_LEN]
                  [--speculative-disable-by-batch-size SPECULATIVE_DISABLE_BY_BATCH_SIZE]
                  [--ngram-prompt-lookup-max NGRAM_PROMPT_LOOKUP_MAX]
                  [--ngram-prompt-lookup-min NGRAM_PROMPT_LOOKUP_MIN]
                  [--spec-decoding-acceptance-method {rejection_sampler,typical_acceptance_sampler}]
                  [--typical-acceptance-sampler-posterior-threshold TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_THRESHOLD]
                  [--typical-acceptance-sampler-posterior-alpha TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_ALPHA]
                  [--disable-logprobs-during-spec-decoding [DISABLE_LOGPROBS_DURING_SPEC_DECODING]]
                  [--model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG]
                  [--ignore-patterns IGNORE_PATTERNS]
                  [--preemption-mode PREEMPTION_MODE]
                  [--served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]]
                  [--qlora-adapter-name-or-path QLORA_ADAPTER_NAME_OR_PATH]
                  [--otlp-traces-endpoint OTLP_TRACES_ENDPOINT]
                  [--collect-detailed-traces COLLECT_DETAILED_TRACES]
                  [--disable-async-output-proc]
                  [--override-neuron-config OVERRIDE_NEURON_CONFIG]
```


### 命名参数

`--model`  

要使用的 Huggingface 模型的名称或路径。

默认值: "facebook/opt-125m"


`--tokenizer`  

要使用的 Huggingface  tokenizer 的名称或路径。如果未指定，将使用模型名称或路径。


`--skip-tokenizer-init`  

跳过 tokenizer 和 detokenizer 的初始化。


`--revision`  

要使用的特定模型版本。它可以是分支名称、标签名称或提交 ID。如果未指定，将使用默认版本。


`--code-revision`  

在 Hugging Face Hub 上使用的模型代码的具体版本。它可以是分支名称、标签名称或提交 ID。如果未指定，将使用默认版本。


`--tokenizer-revision`  

使用的 Huggingface  tokenizer 的版本。它可以是分支名称、标签名称或提交 ID。如果未指定，将使用默认版本。


`--tokenizer-mode`  

可选项：auto, slow, mistral


tokenizer 模式。


「auto」将使用快速 tokenizer（如果可用）。

「slow」将始终使用慢速 tokenizer。

「mistral」将始终使用*mistral_common tokenizer*。


默认值：「auto」


`--trust-remote-code`  

信任来自 Huggingface 的远程代码。


`--download-dir`  

下载和加载权重的目录，默认为 Huggingface 的默认缓存目录。


`--load-format`  

可选项：auto、pt、safetensors、npcache、dummy、tensorizer、sharded_state、gguf、bitsandbytes、mistral


要加载的模型权重的格式。


* 「auto」将尝试以 safetensors 格式加载权重，如果 safetensors 格式不可用，则回退到 pytorch bin 格式。

* 「pt」将以 PyTorch bin 格式加载权重。

* 「safetensors」将以 safetensors 格式加载权重。

* 「npcache」将以 PyTorch 格式加载权重，并存储一个 NumPy 缓存以加快加载速度。

* 「dummy」将使用随机值初始化权重，主要用于性能分析。

* 「tensorizer」将使用 CoreWeave 的张量器加载权重。有关详细信息，请参阅示例部分中的 Tensorize vLLM 模型脚本。

* 「bitsandbytes」将使用 bitsandbytes 量化加载权重。


默认值："auto"


`--config-format`  

可选项：auto、hf、mistral


要加载的模型配置的格式。

"auto" 将尝试以 hf 格式加载配置（如果可用），否则它将尝试以 Mistra 格式加载


默认值: "auto"


`--dtype`  

可选项：auto、half、float16、bfloat16、float、float32


模型权重和激活的数据类型。

* 「auto」将对 FP32 和 FP16 模型使用 FP16 精度，对 BF16 模型使用 BF16 精度。

* 「half」表示 FP16。推荐用于 AWQ 量化。

* 「float16」与「half」相同。

* 「bfloat16」用于精度和范围之间的取得平衡。

* 「float」是 FP32 精度的简写。

* 「float32」表示 FP32 精度。


默认值："auto"


`--kv-cache-dtype`  

可选项：auto, fp8, fp8_e5m2, fp8_e4m3


kv 缓存存储的数据类型。如果为 "auto"，将使用模型的数据类型。 CUDA 11.8+ 支持 fp8 (=fp8_e4m3) 和 fp8_e5m2。 ROCm (AMD GPU) 支持 fp8 (=fp8_e4m3)。


默认值: "auto"


`--quantization-param-path`  

包含 KV 缓存缩放因子的 JSON 文件的路径。当 KV 缓存数据类型为 FP8 时，通常需要提供此路径。否则，KV 缓存缩放因子默认为 1.0，这可能会导致准确性问题。FP8_E5M2（无缩放）仅在高于 11.8 时的 cuda 版本上受支持。在 ROCm (AMD GPU) 上，通用推理标准支持 FP8_E4M3。


`--max-model-len`  

模型上下文长度。如果未指定，将自动从模型配置中派生。


`--guided-decoding-backend`  

可选项：outlines, lm-format-enforcer


默认情况下，用于引导式解码的引擎是哪一个（JSON 模式/正则表达式等）。目前支持 [outlines-dev/outlines](https://github.com/outlines-dev/outlines) 和[noamgat/lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)。此外，可以通过 guided_decoding_backend 参数覆盖每个请求。


默认值: "outlines"


`--distributed-executor-backend`  

可选项：ray, mp


用于分布式服务的后端。当使用超过 1 个 GPU 时，如果已安装 ray，将自动设置为 "ray"，否则将自动设置为 "mp"（多进程）。


`--worker-use-ray`  

已弃用，请使用 –distributed-executor-backend=ray。


`--pipeline-parallel-size, -pp`  

管道阶段数。


默认值: 1


`--tensor-parallel-size, -tp`  

张量并行副本的数量。


默认值: 1


`--max-parallel-loading-workers`  

分多个批次顺序加载模型，以避免使用张量并行和大型模型时出现 RAM OOM。


`--ray-workers-use-nsight`  

如果指定，请使用 nsight 来分析 Ray 进程。


`--block-size`  

可选项：8、16、32


连续 token 块的 token 块大小。在 neuron 设备上此参数被忽略并设置为最大模型长度 (max-model-len)。


默认值：16


`--enable-prefix-caching`  

启用自动前缀缓存。


`--disable-sliding-window`  

禁用滑动窗口，将其限制为滑动窗口大小。


`--use-v2-block-manager`  

使用 BlockSpaceMangerV2。


`--num-lookahead-slots`  

用于推测解码所需的实验调度配置。将来此配置会被推测性配置取代；在此之前，它的存在是为了启用正确性测试。


默认值：0


`--seed`  

用于操作的随机种子。


默认值：0


`--swap-space`  

每个 GPU 的 CPU 交换空间大小（单位为 GiB）。


默认值：4


`--cpu-offload-gb`  

每个 GPU 用于卸载到 CPU 的空间（以 GiB 为单位）。默认值为 0，表示不卸载。直观地说，这个参数可以看作是增加 GPU 内存大小的虚拟方式。例如，如果您有一个 24 GB GPU 并将其设置为 10，那么实际上您可以将其视为 34 GB 的 GPU。然后您可以加载一个 BF16 权重的 13B 模型，这需要至少 26GB 的 GPU 内存。请注意，这需要快速的 CPU-GPU 互连，因为在每次模型前向传播中，模型的一部分是从 CPU 内存动态加载到 GPU 内存中的。


默认值：0


`--gpu-memory-utilization`  

用于模型执行器的 GPU 内存比例，这个比例的范围为 0 到 1。例如，值 0.5 表示 GPU 内存利用率为 50%。如果未指定，将使用默认值 0.9。


默认值：0.9


`--num-gpu-blocks-override`  

如果指定，则忽略 GPU 分析结果并使用此数量的 GPU 块。用于测试抢占。


`--max-num-batched-tokens`  

每次迭代的最大批处理 token 数。


`--max-num-seqs`  

每次迭代的最大序列数。


默认值：256


`--max-logprobs`  

返回 logprobs 的最大日志概率数在 SamplingParams 中指定。


默认值：20


`--disable-log-stats`  

禁用日志记录统计。


`--quantization, -q`  

可选项：aqlm, awq, deepspeedfp, tpu_int8, fp8, fbgemm_fp8, modelopt, marlin, gguf, gptq_marlin_24, gptq_marlin, awq_marlin, gptq, compressed-tensors, bitsandbytes, qqq, experts_int8, neuron_quant, None


用于量化权重的方法。如果没有指定，我们首先检查模型配置文件中的 quantization_config 属性。如果为 None，我们假设模型权重未量化，并使用 dtype 来确定权重的数据类型。


`--rope-scaling`  

JSON 格式的 RoPE 扩展配置。例如，{"type":"dynamic","factor":2.0}。


`--rope-theta`  

相对位置编码的 theta 值，与*rope_scaling*一起使用。在某些情况下，更改 RoPE 的 theta 值可以提高缩放模型的性能。


`--enforce-eager`  

始终使用 eager 模式的 PyTorch。如果为 False，将混合使用 eager 模式和 CUDA 图性，以获得最大性能和灵活性。


`--max-context-len-to-capture`  

CUDA 图表覆盖的最大上下文长度。当序列的上下文长度大于此值时，我们会回退到 eager 模式。（已弃用。请使用 –max-seq-len-to-capture 代替）


`--max-seq-len-to-capture`  

CUDA 图所涵盖的最大序列长度。当序列的上下文长度大于此值时，我们会回退到 eager 模式。此外，对于编码器-解码器模型，如果编码器输入的序列长度大于此值，我们也会回退到 eager 模式。


默认值：8192


`--disable-custom-all-reduce`  

请参阅并行配置。


`--tokenizer-pool-size`  

用于异步分词的分词器池的大小。如果设置为0，则会使用同步分词。


默认值: 0


`--tokenizer-pool-type`  

用于异步分词的分词器池的类型。如果 tokenizer_pool_size 设置为 0，则此选项会被忽略。


默认值："ray"


`--tokenizer-pool-extra-config`  

tokenizer 池的额外配置。这应该是一个将被解析为字典的 JSON 字符串。如果 tokenizer_pool_size 设置为 0，则此选项会被忽略。


`--limit-mm-per-prompt`  

对于每个多模式插件，需要限制每个提示能够处理的输入实例数量。需要以逗号分隔的项目列表形式指定，例如：image=16,video=2 表示每个提示最多有 16 张图像和 2 个视频。默认每种模态的输入实例数量为 1。


`--mm-processor-kwargs` 

多模态输入映射/处理的覆盖设置，例如，图片处理器。例如：{"num_crops": 4}。


`--enable-lora`  

如果为 True，则启用 LoRA 适配器的处理。


`--max-loras`  

单个批次中 LoRAs 的最大数量。


默认值：1


`--max-lora-rank`  

最大 LoRA 秩。


默认值: 16


`--lora-extra-vocab-size`  

LoRA 适配器中可以存在的额外词汇表的最大尺寸（添加到基础模型词汇表中）。


默认值：256


`--lora-dtype`  

可选项：auto、float16、bfloat16、float32


LoRA 的数据类型。如果为自动，则默认为基本模型 dtype。


默认值："auto"


`--long-lora-scaling-factors`  

指定多个缩放因子（可以与基本模型的缩放因子不同--例如长 LoRA），以允许同时使用那些使用这些缩放因子训练的多个 LoRA 适配器。如果未指定，则仅允许使用基本模型缩放因子训练的适配器。


`--max-cpu-loras`  

CPU 内存中存储的 LoRA 的最大数量。必须 >= max_num_seqs。默认设置为 max_num_seqs。


`--fully-sharded-loras`  

默认情况下，只有一半的 LoRA 计算通过张量并行进行分片。启用此功能将使用完全分片的层。在高序列长度、最大秩或张量并行大小下，这样做可能会更快。


`--enable-prompt-adapter`  

如果为 True，则启用 PromptAdapter 的处理。


`--max-prompt-adapters`  

批次中 PromptAdapter 的最大数量。


默认值：1


`--max-prompt-adapter-token`  

PromptAdapters token 的最大数量。


默认值: 0


`--device`  

可选项：auto、cuda、neuron、cpu、openvino、tpu、xpu


vLLM 执行的设备类型。


默认值："auto"


`--num-scheduler-steps`  

每个调度程序调用的最大前进步骤数。


默认值：1


`--multi-step-stream-outputs`  

如果设置为 False，则多步操作将在所有步骤结束时流式传输输出。


默认值：True


`--scheduler-delay-factor`  

在安排下一个提示之前应用延迟（延迟因子乘以先前的提示延迟）。


默认值：0.0


`--enable-chunked-prefill`  

如果设置了该选项，预填充请求可以根据 max_num_batched_tokens 进行分块处理。


`--speculative-model`  

推测解码中使用的草稿模型的名称。


`--speculative-model-quantization`  

可选项：aqlm, awq, deepspeedfp, tpu_int8, fp8, fbgemm_fp8, modelopt, marlin, gguf, gptq_marlin_24, gptq_marlin, awq_marlin, gptq, compressed-tensors, bitsandbytes, qqq, experts_int8, neuron_quant, None


用于量化推测模型权重的方法。如果没有制定，我们首先检查模型配置文件中的 quantization_config 属性。如果为 None，我们假设模型权重未量化，并使用 dtype 来确定权重的数据类型。


`--num-speculative-tokens`  

在推测解码中，从草稿模型中采样的推测 token 的数量。


`--speculative-disable-mqa-scorer`  

如果设置为 True，MQA 评分器将在推测中被禁用并回退到批量扩展。


`--speculative-draft-tensor-parallel-size, -spec-draft-tp`  

推测解码中，草稿模型的张量并行副本数。


`--speculative-max-model-len`  

草稿模型支持的最大序列长度。超过这个长度的序列将跳过推测。


`--speculative-disable-by-batch-size`  

如果入队请求的数量大于此值，则禁用对新传入请求的推测解码。


`--ngram-prompt-lookup-max`  

推测解码中，ngram 提示查找的窗口最大大小。


`--ngram-prompt-lookup-min`  

推测解码中，ngram 提示查找的窗口最小大小。


`--spec-decoding-acceptance-method`  

可选项：rejection_sampler、typly_acceptance_sampler


指定在推测解码中草稿 token 验证期间要使用的接受方法。支持两种类型的接受例程：1）RejectionSampler，不允许更改草稿 token 的接受率；2）TypicalAcceptanceSampler，它是可配置的，允许以较低质量为代价获得更高的接受率，反之亦然。


默认值："rejection_sampler"


`--typical-acceptance-sampler-posterior-threshold`  

设置要接受的 token 后验概率的下限阈值。 TypicalAcceptanceSampler 使用此阈值在推测解码期间做出采样决策。默认为 0.09。


`--typical-acceptance-sampler-posterior-alpha`  

在 TypicalAcceptanceSampler 中，用于基于熵的令牌接受阈值的缩放因子。通常，默认值为 –typical-acceptance-sampler-posterior-threshold 的平方根，即 0.3。


`--disable-logprobs-during-spec-decoding`  

如果设置为 True，则在推测解码期间不会返回 token 对数概率。如果设置为 False，则根据 采样参数 (SamplingParams) 中的设置返回对数概率。如果未指定，则默认为 True。在推测解码期间禁用对数概率可以通过跳过提议采样、目标采样以及确定接受的 token 后的对数概率计算来减少延迟。


`--model-loader-extra-config`  

模型加载器的额外配置。这将被传递到与所选加载格式 (load_format) 相对应的模型加载器。这应该是一个将被解析为字典的 JSON 字符串。


`--ignore-patterns`  

加载模型时要忽略的模式。默认为 'original/**/**'* 以避免重复加载 llama 的检查点。


默认值：[]


`--preemption-mode`  

如果为 ‘recompute’，则引擎通过重新计算进行抢占；如果为“swap”，引擎通过块交换来执行抢占。


`--served-model-name`  

API 中使用的模型名称。如果提供了多个名称，服务器将响应任何提供的名称。响应模型字段中的模型名称将成为此列表中的第一个名称。如果未指定，模型名称将与「–model」参数相同。请注意，此名称也将用于 prometheus 指标的 model_name 标签内容中，如果提供了多个​​名称，指标标签 (metricstag) 将采用第一个名称。


`--qlora-adapter-name-or-path`  

QLoRA 适配器的名称或路径。


`--otlp-traces-endpoint`  

将 OpenTelemetry 跟踪信息发送到的目标 URL。


`--collect-detailed-traces`  

有效选择是 model、worker、all。只有在设置了「–otlp-traces-endpoint」时才有意义。如果设置了此参数，它将收集指定模块的详细跟踪。这涉及到使用可能成本高昂的和/或阻塞操作，因此可能会对性能产生影响。


`--disable-async-output-proc`  

禁用异步输出处理。这可能会导致性能降低。


`--override-neuron-config`  

覆盖或设置 neuron 设备配置。


`--scheduling-policy` 

可选项：fcfs（先来先服务）、priority（优先级）。


要使用的调度策略。「fcfs」（先来先服务，即请求按照到达的顺序进行处理；默认值）或「priority」（请求基于给定的优先级（值越小意味着越早处理）和到达时间来处理，以决定任何平局情况）。


默认值："fcfs"。


## 异步引擎参数

以下是与异步引擎相关的附加参数：

```plain
usage: vllm serve [-h] [--disable-log-requests]
```


### 命名参数

--`disable-log-requests`  

禁用日志请求。

