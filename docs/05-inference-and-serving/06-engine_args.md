---

title: 引擎参数

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


可以在下述找到对每一个 vLLM 引擎参数的说明：

```plain
usage: vllm serve [-h] [--model MODEL]
                  [--task {auto,generate,embedding,embed,classify,score,reward,transcription}]
                  [--tokenizer TOKENIZER] [--hf-config-path HF_CONFIG_PATH]
                  [--skip-tokenizer-init] [--revision REVISION]
                  [--code-revision CODE_REVISION]
                  [--tokenizer-revision TOKENIZER_REVISION]
                  [--tokenizer-mode {auto,slow,mistral,custom}]
                  [--trust-remote-code]
                  [--allowed-local-media-path ALLOWED_LOCAL_MEDIA_PATH]
                  [--download-dir DOWNLOAD_DIR]
                  [--load-format {auto,pt,safetensors,npcache,dummy,tensorizer,sharded_state,gguf,bitsandbytes,mistral,runai_streamer}]
                  [--config-format {auto,hf,mistral}]
                  [--dtype {auto,half,float16,bfloat16,float,float32}]
                  [--kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}]
                  [--max-model-len MAX_MODEL_LEN]
                  [--guided-decoding-backend GUIDED_DECODING_BACKEND]
                  [--logits-processor-pattern LOGITS_PROCESSOR_PATTERN]
                  [--model-impl {auto,vllm,transformers}]
                  [--distributed-executor-backend {ray,mp,uni,external_launcher}]
                  [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE]
                  [--tensor-parallel-size TENSOR_PARALLEL_SIZE]
                  [--enable-expert-parallel]
                  [--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS]
                  [--ray-workers-use-nsight] [--block-size {8,16,32,64,128}]
                  [--enable-prefix-caching | --no-enable-prefix-caching]
                  [--disable-sliding-window] [--use-v2-block-manager]
                  [--num-lookahead-slots NUM_LOOKAHEAD_SLOTS] [--seed SEED]
                  [--swap-space SWAP_SPACE] [--cpu-offload-gb CPU_OFFLOAD_GB]
                  [--gpu-memory-utilization GPU_MEMORY_UTILIZATION]
                  [--num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE]
                  [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS]
                  [--max-num-partial-prefills MAX_NUM_PARTIAL_PREFILLS]
                  [--max-long-partial-prefills MAX_LONG_PARTIAL_PREFILLS]
                  [--long-prefill-token-threshold LONG_PREFILL_TOKEN_THRESHOLD]
                  [--max-num-seqs MAX_NUM_SEQS] [--max-logprobs MAX_LOGPROBS]
                  [--disable-log-stats]
                  [--quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,ptpc_fp8,fbgemm_fp8,modelopt,nvfp4,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,hqq,experts_int8,neuron_quant,ipex,quark,moe_wna16,None}]
                  [--rope-scaling ROPE_SCALING] [--rope-theta ROPE_THETA]
                  [--hf-overrides HF_OVERRIDES] [--enforce-eager]
                  [--max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE]
                  [--disable-custom-all-reduce]
                  [--tokenizer-pool-size TOKENIZER_POOL_SIZE]
                  [--tokenizer-pool-type TOKENIZER_POOL_TYPE]
                  [--tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG]
                  [--limit-mm-per-prompt LIMIT_MM_PER_PROMPT]
                  [--mm-processor-kwargs MM_PROCESSOR_KWARGS]
                  [--disable-mm-preprocessor-cache] [--enable-lora]
                  [--enable-lora-bias] [--max-loras MAX_LORAS]
                  [--max-lora-rank MAX_LORA_RANK]
                  [--lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE]
                  [--lora-dtype {auto,float16,bfloat16}]
                  [--long-lora-scaling-factors LONG_LORA_SCALING_FACTORS]
                  [--max-cpu-loras MAX_CPU_LORAS] [--fully-sharded-loras]
                  [--enable-prompt-adapter]
                  [--max-prompt-adapters MAX_PROMPT_ADAPTERS]
                  [--max-prompt-adapter-token MAX_PROMPT_ADAPTER_TOKEN]
                  [--device {auto,cuda,neuron,cpu,openvino,tpu,xpu,hpu}]
                  [--num-scheduler-steps NUM_SCHEDULER_STEPS]
                  [--use-tqdm-on-load | --no-use-tqdm-on-load]
                  [--multi-step-stream-outputs [MULTI_STEP_STREAM_OUTPUTS]]
                  [--scheduler-delay-factor SCHEDULER_DELAY_FACTOR]
                  [--enable-chunked-prefill [ENABLE_CHUNKED_PREFILL]]
                  [--speculative-model SPECULATIVE_MODEL]
                  [--speculative-model-quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,ptpc_fp8,fbgemm_fp8,modelopt,nvfp4,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,hqq,experts_int8,neuron_quant,ipex,quark,moe_wna16,None}]
                  [--num-speculative-tokens NUM_SPECULATIVE_TOKENS]
                  [--speculative-disable-mqa-scorer]
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
                  [--show-hidden-metrics-for-version SHOW_HIDDEN_METRICS_FOR_VERSION]
                  [--otlp-traces-endpoint OTLP_TRACES_ENDPOINT]
                  [--collect-detailed-traces COLLECT_DETAILED_TRACES]
                  [--disable-async-output-proc]
                  [--scheduling-policy {fcfs,priority}]
                  [--scheduler-cls SCHEDULER_CLS]
                  [--override-neuron-config OVERRIDE_NEURON_CONFIG]
                  [--override-pooler-config OVERRIDE_POOLER_CONFIG]
                  [--compilation-config COMPILATION_CONFIG]
                  [--kv-transfer-config KV_TRANSFER_CONFIG]
                  [--worker-cls WORKER_CLS]
                  [--worker-extension-cls WORKER_EXTENSION_CLS]
                  [--generation-config GENERATION_CONFIG]
                  [--override-generation-config OVERRIDE_GENERATION_CONFIG]
                  [--enable-sleep-mode] [--calculate-kv-scales]
                  [--additional-config ADDITIONAL_CONFIG] [--enable-reasoning]
                  [--reasoning-parser {deepseek_r1}]
```


## 命名参数

**--model**  

要使用的 Huggingface 模型的名称或路径。


默认值："facebook/opt-125m"


**--task**  

可选值：auto, generate, embedding, embed, classify, score, reward, transcription


模型的任务类型：每个 vLLM 实例仅支持一个任务，即使同一模型可以用于多个任务。当模型仅支持一个任务时，可以使用 "auto" 自动选择；否则，必须明确指定要使用的任务。


默认值："auto"


**--tokenizer**  

要使用的 Huggingface 分词器的名称或路径。如果未指定，则使用模型的名称或路径。


**--hf-config-path**  

要使用的 Huggingface 配置文件的名称或路径。如果未指定，则使用模型的名称或路径。


**--skip-tokenizer-init**  

跳过分词器和反分词器的初始化。期望输入中包含有效的 prompt_token_ids，并且 prompt 为 None。生成的输出将包含 token ID。


**--revision**  

要使用的特定模型版本。可以是分支名称、标签名称或提交 ID。如果未指定，则使用默认版本。


**--code-revision**  

Hugging Face Hub 上模型代码的特定版本。可以是分支名称、标签名称或提交 ID。如果未指定，则使用默认版本。


**--tokenizer-revision**  

要使用的 Huggingface 分词器的版本。可以是分支名称、标签名称或提交 ID。如果未指定，则使用默认版本。


**--tokenizer-mode**  

可选值：auto, slow, mistral, custom


分词器模式。

* "auto"：如果可用，则使用快速分词器。

* "slow"：始终使用慢速分词器。

* "mistral"：始终使用 mistral_common 分词器。

* "custom"：使用 --tokenizer 选择预注册的分词器。


默认值："auto"


**--trust-remote-code**  

信任来自 Huggingface 的远程代码。


**--allowed-local-media-path**  

允许 API 请求从服务器文件系统指定的目录中读取本地图像或视频。这是一个安全风险，应仅在受信任的环境中启用。


**--download-dir**  

下载和加载权重的目录，默认为 Huggingface 的默认缓存目录。


**--load-format**  

可选值: auto, pt, safetensors, npcache, dummy, tensorizer, sharded_state, gguf, bitsandbytes, mistral, runai_streamer


加载模型权重的格式。

* "auto"：尝试以 safetensors 格式加载权重，如果不可用，则回退到 pytorch bin 格式。

* "pt"：以 pytorch bin 格式加载权重。

* "safetensors"：以 safetensors 格式加载权重。

* "npcache"：以 pytorch 格式加载权重，并存储 numpy 缓存以加速加载。

* "dummy"：用随机值初始化权重，主要用于性能分析。

* "tensorizer"：使用 CoreWeave 的 tensorizer 加载权重。有关更多信息，请参阅示例部分中的 Tensorize vLLM Model 脚本。

* "runai_streamer"：使用 Run:ai Model Streamer 加载 Safetensors 权重。

* "bitsandbytes"：使用 bitsandbytes 量化加载权重。


默认值: "auto"


**--config-format**  

可选值: auto, hf, mistral


加载模型配置的格式。


* "auto"：如果可用，则尝试以 hf 格式加载配置，否则尝试以 mistral 格式加载。


默认值："ConfigFormat.AUTO"


**--dtype**  

可选值: auto, half, float16, bfloat16, float, float32


模型权重和激活的数据类型。

* "auto": 对于 FP32 和 FP16 模型，使用 FP16 精度；对于 BF16 模型，使用 BF16 精度。

* "half": 使用 FP16 精度。推荐用于 AWQ 量化。

* "float16": 与 "half" 相同。

* "bfloat16": 在精度和范围之间取得平衡。

* "float": FP32 精度的简写。

* "float32": 使用 FP32 精度。


默认值: "auto"


**--kv-cache-dtype**  

可选值: auto, fp8, fp8_e5m2, fp8_e4m3


KV 缓存存储的数据类型。如果为 "auto"，则使用模型的数据类型。CUDA 11.8+ 支持 fp8（即 fp8_e4m3）和 fp8_e5m2。ROCm (AMD GPU) 支持 fp8（即 fp8_e4m3）。


默认值: "auto"


**--max-model-len**  

模型的上下文长度。如果未指定，则从模型配置中自动推导。


**--guided-decoding-backend**  


默认将用于引导解码（JSON 模式 / 正则表达式等）的引擎。目前支持 [outlines-dev/outlines](https://github.com/outlines-dev/outlines)、[mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar) 和 [noamgat/lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)。可以通过 guided_decoding_backend 参数在每个请求中覆盖。可以在后端名称后附加逗号分隔的列表来提供后端特定的选项。有效的后端及其所有可用选项为: [xgrammar:no-fallback, xgrammar:disable-any-whitespace, outlines:no-fallback, lm-format-enforcer:no-fallback]。


默认值："xgrammar"


**--logits-processor-pattern**  

可选的正则表达式模式，指定可以通过 logits_processors 额外完成参数传递的有效 logits 处理器限定名称。默认为 None，表示不允许任何处理器。


**--model-impl**  

可选值: auto, vllm, transformers


要使用的模型实现。

* "auto"：如果存在 vLLM 实现，则尝试使用 vLLM 实现；否则回退到 Transformers 实现。

* "vllm"：使用 vLLM 模型实现。

* "transformers"：使用 Transformers 模型实现。


默认值："auto"


**--distributed-executor-backend**  

可选值: ray, mp, uni, external_launcher


用于分布式模型工作者的后端，可以是「ray」或「mp（多进程）」。如果 pipeline_parallel_size 和 tensor_parallel_size 的乘积小于或等于可用 GPU 的数量，则将使用「mp」以保持在单个主机上处理。否则，如果安装了 Ray，则默认使用「ray」，否则会失败。注意，TPU 仅支持 Ray 进行分布式推理。


**--pipeline-parallel-size, -pp**  

流水线阶段的数量。


默认值：1


**--tensor-parallel-size, -tp**  

张量并行副本的数量。


默认值：1


**--enable-expert-parallel**  

对 MoE 层使用专家并行而不是张量并行。


**--max-parallel-loading-workers**  

以多个批次顺序加载模型，以避免在使用张量并行和大模型时出现 RAM 内存不足 (OOM)。


**--ray-workers-use-nsight**  

如果指定，则使用 nsight 分析 Ray 工作线程。


**--block-size**  

可选值: 8, 16, 32, 64, 128


连续 token 块的 token 块大小。在 neuron 设备上忽略此参数，并设置为 --max-model-len。在 CUDA 设备上，仅支持最大为 32 的块大小。在 HPU 设备上，块大小默认为 128。


**--enable-prefix-caching, --no-enable-prefix-caching**  

启用自动前缀缓存。使用 --no-enable-prefix-caching 显式禁用。


**--disable-sliding-window**  

禁用滑动窗口，限制为滑动窗口大小。


**--use-v2-block-manager**  

[已弃用] 块管理器 v1 已被移除，SelfAttnBlockSpaceManager（即块管理器 v2）现在是默认值。将此标志设置为 True 或 False 对 vLLM 行为没有影响。


**--num-lookahead-slots**  

推测解码所需的实验性调度配置。未来将被推测配置取代；目前用于支持正确性测试。


默认值：0


**--seed**  

操作的随机种子。


**--swap-space**  

每个 GPU 的 CPU 交换空间大小 (GiB)。


默认值：4


**--cpu-offload-gb**  

每个 GPU 卸载到 CPU 的空间 (GiB)。默认值为 0，表示不卸载。直观地说，此参数可以看作是一种虚拟增加 GPU 内存大小的方法。例如，如果你有一个 24 GB 的 GPU 并将此值设置为 10，则虚拟上可以将其视为 34 GB 的 GPU。然后你可以加载一个 13B 的 BF16 权重模型，该模型至少需要 26 GB 的 GPU 内存。请注意，这需要快速的 CPU-GPU 互连，因为部分模型会在每次模型前向传递时从 CPU 内存动态加载到 GPU 内存。


默认值：0


**--gpu-memory-utilization**  

用于模型执行器的 GPU 内存比例，范围为 0 到 1。例如，值为 0.5 表示 50% 的 GPU 内存利用率。如果未指定，则使用默认值 0.9。这是每个实例的限制，仅适用于当前的 vLLM 实例。即使在同一 GPU 上运行另一个 vLLM 实例，也不会影响此设置。例如，如果你在同一 GPU 上运行两个 vLLM 实例，可以将每个实例的 GPU 内存利用率设置为 0.5。


默认值：0.9


**--num-gpu-blocks-override**  

如果指定，则忽略 GPU 分析结果并使用此 GPU 块数。用于测试抢占。


**--max-num-batched-tokens**  

每次迭代的最大批处理 token 数。


**--max-num-partial-prefills**  

对于分块预填充，最大并发部分预填充数。默认为 1。


默认值：1


**--max-long-partial-prefills**  

对于分块预填充，超过 --long-prefill-token-threshold 的提示词的最大并发预填充数。将此值设置为小于 --max-num-partial-prefills 可以在某些情况下允许较短的提示词插队到较长的提示词前面，从而改善延迟。默认为 1。


默认值：1


**--long-prefill-token-threshold**  

对于分块预填充，如果提示词长度超过此 token 数，则视为长提示词。默认为模型上下文长度的 4%。


默认值：0


**--max-num-seqs**  

每次迭代的最大序列数。


**--max-logprobs**  

返回的最大 log probs 数，logprobs 在 SamplingParams 中指定。


默认值：20


**--disable-log-stats**  

禁用统计日志记录。

**--quantization, -q**  

可选值：aqlm, awq, deepspeedfp, tpu_int8, fp8, ptpc_fp8, fbgemm_fp8, modelopt, nvfp4, marlin, gguf, gptq_marlin_24, gptq_marlin, awq_marlin, gptq, compressed-tensors, bitsandbytes, qqq, hqq, experts_int8, neuron_quant, ipex, quark, moe_wna16, None


用于量化权重的方法。如果为 None，则首先检查模型配置文件中的 quantization_config 属性。如果该属性为 None，则假定模型权重未量化，并使用 dtype 确定权重的数据类型。


**--rope-scaling**  

RoPE 缩放的 JSON 格式配置。例如: `{"rope_type":"dynamic","factor":2.0}`。


**--rope-theta**  

RoPE 的 theta 值。与 rope_scaling 一起使用。在某些情况下，更改 RoPE 的 theta 值可以提高缩放模型的性能。


**--hf-overrides**  

HuggingFace 配置的额外参数。这应该是一个 JSON 字符串，将被解析为字典。


**--enforce-eager**  

始终使用 eager 模式的 PyTorch。如果为 False，则将在混合模式下使用 eager 模式和 CUDA 图，以实现最佳性能和灵活性。


**--max-seq-len-to-capture**  

CUDA 图覆盖的最大序列长度。当序列的上下文长度超过此值时，将回退到 eager 模式。此外，对于编码器-解码器模型，如果编码器输入的序列长度超过此值，也将回退到 eager 模式。


默认值：8192


**--disable-custom-all-reduce**  

参见 ParallelConfig。


**--tokenizer-pool-size**  

用于异步分词的分词器池大小。如果为 0，则使用同步分词。


默认值：0


**--tokenizer-pool-type**  

用于异步分词的分词器池类型。如果 tokenizer_pool_size 为 0，则忽略此参数。


默认值：「ray」


**--tokenizer-pool-extra-config**  

分词器池的额外配置。这应该是一个 JSON 字符串，将被解析为字典。如果 tokenizer_pool_size 为 0，则忽略此参数。


**--limit-mm-per-prompt**  

对于每个多模态插件，可以设置每个提示词允许的输入实例数量。输入格式为逗号分隔的列表，例如：「image=16,video=2」，表示每个提示词最多允许上传 16 张图片和 2 个视频。如果不进行设置，系统默认每种模态最多允许 1 个输入实例。


**--mm-processor-kwargs**  

多模态输入映射/处理的覆盖配置，例如图像处理器。例如: `{"num_crops": 4}`。


**--disable-mm-preprocessor-cache**  

如果为 true，则禁用多模态预处理器/映射器的缓存。（不推荐）


**--enable-lora**  

如果为 True，则启用 LoRA 适配器的处理。


**--enable-lora-bias**  

如果为 True，则为 LoRA 适配器启用偏置。


**--max-loras**  

单个批次中 LoRA 的最大数量。


默认值：1


**--max-lora-rank**  

LoRA 的最大秩。


默认值：16


**--lora-extra-vocab-size**  

LoRA 适配器中可能存在的额外词汇表的最大大小（添加到基础模型词汇表中）。


默认值：256


**--lora-dtype**  

可选值：auto, float16, bfloat16


LoRA 的数据类型。如果为 auto，则默认为基础模型的数据类型。


默认值：「auto」


**--long-lora-scaling-factors**  

指定多个缩放因子（可以与基础模型的缩放因子不同，例如 Long LoRA），以允许同时使用使用这些缩放因子训练的多个 LoRA 适配器。如果未指定，则仅允许使用基础模型缩放因子训练的适配器。


**--max-cpu-loras**  

存储在 CPU 内存中的 LoRA 的最大数量。必须大于或等于 max_loras。默认为 max_loras。


**--fully-sharded-loras**  

默认情况下，LoRA 计算的一半通过张量并行进行分片。启用此选项将使用完全分片的层。在高序列长度、最大秩或张量并行大小的情况下，这可能会更快。


**--enable-prompt-adapter**  

如果为 True，则启用 PromptAdapters 的处理。


**--max-prompt-adapters**  

单个批次中 PromptAdapters 的最大数量。


默认值：1


**--max-prompt-adapter-token**  

PromptAdapters 的最大 token 数量。


默认值：0


**--device**  

可选值: auto, cuda, neuron, cpu, openvino, tpu, xpu, hpu


vLLM 执行的设备类型。


默认值：「auto」


**--num-scheduler-steps**  

每次调度器调用的最大前向步骤数。


默认值：1


**--use-tqdm-on-load, --no-use-tqdm-on-load**  

加载模型权重时是否启用/禁用进度条。


默认值：True


**--multi-step-stream-outputs**  

如果为 False，则多步骤将在所有步骤结束时流式输出。


默认值：True


**--scheduler-delay-factor**  

在调度下一个提示词之前应用延迟（延迟因子乘以前一个提示词的延迟）。


默认值：0.0


**--enable-chunked-prefill**  

如果设置，则可以根据 max_num_batched_tokens 对预填充请求进行分块处理。


**--speculative-config**

关于推测性解码的配置。应为一个JSON格式的字符串。


**--speculative-model**  

用于推测解码的草稿模型的名称。


**--speculative-model-quantization**  

可选值: aqlm, awq, deepspeedfp, tpu_int8, fp8, ptpc_fp8, fbgemm_fp8, modelopt, nvfp4, marlin, gguf, gptq_marlin_24, gptq_marlin, awq_marlin, gptq, compressed-tensors, bitsandbytes, qqq, hqq, experts_int8, neuron_quant, ipex, quark, moe_wna16, None


用于量化草稿模型权重的方法。如果为 None，则首先检查模型配置文件中的 quantization_config 属性。如果该属性为 None，则假定模型权重未量化，并使用 dtype 确定权重的数据类型。


**--num-speculative-tokens**  

在推测解码中从草稿模型中采样的推测 token 数量。


**--speculative-disable-mqa-scorer**  

如果设置为 True，则在推测解码中禁用 MQA 评分器，并回退到批次扩展。


**--speculative-draft-tensor-parallel-size, -spec-draft-tp**  

推测解码中草稿模型的张量并行副本数量。


**--speculative-max-model-len**  

草稿模型支持的最大序列长度。超过此长度的序列将跳过推测。


**--speculative-disable-by-batch-size**  

如果排队请求的数量超过此值，则对新传入的请求禁用推测解码。


**--ngram-prompt-lookup-max**  

推测解码中 ngram 提示词查找窗口的最大大小。


**--ngram-prompt-lookup-min**  

推测解码中 ngram 提示词查找窗口的最小大小。


**--spec-decoding-acceptance-method**  

可选值: rejection_sampler, typical_acceptance_sampler


指定在推测解码中草稿 token 验证期间使用的接受方法。支持两种类型的接受例程: 1) RejectionSampler，不允许更改草稿 token 的接受率；2) TypicalAcceptanceSampler，可配置，允许以质量为代价提高接受率，反之亦然。


默认值：「rejection_sampler」


**--typical-acceptance-sampler-posterior-threshold**  

设置 token 被接受的后验概率的下限阈值。此阈值由 TypicalAcceptanceSampler 在推测解码期间用于做出采样决策。默认值为 0.09。


**--typical-acceptance-sampler-posterior-alpha**  

TypicalAcceptanceSampler 中基于熵的 token 接受阈值的缩放因子。通常默认为 --typical-acceptance-sampler-posterior-threshold 的平方根，即 0.3。


**--disable-logprobs-during-spec-decoding**  

如果设置为 True，则在推测解码期间不返回 token 的对数概率。如果设置为 False，则根据 SamplingParams 中的设置返回对数概率。如果未指定，则默认为 True。在推测解码期间禁用对数概率可以减少延迟，因为跳过了提议采样、目标采样和确定接受 token 后的对数概率计算。


**--model-loader-extra-config**  

这是为模型加载器提供的额外配置信息。它将传递给与所选加载格式 (load_format) 对应的模型加载器。这个配置信息需要是一个 JSON 格式的字符串，之后会被解析为一个字典。


**--ignore-patterns**  

加载模型时要忽略的模式。默认为 original/**/*，以避免重复加载 llama 的检查点。


默认值：[]


**--preemption-mode**  

如果为 'recompute'，则引擎通过重新计算执行抢占；如果为 'swap'，则引擎通过块交换执行抢占。


**--served-model-name**  

API 中使用的模型名称。如果提供了多个名称，服务器将响应提供的任何名称。响应中模型字段的模型名称将是此列表中的第一个名称。如果未指定，模型名称将与 --model 参数相同。请注意，此名称也将用于 Prometheus 指标的 model_name 标签内容，如果提供了多个名称，则 metrics 标签将采用第一个名称。


**--qlora-adapter-name-or-path**  

QLoRA 适配器的名称或路径。


**--show-hidden-metrics-for-version**  

启用自指定版本以来已隐藏的已弃用 Prometheus 指标。例如，如果某个先前已弃用的指标自 v0.7.0 版本以来已隐藏，则可以使用 --show-hidden-metrics-for-version=0.7 作为临时解决方案，同时迁移到新指标。该指标可能会在即将发布的版本中完全删除。


**--otlp-traces-endpoint**  

OpenTelemetry 跟踪将发送到的目标 URL。


**--collect-detailed-traces**  

有效选项为 model、worker、all。仅在设置了 --otlp-traces-endpoint 时设置此选项才有意义。如果设置，它将为指定模块收集详细的跟踪信息。这涉及使用可能昂贵或阻塞的操作，因此可能会影响性能。


**--disable-async-output-proc**  

禁用异步输出处理。这可能会导致性能下降。


**--scheduling-policy**  

可选值：fcfs, priority


要使用的调度策略。「fcfs」（先到先服务，即按请求到达的顺序处理；默认）或「priority」（根据给定的优先级处理请求，较低的值意味着较早处理，到达时间决定任何平局）。


默认值：「fcfs」


**--scheduler-cls**  

要使用的调度器类。「vllm.core.scheduler.Scheduler」是默认调度器。可以是类直接路径，也可以是形式为「mod.custom_class」的类路径。


默认值：「vllm.core.scheduler.Scheduler」


**--override-neuron-config**  

覆盖或设置 neuron 设备配置。例如: `{"cast_logits_dtype": "bloat16"}`。


**--override-pooler-config**  

覆盖或设置池化模型的池化方法。例如: `{"pooling_type": "mean", "normalize": false}`。


**--compilation-config, -O**  

模型的 torch.compile 配置。当它是一个数字 (0、1、2、3) 时，它将被解释为优化级别。注意:级别 0 是没有任何优化的默认级别。级别 1 和 2 仅用于内部测试。级别 3 是生产推荐的级别。要指定完整的编译配置，请使用 JSON 字符串。遵循传统编译器的约定，也支持使用不带空格的 -O。-O3 等同于 -O 3。


**--kv-transfer-config**  

分布式 KV 缓存传输的配置。应该是一个 JSON 字符串。


**--worker-cls**  

用于分布式执行的 worker 类。


默认值：「auto」


**--worker-extension-cls**  

worker 类之上的 worker 扩展类，如果你只想向 worker 类添加新功能而不更改现有功能，则此功能很有用。


默认值：""


**--generation-config**  

生成配置的文件夹路径。默认为 'auto'，生成配置将从模型路径加载。如果设置为 'vllm'，则不加载生成配置，将使用 vLLM 默认值。如果设置为文件夹路径，则从指定的文件夹路径加载生成配置。如果生成配置中指定了 max_new_tokens，则它将设置服务器范围内所有请求的输出 token 数量的限制。


默认值：auto


**--override-generation-config**  

以 JSON 格式覆盖或设置生成配置。例如: `{"temperature": 0.5}`。如果与 --generation-config=auto 一起使用，则覆盖参数将与模型的默认配置合并。如果 generation-config 为 None，则仅使用覆盖参数。


**--enable-sleep-mode**  

启用引擎的睡眠模式。（仅支持 cuda 平台）


**--calculate-kv-scales**  

当 kv-cache-dtype 为 fp8 时，启用动态计算 k_scale 和 v_scale。如果 calculate-kv-scales 为 false，那么如果模型检查点中有可用的值，将会从模型检查点中加载这些缩放比例。否则，比例将默认为 1.0。


**--additional-config**  

以 JSON 格式为指定平台提供的额外配置。不同平台可能支持不同的配置。确保配置对你使用的平台有效。输入格式类似于 `{"config_key":"config_value"}`。


**--enable-reasoning**  

是否启用模型的 reasoning_content。如果启用，模型将能够生成推理内容。


**--reasoning-parser**  

可选值：deepseek_r1


根据你使用的模型选择推理解析器。这用于将推理内容解析为 OpenAI API 格式。如果启用了  `--enable-reasoning` 功能，则此选项是必需的。


## 异步引擎参数

以下是异步引擎的附加参数：

```plain
usage: vllm serve [-h] [--disable-log-requests]  
```

### 

### 命名参数

**--disable-log-requests**

禁用请求日志记录。

