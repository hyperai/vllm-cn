---
title: OpenAI 兼容服务器
---


vLLM 提供了一个实现 OpenAI [Completions](https://platform.openai.com/docs/api-reference/completions) 和 [Chat](https://platform.openai.com/docs/api-reference/chat) API 的 HTTP 服务器。


您可以使用 Python 或  [Docker](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html) 启动服务器：

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
```


如需要调用服务器，您可以使用官方的 OpenAI Python 客户端库，或任何其他 HTTP 客户端。

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)


completion = client.chat.completions.create(
  model="NousResearch/Meta-Llama-3-8B-Instruct",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)


print(completion.choices[0].message)
```


## API 参考

有关 API 的更多信息，请参阅 [OpenAI API 参考](https://platform.openai.com/docs/api-reference)。我们支持除以下参数的所有参数：

* Chat：`tools` 和 `tool_choice`。

* Completions：`suffix`。


vLLM 还提供对 OpenAI Vision API 兼容推理的实验性支持。有关更多详细信息，请参阅[使用 VLMs](https://docs.vllm.ai/en/latest/models/vlm.html)。


## 附加参数

vLLM 支持一组不属于 OpenAI API 的部分参数。如需使用这些参数，你可以将它们作为额外参数传递给 OpenAI 客户端，或者直接将它们合并到 JSON 负载中（如果你直接使用 HTTP 调用）。

```python
completion = client.chat.completions.create(
  model="NousResearch/Meta-Llama-3-8B-Instruct",
  messages=[
    {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
  ],
  extra_body={
    "guided_choice": ["positive", "negative"]
  }
)
```

### 

### Chat API 的附加参数

以下[采样参数（点击查看文档）](https://docs.vllm.ai/en/latest/dev/sampling_params.html)均被支持。

```plain
    best_of: Optional[int] = None
    use_beam_search: bool = False
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    prompt_logprobs: Optional[int] = None
```


以下附加参数均被支持：

```plain
    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=
        ("If true, the generation prompt will be added to the chat template. "
         "This is a parameter used by chat template in tokenizer config of the "
         "model."),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    documents: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=
        ("A list of dicts representing documents that will be accessible to "
         "the model if it is performing RAG (retrieval-augmented generation)."
         " If the template does not support RAG, this argument will have no "
         "effect. We recommend that each document should be a dict containing "
         "\"title\" and \"text\" keys."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."),
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description=("If specified, the output will follow the JSON schema."),
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be either "
            "'outlines' / 'lm-format-enforcer'"))
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."))
```


### Completions API 的附加参数

以下[采样参数（点击查看文档）](https://docs.vllm.ai/en/latest/dev/sampling_params.html)均被支持。

```plain
    use_beam_search: bool = False
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    allowed_token_ids: Optional[List[int]] = None
    prompt_logprobs: Optional[int] = None
```


以下附加参数均被支持：

```plain
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."),
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description=
        ("Similar to chat completion, this parameter specifies the format of "
         "output. Only {'type': 'json_object'} or {'type': 'text' } is "
         "supported."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description="If specified, the output will follow the JSON schema.",
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be one of "
            "'outlines' / 'lm-format-enforcer'"))
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."))
```


## 聊天模板

为了使语言模型支持聊天协议，vLLM 要求模型在其 tokenizer 配置中包含一个聊天模板。聊天模板是一个 Jinja2 模板，它指定了角色、消息和其他特定于聊天对 tokens 如何在输入中编码。


`NousResearch/Meta-Llama-3-8B-Instruct` 的示例聊天模板可以在[这里](https://github.com/meta-llama/llama3?tab=readme-ov-file#instruction-tuned-models)找到。


一些模型即使经过了指令/聊天微调，仍然不提供聊天模板。对于这些模型，你可以在 `--chat-template` 参数中手动指定聊天模板的文件路径或字符串形式。如果没有聊天模板，服务器将无法处理聊天请求，所有聊天请求将出错。

```bash
vllm serve <model> --chat-template ./path-to-chat-template.jinja
```


vLLM 社区为流行的模型提供了一组聊天模板，可以在示例目录[这里](https://github.com/vllm-project/vllm/tree/main/examples/)找到它们。


## 服务器的命令行参数

```plain
usage: vllm serve [-h] [--host HOST] [--port PORT]
                  [--uvicorn-log-level {debug,info,warning,error,critical,trace}]
                  [--allow-credentials] [--allowed-origins ALLOWED_ORIGINS]
                  [--allowed-methods ALLOWED_METHODS]
                  [--allowed-headers ALLOWED_HEADERS] [--api-key API_KEY]
                  [--lora-modules LORA_MODULES [LORA_MODULES ...]]
                  [--prompt-adapters PROMPT_ADAPTERS [PROMPT_ADAPTERS ...]]
                  [--chat-template CHAT_TEMPLATE]
                  [--response-role RESPONSE_ROLE] [--ssl-keyfile SSL_KEYFILE]
                  [--ssl-certfile SSL_CERTFILE] [--ssl-ca-certs SSL_CA_CERTS]
                  [--ssl-cert-reqs SSL_CERT_REQS] [--root-path ROOT_PATH]
                  [--middleware MIDDLEWARE] [--return-tokens-as-token-ids]
                  [--disable-frontend-multiprocessing]
                  [--enable-auto-tool-choice]
                  [--tool-call-parser {mistral,hermes}] [--model MODEL]
                  [--tokenizer TOKENIZER] [--skip-tokenizer-init]
                  [--revision REVISION] [--code-revision CODE_REVISION]
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
                  [--disable-log-requests] [--max-log-len MAX_LOG_LEN]
```


### 命名参数

`--host`  

主机名


`--port`  

端口号


默认值：8000


`--uvicorn-log-level`  

可选项：debug, info, warning, error, critical, trace


uvicorn 的日志级别


默认值: 「info」


`--allow-credentials`  

允许凭据


默认值：False


`--allowed-origins`  

允许的来源


默认值：['*']


`--allowed-methods`  

允许的方法


默认值：['*']


`--allowed-headers`  

允许的标头


默认值：['*']


`--api-key`  

如果提供这个密钥，服务器将要求在标头中显示此密钥。


`--lora-modules`  

LoRA 模块配置的格式为 name=path，可以指定多个模块。


`--prompt-adapters`  

以 name=path 的格式提示适配器配置，可以指定多个适配器。


`--chat-template`  

聊天模板的文件路径，或针对指定模型的单行形式的模板。


`--response-role`  

如果设置了 request.add_generation_prompt=true，则返回的角色名称。


默认值：assistant


`--ssl-keyfile`  

SSL 密钥文件的文件路径


`--ssl-certfile`  

SSL 证书文件的文件路径


`--ssl-ca-certs`  

CA 证书文件


`--ssl-cert-reqs`  

是否需要客户端证书（请参阅 stdlib ssl 模块）


默认值：0


`--root-path`  

当应用程序位于基于路径的路由代理之后时的 FastAPI 根路径。


`--middleware`  

要应用于应用程序的额外 ASGI 中间件。我们接受多个 –middleware 参数。该值应为一个导入路径。如果提供了一个函数，vLLM 将使用 @app.middleware('http') 将其添加到服务器中。如果提供了一个类，vLLM 将使用 app.add_middleware() 将其添加到服务器中。


默认值：[]


`--return-tokens-as-token-ids`  

当指定了 –max-logprobs 参数时，它表示将单个标记表示为 "token_id:{token_id}" 形式的字符串，以便可以识别那些无法进行 JSON 编码的标记。


默认值：False


`--disable-frontend-multiprocessing`  

如果指定了该选项，将在与模型服务引擎相同的进程中运行 OpenAI 前端服务器。


默认值：False


`--enable-auto-tool-choice`  

为支持的型号启用自动工具选择。使用 –tool-call-parser 参数指定要使用的解析器


默认值：False


`--tool-call-parser`  

可选项：mistral, hermes


根据您使用的模型选择合适的工具调用解析器。这用于将模型生成的工具调用解析为 OpenAI API 格式。 参数 –enable-auto-tool-choice 是必需的。


`--model`  

要使用的 Huggingface 模型的名称或路径。


默认值：「facebook/opt-125m」


`--tokenizer`  

要使用的 Huggingface tokenizer 的名称或路径。如果未指定，将使用模型名称或路径。


`--skip-tokenizer-init`  

跳过 tokenizer 和 detokenizer 的初始化


默认值：False


`--revision`  

要使用的指定模型版本。它可以是分支名称、标签名称或提交 ID。如果未指定，将使用默认版本。


`--code-revision`  

在 Hugging Face Hub 上用于模型代码的指定修订版本。它可以是分支名称、标签名称或提交 ID。如果未指定，将使用默认版本。


`--tokenizer-revision`  

使用 Huggingface tokenizer 的修订版本。它可以是分支名称、标签名称或提交 ID。如果未指定，将使用默认版本。


`--tokenizer-mode`  

可选项：auto, slow, mistral


tokenizer 模式。


* 「auto」将使用快速 tokenizer (如果可用)。

* 「slow」将始终使用慢 tokenizer。

* 「mistral」将始终使用 mistral_common tokenizer 。


默认值：「auto」


`--trust-remote-code`  

信任来自 Huggingface 的远程代码。


默认值：False


`--download-dir`  

用于下载和加载权重的目录，默认为 huggingface 的默认缓存目录。


`--load-format`  

可选项：auto、pt、safetensors、npcache、dummy、tensorizer、sharded_state、gguf、bitsandbytes、mistral


要加载的模型权重的格式。


* 「auto」将尝试以 safetensors 格式加载权重，如果 safetensors 格式不可用，则会回退到 pytorch bin 格式。

* 「pt」将以 pytorch bin 格式加载权重。

* 「safetensors」将以 safetensors 格式加载权重。

* 「npcache」将以 pytorch 格式加载权重，并存储 numpy 缓存以加快加载速度。

* 「dummy」将使用随机值初始化权重，主要用于性能分析。

* 「tensorizer」将使用 CoreWeave 的张量器加载权重。有关详细信息，请参阅示例部分中的 Tensorize vLLM 模型脚本。

* 「bitsandbytes」将使用 bitsandbytes 量化来加载权重。


默认值：「auto」


`--config-format`  

可选项：auto、hf、mistral


要加载的模型配置的格式。


「auto」将尝试以 hf 格式加载配置（如果可用），否则它将尝试以 Mistra 格式加载


默认值：「auto」


`--dtype`  

可选项：auto、half、float16、bfloat16、float、float32


模型权重和激活的数据类型。

* 「auto」将对 FP32 和 FP16 型号使用 FP16 精度，对 BF16 型号使用 BF16 精度。

* 「half」用于 FP16，推荐用于 AWQ 量化。

* 「float16」与「half」相同。

* 「bfloat16」用于在精度和范围之间取得平衡。

* 「float」是 FP32 精度的简写。

* 「float32」表示 FP32 精度。


默认值：「auto」


`--kv-cache-dtype`  

可选项：auto、fp8、fp8_e5m2、fp8_e4m3


kv 缓存存储的数据类型。如果设置为 "auto"，将使用模型的数据类型。CUDA 11.8+ 版本支持 fp8 (=fp8_e4m3) 和 fp8_e5m2。ROCm (AMD GPU) 支持 fp8 (=fp8_e4m3)。


默认值：「auto」


`--quantization-param-path`  

路径指向包含 KV 缓存缩放因子的JSON文件。通常，当 KV 缓存数据类型设定为 FP8 时，需要提供此文件。否则，KV 缓存缩放因子默认为 1.0，这可能会影响准确性。只有在 CUDA 版本高于 11.8 时，才支持不使用缩放的 FP8_E5M2。对于 ROCm (AMD GPU) ，则支持 FP8_E4M3，以满足一般推理标准。


`--max-model-len`  

模型的上下文长度。如果未指定，将自动从模型配置中推导出来。


`--guided-decoding-backend`  

可选项：outlines, lm-format-enforcer


默认情况下，用于引导解码（如 JSON 模式、正则表达式等）的引擎是哪种。当前支持的选项包括 [outlines-dev/outlines](https://github.com/outlines-dev/outlines) 和 [noamgat/lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)。此外，可以通过 guided_decoding_backend 在每个请求中自定义选择不同的引擎。


默认值：「outlines」


`--distributed-executor-backend`  

可选项：ray, mp


用于分布式服务的后端。当使用超过 1 个 GPU 时，如果已安装 ray，将自动设置为「ray」，否则将自动设置为「mp」(multiprocessing)。


`--worker-use-ray`  

已弃用，请使用 –distributed-executor-backend=ray。


默认值：False


`--pipeline-parallel-size,`  

管道阶段的数量。


默认值：1


`--tensor-parallel-size,`  

张量并行副本的数量。


默认值：1


`--max-parallel-loading-workers`  

分多个批次顺序加载模型，这样可以避免使用张量并行和大型模型时出现内存不足 (RAM OOM) 的情况。


`--ray-workers-use-nsight`  

如果指定了该选项，则使用 Nsight 对 Ray 工作进程进行性能分析。


默认值：False


`--block-size`  

可选项：8、16、32


用于连续 token 块的 token 块大小。在神经元设备上此参数被忽略，并设置为最大模型长度。


默认值: 16


`--enable-prefix-caching`  

启用自动前缀缓存。


默认值：False


`--disable-sliding-window`  

禁用滑动窗口，将其限制为滑动窗口大小。


默认值：False


`--use-v2-block-manager`  

使用 BlockSpaceMangerV2。


默认值：False


`--num-lookahead-slots`  

用于推测解码的实验性调度配置。未来，它将被推测性配置所替代；在此之前，它是为了确保正确性测试的进行而保留的。


默认值：0


`--seed`  

用于操作的随机种子。


默认值：0


`--swap-space`  

每个 GPU 的 CPU 交换空间大小 (GiB)。


默认值：4


`--cpu-offload-gb`  

每个 GPU 用于卸载到 CPU 的空间大小（以 GiB 为单位）。默认值为 0，表示不卸载。直观地说，这个参数可以被视为一种虚拟地增加 GPU 内存大小的方式。例如，如果您有一个 24 GB GPU 并将其参数设置为 10，那么实际上您可以把它看作是一个 34GB 的 GPU。这样您就可以加载一个 BF16 权重的 13B 模型，该模型至少需要 26GB 的 GPU 内存。请注意，这需要快速的 CPU-GPU 互连，因为在每次模型前向传播中，部分模型是从 CPU 内存动态加载到 GPU 内存中的。


默认值：0


`--gpu-memory-utilization`  

用于模型执行器的 GPU 内存比例，其范围从 0 到 1。例如，值为 0.5 表示 GPU 内存利用率为 50%。如果未指定，将使用默认值 0.9。


默认值：0.9


`--num-gpu-blocks-override`  

如果指定了该参数，则忽略 GPU 分析结果并使用这个数量的 GPU 块。此操作用于测试抢占。


`--max-num-batched-tokens`  

每次迭代的最大批处理 token 数量。


`--max-num-seqs`  

每次迭代的最大序列数量。


默认值：256


`--max-logprobs`  

在采样参数 (SamplingParams) 中指定了对数概率 (logprobs) 返回的最大值。


默认值：20


`--disable-log-stats`  

禁用日志记录统计。


默认值：False


`--quantization,-q`

可选项：aqlm, awq, deepspeedfp, tpu_int8, fp8, fbgemm_fp8, modelopt, marlin, gguf, gptq_marlin_24, gptq_marlin, awq_marlin, gptq, compressed-tensors, bitsandbytes, qqq, experts_int8, neuron_quant, None


用于量化权重的方法。如果没有制定，我们首先检查模型配置文件中的 quantization_config 属性。如果为 None，我们假设模型权重未量化，并使用 dtype 来确定权重的数据类型。


`--rope-scaling`  

JSON 格式的 RoPE 扩展配置。例如，{"type":"dynamic","factor":2.0}


`--rope-theta`  

RoPE theta。与 rope_scaling 一起使用。在某些情况下，更改 RoPE theta 可以提高缩放模型的性能。


`--enforce-eager`  

始终使用 eager 模式的 PyTorch。如果为 False，将会使用 eager 模式和 hybrid CUDA 图混合模式，以获得最大性能和灵活性。


默认值：False


`--max-context-len-to-capture`  

CUDA 图表覆盖的最大上下文长度。当序列的上下文长度大于此值时，我们会回退到 eager 模式。 (已弃用。请使用 –max-seq-len-to-capture 代替)

`--max-seq-len-to-capture`  

CUDA 图涵盖的最大序列长度。当序列的上下文长度大于此值时，我们会退回到 eager 模式。


默认值：8192


`--disable-custom-all-reduce`  

请参阅并行配置 (ParallelConfig)。


默认值：False


`--tokenizer-pool-size`  

用于异步标记化的标记 tokenizer 池的大小。如果为 0，将使用同步标记化。


默认值：0


`--tokenizer-pool-type`  

用于异步标记化 (asynchronous tokenization) 的标记 tokenizer 池的类型。如果 tokenizer_pool_size 为 0，则使用同步标记化。


默认值：「ray」


`--tokenizer-pool-extra-config`  

tokenizer 池的额外配置。这应该是一个将被解析为字典的 JSON 字符串。如果 tokenizer_pool_size 为 0，则被忽略。


`--limit-mm-per-prompt`  

对于每个多模态插件，限制每个提示允许的输入实例数量。需要提供一个以逗号分隔的项目列表，例如：「image=16,video=2」表示每个提示最多可以包含 16 个图像和 2 个视频。默认情况下，每种模态对应的数量为 1。


`--enable-lora`  

如果为 True，则启用对 LoRA 适配器的处理。


默认值：False


`--max-loras`  

单批次中 LoRA 的最大数量。


默认值：1


`--max-lora-rank`  

最大 LoRA rank。


默认值：16


`--lora-extra-vocab-size`  

LoRA 适配器可以包含的最大额外词汇量，此词汇量将被添加到基础模型的词汇表中。


默认值：256


`--lora-dtype`  

可选项：auto、float16、bfloat16、float32


LoRA 的数据类型。如果为 auto，则默认为基本模型 dtype。


默认值：「auto」


`--long-lora-scaling-factors`  

指定多个缩放因子（可以与基础模型的缩放因子不同，例如长 LoRA)，以便能够同时使用用这些缩放因子训练的多个低秩自适应 (LoRA) 适配器。如果未指定，则仅允许使基础模型缩放因子训练的适配器。


`--max-cpu-loras`  

CPU 内存中存储的 LoRA 的最大数量。必须 >= max_num_seqs。默认为 max_num_seqs。


`--fully-sharded-loras`  

默认情况下，只有一半的 LoRA 计算通过张量并行进行分片。启用此功能将使用完全分片的层。在高序列长度、最大秩或张量并行大小的情况下，这可能会更快。


默认值：False


`--enable-prompt-adapter`  

如果为 True，则启用 PromptAdapter 处理。


默认值：False


`--max-prompt-adapters`  

每批次中 PromptAdapter 的最大数量。


默认值：1


`--max-prompt-adapter-token`  

PromptAdapters 的最大 token 数量。


默认值：0


`--device`  

可选项：auto、cuda、neuron、cpu、openvino、tpu、xpu


vLLM 执行的设备类型。


默认值：「auto」


`--num-scheduler-steps`  

每个调度器调用的最大前进步数。


默认值：1


`--scheduler-delay-factor`  

在调度下一个提示之前应用一个延迟（延迟因子乘以前一个提示的延迟时间）。


默认值：0.0


`--enable-chunked-prefill`  

如果设置了该参数，预填充请求可以根据最大批处理令牌数 (max_num_batched_tokens) 进行分块。


`--speculative-model`  

推测解码中要使用的 draft model 的名称。


`--speculative-model-quantization`  

可选项：aqlm, awq, deepspeedfp, tpu_int8, fp8, fbgemm_fp8, modelopt, marlin, gguf, gptq_marlin_24, gptq_marlin, awq_marlin, gptq, compressed-tensors, bitsandbytes, qqq, experts_int8, neuron_quant, None


用于量化推测模型权重的方法。如果为 None，我们首先检查模型配置文件中的 quantization_config 属性。如果该属性也为 None，我们假设模型权重未被量化，并使用数据类型（dtype）来确定权重的数据类型。


`--num-speculative-tokens`  

在推测解码中从 draft model 中采样的推测 token 数量。


`--speculative-draft-tensor-parallel-size,-spec-draft-tp`  

推测解码中 draft model 的张量并行副本数。


`--speculative-max-model-len`  

draft model 支持的最大序列长度。超过这个长度的序列将跳过推测。


`--speculative-disable-by-batch-size`  

如果入队请求的数量大于此值，则禁用对新传入请求的推测解码。


`--ngram-prompt-lookup-max`  

在推测解码中，ngram 提示查找的窗口最大大小。


`--ngram-prompt-lookup-min`  

在推测解码中，ngram 提示查找的窗口最小大小。


`--spec-decoding-acceptance-method`  

可选项：rejection_sampler、typly_acceptance_sampler


在推测性解码的草稿 token 验证期间指定要使用的接受方法。支持 2 种类型的接受例程：1) RejectionSampler，不允许更改草稿 token 的接受率；2) TypicalAcceptanceSampler，它是可配置的，允许以较低质量为代价获得更高的接受率，反之亦然。


默认值：「rejection_sampler」


`--typical-acceptance-sampler-posterior-threshold`  

设置一个要接受的 token 后验概率的下限阈值。 TypicalAcceptanceSampler 使用此阈值在推测解码期间做出采样决策，默认值为 0.09。


`--typical-acceptance-sampler-posterior-alpha`  

在 TypicalAcceptanceSampler 中，用于 token 接受的基于熵的阈值的缩放因子。通常，默认值为 --typical-acceptance-sampler-posterior-threshold 的平方根，即 0.3。


`--disable-logprobs-during-spec-decoding`  

如果设置为 True，则在推测解码期间不会返回 token 对数概率。如果设置为 False，则根据 SamplingParams 中的设置返回对数概率。如果未指定，则默认为 True。在推测解码期间禁用对数概率可以减少延迟，方法是跳过在提议采样、目标采样和确定接受的 tokens 之后的 logprob 计算。


`--model-loader-extra-config`  

模型加载器的额外配置。这将被传递到与所选 load_format 相对应的模型加载器。这应该是一个将被解析为字典的 JSON 字符串。


`--ignore-patterns`  

加载模型时要忽略的模式。默认为「original/**/*」以避免重复加载 llama 的 checkpoints。


默认值：[]


`--preemption-mode`  

如果设置为 'recompute'，则引擎通过重新计算进行抢占；如果设置为 "swap"，则引擎通过块交换来执行抢占。


`--served-model-name`  

API 中使用的模型名称。如果提供了多个名称，服务器将响应任何提供的名称。响应模型字段中的模型名称将成为此列表中的第一个名称。如果未指定，模型名称将与 –model 参数相同。请注意，此名称也将用于 prometheus 指标的 model_name 标签内容中，如果提供多个​​名称，metricstag 将采用第一个名称。


`--qlora-adapter-name-or-path`  

QLoRA 适配器的名称或路径。


`--otlp-traces-endpoint`  

用于将 OpenTelemetry 追踪数据发送到的目标 URL。


`--collect-detailed-traces`  

有效选项是 model、worker、all。只有当设置了 –otlp-traces-endpoint 时设置此项才有意义。如果设置了此项，它将收集指定模块的详细跟踪。使用可能成本较高或阻塞的操作，因此可能会对性能产生影响。


`--disable-async-output-proc`  

禁用异步输出处理。这可能会导致性能下降。


默认值：False


`--override-neuron-config`  

覆盖或设置神经元设备配置。


`--disable-log-requests`  

禁用日志记录请求。


默认值：False


`--max-log-len`  

日志中打印的最大提示字符数或提示 ID 号。


默认值：Unlimited


## chat completion API 中的工具调用

### 命名函数调用

默认情况下，vLLM 在聊天完成 API 中仅支持命名函数调用。它通过 Outlines 实现这一点，因此默认情况下是启用的，并且可以与任何受支持的模型一起使用。你将获得一个可有效解析的函数调用 —— 但不一定是高质量的。


要使用命名函数，你需要在 chat completion 请求的 `tools` 参数中定义函数，并在 `tool_choice` 参数中指定其中一个工具的 `name`。


### 配置文件

`serve` 模块也可以接受来自 `yaml` 格式配置文件的参数。yaml 中的参数必须使用[这里](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server)概述的参数长格式来指定。


例如：

```yaml
# config.yaml


host: "127.0.0.1"
port: 6379
uvicorn-log-level: "info"
$ vllm serve SOME_MODEL --config config.yaml

---
```
  
**注意**  

如果通过命令行和配置文件提供了参数，则命令行中的值将优先。优先级顺序为 `command line > config file values > defaults`。


---

## chat completion API 中的工具调用

vLLM 默认仅支持 chat completion API 中的命名函数调用。`tool_choice` 选项 `auto` 和 `required` 目前**尚未****支持**，但已在开发计划中。


调用者需要负责向模型提示工具信息，vLLM 不会自动处理这些提示。


vLLM 将使用引导式解码机制，以确保响应内容与 `tools` 参数中定义的工具参数对象的 JSON 模式保持一致。


### 自动函数调用

要启用此功能，你需要设置以下标志：

* `--enable-auto-tool-choice` — **强制**启用自动工具选择。它告诉 vLLM，当模型认为适当时，你希望启用模型自行生成工具调用。

* `--tool-call-parser` — 选择要使用的工具解析器——目前可以选择 `hermes` 或 `mistral`。未来将继续添加其他工具解析器。

* `--chat-template` — 自动工具选择的**可选配置**。指定聊天模板的路径，该模板用于处理 `tool` 角色消息和包含先前生成的工具调用的 `assistant` 角色消息。Hermes 和 Mistral 模型在其 `tokenizer_config.json` 文件中有与工具兼容的聊天模板，但你可以指定自定义模板。如果您的模型在 `tokenizer_config.json` 中配置了特定于工具使用的场景聊天模板，这个参数可以设置为 `tool_use`。这种情况下，它将按照 `transformers` 规范使用。有关更多信息，可以在 HuggingFace 的[相关文档](https://huggingface.co/docs/transformers/en/chat_templating#why-do-some-models-have-multiple-templates)中找到；您可以在 tokenizer_config.json [示例](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/blob/main/tokenizer_config.json)中看到这一点。


如果您喜欢的工具调用模型尚未得到支持，欢迎随时贡献解析器和工具使用聊天模板！


#### Hermes 模型

支持所有 Nous Research Hermes 系列的模型（Hermes 2 Pro 及更高版本）。

* `NousResearch/Hermes-2-Pro-*`

* `NousResearch/Hermes-2-Theta-*`

* `NousResearch/Hermes-3-*`


**注意：**由于合并操作，**Hermes 2 Theta** 模型的工具调用质量和能力据悉已有所降低。


标志：`--tool-call-parser hermes`


#### Mistral 模型

支持的模型：

* `mistralai/Mistral-7B-Instruct-v0.3`（已确认）

* 其他 Mistral 函数调用模型也兼容。


已知问题：

1. Mistral 7B 在正确生成并行工具调用时存在困难。

2. Mistral 的 `tokenizer_config.json` 聊天模板要求工具调用 ID 恰好为 9 位，这比 vLLM 生成的 ID 短的多。由于不符合该条件时会引发异常，因此提供了以下额外的聊天模板：

* `examples/tool_chat_template_mistral.jinja` - 这是「官方」的 Mistral 聊天模板，但经过了调整，以适应 vLLM 的工具调用 ID（提供的 `tool_call_id` 字段被截断为最后 9 位）

* `examples/tool_chat_template_mistral_parallel.jinja` - 这是一个「更好」的版本，当提供工具时，它会添加一个工具使用系统提示，这在处理并行工具调用时可靠性会更高。


推荐标志： `--tool-call-parser mistral --chat-template examples/tool_chat_template_mistral_parallel.jinja`

