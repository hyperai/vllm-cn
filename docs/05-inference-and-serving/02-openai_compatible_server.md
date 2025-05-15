---
title: OpenAI 兼容服务器
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 提供实现了 OpenAI  [Completions API](https://platform.openai.com/docs/api-reference/completions), [Chat API](https://platform.openai.com/docs/api-reference/chat) 等接口的 HTTP 服务器。

您可以通过`vllm serve`命令或 [Docker](https://docs.vllm.ai/en/latest/deployment/docker.html#deployment-docker) 容器启动服务：

```plain
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
```

调用服务时，可使用官方 [OpenAI Python 客户端](https://github.com/openai/openai-python)或任意 HTTP 客户端：

```plain
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

**提示**

vLLM支持部分OpenAI未包含的参数（如`top_k`），可通过在请求的`extra_body`参数中传递，例如：`extra_body={"top_k": 50}`。

重要信息:

默认情况下，服务器会加载 Hugging Face 模型仓库中的`generation_config.json`文件（若存在）。这意味着某些采样参数的默认值可能被模型创建者推荐的配置覆盖。如需禁用此行为，请在启动服务时添加`--generation-config vllm`参数。

## 支持的 API

我们目前支持以下 OpenAI API：

- [Completions API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#completions-api) (`/v1/completions`)

  - 仅适用于[文本生成模型](https://docs.vllm.ai/en/latest/models/generative_models.html) (`--task generate`)。

  - _注意：不支持_`suffix`_参数。_

- [Chat Completions API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#chat-api) (`/v1/chat/completions`)

  - 仅适用于带有[聊天模板](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#chat-template)的[文本生成模型](https://docs.vllm.ai/en/latest/models/generative_models.html) (`--task generate`)。

  - _注意：忽略_`parallel_tool_calls`_和_`user`_参数。_

- [Embeddings API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#embeddings-api) (`/v1/embeddings`)

  - 仅适用于[嵌入模型](https://docs.vllm.ai/en/latest/models/pooling_models.html) (`--task embed`)。

- [Transcriptions API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#transcriptions-api) (`/v1/audio/transcriptions`)

  - 仅适用于自动语音识别 (ASR) 模型 (OpenAI Whisper) (`--task generate`)。

此外，我们还提供以下自定义 API：

- [Tokenizer API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#tokenizer-api) (`/tokenize`, `/detokenize`)

  - 适用于任何带有分词器的模型。

- [Pooling API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#pooling-api) (`/pooling`)

  - 适用于所有[池化模型](https://docs.vllm.ai/en/latest/models/pooling_models.html)。

- [Score API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#score-api) (`/score`)

  - 适用于嵌入模型和[交叉编码器模型](https://docs.vllm.ai/en/latest/models/pooling_models.html) (`--task score`)。

- [Re-rank API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#rerank-api) (`/rerank`, `/v1/rerank`, `/v2/rerank`)

  - 实现 [Jina AI 的 v1 re-rank API](https://jina.ai/reranker/)

  - 同时兼容 [Cohere 的 v1 & v2 re-rank APIs](https://docs.cohere.com/v2/reference/rerank)

  - Jina 和 Cohere 的 API 非常相似；Jina 的 API 在 rerank 端点响应中包含额外信息。

  - 仅适用于[交叉编码器模型](https://docs.vllm.ai/en/latest/models/pooling_models.html) (`--task score`)。

## 聊天模板

为了使语言模型支持聊天协议，vLLM 要求模型在其分词器配置中包含聊天模板。聊天模板是一个 Jinja2 模板，指定了角色、消息和其他聊天特定 token 在输入中的编码方式。

`NousResearch/Meta-Llama-3-8B-Instruct` 的示例聊天模板可以在[这里](https://github.com/meta-llama/llama3?tab=readme-ov-file#instruction-tuned-models)找到。

有些模型即使经过指令/聊天微调也没有提供聊天模板。对于这些模型，您可以在 `--chat-template` 参数中手动指定聊天模板，参数可以是模板文件的路径，也可以是模板字符串形式。没有聊天模板，服务器将无法处理聊天请求，所有聊天请求都会出错。

```plain
vllm serve <model> --chat-template ./path-to-chat-template.jinja
```

vLLM 社区为热门模型提供了一组聊天模板。您可以在[示例](https://github.com/vllm-project/vllm/tree/main/examples)目录下找到它们。

随着多模态聊天 API 的加入，OpenAI 规范现在接受一种新的聊天消息格式，该格式同时指定 `type` 和 `text` 字段。下面是一个示例：

```plain
completion = client.chat.completions.create(
  model="NousResearch/Meta-Llama-3-8B-Instruct",
  messages=[
    {"role": "user", "content": [{"type": "text", "text": "Classify this sentiment: vLLM is wonderful!"}]}
  ]
)
```

大多数 LLM 的聊天模板期望 `content` 字段是字符串，但一些较新的模型如 `meta-llama/Llama-Guard-3-1B` 期望内容按照请求中的 OpenAI 模式格式化。vLLM 提供尽力而为的自动检测支持，这会记录为类似「*Detected the chat template content format to be...\*\*」*的字符串，并在内部转换传入请求以匹配检测到的格式，可能是以下之一：

- `"string"`：字符串。

  - 示例：`"Hello world"`

- `"openai"`：字典列表，类似于 OpenAI 模式。

  - 示例：`[{"type": "text", "text": "Hello world!"}]`

如果结果不符合您的预期，您可以设置 `--chat-template-content-format` CLI 参数来覆盖使用的格式。

## 额外参数

vLLM 支持一组不属于 OpenAI API 的参数。要使用它们，您可以在 OpenAI 客户端中作为额外参数传递。或者，如果您直接使用 HTTP 调用，可以直接将它们合并到 JSON 负载中。

```plain
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

## 额外 HTTP 头

目前仅支持 `X-Request-Id` HTTP 请求头。可以通过 `--enable-request-id-headers` 启用。

> 注意，在高 QPS 率下，启用头可能会显著影响性能。出于这个原因，我们建议在路由器级别（例如通过 Istio）而不是在 vLLM 层实现 HTTP 头。更多详情请参阅此 PR。

```plain
completion = client.chat.completions.create(
  model="NousResearch/Meta-Llama-3-8B-Instruct",
  messages=[
    {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
  ],
  extra_headers={
    "x-request-id": "sentiment-classification-00001",
  }
)
print(completion._request_id)


completion = client.completions.create(
  model="NousResearch/Meta-Llama-3-8B-Instruct",
  prompt="A robot may not injure a human being",
  extra_headers={
    "x-request-id": "completion-test",
  }
)
print(completion._request_id)
```

## CLI 参数参考

### `vllm serve`

`vllm serve` 命令用于启动 OpenAI 兼容服务。

```plain
usage: vllm serve [-h] [--host HOST] [--port PORT]
                  [--uvicorn-log-level {debug,info,warning,error,critical,trace}]
                  [--disable-uvicorn-access-log] [--allow-credentials]
                  [--allowed-origins ALLOWED_ORIGINS]
                  [--allowed-methods ALLOWED_METHODS]
                  [--allowed-headers ALLOWED_HEADERS] [--api-key API_KEY]
                  [--lora-modules LORA_MODULES [LORA_MODULES ...]]
                  [--prompt-adapters PROMPT_ADAPTERS [PROMPT_ADAPTERS ...]]
                  [--chat-template CHAT_TEMPLATE]
                  [--chat-template-content-format {auto,string,openai}]
                  [--response-role RESPONSE_ROLE] [--ssl-keyfile SSL_KEYFILE]
                  [--ssl-certfile SSL_CERTFILE] [--ssl-ca-certs SSL_CA_CERTS]
                  [--enable-ssl-refresh] [--ssl-cert-reqs SSL_CERT_REQS]
                  [--root-path ROOT_PATH] [--middleware MIDDLEWARE]
                  [--return-tokens-as-token-ids]
                  [--disable-frontend-multiprocessing]
                  [--enable-request-id-headers] [--enable-auto-tool-choice]
                  [--tool-call-parser {granite-20b-fc,granite,hermes,internlm,jamba,llama3_json,mistral,pythonic} or name registered in --tool-parser-plugin]
                  [--tool-parser-plugin TOOL_PARSER_PLUGIN] [--model MODEL]
                  [--task {auto,generate,embedding,embed,classify,score,reward,transcription}]
                  [--tokenizer TOKENIZER] [--hf-config-path HF_CONFIG_PATH]
                  [--skip-tokenizer-init] [--revision REVISION]
                  [--code-revision CODE_REVISION]
                  [--tokenizer-revision TOKENIZER_REVISION]
                  [--tokenizer-mode {auto,slow,mistral,custom}]
                  [--trust-remote-code]
                  [--allowed-local-media-path ALLOWED_LOCAL_MEDIA_PATH]
                  [--download-dir DOWNLOAD_DIR]
                  [--load-format {auto,pt,safetensors,npcache,dummy,tensorizer,sharded_state,gguf,bitsandbytes,mistral,runai_streamer,fastsafetensors}]
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
                  [--data-parallel-size DATA_PARALLEL_SIZE]
                  [--enable-expert-parallel]
                  [--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS]
                  [--ray-workers-use-nsight] [--block-size {8,16,32,64,128}]
                  [--enable-prefix-caching | --no-enable-prefix-caching]
                  [--prefix-caching-hash-algo {builtin,sha256}]
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
                  [--device {auto,cuda,neuron,cpu,tpu,xpu,hpu}]
                  [--num-scheduler-steps NUM_SCHEDULER_STEPS]
                  [--use-tqdm-on-load | --no-use-tqdm-on-load]
                  [--multi-step-stream-outputs [MULTI_STEP_STREAM_OUTPUTS]]
                  [--scheduler-delay-factor SCHEDULER_DELAY_FACTOR]
                  [--enable-chunked-prefill [ENABLE_CHUNKED_PREFILL]]
                  [--speculative-config SPECULATIVE_CONFIG]
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
                  [--reasoning-parser {deepseek_r1,granite}]
                  [--disable-cascade-attn] [--disable-log-requests]
                  [--max-log-len MAX_LOG_LEN] [--disable-fastapi-docs]
                  [--enable-prompt-tokens-details]
                  [--enable-server-load-tracking]
```

#### 命名参数

**--host**

主机名。

**​--port**

端口号。

默认值：8000

**​--uvicorn-log-level**

可选值：debug, info, warning, error, critical, trace

Uvicorn 日志级别。

默认值："info"

**​--disable-uvicorn-access-log**

禁用 Uvicorn 访问日志。

默认值：False

**​--allow-credentials**

允许跨域凭证。

默认值：False

**​--allowed-origins**

允许的跨域源。

默认值：['*']

**​--allowed-methods**

允许的 HTTP 方法。

默认值：['*']

**​--allowed-headers**

允许的 HTTP 头。

默认值：['*']

**​--api-key**

若指定，服务端将在请求头中验证此密钥。

**​--lora-modules**

LoRA 模块配置，支持 'name=path' 或 JSON 格式。

旧格式示例：`'name=path'`

新格式示例：`{"name": "name", "path": "lora_path", "base_model_name": "id"}`

**​--prompt-adapters**

Prompt 适配器配置，格式为 name=path。可指定多个适配器。

**​--chat-template**

对话模板文件路径或单行模板字符串。

**​--chat-template-content-format**

可选值：auto, string, openai

消息内容渲染格式：

• "string" 按字符串渲染（示例：`"Hello World"`）

• "openai" 按 OpenAI 字典列表渲染（示例：`[{"type": "text", "text": "Hello world!"}]`）

默认值："auto"

**​--response-role**

当 `request.add_generation_prompt=true` 时的默认返回角色。

默认值：assistant

**​--ssl-keyfile**

SSL 密钥文件路径。

**​--ssl-certfile**

SSL 证书文件路径。

**​--ssl-ca-certs**

CA 证书文件路径。

**​--enable-ssl-refresh**

SSL 证书变更时自动刷新上下文。

默认值：False

**​--ssl-cert-reqs**

客户端证书验证等级（参考 Python ssl 模块）。

默认值：0

**​--root-path**

反向代理路径前缀配置。

**​--middleware**

附加 ASGI 中间件（支持多个 --middleware 参数）。

默认值：[]

**​--return-tokens-as-token-ids**

当指定 --max-logprobs 时，将 token 表示为 'token_id:{token_id}' 格式。

默认值：False

**​--disable-frontend-multiprocessing**

在模型服务进程中运行前端服务。

默认值：False

**​--enable-request-id-headers**

在响应中添加 X-Request-Id 头（高 QPS 时影响性能）。

默认值：False

**​--enable-auto-tool-choice**

启用自动工具选择（需配合 --tool-call-parser 使用）。

默认值：False

**​--tool-call-parser**

工具调用解析器选择（需配合 --enable-auto-tool-choice 使用）。

**​--tool-parser-plugin**

自定义工具解析插件注册名称。

默认值：""

**​--model**

HuggingFace 模型名称或路径。

默认值："facebook/opt-125m"

**​--task**

可选值：auto, generate, embedding, embed, classify, score, reward, transcription

模型任务类型。

默认值："auto"

**​--tokenizer**

HuggingFace 分词器名称或路径。

**​--hf-config-path**

HuggingFace 配置文件路径。

**​--skip-tokenizer-init**

跳过分词器初始化（需自行处理 token 输入）。

默认值：False

**​--revision**

模型版本标识（分支/标签/commit ID）。

**​--code-revision**

模型代码版本标识。

**​--tokenizer-revision**

分词器版本标识。

**​--tokenizer-mode**

可选值：auto, slow, mistral, custom

分词器模式选择。

默认值："auto"

**​--trust-remote-code**

信任远程代码执行。

默认值：False

**​--allowed-local-media-path**

允许读取本地多媒体文件的安全路径（仅限可信环境）。

**​--download-dir**

模型下载存储目录。

**​--load-format**

可选值：auto, pt, safetensors 等 14 种格式

权重加载格式。

默认值："auto"

**​--config-format**

可选值：auto, hf, mistral

配置文件格式。

默认值："ConfigFormat.AUTO"

**​--dtype**

可选值：auto, half, float16 等 6 种精度

模型权重和激活值精度。

默认值："auto"

**​--kv-cache-dtype**

可选值：auto, fp8, fp8_e5m2, fp8_e4m3

KV 缓存精度。

默认值："auto"

**​--max-model-len**

模型上下文长度（自动检测默认值）。

**​--guided-decoding-backend**

引导式解码后端选择（xgrammar/guidance/auto）。

默认值："xgrammar"

**​--logits-processor-pattern**

日志概率处理器正则模式。

**​--model-impl**

可选值：auto, vllm, transformers

模型实现选择。

默认值："auto"

**​--distributed-executor-backend**

可选值：ray, mp, uni, external_launcher

分布式执行后端。

默认值："ray"

**​--pipeline-parallel-size, -pp**

流水线并行度。

默认值：1

**​--tensor-parallel-size, -tp**

张量并行度。

默认值：1

**​--data-parallel-size, -dp**

数据并行度。

默认值：1

**​--enable-expert-parallel**

启用专家并行（MoE 场景）。

默认值：False

**​--max-parallel-loading-workers**

大模型分批次加载参数。

**​--ray-workers-use-nsight**

使用 Nsight 分析 Ray 工作节点。

默认值：False

**​--block-size**

可选值：8, 16, 32, 64, 128

Token 块大小。

默认值：32（CUDA）/128（HPU）

**​--enable-prefix-caching, --no-enable-prefix-caching**

启用前缀缓存。

默认值：True

**​--prefix-caching-hash-algo**

可选值：builtin, sha256

前缀缓存哈希算法。

默认值："builtin"

**​--disable-sliding-window**

禁用滑动窗口机制。

默认值：False

**​--use-v2-block-manager**

[已弃用] 强制使用 V2 块管理器。

默认值：True

**​--num-lookahead-slots**

前瞻槽位数（推测解码实验参数）。

默认值：0

**​--seed**

随机数种子。

**​--swap-space**

每 GPU 的 CPU 交换空间（GiB）。

默认值：4

**​--cpu-offload-gb**

每 GPU 的 CPU 卸载空间（GiB）。

默认值：0

**​--gpu-memory-utilization**

GPU 内存利用率（0-1）。

默认值：0.9

**​--num-gpu-blocks-override**

覆盖 GPU 块数（测试用）。

**​--max-num-batched-tokens**

单次批处理最大 Token 数。

**​--max-num-partial-prefills**

分块预填充最大并发数。

默认值：1

**​--max-long-partial-prefills**

长提示最大并发数。

默认值：1

**​--long-prefill-token-threshold**

长提示判定阈值。

默认值：0

**​--max-num-seqs**

单次迭代最大序列数。

**​--max-logprobs**

最大返回日志概率数。

默认值：20

**​--disable-log-stats**

禁用统计日志。

默认值：False

**​--quantization, -q**

可选值：aqlm, awq 等 20 种量化方法

权重量化方式。

**​--rope-scaling**

RoPE 缩放配置（JSON 格式）。

示例：`{"rope_type":"dynamic","factor":2.0}`

**​--rope-theta**

RoPE theta 参数。

**​--hf-overrides**

HuggingFace 配置覆盖（JSON 格式）。

**​--enforce-eager**

强制使用 Eager 模式。

默认值：False

**​--max-seq-len-to-capture**

CUDA 图捕获最大序列长度。

默认值：8192

**​--disable-custom-all-reduce**

禁用自定义 AllReduce。

默认值：False

**​--tokenizer-pool-size**

异步分词器池大小。

默认值：0

**​--tokenizer-pool-type**

分词器池类型。

默认值："ray"

**​--tokenizer-pool-extra-config**

分词器池额外配置（JSON 格式）。

**​--limit-mm-per-prompt**

多模态输入限制（示例：image=16,video=2）。

**​--mm-processor-kwargs**

多模态处理器参数覆盖（JSON 格式）。

**​--disable-mm-preprocessor-cache**

禁用多模态预处理器缓存。

默认值：False

**​--enable-lora**

启用 LoRA 支持。

默认值：False

**​--enable-lora-bias**

启用 LoRA 偏置项。

默认值：False

**​--max-loras**

单批次最大 LoRA 数量。

默认值：1

**​--max-lora-rank**

最大 LoRA 秩。

默认值：16

**​--lora-extra-vocab-size**

LoRA 额外词汇量上限。

默认值：256

**​--lora-dtype**

可选值：auto, float16, bfloat16

LoRA 数据类型。

默认值："auto"

**​--long-lora-scaling-factors**

长 LoRA 缩放因子配置。

**​--max-cpu-loras**

CPU 内存最大 LoRA 数量。

默认值：等于 max_loras

**​--fully-sharded-loras**

启用全分片 LoRA。

默认值：False

**​--enable-prompt-adapter**

启用 Prompt 适配器。

默认值：False

**​--max-prompt-adapters**

单批次最大 Prompt 适配器数量。

默认值：1

**​--max-prompt-adapter-token**

Prompt 适配器最大 Token 数。

默认值：0

**​--device**

可选值：auto, cuda, neuron 等 6 种设备

运行设备选择。

默认值："auto"

**​--num-scheduler-steps**

调度器单次最大步数。

默认值：1

**​--use-tqdm-on-load, --no-use-tqdm-on-load**

加载进度条显示。

默认值：True

**​--multi-step-stream-outputs**

多步推理流式输出控制。

默认值：True

**​--scheduler-delay-factor**

调度延迟因子。

默认值：0.0

**​--enable-chunked-prefill**

启用分块预填充。

**​--speculative-config**

推测解码配置（JSON 格式）。

**​--speculative-model**

推测解码草稿模型名称。

**​--speculative-model-quantization**

草稿模型量化方式。

**​--num-speculative-tokens**

推测解码 Token 数。

**​--speculative-disable-mqa-scorer**

禁用 MQA 评分器。

默认值：False

**​--speculative-draft-tensor-parallel-size, -spec-draft-tp**

草稿模型张量并行度。

**​--speculative-max-model-len**

草稿模型最大序列长度。

**​--speculative-disable-by-batch-size**

按批次大小禁用推测解码。

**​--ngram-prompt-lookup-max**

N-gram 提示查找最大窗口。

**​--ngram-prompt-lookup-min**

N-gram 提示查找最小窗口。

**​--spec-decoding-acceptance-method**

可选值：rejection_sampler, typical_acceptance_sampler

推测解码验收方法。

默认值："rejection_sampler"

**​--typical-acceptance-sampler-posterior-threshold**

典型采样器后验概率阈值。

默认值：0.09

**​--typical-acceptance-sampler-posterior-alpha**

典型采样器熵缩放因子。

默认值：0.3

**​--disable-logprobs-during-spec-decoding**

推测解码期间禁用日志概率。

默认值：True

**​--model-loader-extra-config**

模型加载器额外配置（JSON 格式）。

**​--ignore-patterns**

模型加载忽略模式。

默认值：[]

**​--preemption-mode**

抢占模式选择（recompute/swap）。

**​--served-model-name**

API 使用的模型名称。

**​--qlora-adapter-name-or-path**

QLoRA 适配器名称或路径。

**​--show-hidden-metrics-for-version**

显示指定版本的隐藏指标。

**​--otlp-traces-endpoint**

OpenTelemetry 追踪端点。

**​--collect-detailed-traces**

详细追踪配置（model/worker/all）。

**​--disable-async-output-proc**

禁用异步输出处理。

默认值：False

**​--scheduling-policy**

可选值：fcfs, priority

调度策略选择。

默认值："fcfs"

**​--scheduler-cls**

调度器类路径。

默认值："vllm.core.scheduler.Scheduler"

**​--override-neuron-config**

Neuron 设备配置覆盖（JSON 格式）。

**​--override-pooler-config**

池化方法配置覆盖（JSON 格式）。

**​--compilation-config, -O**

模型编译配置（数字等级或 JSON）。

**​--kv-transfer-config**

分布式 KV 缓存传输配置（JSON 格式）。

**​--worker-cls**

分布式工作节点类。

默认值："auto"

**​--worker-extension-cls**

工作节点扩展类。

默认值：""

**​--generation-config**

生成配置文件路径。

默认值："auto"

**​--override-generation-config**

生成配置覆盖（JSON 格式）。

**​--enable-sleep-mode**

启用引擎睡眠模式（仅限 CUDA）。

默认值：False

**​--calculate-kv-scales**

动态计算 FP8 KV 缩放因子。

默认值：False

**​--additional-config**

平台特定附加配置（JSON 格式）。

**​--enable-reasoning**

启用推理内容生成。

默认值：False

**​--reasoning-parser**

可选值：deepseek_r1, granite

推理内容解析器选择。

**​--disable-cascade-attn**

禁用级联注意力。

默认值：False

**​--disable-log-requests**

禁用请求日志。

默认值：False

**​--max-log-len**

日志最大显示长度。

默认值：无限制

**​--disable-fastapi-docs**

禁用 API 文档。

默认值：False

**​--enable-prompt-tokens-details**

启用详细 Token 统计。

默认值：False

**​--enable-server-load-tracking**

启用服务负载监控。

默认值：False

#### 配置文件

您可以通过 [YAML](https://yaml.org/) 配置文件加载 CLI 参数。参数名称必须使用[前文](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#vllm-serve)所述的长格式。

例如：

```plain
# config.yaml


host: "127.0.0.1"
port: 6379
uvicorn-log-level: "info"
```

使用上述配置文件：

```plain
vllm serve SOME_MODEL --config config.yaml
```

> **注意**
> 
> 如果同时通过命令行和配置文件提供参数，命令行参数值将优先。优先级顺序为：`命令行 > 配置文件值 > 默认值`。

## API 参考

### Completions API

我们的 Completions API 兼容 [OpenAI 的 Completions API](https://platform.openai.com/docs/api-reference/completions)，您可以使用 [官方 OpenAI Python 客户端](https://github.com/openai/openai-python) 与其交互。

代码示例：[examples/online_serving/openai_completion_client.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_completion_client.py)

#### 额外参数

支持以下[采样参数](https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-params)：

```plain
    use_beam_search: bool = False
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    length_penalty: float = 1.0
    stop_token_ids: Optional[list[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    allowed_token_ids: Optional[list[int]] = None
    prompt_logprobs: Optional[int] = None
```

支持以下额外参数：

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
         "output. Only {'type': 'json_object'}, {'type': 'json_schema'} or "
         "{'type': 'text' } is supported."),
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
    guided_choice: Optional[list[str]] = Field(
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
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."))
    logits_processors: Optional[LogitsProcessors] = Field(
        default=None,
        description=(
            "A list of either qualified names of logits processors, or "
            "constructor objects, to apply when sampling. A constructor is "
            "a JSON object with a required 'qualname' field specifying the "
            "qualified name of the processor class/factory, and optional "
            "'args' and 'kwargs' fields containing positional and keyword "
            "arguments. For example: {'qualname': "
            "'my_module.MyLogitsProcessor', 'args': [1, 2], 'kwargs': "
            "{'param': 'value'}}."))
    return_tokens_as_token_ids: Optional[bool] = Field(
        default=None,
        description=(
            "If specified with 'logprobs', tokens are represented "
            " as strings of the form 'token_id:{token_id}' so that tokens "
            "that are not JSON-encodable can be identified."))


```

### Chat API

我们的 Chat API 兼容 [OpenAI 的 Chat Completions API](https://platform.openai.com/docs/api-reference/chat)，您可以使用 [官方 OpenAI Python 客户端](https://github.com/openai/openai-python) 与其交互。

我们同时支持 [视觉](https://platform.openai.com/docs/guides/vision) 和 [音频](https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-in) 相关参数，详见 [多模态输入](https://docs.vllm.ai/en/latest/serving/multimodal_inputs.html#multimodal-inputs) 指南。

- _注意：不支持_`image_url.detail`_参数。_

代码示例：[examples/online_serving/openai_chat_completion_client.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_client.py)

#### 额外参数

支持以下[采样参数](https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-params)。

```plain
    best_of: Optional[int] = None
    use_beam_search: bool = False
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    length_penalty: float = 1.0
    stop_token_ids: Optional[list[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    prompt_logprobs: Optional[int] = None
```

支持以下额外参数：

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
    continue_final_message: bool = Field(
        default=False,
        description=
        ("If this is set, the chat will be formatted so that the final "
         "message in the chat is open-ended, without any EOS tokens. The "
         "model will continue this message rather than starting a new one. "
         "This allows you to \"prefill\" part of the model's response for it. "
         "Cannot be used at the same time as `add_generation_prompt`."),
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
    documents: Optional[list[dict[str, str]]] = Field(
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
    chat_template_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    mm_processor_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
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
    guided_choice: Optional[list[str]] = Field(
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
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."))
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."))
    logits_processors: Optional[LogitsProcessors] = Field(
        default=None,
        description=(
            "A list of either qualified names of logits processors, or "
            "constructor objects, to apply when sampling. A constructor is "
            "a JSON object with a required 'qualname' field specifying the "
            "qualified name of the processor class/factory, and optional "
            "'args' and 'kwargs' fields containing positional and keyword "
            "arguments. For example: {'qualname': "
            "'my_module.MyLogitsProcessor', 'args': [1, 2], 'kwargs': "
            "{'param': 'value'}}."))
    return_tokens_as_token_ids: Optional[bool] = Field(
        default=None,
        description=(
            "If specified with 'logprobs', tokens are represented "
            " as strings of the form 'token_id:{token_id}' so that tokens "
            "that are not JSON-encodable can be identified."))


```

### Embeddings API

我们的 Embeddings API 兼容 [OpenAI 的 Embeddings API](https://platform.openai.com/docs/api-reference/embeddings)，您可以使用 [官方 OpenAI Python 客户端](https://github.com/openai/openai-python) 与其交互。

如果模型具有 [聊天模板](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#chat-template)，您可以用 `messages` 列表（与 [Chat API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#chat-api) 相同格式）替换 `inputs`，这些消息将被视为模型的单个提示。

代码示例：[examples/online_serving/openai_embedding_client.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_embedding_client.py)

#### 多模态输入

您可以通过为服务器定义自定义聊天模板并在请求中传递 `messages` 列表，向嵌入模型传递多模态输入。参考以下示例：

**VLM2Vec**

启动模型服务：

```plain
vllm serve TIGER-Lab/VLM2Vec-Full --task embed \
  --trust-remote-code --max-model-len 4096 --chat-template examples/template_vlm2vec.jinja
```

> **重要信息**
> 
> 由于 VLM2Vec 与 Phi-3.5-Vision 具有相同的模型架构，我们必须显式传递 `--task embed` 以在嵌入模式下运行此模型而非文本生成模式。
>
> 自定义聊天模板与此模型的原始模板完全不同，可在此处找到：[examples/template_vlm2vec.jinja](https://github.com/vllm-project/vllm/blob/main/examples/template_vlm2vec.jinja)

由于 OpenAI 客户端未定义请求模式，我们使用底层 `requests` 库向服务器发送请求：

```plain
import requests


image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


response = requests.post(
    "http://localhost:8000/v1/embeddings",
    json={
        "model": "TIGER-Lab/VLM2Vec-Full",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Represent the given image."},
            ],
        }],
        "encoding_format": "float",
    },
)
response.raise_for_status()
response_json = response.json()
print("Embedding output:", response_json["data"][0]["embedding"])
```

**DSE-Qwen2-MRL**

要为模型提供服务，请执行以下操作：

```python
vllm serve MrLight/dse-qwen2-2b-mrl-v1 --task embed \
  --trust-remote-code --max-model-len 8192 --chat-template examples/template_dse_qwen2_vl.jinja
```

> **重要信息**
> 
> 与 VLM2Vec 一样，我们必须显式地传递 `--task embed`。
>
> 此外，`MrLight/dse-qwen2-2b-mrl-v1` 需要一个 EOS 令牌进行嵌入，这由自定义聊天模板处理：[examples/template_dse_qwen2_vl.jinja](https://github.com/vllm-project/vllm/blob/main/examples/template_dse_qwen2_vl.jinja)。

> **重要信息**
>
>`MrLight/dse-qwen2-2b-mrl-v1` 需要文本查询嵌入的最小图像大小的占位符图像。有关详细信息，请参阅下面的完整代码示例。

#### 额外参数

支持以下 [池化参数](https://docs.vllm.ai/en/latest/api/inference_params.html#pooling-params)。

```plain
    additional_data: Optional[Any] = None
```

默认支持以下额外参数：

```plain
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."))


```

对于类聊天输入（即传递 `messages` 时），改为支持以下额外参数：

```plain
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."),
    )
    chat_template_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    mm_processor_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."))
```

### Transcriptions API

我们的 Transcriptions API 兼容 [OpenAI 的 Transcriptions API](https://platform.openai.com/docs/api-reference/audio/createTranscription)，您可以使用 [官方 OpenAI Python 客户端](https://github.com/openai/openai-python) 与其交互。

> **注意**
> 
> 要使用 Transcriptions API，请通过 `pip install vllm[audio]` 安装额外的音频依赖。

代码示例：[examples/online_serving/openai_transcription_client.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_transcription_client.py)

### Tokenizer API

我们的 Tokenizer API 是 [HuggingFace 风格分词器](https://huggingface.co/docs/transformers/en/main_classes/tokenizer) 的简单封装，包含两个端点：

- `/tokenize` 对应调用 `tokenizer.encode()`
- `/detokenize` 对应调用 `tokenizer.decode()`

### Pooling API

我们的 Pooling API 使用 [池化模型](https://docs.vllm.ai/en/latest/models/pooling_models.html) 对输入提示进行编码，并返回相应的隐藏状态。

输入格式与 [Embeddings API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#embeddings-api) 相同，但输出数据可以包含任意嵌套列表，而不仅是一维浮点数列表。

代码示例：[examples/online_serving/openai_pooling_client.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_pooling_client.py)

### Score API

我们的 Score API 可以应用交叉编码器模型或嵌入模型来预测句子对的分数。使用嵌入模型时，分数对应于每对嵌入之间的余弦相似度。通常，句子对的分数表示两个句子之间的相似度，范围在 0 到 1 之间。

交叉编码器模型的文档见 [sbert.net](https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html)。

代码示例：[examples/online_serving/openai_cross_encoder_score.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_cross_encoder_score.py)

#### 单次推理 (Single inference)

您可以向 `text_1` 和 `text_2` 传递字符串，形成一个句子对。

请求：

```plain
curl -X 'POST' \
  'http://127.0.0.1:8000/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "BAAI/bge-reranker-v2-m3",
  "encoding_format": "float",
  "text_1": "What is the capital of France?",
  "text_2": "The capital of France is Paris."
}'
```

响应：

```plain
{
  "id": "score-request-id",
  "object": "list",
  "created": 693447,
  "model": "BAAI/bge-reranker-v2-m3",
  "data": [
    {
      "index": 0,
      "object": "score",
      "score": 1
    }
  ],
  "usage": {}
}
```

#### 批量推理

您可以向 `text_1` 传递字符串，向 `text_2` 传递列表，形成多个句子对，每对由 `text_1` 和 `text_2` 中的一个字符串组成。总对数等于 `len(text_2)`。

请求：

```plain
curl -X 'POST' \
  'http://127.0.0.1:8000/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "BAAI/bge-reranker-v2-m3",
  "text_1": "What is the capital of France?",
  "text_2": [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris."
  ]
}'
```

响应：

```plain
{
  "id": "score-request-id",
  "object": "list",
  "created": 693570,
  "model": "BAAI/bge-reranker-v2-m3",
  "data": [
    {
      "index": 0,
      "object": "score",
      "score": 0.001094818115234375
    },
    {
      "index": 1,
      "object": "score",
      "score": 1
    }
  ],
  "usage": {}
}
```

您可以向 `text_1` 和 `text_2` 都传递列表，形成多个句子对，每对由 `text_1` 中的一个字符串和 `text_2` 中对应的字符串组成（类似 `zip()`）。总对数等于 `len(text_2)`。

请求：

```plain
curl -X 'POST' \
  'http://127.0.0.1:8000/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "BAAI/bge-reranker-v2-m3",
  "encoding_format": "float",
  "text_1": [
    "What is the capital of Brazil?",
    "What is the capital of France?"
  ],
  "text_2": [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris."
  ]
}'
```

响应：

```plain
{
  "id": "score-request-id",
  "object": "list",
  "created": 693447,
  "model": "BAAI/bge-reranker-v2-m3",
  "data": [
    {
      "index": 0,
      "object": "score",
      "score": 1
    },
    {
      "index": 1,
      "object": "score",
      "score": 1
    }
  ],
  "usage": {}
}
```

#### 额外参数

支持以下[池化参数](https://docs.vllm.ai/en/latest/api/inference_params.html#pooling-params)。

```plain
    additional_data: Optional[Any] = None
```

支持以下额外参数：

```plain
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."))


```

### Re-rank API

我们的 Re-rank API 可以应用嵌入模型或交叉编码器模型来预测单个查询与文档列表中每个文档之间的相关分数。通常，句子对的分数表示两个句子之间的相似度，范围在0到1之间。

交叉编码器模型的文档见 [sbert.net](https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html)。

rerank端点支持流行的重排序模型如 `BAAI/bge-reranker-base` 和其他支持 `score` 任务的模型。此外，`/rerank`、`/v1/rerank` 和 `/v2/rerank` 端点兼容 [Jina AI的重排序API接口](https://jina.ai/reranker/) 和 [Cohere的重排序API接口](https://docs.cohere.com/v2/reference/rerank)，以确保与流行开源工具的兼容性。

代码示例：[examples/online_serving/jinaai_rerank_client.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/jinaai_rerank_client.py)

#### 示例请求

注意 `top_n` 请求参数是可选的，默认为 `documents` 字段的长度。结果文档将按相关性排序，`index` 属性可用于确定原始顺序。

请求：

```plain
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/rerank' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "BAAI/bge-reranker-base",
  "query": "What is the capital of France?",
  "documents": [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris.",
    "Horses and cows are both animals"
  ]
}'
```

响应：

```plain
{
  "id": "rerank-fae51b2b664d4ed38f5969b612edff77",
  "model": "BAAI/bge-reranker-base",
  "usage": {
    "total_tokens": 56
  },
  "results": [
    {
      "index": 1,
      "document": {
        "text": "The capital of France is Paris."
      },
      "relevance_score": 0.99853515625
    },
    {
      "index": 0,
      "document": {
        "text": "The capital of Brazil is Brasilia."
      },
      "relevance_score": 0.0005860328674316406
    }
  ]
}
```

#### 额外参数

支持以下[池化参数](https://docs.vllm.ai/en/latest/api/inference_params.html#pooling-params)。

```plain
    additional_data: Optional[Any] = None
```

支持以下额外参数：

```plain
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."))
```
