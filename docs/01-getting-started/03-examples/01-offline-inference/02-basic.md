---
title: 基础指南
---



[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/basic](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic)

# Basic

`LLM` 类提供了主要的 Python 接口，用于离线推理，即在不使用独立推理服务器的情况下与模型交互。

## 使用方法

示例中的第一个脚本展示了 vLLM 最基本的用法。如果你是 Python 和 vLLM 的新手，建议从这里开始。

```bash
python examples/offline_inference/basic/basic.py
```

其余的脚本包含一个[参数解析器](https://docs.python.org/3/library/argparse.html)，你可以使用它来传递任何与 [`LLM`](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html) 兼容的参数。尝试使用 `--help` 运行脚本，查看所有可用参数列表。

```bash
python examples/offline_inference/basic/classify.py
```

```bash
python examples/offline_inference/basic/embed.py
```

```bash
python examples/offline_inference/basic/score.py
```

聊天（chat）和文本生成（generate）脚本还支持采样参数： `max_tokens`, `temperature`, `top_p` 和 `top_k`。

```bash
python examples/offline_inference/basic/chat.py
```

```bash
python examples/offline_inference/basic/generate.py
```

## 功能
在支持参数传递的脚本中，你可以尝试以下功能。

### 默认生成配置

`--generation-config` 参数用于指定调用 `LLM.get_default_sampling_params()` 时加载生成配置的路径。若设置为 'auto'，则从模型路径加载生成配置。若设置为文件夹路径，则从指定路径加载配置。若不提供该参数，则使用 vLLM 默认值。

> 若生成配置中指定了 max_new_tokens，则会在服务器范围内对所有请求的输出 token 数量施加限制。

尝试以下参数进行测试：

```bash
--generation-config auto
```

### 量化
#### AQLM

vLLM 支持使用 AQLM 量化的模型。

可以使用 `--model` 参数测试以下任一模型：

- `ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf`
- `ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf`
- `ISTA-DASLab/Llama-2-13b-AQLM-2Bit-1x16-hf`
- `ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf`
- `BlackSamorez/TinyLlama-1_1B-Chat-v1_0-AQLM-2Bit-1x16-hf`
 
> 部分模型可能过大，无法在单块 GPU 上运行。你可以通过设置 `--tensor-parallel-size` 来将其拆分到多块 GPU 上运行。

#### GGUF

vLLM 还支持使用 GGUF 量化的模型。

你可以下载一个 GGUF 量化模型，并使用以下参数进行测试：

```python
from huggingface_hub import hf_hub_download
repo_id = "bartowski/Phi-3-medium-4k-instruct-GGUF"
filename = "Phi-3-medium-4k-instruct-IQ2_M.gguf"
print(hf_hub_download(repo_id, filename=filename))
```

```bash
--model {local-path-printed-above} --tokenizer microsoft/Phi-3-medium-4k-instruct
```

### CPU 内存卸载

`--cpu-offload-gb` 参数可以视作一种虚拟扩展 GPU 内存的方式。例如，如果你有一块 24GB 的 GPU，并将该参数设置为 10GB，则可以将其视为一块 34GB 的 GPU，从而加载一个 13B 规模的 BF16 模型（最低需要 26GB GPU 内存）。但需要注意，此方法依赖于高速的 CPU-GPU 互连，因为部分模型会在每次前向传播时从 CPU 内存动态加载到 GPU 内存。

尝试以下参数进行测试：

```bash
--model meta-llama/Llama-2-13b-chat-hf --cpu-offload-gb 10
```

# 示例材料
## basic.py
```python
# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

# Sample prompts.
# 样本提示。
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
# 创建一个采样参数对象。
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
# 创建一个 LLM。
llm = LLM(model="facebook/opt-125m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
# 从提示中生成文本。输出是 RequestOutput 的包含提示，生成的文本和其他信息对象列表。
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
# 打印输出。
print("\nGenerated Outputs:\n" + "-" * 60)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt:    {prompt!r}")
    print(f"Output:    {generated_text!r}")
    print("-" * 60)
```

## chat.py
```python
# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def main(args: dict):
    # Pop arguments not used by LLM
    # 弹出 LLM 未使用的参数
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    chat_template_path = args.pop("chat_template_path")

    # Create an LLM
    # 创建一个 LLM
    llm = LLM(**args)

    # Create sampling params object
    # 创建采样参数对象
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    def print_outputs(outputs):
        print("\nGenerated Outputs:\n" + "-" * 80)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}\n")
            print(f"Generated text: {generated_text!r}")
            print("-" * 80)

    print("=" * 80)

    # In this script, we demonstrate how to pass input to the chat method:
    # 在此脚本中，我们演示了如何将输入传递到 chat 方法:
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
        },
        {
            "role": "user",
            "content":
            "Write an essay about the importance of higher education.",
        },
    ]
    outputs = llm.chat(conversation, sampling_params, use_tqdm=False)
    print_outputs(outputs)

    # You can run batch inference with llm.chat API
    # 您可以使用 llm.chat API 进行批处理推断
    conversations = [conversation for _ in range(10)]

    # We turn on tqdm progress bar to verify it's indeed running batch inference
    # 我们打开 tqdm 进度栏以验证其确实正在运行批处理推断
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
    print_outputs(outputs)

    # A chat template can be optionally supplied.
    # If not, the model will use its default chat template.
    # 可以选择提供聊天模板。
    # 如果没有，该模型将使用其默认聊天模板。
    if chat_template_path is not None:
        with open(chat_template_path) as f:
            chat_template = f.read()

        outputs = llm.chat(
            conversations,
            sampling_params,
            use_tqdm=False,
            chat_template=chat_template,
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    # Add engine args
    # 添加引擎 Args
    engine_group = parser.add_argument_group("Engine arguments")
    EngineArgs.add_cli_args(engine_group)
    engine_group.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    # Add sampling params
    # 添加采样参数
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    # Add example params
    # 添加示例参数
    parser.add_argument("--chat-template-path", type=str)
    args: dict = vars(parser.parse_args())
    main(args)

```

## classify.py
```python
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def main(args: Namespace):
    # Sample prompts.
    # 样本提示。
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create an LLM.
    # You should pass task="classify" for classification models
    # 创建一个 LLM。
    # 您应该传递 task="classify" 给分类模型
    model = LLM(**vars(args))

    # Generate logits. The output is a list of ClassificationRequestOutputs.
    # 生成 logits。输出是 ClassificationRequestOutputs 的列表。
    outputs = model.classify(prompts)

    # Print the outputs.
    # 打印输出。
    print("\nGenerated Outputs:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs):
        probs = output.outputs.probs
        probs_trimmed = ((str(probs[:16])[:-1] +
                          ", ...]") if len(probs) > 16 else probs)
        print(f"Prompt: {prompt!r} \n"
              f"Class Probabilities: {probs_trimmed} (size={len(probs)})")
        print("-" * 60)


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    # 设置示例特定参数
    parser.set_defaults(model="jason9693/Qwen2.5-1.5B-apeach",
                        task="classify",
                        enforce_eager=True)
    args = parser.parse_args()
    main(args)

```

## embed.py
```python
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def main(args: Namespace):
    # Sample prompts.
    # 样本提示。
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create an LLM.
    # You should pass task="embed" for embedding models
    # 创建一个 LLM。
    # 您应该传递 task="embed 给嵌入模型"
    model = LLM(**vars(args))

    # Generate embedding. The output is a list of EmbeddingRequestOutputs.
    # 生成嵌入。输出是 EmbeddingRequestOutputs 的列表。
    outputs = model.embed(prompts)

    # Print the outputs.
    # 打印输出。
    print("\nGenerated Outputs:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs):
        embeds = output.outputs.embedding
        embeds_trimmed = ((str(embeds[:16])[:-1] +
                           ", ...]") if len(embeds) > 16 else embeds)
        print(f"Prompt: {prompt!r} \n"
              f"Embeddings: {embeds_trimmed} (size={len(embeds)})")
        print("-" * 60)


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    # 设置示例特定参数
    parser.set_defaults(model="intfloat/e5-mistral-7b-instruct",
                        task="embed",
                        enforce_eager=True)
    args = parser.parse_args()
    main(args)

```

## generate.py
```python
# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def main(args: dict):
    # Pop arguments not used by LLM
    # 弹出 LLM 未使用的参数
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    # Create an LLM
    # 创建一个 LLM
    llm = LLM(**args)

    # Create a sampling params object
    # 创建一个采样参数对象
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    # 从提示中生成文本。输出是 RequestOutput 的包含提示，生成文本和其他信息的对象列表。
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    # 打印输出。
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    # Add engine args
    # 添加引擎 Args
    engine_group = parser.add_argument_group("Engine arguments")
    EngineArgs.add_cli_args(engine_group)
    engine_group.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    # Add sampling params
    # 添加采样参数
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    args: dict = vars(parser.parse_args())
    main(args)

```

## score.py
```python
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def main(args: Namespace):
    # Sample prompts.
    # 样本提示。
    text_1 = "What is the capital of France?"
    texts_2 = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]

    # Create an LLM.
    # You should pass task="score" for cross-encoder models
    # 创建一个 LLM。
    # 您应该传递 task="score" 给跨编码模型
    model = LLM(**vars(args))

    # Generate scores. The output is a list of ScoringRequestOutputs.
    # 生成分数。输出是 ScoringRequestOutputs 的列表。
    outputs = model.score(text_1, texts_2)

    # Print the outputs.
    # 打印输出。
    print("\nGenerated Outputs:\n" + "-" * 60)
    for text_2, output in zip(texts_2, outputs):
        score = output.outputs.score
        print(f"Pair: {[text_1, text_2]!r} \nScore: {score}")
        print("-" * 60)


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    # 设置示例特定参数
    parser.set_defaults(model="BAAI/bge-reranker-v2-m3",
                        task="score",
                        enforce_eager=True)
    args = parser.parse_args()
    main(args)

```