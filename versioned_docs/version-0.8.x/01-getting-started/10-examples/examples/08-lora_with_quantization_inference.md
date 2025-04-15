---
title: 带量化的 LoRA 推理
---


源代码: [vllm-project/vllm](https://raw.githubusercontent.com/vllm-project/vllm/main/examples/lora_with_quantization_inference.py)

```python
"""
This example shows how to use LoRA with different quantization techniques
for offline inference.

Requires HuggingFace credentials for access.
该示例展示了如何在离线推理中使用不同的量化技术进行 LoRA（低秩适配）。    
访问需要 HuggingFace 凭据。

"""

import gc
from typing import List, Optional, Tuple

import torch
from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


def create_test_prompts(
        lora_path: str
) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    return [
        # this is an example of using quantization without LoRA
        # 这是一个不使用 LoRA 量化的示例
        ("My name is",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128), None),
        # the next three examples use quantization with LoRA
        # 接下来的三个示例使用了 LoRA 量化
        ("my name is",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128),
         LoRARequest("lora-test-1", 1, lora_path)),
        ("The capital of USA is",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128),
         LoRARequest("lora-test-2", 1, lora_path)),
        ("The capital of France is",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128),
         LoRARequest("lora-test-3", 1, lora_path)),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams,
                                              Optional[LoRARequest]]]):
    """Continuously process a list of prompts and handle the outputs."""
    """持续处理一系列提示并处理输出"""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print("----------------------------------------------------")
                print(f"Prompt: {request_output.prompt}")
                print(f"Output: {request_output.outputs[0].text}")


def initialize_engine(model: str, quantization: str,
                      lora_repo: Optional[str]) -> LLMEngine:
    """Initialize the LLMEngine."""
    """初始化 LLMEngine"""

    if quantization == "bitsandbytes":
        # QLoRA (https://arxiv.org/abs/2305.14314) is a quantization technique.
        # It quantizes the model when loading, with some config info from the
        # LoRA adapter repo. So need to set the parameter of load_format and
        # qlora_adapter_name_or_path as below.
        # QLoRA（https://arxiv.org/abs/2305.14314）是一种量化技术。
        # 它在加载模型时进行量化，并使用来自 LoRA 适配器库的一些配置信息。
        #因此，需要按以下方式设置 load_format 和 qlora_adapter_name_or_path 参数。



        engine_args = EngineArgs(
            model=model,
            quantization=quantization,
            qlora_adapter_name_or_path=lora_repo,
            load_format="bitsandbytes",
            enable_lora=True,
            max_lora_rank=64,
            # set it only in GPUs of limited memory
            # 仅在内存有限的 GPU 上设置它
            enforce_eager=True)
    else:
        engine_args = EngineArgs(
            model=model,
            quantization=quantization,
            enable_lora=True,
            max_loras=4,
            # set it only in GPUs of limited memory
            # 仅在内存有限的 GPU 上设置它
            enforce_eager=True)
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Main function that sets up and runs the prompt processing."""
    """主函数，用于设置和运行提示处理"""

    test_configs = [{
        "name": "qlora_inference_example",
        'model': "huggyllama/llama-7b",
        'quantization': "bitsandbytes",
        'lora_repo': 'timdettmers/qlora-flan-7b'
    }, {
        "name": "AWQ_inference_with_lora_example",
        'model': 'TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ',
        'quantization': "awq",
        'lora_repo': 'jashing/tinyllama-colorist-lora'
    }, {
        "name": "GPTQ_inference_with_lora_example",
        'model': 'TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ',
        'quantization': "gptq",
        'lora_repo': 'jashing/tinyllama-colorist-lora'
    }]

    for test_config in test_configs:
        print(
            f"~~~~~~~~~~~~~~~~ Running: {test_config['name']} ~~~~~~~~~~~~~~~~"
        )
        engine = initialize_engine(test_config['model'],
                                   test_config['quantization'],
                                   test_config['lora_repo'])
        lora_path = snapshot_download(repo_id=test_config['lora_repo'])
        test_prompts = create_test_prompts(lora_path)
        process_requests(engine, test_prompts)

        # Clean up the GPU memory for the next test
        # 清理 GPU 内存以进行下一次测试
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
```


