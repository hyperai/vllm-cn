---
title: Vision Language Multi Image
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/vision_language_multi_image.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language_multi_image.py)

```python
# SPDX-License-Identifier: Apache-2.0

"""
本示例展示如何使用 vLLM 在视觉语言模型上执行离线推理，
处理多图像输入并生成文本，
整个过程使用模型定义的对话模板(chat template)。
"""
import os
from argparse import Namespace
from dataclasses import asdict
from typing import NamedTuple, Optional

from huggingface_hub import snapshot_download
from PIL.Image import Image
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser

QUESTION = "What is the content of each image?"
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
]


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[Image]
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list[LoRARequest]] = None


# 注意：默认的 `max_num_seqs` 和 `max_model_len` 可能会导致低端 GPU 出现 OOM（内存溢出）。
# 除非另有说明，这些设置已在单张 L4 GPU 上经过测试可正常运行。


def load_aria(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "rhymes-ai/Aria"
    engine_args = EngineArgs(
        model=model_name,
        tokenizer_mode="slow",
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": len(image_urls)},
    )
    placeholders = "<fim_prefix><|img|><fim_suffix>\n" * len(image_urls)
    prompt = (f"<|im_start|>user\n{placeholders}{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = [93532, 93653, 944, 93421, 1019, 93653, 93519]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_deepseek_vl2(question: str,
                      image_urls: list[str]) -> ModelRequestData:
    model_name = "deepseek-ai/deepseek-vl2-tiny"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholder = "".join(f"image_{i}:<image>\n"
                          for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|User|>: {placeholder}{question}\n\n<|Assistant|>:"

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_gemma3(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "google/gemma-3-4b-it"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [{
        "role":
        "user",
        "content": [
            *placeholders,
            {
                "type": "text",
                "text": question
            },
        ],
    }]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_h2ovl(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "h2oai/h2ovl-mississippi-800m"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"max_dynamic_patch": 4},
    )

    placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i, _ in enumerate(image_urls, start=1))
    messages = [{'role': 'user', 'content': f"{placeholders}\n{question}"}]

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)


    # H2OVL-Mississippi 的停止 token
    # https://huggingface.co/h2oai/h2ovl-mississippi-800m
    stop_token_ids = [tokenizer.eos_token_id]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_idefics3(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "HuggingFaceM4/Idefics3-8B-Llama3"

    # 以下配置已确认可以在单个 L40 GPU 上启动。
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=16,
        enforce_eager=True,
        limit_mm_per_prompt={"image": len(image_urls)},
        # 如果您的内存不足，则可以减少 "LINGEST_EDDE"。
        # 请参阅:https://huggingface.co/huggingfacem4/idefics3-8b-llama3#model-optimization
        mm_processor_kwargs={
            "size": {
                "longest_edge": 2 * 364
            },
        },
    )

    placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|begin_of_text|>User:{placeholders}\n{question}<end_of_utterance>\nAssistant:"  # noqa: E501
    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_internvl(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "OpenGVLab/InternVL2-2B"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"max_dynamic_patch": 4},
    )

    placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i, _ in enumerate(image_urls, start=1))
    messages = [{'role': 'user', 'content': f"{placeholders}\n{question}"}]

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Internvl 的停止 token
    # 型号变体可能具有不同的停止 token
    # 请参考正确的"停止单词"的模型卡:
    # https://huggingface.co/opengvlab/internvl2-2b/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_mllama(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # 以下配置已确认可以在单个 L40 GPU 上启动。
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = "<|image|>" * len(image_urls)
    prompt = f"{placeholders}<|begin_of_text|>{question}"
    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_nvlm_d(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "nvidia/NVLM-D-72B"


    # 根据需要进行调整以适合 GPU
    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        tensor_parallel_size=4,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"max_dynamic_patch": 4},
    )

    placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i, _ in enumerate(image_urls, start=1))
    messages = [{'role': 'user', 'content': f"{placeholders}\n{question}"}]

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_pixtral_hf(question: str, image_urls: list[str]) -> ModelRequestData:
    model_name = "mistral-community/pixtral-12b"

    # 根据需要进行调整以适合 GPU
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        tensor_parallel_size=2,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = "[IMG]" * len(image_urls)
    prompt = f"<s>[INST]{question}\n{placeholders}[/INST]"

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_phi3v(question: str, image_urls: list[str]) -> ModelRequestData:

    # num_crops 是传递给多模态图像处理器的一个覆盖（override）关键字参数；
    # 对于某些模型，例如 Phi-3.5-vision-instruct，建议在单帧（single frame）场景下使用 16，
    # 而在多帧（multi-frame）场景下使用 4。
    #
    # 一般来说，较大的 num_crops 值会导致每个图像实例生成更多的 token，
    # 因为它可能在图像预处理过程中对图像进行更大范围的缩放。
    #
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct#loading-the-model-locally
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/main/processing_phi3_v.py#L194
    engine_args = EngineArgs(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"num_crops": 4},
    )
    placeholders = "\n".join(f"<|image_{i}|>"
                             for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|user|>\n{placeholders}\n{question}<|end|>\n<|assistant|>\n"

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
    )


def load_phi4mm(question: str, image_urls: list[str]) -> ModelRequestData:
    """
    Phi-4-multimodal-instruct 支持图像和音频输入。
    此示例展示了如何处理多图像输入。
    """

    model_path = snapshot_download("microsoft/Phi-4-multimodal-instruct")

    # 由于 vision-lora 和 speech-lora 与基本模型共存，所以
    # 我们必须手动指定 Lora 权重的路径。
    vision_lora_path = os.path.join(model_path, "vision-lora")
    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=10000,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
        enable_lora=True,
        max_lora_rank=320,
    )

    placeholders = "".join(f"<|image_{i}|>"
                           for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|user|>{placeholders}{question}<|end|><|assistant|>"

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_image(url) for url in image_urls],
        lora_requests=[LoRARequest("vision", 1, vision_lora_path)],
    )


def load_qwen_vl_chat(question: str,
                      image_urls: list[str]) -> ModelRequestData:
    model_name = "Qwen/Qwen-VL-Chat"
    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=1024,
        max_num_seqs=2,
        hf_overrides={"architectures": ["QwenVLForConditionalGeneration"]},
        limit_mm_per_prompt={"image": len(image_urls)},
    )
    placeholders = "".join(f"Picture {i}: <img></img>\n"
                           for i, _ in enumerate(image_urls, start=1))

    # 此模型在其 tokenizer 上没有 chat_template 属性，
    # 因此，我们需要显式传递它。我们使用 ChatML，因为它已在模型的生成工具中使用。
    # https://huggingface.co/qwen/qwen-vl-chat/blob/main/qwen_generation_utils.py#l265
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)

    # 从:https://huggingface.co/docs/transformers/main/en/chat_templating 复制
    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"  # noqa: E501

    messages = [{'role': 'user', 'content': f"{placeholders}\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True,
                                           chat_template=chat_template)

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
        chat_template=chat_template,
    )


def load_qwen2_vl(question: str, image_urls: list[str]) -> ModelRequestData:
    try:
        from qwen_vl_utils import process_vision_info
    except ModuleNotFoundError:
        print('WARNING: `qwen-vl-utils` not installed, input images will not '
              'be automatically resized. You can enable this functionality by '
              '`pip install qwen-vl-utils`.')
        process_vision_info = None

    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    # 在 L40上测试
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=32768 if process_vision_info is None else 4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role":
        "user",
        "content": [
            *placeholders,
            {
                "type": "text",
                "text": question
            },
        ],
    }]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    if process_vision_info is None:
        image_data = [fetch_image(url) for url in image_urls]
    else:
        image_data, _ = process_vision_info(messages)

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


def load_qwen2_5_vl(question: str, image_urls: list[str]) -> ModelRequestData:
    try:
        from qwen_vl_utils import process_vision_info
    except ModuleNotFoundError:
        print('WARNING: `qwen-vl-utils` not installed, input images will not '
              'be automatically resized. You can enable this functionality by '
              '`pip install qwen-vl-utils`.')
        process_vision_info = None

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=32768 if process_vision_info is None else 4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role":
        "user",
        "content": [
            *placeholders,
            {
                "type": "text",
                "text": question
            },
        ],
    }]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    if process_vision_info is None:
        image_data = [fetch_image(url) for url in image_urls]
    else:
        image_data, _ = process_vision_info(messages,
                                            return_video_kwargs=False)

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


model_example_map = {
    "aria": load_aria,
    "deepseek_vl_v2": load_deepseek_vl2,
    "gemma3": load_gemma3,
    "h2ovl_chat": load_h2ovl,
    "idefics3": load_idefics3,
    "internvl_chat": load_internvl,
    "mllama": load_mllama,
    "NVLM_D": load_nvlm_d,
    "phi3_v": load_phi3v,
    "phi4_mm": load_phi4mm,
    "pixtral_hf": load_pixtral_hf,
    "qwen_vl_chat": load_qwen_vl_chat,
    "qwen2_vl": load_qwen2_vl,
    "qwen2_5_vl": load_qwen2_5_vl,
}


def run_generate(model, question: str, image_urls: list[str],
                 seed: Optional[int]):
    req_data = model_example_map[model](question, image_urls)

    engine_args = asdict(req_data.engine_args) | {"seed": args.seed}
    llm = LLM(**engine_args)

    # 要维护此脚本中的代码兼容性，我们在此处添加 Lora。
    # 您还可以使用:
    # llm.generate(prompts, lora_request=lora_request,...)
    if req_data.lora_requests:
        for lora_request in req_data.lora_requests:
            llm.llm_engine.add_lora(lora_request=lora_request)

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=128,
                                     stop_token_ids=req_data.stop_token_ids)

    outputs = llm.generate(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": {
                "image": req_data.image_data
            },
        },
        sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def run_chat(model: str, question: str, image_urls: list[str],
             seed: Optional[int]):
    req_data = model_example_map[model](question, image_urls)

    engine_args = asdict(req_data.engine_args) | {"seed": seed}
    llm = LLM(**engine_args)

    # 要维护此脚本中的代码兼容性，我们在此处添加 Lora。
    # 您还可以使用:
    # llm.generate(prompts, lora_request=lora_request,...)
    if req_data.lora_requests:
        for lora_request in req_data.lora_requests:
            llm.llm_engine.add_lora(lora_request=lora_request)

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=128,
                                     stop_token_ids=req_data.stop_token_ids)
    outputs = llm.chat(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": question,
                },
                *({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                } for image_url in image_urls),
            ],
        }],
        sampling_params=sampling_params,
        chat_template=req_data.chat_template,
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def main(args: Namespace):
    model = args.model_type
    method = args.method
    seed = args.seed

    if method == "generate":
        run_generate(model, QUESTION, IMAGE_URLS, seed)
    elif method == "chat":
        run_chat(model, QUESTION, IMAGE_URLS, seed)
    else:
        raise ValueError(f"Invalid method: {method}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models that support multi-image input for text '
        'generation')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="phi3_v",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument("--method",
                        type=str,
                        default="generate",
                        choices=["generate", "chat"],
                        help="The method to run in `vllm.LLM`.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Set the seed when initializing `vllm.LLM`.")

    args = parser.parse_args()
    main(args)

```
