---
title: Audio Language
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/audio_language.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/audio_language.py)

```python
# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference
with the correct prompt format on audio language models.
For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
"""
此示例显示了如何使用 vLLM 进行离线推理运行
使用音频语言模型的正确及时格式。
对于大多数型号，及时格式应遵循相应的示例
在 HuggingFace 模型存储库上。
"""
import os
from dataclasses import asdict
from typing import NamedTuple, Optional

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.lora.request import LoRARequest
from vllm.utils import FlexibleArgumentParser

audio_assets = [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]
question_per_audio_count = {
    0: "What is 1+1?",
    1: "What is recited in the audio?",
    2: "What sport and what nursery rhyme are referenced?"
}


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.
# 注意：默认的 `max_num_seqs` 和 `max_model_len` 可能会导致低端 GPU 出现 OOM（内存溢出）。
# 除非另有说明，这些设置已在单张 L4 GPU 上经过测试可正常运行。


# MiniCPM-O
def run_minicpmo(question: str, audio_count: int) -> ModelRequestData:
    model_name = "openbmb/MiniCPM-o-2_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": audio_count},
    )

    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    audio_placeholder = "(<audio>./</audio>)" * audio_count
    audio_chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>' }}{% endif %}"  # noqa: E501
    messages = [{
        'role': 'user',
        'content': f'{audio_placeholder}\n{question}'
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True,
                                           chat_template=audio_chat_template)

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
    )


# Phi-4-multimodal-instruct
def run_phi4mm(question: str, audio_count: int) -> ModelRequestData:
    """
    Phi-4-multimodal-instruct supports both image and audio inputs. Here, we
    show how to process audio inputs.
    """
    model_path = snapshot_download("microsoft/Phi-4-multimodal-instruct")
    # Since the vision-lora and speech-lora co-exist with the base model,
    # we have to manually specify the path of the lora weights.
    # 由于 vision-lora 和 speech-lora 与基本模型共存，所以
    # 我们必须手动指定 lora 权重的路径。
    speech_lora_path = os.path.join(model_path, "speech-lora")
    placeholders = "".join([f"<|audio_{i+1}|>" for i in range(audio_count)])

    prompts = f"<|user|>{placeholders}{question}<|end|><|assistant|>"

    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        enable_lora=True,
        max_lora_rank=320,
        limit_mm_per_prompt={"audio": audio_count},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompts,
        lora_requests=[LoRARequest("speech", 1, speech_lora_path)],
    )


# Qwen2-Audio
def run_qwen2_audio(question: str, audio_count: int) -> ModelRequestData:
    model_name = "Qwen/Qwen2-Audio-7B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": audio_count},
    )

    audio_in_prompt = "".join([
        f"Audio {idx+1}: "
        f"<|audio_bos|><|AUDIO|><|audio_eos|>\n" for idx in range(audio_count)
    ])

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n"
              f"{audio_in_prompt}{question}<|im_end|>\n"
              "<|im_start|>assistant\n")

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
    )


# Ultravox 0.5-1B
# Ultravox 0.5-1b
def run_ultravox(question: str, audio_count: int) -> ModelRequestData:
    model_name = "fixie-ai/ultravox-v0_5-llama-3_2-1b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [{
        'role': 'user',
        'content': "<|audio|>\n" * audio_count + question
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        trust_remote_code=True,
        limit_mm_per_prompt={"audio": audio_count},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
    )


# Whisper
# 耳语
def run_whisper(question: str, audio_count: int) -> ModelRequestData:
    assert audio_count == 1, (
        "Whisper only support single audio input per prompt")
    model_name = "openai/whisper-large-v3-turbo"

    prompt = "<|startoftranscript|>"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=448,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": audio_count},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
    )


model_example_map = {
    "minicpmo": run_minicpmo,
    "phi4_mm": run_phi4mm,
    "qwen2_audio": run_qwen2_audio,
    "ultravox": run_ultravox,
    "whisper": run_whisper,
}


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    audio_count = args.num_audios
    req_data = model_example_map[model](question_per_audio_count[audio_count],
                                        audio_count)

    engine_args = asdict(req_data.engine_args) | {"seed": args.seed}
    llm = LLM(**engine_args)

    # To maintain code compatibility in this script, we add LoRA here.
    # You can also add LoRA using:
    # llm.generate(prompts, lora_request=lora_request,...)
    # 要维护此脚本中的代码兼容性，我们在此处添加 Lora。
    # 您还可以使用:
    # llm.generate (提示，lora_request = lora_request，...)
    if req_data.lora_requests:
        for lora_request in req_data.lora_requests:
            llm.llm_engine.add_lora(lora_request=lora_request)

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    # 我们将温度设置为0.2，以便输出可能不同
    # 即使在运行批处理推理时所有提示都相同。
    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=64,
                                     stop_token_ids=req_data.stop_token_ids)

    mm_data = {}
    if audio_count > 0:
        mm_data = {
            "audio": [
                asset.audio_and_sample_rate
                for asset in audio_assets[:audio_count]
            ]
        }

    assert args.num_prompts > 0
    inputs = {"prompt": req_data.prompt, "multi_modal_data": mm_data}
    if args.num_prompts > 1:
        # Batch inference
        # 批次推理
        inputs = [inputs] * args.num_prompts

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'audio language models')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="ultravox",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')
    parser.add_argument("--num-audios",
                        type=int,
                        default=1,
                        choices=[0, 1, 2],
                        help="Number of audio items per prompt.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Set the seed when initializing `vllm.LLM`.")

    args = parser.parse_args()
    main(args)

```
