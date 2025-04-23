---
title: Vision Language Embedding
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/vision_language_embedding.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language_embedding.py)

```python
# SPDX-License-Identifier: Apache-2.0

"""
本示例展示如何使用 vLLM 执行离线推理，并在视觉语言模型上
使用正确的提示格式生成多模态嵌入。
对于大多数模型，提示格式应遵循 HuggingFace 模型库中
相应的示例格式。
"""
from argparse import Namespace
from dataclasses import asdict
from typing import Literal, NamedTuple, Optional, TypedDict, Union, get_args

from PIL.Image import Image

from vllm import LLM, EngineArgs
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser


class TextQuery(TypedDict):
    modality: Literal["text"]
    text: str


class ImageQuery(TypedDict):
    modality: Literal["image"]
    image: Image


class TextImageQuery(TypedDict):
    modality: Literal["text+image"]
    text: str
    image: Image


QueryModality = Literal["text", "image", "text+image"]
Query = Union[TextQuery, ImageQuery, TextImageQuery]


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image: Optional[Image]


def run_e5_v(query: Query) -> ModelRequestData:
    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'  # noqa: E501

    if query["modality"] == "text":
        text = query["text"]
        prompt = llama3_template.format(
            f"{text}\nSummary above sentence in one word: ")
        image = None
    elif query["modality"] == "image":
        prompt = llama3_template.format(
            "<image>\nSummary above image in one word: ")
        image = query["image"]
    else:
        modality = query['modality']
        raise ValueError(f"Unsupported query modality: '{modality}'")

    engine_args = EngineArgs(
        model="royokong/e5-v",
        task="embed",
        max_model_len=4096,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image=image,
    )


def run_vlm2vec(query: Query) -> ModelRequestData:
    if query["modality"] == "text":
        text = query["text"]
        prompt = f"Find me an everyday image that matches the given caption: {text}"  # noqa: E501
        image = None
    elif query["modality"] == "image":
        prompt = "<|image_1|> Find a day-to-day image that looks similar to the provided image."  # noqa: E501
        image = query["image"]
    elif query["modality"] == "text+image":
        text = query["text"]
        prompt = f"<|image_1|> Represent the given image with the following question: {text}"  # noqa: E501
        image = query["image"]
    else:
        modality = query['modality']
        raise ValueError(f"Unsupported query modality: '{modality}'")

    engine_args = EngineArgs(
        model="TIGER-Lab/VLM2Vec-Full",
        task="embed",
        trust_remote_code=True,
        mm_processor_kwargs={"num_crops": 4},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image=image,
    )


def get_query(modality: QueryModality):
    if modality == "text":
        return TextQuery(modality="text", text="A dog sitting in the grass")

    if modality == "image":
        return ImageQuery(
            modality="image",
            image=fetch_image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/American_Eskimo_Dog.jpg/360px-American_Eskimo_Dog.jpg"  # noqa: E501
            ),
        )

    if modality == "text+image":
        return TextImageQuery(
            modality="text+image",
            text="A cat standing in the snow.",
            image=fetch_image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/179px-Felis_catus-cat_on_snow.jpg"  # noqa: E501
            ),
        )

    msg = f"Modality {modality} is not supported."
    raise ValueError(msg)


def run_encode(model: str, modality: QueryModality, seed: Optional[int]):
    query = get_query(modality)
    req_data = model_example_map[model](query)

    engine_args = asdict(req_data.engine_args) | {"seed": seed}
    llm = LLM(**engine_args)

    mm_data = {}
    if req_data.image is not None:
        mm_data["image"] = req_data.image

    outputs = llm.embed({
        "prompt": req_data.prompt,
        "multi_modal_data": mm_data,
    })

    for output in outputs:
        print(output.outputs.embedding)


def main(args: Namespace):
    run_encode(args.model_name, args.modality, args.seed)


model_example_map = {
    "e5_v": run_e5_v,
    "vlm2vec": run_vlm2vec,
}

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for multimodal embedding')
    parser.add_argument('--model-name',
                        '-m',
                        type=str,
                        default="vlm2vec",
                        choices=model_example_map.keys(),
                        help='The name of the embedding model.')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=get_args(QueryModality),
                        help='Modality of the input.')
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Set the seed when initializing `vllm.LLM`.")

    args = parser.parse_args()
    main(args)

```
