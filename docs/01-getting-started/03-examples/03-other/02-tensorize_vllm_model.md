---
title: Tensorize Vllm Model
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/other/tensorize_vllm_model.py](https://github.com/vllm-project/vllm/blob/main/examples/other/tensorize_vllm_model.py)

```python
# SPDX-License-Identifier: Apache-2.0

import argparse
import dataclasses
import json
import os
import uuid

from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.tensorizer import (TensorizerArgs,
                                                         TensorizerConfig,
                                                         tensorize_vllm_model)
from vllm.utils import FlexibleArgumentParser

# yapf 与本文档字符串的 isort 存在格式冲突
# yapf: disable

"""
tensorize_vllm_model.py 脚本可用于序列化和反序列化 vLLM 模型。这些模型可以通过 tensorizer
极速加载到 GPU，支持通过 HTTP/HTTPS 端点、S3 端点或本地路径加载。同时支持张量加密解密功能
（需安装 libsodium）。安装支持 tensorizer 的 vLLM：`pip install vllm[tensorizer]`。
了解更多 tensorizer 信息请访问：https://github.com/coreweave/tensorizer

序列化模型：
1. 从源码安装 vLLM
2. 在项目根目录运行：
python -m examples.other.tensorize_vllm_model \\
   --model facebook/opt-125m \\
   serialize \\
   --serialized-directory s3://my-bucket \\
   --suffix v1

该命令会从 HuggingFace 下载模型，加载到 vLLM 中序列化后保存到 S3 存储桶（也可使用本地目录）。
需要设置以下环境变量：
`S3_ACCESS_KEY_ID`、`S3_SECRET_ACCESS_KEY` 和 `S3_ENDPOINT_URL`。
或直接通过命令行参数提供：
`--s3-access-key-id`、`--s3-secret-access-key` 和 `--s3-endpoint`。
使用 `--keyfile` 参数可加密模型权重。

反序列化模型：
python -m examples.other.tensorize_vllm_model \\
   --model EleutherAI/gpt-j-6B \\
   --dtype float16 \\
   deserialize \\
   --path-to-tensors s3://my-bucket/vllm/EleutherAI/gpt-j-6B/v1/model.tensors

该命令会从 S3 存储桶下载模型张量并反序列化。若序列化时使用了加密，可通过 `--keyfile` 参数解密。

分布式张量并行支持：
每个模型分片会序列化为单独文件。tensorizer_uri 应指定为包含格式说明符（如 '%03d'）的字符串模板。
分片模型序列化后的命名格式为：model-rank-%03d.tensors

查看完整参数：
序列化：`python -m examples.other.tensorize_vllm_model serialize --help`
反序列化：`python -m examples.other.tensorize_vllm_model deserialize --help`

序列化后模型加载示例：
    llm = LLM(model="facebook/opt-125m",
              load_format="tensorizer",
              model_loader_extra_config=TensorizerConfig(
                    tensorizer_uri = path_to_tensors,
                    num_readers=3,
                    )
              )

vLLM OpenAI 推理服务器也支持加载序列化模型。`model_loader_extra_config` 对应 CLI 参数
`--model-loader-extra-config`，接受 TensorizerConfig 参数的 JSON 字符串。

查看所有 tensorizer 配置参数：
运行 `python -m examples.other.tensorize_vllm_model deserialize --help`
在 `tensorizer options` 部分查看。注意 `--tensorizer-uri` 和 `--path-to-tensors`
在本脚本中功能相同。
"""

def parse_args():
    parser = FlexibleArgumentParser(
        description="An example script that can be used to serialize and "
        "deserialize vLLM models. These models "
        "can be loaded using tensorizer directly to the GPU "
        "extremely quickly. Tensor encryption and decryption is "
        "also supported, although libsodium must be installed to "
        "use it.")
    parser = EngineArgs.add_cli_args(parser)
    subparsers = parser.add_subparsers(dest='command')

    serialize_parser = subparsers.add_parser(
        'serialize', help="Serialize a model to `--serialized-directory`")

    serialize_parser.add_argument(
        "--suffix",
        type=str,
        required=False,
        help=(
            "The suffix to append to the serialized model directory, which is "
            "used to construct the location of the serialized model tensors, "
            "e.g. if `--serialized-directory` is `s3://my-bucket/` and "
            "`--suffix` is `v1`, the serialized model tensors will be "
            "saved to "
            "`s3://my-bucket/vllm/EleutherAI/gpt-j-6B/v1/model.tensors`. "
            "If none is provided, a random UUID will be used."))
    serialize_parser.add_argument(
        "--serialized-directory",
        type=str,
        required=True,
        help="The directory to serialize the model to. "
        "This can be a local directory or S3 URI. The path to where the "
        "tensors are saved is a combination of the supplied `dir` and model "
        "reference ID. For instance, if `dir` is the serialized directory, "
        "and the model HuggingFace ID is `EleutherAI/gpt-j-6B`, tensors will "
        "be saved to `dir/vllm/EleutherAI/gpt-j-6B/suffix/model.tensors`, "
        "where `suffix` is given by `--suffix` or a random UUID if not "
        "provided.")

    serialize_parser.add_argument(
        "--keyfile",
        type=str,
        required=False,
        help=("Encrypt the model weights with a randomly-generated binary key,"
              " and save the key at this path"))

    deserialize_parser = subparsers.add_parser(
        'deserialize',
        help=("Deserialize a model from `--path-to-tensors`"
              " to verify it can be loaded and used."))

    deserialize_parser.add_argument(
        "--path-to-tensors",
        type=str,
        required=True,
        help="The local path or S3 URI to the model tensors to deserialize. ")

    deserialize_parser.add_argument(
        "--keyfile",
        type=str,
        required=False,
        help=("Path to a binary key to use to decrypt the model weights,"
              " if the model was serialized with encryption"))

    TensorizerArgs.add_cli_args(deserialize_parser)

    return parser.parse_args()



def deserialize():
    llm = LLM(model=args.model,
              load_format="tensorizer",
              tensor_parallel_size=args.tensor_parallel_size,
              model_loader_extra_config=tensorizer_config
    )
    return llm


if __name__ == '__main__':
    args = parse_args()

    s3_access_key_id = (getattr(args, 's3_access_key_id', None)
                        or os.environ.get("S3_ACCESS_KEY_ID", None))
    s3_secret_access_key = (getattr(args, 's3_secret_access_key', None)
                            or os.environ.get("S3_SECRET_ACCESS_KEY", None))
    s3_endpoint = (getattr(args, 's3_endpoint', None)
                or os.environ.get("S3_ENDPOINT_URL", None))

    credentials = {
        "s3_access_key_id": s3_access_key_id,
        "s3_secret_access_key": s3_secret_access_key,
        "s3_endpoint": s3_endpoint
    }

    model_ref = args.model

    model_name = model_ref.split("/")[1]

    keyfile = args.keyfile if args.keyfile else None

    if args.model_loader_extra_config:
        config = json.loads(args.model_loader_extra_config)
        tensorizer_args = \
            TensorizerConfig(**config)._construct_tensorizer_args()
        tensorizer_args.tensorizer_uri = args.path_to_tensors
    else:
        tensorizer_args = None

    if args.command == "serialize":
        eng_args_dict = {f.name: getattr(args, f.name) for f in
                        dataclasses.fields(EngineArgs)}

        engine_args = EngineArgs.from_cli_args(
            argparse.Namespace(**eng_args_dict)
        )

        input_dir = args.serialized_directory.rstrip('/')
        suffix = args.suffix if args.suffix else uuid.uuid4().hex
        base_path = f"{input_dir}/vllm/{model_ref}/{suffix}"
        if engine_args.tensor_parallel_size > 1:
            model_path = f"{base_path}/model-rank-%03d.tensors"
        else:
            model_path = f"{base_path}/model.tensors"

        tensorizer_config = TensorizerConfig(
            tensorizer_uri=model_path,
            encryption_keyfile=keyfile,
            **credentials)

        tensorize_vllm_model(engine_args, tensorizer_config)

    elif args.command == "deserialize":
        if not tensorizer_args:
            tensorizer_config = TensorizerConfig(
                tensorizer_uri=args.path_to_tensors,
                encryption_keyfile = keyfile,
                **credentials
            )
        deserialize()
    else:
        raise ValueError("Either serialize or deserialize must be specified.")

```
