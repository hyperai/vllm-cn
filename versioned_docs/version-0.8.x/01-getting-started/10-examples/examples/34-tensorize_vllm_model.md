---
title: Tensorize vLLM 模型
---

源代码: [vllm-project/vllm](https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tensorize_vllm_model.py)

```python
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

# yapf conflicts with isort for this docstring
# yapf 与该 docstring 的 isort 冲突

# yapf: disable
"""
tensorize_vllm_model.py is a script that can be used to serialize and 
deserialize vLLM models. These models can be loaded using tensorizer 
to the GPU extremely quickly over an HTTP/HTTPS endpoint, an S3 endpoint,
or locally. Tensor encryption and decryption is also supported, although 
libsodium must be installed to use it. Install vllm with tensorizer support 
using `pip install vllm[tensorizer]`. To learn more about tensorizer, visit
https://github.com/coreweave/tensorizer
tensorize_vllm_model.py 是一个用于序列化和反序列化 vLLM 模型的脚本。
这些模型可以通过 tensorizer 以极快的速度加载到 GPU 上，无论是通过 HTTP/HTTPS 端
点、S3 端点，还是本地加载。脚本还支持张量的加密和解密，但需要安装 libsodium 才能使
用此功能。可以通过 `pip install vllm[tensorizer]` 安装带 tensorizer 支持
的 vLLM。想了解更多关于 tensorizer 的信息，
请访问 https://github.com/coreweave/tensorizer。




To serialize a model, install vLLM from source, then run something 
like this from the root level of this repository:
要序列化模型，先从源码安装 vLLM，然后在本项目的根目录运行类似以下命令：


python -m examples.tensorize_vllm_model \
   --model facebook/opt-125m \
   serialize \
   --serialized-directory s3://my-bucket \
   --suffix v1
   
Which downloads the model from HuggingFace, loads it into vLLM, serializes it,
and saves it to your S3 bucket. A local directory can also be used. This
assumes your S3 credentials are specified as environment variables
in the form of `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, and 
`S3_ENDPOINT_URL`. To provide S3 credentials directly, you can provide 
`--s3-access-key-id` and `--s3-secret-access-key`, as well as `--s3-endpoint` 
as CLI args to this script.
这会从 HuggingFace 下载模型，将其加载到 vLLM 中，进行序列化，并将其保存到你
的 S3 存储桶中。也可以使用本地目录保存。此操作假设你的 S3 凭证已经作为环境变量指定，
格式为 `S3_ACCESS_KEY_ID`、`S3_SECRET_ACCESS_KEY` 和 `S3_ENDPOINT_URL`。
你也可以通过命令行参数提供 S3 凭证，
如 `--s3-access-key-id`、`--s3-secret-access-key` 和 `--s3-endpoint`。




You can also encrypt the model weights with a randomly-generated key by 
providing a `--keyfile` argument.
你还可以通过提供 `--keyfile` 参数使用随机生成的密钥来加密模型权重。




To deserialize a model, you can run something like this from the root 
level of this repository:
要反序列化模型，你可以在本项目根目录运行类似以下命令：




python -m examples.tensorize_vllm_model \
   --model EleutherAI/gpt-j-6B \
   --dtype float16 \
   deserialize \
   --path-to-tensors s3://my-bucket/vllm/EleutherAI/gpt-j-6B/v1/model.tensors

Which downloads the model tensors from your S3 bucket and deserializes them.
这会从你的 S3 存储桶中下载模型张量并进行反序列化。




You can also provide a `--keyfile` argument to decrypt the model weights if 
they were serialized with encryption.
如果模型权重在序列化时进行了加密，你还可以通过提供 `--keyfile` 参数来解密权重。




To support distributed tensor-parallel models, each model shard will be
serialized to a separate file. The tensorizer_uri is then specified as a string
template with a format specifier such as '%03d' that will be rendered with the
shard's rank. Sharded models serialized with this script will be named as
model-rank-%03d.tensors
为支持分布式张量并行模型，每个模型分片将序列化到单独的文件中。然后 `tensorizer_uri`
将作为一个字符串模板，使用类似 `%03d` 的格式说明符来表示分片的 rank。使用此脚本序列
化的分片模型将命名为 `model-rank-%03d.tensors`。




For more information on the available arguments for serializing, run 
`python -m examples.tensorize_vllm_model serialize --help`.
要查看序列化时可用的所有参数，请运行：`python -m examples.tensorize_vllm_model serialize --help`


Or for deserializing:
反序列化时请运行：


`python -m examples.tensorize_vllm_model deserialize --help`.




Once a model is serialized, tensorizer can be invoked with the `LLM` class 
directly to load models:
一旦模型被序列化，可以直接使用 `LLM` 类通过 tensorizer 加载模型：


    llm = LLM(model="facebook/opt-125m",
              load_format="tensorizer",
              model_loader_extra_config=TensorizerConfig(
                    tensorizer_uri = path_to_tensors,
                    num_readers=3,
                    )
              )
            
A serialized model can be used during model loading for the vLLM OpenAI
inference server. `model_loader_extra_config` is exposed as the CLI arg
`--model-loader-extra-config`, and accepts a JSON string literal of the
TensorizerConfig arguments desired.
序列化后的模型可以用于 vLLM OpenAI 推理服务器的模型加载。
`model_loader_extra_config` 以 CLI 参数 `--model-loader-extra-config` 展示，
并接受 TensorizerConfig 参数的 JSON 字符串。




In order to see all of the available arguments usable to configure 
loading with tensorizer that are given to `TensorizerConfig`, run:
要查看可用于配置 tensorizer 加载的所有参数，这些参数会传递给 `TensorizerConfig`
，请运行：




`python -m examples.tensorize_vllm_model deserialize --help`

under the `tensorizer options` section. These can also be used for
deserialization in this example script, although `--tensorizer-uri` and
`--path-to-tensors` are functionally the same in this case.
在 `tensorizer options` 部分下的选项可以用于此示例脚本的反序列化操作，
尽管 `--tensorizer-uri` 和 `--path-to-tensors` 在此情况下功能相同。

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


