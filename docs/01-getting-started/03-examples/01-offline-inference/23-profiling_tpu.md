---
title: vLLM TPU 分析
---

[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/profiling_tpu](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/profiling_tpu)

此脚本用于分析 vLLM 在特定预填充(prefill)或解码(decode)令牌形状下的 TPU 性能表现。

注意：实际运行的服务器会混合处理多种形状的预填充和解码请求。

假设您已在使用 TPU 环境(本测试基于 TPU v6e)并已按照[安装指南](https://docs.vllm.ai/en/latest/getting_started/installation/ai_accelerator/index.html)完成 vLLM 安装。

>以下所有示例中，我们都先进行若干次预热运行(因此使用--enforce-eager参数是可行的)

## 性能分析示例

### 生成预填充分析数据

此示例运行 Qwen/Qwen2.5-7B-Instruct 模型，处理包含1024个输入令牌的单个请求。该设置旨在专门分析预填充阶段的时间和操作。

```bash
export XLA_HLO_DEBUG=1
export MODEL=Qwen/Qwen2.5-7B-Instruct
export VLLM_TPU_PROFILE_DURATION_MS=3000
export VLLM_TPU_PROFILE_DELAY_MS=0


python3 profiling.py \
    --model $MODEL \
    --input-len 1024 --output-len 1 \
    --batch-size 1 --enforce-eager \
    --max-model-len 2048 \
    --tensor-parallel-size 1 \
    --profile-result-dir profiles
```

### 生成解码分析数据

此示例运行 Llama 3.1 70B 模型，处理32个并行请求的批次，每个请求包含1个输入令牌和128个输出令牌。通过设置极小的1个令牌预填充，并配置`VLLM_TPU_PROFILE_DELAY_MS=1000`跳过前1秒的推理(预计是预填充阶段)，专门分析32个并行解码过程。

```bash
export XLA_HLO_DEBUG=1
export MODEL=meta-llama/Llama-3.1-70B-Instruct
export VLLM_TPU_PROFILE_DURATION_MS=2000
export VLLM_TPU_PROFILE_DELAY_MS=1000


rm -rf ~/.cache/vllm/xla_cache
python3 profiling.py \
    --model $MODEL \
    --input-len 1 \
    --output-len 128 \
    --batch-size 32 \
    --enforce-eager \
    --profile-result-dir profiles \
    --max-model-len 2048 --tensor-parallel-size 8
```

## 可视化分析结果

收集到性能分析数据后，您可以使用[TensorBoard](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)进行可视化分析。

需要安装的依赖项通常包括：

```bash
pip install tensorflow-cpu tensorboard-plugin-profile etils importlib_resources
```
Then you just need to point TensorBoard to the directory where you saved the profiles and visit `http://localhost:6006/` in your browser:
然后只需将TensorBoard指向保存分析数据的目录，并在浏览器中访问`http://localhost:6006/`：

```bash
tensorboard --logdir profiles/ --port 6006
```
# 示例材料

## profiling.py

```plain
# SPDX-License-Identifier: Apache-2.0


import argparse
import dataclasses
import os
import time


import numpy as np
import torch_xla.debug.profiler as xp
from tqdm import tqdm


from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.utils import FlexibleArgumentParser


DURATION_MS = int(os.getenv("VLLM_TPU_PROFILE_DURATION_MS", 3000))
DELAY_MS = int(os.getenv("VLLM_TPU_PROFILE_DELAY_MS", 0))




def main(args: argparse.Namespace):
    print(args)


    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))
    server = xp.start_server(9012)  # noqa: F841


    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: list[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]


    def run_to_completion():
        start_time = time.perf_counter()
        llm.generate(dummy_prompts,
                     sampling_params=sampling_params,
                     use_tqdm=False)
        end_time = time.perf_counter()
        latency = end_time - start_time
        return latency


    # Warmup
    # 预热
    print("Warming up...")
    warmup_latencies = []
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        warmup_latencies.append(run_to_completion())
    print(f"Average warmup latency: {np.mean(warmup_latencies):.4f}s")


    # Profile
    # 分析
    profile_dir = args.profile_result_dir
    print(f"Profiling (results will be saved to '{profile_dir}')...")
    # Enable tracing on server
    # 在服务器上启用跟踪
    xp.trace_detached("localhost:9012",
                      profile_dir,
                      delay_ms=DELAY_MS,
                      duration_ms=DURATION_MS)
    if DELAY_MS == 0:
        time.sleep(1.0)
    profile_latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profile iterations"):
        profile_latencies.append(run_to_completion())
    print(f"Average profile latency: {np.mean(profile_latencies):.4f}s")


    return




if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=5,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=1,
                        help='Number of iterations to run for profiling.')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default="profiles",
        help=
        ('path to save the pytorch profiler output. Can be visualized '
         'with ui.perfetto.dev or Tensorboard '
         '(https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm).'
         ))


    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)




```


