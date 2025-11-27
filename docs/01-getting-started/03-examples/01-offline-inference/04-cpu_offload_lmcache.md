---
title: Cpu Offload Lmcache
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/cpu_offload_lmcache.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/cpu_offload_lmcache.py)

```python
# SPDX-License-Identifier: Apache-2.0

"""
该文件演示了 CPU 卸载的示例用法
与 LMCache。
请注意，运行此示例需要 "pip install lmcache"。
在 https://github.com/LMCache/LMCache 中了解有关 LMCache 的更多信息。
"""
import os
import time

from lmcache.experimental.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# 与 LMCache 相关的环境变量
# 在 LMCache 中使用实验功能
os.environ["LMCache_USE_EXPERIMENTAL"] = "True"
# LMCache 设置为每块使用256个 token
os.environ["LMCache_CHUNK_SIZE"] = "256"
# 在 LMCache 中启用本地 CPU 后端
os.environ["LMCache_LOCAL_CPU"] = "True"
# 将本地 CPU 内存限制设置为 5.0 GB
os.environ["LMCache_MAX_LOCAL_CPU_SIZE"] = "5.0"

# 此示例脚本以共享前缀运行两个请求。
shared_prompt = "Hello, how are you?" * 1000
first_prompt = [
    shared_prompt + "Hello, my name is",
]
second_prompt = [
    shared_prompt + "Tell me a very long story",
]

sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

ktc = KVTransferConfig.from_cli(
    '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}')
# 将 GPU 内存利用设置为 0.8，用于 40GB 显存的 A40 GPU。
# 如果您的 GPU 的内存较少，则降低值。
# 请注意，LMCache 目前与块预填充不兼容。
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",
          kv_transfer_config=ktc,
          max_model_len=8000,
          enable_chunked_prefill=False,
          gpu_memory_utilization=0.8)

outputs = llm.generate(first_prompt, sampling_params)
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")
print("First request done.")

time.sleep(1)

outputs = llm.generate(second_prompt, sampling_params)
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")
print("Second request done.")

# 清理 LMCache 后端
LMCacheEngineBuilder.destroy(ENGINE_NAME)

```
