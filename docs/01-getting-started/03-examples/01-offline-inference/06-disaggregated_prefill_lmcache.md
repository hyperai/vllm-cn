---
title: Disaggregated Prefill Lmcache
---

[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/disaggregated_prefill_lmcache.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/disaggregated_prefill_lmcache.py)

```python
# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of disaggregated prefilling
with LMCache.
We will launch 2 vllm instances (GPU 0 for prefill and GPU 1 for decode),
and launch an additional LMCache server.
KV cache is transferred in the following manner: 
vLLM prefill node -> LMCache server -> vLLM decode node.
Note that `pip install lmcache` is needed to run this example.
Learn more about LMCache in https://github.com/LMCache/LMCache.
"""
"""
该文件演示了分解预填充的示例用法
与 LMCache。
我们将启动2个 vLLM 实例 (Preill 的 GPU 0和 Decode 的 GPU 1) ，
并启动额外的 LMCache 服务器。
KV 缓存以以下方式传输: 
vLLM 预填充节点 - > lmcache Server-> vllm 解码节点。
请注意，运行此示例需要运行 `pip install lmcache`。
在 https://github.com/LMCache/LMCache 中了解有关 LMCache 的更多信息。
"""
import os
import subprocess
import time
from multiprocessing import Event, Process

from lmcache.experimental.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# LMCache-related environment variables
# The port to start LMCache server
# 与 LMCache 相关的环境变量
# 要启动 LMCache 服务器的端口
port = 8100
# Use experimental features in LMCache
# 在 LMCache 中使用实验功能
os.environ["LMCache_USE_EXPERIMENTAL"] = "True"
# LMCache is set to use 256 tokens per chunk
# lmcache 设置为每块使用256个 token 
os.environ["LMCache_CHUNK_SIZE"] = "256"
# Disable local CPU backend in LMCache
# 禁用 LMCache 中的本地 CPU 后端
os.environ["LMCache_LOCAL_CPU"] = "False"
# Set local CPU memory buffer limit to 5.0 GB
# 将本地 CPU 内存缓冲区限制设置为5.0 GB
os.environ["LMCache_MAX_LOCAL_CPU_SIZE"] = "5.0"
# Set the remote URL for LMCache server
# 设置 LMCache 服务器的远程 URL
os.environ["LMCache_REMOTE_URL"] = f"lm://localhost:{port}"
# Set the serializer/deserializer between vllm and LMCache server
# `naive` indicates using raw bytes of the tensor without any compression
# 在 vLLM 和 LMCache 服务器之间设置序列化器/求职者
# `naive` 表示使用张量的原始字符而无需任何压缩
os.environ["LMCache_REMOTE_SERDE"] = "naive"


def run_prefill(prefill_done, prompts):
    # We use GPU 0 for prefill node.
    # 我们将 GPU 0 用于预填充节点。
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"LMCacheConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
    )
    # Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
    # memory. Reduce the value if your GPU has less memory.
    # 将 GPU 内存利用设置为0.8，用于40GB 显存的 A40 GPU。
    # 如果您的 GPU 的内存较少，则降低值。
    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",
              kv_transfer_config=ktc,
              max_model_len=8000,
              gpu_memory_utilization=0.8,
              enforce_eager=True)

    #llm.generate(prompts, sampling_params)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("Prefill node is finished.")
    prefill_done.set()

    # Clean up lmcache backend
    # 清理 LMCache 后端
    LMCacheEngineBuilder.destroy(ENGINE_NAME)


def run_decode(prefill_done, prompts, timeout=1):
    # We use GPU 1 for decode node.
    # 我们将 GPU 1 用于解码节点。
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"LMCacheConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
    )
    # Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
    # of memory. Reduce the value if your GPU has less memory.
    # 将 GPU 内存利用设置为0.8，用于40GB 显存的 A40 GPU。
    # 如果您的 GPU 的内存较少，则降低值。
    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",
              kv_transfer_config=ktc,
              max_model_len=8000,
              gpu_memory_utilization=0.8,
              enforce_eager=True)

    print("Waiting for prefill node to finish...")
    prefill_done.wait()
    time.sleep(timeout)

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")

    # Clean up lmcache backend
    # 清理 LMCache 后端
    LMCacheEngineBuilder.destroy(ENGINE_NAME)


def run_lmcache_server(port):
    server_proc = subprocess.Popen([
        "python", "-m", "lmcache.experimental.server", "localhost",
        str(port)
    ])
    return server_proc


if __name__ == "__main__":

    prompts = [
        "Hello, how are you?" * 1000,
    ]

    prefill_done = Event()
    prefill_process = Process(target=run_prefill, args=(prefill_done, prompts))
    decode_process = Process(target=run_decode, args=(prefill_done, prompts))
    lmcache_server_process = run_lmcache_server(port)

    # Start prefill node
    # 开始预填充节点
    prefill_process.start()

    # Start decode node
    # 开始解码节点
    decode_process.start()

    # Clean up the processes
    # 清理过程
    decode_process.join()
    prefill_process.terminate()
    lmcache_server_process.terminate()
    lmcache_server_process.wait()

```
