---
title: Disaggregated Prefill
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/disaggregated_prefill.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/disaggregated_prefill.py)

```python
# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of disaggregated prefilling
We will launch 2 vllm instances (GPU 0 for prefill and GPU 1 for decode),
and then transfer the KV cache between them.
"""
"""
该文件演示了分解预填充的示例用法
我们将启动2个 vLLM 实例 (Preill 的 GPU 0和 Decode 的 GPU 1) ，
然后将 KV 缓存在它们之间传递。
"""
import os
import time
from multiprocessing import Event, Process

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def run_prefill(prefill_done):
    # We use GPU 0 for prefill node.
    # 我们将 GPU 0用于预填充节点。
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # The prefill node receives two requests, while the decode node receives
    # three requests. So the decode node will only receive the KV Cache for
    # requests 1 and 3. The decode node will use the KV Cache of requests 1
    # and 3 and do prefilling on request 2.
    # 预填充节点接收两个请求，而解码节点接收到
    # 三个请求。因此，解码节点只会接收 KV 缓存
    # 请求1和3。解码节点将使用请求的 KV 缓存 1
    # 和 3，并根据要求进行预填充 2。
    prompts = [
        "Hello, my name is",
        "Hi, your name is",
        # The decode node will actually "prefill" this request.
        # 解码节点实际上将"预填充"此请求。
        "Tell me a very long story",
    ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Using PyNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the prefill node (kv_producer, rank 0).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for PyNcclConnector.
    # 使用 pyncclconnector 在 vLLM 实例之间传输 KV 缓存。
    # 此实例是预填充节点 (kv_producer， rank 0)。
    # KV 缓存传输的并行实例数设置为 2，
    # 根据 PyncclConnector 的要求。
    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    # 将 GPU 内存利用设置为0.8，用于40GB 显存的 A6000 GPU。
    # 您可能需要调整值以适合您的 GPU。
    llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
              kv_transfer_config=ktc,
              max_model_len=2000,
              gpu_memory_utilization=0.8)

    llm.generate(prompts, sampling_params)
    print("Prefill node is finished.")
    prefill_done.set()

    # To keep the prefill node running in case the decode node is not done;
    # otherwise, the script might exit prematurely, causing incomplete decoding.
    # 如果未完成解码节点，则保持预填充节点运行；
    # 否则，脚本可能会过早退出，从而导致不完整的解码。
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script stopped by user.")


def run_decode(prefill_done):
    # We use GPU 1 for decode node.
    # 我们将 GPU 1 用于解码节点。
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    prompts = [
        "Hello, my name is",
        "Hi, your name is",
        "Tell me a very long story",
    ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95)

    # Using PyNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the decode node (kv_consumer, rank 1).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for PyNcclConnector.
    # 使用 PyNcclConnector 在 vLLM 实例之间传输 KV 缓存。
    # 此实例是解码节点 (KV_Consumer， rank 1)。
    # KV 缓存传输的并行实例数设置为2，
    # 根据 PyNcclConnector 的要求。
    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    # 将 GPU 内存利用设置为0.8，用于40GB 显存的 A6000 GPU
    # 您可能需要调整值以适合您的 GPU。
    llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
              kv_transfer_config=ktc,
              max_model_len=2000,
              gpu_memory_utilization=0.8)

    # Wait for the producer to start the pipe
    # 等待生产者启动管道
    print("Waiting for prefill node to finish...")
    prefill_done.wait()

    # At this point when the prefill_done is set, the kv-cache should have been
    # transferred to this decode node, so we can start decoding.
    # 在设置 prefill_done 时，KV-CACHE 应该
    # 转到此解码节点，因此我们可以开始解码。
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    prefill_done = Event()
    prefill_process = Process(target=run_prefill, args=(prefill_done, ))
    decode_process = Process(target=run_decode, args=(prefill_done, ))

    # Start prefill node
    # 开始预填充节点
    prefill_process.start()

    # Start decode node
    # 开始解码节点
    decode_process.start()

    # Terminate the prefill node when decode is finished
    # 解释完成后终止预填充节点
    decode_process.join()
    prefill_process.terminate()

```
