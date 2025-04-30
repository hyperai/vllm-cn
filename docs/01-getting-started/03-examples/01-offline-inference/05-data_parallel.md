---
title: Data Parallel
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/data_parallel.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/data_parallel.py)

```python
# SPDX-License-Identifier: Apache-2.0
"""
用法:
单节点:
python examples/offline_inference/data_parallel.py \
--model="ibm-research/PowerMoE-3b" \
--dp-size=2 \
--tp-size=2
多节点:
Node 0 (assume the node has ip of 10.99.48.128):
python examples/offline_inference/data_parallel.py \
--model="ibm-research/PowerMoE-3b" \
--dp-size=2 \
--tp-size=2 \
--node-size=2 \
--node-rank=0 \
--master-addr=10.99.48.128 \
--master-port=13345
Node 1:
python examples/offline_inference/data_parallel.py \
--model="ibm-research/PowerMoE-3b" \
--dp-size=2 \
--tp-size=2 \
--node-size=2 \
--node-rank=1 \
--master-addr=10.99.48.128 \
--master-port=13345
"""
import os
from time import sleep

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port


def main(model, dp_size, local_dp_rank, global_dp_rank, dp_master_ip,
         dp_master_port, GPUs_per_dp_rank):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)


    # cuda_visible_devices 的每个 DP rank 都会自动设置
    # 引擎过程。

    # 样本提示。
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 100


    # 使用 DP，每个 rank 都应处理不同的提示。
    # 通常所有 DP  rank 完整的数据集，
    # 并且每个 rank 都处理数据集的不同部分。
    promts_per_rank = len(prompts) // dp_size
    start = global_dp_rank * promts_per_rank
    end = start + promts_per_rank
    prompts = prompts[start:end]
    if len(prompts) == 0:

        # 如果任何 rank 都没有提示进行处理，
        # 我们需要设置占位符提示
        prompts = ["Placeholder"]
    print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")


    # 创建一个采样参数对象。
    # 由于我们正在平行数据，因此每个 rank 都可以具有不同的
    # 采样参数。在这里，我们为不同的 max_tokens 设置了不同的 max_tokens
    # 示范 rank 。
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=[16, 20][global_dp_rank % 2])

    # 创建一个 LLM。
    llm = LLM(model=model,
              tensor_parallel_size=GPUs_per_dp_rank,
              enforce_eager=True,
              enable_expert_parallel=True)
    outputs = llm.generate(prompts, sampling_params)
    # 打印输出。
    for i, output in enumerate(outputs):
        if i >= 5:
            # print only 5 outputs
            # 仅打印5个输出
            break
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
              f"Generated text: {generated_text!r}")

    # 让引擎有时间在退出之前暂停处理循环。
    sleep(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data Parallel Inference")
    parser.add_argument("--model",
                        type=str,
                        default="ibm-research/PowerMoE-3b",
                        help="Model name or path")
    parser.add_argument("--dp-size",
                        type=int,
                        default=2,
                        help="Data parallel size")
    parser.add_argument("--tp-size",
                        type=int,
                        default=2,
                        help="Tensor parallel size")
    parser.add_argument("--node-size",
                        type=int,
                        default=1,
                        help="Total number of nodes")
    parser.add_argument("--node-rank",
                        type=int,
                        default=0,
                        help="Rank of the current node")
    parser.add_argument("--master-addr",
                        type=str,
                        default="",
                        help="Master node IP address")
    parser.add_argument("--master-port",
                        type=int,
                        default=0,
                        help="Master node port")
    args = parser.parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    from multiprocessing import Process

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
            range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)):
        proc = Process(target=main,
                       args=(args.model, dp_size, local_dp_rank,
                             global_dp_rank, dp_master_ip, dp_master_port,
                             tp_size))
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join(timeout=300)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that "
                  f"didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)

```
