---
title: Disaggregated Prefill
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/disaggregated_prefill.sh](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/disaggregated_prefill.sh)

```bash
#!/bin/bash
# 本文件演示解耦预填充 (disaggregated prefilling) 的示例用法
# 我们将启动 2 个 vllm 实例(1 个用于预填充，1 个用于解码)
# 然后在它们之间传输 KV 缓存

set -xe

echo "🚧🚧 Warning: The usage of disaggregated prefill is experimental and subject to change 🚧🚧"
sleep 1

# 可选项：meta-llama/Meta-Llama-3.1-8B-Instruct 或 deepseek-ai/DeepSeek-V2-Lite
MODEL_NAME=${HF_MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}

# 捕获 Ctrl+C 中断信号
trap 'cleanup' INT

# 清理函数
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    # 清理命令
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# 首先安装 quart——解耦预填充代理服务所需
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi


# 等待 vLLM 服务器启动的函数
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


# 您也可以调整 --kv-ip 和 --kv-port 参数用于分布式推理

# 预填充实例，作为 KV 生产者
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_NAME \
    --port 8100 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' &

# 解码实例，作为 KV 消费者
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_NAME \
    --port 8200 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' &

# 等待预填充和解码实例就绪
wait_for_server 8100
wait_for_server 8200


# 启动代理服务器，开放 8000 端口服务
# 该代理的工作流程：
# 1. 将请求发送到预填充 vLLM 实例(端口 8100)，将 max_tokens 设为 1
# 2. 预填充完成后，将请求转发到解码 vLLM 实例
# 注意：此 API 用法可能会变更——未来我们将引入"vllm connect"来连接预填充和解码实例
python3 ../../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py &
sleep 1

# 发送两个示例请求
output1=$(curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0
}')

output2=$(curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"prompt": "Santa Clara is a",
"max_tokens": 10,
"temperature": 0
}')


# 清理命令
pgrep python | xargs kill -9
pkill -f python

echo ""

sleep 1

# 打印curl请求的输出
echo ""
echo "Output of first request: $output1"
echo "Output of second request: $output2"

echo "🎉🎉 Successfully finished 2 test requests! 🎉🎉"
echo ""

```
