---
title: Run Cluster
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/run_cluster.sh](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/run_cluster.sh)

```bash
#!/bin/bash

# Check for minimum number of required arguments
# 检查是否提供了最少数量的必需的参数
if [ $# -lt 4 ]; then
    echo "Usage: $0 docker_image head_node_address --head|--worker path_to_hf_home [additional_args...]"
    exit 1
fi

# Assign the first three arguments and shift them away
# 读取前四个参数并移除它们
DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"  # Should be --head or --worker # 应该是 --head 或 --worker
PATH_TO_HF_HOME="$4"
shift 4

# Additional arguments are passed directly to the Docker command
# 其余参数作为额外参数传递给 Docker 命令
ADDITIONAL_ARGS=("$@")

# Validate node type
# 验证节点类型
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

# Define a function to cleanup on EXIT signal
# 定义一个函数，在脚本退出时清理 Docker 容器
cleanup() {
    docker stop node
    docker rm node
}
trap cleanup EXIT

# Command setup for head or worker node
# 根据节点类型设置 Ray 启动命令
RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
    RAY_START_CMD+=" --head --port=6379"
else
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
fi

# Run the docker command with the user specified parameters and additional arguments
# 运行 Docker 容器，传入用户指定的参数和额外参数
docker run \
    --entrypoint /bin/bash \
    --network host \
    --name node \
    --shm-size 10.24g \
    --gpus all \
    -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
    "${ADDITIONAL_ARGS[@]}" \
    "${DOCKER_IMAGE}" -c "${RAY_START_CMD}"

```
