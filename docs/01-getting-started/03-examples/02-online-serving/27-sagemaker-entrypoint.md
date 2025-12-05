---
title: Sagemaker-entrypoint
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/sagemaker-entrypoint.sh](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/sagemaker-entrypoint.sh)

```bash
#!/bin/bash

# Define the prefix for environment variables to look for
# 定义要查找的环境变量前缀
PREFIX="SM_VLLM_"
ARG_PREFIX="--"

# Initialize an array for storing the arguments
# port 8080 required by sagemaker, https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-code-container-response
# 初始化一个数组用于存储参数
# 端口 8080 是 SageMaker 必需的端口，参考：https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-code-container-response
ARGS=(--port 8080)

# Loop through all environment variables
# 遍历所有环境变量
while IFS='=' read -r key value; do
    # Remove the prefix from the key, convert to lowercase, and replace underscores with dashes
    # 去掉前缀，将键名转换为小写，并用短横线替换下划线
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')

    # Add the argument name and value to the ARGS array
    # 将参数名和对应的值添加到 ARGS 数组
    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}")

# Pass the collected arguments to the main entrypoint
# 将收集到的参数传递给主入口点
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
```
