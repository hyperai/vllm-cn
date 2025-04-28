---
title: SkePilot
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

![图片](/img/docs/v1-deployment/07-skypilot_1.png)

**vLLM 可以通过 [SkyPilot](https://github.com/skypilot-org/skypilot) 在云和 Kubernetes 上运行并扩充为多个服务副本**，SkyPilot 是一个用于在任何云上运行 LLM 的开源框架。更多各种开放模型的示例，例如 Llama-3、Mixtral 等，可以在 [SkyPilot AI gallery](https://skypilot.readthedocs.io/en/latest/gallery/index.html) 中找到。

## 依赖

- 前往 [HuggingFace 模型页面](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) ， 请求获取模型 `meta-llama/Meta-Llama-3-8B -指导`。

- 检查您是否已安装 SkyPilot ([文档](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html))。

- 检查 `sky check`是否显示云或 Kubernetes 已启用。

```plain
pip install skypilot-nightly
sky check
```

## 在单个实例上运行

请参阅用于服务的 vLLM SkyPilot YAML，[serving.yaml](https://github.com/skypilot-org/skypilot/blob/master/llm/vllm/serve.yaml)。

```yaml
resources:
  accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # We can use cheaper accelerators for 8B model.


  accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # 我们可以为 8B 型号使用更便宜的加速器。


  use_spot: True
  disk_size: 512 # 确保模型检查点足够容纳。


  disk_tier: best
  ports: 8081 # 公开于互联网的端口。




envs:
  MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
  HF_TOKEN: <your-huggingface-token> # 更改为你自己的 huggingface token，或者使用 --env 来传递。




setup: |
  conda create -n vllm python=3.10 -y
  conda activate vllm


  pip install vllm==0.4.0.post1
  # 为 web UI 安装 Gradio。


  pip install gradio openai
  pip install flash-attn==2.5.7


run: |
  conda activate vllm
  echo 'Starting vllm api server...'
  python -u -m vllm.entrypoints.openai.api_server \
    --port 8081 \
    --model $MODEL_NAME \
    --trust-remote-code \
    --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
    2>&1 | tee api_server.log &


  echo 'Waiting for vllm api server to start...'
  while ! `cat api_server.log | grep -q 'Uvicorn running on'`; do sleep 1; done


  echo 'Starting gradio server...'
  git clone https://github.com/vllm-project/vllm.git || true
  python vllm/examples/gradio_openai_chatbot_webserver.py \
    -m $MODEL_NAME \
    --port 8811 \
    --model-url http://localhost:8081/v1 \
    --stop-token-ids 128009,128001
```

在列表中的任何一个候选 GPU（L4、A10g、...） 上启动 Llama-3 8B 模型的服务：

```plain
HF_TOKEN="your-huggingface-token" sky launch serving.yaml --env HF_TOKEN
```

检查命令的输出。会有一个可分享的 gradio 链接 （如下面的最后一行）。在浏览器中打开它，可以使用 LLaMA 模型进行文本补全。

```plain
(task, pid=7431) Running on public URL: https://<gradio-hash>.gradio.live
```

**可选**: 使用 70B 模型代替默认的 8B 模型，并且使用更多的 GPU：

```plain
HF_TOKEN="your-huggingface-token" sky launch serving.yaml --gpus A100:8 --env HF_TOKEN --env MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct
```

## 扩充到多个副本

SkyPilot 可以通过内置的自动扩充、负载平衡和容错功能将服务扩充到多个服务副本。您可以通过向 YAML 文件添加服务部分来完成此操作。

```yaml
service:
  replicas: 2
  # 准备就绪探针的实际请求。

  readiness_probe:
    path: /v1/chat/completions
    post_data:
    model: $MODEL_NAME
    messages:
      - role: user
        content: Hello! What is your name?
  max_tokens: 1
```

完整的 YAML 配置如下：

```yaml
service:
  replicas: 2
  # 准备就绪探针的实际请求。


  readiness_probe:
    path: /v1/chat/completions
    post_data:
      model: $MODEL_NAME
      messages:
        - role: user
          content: Hello! What is your name?
      max_tokens: 1


resources:
  accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # We can use cheaper accelerators for 8B model.


  accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # 我们可以为 8B 型号使用更便宜的加速器。


  use_spot: True
  disk_size: 512 # 确保模型检查点足够容纳。


  disk_tier: best
  ports: 8081 # 公开于互联网的端口。




envs:
  MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
  HF_TOKEN: <your-huggingface-token> # 更改为你自己的huggingface token，或者使用 --env 来传递。




setup: |
  conda create -n vllm python=3.10 -y
  conda activate vllm


  pip install vllm==0.4.0.post1
  # 安装 Gradio for Web UI。


  pip install gradio openai
  pip install flash-attn==2.5.7


run: |
  conda activate vllm
  echo 'Starting vllm api server...'
  python -u -m vllm.entrypoints.openai.api_server \
    --port 8081 \
    --model $MODEL_NAME \
    --trust-remote-code \
    --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
    2>&1 | tee api_server.log
```

开始在多个副本上提供 Llama-3 8B 模型服务：

```plain
HF_TOKEN="your-huggingface-token" sky serve up -n vllm serving.yaml --env HF_TOKEN
```

等待服务准备就绪：

```plain
watch -n10 sky serve status vllm
```

示例输出：

```plain
Services
NAME  VERSION  UPTIME  STATUS  REPLICAS  ENDPOINT
vllm  1        35s     READY   2/2       xx.yy.zz.100:30001


Service Replicas
SERVICE_NAME  ID  VERSION  IP            LAUNCHED     RESOURCES                STATUS  REGION
vllm          1   1        xx.yy.zz.121  18 mins ago  1x GCP([Spot]{'L4': 1})  READY   us-east4
vllm          2   1        xx.yy.zz.245  18 mins ago  1x GCP([Spot]{'L4': 1})  READY   us-east4
```

服务就绪后，您可以找到该服务的单个端点并使用该端点访问该服务：

```plain
ENDPOINT=$(sky serve status --endpoint 8081 vllm)
curl -L http://$ENDPOINT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Who are you?"
    }
    ],
    "stop_token_ids": [128009,  128001]
  }'
```

要启用自动缩放，您可以将 [replicas] 替换为 _service_ 中的以下配置：

```yaml
service:
  replica_policy:
    min_replicas: 2
    max_replicas: 4
    target_qps_per_replica: 2
```

当 QPS 超过 2 时，会将服务扩充到每个副本。

完整的 YAML 配置如下：

```yaml
service:
  replica_policy:
    min_replicas: 2
    max_replicas: 4
    target_qps_per_replica: 2
  # 准备就绪探针的实际请求。


  readiness_probe:
    path: /v1/chat/completions
    post_data:
      model: $MODEL_NAME
      messages:
        - role: user
          content: Hello! What is your name?
      max_tokens: 1


resources:
  accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # We can use cheaper accelerators for 8B model.


  accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # 我们可以为 8B 型号使用更便宜的加速器。


  use_spot: True
  disk_size: 512 # 确保模型检查点足够容纳。


  disk_tier: best
  ports: 8081 # 公开于互联网的端口。




envs:
  MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
  HF_TOKEN: <your-huggingface-token> # 更改为你自己的 huggingface token，或者使用 --env 来传递。




setup: |
  conda create -n vllm python=3.10 -y
  conda activate vllm


  pip install vllm==0.4.0.post1
  # 安装 Gradio Web UI 。


  pip install gradio openai
  pip install flash-attn==2.5.7


run: |
  conda activate vllm
  echo 'Starting vllm api server...'
  python -u -m vllm.entrypoints.openai.api_server \
    --port 8081 \
    --model $MODEL_NAME \
    --trust-remote-code \
    --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
    2>&1 | tee api_server.log
```

如需使用新配置更新服务，输入以下命令：

```plain
HF_TOKEN="your-huggingface-token" sky serve update vllm serving.yaml --env HF_TOKEN
```

停止服务命令：

```plain
sky serve down vllm
```

### **可选**：将 GUI 连接到端点

还可以使用单独的 GUI 前端访问 Llama-3 服务，因此发送到 GUI 的用户请求将在副本之间进行负载平衡。

查看完整的 GUI YAML 如下：

```yaml
envs:
  MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
  ENDPOINT: x.x.x.x:3031 # Address of the API server running vllm.


  ENDPOINT: x.x.x.x:3031 # 运行 vllm 的 API 服务器的地址。




resources:
  cpus: 2


setup: |
  conda create -n vllm python=3.10 -y
  conda activate vllm

  # 安装 Gradio Web UI。


  pip install gradio openai


run: |
  conda activate vllm
  export PATH=$PATH:/sbin


  echo 'Starting gradio server...'
  git clone https://github.com/vllm-project/vllm.git || true
  python vllm/examples/gradio_openai_chatbot_webserver.py \
    -m $MODEL_NAME \
    --port 8811 \
    --model-url http://$ENDPOINT/v1 \
    --stop-token-ids 128009,128001 | tee ~/gradio.log
```

1. 启动聊天 Web UI:

```plain
sky launch -c gui ./gui.yaml --env ENDPOINT=$(sky serve status --endpoint vllm)
```

2. 然后，我们可以通过返回的 gradio 链接访问 GUI:

```plain
| INFO | stdout | Running on public URL: https://6141e84201ce0bb4ed.gradio.live
```
