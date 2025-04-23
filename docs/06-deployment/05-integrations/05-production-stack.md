---

title: 生产环境技术栈

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


在 Kubernetes 上部署 vLLM 是一种可扩展且高效的服务机器学习模型的方式。本指南将引导您使用 [vLLM 生产环境技术栈](https://github.com/vllm-project/production-stack) 部署 vLLM。该技术栈源于伯克利-芝加哥大学的合作，是 [vLLM 项目](https://github.com/vllm-project) 下正式发布的生产优化代码库，专为 LLM 部署设计，具有以下特点：

* **与上游 vLLM 兼容** – 它封装了上游 vLLM，无需修改其代码。
* **易于使用** – 通过 Helm 图表简化部署，并通过 Grafana 仪表板实现可观测性。
* **高性能** – 针对 LLM 工作负载进行了优化，支持多模型、模型感知和前缀感知路由、快速 vLLM 引导以及使用 [LMCache](https://github.com/LMCache/LMCache) 的 KV 缓存卸载等功能。


如果您是 Kubernetes 的新手，不用担心：在 vLLM 生产环境技术栈的 [仓库](https://github.com/vllm-project/production-stack) 中，我们提供了逐步的[指南](https://github.com/vllm-project/production-stack/blob/main/tutorials/00-install-kubernetes-env.md)和一个[短视频](https://www.youtube.com/watch?v=EsTJbQtzj0g)，**帮助您在 4 分钟内完成所有设置并开始使用！**


## 依赖

确保您有一个运行中的 Kubernetes 环境并配备了 GPU（您可以按照[本教程](https://github.com/vllm-project/production-stack/blob/main/tutorials/00-install-kubernetes-env.md)在裸机 GPU 机器上安装 Kubernetes 环境）。


## 使用 vLLM 生产环境技术栈进行部署

使用 Helm 图表安装标准的 vLLM 生产环境技术栈。您可以运行该 [bash 脚本](https://github.com/vllm-project/production-stack/blob/main/tutorials/install-helm.sh) 在您的 GPU 服务器上安装 Helm。


要安装 vLLM 生产环境技术栈，请在您的主机上运行以下命令：

```plain
sudo helm repo add vllm https://vllm-project.github.io/production-stack
sudo helm install vllm vllm/vllm-stack -f tutorials/assets/values-01-minimal-example.yaml
```


这将实例化一个名为 `vllm` 的基于 vLLM 生产环境技术栈的部署，运行一个小型 LLM（Facebook opt-125M 模型）。


### 验证安装

使用以下命令监控部署状态：

```plain
sudo kubectl get pods
```


您将看到 `vllm` 部署的 Pod 将过渡到 `Running` 状态。

```plain
NAME                                           READY   STATUS    RESTARTS   AGE
vllm-deployment-router-859d8fb668-2x2b7        1/1     Running   0          2m38s
vllm-opt125m-deployment-vllm-84dfc9bd7-vb9bs   1/1     Running   0          2m38s
```


**注意**：容器下载 Docker 镜像和 LLM 权重可能需要一些时间。

### 

### 向技术栈发送查询

将 `vllm-router-service` 端口转发到主机：

```plain
sudo kubectl port-forward svc/vllm-router-service 30080:80
```


然后，您可以向 OpenAI 兼容的 API 发送查询以检查可用模型：

```plain
curl -o- http://localhost:30080/models
```


预期输出：

```plain
{
  "object": "list",
  "data": [
    {
      "id": "facebook/opt-125m",
      "object": "model",
      "created": 1737428424,
      "owned_by": "vllm",
      "root": null
    }
  ]
}
```


要发送实际的聊天请求，您可以向 OpenAI 的 `/completion` 端点发出 curl 请求：

```plain
curl -X POST http://localhost:30080/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Once upon a time,",
    "max_tokens": 10
  }'
```


预期输出：

```plain
{
  "id": "completion-id",
  "object": "text_completion",
  "created": 1737428424,
  "model": "facebook/opt-125m",
  "choices": [
    {
      "text": " there was a brave knight who...",
      "index": 0,
      "finish_reason": "length"
    }
  ]
}
```


### 卸载

如需删除部署，请运行：

```plain
sudo helm uninstall vllm
```



---
### （高级）配置 vLLM 生产环境技术栈

vLLM 生产环境技术栈的核心配置通过 YAML 管理。以下是上述安装中使用的示例配置：

```plain
servingEngineSpec:
  runtimeClassName: ""
  modelSpec:
  - name: "opt125m"
    repository: "vllm/vllm-openai"
    tag: "latest"
    modelURL: "facebook/opt-125m"


    replicaCount: 1


    requestCPU: 6
    requestMemory: "16Gi"
    requestGPU: 1


    pvcStorage: "10Gi"
```


在此 YAML 配置中：

* `modelSpec` 包括：

   * `name`：您希望为模型命名的昵称。

   * `repository`：vLLM 的 Docker 仓库。

   * `tag`：Docker 镜像标签。

   * `modelURL`：您希望使用的 LLM 模型。
* `replicaCount`：副本数量。
* `requestCPU` 和 `requestMemory`：指定 Pod 的 CPU 和内存资源请求。
* `requestGPU`：指定所需的 GPU 数量。
* `pvcStorage`：为模型分配持久存储。


**注意**：如果您打算设置两个 Pod，请参考此 [YAML 文件](https://github.com/vllm-project/production-stack/blob/main/tutorials/assets/values-01-2pods-minimal-example.yaml)。


**注意**：vLLM 生产环境技术栈提供了更多功能（例如 CPU 卸载和多种路由算法）。请查看这些[示例和教程](https://github.com/vllm-project/production-stack/tree/main/tutorials)以及我们的[仓库](https://github.com/vllm-project/production-stack)以获取更多详细信息！

