---

title: Helm

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


用于在 Kubernetes 上部署 vLLM 的 Helm Chart。


Helm 是 Kubernetes 的软件包管理器，它可以帮助您在 k8s 上部署 vLLM，并自动化 vLLM Kubernetes 应用程序的部署。借助 Helm，您可以通过覆盖变量值，在多个命名空间中使用不同的配置部署相同的框架架构。


本指南将引导您完成使用 Helm 部署 vLLM 的过程，包括必要的先决条件、Helm 安装步骤以及架构和 `values.yaml` 配置文件的相关文档。


## 依赖

在开始之前，请确保您具备以下条件：

* 运行中的 Kubernetes 集群

* NVIDIA Kubernetes 设备插件（`k8s-device-plugin`）：可在 [NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) 找到

* 集群中可用的 GPU 资源

* 包含要部署模型的 S3 存储

##

## 安装 Helm Chart

使用 `test-vllm` 作为发布名称安装 Chart，运行以下命令：

```go
helm upgrade --install --create-namespace --namespace=ns-vllm test-vllm . -f values.yaml --set secrets.s3endpoint=$ACCESS_POINT --set secrets.s3bucketname=$BUCKET --set secrets.s3accesskeyid=$ACCESS_KEY --set secrets.s3accesskey=$SECRET_KEY
```


## 卸载 Helm Chart

如果要卸载 `test-vllm` 部署，可以运行以下命令：

```go
helm uninstall test-vllm --namespace=ns-vllm
```
该命令将删除与 Chart 相关的所有 Kubernetes 组件（**包括持久卷**），并删除该发布。
##

## 架构

![图片](/img/docs/v1-deployment/04-helm_1.png)
##

## 值

|键 (Key)|类型|默认值|描述|
|:----|:----|:----|:----|
|autoscaling|object|{“enabled”:false,”maxReplicas”:100,”minReplicas”:1,”targetCPUUtilizationPercentage”:80}|Autoscaling configuration|
|autoscaling.enabled|bool|false|Enable autoscaling|
|autoscaling.maxReplicas|int|100|Maximum replicas|
|autoscaling.minReplicas|int|1|Minimum replicas|
|autoscaling.targetCPUUtilizationPercentage|int|80|Target CPU utilization for autoscaling|
|configs|object|{}|Configmap|
|containerPort|int|8000|Container port|
|customObjects|list|[]|Custom Objects configuration|
|deploymentStrategy|object|{}|Deployment strategy configuration|
|externalConfigs|list|[]|External configuration|
|extraContainers|list|[]|Additional containers configuration|
|extraInit|object|{“pvcStorage”:”1Gi”,”s3modelpath”:”relative_s3_model_path/opt-125m”, “awsEc2MetadataDisabled”: true}|Additional configuration for the init container|
|extraInit.pvcStorage|string|“50Gi”|Storage size of the s3|
|extraInit.s3modelpath|string|“relative_s3_model_path/opt-125m”|Path of the model on the s3 which hosts model weights and config files|
|extraInit.awsEc2MetadataDisabled|boolean|true|Disables the use of the Amazon EC2 instance metadata service|
|extraPorts|list|[]|Additional ports configuration|
|gpuModels|list|[“TYPE_GPU_USED”]|Type of gpu used|
|image|object|{“command”:[“vllm”,”serve”,”/data/”,”–served-model-name”,”opt-125m”,”–host”,”0.0.0.0”,”–port”,”8000”],”repository”:”vllm/vllm-openai”,”tag”:”latest”}|Image configuration|
|image.command|list|[“vllm”,”serve”,”/data/”,”–served-model-name”,”opt-125m”,”–host”,”0.0.0.0”,”–port”,”8000”]|Container launch command|
|image.repository|string|“vllm/vllm-openai”|Image repository|
|image.tag|string|“latest”|Image tag|
|livenessProbe|object|{“failureThreshold”:3,”httpGet”:{“path”:”/health”,”port”:8000},”initialDelaySeconds”:15,”periodSeconds”:10}|Liveness probe configuration|
|livenessProbe.failureThreshold|int|3|Number of times after which if a probe fails in a row, Kubernetes considers that the overall check has failed: the container is not alive|
|livenessProbe.httpGet|object|{“path”:”/health”,”port”:8000}|Configuration of the Kubelet http request on the server|
|livenessProbe.httpGet.path|string|“/health”|Path to access on the HTTP server|
|livenessProbe.httpGet.port|int|8000|Name or number of the port to access on the container, on which the server is listening|
|livenessProbe.initialDelaySeconds|int|15|Number of seconds after the container has started before liveness probe is initiated|
|livenessProbe.periodSeconds|int|10|How often (in seconds) to perform the liveness probe|
|maxUnavailablePodDisruptionBudget|string|“”|Disruption Budget Configuration|
|readinessProbe|object|{“failureThreshold”:3,”httpGet”:{“path”:”/health”,”port”:8000},”initialDelaySeconds”:5,”periodSeconds”:5}|Readiness probe configuration|
|readinessProbe.failureThreshold|int|3|Number of times after which if a probe fails in a row, Kubernetes considers that the overall check has failed: the container is not ready|
|readinessProbe.httpGet|object|{“path”:”/health”,”port”:8000}|Configuration of the Kubelet http request on the server|
|readinessProbe.httpGet.path|string|“/health”|Path to access on the HTTP server|
|readinessProbe.httpGet.port|int|8000|Name or number of the port to access on the container, on which the server is listening|
|readinessProbe.initialDelaySeconds|int|5|Number of seconds after the container has started before readiness probe is initiated|
|readinessProbe.periodSeconds|int|5|How often (in seconds) to perform the readiness probe|
|replicaCount|int|1|Number of replicas|
|resources|object|{“limits”:{“cpu”:4,”memory”:”16Gi”,”nvidia.com/gpu”:1},”requests”:{“cpu”:4,”memory”:”16Gi”,”nvidia.com/gpu”:1}}|Resource configuration|
|resources.limits.”nvidia.com/gpu”|int|1|Number of gpus used|
|resources.limits.cpu|int|4|Number of CPUs|
|resources.limits.memory|string|“16Gi”|CPU memory configuration|
|resources.requests.”nvidia.com/gpu”|int|1|Number of gpus used|
|resources.requests.cpu|int|4|Number of CPUs|
|resources.requests.memory|string|“16Gi”|CPU memory configuration|
|secrets|object|{}|Secrets configuration|
|serviceName|string||Service name|
|servicePort|int|80|Service port|
|labels.environment|string|test|Environment name|
|labels.release|string|test||
