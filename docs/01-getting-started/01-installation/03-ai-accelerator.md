---

title: 其他 AI 加速器

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


vLLM 是一个 Python 库，支持以下 AI 加速器。根据您的 AI 加速器类型查看供应商特定说明：

#### Google TPU

张量处理单元 (TPU) 是 Google 定制开发的专用集成电路 (ASIC)，用于加速机器学习工作负载。TPU 有不同的版本，每个版本具有不同的硬件规格。有关 TPU 的更多信息，请参阅 [TPU 系统架构](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)。有关 vLLM 支持的 TPU 版本信息，请参阅：


* [TPU v6e](https://cloud.google.com/tpu/docs/v6e)

* [TPU v5e](https://cloud.google.com/tpu/docs/v5e)

* [TPU v5p](https://cloud.google.com/tpu/docs/v5p)

* [TPU v4](https://cloud.google.com/tpu/docs/v4)


这些 TPU 版本允许您配置 TPU 芯片的物理排列方式。这可以提高吞吐量和网络性能。更多信息请参阅：


* [TPU v6e 拓扑结构](https://cloud.google.com/tpu/docs/v6e#configurations)

* [TPU v5e 拓扑结构](https://cloud.google.com/tpu/docs/v5e#tpu-v5e-config)

* [TPU v5p 拓扑结构](https://cloud.google.com/tpu/docs/v5p#tpu-v5p-config)

* [TPU v4 拓扑结构](https://cloud.google.com/tpu/docs/v4#tpu-v4-config)


要使用 Cloud TPU，您需要为 Google Cloud Platform 项目授予 TPU 配额。TPU 配额指定您可以在 GPC 项目中使用的 TPU 数量，并根据 TPU 版本、所需 TPU 数量和配额类型进行定义。更多信息请参阅 [TPU 配额](https://cloud.google.com/tpu/docs/quota#tpu_quota)。


有关 TPU 定价信息，请参阅 [Cloud TPU 定价](https://cloud.google.com/tpu/pricing)。


您可能需要为 TPU 虚拟机提供额外的持久存储。更多信息请参阅 [Cloud TPU 数据存储选项](https://cloud.devsite.corp.google.com/tpu/docs/storage-options)。


**注意**

此设备没有预构建的 wheels，因此您必须使用预构建的 Docker 镜像或从源代码构建 vLLM。


#### Intel Gaudi

此节提供了在 Intel Gaudi 设备上运行 vLLM 的说明。


>**注意**
>此设备没有预构建的 wheels 或镜像，因此您必须从源代码构建 vLLM。

#### AWS Neuron

vLLM 0.3.3 及以上版本支持通过 Neuron SDK 在 AWS Trainium/Inferentia 上进行模型推理和服务，并支持连续批处理。分页注意力（Paged Attention）和分块预填充（Chunked Prefill）功能目前正在开发中，即将推出。Neuron SDK 当前支持的数据类型为 FP16 和 BF16。


>**注意**
>此设备没有预构建的 wheels 或镜像，因此您必须从源代码构建 vLLM。

## 环境要求

### Google TPU

* Google Cloud TPU 虚拟机

* TPU 版本：v6e、v5e、v5p、v4

* Python：3.10 或更高版本


#### 配置 Cloud TPU

您可以使用 [Cloud TPU API](https://cloud.google.com/tpu/docs/reference/rest) 或 [队列资源](https://cloud.google.com/tpu/docs/queued-resources) API 配置 Cloud TPU。本节展示如何使用队列资源 API 创建 TPU。有关使用 Cloud TPU API 的更多信息，请参阅 [使用 Create Node API 创建 Cloud TPU](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm#create-node-api)。队列资源允许您以队列方式请求 Cloud TPU 资源。当您请求队列资源时，请求会被添加到 Cloud TPU 服务维护的队列中。当请求的资源可用时，它将分配给您的 Google Cloud 项目供您独占使用。


>**注意**
>在以下所有命令中，请将全大写的参数名称替换为适当的值。有关参数描述，请参阅参数描述表。
#### 

#### 使用 GKE 配置 Cloud TPU

有关在 GKE 中使用 TPU 的更多信息，请参阅：

* [https://cloud.google.com/kubernetes-engine/docs/how-to/tpus](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus)

* [https://cloud.google.com/kubernetes-engine/docs/concepts/tpus](https://cloud.google.com/kubernetes-engine/docs/concepts/tpus)

* [https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus](https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus)


### Intel Gaudi

* 操作系统：Ubuntu 22.04 LTS

* Python：3.10

* Intel Gaudi 加速器

* Intel Gaudi 软件版本 1.18.0


请按照 [Gaudi 安装指南](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 中的说明设置执行环境。要获得最佳性能，请按照 [优化训练平台指南](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html) 中概述的方法操作。


### AWS Neuron

* 操作系统：Linux

* Python：3.9 – 3.11

* 加速器：NeuronCore_v2（在 trn1/inf2 实例中）

* Pytorch 2.0.1/2.1.1

* AWS Neuron SDK 2.16/2.17（已验证于 Python 3.8）


## 配置新环境

### Google TPU

#### 使用队列资源 API 配置 Cloud TPU

创建具有 4 个 TPU 芯片的 TPU v5e：

```go
gcloud alpha compute tpus queued-resources create QUEUED_RESOURCE_ID \
--node-id TPU_NAME \
--project PROJECT_ID \
--zone ZONE \
--accelerator-type ACCELERATOR_TYPE \
--runtime-version RUNTIME_VERSION \
--service-account SERVICE_ACCOUNT
```


|参数名称|描述|
|:----|:----|
|QUEUED_RESOURCE_ID|用户分配的队列资源请求 ID。|
|TPU_NAME|当队列资源请求被分配时创建的 TPU 的用户分配名称。|
|PROJECT_ID|您的 Google Cloud 项目。|
|ZONE|要在其中创建 Cloud TPU 的 GCP 区域。您使用的值取决于您使用的 TPU 版本。更多信息请参阅 [TPU 区域和区域](https://cloud.google.com/tpu/docs/regions-zones)|
|ACCELERATOR_TYPE|要使用的 TPU 版本。指定 TPU 版本，例如 v5litepod-4 指定具有 4 个核心的 v5e TPU。更多信息请参阅 [TPU 版本](https://cloud.devsite.corp.google.com/tpu/docs/system-architecture-tpu-vm#versions)。|
|RUNTIME_VERSION|要使用的 TPU 虚拟机运行时版本。更多信息请参阅 [TPU 虚拟机镜像](https://cloud.google.com/tpu/docs/runtimes)。|
|SERVICE_ACCOUNT|您的服务账号的电子邮件地址。您可以在 IAM 云控制台的 *服务账号* 下找到它。例如：tpu-service-account@<your_project_ID>.iam.gserviceaccount.com|


通过 SSH 连接到您的 TPU：

```plain
gcloud compute tpus tpu-vm ssh TPU_NAME --zone ZONE
```


### Intel Gaudi

#### 环境验证

要验证 Intel Gaudi 软件是否正确安装，请运行：

```go
hl-smi # 验证 hl-smi 是否在您的 PATH 中，并且每个 Gaudi 加速器可见  
apt list --installed | grep habana # 验证 habanalabs-firmware-tools、habanalabs-graph、habanalabs-rdma-core、habanalabs-thunk 和 habanalabs-container-runtime 是否已安装  
pip list | grep habana # 验证 habana-torch-plugin、habana-torch-dataloader、habana-pyhlml 和 habana-media-loader 是否已安装  
pip list | grep neural # 验证 neural_compressor 是否已安装  
```


更多详细信息请参阅 [Intel Gaudi 软件堆栈验证](https://docs.habana.ai/en/latest/Installation_Guide/SW_Verification.html#platform-upgrade)。

#### 

#### 运行 Docker 镜像

强烈建议使用来自 Intel Gaudi 仓库的最新 Docker 镜像。更多详细信息请参阅 [Intel Gaudi 文档](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#pull-prebuilt-containers)。

使用以下命令运行 Docker 镜像：

```go
docker pull vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest
```


### AWS Neuron

#### 启动 Trn1/Inf2 实例

以下是启动 trn1/inf2 实例的步骤，以便安装 [Ubuntu 22.04 LTS 上的 PyTorch Neuron (](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22.html)["](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22.html)[torch-neuronx](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22.html)["](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22.html)[) 设置](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22.html)。

* 请按照 [启动 Amazon EC2 实例](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance) 中的说明启动实例。在 EC2 控制台选择实例类型时，请确保选择正确的实例类型。

* 有关实例规格和定价的更多信息，请参阅：[Trn1 网页](https://aws.amazon.com/ec2/instance-types/trn1/)、[Inf2 网页](https://aws.amazon.com/ec2/instance-types/inf2/)

* 选择 Ubuntu Server 22.04 TLS AMI

* 启动 Trn1/Inf2 实例时，请将主 EBS 卷大小调整为至少 512GB。

* 启动实例后，按照 [连接到您的实例](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html) 中的说明连接到实例

#### 

#### 安装驱动程序和工具

如果已安装 [Deep Learning AMI Neuron](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html)，则无需安装驱动程序和工具。如果操作系统未安装驱动程序和工具，请按照以下步骤操作：

```plain
# Configure Linux for Neuron repository updates
# 为 Neuron 仓库配置 Linux  
. /etc/os-release
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -


# Update OS packages
# 更新操作系统包  
sudo apt-get update -y


# Install OS headers
# 安装操作系统头文件  
sudo apt-get install linux-headers-$(uname -r) -y


# Install git
# 安装 git  
sudo apt-get install git -y


# install Neuron Driver
# 安装 Neuron 驱动
sudo apt-get install aws-neuronx-dkms=2.* -y


# Install Neuron Runtime
# 安装 Neuron 运行时  
sudo apt-get install aws-neuronx-collectives=2.* -y
sudo apt-get install aws-neuronx-runtime-lib=2.* -y


# Install Neuron Tools
# 安装 Neuron 工具  
sudo apt-get install aws-neuronx-tools=2.* -y


# Add PATH
# 添加 PATH  
export PATH=/opt/aws/neuron/bin:$PATH
```


## 使用 Python 设置

### 预构建的 wheels

#### Google TPU

当前没有预构建的 TPU wheels。

#### Intel Gaudi

当前没有预构建的 Intel Gaudi wheels。

#### AWS Neuron

当前没有预构建的 Neuron wheels。

### 

### 从源代码构建 wheel

### Google TPU

安装 Miniconda：

```plain
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```


创建并激活 vLLM 的 Conda 环境：

```plain
conda create -n vllm python=3.10 -y
conda activate vllm
```


克隆 vLLM 仓库并进入 vLLM 目录：

```plain
git clone https://github.com/vllm-project/vllm.git && cd vllm
```


卸载现有的 `torch` 和 `torch_xla` 包：

```plain
pip uninstall torch torch-xla -y
```


安装构建依赖项：

```plain
pip install -r requirements/tpu.txt
sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev
```


运行安装脚本：

```plain
VLLM_TARGET_DEVICE="tpu" python setup.py develop
```


### Intel Gaudi

要从源代码构建并安装 vLLM，请运行：

```go
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/hpu.txt
python setup.py develop
```


当前，最新的功能和性能优化在 Gaudi 的 [vLLM-fork](https://github.com/HabanaAI/vllm-fork) 中开发，并定期同步到 vLLM 主仓库。要安装最新的 [HabanaAI/vLLM-fork](https://github.com/HabanaAI/vllm-fork)，请运行以下命令：

```go
git clone https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
git checkout habana_main
pip install -r requirements/hpu.txt
python setup.py develop
```


### AWS Neuron

**注意**

当前支持的 Neuron Pytorch 版本安装了 `triton` 版本 `2.1.0`。这与 `vllm >= 0.5.3` 不兼容。您可能会看到错误 `cannot import name 'default_dump_dir...`。要解决此问题，请在安装 vLLM wheel 后运行 `pip install --upgrade triton==3.0.0`。


以下说明适用于 Neuron SDK 2.16 及更高版本。

#### 

#### 安装 transformers-neuronx 及其依赖项

[transformers-neuronx](https://github.com/aws-neuron/transformers-neuronx) 将作为在 trn1/inf2 实例上支持推理的后端。按照以下步骤安装 transformer-neuronx 包及其依赖项。

```plain
# Install Python venv
# 安装 Python venv  
sudo apt-get install -y python3.10-venv g++


# Create Python venv
# 创建 Python venv  
python3.10 -m venv aws_neuron_venv_pytorch


# Activate Python venv
# 激活 Python venv  
source aws_neuron_venv_pytorch/bin/activate


# Install Jupyter notebook kernel
# 安装 Jupyter notebook 内核  
pip install ipykernel
python3.10 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
pip install jupyter notebook
pip install environment_kernels


# Set pip repository pointing to the Neuron repository
# 将 pip 仓库指向 Neuron 仓库 
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com


# Install wget, awscli
# 安装 wget、awscli  
python -m pip install wget
python -m pip install awscli


# Update Neuron Compiler and Framework
# 更新 Neuron 编译器和框架  
python -m pip install --upgrade neuronx-cc==2.* --pre torch-neuronx==2.1.* torchvision transformers-neuronx
```


#### 从源代码安装 vLLM

安装 neuronx-cc 和 transformers-neuronx 包后，可按如下方式安装 vllm：

```go
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -U -r requirements/neuron.txt
VLLM_TARGET_DEVICE="neuron" pip install .
```


如果安装过程中正确检测到 neuron 包，将安装 `vllm-0.3.0+neuron212`。

## 

## 使用 Docker 设置

### Pre-built images

### 预构建的镜像

#### Google TPU

请参阅 [使用 vLLM 的官方 Docker 镜像](https://docs.vllm.ai/en/latest/deployment/docker.html#deployment-docker-pre-built-image) 以获取使用官方 Docker 镜像的说明，确保将镜像名称 `vllm/vllm-openai` 替换为 `vllm/vllm-tpu`。

#### Intel Gaudi

当前没有预构建的 Intel Gaudi 镜像。

#### AWS Neuron

当前没有预构建的 Neuron 镜像。

### 

### 从源代码构建镜像

#### Google TPU

您可以使用 [Dockerfile.tpu](https://github.com/vllm-project/vllm/blob/main/Dockerfile.tpu) 构建支持 TPU 的 Docker 镜像。

```go
docker build -f Dockerfile.tpu -t vllm-tpu .
```


使用以下命令运行 Docker 镜像：

```plain
# Make sure to add `--privileged --net host --shm-size=16G`.
# 确保添加 `--privileged --net host --shm-size=16G`  
docker run --privileged --net host --shm-size=16G -it vllm-tpu
```


**注意**

由于 TPU 依赖需要静态形状的 XLA，vLLM 将可能的输入形状分桶，并为每个形状编译一个 XLA 图。首次运行时编译可能需要 20~30 分钟。但由于 XLA 图会缓存到磁盘（默认在 `VLLM_XLA_CACHE_PATH` 或 `~/.cache/vllm/xla_cache`），后续编译时间将减少到约 5 分钟。


**提示**

如果遇到以下错误：

```go
from torch._C import *  # noqa: F403
ImportError: libopenblas.so.0: cannot open shared object file: No such
file or directory
```


请使用以下命令安装 OpenBLAS：

```go
sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev
```


#### Intel Gaudi

```go
docker build -f Dockerfile.hpu -t vllm-hpu-env  .
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --rm vllm-hpu-env
```


**提示**

如果遇到以下错误：`docker: Error response from daemon: Unknown runtime specified habana.`，请参阅 [Intel Gaudi 软件堆栈和驱动安装](https://docs.habana.ai/en/v1.18.0/Installation_Guide/Bare_Metal_Fresh_OS.html) 的「使用容器安装」部分。确保已安装 `habana-container-runtime` 包，并且 `habana` 容器运行时已注册。


#### AWS Neuron

有关构建 Docker 镜像的说明，请参阅 [从源代码构建 vLLM 的 Docker 镜像](https://docs.vllm.ai/en/latest/deployment/docker.html#deployment-docker-build-image-from-source)。

确保使用 [Dockerfile.neuron](https://github.com/vllm-project/vllm/blob/main/Dockerfile.neuron) 替代默认的 Dockerfile。


# Extra information

## Google TPU

此设备没有额外信息。


## Intel Gaudi

### 支持的功能

* [离线推理](https://docs.vllm.ai/en/latest/serving/offline_inference.html#offline-inference)

* 通过 [OpenAI 兼容服务器](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#openai-compatible-server) 进行在线服务

* HPU 自动检测 - 无需在 vLLM 中手动选择设备

* 针对 Intel Gaudi 加速器优化的分页 KV 缓存算法

* 针对 Intel Gaudi 的定制分页注意力、KV 缓存操作、预填充注意力、均方根层归一化、旋转位置编码实现

* 多卡推理的張量并行支持

* 使用 [HPU 图](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) 加速低批量延迟和吞吐量的推理

* 带线性偏置的注意力（ALiBi）


### 不支持的功能

* 波束搜索

* LoRA 适配器

* 量化

* 预填充分块（混合批量推理）


### 支持的配置

以下配置已验证可在 Gaudi2 设备上运行。未列出的配置可能无法正常工作。

* [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) 在单 HPU 上，或通过張量并行在 2x 和 8x HPU 上，BF16 数据类型，随机或贪婪采样

* [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) 在单 HPU 上，或通过張量并行在 2x 和 8x HPU 上，BF16 数据类型，随机或贪婪采样

* [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) 在单 HPU 上，或通过張量并行在 2x 和 8x HPU 上，BF16 数据类型，随机或贪婪采样

* [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 在单 HPU 上，或通过張量并行在 2x 和 8x HPU 上，BF16 数据类型，随机或贪婪采样

* [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) 在单 HPU 上，或通过張量并行在 2x 和 8x HPU 上，BF16 数据类型，随机或贪婪采样

* [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) 在单 HPU 上，或通过張量并行在 2x 和 8x HPU 上，BF16 数据类型，随机或贪婪采样

* [meta-llama/Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b) 通过張量并行在 8x HPU 上，BF16 数据类型，随机或贪婪采样

* [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) 通过張量并行在 8x HPU 上，BF16 数据类型，随机或贪婪采样

* [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) 通过張量并行在 8x HPU 上，BF16 数据类型，随机或贪婪采样

* [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) 通过張量并行在 8x HPU 上，BF16 数据类型，随机或贪婪采样

* [meta-llama/Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B) 通过張量并行在 8x HPU 上，BF16 数据类型，随机或贪婪采样

* [meta-llama/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) 通过張量并行在 8x HPU 上，BF16 数据类型，随机或贪婪采样

### 

### 性能调优

#### 执行模式

当前在 vLLM 中，HPU 支持四种执行模式，具体取决于所选的 HPU PyTorch Bridge 后端（通过 `PT_HPU_LAZY_MODE` 环境变量）和 `--enforce-eager` 标志。

|PT_HPU_LAZY_MODE|enforce_eager|执行模式|
|:----|:----|:----|
|0|0|torch.compile|
|0|1|PyTorch 即时模式|
|1|0|HPU 图|
|1|1|PyTorch 延迟模式|


>**警告**
>在 1.18.0 版本中，所有使用 `PT_HPU_LAZY_MODE=0` 的模式均为高度实验性，仅应用于验证功能正确性。它们的性能将在后续版本中改进。要在 1.18.0 中获得最佳性能，请使用 HPU 图或 PyTorch 延迟模式。
#### 

#### 分桶机制

Intel Gaudi 加速器在操作固定张量形状的模型时表现最佳。[Intel Gaudi 图编译器](https://docs.habana.ai/en/latest/Gaudi_Overview/Intel_Gaudi_Software_Suite.html#graph-compiler-and-runtime) 负责生成在 Gaudi 上实现给定模型拓扑的优化二进制代码。在默认配置中，生成的二进制代码可能高度依赖输入和输出张量形状，并且在同一拓扑中遇到不同形状的张量时可能需要重新编译图。虽然生成的二进制代码能高效利用 Gaudi，但编译本身可能会在端到端执行中引入明显的开销。在动态推理服务场景中，需要尽量减少图编译次数，并降低图编译在服务器运行时发生的风险。目前通过「分桶」模型的向前传递在两个维度（`batch_size` 和 `sequence_length`）来实现这一点。


**注意**

分桶允许我们显著减少所需图的数量，但它不处理任何图编译和设备代码生成——这是在预热和 HPUGraph 捕获阶段完成的。


分桶范围由三个参数确定——`min`、`step` 和 `max`。它们可以分别为提示和解码阶段以及批量大小和序列长度维度单独设置。这些参数可以在 vLLM 启动期间的日志中观察到：

```plain
INFO 08-01 21:37:59 hpu_model_runner.py:493] Prompt bucket config (min, step, max_warmup) bs:[1, 32, 4], seq:[128, 128, 1024]
INFO 08-01 21:37:59 hpu_model_runner.py:499] Generated 24 prompt buckets: [(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024)]
INFO 08-01 21:37:59 hpu_model_runner.py:504] Decode bucket config (min, step, max_warmup) bs:[1, 128, 4], seq:[128, 128, 2048]
INFO 08-01 21:37:59 hpu_model_runner.py:509] Generated 48 decode buckets: [(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (1, 1152), (1, 1280), (1, 1408), (1, 1536), (1, 1664), (1, 1792), (1, 1920), (1, 2048), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (2, 1152), (2, 1280), (2, 1408), (2, 1536), (2, 1664), (2, 1792), (2, 1920), (2, 2048), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024), (4, 1152), (4, 1280), (4, 1408), (4, 1536), (4, 1664), (4, 1792), (4, 1920), (4, 2048)]
```


`min` 确定桶的最低值。`step` 确定桶之间的间隔，`max` 确定桶的上限。此外，`min` 和 `step` 之间的间隔有特殊处理——`min` 会被连续乘以二的幂次，直到达到 `step`。我们称此为“上升阶段”，它用于以最小浪费处理较低批量大小，同时允许在较大批量大小上进行较大填充。


示例（带上升阶段）

```plain
min = 2, step = 32, max = 64
=> ramp_up = (2, 4, 8, 16)
=> stable = (32, 64)
=> buckets = ramp_up + stable => (2, 4, 8, 16, 32, 64)
```


示例（不带上升阶段）

```plain
min = 128, step = 128, max = 512
=> ramp_up = ()
=> stable = (128, 256, 384, 512)
=> buckets = ramp_up + stable => (128, 256, 384, 512)
```


在记录的示例中，为提示（预填充）阶段生成了 24 个桶，为解码阶段生成了 48 个桶。每个桶对应具有指定张量形状的模型的单独优化设备二进制文件。每当处理一批请求时，它会在批量和序列长度维度上填充到最小的可能桶。


>**警告**
>如果请求在任何维度上超过最大桶大小，则将在不填充的情况下处理，并且其处理可能需要图编译，从而显著增加端到端延迟。桶的边界可通过环境变量由用户配置，可以增加桶的上限以避免此类情况。

例如，如果空闲的 vLLM 服务器收到一个包含 3 个序列、最大序列长度为 412 的请求，则其将被填充为 `(4, 512)` 预填充桶，因为批量大小（序列数）将被填充到 4（最接近且大于 3 的批量大小维度），最大序列长度将被填充到 512（最接近且大于 412 的序列长度维度）。预填充阶段后，它将作为 `(4, 512)` 解码桶执行，并持续作为该桶，直到批量维度发生变化（由于请求完成）——此时它将变为 `(2, 512)` 桶，或上下文长度超过 512 个 token——此时它将变为 `(4, 640)` 桶。


>**注意**
>分桶对客户端透明——序列长度维度的填充永远不会返回给客户端，批量维度的填充不会创建新请求。

#### 预热 (Warmup)

预热是在 vLLM 服务器开始监听之前可选但强烈推荐的步骤。它使用虚拟数据为每个桶执行向前传递。目标是在服务器运行时预编译所有图，避免在桶边界内产生任何图编译开销。每个预热步骤在 vLLM 启动期间记录：

```plain
INFO 08-01 22:26:47 hpu_model_runner.py:1066] [Warmup][Prompt][1/24] batch_size:4 seq_len:1024 free_mem:79.16 GiB
INFO 08-01 22:26:47 hpu_model_runner.py:1066] [Warmup][Prompt][2/24] batch_size:4 seq_len:896 free_mem:55.43 GiB
INFO 08-01 22:26:48 hpu_model_runner.py:1066] [Warmup][Prompt][3/24] batch_size:4 seq_len:768 free_mem:55.43 GiB
...
INFO 08-01 22:26:59 hpu_model_runner.py:1066] [Warmup][Prompt][24/24] batch_size:1 seq_len:128 free_mem:55.43 GiB
INFO 08-01 22:27:00 hpu_model_runner.py:1066] [Warmup][Decode][1/48] batch_size:4 seq_len:2048 free_mem:55.43 GiB
INFO 08-01 22:27:00 hpu_model_runner.py:1066] [Warmup][Decode][2/48] batch_size:4 seq_len:1920 free_mem:55.43 GiB
INFO 08-01 22:27:01 hpu_model_runner.py:1066] [Warmup][Decode][3/48] batch_size:4 seq_len:1792 free_mem:55.43 GiB
...
INFO 08-01 22:27:16 hpu_model_runner.py:1066] [Warmup][Decode][47/48] batch_size:2 seq_len:128 free_mem:55.43 GiB
INFO 08-01 22:27:16 hpu_model_runner.py:1066] [Warmup][Decode][48/48] batch_size:1 seq_len:128 free_mem:55.43 GiB
```


此示例使用与 [分桶机制](https://docs.vllm.ai/en/latest/getting_started/installation/ai_accelerator.html?device=hpu-gaudi#gaudi-bucketing-mechanism) 部分相同的桶。每行输出对应单个桶的执行。当桶首次执行时，其图将被编译并可在后续重复使用，跳过进一步的图编译。


>**提示**
>编译所有桶可能需要一些时间，可以通过 `VLLM_SKIP_WARMUP=true` 环境变量关闭。请注意，如果这样做，您可能在首次执行给定桶时面临图编译。在开发过程中关闭预热是可以的，但在部署中强烈建议启用。
#### 

#### HPU 图捕获

[HPU 图](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) 当前是 Intel Gaudi 上 vLLM 最高性能的执行方法。启用 HPU 图后，执行图将提前追踪（记录），以便在推理期间重放，显著减少主机开销。记录可能会占用大量内存，这在分配 KV 缓存时需要考虑。启用 HPU 图将影响可用 KV 缓存块的数量，但 vLLM 提供用户可配置的变量来控制内存管理。


当使用 HPU 图时，它们与 KV 缓存共享公共内存池（“可用内存”），由 `gpu_memory_utilization` 标志（默认为 `0.9`）确定。在分配 KV 缓存之前，模型权重将加载到设备上，并在虚拟数据上执行模型的向前传递以估计内存使用情况。之后，`gpu_memory_utilization` 标志将被使用——其默认值将标记此时设备空闲内存的 90% 为可用。接下来分配 KV 缓存，模型被预热，并捕获 HPU 图。环境变量 `VLLM_GRAPH_RESERVED_MEM` 定义保留用于 HPU 图捕获的内存比例。其默认值（`VLLM_GRAPH_RESERVED_MEM=0.1`）表示 10% 的可用内存将保留给图捕获（后称为“可用图内存”），剩余的 90% 用于 KV 缓存。环境变量 `VLLM_GRAPH_PROMPT_RATIO` 确定保留给预填充和解码图的可用图内存比例。默认情况下（`VLLM_GRAPH_PROMPT_RATIO=0.3`），两个阶段具有相同的内存约束。较低的值对应预填充阶段保留的可用图内存较少，例如 `VLLM_GRAPH_PROMPT_RATIO=0.2` 将为预填充图保留 20% 的可用图内存，为解码图保留 80%。


>**注意**
>`gpu_memory_utilization` 并不对应 HPU 的绝对内存使用量。它指定在加载模型并执行性能分析运行后的内存余量。如果设备总内存为 100 GiB，加载模型权重并执行性能分析运行后空闲内存为 50 GiB，默认 `gpu_memory_utilization` 将标记 50 GiB 的 90% 为可用，留出 5 GiB 余量，无论设备总内存如何。

用户还可以分别为预填充和解码阶段配置 HPU 图的捕获策略。策略影响图的捕获顺序。已实现两种策略：- `max_bs` - 图捕获队列按批量大小降序排序。批量大小相同的桶按序列长度升序排序（例如 `(64, 128)`、`(64, 256)`、`(32, 128)`、`(32, 256)`、`(1, 128)`、`(1, 256)`），解码的默认策略 - `min_tokens` - 图捕获队列按每个图处理的 token 数量（`batch_size*sequence_length`）升序排序，预填充的默认策略。


当有大量请求挂起时，vLLM 调度器将尝试尽快填充解码的最大批量大小。当请求完成时，解码批量大小减少。此时，vLLM 将尝试为等待队列中的请求安排预填充迭代，以将解码批量大小恢复到之前的状态。这意味着在满载场景中，解码批量大小通常处于最大值，这使得捕获大批量 HPU 图至关重要，这反映在 `max_bs` 策略中。另一方面，预填充最常以极低的批量大小（1-4）执行，这反映在 `min_tokens` 策略中。


>**注意**
>`VLLM_GRAPH_PROMPT_RATIO` 并未为每个阶段（预填充和解码）设置严格的内存限制。vLLM 将首先尝试为预填充 HPU 图使用全部可用预填充图内存（可用图内存 * `VLLM_GRAPH_PROMPT_RATIO`），接着对解码图执行相同操作。如果一个阶段已完全捕获，并且可用图内存池中有剩余内存，vLLM 将尝试为另一个阶段捕获更多图，直到无法在不超出保留内存池的情况下捕获更多 HPU 图。此机制的行为可在以下示例中观察到。

每个描述的步骤均由 vLLM 服务器记录，如下所示（负值表示内存被释放）：

```plain
INFO 08-02 17:37:44 hpu_model_runner.py:493] Prompt bucket config (min, step, max_warmup) bs:[1, 32, 4], seq:[128, 128, 1024]
INFO 08-02 17:37:44 hpu_model_runner.py:499] Generated 24 prompt buckets: [(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024)]
INFO 08-02 17:37:44 hpu_model_runner.py:504] Decode bucket config (min, step, max_warmup) bs:[1, 128, 4], seq:[128, 128, 2048]
INFO 08-02 17:37:44 hpu_model_runner.py:509] Generated 48 decode buckets: [(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (1, 1152), (1, 1280), (1, 1408), (1, 1536), (1, 1664), (1, 1792), (1, 1920), (1, 2048), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (2, 1152), (2, 1280), (2, 1408), (2, 1536), (2, 1664), (2, 1792), (2, 1920), (2, 2048), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024), (4, 1152), (4, 1280), (4, 1408), (4, 1536), (4, 1664), (4, 1792), (4, 1920), (4, 2048)]
INFO 08-02 17:37:52 hpu_model_runner.py:430] Pre-loading model weights on hpu:0 took 14.97 GiB of device memory (14.97 GiB/94.62 GiB used) and 2.95 GiB of host memory (475.2 GiB/1007 GiB used)
INFO 08-02 17:37:52 hpu_model_runner.py:438] Wrapping in HPU Graph took 0 B of device memory (14.97 GiB/94.62 GiB used) and -252 KiB of host memory (475.2 GiB/1007 GiB used)
INFO 08-02 17:37:52 hpu_model_runner.py:442] Loading model weights took in total 14.97 GiB of device memory (14.97 GiB/94.62 GiB used) and 2.95 GiB of host memory (475.2 GiB/1007 GiB used)
INFO 08-02 17:37:54 hpu_worker.py:134] Model profiling run took 504 MiB of device memory (15.46 GiB/94.62 GiB used) and 180.9 MiB of host memory (475.4 GiB/1007 GiB used)
INFO 08-02 17:37:54 hpu_worker.py:158] Free device memory: 79.16 GiB, 39.58 GiB usable (gpu_memory_utilization=0.5), 15.83 GiB reserved for HPUGraphs (VLLM_GRAPH_RESERVED_MEM=0.4), 23.75 GiB reserved for KV cache
INFO 08-02 17:37:54 hpu_executor.py:85] # HPU blocks: 1519, # CPU blocks: 0
INFO 08-02 17:37:54 hpu_worker.py:190] Initializing cache engine took 23.73 GiB of device memory (39.2 GiB/94.62 GiB used) and -1.238 MiB of host memory (475.4 GiB/1007 GiB used)
INFO 08-02 17:37:54 hpu_model_runner.py:1066] [Warmup][Prompt][1/24] batch_size:4 seq_len:1024 free_mem:55.43 GiB
...
INFO 08-02 17:38:22 hpu_model_runner.py:1066] [Warmup][Decode][48/48] batch_size:1 seq_len:128 free_mem:55.43 GiB
INFO 08-02 17:38:22 hpu_model_runner.py:1159] Using 15.85 GiB/55.43 GiB of free device memory for HPUGraphs, 7.923 GiB for prompt and 7.923 GiB for decode (VLLM_GRAPH_PROMPT_RATIO=0.3)
INFO 08-02 17:38:22 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][1/24] batch_size:1 seq_len:128 free_mem:55.43 GiB
...
INFO 08-02 17:38:26 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][11/24] batch_size:1 seq_len:896 free_mem:48.77 GiB
INFO 08-02 17:38:27 hpu_model_runner.py:1066] [Warmup][Graph/Decode][1/48] batch_size:4 seq_len:128 free_mem:47.51 GiB
...
INFO 08-02 17:38:41 hpu_model_runner.py:1066] [Warmup][Graph/Decode][48/48] batch_size:1 seq_len:2048 free_mem:47.35 GiB
INFO 08-02 17:38:41 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][12/24] batch_size:4 seq_len:256 free_mem:47.35 GiB
INFO 08-02 17:38:42 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][13/24] batch_size:2 seq_len:512 free_mem:45.91 GiB
INFO 08-02 17:38:42 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][14/24] batch_size:1 seq_len:1024 free_mem:44.48 GiB
INFO 08-02 17:38:43 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][15/24] batch_size:2 seq_len:640 free_mem:43.03 GiB
INFO 08-02 17:38:43 hpu_model_runner.py:1128] Graph/Prompt captured:15 (62.5%) used_mem:14.03 GiB buckets:[(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (4, 128), (4, 256)]
INFO 08-02 17:38:43 hpu_model_runner.py:1128] Graph/Decode captured:48 (100.0%) used_mem:161.9 MiB buckets:[(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (1, 1152), (1, 1280), (1, 1408), (1, 1536), (1, 1664), (1, 1792), (1, 1920), (1, 2048), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (2, 1152), (2, 1280), (2, 1408), (2, 1536), (2, 1664), (2, 1792), (2, 1920), (2, 2048), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024), (4, 1152), (4, 1280), (4, 1408), (4, 1536), (4, 1664), (4, 1792), (4, 1920), (4, 2048)]
INFO 08-02 17:38:43 hpu_model_runner.py:1206] Warmup finished in 49 secs, allocated 14.19 GiB of device memory
INFO 08-02 17:38:43 hpu_executor.py:91] init_cache_engine took 37.92 GiB of device memory (53.39 GiB/94.62 GiB used) and 57.86 MiB of host memory (475.4 GiB/1007 GiB used)
```


#### 推荐的 vLLM 参数

* 我们建议在 Gaudi 2 上使用 `block_size` 为 128 进行 BF16 数据类型的推理。使用默认值（16、32）可能由于矩阵乘法引擎利用不足而导致性能欠佳（参阅 [Gaudi 架构](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)）。

* 对于 Llama 7B 的最大吞吐量，我们建议启用 HPU 图，批量大小为 128 或 256，最大上下文长度为 2048。如果遇到内存不足问题，请参阅故障排除部分。

#### 

#### 环境变量

**诊断和分析旋钮：**

* `VLLM_PROFILER_ENABLED`：如果为 `true`，将启用高级分析器。生成的 JSON 跟踪可在 [perfetto.habana.ai](https://perfetto.habana.ai/#!/viewer) 查看。默认禁用。

* `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION`：如果为 `true`，将记录每个 vLLM 引擎步骤的图编译（仅在发生编译时）。强烈建议与 `PT_HPU_METRICS_GC_DETAILS=1` 一起使用。默认禁用。

* `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL`：如果为 `true`，将始终记录每个 vLLM 引擎步骤的图编译（即使未发生）。默认禁用。

* `VLLM_HPU_LOG_STEP_CPU_FALLBACKS`：如果为 `true`，将记录每个 vLLM 引擎步骤的 CPU 回退（仅在发生回退时）。默认禁用。

* `VLLM_HPU_LOG_STEP_CPU_FALLBACKS_ALL`：如果为 `true`，将始终记录每个 vLLM 引擎步骤的 CPU 回退（即使未发生）。默认禁用。


**性能调优旋钮：**

* `VLLM_SKIP_WARMUP`：如果为 `true`，将跳过预热，默认 `false`

* `VLLM_GRAPH_RESERVED_MEM`：保留用于 HPU 图捕获的内存百分比，默认 `0.1`

* `VLLM_GRAPH_PROMPT_RATIO`：保留用于预填充图的图内存百分比，默认 `0.3`

* `VLLM_GRAPH_PROMPT_STRATEGY`：确定预填充图捕获顺序的策略，`min_tokens` 或 `max_bs`，默认 `min_tokens`

* `VLLM_GRAPH_DECODE_STRATEGY`：确定解码图捕获顺序的策略，`min_tokens` 或 `max_bs`，默认 `max_bs`

* `VLLM_{phase}_{dim}_BUCKET_{param}` - 共 12 个环境变量，用于配置分桶机制的范围

   * `{phase}` 为 `PROMPT` 或 `DECODE`

   * `{dim}` 为 `BS`、`SEQ` 或 `BLOCK`

   * `{param}` 为 `MIN`、`STEP` 或 `MAX`

   * 默认值：

      * 预填充：

         * 批量大小最小值（`VLLM_PROMPT_BS_BUCKET_MIN`）：`1`

         * 批量大小步长（`VLLM_PROMPT_BS_BUCKET_STEP`）：`min(max_num_seqs, 32)`

         * 批量大小最大值（`VLLM_PROMPT_BS_BUCKET_MAX`）：`min(max_num_seqs, 64)`

         * 序列长度最小值（`VLLM_PROMPT_SEQ_BUCKET_MIN`）：`block_size`

         * 序列长度步长（`VLLM_PROMPT_SEQ_BUCKET_STEP`）：`block_size`

         * 序列长度最大值（`VLLM_PROMPT_SEQ_BUCKET_MAX`）：`max_model_len`

      * 解码：

         * 批量大小最小值（`VLLM_DECODE_BS_BUCKET_MIN`）：`1`

         * 批量大小步长（`VLLM_DECODE_BS_BUCKET_STEP`）：`min(max_num_seqs, 32)`

         * 批量大小最大值（`VLLM_DECODE_BS_BUCKET_MAX`）：`max_num_seqs`

         * 序列长度最小值（`VLLM_DECODE_BLOCK_BUCKET_MIN`）：`block_size`

         * 序列长度步长（`VLLM_DECODE_BLOCK_BUCKET_STEP`）：`block_size`

         * 序列长度最大值（`VLLM_DECODE_BLOCK_BUCKET_MAX`）：`max(128, (max_num_seqs*max_model_len)/block_size)`


此外，以下 HPU PyTorch Bridge 环境变量会影响 vLLM 执行：

* `PT_HPU_LAZY_MODE`：如果为 `0`，将使用 Gaudi 的 PyTorch 即时后端；如果为 `1`，将使用 Gaudi 的 PyTorch 延迟后端，默认 `1`

* `PT_HPU_ENABLE_LAZY_COLLECTIVES`：必须为 `true` 以支持使用 HPU 图的张量并行推理

### 

### 故障排除：调整 HPU 图

如果您遇到设备内存不足问题或尝试以更高批量大小进行推理，请按照以下步骤调整 HPU 图：

* 调整 `gpu_memory_utilization` 旋钮。这将减少 KV 缓存的分配，为捕获更大批量的图留出空间。默认 `gpu_memory_utilization` 设置为 0.9，尝试分配约 90% 的 HBM 剩余内存用于 KV 缓存。注意，减少此值会减少可用 KV 缓存块的数量，从而减少给定时间内可处理的最大 token 数。

* 如果此方法无效，您可以完全禁用 `HPUGraph`。禁用 HPU 图后，您将以较低批次的延迟和吞吐量为代价，换取较高批次下潜在的更高吞吐量。您可以通过向服务器添加 `--enforce-eager` 标志（用于在线服务）或向 LLM 构造函数传递 `enforce_eager=True` 参数（用于离线推理）来实现此目的。


## AWS Neuron

此设备没有额外信息。

