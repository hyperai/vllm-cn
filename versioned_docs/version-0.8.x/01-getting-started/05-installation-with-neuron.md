---
title: 使用 Neuron 安装
---


从 vLLM 0.3.3 版本起，支持在带有 Neuron SDK 的 AWS Trainium/Inferentia 上进行模型推理和服务。目前 Neuron SDK 不支持分页注意力 (Paged Attention)，但 Transformers-neuronx 支持简单的连续批处理。Neuron SDK 目前支持的数据类型为 FP16 和 BF16。


## 依赖环境

* 操作系统：Linux

* Python：3.8 -- 3.11

* 加速器：NeuronCore_v2（在 trn1/inf2 实例中）

* Pytorch 2.0.1/2.1.1

* AWS Neuron SDK 2.16/2.17（在 python 3.8 上验证）


安装步骤:


*  [从源代码构建](#从源代码构建)

   * [步骤 0. 启动 Trn1/Inf2 实例](#步骤-0-启动-trn1inf2-实例)

   * [步骤 1. 安装驱动程序和工具](#步骤-1-安装驱动程序和工具)

   * [步骤 2. 安装 Transformers-neuronx 及其依赖](#步骤-2-安装-transformers-neuronx-及其依赖)

   * [步骤 3. 从源代码安装 vLLM](#步骤-3-从源代码安装-vllm)

##

## 从源代码构建

以下说明适用于 Neuron SDK 2.16 及更高版本。


### 步骤 0. 启动 Trn1/Inf2 实例

以下是启动 trn1/inf2 实例的步骤，以便在 Ubuntu 22.04 LTS 上安装 [PyTorch Neuron ("torch-neuronx") 设置](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22.html)。

* 请按照[启动 Amazon EC2 实例](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance)中的说明启动实例。在 EC2 控制台选择实例类型时，请确保选择正确的实例类型。

* 要获取有关实例大小和定价的更多信息，请参阅：[Trn1 网页](https://aws.amazon.com/ec2/instance-types/trn1/)、[Inf2 网页](https://aws.amazon.com/ec2/instance-types/inf2/)。

* 选择 Ubuntu Server 22.04 TLS AMI

* 启动 Trn1/Inf2 时，请将您的主 EBS 卷大小调整为至少 512GB。

* 启动实例后，按照[连接到您的实例](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)中的说明连接到实例


### 步骤 1. 安装驱动程序和工具

如果 [Deep Learning AMI Neuron](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html) 已安装，则无需安装驱动程序和工具。如果操作系统上未安装驱动程序和工具，请按照以下步骤操作:

```plain
# Configure Linux for Neuron repository updates
# 配置 Linux 以进行 Neuron 存储库更新

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
# 安装git

sudo apt-get install git -y

# install Neuron Driver
# 安装 Neuron 驱动

sudo apt-get install aws-neuronx-dkms=2.* -y

# Install Neuron Runtime
# 安装 Neuron 运行时

sudo apt-get install aws-neuronx-collectives=2.* -y
sudo apt-get install aws-neuronx-runtime-lib=2.* -y

# Install Neuron Tools
# 安装 Neuron Tools

sudo apt-get install aws-neuronx-tools=2.* -y

# Add PATH
# 添加路径

export PATH=/opt/aws/neuron/bin:$PATH
```


### 步骤 2. 安装 Transformers-neuronx 及其依赖

[transformers-neuronx](https://github.com/aws-neuron/transformers-neuronx) 将作为后端来支持在 trn1/inf2 实例上进行推理。请按照以下步骤安装 Transformer-neuronx 包及其依赖。

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
# 安装 Jupyter Notebook 内核

pip install ipykernel
python3.10 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
pip install jupyter notebook
pip install environment_kernels

# Set pip repository pointing to the Neuron repository
# 设置 pip 存储库指向 Neuron 存储库

python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install wget, awscli
# 安装 wget、awscli

python -m pip install wget
python -m pip install awscli

# Update Neuron Compiler and Framework
# 更新 Neuron 编译器和框架

python -m pip install --upgrade neuronx-cc==2.* --pre torch-neuronx==2.1.* torchvision transformers-neuronx
```


### 步骤 3. 从源代码安装 vLLM

一旦安装了 neuronx-cc 和 transformers-neuronx 软件包，我们就能安装 vllm 了，如下所示：

```plain
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -U -r requirements-neuron.txt
VLLM_TARGET_DEVICE="neuron" pip install .
```


如果在安装过程中正确检测到 neuron 包，则会安装 `vllm-0.3.0+neuron212`。
