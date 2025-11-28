---
title: 使用 Nginx
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

本文档介绍如何启动多个 vLLM 服务器容器，并使用 Nginx 作为负载均衡器在这些服务器之间进行流量分配。

目录

1. [构建 Nginx 容器](https://docs.vllm.ai/en/latest/deployment/nginx.html#nginxloadbalancer-nginx-build)

2. [创建简单的 Nginx 配置文件](https://docs.vllm.ai/en/latest/deployment/nginx.html#nginxloadbalancer-nginx-conf)

3. [构建 vLLM 容器](https://docs.vllm.ai/en/latest/deployment/nginx.html#nginxloadbalancer-nginx-vllm-container)

4. [创建 Docker 网络](https://docs.vllm.ai/en/latest/deployment/nginx.html#nginxloadbalancer-nginx-docker-network)

5. [启动 vLLM 容器](https://docs.vllm.ai/en/latest/deployment/nginx.html#nginxloadbalancer-nginx-launch-container)

6. [启动 Nginx](https://docs.vllm.ai/en/latest/deployment/nginx.html#nginxloadbalancer-nginx-launch-nginx)

7. [验证 vLLM 服务器是否准备就绪](https://docs.vllm.ai/en/latest/deployment/nginx.html#nginxloadbalancer-nginx-verify-nginx)

## 构建 Nginx 容器

本文假设你已克隆 vLLM 项目，并位于 vLLM 根目录。

```go
export vllm_root=`pwd`
```

创建一个名为 `Dockerfile.nginx` 的文件：

```go
FROM nginx:latest
RUN rm /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

构建容器：

```go
docker build . -f Dockerfile.nginx --tag nginx-lb
```

## 创建简单的 Nginx 配置文件

创建一个名为 `nginx_conf/nginx.conf` 的文件。你可以根据需要添加多个服务器，以下示例中默认包含两个。如果需要更多服务器，可以在 `upstream backend` 部分添加更多 `server vllmN:8000 max_fails=3 fail_timeout=10000s;` 条目。

```go
upstream backend {
    least_conn;
    server vllm0:8000 max_fails=3 fail_timeout=10000s;
    server vllm1:8000 max_fails=3 fail_timeout=10000s;
}
server {
    listen 80;
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 构建 vLLM 容器

```go
cd $vllm_root
docker build -f Dockerfile . --tag vllm
```

如果你在代理网络环境下，可以在构建时传递代理参数：

```go
cd $vllm_root
docker build -f Dockerfile . --tag vllm --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy
```

## 创建 Docker 网络

```go
docker network create vllm_nginx
```

## 启动 vLLM 容器

注意：

- 如果你的 HuggingFace 模型缓存目录不同，请修改 `hf_cache_dir` 变量。
- 如果你没有 HuggingFace 缓存，建议先启动 `vllm0`，等待模型下载完成并服务器准备就绪后再启动 `vllm1`，以避免重复下载模型。
- 以下示例假设使用 GPU，如使用 CPU，请移除 `--gpus device=ID`，并添加 `VLLM_CPU_KVCACHE_SPACE` 和 `VLLM_CPU_OMP_THREADS_BIND` 环境变量。
- 若需更改 vLLM 服务器使用的模型，请调整 `meta-llama/Llama-2-7b-chat-hf` 为所需模型名称。

```go
mkdir -p ~/.cache/huggingface/hub/
hf_cache_dir=~/.cache/huggingface/
docker run -itd --ipc host --network vllm_nginx --gpus device=0 --shm-size=10.24gb -v $hf_cache_dir:/root/.cache/huggingface/ -p 8081:8000 --name vllm0 vllm --model meta-llama/Llama-2-7b-chat-hf
docker run -itd --ipc host --network vllm_nginx --gpus device=1 --shm-size=10.24gb -v $hf_cache_dir:/root/.cache/huggingface/ -p 8082:8000 --name vllm1 vllm --model meta-llama/Llama-2-7b-chat-hf
```

> **注意**
> 
> 如果您处于代理环境中，可以通过 `-e http_proxy=$http_proxy -e https_proxy=$https_proxy` 参数将代理设置传递给 `docker run` 命令。

## 启动 Nginx

```go
docker run -itd -p 8000:80 --network vllm_nginx -v ./nginx_conf/:/etc/nginx/conf.d/ --name nginx-lb nginx-lb:latest
```

## 验证 vLLM 服务器是否准备就绪

```go
docker logs vllm0 | grep Uvicorn
docker logs vllm1 | grep Uvicorn
```

如果一切正常，输出应类似于如下所示：

```go
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
