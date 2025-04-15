---
title: 使用 LWS 进行部署
---


LeaderWorkerSet (LWS) 是一个 Kubernetes API，旨在解决 AI/ML 推理工作负载的常见部署模式。一个主要用例是多主机/多节点分布式推理。


vLLM 可以与 [LWS](https://github.com/kubernetes-sigs/lws) 一起部署在 Kubernetes 上，实现分布式模型服务。


有关使用 LWS 在 Kubernetes 上部署 vLLM 的更多详细信息，请参阅[本指南](https://github.com/kubernetes-sigs/lws/tree/main/docs/examples/vllm)。

