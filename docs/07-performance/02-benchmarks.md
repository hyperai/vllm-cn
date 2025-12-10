---
title: 基准套件
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 包含两组基准：

- **性能基准**
- **夜间基准测试**

## 性能基准测试

性能基准测试用于开发过程中，以确认新更改是否在各种工作负载下提升了性能。每次提交带有 `perf-benchmarks` 和 `ready` 标签的代码时，以及当 PR 合并到 vLLM 时，都会触发这些测试。

最新的性能结果托管在公开的 [vLLM 性能仪表板](https://perf.vllm.ai/)上。

有关性能基准测试及其参数的更多信息，请参阅[此处](https://github.com/vllm-project/vllm/blob/main/.buildkite/nightly-benchmarks/performance-benchmarks-descriptions.md)。

## 夜间基准测试

这些测试在 vLLM 有重大更新（例如升级到新版本）时，将 vLLM 的性能与其他替代方案（如 `tgi`、`trt-llm` 和 `lmdeploy`）进行比较。它们主要用于帮助用户评估何时选择 vLLM 而非其他选项，并在每次提交带有 `perf-benchmarks` 和 `nightly-benchmarks` 标签的代码时触发。

最新的夜间基准测试结果会在主要版本发布的博客文章中分享，例如 [vLLM v0.6.0](https://blog.vllm.ai/2024/09/05/perf-update.html)。

有关夜间基准测试及其参数的更多信息，请参阅[此处](https://github.com/vllm-project/vllm/blob/main/.buildkite/nightly-benchmarks/nightly-descriptions.md)。
