---
title: vLLM 基准套件
---


vLLM 包含 2 组基准:

* **性能基准测试**: 在 vLLM 的高频工作负载下（当 vLLM 的拉取请求（简称 PR）被合并时）对其进行性能基准测试。参阅 [vLLM 性能仪表板](https://perf.vllm.ai)了解最新性能结果。

* **Nightly 基准测试**: 当 vLLM 发生重大更新时（例如，升级到新版本），比较 vLLM 与替代方案（tgi、trt-llm 和 lmdeploy）的性能。最新结果可在 [vLLM GitHub README](https://github.com/vllm-project/vllm/blob/main/README.md) 中找到。


## 触发基准测试

性能基准测试和 nightly 基准测试可以通过向 vLLM 提交 PR 来触发，并使用 *perf-benchmarks* 和 *nightly-benchmarks* 标记 PR。


**注意：**

有关基准测试环境、工作负载和指标的详细说明，请参考 [vLLM 性能基准说明](https://github.com/vllm-project/vllm/blob/main/.buildkite/nightly-benchmarks/performance-benchmarks-descriptions.md) 和 [vLLM nightly 基准说明](https://github.com/vllm-project/vllm/blob/main/.buildkite/nightly-benchmarks/nightly-descriptions.md)。
