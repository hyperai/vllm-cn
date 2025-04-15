---
title: 分析 vLLM
---


我们支持使用 `torch.profiler` 模块对 vLLM 工作进程进行跟踪。您可以通过将 `VLLM_TORCH_PROFILER_DIR` 环境变量设置为要保存跟踪日志的目录来启用跟踪： `VLLM_TORCH_PROFILER_DIR=/mnt/traces/` 


OpenAI 服务器还需要使用 `VLLM_TORCH_PROFILER_DIR` 环境变量集启动。


当使用 `benchmarks/benchmark_serving.py` 时，您可以通过传递 `--profile` 标志来启用分析。


**警告：**

仅在开发环境中启用分析。


可以使用 [https://ui.perfetto.dev/](https://ui.perfetto.dev/) 来可视化跟踪。


**提示：**

在进行性能分析时，仅通过 vLLM 发送少量请求，因为跟踪数据可能会变得相当庞大。而且，无需对跟踪文件进行解压缩操作，它们能够直接被查看。


**提示：**

如需停止分析器————它会将所有的性能分析追踪文件刷新到目录中。这需要一些时间，例如，对于大约 100 个请求的 Llama 70B 模型的数据，在 H100 上刷新大约需要 10 分钟。在启动服务器之前，请将环境变量 VLLM_RPC_TIMEOUT 设置为一个较大的值，比如 30 分钟，即 `export VLLM_RPC_TIMEOUT=1800000`


## 命令和使用示例：

### 离线推理

参见 [examples/offline_inference_with_profiler.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_with_profiler.py) 获取示例。

### OpenAI 服务器：

```bash
VLLM_TORCH_PROFILER_DIR=/mnt/traces/ python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B 
```
benchmark_serving.py:
```bash
python benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-70B --dataset-name sharegpt --dataset-path sharegpt.json --profile --num-prompts 2 
```


