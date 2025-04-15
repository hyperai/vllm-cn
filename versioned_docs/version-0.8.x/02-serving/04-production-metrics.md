---
title: 生产指标
---


vLLM 公布了许多指标，可用于监控系统运行状况。这些指标通过 vLLM 与 OpenAI 兼容的 API 服务器上的 */metrics* 端点公开。

The following metrics are exposed:

以下是公布的指标：

```python
class Metrics:
    """
    vLLM uses a multiprocessing-based frontend for the OpenAI server.
    This means that we need to run prometheus_client in multiprocessing mode
    See https://prometheus.github.io/client_python/multiprocess/ for more
    details on limitations.
    vLLM 使用基于多进程的前端来运行 OpenAI 服务器。
    也就是说我们需要以多进程模式运行 prometheus_client。
    详情请参阅 https://prometheus.github.io/client_python/multiprocess/ 了解相关限制。
    """
    labelname_finish_reason = "finished_reason"
    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter
    _histogram_cls = prometheus_client.Histogram

    def __init__(self, labelnames: List[str], max_model_len: int):
        # Unregister any existing vLLM collectors (for CI/CD)
        # 注销所有存在的 vLLM 收集器 （对 CI/CD）
        self._unregister_vllm_metrics()

        # System stats
        # 系统统计
        #   Scheduler State
        #   调度统计
        self.gauge_scheduler_running = self._gauge_cls(
            name="vllm:num_requests_running",
            documentation="Number of requests currently running on GPU.",
            labelnames=labelnames,
            multiprocess_mode="sum")
        self.gauge_scheduler_waiting = self._gauge_cls(
            name="vllm:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            labelnames=labelnames,
            multiprocess_mode="sum")
        self.gauge_scheduler_swapped = self._gauge_cls(
            name="vllm:num_requests_swapped",
            documentation="Number of requests swapped to CPU.",
            labelnames=labelnames,
            multiprocess_mode="sum")
        #   KV Cache Usage in %
        #   KV Cache 使用占比（%）
        self.gauge_gpu_cache_usage = self._gauge_cls(
            name="vllm:gpu_cache_usage_perc",
            documentation="GPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames,
            multiprocess_mode="sum")
        self.gauge_cpu_cache_usage = self._gauge_cls(
            name="vllm:cpu_cache_usage_perc",
            documentation="CPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames,
            multiprocess_mode="sum")
        #   Prefix caching block hit rate
        #   前缀缓存块命中率
        self.gauge_cpu_prefix_cache_hit_rate = self._gauge_cls(
            name="vllm:cpu_prefix_cache_hit_rate",
            documentation="CPU prefix cache block hit rate.",
            labelnames=labelnames,
            multiprocess_mode="sum")
        self.gauge_gpu_prefix_cache_hit_rate = self._gauge_cls(
            name="vllm:gpu_prefix_cache_hit_rate",
            documentation="GPU prefix cache block hit rate.",
            labelnames=labelnames,
            multiprocess_mode="sum")

        # Iteration stats
        # 迭代统计
        self.counter_num_preemption = self._counter_cls(
            name="vllm:num_preemptions_total",
            documentation="Cumulative number of preemption from the engine.",
            labelnames=labelnames)
        self.counter_prompt_tokens = self._counter_cls(
            name="vllm:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames)
        self.counter_generation_tokens = self._counter_cls(
            name="vllm:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames)
        self.histogram_time_to_first_token = self._histogram_cls(
            name="vllm:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0
            ])
        self.histogram_time_per_output_token = self._histogram_cls(
            name="vllm:time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5
            ])

        # Request stats
        # 请求统计
        #   Latency
        #   延迟
        self.histogram_e2e_time_request = self._histogram_cls(
            name="vllm:e2e_request_latency_seconds",
            documentation="Histogram of end to end request latency in seconds.",
            labelnames=labelnames,
            buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        #   Metadata
        #   元数据
        self.histogram_num_prompt_tokens_request = self._histogram_cls(
            name="vllm:request_prompt_tokens",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.histogram_num_generation_tokens_request = \
            self._histogram_cls(
                name="vllm:request_generation_tokens",
                documentation="Number of generation tokens processed.",
                labelnames=labelnames,
                buckets=build_1_2_5_buckets(max_model_len),
            )
        self.histogram_best_of_request = self._histogram_cls(
            name="vllm:request_params_best_of",
            documentation="Histogram of the best_of request parameter.",
            labelnames=labelnames,
            buckets=[1, 2, 5, 10, 20],
        )
        self.histogram_n_request = self._histogram_cls(
            name="vllm:request_params_n",
            documentation="Histogram of the n request parameter.",
            labelnames=labelnames,
            buckets=[1, 2, 5, 10, 20],
        )
        self.counter_request_success = self._counter_cls(
            name="vllm:request_success_total",
            documentation="Count of successfully processed requests.",
            labelnames=labelnames + [Metrics.labelname_finish_reason])

        # Speculatie decoding stats
        # 推测解码统计
        self.gauge_spec_decode_draft_acceptance_rate = self._gauge_cls(
            name="vllm:spec_decode_draft_acceptance_rate",
            documentation="Speulative token acceptance rate.",
            labelnames=labelnames,
            multiprocess_mode="sum")
        self.gauge_spec_decode_efficiency = self._gauge_cls(
            name="vllm:spec_decode_efficiency",
            documentation="Speculative decoding system efficiency.",
            labelnames=labelnames,
            multiprocess_mode="sum")
        self.counter_spec_decode_num_accepted_tokens = (self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens_total",
            documentation="Number of accepted tokens.",
            labelnames=labelnames))
        self.counter_spec_decode_num_draft_tokens = self._counter_cls(
            name="vllm:spec_decode_num_draft_tokens_total",
            documentation="Number of draft tokens.",
            labelnames=labelnames)
        self.counter_spec_decode_num_emitted_tokens = (self._counter_cls(
            name="vllm:spec_decode_num_emitted_tokens_total",
            documentation="Number of emitted tokens.",
            labelnames=labelnames))

        # Deprecated in favor of vllm:prompt_tokens_total
        # 已弃用，推荐使用 vllm:prompt_tokens_total
        self.gauge_avg_prompt_throughput = self._gauge_cls(
            name="vllm:avg_prompt_throughput_toks_per_s",
            documentation="Average prefill throughput in tokens/s.",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        # Deprecated in favor of vllm:generation_tokens_total
        # 已弃用，推荐使用 vllm:generation_tokens_total
        self.gauge_avg_generation_throughput = self._gauge_cls(
            name="vllm:avg_generation_throughput_toks_per_s",
            documentation="Average generation throughput in tokens/s.",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
```


