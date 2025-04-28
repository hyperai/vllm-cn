---
title: 生产指标
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 公布了许多可用于监控系统运行状况的指标。这些指标通过 vLLM OpenAI 兼容 API 服务器上的 `/metrics` 端点公开。

您可以使用　Python　或　[Docker](https://docs.vllm.ai/en/latest/deployment/docker.html#deployment-docker)　启动服务：

```go
vllm serve unsloth/Llama-3.2-1B-Instruct
```

然后查询终端节点以从服务器获取最新指标：

```plain
$ curl http://0.0.0.0:8000/metrics

# HELP vllm:iteration_tokens_total 每个 engine_step 的 token 数量的直方图。
# TYPE vllm:iteration_tokens_total histogram


vllm:iteration_tokens_total_sum{model_name="unsloth/Llama-3.2-1B-Instruct"} 0.0
vllm:iteration_tokens_total_bucket{le="1.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="8.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="16.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="32.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="64.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="128.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="256.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="512.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
...
```

以下指标已公布：

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
    labelname_waiting_lora_adapters = "waiting_lora_adapters"
    labelname_running_lora_adapters = "running_lora_adapters"
    labelname_max_lora = "max_lora"
    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter
    _histogram_cls = prometheus_client.Histogram


    def __init__(self, labelnames: List[str], vllm_config: VllmConfig):
        # 注销所有存在的 vLLM 收集器 （对 CI/CD）
        self._unregister_vllm_metrics()


        max_model_len = vllm_config.model_config.max_model_len


        # 系统统计
        # 调度统计
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
        self.gauge_lora_info = self._gauge_cls(
            name="vllm:lora_requests_info",
            documentation="Running stats on lora requests.",
            labelnames=[
                self.labelname_running_lora_adapters,
                self.labelname_max_lora,
                self.labelname_waiting_lora_adapters,
            ],
            multiprocess_mode="livemostrecent",
        )


        #　废弃于　0.8　－　v1 中没有使用 KV 缓存卸载　
        # TODO: 在 0.9 里如果 show_hidden_metrics=True 则只能开启
        self.gauge_scheduler_swapped = self._gauge_cls(
            name="vllm:num_requests_swapped",
            documentation=(
                "Number of requests swapped to CPU. "
                "DEPRECATED: KV cache offloading is not used in V1"),
            labelnames=labelnames,
            multiprocess_mode="sum")


        #   KV Cache 使用占比（%）
        self.gauge_gpu_cache_usage = self._gauge_cls(
            name="vllm:gpu_cache_usage_perc",
            documentation="GPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames,
            multiprocess_mode="sum")


        #　废弃于　0.8　－　v1 中没有使用 KV 缓存卸载　
        # TODO: 在 0.9 里如果 show_hidden_metrics=True 则只能开启
        self.gauge_cpu_cache_usage = self._gauge_cls(
            name="vllm:cpu_cache_usage_perc",
            documentation=(
                "CPU KV-cache usage. 1 means 100 percent usage. "
                "DEPRECATED: KV cache offloading is not used in V1"),
            labelnames=labelnames,
            multiprocess_mode="sum")


        #　废弃于　0.8　－　v1 中没有使用 KV 缓存卸载　
        # TODO: 在 0.9 里如果 show_hidden_metrics=True 则只能开启
        self.gauge_cpu_prefix_cache_hit_rate = self._gauge_cls(
            name="vllm:cpu_prefix_cache_hit_rate",
            documentation=(
                "CPU prefix cache block hit rate. "
                "DEPRECATED: KV cache offloading is not used in V1"),
            labelnames=labelnames,
            multiprocess_mode="sum")


        #　废弃于　0.8　－　v1 中没有使用 KV 缓存卸载　
        # TODO: 在 0.9 里如果 show_hidden_metrics=True 则只能开启
        self.gauge_gpu_prefix_cache_hit_rate = self._gauge_cls(
            name="vllm:gpu_prefix_cache_hit_rate",
            documentation=("GPU prefix cache block hit rate. "
                           "DEPRECATED: use vllm:gpu_prefix_cache_queries and "
                           "vllm:gpu_prefix_cache_queries in V1"),
            labelnames=labelnames,
            multiprocess_mode="sum")


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
        buckets = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8096]
        if not vllm_config.model_config.enforce_eager:
            buckets = vllm_config.compilation_config.\
                cudagraph_capture_sizes.copy()
            buckets.sort()
        self.histogram_iteration_tokens = self._histogram_cls(
            name="vllm:iteration_tokens_total",
            documentation="Histogram of number of tokens per engine_step.",
            labelnames=labelnames,
            buckets=buckets)
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


        # 请求统计
        #   延迟
        request_latency_buckets = [
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
            40.0, 50.0, 60.0
        ]
        self.histogram_e2e_time_request = self._histogram_cls(
            name="vllm:e2e_request_latency_seconds",
            documentation="Histogram of end to end request latency in seconds.",
            labelnames=labelnames,
            buckets=request_latency_buckets)
        self.histogram_queue_time_request = self._histogram_cls(
            name="vllm:request_queue_time_seconds",
            documentation=
            "Histogram of time spent in WAITING phase for request.",
            labelnames=labelnames,
            buckets=request_latency_buckets)
        self.histogram_inference_time_request = self._histogram_cls(
            name="vllm:request_inference_time_seconds",
            documentation=
            "Histogram of time spent in RUNNING phase for request.",
            labelnames=labelnames,
            buckets=request_latency_buckets)
        self.histogram_prefill_time_request = self._histogram_cls(
            name="vllm:request_prefill_time_seconds",
            documentation=
            "Histogram of time spent in PREFILL phase for request.",
            labelnames=labelnames,
            buckets=request_latency_buckets)
        self.histogram_decode_time_request = self._histogram_cls(
            name="vllm:request_decode_time_seconds",
            documentation=
            "Histogram of time spent in DECODE phase for request.",
            labelnames=labelnames,
            buckets=request_latency_buckets)

        #　废弃于　0.8　－　v1 中没有使用 KV 缓存卸载　
        # TODO: 在 0.9 里如果 show_hidden_metrics=True 则只能开启
        self.histogram_time_in_queue_request = self._histogram_cls(
            name="vllm:time_in_queue_requests",
            documentation=(
                "Histogram of time the request spent in the queue in seconds. "
                "DEPRECATED: use vllm:request_queue_time_seconds instead."),
            labelnames=labelnames,
            buckets=request_latency_buckets)


        #　废弃于　0.8　－　v1 中没有使用 KV 缓存卸载　
        # TODO: 在 0.9 里如果 show_hidden_metrics=True 则只能开启
        self.histogram_model_forward_time_request = self._histogram_cls(
            name="vllm:model_forward_time_milliseconds",
            documentation=(
                "Histogram of time spent in the model forward pass in ms. "
                "DEPRECATED: use prefill/decode/inference time metrics instead."
            ),
            labelnames=labelnames,
            buckets=build_1_2_3_5_8_buckets(3000))
        self.histogram_model_execute_time_request = self._histogram_cls(
            name="vllm:model_execute_time_milliseconds",
            documentation=(
                "Histogram of time spent in the model execute function in ms."
                "DEPRECATED: use prefill/decode/inference time metrics instead."
            ),
            labelnames=labelnames,
            buckets=build_1_2_3_5_8_buckets(3000))


        #   元数据
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
        self.histogram_max_num_generation_tokens_request = self._histogram_cls(
            name="vllm:request_max_num_generation_tokens",
            documentation=
            "Histogram of maximum number of requested generation tokens.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len))
        self.histogram_n_request = self._histogram_cls(
            name="vllm:request_params_n",
            documentation="Histogram of the n request parameter.",
            labelnames=labelnames,
            buckets=[1, 2, 5, 10, 20],
        )
        self.histogram_max_tokens_request = self._histogram_cls(
            name="vllm:request_params_max_tokens",
            documentation="Histogram of the max_tokens request parameter.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.counter_request_success = self._counter_cls(
            name="vllm:request_success_total",
            documentation="Count of successfully processed requests.",
            labelnames=labelnames + [Metrics.labelname_finish_reason])


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


```

以下指标已弃用，并将在未来版本中删除：

- `vllm:num_requests_swapped`, `vllm:cpu_cache_usage_perc`, 和 `vllm:cpu_prefix_cache_hit_rate` 因为 V1 中没有使用 KV 缓存卸载。
- `vllm:gpu_prefix_cache_hit_rate` 在 V1 中替换为 queries+hits 计数器。
- `vllm:time_in_queue_requests` 与 `vllm:request_queue_time_seconds`重复
- `vllm:model_forward_time_milliseconds` 和 `vllm:model_execute_time_milliseconds` ：因为应该改用预填充/解码/推理时间指标。

注意：当指标在版本 `X.Y` 中被弃用时，它们将在版本 `X.Y+1` 中隐藏 但可以使用 `--show-hidden-metrics-for-version=X.Y` 转义舱口重新启用，然后在版本 `X.Y+2` 中删除。
