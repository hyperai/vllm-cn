---
title: 环境
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 使用以下环境变量来配置系统：

> **警告**
> 
> 请注意，`VLLM_PORT` 和 `VLLM_HOST_IP` 仅用于 vLLM **内部使用**，它们并不是 API 服务器的端口和 IP。如果你使用 `--host $VLLM_HOST_IP` 和 `--port $VLLM_PORT` 来启动 API 服务器，它将无法正常工作。

所有 vLLM 使用的环境变量都以 `VLLM_` 作为前缀。**Kubernetes 用户需要特别注意**：请不要将服务命名为 `vllm`，否则 Kubernetes 设置的环境变量可能会与 vLLM 的环境变量发生冲突。因为 [Kubernetes 会为每个服务设置环境变量，并使用大写的服务名称作为前缀](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables)。

```plain


environment_variables: dict[str, Callable[[], Any]] = {



    # ================== 安装时环境变量 ==================




    # vLLM 的目标设备，支持 [cuda（默认）、rocm、neuron、cpu、openvino]
    "VLLM_TARGET_DEVICE":
    lambda: os.getenv("VLLM_TARGET_DEVICE", "cuda"),




    # 并行运行的编译任务的最大数量。默认是 CPU 的数量
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),




    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    # 用于 nvcc 的线程数。默认为 1。如果设置，`MAX_JOBS` 将减少以避免 CPU 过载
    "NVCC_THREADS":
    lambda: os.getenv("NVCC_THREADS", None),




    # 如果设置，vllm 将使用预编译的二进制文件 (*.so)
    "VLLM_USE_PRECOMPILED":
    lambda: bool(os.environ.get("VLLM_USE_PRECOMPILED")) or bool(
        os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION")),




    # 是否在 Python 构建中强制使用 nightly wheel。用于在 Python 构建中测试 nightly wheel
    "VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL":
    lambda: bool(int(os.getenv("VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL", "0"))
                 ),




    # CMake 构建类型
    # 如果未设置，默认为 "Debug" 或 "RelWithDebInfo"
    # 可用选项："Debug"、"Release"、"RelWithDebInfo"
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),




    # 如果设置，vllm 将在安装期间打印详细日志
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),




    # vLLM 配置文件的根目录
    # 默认为 `~/.config/vllm`，除非设置了 `XDG_CONFIG_HOME`
    # 注意，这不仅影响 vllm 在运行时如何查找其配置文件，还影响 vllm 在**安装**期间如何安装其配置文件
    "VLLM_CONFIG_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "vllm"),
        )),




    # ================== 运行时环境变量 ==================




    # vLLM 缓存文件的根目录
    # 默认为 `~/.cache/vllm`，除非设置了 `XDG_CACHE_HOME`
    "VLLM_CACHE_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_CACHE_ROOT",
            os.path.join(get_default_cache_root(), "vllm"),
        )),



    # 在分布式环境中用于确定当前节点的 IP 地址，当节点有多个网络接口时
    # 如果你使用多节点推理，你应该在每个节点上设置不同的值
    'VLLM_HOST_IP':
    lambda: os.getenv('VLLM_HOST_IP', ""),




    # 在分布式环境中用于手动设置通信端口
    # 注意：如果设置了 VLLM_PORT，并且某些代码请求多个端口，VLLM_PORT 将用作第一个端口，其余的将通过递增 VLLM_PORT 值生成
    # '0' 用于使 mypy 满意
    'VLLM_PORT':
    lambda: int(os.getenv('VLLM_PORT', '0'))
    if 'VLLM_PORT' in os.environ else None,



    # 当前端 API 服务器以多进程模式运行时用于 IPC 的路径，以便与后端引擎进程通信
    'VLLM_RPC_BASE_PATH':
    lambda: os.getenv('VLLM_RPC_BASE_PATH', tempfile.gettempdir()),



    # 如果为 true，将从 ModelScope 加载模型，而不是 Hugging Face Hub
    # 注意，值为 true 或 false，而不是数字
    "VLLM_USE_MODELSCOPE":
    lambda: os.environ.get("VLLM_USE_MODELSCOPE", "False").lower() == "true",



    # 当环形缓冲区满时记录警告消息的时间间隔（秒）
    "VLLM_RINGBUFFER_WARNING_INTERVAL":
    lambda: int(os.environ.get("VLLM_RINGBUFFER_WARNING_INTERVAL", "60")),




    # cudatoolkit 主目录的路径，其下应有 bin、include 和 lib 目录
    "CUDA_HOME":
    lambda: os.environ.get("CUDA_HOME", None),




    # NCCL 库文件的路径。这是必需的，因为 PyTorch 自带的 nccl>=2.19 包含一个 bug：https://github.com/NVIDIA/nccl/issues/1234
    "VLLM_NCCL_SO_PATH":
    lambda: os.environ.get("VLLM_NCCL_SO_PATH", None),




    # 当 `VLLM_NCCL_SO_PATH` 未设置时，vllm 将尝试在 `LD_LIBRARY_PATH` 指定的位置查找 nccl 库文件
    "LD_LIBRARY_PATH":
    lambda: os.environ.get("LD_LIBRARY_PATH", None),



    # 控制 vllm 是否使用 triton flash attention 的标志
    "VLLM_USE_TRITON_FLASH_ATTN":
    lambda: (os.environ.get("VLLM_USE_TRITON_FLASH_ATTN", "True").lower() in
             ("true", "1")),



    # 强制 vllm 使用特定的 flash-attention 版本（2 或 3），仅在使用 flash-attention 后端时有效
    "VLLM_FLASH_ATTN_VERSION":
    lambda: maybe_convert_int(os.environ.get("VLLM_FLASH_ATTN_VERSION", None)),



    # 内部标志，用于启用 Dynamo fullgraph 捕获
    "VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE":
    lambda: bool(
        os.environ.get("VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1") != "0"),



    # 分布式设置中进程的本地 rank，用于确定 GPU 设备 ID
    "LOCAL_RANK":
    lambda: int(os.environ.get("LOCAL_RANK", "0")),




    # 用于控制分布式设置中的可见设备
    "CUDA_VISIBLE_DEVICES":
    lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),



    # 引擎中每次迭代的超时时间
    "VLLM_ENGINE_ITERATION_TIMEOUT_S":
    lambda: int(os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "60")),



    # vLLM API 服务器的 API 密钥
    "VLLM_API_KEY":
    lambda: os.environ.get("VLLM_API_KEY", None),



    # S3 访问信息，用于 tensorizer 从 S3 加载模型
    "S3_ACCESS_KEY_ID":
    lambda: os.environ.get("S3_ACCESS_KEY_ID", None),
    "S3_SECRET_ACCESS_KEY":
    lambda: os.environ.get("S3_SECRET_ACCESS_KEY", None),
    "S3_ENDPOINT_URL":
    lambda: os.environ.get("S3_ENDPOINT_URL", None),




    # 使用统计收集
    "VLLM_USAGE_STATS_SERVER":
    lambda: os.environ.get("VLLM_USAGE_STATS_SERVER", "https://stats.vllm.ai"),
    "VLLM_NO_USAGE_STATS":
    lambda: os.environ.get("VLLM_NO_USAGE_STATS", "0") == "1",
    "VLLM_DO_NOT_TRACK":
    lambda: (os.environ.get("VLLM_DO_NOT_TRACK", None) or os.environ.get(
        "DO_NOT_TRACK", None) or "0") == "1",
    "VLLM_USAGE_SOURCE":
    lambda: os.environ.get("VLLM_USAGE_SOURCE", "production"),




    # 日志配置
    # 如果设置为 0，vllm 将不配置日志
    # 如果设置为 1，vllm 将使用默认配置或 VLLM_LOGGING_CONFIG_PATH 指定的配置文件配置日志
    "VLLM_CONFIGURE_LOGGING":
    lambda: int(os.getenv("VLLM_CONFIGURE_LOGGING", "1")),
    "VLLM_LOGGING_CONFIG_PATH":
    lambda: os.getenv("VLLM_LOGGING_CONFIG_PATH"),




    # 用于配置默认日志级别
    "VLLM_LOGGING_LEVEL":
    lambda: os.getenv("VLLM_LOGGING_LEVEL", "INFO"),




    # 如果设置，VLLM_LOGGING_PREFIX 将被添加到所有日志消息的前面
    "VLLM_LOGGING_PREFIX":
    lambda: os.getenv("VLLM_LOGGING_PREFIX", ""),




    # 如果设置，vllm 将在一个线程池中调用 logits processors，线程池的大小为此值
    # 当使用自定义 logits processors 时，这很有用，这些 processors 可能（a）启动额外的 CUDA 内核，或（b）在不持有 Python GIL 的情况下执行大量 CPU 密集型工作，或两者兼有
    "VLLM_LOGITS_PROCESSOR_THREADS":
    lambda: int(os.getenv("VLLM_LOGITS_PROCESSOR_THREADS", "0"))
    if "VLLM_LOGITS_PROCESSOR_THREADS" in os.environ else None,




    # 跟踪函数调用
    # 如果设置为 1，vllm 将跟踪函数调用
    # 对调试很有用
    "VLLM_TRACE_FUNCTION":
    lambda: int(os.getenv("VLLM_TRACE_FUNCTION", "0")),




    # 注意力计算的后端
    # 可用选项：
    # - "TORCH_SDPA"：使用 torch.nn.MultiheadAttention
    # - "FLASH_ATTN"：使用 FlashAttention
    # - "XFORMERS"：使用 XFormers
    # - "ROCM_FLASH"：使用 ROCmFlashAttention
    # - "FLASHINFER"：使用 flashinfer
    # - "FLASHMLA"：使用 FlashMLA
    "VLLM_ATTENTION_BACKEND":
    lambda: os.getenv("VLLM_ATTENTION_BACKEND", None),



    # 如果设置，vllm 将使用 flashinfer 采样器
    "VLLM_USE_FLASHINFER_SAMPLER":
    lambda: bool(int(os.environ["VLLM_USE_FLASHINFER_SAMPLER"]))
    if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ else None,



    # 如果设置，vllm 将强制 flashinfer 使用张量核心；否则将基于模型架构使用启发式方法
    "VLLM_FLASHINFER_FORCE_TENSOR_CORES":
    lambda: bool(int(os.getenv("VLLM_FLASHINFER_FORCE_TENSOR_CORES", "0"))),




    # 流水线阶段分区策略
    "VLLM_PP_LAYER_PARTITION":
    lambda: os.getenv("VLLM_PP_LAYER_PARTITION", None),




    # （仅 CPU 后端）CPU 键值缓存空间
    # 默认为 4GB
    "VLLM_CPU_KVCACHE_SPACE":
    lambda: int(os.getenv("VLLM_CPU_KVCACHE_SPACE", "0")),




    # （仅 CPU 后端）OpenMP 线程绑定的 CPU 核心 ID，例如 "0-31"、"0,1,2"、"0-31,33"。不同 rank 的 CPU 核心用 '|' 分隔
    "VLLM_CPU_OMP_THREADS_BIND":
    lambda: os.getenv("VLLM_CPU_OMP_THREADS_BIND", "all"),




    # （仅 CPU 后端）是否对 MoE 层使用预打包。这将传递给 ipex.llm.modules.GatedMLPMOE。在不支持的 CPU 上，你可能需要将其设置为 "0"（False）。
    "VLLM_CPU_MOE_PREPACK":
    lambda: bool(int(os.getenv("VLLM_CPU_MOE_PREPACK", "1"))),




    # OpenVINO 设备选择
    # 默认为 CPU
    "VLLM_OPENVINO_DEVICE":
    lambda: os.getenv("VLLM_OPENVINO_DEVICE", "CPU").upper(),




    # OpenVINO 键值缓存空间
    # 默认为 4GB
    "VLLM_OPENVINO_KVCACHE_SPACE":
    lambda: int(os.getenv("VLLM_OPENVINO_KVCACHE_SPACE", "0")),




    # OpenVINO KV 缓存精度
    # 如果平台原生支持，则默认为 bf16，否则为 f16
    # 要启用 KV 缓存压缩，请显式指定 u8
    "VLLM_OPENVINO_CPU_KV_CACHE_PRECISION":
    lambda: os.getenv("VLLM_OPENVINO_CPU_KV_CACHE_PRECISION", None),




    # 在通过 HF Optimum 导出模型时启用权重压缩
    # 默认为 False
    "VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS":
    lambda:
    (os.environ.get("VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS", "0").lower() in
     ("on", "true", "1")),

    # 如果设置了环境变量，则所有工作者将作为与引擎分离的进程执行，并且我们使用相同的机制在所有工作者上触发执行。
    # 运行 vLLM 时设置 VLLM_USE_RAY_SPMD_WORKER=1 以启用此功能。
    "VLLM_USE_RAY_SPMD_WORKER":
    lambda: bool(int(os.getenv("VLLM_USE_RAY_SPMD_WORKER", "0"))),



    # 如果设置了环境变量，它将使用 Ray 的 Compiled Graph（以前称为 ADAG）API，该 API 优化了控制平面开销。
    # 运行 vLLM 时设置 VLLM_USE_RAY_COMPILED_DAG=1 以启用此功能。
    "VLLM_USE_RAY_COMPILED_DAG":
    lambda: bool(int(os.getenv("VLLM_USE_RAY_COMPILED_DAG", "0"))),



    # 如果设置了环境变量，它将在 Ray 的 Compiled Graph 中使用 NCCL 进行通信。如果未设置 VLLM_USE_RAY_COMPILED_DAG，则忽略此标志。
    "VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL":
    lambda: bool(int(os.getenv("VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL", "1"))
                 ),




    # 如果设置了环境变量，它将在 Ray 的 Compiled Graph 中启用 GPU 通信重叠（实验性功能）。如果未设置 VLLM_USE_RAY_COMPILED_DAG，则忽略此标志。
    "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM":
    lambda: bool(int(os.getenv("VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM", "0"))
                 ),




    # 为工作者使用专用的多进程上下文。
    # spawn 和 fork 都有效
    "VLLM_WORKER_MULTIPROC_METHOD":
    lambda: os.getenv("VLLM_WORKER_MULTIPROC_METHOD", "fork"),



    # 存储下载资源的缓存路径
    "VLLM_ASSETS_CACHE":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_ASSETS_CACHE",
            os.path.join(get_default_cache_root(), "vllm", "assets"),
        )),




    # 为多模态模型服务时获取图像的超时时间
    # 默认为 5 秒
    "VLLM_IMAGE_FETCH_TIMEOUT":
    lambda: int(os.getenv("VLLM_IMAGE_FETCH_TIMEOUT", "5")),




    # 为多模态模型服务时获取视频的超时时间
    # 默认为 30 秒
    "VLLM_VIDEO_FETCH_TIMEOUT":
    lambda: int(os.getenv("VLLM_VIDEO_FETCH_TIMEOUT", "30")),



    # 为多模态模型服务时获取音频的超时时间
    # 默认为 10 秒
    "VLLM_AUDIO_FETCH_TIMEOUT":
    lambda: int(os.getenv("VLLM_AUDIO_FETCH_TIMEOUT", "10")),




    # 多模态输入缓存的缓存大小（以 GiB 为单位）
    # 默认为 8 GiB
    "VLLM_MM_INPUT_CACHE_GIB":
    lambda: int(os.getenv("VLLM_MM_INPUT_CACHE_GIB", "8")),




    # XLA 持久缓存目录的路径。
    # 仅用于 XLA 设备（如 TPU）。
    "VLLM_XLA_CACHE_PATH":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_XLA_CACHE_PATH",
            os.path.join(get_default_cache_root(), "vllm", "xla_cache"),
        )),
    "VLLM_FUSED_MOE_CHUNK_SIZE":
    lambda: int(os.getenv("VLLM_FUSED_MOE_CHUNK_SIZE", "32768")),




    # 如果设置，vllm 将跳过弃用警告。
    "VLLM_NO_DEPRECATION_WARNING":
    lambda: bool(int(os.getenv("VLLM_NO_DEPRECATION_WARNING", "0"))),




    # 如果设置，即使底层的 AsyncLLMEngine 出错并停止服务请求，OpenAI API 服务器也将保持活动状态
    "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH":
    lambda: bool(os.getenv("VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", 0)),




    # 如果设置了环境变量 VLLM_ALLOW_LONG_MAX_MODEL_LEN，则允许用户指定大于从模型 config.json 派生的最大长度的最大序列长度。
    # 要启用此功能，请设置 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1。
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN":
    lambda:
    (os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "0").strip().lower() in
     ("1", "true")),




    # 如果设置，无论硬件是否支持 FP8 计算，都将强制使用 FP8 Marlin 进行 FP8 量化。
    "VLLM_TEST_FORCE_FP8_MARLIN":
    lambda:
    (os.environ.get("VLLM_TEST_FORCE_FP8_MARLIN", "0").strip().lower() in
     ("1", "true")),
    "VLLM_TEST_FORCE_LOAD_FORMAT":
    lambda: os.getenv("VLLM_TEST_FORCE_LOAD_FORMAT", "dummy"),




    # zmq 客户端等待后端服务器响应简单数据操作的时间（以毫秒为单位）
    "VLLM_RPC_TIMEOUT":
    lambda: int(os.getenv("VLLM_RPC_TIMEOUT", "10000")),




    # 要加载的插件名称列表，以逗号分隔。
    # 如果未设置，则表示将加载所有插件。
    # 如果设置为空字符串，则不加载任何插件。
    "VLLM_PLUGINS":
    lambda: None if "VLLM_PLUGINS" not in os.environ else os.environ[
        "VLLM_PLUGINS"].split(","),





    # 如果设置，则启用 torch profiler。torch profiler 跟踪保存的目录路径。注意，它必须是绝对路径。
    "VLLM_TORCH_PROFILER_DIR":
    lambda: (None if os.getenv("VLLM_TORCH_PROFILER_DIR", None) is None else os
             .path.expanduser(os.getenv("VLLM_TORCH_PROFILER_DIR", "."))),



    # 如果设置，vLLM 将使用 Triton 实现的 AWQ。
    "VLLM_USE_TRITON_AWQ":
    lambda: bool(int(os.getenv("VLLM_USE_TRITON_AWQ", "0"))),



    # 如果设置，允许在运行时加载或卸载 lora 适配器，
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING":
    lambda:
    (os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "0").strip().lower() in
     ("1", "true")),




    # 默认情况下，vLLM 会自行检查点对点能力，以防驱动程序损坏。有关详细信息，请参阅 https://github.com/vllm-project/vllm/blob/a9b15c606fea67a072416ea0ea115261a2756058/vllm/distributed/device_communicators/custom_all_reduce_utils.py#L101-L108。
    # 如果此环境变量设置为 1，vLLM 将跳过点对点检查，并信任驱动程序的点对点能力报告。
    "VLLM_SKIP_P2P_CHECK":
    lambda: os.getenv("VLLM_SKIP_P2P_CHECK", "0") == "1",




    # 应禁用的量化内核列表，用于测试和性能比较。目前仅影响 MPLinearKernel 选择
    # （内核：MacheteLinearKernel、MarlinLinearKernel、ExllamaLinearKernel）
    "VLLM_DISABLED_KERNELS":
    lambda: [] if "VLLM_DISABLED_KERNELS" not in os.environ else os.environ[
        "VLLM_DISABLED_KERNELS"].split(","),



    # 如果设置，则使用 V1 代码路径。
    "VLLM_USE_V1":
    lambda: bool(int(os.getenv("VLLM_USE_V1", "1"))),




    # 为 ROCm 将 fp8 权重填充到 256 字节
    "VLLM_ROCM_FP8_PADDING":
    lambda: bool(int(os.getenv("VLLM_ROCM_FP8_PADDING", "1"))),
    # 用于 FP8 KV 缓存的动态键缩放因子计算的除数
    "K_SCALE_CONSTANT":
    lambda: int(os.getenv("K_SCALE_CONSTANT", "200")),




    # 用于 FP8 KV 缓存的动态值缩放因子计算的除数
    "V_SCALE_CONSTANT":
    lambda: int(os.getenv("V_SCALE_CONSTANT", "100")),
    # 如果设置，则在 V1 代码路径中启用 LLM 的多进程处理。
    "VLLM_ENABLE_V1_MULTIPROCESSING":
    lambda: bool(int(os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1"))),
    "VLLM_LOG_BATCHSIZE_INTERVAL":
    lambda: float(os.getenv("VLLM_LOG_BATCHSIZE_INTERVAL", "-1")),
    "VLLM_DISABLE_COMPILE_CACHE":
    lambda: bool(int(os.getenv("VLLM_DISABLE_COMPILE_CACHE", "0"))),




    # 如果设置，vllm 将以开发模式运行，这将启用一些用于开发和调试的额外端点，例如 `/reset_prefix_cache`
    "VLLM_SERVER_DEV_MODE":
    lambda: bool(int(os.getenv("VLLM_SERVER_DEV_MODE", "0"))),




    # 控制在 V1 AsyncLLM 接口中处理每个 token 输出时，单个 asyncio 任务中处理的最大请求数。适用于处理高并发的流式请求。
    # 将此值设置得太高可能会导致消息间延迟的更大差异。设置得太低可能会对 TTFT 和整体吞吐量产生负面影响。
    "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE":
    lambda: int(os.getenv("VLLM_V1_OUTPUT_PROC_CHUNK_SIZE", "128")),
    # If set, vLLM will disable the MLA attention optimizations.
    # 如果设置，vLLM 将禁用 MLA 注意力优化。
    "VLLM_MLA_DISABLE":
    lambda: bool(int(os.getenv("VLLM_MLA_DISABLE", "0"))),



    # 如果设置，vLLM 将使用 Triton 实现的 moe_align_block_size，即 fused_moe.py 中的 moe_align_block_size_triton。
    "VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON":
    lambda: bool(int(os.getenv("VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON", "0"))
                 ),




    # Ray 中每个工作者的 GPU 数量，如果设置为分数，则允许 Ray 在单个 GPU 上调度多个 actor，以便用户可以将其他 actor 与 vLLM 放在同一 GPU 上。
    "VLLM_RAY_PER_WORKER_GPUS":
    lambda: float(os.getenv("VLLM_RAY_PER_WORKER_GPUS", "1.0")),



    # Ray 的 bundle 索引，如果设置，它可以精确控制每个工作者使用的 Ray bundle 的索引。
    # 格式：逗号分隔的整数列表，例如 "0,1,2,3"
    "VLLM_RAY_BUNDLE_INDICES":
    lambda: os.getenv("VLLM_RAY_BUNDLE_INDICES", ""),



    # 在某些系统中，find_loaded_library() 可能无法工作。因此，我们允许用户通过环境变量 VLLM_CUDART_SO_PATH 指定路径。
    "VLLM_CUDART_SO_PATH":
    lambda: os.getenv("VLLM_CUDART_SO_PATH", None),



    # 连续缓存获取，以避免在 Gaudi3 上使用昂贵的 gather 操作。这仅适用于 HPU 连续缓存。如果设置为 true，则将使用连续缓存获取。
    "VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH":
    lambda: os.environ.get("VLLM_CONTIGUOUS_PA", "true").lower() in
    ("1", "true"),


    # 数据并行设置中进程的 rank
    "VLLM_DP_RANK":
    lambda: int(os.getenv("VLLM_DP_RANK", "0")),



    # 数据并行设置中的 world size
    "VLLM_DP_SIZE":
    lambda: int(os.getenv("VLLM_DP_SIZE", "1")),




    # 数据并行设置中主节点的 IP 地址
    "VLLM_DP_MASTER_IP":
    lambda: os.getenv("VLLM_DP_MASTER_IP", "127.0.0.1"),




    # 数据并行设置中主节点的端口
    "VLLM_DP_MASTER_PORT":
    lambda: int(os.getenv("VLLM_DP_MASTER_PORT", "0")),




    # 是否在 CI 中通过 RunAI Streamer 使用 S3 路径加载模型
    "VLLM_CI_USE_S3":
    lambda: os.environ.get("VLLM_CI_USE_S3", "0") == "1",



    # 是否在 gptq/awq marlin 内核中使用 atomicAdd reduce。
    "VLLM_MARLIN_USE_ATOMIC_ADD":
    lambda: os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD", "0") == "1",




    # 是否为 V0 启用 outlines 缓存
    # 此缓存是无限制的并且存储在磁盘上，因此在可能存在恶意用户的环境中不安全使用。
    "VLLM_V0_USE_OUTLINES_CACHE":
    lambda: os.environ.get("VLLM_V0_USE_OUTLINES_CACHE", "0") == "1",
}




```
