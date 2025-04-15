---
title: 环境变量
---

vLLM 使用以下环境变量来配置系统：

**警告**

请注意， `VLLM_PORT` 和 `VLLM_HOST_IP` 是用于设置 vLLM **内部使用**的端口和 IP，它们并非 API 服务器的端口和 IP。如果您使用 `--host $VLLM_HOST_IP` 和 `--port $VLLM_PORT` 来启动 API 服务器，将无法正常运行。


vLLM 所使用的所有环境变量都以 `VLLM_` 为前缀。 **Kubernetes 用户****需要****特别注意**：请不要将服务命名为 `vllm`，否则 Kubernetes 设置的环境变量可能会与 vLLM 的环境变量冲突，因为 [Kubernetes 会以大写的服务名称作为前缀为每个服务设置环境变量](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables)。 

```python
environment_variables: Dict[str, Callable[[], Any]] = {


    # ================== Installation Time Env Vars ==================
    # ================== 安装时的环境变量 ==================


    # Target device of vLLM, supporting [cuda (by default),
    # rocm, neuron, cpu, openvino]
    # vLLM的目标设备，支持[cuda（默认）、rocm、neuron、cpu、openvino]
    "VLLM_TARGET_DEVICE":
    lambda: os.getenv("VLLM_TARGET_DEVICE", "cuda"),


    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    # 并行运行的最大编译任务数。
    # 默认是CPU的数量
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),


    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    # 用于nvcc的线程数
    # 默认是1。
    # 如果设置，`MAX_JOBS` 将减少以避免 CPU 过载。
    "NVCC_THREADS":
    lambda: os.getenv("NVCC_THREADS", None),


    # If set, vllm will use precompiled binaries (*.so)
    # 如果设置，vllm将使用预编译的二进制文件（*.so）
    "VLLM_USE_PRECOMPILED":
    lambda: bool(os.environ.get("VLLM_USE_PRECOMPILED")),


    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    # CMake 的构建类型
    # 如果未设置，默认是 “Debug” 或 “RelWithDebInfo”
    # 可用选项：“Debug”、“Release”、“RelWithDebInfo”
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),


    # If set, vllm will print verbose logs during installation
    # 如果设置，vllm 将在安装过程中打印详细日志
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),


    # Root directory for VLLM configuration files
    # Defaults to `~/.config/vllm` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how vllm finds its configuration files
    # during runtime, but also affects how vllm installs its configuration
    # files during **installation**.
    # VLLM 配置文件的根目录
    # 默认为 `~/.config/vllm`，除非设置了 `XDG_CONFIG_HOME`
    # 注意，这不仅影响 vllm 在运行时如何找到其配置文件，还影响 vllm 在**安装**期间如何安装其配置文件。
    "VLLM_CONFIG_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "vllm"),
        )),


    # ================== Runtime Env Vars ==================
     # ================== 运行时的环境变量 ==================


    # Root directory for VLLM cache files
    # Defaults to `~/.cache/vllm` unless `XDG_CACHE_HOME` is set
    # VLLM 缓存文件的根目录
    # 默认为 `~/.cache/vllm`，除非设置了 `XDG_CACHE_HOME`
    "VLLM_CACHE_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_CACHE_ROOT",
            os.path.join(get_default_cache_root(), "vllm"),
        )),


    # used in distributed environment to determine the ip address
    # of the current node, when the node has multiple network interfaces.
    # If you are using multi-node inference, you should set this differently
    # on each node.
    # 在分布式环境中，用于确定当前节点的 IP 地址，当节点有多个网络接口时。
    # 如果你使用多节点推理，你应该在每个节点上分别设置它。
    'VLLM_HOST_IP':
    lambda: os.getenv('VLLM_HOST_IP', "") or os.getenv("HOST_IP", ""),


    # used in distributed environment to manually set the communication port
    # Note: if VLLM_PORT is set, and some code asks for multiple ports, the
    # VLLM_PORT will be used as the first port, and the rest will be generated
    # by incrementing the VLLM_PORT value.
    # '0' is used to make mypy happy
    # 在分布式环境中，用于手动设置通信端口
    # 注意：如果设置了 VLLM_PORT，并且某些代码需要多个端口，
    # VLLM_PORT 将用作第一个端口，其他端口通过递增 VLLM_PORT 值生成。
    # '0' 是为了通过 mypy 检查
    'VLLM_PORT':
    lambda: int(os.getenv('VLLM_PORT', '0'))
    if 'VLLM_PORT' in os.environ else None,


    # path used for ipc when the frontend api server is running in
    # multi-processing mode to communicate with the backend engine process.
    # 当前端 API 服务器在多进程模式下运行时，用于与后端引擎进程通信的 ipc 路径
    'VLLM_RPC_BASE_PATH':
    lambda: os.getenv('VLLM_RPC_BASE_PATH', tempfile.gettempdir()),


    # If true, will load models from ModelScope instead of Hugging Face Hub.
    # note that the value is true or false, not numbers
    # 如果为 true，将从 ModelScope 加载模型，而不是从 Hugging Face Hub 加载。
    # 注意，值为 true 或 false，而不是数字
    "VLLM_USE_MODELSCOPE":
    lambda: os.environ.get("VLLM_USE_MODELSCOPE", "False").lower() == "true",


    # Instance id represents an instance of the VLLM. All processes in the same
    # instance should have the same instance id.
    # 实例 ID 表示 VLLM 的一个实例。所有在同一实例中的进程应该具有相同的实例 ID
    "VLLM_INSTANCE_ID":
    lambda: os.environ.get("VLLM_INSTANCE_ID", None),


    # Interval in seconds to log a warning message when the ring buffer is full
    # 当环形缓冲区满时，记录警告消息的间隔（秒）
    "VLLM_RINGBUFFER_WARNING_INTERVAL":
    lambda: int(os.environ.get("VLLM_RINGBUFFER_WARNING_INTERVAL", "60")),


    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    # cudatoolkit 主目录的路径，其中应包含 bin、include 和 lib 目录。
    "CUDA_HOME":
    lambda: os.environ.get("CUDA_HOME", None),


    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    # NCCL 库文件的路径。由于 nccl>=2.19 带来的错误，PyTorch 需要此文件：
    # https://github.com/NVIDIA/nccl/issues/1234
    "VLLM_NCCL_SO_PATH":
    lambda: os.environ.get("VLLM_NCCL_SO_PATH", None),


    # when `VLLM_NCCL_SO_PATH` is not set, vllm will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    # 当未设置 `VLLM_NCCL_SO_PATH` 时，vllm 将尝试在 `LD_LIBRARY_PATH` 
    # 指定的位置查找 NCCL 库文件
    "LD_LIBRARY_PATH":
    lambda: os.environ.get("LD_LIBRARY_PATH", None),


    # flag to control if vllm should use triton flash attention
    # 控制 vllm 是否应使用 Triton Flash Attention 的标志
    "VLLM_USE_TRITON_FLASH_ATTN":
    lambda: (os.environ.get("VLLM_USE_TRITON_FLASH_ATTN", "True").lower() in
             ("true", "1")),


    # Internal flag to enable Dynamo graph capture
    # 内部标志以启用 Dynamo 图捕获
    "VLLM_TEST_DYNAMO_GRAPH_CAPTURE":
    lambda: int(os.environ.get("VLLM_TEST_DYNAMO_GRAPH_CAPTURE", "0")),
    "VLLM_DYNAMO_USE_CUSTOM_DISPATCHER":
    lambda:
    (os.environ.get("VLLM_DYNAMO_USE_CUSTOM_DISPATCHER", "True").lower() in
     ("true", "1")),


    # Internal flag to control whether we use custom op,
    # or use the native pytorch implementation
    # 内部标志以控制我们是否使用自定义操作，
    # 或使用本机 PyTorch 实现
    "VLLM_TEST_COMPILE_NO_CUSTOM_OPS":
    lambda: int(os.environ.get("VLLM_TEST_COMPILE_NO_CUSTOM_OPS", "0")),


    # Internal flag to enable Dynamo fullgraph capture
    # 内部标志以启用 Dynamo 全图捕获
    "VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE":
    lambda: bool(
        os.environ.get("VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1") != "0"),


    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    # 在分布式设置中进程的本地排名，用于确定 GPU 设备 ID
    "LOCAL_RANK":
    lambda: int(os.environ.get("LOCAL_RANK", "0")),


    # used to control the visible devices in the distributed setting
    # 用于控制在分布式设置中可见设备的设置
    "CUDA_VISIBLE_DEVICES":
    lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),


    # timeout for each iteration in the engine
    # 引擎中每次迭代的超时时间
    "VLLM_ENGINE_ITERATION_TIMEOUT_S":
    lambda: int(os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "60")),


    # API key for VLLM API server
    # VLLM API 服务器的 API 密钥
    "VLLM_API_KEY":
    lambda: os.environ.get("VLLM_API_KEY", None),


    # S3 access information, used for tensorizer to load model from S3
    # S3 访问信息，用于 Tensorizer 从 S3 加载模型
    "S3_ACCESS_KEY_ID":
    lambda: os.environ.get("S3_ACCESS_KEY_ID", None),
    "S3_SECRET_ACCESS_KEY":
    lambda: os.environ.get("S3_SECRET_ACCESS_KEY", None),
    "S3_ENDPOINT_URL":
    lambda: os.environ.get("S3_ENDPOINT_URL", None),


    # Usage stats collection
    # 使用统计数据收集
    "VLLM_USAGE_STATS_SERVER":
    lambda: os.environ.get("VLLM_USAGE_STATS_SERVER", "https://stats.vllm.ai"),
    "VLLM_NO_USAGE_STATS":
    lambda: os.environ.get("VLLM_NO_USAGE_STATS", "0") == "1",
    "VLLM_DO_NOT_TRACK":
    lambda: (os.environ.get("VLLM_DO_NOT_TRACK", None) or os.environ.get(
        "DO_NOT_TRACK", None) or "0") == "1",
    "VLLM_USAGE_SOURCE":
    lambda: os.environ.get("VLLM_USAGE_SOURCE", "production"),


    # Logging configuration
    # If set to 0, vllm will not configure logging
    # If set to 1, vllm will configure logging using the default configuration
    #    or the configuration file specified by VLLM_LOGGING_CONFIG_PATH
    # 日志配置
    # 如果设置为 0，vllm 将不配置日志
    # 如果设置为 1，vllm 将使用默认配置或由 VLLM_LOGGING_CONFIG_PATH 指定的
    # 配置文件配置日志
    "VLLM_CONFIGURE_LOGGING":
    lambda: int(os.getenv("VLLM_CONFIGURE_LOGGING", "1")),
    "VLLM_LOGGING_CONFIG_PATH":
    lambda: os.getenv("VLLM_LOGGING_CONFIG_PATH"),


    # this is used for configuring the default logging level
    # 用于配置默认日志级别
    "VLLM_LOGGING_LEVEL":
    lambda: os.getenv("VLLM_LOGGING_LEVEL", "INFO"),


    # Trace function calls
    # If set to 1, vllm will trace function calls
    # Useful for debugging
    # 跟踪函数调用
    # 如果设置为 1，vllm 将跟踪函数调用
    # 调试时很有用
    "VLLM_TRACE_FUNCTION":
    lambda: int(os.getenv("VLLM_TRACE_FUNCTION", "0")),


    # Backend for attention computation
    # Available options:
    # - "TORCH_SDPA": use torch.nn.MultiheadAttention
    # - "FLASH_ATTN": use FlashAttention
    # - "XFORMERS": use XFormers
    # - "ROCM_FLASH": use ROCmFlashAttention
    # - "FLASHINFER": use flashinfer
    # 注意力计算的后端
    # 可用选项：
    # - "TORCH_SDPA": 使用 torch.nn.MultiheadAttention
    # - "FLASH_ATTN": 使用 FlashAttention
    # - "XFORMERS": 使用 XFormers
    # - "ROCM_FLASH": 使用 ROCmFlashAttention
    # - "FLASHINFER": 使用 flashinfer
    "VLLM_ATTENTION_BACKEND":
    lambda: os.getenv("VLLM_ATTENTION_BACKEND", None),


    # If set, vllm will use flashinfer sampler
    # 如果设置，vllm 将使用 flashinfer 采样器
    "VLLM_USE_FLASHINFER_SAMPLER":
    lambda: bool(int(os.getenv("VLLM_USE_FLASHINFER_SAMPLER", "0"))),


    # Pipeline stage partition strategy
    # 流水线阶段分区策略
    "VLLM_PP_LAYER_PARTITION":
    lambda: os.getenv("VLLM_PP_LAYER_PARTITION", None),


    # (CPU backend only) CPU key-value cache space.
    # default is 4GB
    # （仅 CPU 后端）CPU 键值缓存空间。
    # 默认是 4GB
    "VLLM_CPU_KVCACHE_SPACE":
    lambda: int(os.getenv("VLLM_CPU_KVCACHE_SPACE", "0")),


    # (CPU backend only) CPU core ids bound by OpenMP threads, e.g., "0-31",
    # "0,1,2", "0-31,33". CPU cores of different ranks are separated by '|'.
    # （仅 CPU 后端）通过 OpenMP 线程绑定的 CPU 核心 ID，例如 "0-31"，
    # "0,1,2"，"0-31,33"。不同排名的 CPU 核心用 '|' 分隔。
    "VLLM_CPU_OMP_THREADS_BIND":
    lambda: os.getenv("VLLM_CPU_OMP_THREADS_BIND", "all"),


    # OpenVINO key-value cache space
    # default is 4GB
    # OpenVINO 键值缓存空间
    # 默认是 4GB
    "VLLM_OPENVINO_KVCACHE_SPACE":
    lambda: int(os.getenv("VLLM_OPENVINO_KVCACHE_SPACE", "0")),


    # OpenVINO KV cache precision
    # default is bf16 if natively supported by platform, otherwise f16
    # To enable KV cache compression, please, explicitly specify u8
    # OpenVINO KV 缓存精度
    # 默认是 bf16 如果平台原生支持，否则是 f16
    # 要启用 KV 缓存压缩，请明确指定 u8
    "VLLM_OPENVINO_CPU_KV_CACHE_PRECISION":
    lambda: os.getenv("VLLM_OPENVINO_CPU_KV_CACHE_PRECISION", None),


    # Enables weights compression during model export via HF Optimum
    # default is False
    # 在通过 HF Optimum 导出模型时启用权重压缩
    # 默认是 False
    "VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS":
    lambda: bool(os.getenv("VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS", False)),


    # If the env var is set, then all workers will execute as separate
    # processes from the engine, and we use the same mechanism to trigger
    # execution on all workers.
    # Run vLLM with VLLM_USE_RAY_SPMD_WORKER=1 to enable it.
    # 如果设置了该环境变量，则所有工作进程将作为引擎的单独进程执行，
    # 并且我们使用相同的机制在所有工作进程上触发执行。
    # 运行 vLLM 时将 VLLM_USE_RAY_SPMD_WORKER=1 启用它。
    "VLLM_USE_RAY_SPMD_WORKER":
    lambda: bool(int(os.getenv("VLLM_USE_RAY_SPMD_WORKER", "0"))),


    # If the env var is set, it uses the Ray's compiled DAG API
    # which optimizes the control plane overhead.
    # Run vLLM with VLLM_USE_RAY_COMPILED_DAG=1 to enable it.
    # 如果设置了该环境变量，则使用 Ray 的编译 DAG API
    # 优化控制平面开销。
    # 运行 vLLM 时将 VLLM_USE_RAY_COMPILED_DAG=1 启用它。
    "VLLM_USE_RAY_COMPILED_DAG":
    lambda: bool(int(os.getenv("VLLM_USE_RAY_COMPILED_DAG", "0"))),


    # If the env var is set, it uses NCCL for communication in
    # Ray's compiled DAG. This flag is ignored if
    # VLLM_USE_RAY_COMPILED_DAG is not set.
    # 如果设置了该环境变量，则在 Ray 的编译 DAG 中使用 NCCL 进行通信。
    # 如果未设置 VLLM_USE_RAY_COMPILED_DAG，此标志将被忽略。
    "VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL":
    lambda: bool(int(os.getenv("VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL", "1"))
                 ),


    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    # 为工作进程使用专用的多进程上下文。
    # spawn 和 fork 均可使用
    "VLLM_WORKER_MULTIPROC_METHOD":
    lambda: os.getenv("VLLM_WORKER_MULTIPROC_METHOD", "fork"),


    # Path to the cache for storing downloaded assets
    # 存储下载资产的缓存
    "VLLM_ASSETS_CACHE":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_ASSETS_CACHE",
            os.path.join(get_default_cache_root(), "vllm", "assets"),
        )),


    # Timeout for fetching images when serving multimodal models
    # Default is 5 seconds
    # 在提供多模态模型时获取图像的超时时间
    # 默认是 5 秒
    "VLLM_IMAGE_FETCH_TIMEOUT":
    lambda: int(os.getenv("VLLM_IMAGE_FETCH_TIMEOUT", "5")),


    # Timeout for fetching audio when serving multimodal models
    # Default is 5 seconds
    # 在提供多模态模型时获取音频的超时时间
    # 默认是 5 秒
    "VLLM_AUDIO_FETCH_TIMEOUT":
    lambda: int(os.getenv("VLLM_AUDIO_FETCH_TIMEOUT", "5")),


    # Path to the XLA persistent cache directory.
    # Only used for XLA devices such as TPUs.
    # XLA 持久缓存目录的路径。
    # 仅用于 XLA 设备，例如 TPU。
    "VLLM_XLA_CACHE_PATH":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_XLA_CACHE_PATH",
            os.path.join(get_default_cache_root(), "vllm", "xla_cache"),
        )),
    "VLLM_FUSED_MOE_CHUNK_SIZE":
    lambda: int(os.getenv("VLLM_FUSED_MOE_CHUNK_SIZE", "32768")),


    # If set, vllm will skip the deprecation warnings.
    # 如果设置，vllm 将跳过弃用警告
    "VLLM_NO_DEPRECATION_WARNING":
    lambda: bool(int(os.getenv("VLLM_NO_DEPRECATION_WARNING", "0"))),


    # If set, the OpenAI API server will stay alive even after the underlying
    # AsyncLLMEngine errors and stops serving requests
    # 如果设置，OpenAI API 服务器将在底层
    # AsyncLLMEngine 错误后保持活动状态，停止提供请求
    "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH":
    lambda: bool(os.getenv("VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", 0)),


    # If the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN is set, it allows
    # the user to specify a max sequence length greater than
    # the max length derived from the model's config.json.
    # To enable this, set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
    # 如果环境变量 VLLM_ALLOW_LONG_MAX_MODEL_LEN 被设置，它允许
    # 用户指定一个大于从模型的 config.json 得到的最大长度的最大序列长度。
    # 要启用此功能，设置 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1。
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN":
    lambda:
    (os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "0").strip().lower() in
     ("1", "true")),


    # If set, forces FP8 Marlin to be used for FP8 quantization regardless
    # of the hardware support for FP8 compute.
    # 如果设置，无论硬件对 FP8 计算的支持如何，都强制使用 FP8 Marlin 进行 FP8 量化。
    "VLLM_TEST_FORCE_FP8_MARLIN":
    lambda:
    (os.environ.get("VLLM_TEST_FORCE_FP8_MARLIN", "0").strip().lower() in
     ("1", "true")),


    # Time in ms for the zmq client to wait for a response from the backend
    # server for simple data operations
    # zmq 客户端等待从后端的简单数据操作响应的时间（毫秒）
    "VLLM_RPC_TIMEOUT":
    lambda: int(os.getenv("VLLM_RPC_TIMEOUT", "10000")),


    # a list of plugin names to load, separated by commas.
    # if this is not set, it means all plugins will be loaded
    # if this is set to an empty string, no plugins will be loaded
    # 要加载的插件名称列表，用逗号分隔。
    # 如果没有设置，则意味着将加载所有插件
    # 如果设置为空字符串，则不会加载任何插件
    "VLLM_PLUGINS":
    lambda: None if "VLLM_PLUGINS" not in os.environ else os.environ[
        "VLLM_PLUGINS"].split(","),


    # Enables torch profiler if set. Path to the directory where torch profiler
    # traces are saved. Note that it must be an absolute path.
    # 如果设置，将启用 torch profiler。保存 torch profiler
    # 跟踪的目录路径。请注意，它必须是绝对路径。


    "VLLM_TORCH_PROFILER_DIR":
    lambda: (None if os.getenv("VLLM_TORCH_PROFILER_DIR", None) is None else os
             .path.expanduser(os.getenv("VLLM_TORCH_PROFILER_DIR", "."))),


    # If set, vLLM will use Triton implementations of AWQ.
    # 如果设置，vLLM 将使用 Triton 的 AWQ 实现
    "VLLM_USE_TRITON_AWQ":
    lambda: bool(int(os.getenv("VLLM_USE_TRITON_AWQ", "0"))),


    # If set, allow loading or unloading lora adapters in runtime,
    # 如果设置，允许在运行时加载或卸载 lora 适配器，
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING":
    lambda:
    (os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "0").strip().lower() in
     ("1", "true")),
}
```


