---

title: FP8 KV 缓存

---

[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


将 KV 缓存量化为 FP8 可减少其内存占用。这增加了存储在缓存中的 token 数量，从而提高了吞吐量。


## FP8 格式

OCP ([Open Compute Project](https://www.opencompute.org/))  指定了两种常见的 8 位浮点数据格式：

* E5M2（5 个指数位和 2 个尾数位）

* E4M3FN（4 个指数位和 3 个尾数位）

与 E5M2 相比，E4M3 格式的优点之一是浮点数以更高的精度表示。然而，FP8 E4M3 的动态范围较小（可以表示 ±240.0），通常需要在每个量化张量旁边使用更高精度的缩放因子（通常是 FP32） 。

## 当前局限性

目前，仅支持每个张量（标量）的缩放因子。我们正在开发支持更细粒度的缩放因子 （例如每个通道）。


### 性能影响

当前的 FP8 KV 缓存实现主要通过允许大约两倍的 KV 缓存分配空间来提高吞吐量。这使得：

* 能够处理更长的上下文长度

* 处理更多的并发请求批次


然而，由于当前的实现尚未包含融合的去量化和注意力操作，因此暂时没有延迟方面的改进。未来的版本将支持硬件加速的量化注意力操作，从而提供额外的性能优势。虽然最新的硬件（如 AMD MI300、NVIDIA Hopper 或更高版本的产品）支持 FP8 与其他格式（如 fp32、fp16、bf16）之间的原生硬件转换，但这一优势尚未完全实现。


研究表明，FP8 E4M3 量化通常只会对推理精度产生最小的影响，这使其成为吞吐量优化的实用选择。

### 使用示例

以下是如何启用 FP8 量化的示例：

```plain
# To calculate kv cache scales on the fly enable the calculate_kv_scales
# parameter
# 启用 calculate_kv_scales 参数以动态计算 kv 缓存缩放因子




from vllm import LLM, SamplingParams


sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf",
          kv_cache_dtype="fp8",
          calculate_kv_scales=True)
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```


`kv_cache_dtype`参数指定 KV 缓存存储的数据类型：

* `"auto"`：使用模型的默认“未量化”数据类型

* `"fp8"` 或 `"fp8_e4m3"`：支持 CUDA 11.8+ 和 ROCm (AMD GPU)

* `"fp8_e5m2"`：支持 CUDA 11.8+


### 校准缩放因子以提高精度

为了在使用 FP8 KV 缓存时获得最佳模型质量，我们建议使用针对代表性推理数据校准的缩放因子。推荐使用 [LLM Compressor](https://github.com/vllm-project/llm-compressor/) 工具来完成此过程。

### 安装

首先安装所需的依赖项：

```go
pip install llmcompressor
```


### 示例用法

以下是使用 `meta-llama/Llama-3.1-8B-Instruct` 的完整示例（大多数模型可以使用相同的模式）：

```plain
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.transformers import oneshot


# Select model and load it
# 选择模型并加载
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# Select calibration dataset
# 选择校准数据集
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"


# Configure calibration parameters
# 配置校准参数
NUM_CALIBRATION_SAMPLES = 512  # 512 samples is a good starting point
MAX_SEQUENCE_LENGTH = 2048


# Load and preprocess dataset
# 加载并预处理数据集
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


def process_and_tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)


# Configure quantization settings
# 配置量化设置
recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""


# Apply quantization
# 应用量化
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)


# Save quantized model
# 保存量化模型
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-KV"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```


上述脚本将在当前目录中创建一个包含量化模型的文件夹（例如 `Llama-3.1-8B-Instruct-FP8-KV`），其中包含校准后的缩放因子。


在运行模型时，必须指定 `kv_cache_dtype="fp8"` 以启用 KV 缓存量化并使用缩放因子：

```plain
from vllm import LLM, SamplingParams


sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(model="Llama-3.1-8B-Instruct-FP8-KV", kv_cache_dtype="fp8")
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```


