---

title: 多模态输入

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


本页教你如何在 vLLM 中向[多模态模型](https://docs.vllm.ai/en/latest/models/supported_models.html#supported-mm-models)传递多模态输入。


>**注意**
>我们正在积极迭代多模态支持功能。有关即将到来的变更，请参阅[此 RFC](https://github.com/vllm-project/vllm/issues/4194#) 。如果您有任何反馈或功能请求，请在 [GitHub 上提交问题](https://github.com/vllm-project/vllm/issues/new/choose) 。

## 离线推理

要输入多模态数据，请按照 `vllm.inputs.PromptType` 中的模式操作：

* `prompt`：提示词应遵循 HuggingFace 文档中记录的格式。
* `multi_modal_data`：这是一个字典，遵循 `vllm.multimodal.inputs.MultiModalDataDict` 中定义的模式。

### 

### 图像输入

您可以将单个图像传递到多模态字典的 `'image'` 字段中，如下例所示：

```plain
llm = LLM(model="llava-hf/llava-1.5-7b-hf")


# Refer to the HuggingFace repo for the correct format to use
# 参考 HuggingFace 仓库以使用正确的格式
prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"


# Load the image using PIL.Image
# 使用 PIL.Image 加载图像
image = PIL.Image.open(...)


# Single prompt inference
# 单提示词推理
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image},
})


for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)


# Batch inference
# 批量推理
image_1 = PIL.Image.open(...)
image_2 = PIL.Image.open(...)
outputs = llm.generate(
    [
        {
            "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
            "multi_modal_data": {"image": image_1},
        },
        {
            "prompt": "USER: <image>\nWhat's the color of this image?\nASSISTANT:",
            "multi_modal_data": {"image": image_2},
        }
    ]
)


for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
```


完整示例：[examples/offline_inference/vision_language.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py)


要在同一文本提示中替换多张图像，可以传递一个图像列表：

```plain
llm = LLM(
    model="microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,  # Required to load Phi-3.5-vision
    max_model_len=4096,  # Otherwise, it may not fit in smaller GPUs
    limit_mm_per_prompt={"image": 2},  # The maximum number to accept
)


# Refer to the HuggingFace repo for the correct format to use
# 参考 HuggingFace 仓库以使用正确的格式
prompt = "<|user|>\n<|image_1|>\n<|image_2|>\nWhat is the content of each image?<|end|>\n<|assistant|>\n"


# Load the images using PIL.Image
# 使用 PIL.Image 加载图像
image1 = PIL.Image.open(...)
image2 = PIL.Image.open(...)


outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {
        "image": [image1, image2]
    },
})


for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
```


完整示例：[examples/offline_inference/vision_language_multi_image.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language_multi_image.py)


多图像输入可以扩展到视频字幕生成。我们以 [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) 为例，因为它支持视频：

```plain
# Specify the maximum number of frames per video to be 4. This can be changed.
# 指定每段视频的最大帧数为 4。可以调整此值。
llm = LLM("Qwen/Qwen2-VL-2B-Instruct", limit_mm_per_prompt={"image": 4})


# Create the request payload.
# 创建请求负载。
video_frames = ... # load your video making sure it only has the number of frames specified earlier.
video_frames = ... # 加载视频，确保帧数不超过之前指定的数量。
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this set of frames. Consider the frames to be a part of the same video."},
    ],
}
for i in range(len(video_frames)):
    base64_image = encode_image(video_frames[i]) # base64 encoding.
    new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    message["content"].append(new_image)


# Perform inference and log output.
# 执行推理并记录输出。
outputs = llm.chat([message])


for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
```


### 视频输入

您可以直接将 NumPy 数组列表传递到多模态字典的 `'video'` 字段中，而无需使用多图像输入。

完整示例：[examples/offline_inference/vision_language.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py)

### 

### 音频输入

您可以将元组 `(array, sampling_rate)` 传递到多模态字典的 `'audio'` 字段中。

完整示例：[examples/offline_inference/audio_language.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/audio_language.py)

### 

### 嵌入输入

要将预计算的嵌入（属于某种数据类型，如图像、视频或音频）直接输入到语言模型中，请将形状为 `(num_items, feature_size, hidden_size of LM)` 的张量传递到多模态字典的相应字段中。

```plain
# Inference with image embeddings as input
# 使用图像嵌入作为输入进行推理
llm = LLM(model="llava-hf/llava-1.5-7b-hf")


# Refer to the HuggingFace repo for the correct format to use
# 参考 HuggingFace 仓库以使用正确的格式
prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"


# Embeddings for single image
# 单张图像的嵌入
# torch.Tensor of shape (1, image_feature_size, hidden_size of LM)
# torch.Tensor，形状为 (1, image_feature_size, hidden_size of LM)
image_embeds = torch.load(...)


outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image_embeds},
})


for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
```


对于 Qwen2-VL 和 MiniCPM-V，我们接受与嵌入一起的额外参数：

```plain
# Construct the prompt based on your model
# 根据模型构建提示词
prompt = ...


# Embeddings for multiple images
# 多张图像的嵌入
# torch.Tensor of shape (num_images, image_feature_size, hidden_size of LM)
# torch.Tensor，形状为 (num_images, image_feature_size, hidden_size of LM)
image_embeds = torch.load(...)


# Qwen2-VL
llm = LLM("Qwen/Qwen2-VL-2B-Instruct", limit_mm_per_prompt={"image": 4})
mm_data = {
    "image": {
        "image_embeds": image_embeds,
        # image_grid_thw is needed to calculate positional encoding.
        "image_grid_thw": torch.load(...),  # torch.Tensor of shape (1, 3),
    }
}


# MiniCPM-V
llm = LLM("openbmb/MiniCPM-V-2_6", trust_remote_code=True, limit_mm_per_prompt={"image": 4})
mm_data = {
    "image": {
        "image_embeds": image_embeds,
        # image_sizes is needed to calculate details of the sliced image.
        "image_sizes": [image.size for image in images],  # list of image sizes
    }
}


outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": mm_data,
})


for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
```


## 在线服务

我们的 OpenAI 兼容服务器通过 [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) 接受多模态数据。


**重要**

使用 Chat Completions API 需要**聊天模板** 。


虽然大多数模型都附带聊天模板，但其他模型需要您自行定义。聊天模板可以根据模型 HuggingFace 仓库中的文档推断。例如，LLaVA-1.5 (`llava-hf/llava-1.5-7b-hf`) 需要一个可以在此处找到的聊天模板：[examples/template_llava.jinja](https://github.com/vllm-project/vllm/blob/main/examples/template_llava.jinja)

### 

### 图像输入

图像输入根据 [OpenAI Vision API](https://platform.openai.com/docs/guides/vision) 支持。以下是一个使用 Phi-3.5-Vision 的简单示例。

首先，启动 OpenAI 兼容服务器：

```plain
vllm serve microsoft/Phi-3.5-vision-instruct --task generate \
  --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt image=2
```


然后，您可以按如下方式使用 OpenAI 客户端：

```plain
from openai import OpenAI


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


# Single-image input inference
# 单图像输入推理
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


chat_response = client.chat.completions.create(
    model="microsoft/Phi-3.5-vision-instruct",
    messages=[{
        "role": "user",
        "content": [
            # NOTE: The prompt formatting with the image token `<image>` is not needed
            # since the prompt will be processed automatically by the API server.
            {"type": "text", "text": "What’s in this image?"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }],
)
print("Chat completion output:", chat_response.choices[0].message.content)


# Multi-image input inference
# 多图像输入推理
image_url_duck = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
image_url_lion = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"


chat_response = client.chat.completions.create(
    model="microsoft/Phi-3.5-vision-instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What are the animals in these images?"},
            {"type": "image_url", "image_url": {"url": image_url_duck}},
            {"type": "image_url", "image_url": {"url": image_url_lion}},
        ],
    }],
)
print("Chat completion output:", chat_response.choices[0].message.content)
```


完整示例：[examples/online_serving/openai_chat_completion_client_for_multimodal.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_client_for_multimodal.py)


>**提示**
>vLLM 也支持从本地文件路径加载：您可以通过 `--allowed-local-media-path` 在启动 API 服务器/引擎时指定允许的本地媒体路径，并在 API 请求中将文件路径作为 `url` 传递。

>**提示**
>不需要在 API 请求的文本内容中放置图像占位符——它们已经由图像内容表示。事实上，您可以通过交错文本和图像内容在文本中间放置图像占位符。

**注意**

默认情况下，通过 HTTP URL 获取图像的超时时间为 `5` 秒。您可以通过设置环境变量覆盖此值：

```go
export VLLM_IMAGE_FETCH_TIMEOUT=<timeout>
```


### 视频输入

您可以通过 `video_url` 传递视频文件，而不是 `image_url`。以下是一个使用 [LLaVA-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) 的简单示例。

首先，启动 OpenAI 兼容服务器：

```plain
vllm serve llava-hf/llava-onevision-qwen2-0.5b-ov-hf --task generate --max-model-len 8192
```


然后，您可以按如下方式使用 OpenAI 客户端：

```plain
from openai import OpenAI


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"


## Use video url in the payload
## 在负载中使用视频 URL
chat_completion_from_url = client.chat.completions.create(
    messages=[{
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this video?"
            },
            {
                "type": "video_url",
                "video_url": {
                    "url": video_url
                },
            },
        ],
    }],
    model=model,
    max_completion_tokens=64,
)


result = chat_completion_from_url.choices[0].message.content
print("Chat completion output from image url:", result)
```


完整示例：[examples/online_serving/openai_chat_completion_client_for_multimodal.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_client_for_multimodal.py)


**注意**

默认情况下，通过 HTTP URL 获取视频的超时时间为 `30` 秒。您可以通过设置环境变量覆盖此值：

```go
export VLLM_VIDEO_FETCH_TIMEOUT=<timeout>
```


### 音频输入

音频输入根据 [OpenAI Audio API](https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-in) 支持。以下是一个使用 Ultravox-v0.5-1B 的简单示例。

首先，启动 OpenAI 兼容服务器：

```plain
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b
```


然后，您可以按如下方式使用 OpenAI 客户端：

```plain
import base64
import requests
from openai import OpenAI
from vllm.assets.audio import AudioAsset


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""
    """将远程 URL 获取的内容编码为 base64 格式。"""


    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')


    return result


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


# Any format supported by librosa is supported
# 支持 librosa 支持的任何格式
audio_url = AudioAsset("winning_call").url
audio_base64 = encode_base64_content_from_url(audio_url)


chat_completion_from_base64 = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this audio?"
            },
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_base64,
                    "format": "wav"
                },
            },
        ],
    }],
    model=model,
    max_completion_tokens=64,
)


result = chat_completion_from_base64.choices[0].message.content
print("Chat completion output from input audio:", result)
```


或者，您可以传递 `audio_url`，这是图像输入中 `image_url` 的音频对应物：

```plain
chat_completion_from_url = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this audio?"
            },
            {
                "type": "audio_url",
                "audio_url": {
                    "url": audio_url
                },
            },
        ],
    }],
    model=model,
    max_completion_tokens=64,
)


result = chat_completion_from_url.choices[0].message.content
print("Chat completion output from audio url:", result)
```


完整示例：[examples/online_serving/openai_chat_completion_client_for_multimodal.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_client_for_multimodal.py)


**注意**

默认情况下，通过 HTTP URL 获取音频的超时时间为 `10` 秒。您可以通过设置环境变量覆盖此值：

```go
export VLLM_AUDIO_FETCH_TIMEOUT=<timeout>
```


### 嵌入输入

要将预计算的嵌入（属于某种数据类型，如图像、视频或音频）直接输入到语言模型中，请将张量传递到多模态字典的相应字段中。

#### 

#### 图像嵌入输入

对于图像嵌入，您可以将 base64 编码的张量传递到 `image_embeds` 字段。以下示例演示了如何将图像嵌入传递到 OpenAI 服务器：

```plain
image_embedding = torch.load(...)
grid_thw = torch.load(...) # Required by Qwen/Qwen2-VL-2B-Instruct


buffer = io.BytesIO()
torch.save(image_embedding, buffer)
buffer.seek(0)
binary_data = buffer.read()
base64_image_embedding = base64.b64encode(binary_data).decode('utf-8')


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)


# Basic usage - this is equivalent to the LLaVA example for offline inference
# 基本用法 - 这等同于离线推理中的 LLaVA 示例
model = "llava-hf/llava-1.5-7b-hf"
embeds =  {
    "type": "image_embeds",
    "image_embeds": f"{base64_image_embedding}" 
}


# Pass additional parameters (available to Qwen2-VL and MiniCPM-V)
# 传递额外参数（适用于 Qwen2-VL 和 MiniCPM-V）
model = "Qwen/Qwen2-VL-2B-Instruct"
embeds =  {
    "type": "image_embeds",
    "image_embeds": {
        "image_embeds": f"{base64_image_embedding}" , # Required
        "image_grid_thw": f"{base64_image_grid_thw}"  # Required by Qwen/Qwen2-VL-2B-Instruct
    },
}
model = "openbmb/MiniCPM-V-2_6"
embeds =  {
    "type": "image_embeds",
    "image_embeds": {
        "image_embeds": f"{base64_image_embedding}" , # Required
        "image_sizes": f"{base64_image_sizes}"  # Required by openbmb/MiniCPM-V-2_6
    },
}
chat_completion = client.chat.completions.create(
    messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {
            "type": "text",
            "text": "What's in this image?",
        },
        embeds,
        ],
    },
],
    model=model,
)
```


**注意**

只有一条消息可以包含 `{"type": "image_embeds"}`。如果与需要额外参数的模型一起使用，您还必须为每个参数提供一个张量，例如 `image_grid_thw`、`image_sizes` 等。


