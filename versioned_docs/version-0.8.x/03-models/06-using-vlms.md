---
title: 使用 VLM
---


vLLM 为视觉语言模型 (VLM) 提供实验性支持，可以参阅「支持的 VLM 列表」。本文档将向您展示如何使用 vLLM 运行并提供这些模型的服务。

**注意：**

我们正在积极改进对 VLM 的支持。预计在即将发布的版本中，VLM 的使用和开发会发生重大变化，但无需事先弃用。

We are continuously improving user & developer experience for VLMs. Please [open an issue on GitHub](https://github.com/vllm-project/vllm/issues/new/choose) if you have any feedback or feature requests.

我们不断改善 VLMs 的用户和开发人员体验。如果您有任何反馈或功能请求，请[访问  GitHub 并提出 issue](https://github.com/vllm-project/vllm/issues/new/choose)。


## 离线推理

### 单图像输入

 `LLM` 类的实例化过程与语言模型的实例化方式大致相同。

```python
llm = LLM(model="llava-hf/llava-1.5-7b-hf")
```


要将图像传递给模型，请注意 `vllm.inputs.PromptInputs` 中的以下内容: 


* `prompt`: 提示应遵循 HuggingFace 中记录的格式。

* `multi_modal_data`: 这是一个字典，它遵循 `vllm.multimodal.MultiModalDataDict` 中定义的模式。

```python
# Refer to the HuggingFace repo for the correct format to use
# 请参阅 HuggingFace 存储库以了解要使用的正确格式


prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"


# Load the image using PIL.Image
# 使用 PIL.Image 加载图像


image = PIL.Image.open(...)


# Single prompt inference
# 单提示推理


outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image},
})


for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)


# Inference with image embeddings as input
# 以图像嵌入作为输入进行推理


image_embeds = torch.load(...) # torch.Tensor of shape (1, image_feature_size, hidden_size of LM)


image_embeds = torch.load(...) # torch.Tensor 形状为 (1, image_feature_size, hide_size of LM)


outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image_embeds},
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
代码示例可以在 [examples/offline_inference_vision_language.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_vision_language.py) 中找到。


### 多图像输入

多图像输入仅被一部分视觉语言模型 (VLMs) 支持，如[此处](https://docs.vllm.ai/en/latest/models/supported_models.html#supported-vlms)所示。


若要在单个文本提示中启用多个多模态项目，您需要为 `LLM`类设置 `limit_mm_per_prompt` 参数。

```python
llm = LLM(
    model="microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,  # Required to load Phi-3.5-vision 需要加载 Phi-3.5-vision 模型


    max_model_len=4096,  # Otherwise, it may not fit in smaller GPUs 否则，可能无法适配较小的 GPU


    limit_mm_per_prompt={"image": 2},  # The maximum number to accept 每个文本提示允许的最大多模态项数量
)
```


您可以传入一个图像列表，而不是传入一张单独的图像。

```python
# Refer to the HuggingFace repo for the correct format to use
# 参考 HuggingFace 仓库中的正确格式来使用


prompt = "<|user|>\n<|image_1|>\n<|image_2|>\nWhat is the content of each image?<|end|>\n<|assistant|>\n"


# Load the images using PIL.Image
# 使用 PIL.Image 加载图片
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
代码示例可以在 [examples/offline_inference_vision_language_multi_image.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_vision_language_multi_image.py) 中找到。

多图像输入功能可以扩展应用于视频描述任务。以下展示了如何使用 Qwen2-VL 模型来实现这一点，因为该模型支持视频处理：


```python
# Specify the maximum number of frames per video to be 4. This can be changed.
# 指定每个视频的最大帧数为 4，这个数值可以根据需要调整。
llm = LLM("Qwen/Qwen2-VL-2B-Instruct", limit_mm_per_prompt={"image": 4})


# Create the request payload.
# 创建请求数据载荷。
video_frames = ... # load your video making sure it only has the number of frames specified earlier.
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


## 在线推理

您可以使用兼容 [OpenAI Vision API](https://platform.openai.com/docs/guides/vision) 的 vLLM HTTP 服务器提供视觉语言模型。


以下是一个关于如何使用 vLLM 的 OpenAI 兼容 API 服务器启动同一个 `microsoft/Phi-3.5-vision-instruct` 的示例。


```bash
vllm serve microsoft/Phi-3.5-vision-instruct --task generate \
  --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt image=2
```
**重要**
由于 OpenAI Vision API 基于 [Chat](https://platform.openai.com/docs/api-reference/chat) API，因此需要聊天模板来启动 API 服务器。


虽然 Phi-3.5-Vision 自带了聊天模板，但如果您使用的模型的分词器没有包含聊天模板，您可能需要自行提供。聊天模板通常可以根据 HuggingFace 存储库中模型文档的说明来推断。例如，LLaVA-1.5（`llava-hf/llava-1.5-7b-hf`) 模型就需要一个聊天模板，您可以[在这里](https://github.com/vllm-project/vllm/blob/main/examples/template_llava.jinja)找到该模板。


要使用服务器，您可以使用 OpenAI 客户端，如下例所示:

```python
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
            # 注意：不需要使用图像标记 `<image>` 格式化提示
            # since the prompt will be processed automatically by the API server.
            # 因为 API 服务器将自动处理提示。


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
完整的代码示例可以在 [examples/openai_api_client_for_multimodal.py](https://github.com/vllm-project/vllm/blob/main/examples/openai_api_client_for_multimodal.py) 中找到。

**注意：**

默认情况下，通过 http url 获取图像的超时时间为 `5` 秒。您可以通过设置环境变量来覆盖这个设置：

```plain
export VLLM_IMAGE_FETCH_TIMEOUT=<timeout>
```
**注意：**
在 API 请求中无需格式化提示词，因为它将由服务器进行处理。

