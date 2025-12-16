---
title: 多模态支持
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

本文档将引导您扩展基础模型，使其能够接受[多模态输入](https://docs.vllm.ai/en/latest/serving/multimodal_inputs.html#multimodal-inputs)。

## 1. 更新基础 vLLM 模型

假设您已经按照[这些步骤](https://docs.vllm.ai/en/latest/contributing/model/basic.html#new-model-basic)在 vLLM 中实现了模型。进一步更新模型如下：

在 `forward()` 中为每个对应于多模态输入的输入张量保留一个关键字参数，如下例所示：

```plain
  def forward(
      self,
      input_ids: torch.Tensor,
      positions: torch.Tensor,
+     pixel_values: torch.Tensor,
  ) -> SamplerOutput:
```

- 更方便的是，您可以简单地将 `**kwargs` 传递给 `forward()` 方法，并从中检索多模态输入的关键字参数。

实现 `get_multimodal_embeddings()`，该方法通过模型的多模态分词器运行多模态输入并返回嵌入。下面我们提供了一个典型实现模式的样板，但请根据您的需求进行调整。

```plain
class YourModelForImage2Seq(nn.Module):
    ...


    def _process_image_input(self, image_input: YourModelImageInputs) -> torch.Tensor:


        assert self.vision_encoder is not None
        image_features = self.vision_encoder(image_input)
        return self.multi_modal_projector(image_features)


    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:


        # 验证多模态输入关键字参数
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None


        # 通过编码器和投影器运行多模态输入
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings
```

> **重要**
> 
> 返回的 `multimodal_embeddings` 必须是形状为 `(num_items, feature_size, hidden_size)` 的 **3D** `torch.Tensor`，或者是形状为 `(feature_size, hidden_size)` 的 **2D** `torch.Tensor` 的 **列表/元组**，以便 `multimodal_embeddings[i]` 检索从请求的第 `i` 个多模态数据项（例如图像）生成的嵌入。

- 实现 `get_input_embeddings()` 以将 `multimodal_embeddings` 与来自 `input_ids` 的文本嵌入合并。如果模型的输入处理已正确实现（见下文），那么您可以利用我们提供的实用函数轻松合并嵌入。

```plain
from .utils import merge_multimodal_embeddings


class YourModelForImage2Seq(nn.Module):
    ...


    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:

        # `get_input_embeddings` 应该已经作为基础 vLLM 模型实现的要求之一实现。
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)


        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=self.config.image_token_index)


        return inputs_embeds
```

- 完成上述步骤后，使用 `SupportsMultiModal` 接口更新模型类。

```plain
+ from vllm.model_executor.models.interfaces import SupportsMultiModal


- class YourModelForImage2Seq(nn.Module):
+ class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```

> **注意**
> 
> 模型类不必命名为 `*ForCausalLM`。查看 [HuggingFace Transformers 文档](https://huggingface.co/docs/transformers/model_doc/auto#multimodal) 以获取一些示例。

## 2. 指定处理信息

接下来，创建 `BaseProcessingInfo` 的子类以提供与 HF 处理相关的基本信息。

### 输入项的最大数量

您需要重写抽象方法 `get_supported_mm_limits()` 以返回模型支持的每种模态的输入项的最大数量。

例如，如果模型支持任意数量的图像但每个提示仅支持一个视频：

```plain
def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
    return {"image": None, "video": 1}
```

### 占位符特征 token 的最大数量

此外，重写抽象方法 `get_mm_max_tokens_per_item()` 以返回每种模态的每个输入项的占位符特征 token 的最大数量。

调用模型时，视觉编码器的输出嵌入被分配给包含占位符特征 token 的输入位置。因此，占位符特征 token 的数量应等于输出嵌入的大小。

#### 基础示例：LLaVA

查看 HuggingFace 的 `LlavaForConditionalGeneration` 代码：

```plain
# https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llava/modeling_llava.py#L530-L544
n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
n_image_features = image_features.shape[0] * image_features.shape[1]


if n_image_tokens != n_image_features:
    raise ValueError(
        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
    )
special_image_mask = (
    (input_ids == self.config.image_token_index)
    .unsqueeze(-1)
    .expand_as(inputs_embeds)
    .to(inputs_embeds.device)
)
image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
```

每张图像的占位符特征 token 数量为 `image_features.shape[1]`。`image_features` 是在 `get_image_features` 方法中计算的：

```plain
# https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llava/modeling_llava.py#L290-L300
image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)


selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
if vision_feature_select_strategy == "default":
    selected_image_feature = selected_image_feature[:, 1:]
elif vision_feature_select_strategy == "full":
    selected_image_feature = selected_image_feature
else:
    raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")
image_features = self.multi_modal_projector(selected_image_feature)
return image_features
```

我们可以推断 `image_features.shape[1]` 基于视觉塔（`CLIPVisionModel` 对于 `llava-hf/llava-1.5-7b-hf` 模型）的 `image_outputs.hidden_states.shape[1]`。此外，我们只需要序列长度（张量的第二维度）来获取 `image_features.shape[1]`。序列长度由 `CLIPVisionTransformer` 中的初始隐藏状态决定，因为注意力机制不会改变输出隐藏状态的序列长度。

```plain
# https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L1094-L1102
hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
hidden_states = self.pre_layrnorm(hidden_states)


encoder_outputs = self.encoder(
    inputs_embeds=hidden_states,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)
```

为了找到序列长度，我们查看 `CLIPVisionEmbeddings` 的代码：

```plain
# https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L247-L257
target_dtype = self.patch_embedding.weight.dtype
patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
patch_embeds = patch_embeds.flatten(2).transpose(1, 2)


class_embeds = self.class_embedding.expand(batch_size, 1, -1)
embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
if interpolate_pos_encoding:
    embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
else:
    embeddings = embeddings + self.position_embedding(self.position_ids)
return embeddings
```

我们可以推断 `embeddings.shape[1] == self.num_positions`，其中

```plain
# https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L195-L196
self.num_patches = (self.image_size // self.patch_size) ** 2
self.num_positions = self.num_patches + 1
```

总的来说，图像的占位符特征 token 数量可以计算为：

```plain
def get_num_image_tokens(
    self,
    *,
    image_width: int,
    image_height: int,
) -> int:
    hf_config = self.get_hf_config()
    hf_processor = self.get_hf_processor()


    image_size = hf_config.vision_config.image_size
    patch_size = hf_config.vision_config.patch_size


    num_image_tokens = (image_size // patch_size) ** 2 + 1
    if hf_processor.vision_feature_select_strategy == "default":
        num_image_tokens -= 1


    return num_image_tokens
```

注意，图像 token 的数量不依赖于图像的宽度和高度。因此，我们可以使用任何图像大小计算最大图像 token 数量：

```plain
def get_image_size_with_most_features(self) -> ImageSize:
    hf_config = self.get_hf_config()
    width = height = hf_config.image_size
    return ImageSize(width=width, height=height)


def get_max_image_tokens(self) -> int:
    target_width, target_height = self.get_image_size_with_most_features()


    return self.get_num_image_tokens(
        image_width=target_width,
        image_height=target_height,
    )
```

因此，我们可以重写该方法为：

```plain
def get_mm_max_tokens_per_item(
    self,
    seq_len: int,
    mm_counts: Mapping[str, int],
) -> Mapping[str, int]:
    return {"image": self.get_max_image_tokens()}
```

**注意**

我们的[实际代码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava.py)更加抽象，以支持除 CLIP 之外的其他视觉编码器。

#### 非连续特征 token：Fuyu

查看 HuggingFace 的 `FuyuForCausalLM` 代码：

```plain
# https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/modeling_fuyu.py#L311-L322
if image_patches is not None and past_key_values is None:
    patch_embeddings = [
        self.vision_embed_tokens(patch.to(self.vision_embed_tokens.weight.dtype))
        .squeeze(0)
        .to(inputs_embeds.device)
        for patch in image_patches
    ]
    inputs_embeds = self.gather_continuous_embeddings(
        word_embeddings=inputs_embeds,
        continuous_embeddings=patch_embeddings,
        image_patch_input_indices=image_patches_indices,
    )
```

批次中第 `i` 项的占位符特征 token 数量为 `patch_embeddings[i].shape[0]`，与 `image_patches[i].shape[0]` 相同，即 `num_total_patches`。

与 LLaVA 不同，Fuyu 没有在建模文件中定义 patch 的数量。我们可以在哪里找到更多信息？考虑到模型输入来自 `FuyuProcessor` 的输出，让我们**查看预处理文件**。

图像输出是通过调用 `FuyuImageProcessor.preprocess` 然后调用 `FuyuImageProcessor.preprocess_with_tokenizer_info` 在 `FuyuProcessor` 中获得的。

在 `FuyuImageProcessor.preprocess` 中，图像被调整大小并填充到目标 `FuyuImageProcessor.size`，返回调整大小后的尺寸（但填充前）作为元数据。

```plain
# https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/processing_fuyu.py#L541-L544
image_encoding = self.image_processor.preprocess(images, **output_kwargs["images_kwargs"])
batch_images = image_encoding["images"]
image_unpadded_heights = image_encoding["image_unpadded_heights"]
image_unpadded_widths = image_encoding["image_unpadded_widths"]


# https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/image_processing_fuyu.py#L480-L
if do_resize:
    batch_images = [
        [self.resize(image, size=size, input_data_format=input_data_format) for image in images]
        for images in batch_images
    ]


image_sizes = [get_image_size(images[0], channel_dim=input_data_format) for images in batch_images]
image_unpadded_heights = [[image_size[0]] for image_size in image_sizes]
image_unpadded_widths = [[image_size[1]] for image_size in image_sizes]


if do_pad:
    batch_images = [
        [
            self.pad_image(
                image,
                size=size,
                mode=padding_mode,
                constant_values=padding_value,
                input_data_format=input_data_format,
            )
            for image in images
        ]
        for images in batch_images
    ]
```

在 `FuyuImageProcessor.preprocess_with_tokenizer_info` 中，图像根据此元数据被分割成 patch：

```plain
# https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/processing_fuyu.py#L417-L425
model_image_input = self.image_processor.preprocess_with_tokenizer_info(
    image_input=tensor_batch_images,
    image_present=image_present,
    image_unpadded_h=image_unpadded_heights,
    image_unpadded_w=image_unpadded_widths,
    image_placeholder_id=image_placeholder_id,
    image_newline_id=image_newline_id,
    variable_sized=True,
)


# https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/image_processing_fuyu.py#L638-L658
image_height, image_width = image.shape[1], image.shape[2]
if variable_sized:  # variable_sized=True
    new_h = min(
        image_height,
        math.ceil(image_unpadded_h[batch_index, subseq_index] / patch_height) * patch_height,
    )
    new_w = min(
        image_width,
        math.ceil(image_unpadded_w[batch_index, subseq_index] / patch_width) * patch_width,
    )
    image = image[:, :new_h, :new_w]
    image_height, image_width = new_h, new_w


num_patches = self.get_num_patches(image_height=image_height, image_width=image_width)
tensor_of_image_ids = torch.full(
    [num_patches], image_placeholder_id, dtype=torch.int32, device=image_input.device
)
patches = self.patchify_image(image=image.unsqueeze(0)).squeeze(0)
assert num_patches == patches.shape[0]
```

patch 的数量由 `FuyuImageProcessor.get_num_patches` 定义：

```plain
# https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/image_processing_fuyu.py#L552-L562
patch_size = patch_size if patch_size is not None else self.patch_size
patch_height, patch_width = self.patch_size["height"], self.patch_size["width"]


if image_height % patch_height != 0:
    raise ValueError(f"{image_height=} must be divisible by {patch_height}")
if image_width % patch_width != 0:
    raise ValueError(f"{image_width=} must be divisible by {patch_width}")


num_patches_per_dim_h = image_height // patch_height
num_patches_per_dim_w = image_width // patch_width
num_patches = num_patches_per_dim_h * num_patches_per_dim_w
```

我们可以在 vLLM 中使用以下代码计算：

```plain
def get_num_image_patches(
    self,
    *,
    image_width: int,
    image_height: int,
) -> int:
    image_processor = self.get_image_processor()
    target_width = image_processor.size["width"]
    target_height = image_processor.size["height"]
    patch_width = image_processor.patch_size["width"]
    patch_height = image_processor.patch_size["height"]


    if not (image_width <= target_width and image_height <= target_height):
        height_scale_factor = target_height / image_height
        width_scale_factor = target_width / image_width
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)


        image_height = int(image_height * optimal_scale_factor)
        image_width = int(image_width * optimal_scale_factor)


    ncols = math.ceil(image_width / patch_width)
    nrows = math.ceil(image_height / patch_height)
    return ncols * nrows
```

这些图像 patch 对应于占位符 token（`|SPEAKER|`）。然而，处理器还会插入换行 token（`|NEWLINE|`），如下所示：

```plain
# https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/image_processing_fuyu.py#L654-L670
tensor_of_image_ids = torch.full(
    [num_patches], image_placeholder_id, dtype=torch.int32, device=image_input.device
)
patches = self.patchify_image(image=image.unsqueeze(0)).squeeze(0)
assert num_patches == patches.shape[0]


if variable_sized:
    # 现在使用 |NEWLINE| 终止每行
    tensor_of_image_ids = tensor_of_image_ids.reshape(-1, image_width // patch_width)
    newline_ids = torch.full(
        [tensor_of_image_ids.shape[0], 1],
        image_newline_id,
        dtype=torch.int32,
        device=image_input.device,
    )
    tensor_of_image_ids = torch.cat([tensor_of_image_ids, newline_ids], dim=1)
    tensor_of_image_ids = tensor_of_image_ids.reshape(-1)
```

因此，图像的 token 布局为：

```plain
|SPEAKER||SPEAKER|...|SPEAKER||NEWLINE|
|SPEAKER||SPEAKER|...|SPEAKER||NEWLINE|
...
|SPEAKER||SPEAKER|...|SPEAKER||NEWLINE|
```

这使得占位符 token 在提示中不连续。由于 vLLM 要求特征 token 是连续的，**我们也将换行 token 视为特征 token**。

因此，总的特征 token 数量为

```plain
def get_num_image_tokens(
    self,
    *,
    image_width: int,
    image_height: int,
) -> int:
    image_processor = self.get_image_processor()
    target_width = image_processor.size["width"]
    target_height = image_processor.size["height"]
    patch_width = image_processor.patch_size["width"]
    patch_height = image_processor.patch_size["height"]


    if not (image_width <= target_width and image_height <= target_height):
        height_scale_factor = target_height / image_height
        width_scale_factor = target_width / image_width
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)


        image_height = int(image_height * optimal_scale_factor)
        image_width = int(image_width * optimal_scale_factor)


    ncols = math.ceil(image_width / patch_width)
    nrows = math.ceil(image_height / patch_height)
    return (ncols + 1) * nrows
```

要计算最大图像 token 数量，请记住输入图像首先被调整大小以适应 `image_processor.size`。因此，在转换为 patch 之前，图像的最大可能尺寸等于 `image_processor.size`。

```plain
def get_image_size_with_most_features(self) -> ImageSize:
    image_processor = self.get_image_processor()
    return ImageSize(width=image_processor.size["width"],
                        height=image_processor.size["height"])


def get_max_image_tokens(self) -> int:
    target_width, target_height = self.get_image_size_with_most_features()


    return self.get_num_image_tokens(
        image_width=target_width,
        image_height=target_height,
    )
```

因此，我们可以重写该方法为：

```plain
def get_mm_max_tokens_per_item(
    self,
    seq_len: int,
    mm_counts: Mapping[str, int],
) -> Mapping[str, int]:
    return {"image": self.get_max_image_tokens()}
```

> **注意**
> 
> 我们的[实际代码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/fuyu.py)直接返回 `ncols` 和 `nrows` 而不是总 token 数量。这是因为 `ncols` 和 `nrows` 用于指定特征 token 的布局（如本指南的第 4 步所示）。

## 3. 指定虚拟输入

接下来，继承 `BaseDummyInputsBuilder` 以构建用于 HF 处理和内存分析的虚拟输入。

### 用于内存分析

重写抽象方法 `get_dummy_processor_inputs()` 以构建用于内存分析的虚拟输入。此虚拟输入应导致模型的最坏情况内存使用，以便 vLLM 可以为其保留正确数量的内存。

假设内存使用量随着 token 数量的增加而增加，虚拟输入可以基于 `get_mm_max_tokens_per_item()` 的代码构建。

#### 基础示例：LLaVA

利用第 2 步中实现的 `get_image_size_with_most_features` 方法：

```plain
def get_dummy_processor_inputs(
    self,
    seq_len: int,
    mm_counts: Mapping[str, int],
) -> ProcessorInputs:
    num_images = mm_counts.get("image", 0)


    processor = self.info.get_hf_processor()
    image_token = processor.image_token

    hf_config = self.get_hf_config()
    target_width, target_height = self.info.get_image_size_with_most_features()


    mm_data = {
        "image":
        self._get_dummy_images(width=target_width,
                               height=target_height,
                               num_images=num_images)
    }


    return ProcessorInputs(
        prompt_text=image_token * num_images,
        mm_data=mm_data,
    )
```

#### 非连续特征 token：Fuyu

Fuyu 不需要在 HF 处理器的输入中出现图像占位符，因此无论图像数量如何，虚拟提示文本都为空。除此之外，此方法的逻辑与 LLaVA 非常相似：

```plain
def get_dummy_processor_inputs(
    self,
    seq_len: int,
    mm_counts: Mapping[str, int],
) -> ProcessorInputs:
    target_width, target_height = \
        self.info.get_image_size_with_most_features()
    num_images = mm_counts.get("image", 0)


    mm_data = {
        "image":
        self._get_dummy_images(width=target_width,
                                height=target_height,
                                num_images=num_images)
    }


    return ProcessorInputs(
        prompt_text="",
        mm_data=mm_data,
    )
```

## 4. 指定处理细节

接下来，创建 `BaseMultiModalProcessor` 的子类以填充有关 HF 处理的缺失细节。

> **另请参阅** >[多模态数据处理](https://docs.vllm.ai/en/latest/design/mm_processing.html#mm-processing)

### 多模态字段

重写 `_get_mm_fields_config()` 以返回由 HF 处理器输出的与输入多模态项相关的张量模式。

#### 基础示例：LLaVA

`CLIPImageProcessor` 的输出是一个形状为 `(num_images, num_channels, image_height, image_width)` 的简单张量：

```plain
# https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/image_processing_clip.py#L339-L345
images = [
    to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
    for image in all_images
]


data = {"pixel_values": images}
return BatchFeature(data=data, tensor_type=return_tensors)
```

因此，我们按如下方式重写 `_get_mm_fields_config()` ：

```plain
def _get_mm_fields_config(
    self,
    hf_inputs: BatchFeature,
    hf_processor_mm_kwargs: Mapping[str, object],
) -> Mapping[str, MultiModalFieldConfig]:
    return dict(
        pixel_values=MultiModalFieldConfig.batched("image"),
    )
```

> **注意**
> 
> 我们的[实际代码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava.py) 还支持预计算的图像嵌入，可以通过 `image_embeds` 参数传递给模型。

#### 非连续特征 token：Fuyu

`FuyuImageProcessor.preprocess_with_tokenizer_info` 的 `image_patches` 输出连接了属于批次中每个项目的图像的 patch：

```plain
# https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/image_processing_fuyu.py#L673-L679
        image_input_ids.append(tensor_of_image_ids)
        image_patches.append(patches)
    else:
        image_input_ids.append(torch.tensor([], dtype=torch.int32, device=image_input.device))


batch_image_input_ids.append(image_input_ids)
batch_image_patches.append(image_patches)
```

因此，`FuyuImageProcessor` 输出的 `image_patches` 的形状为 `(1, num_images, num_patches, patch_width * patch_height * num_channels)`。

为了支持像 LLaVA 中那样使用 `MultiModalFieldConfig.batched()`，我们通过重写 `BaseMultiModalProcessor._call_hf_processor()` 来移除额外的批次维度：

```plain
def _call_hf_processor(
    self,
    prompt: str,
    mm_data: Mapping[str, object],
    mm_kwargs: Mapping[str, object],
) -> BatchFeature:
    processed_outputs = super()._call_hf_processor(
        prompt=prompt,
        mm_data=mm_data,
        mm_kwargs=mm_kwargs,
    )


    image_patches = processed_outputs.get("image_patches")
    if image_patches is not None:
        images = mm_data["images"]
        assert isinstance(images, list)


        # 原始输出：(1, num_images, Pn, Px * Py * C)
        # 新输出：(num_images, Pn, Px * Py * C)


        assert (isinstance(image_patches, list)
                and len(image_patches) == 1)
        assert (isinstance(image_patches[0], torch.Tensor)
                and len(image_patches[0]) == len(images))


        processed_outputs["image_patches"] = image_patches[0]


    return processed_outputs
```

> **注意**
> 
> 我们的[实际代码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/fuyu.py)对纯文本输入有特殊处理，以防止 HF 处理器产生不必要的警告。

这使我们能够按以下方式重写 `_get_mm_fields_config()` ：

```plain
def _get_mm_fields_config(
    self,
    hf_inputs: BatchFeature,
    hf_processor_mm_kwargs: Mapping[str, object],
) -> Mapping[str, MultiModalFieldConfig]:
    return dict(image_patches=MultiModalFieldConfig.batched("image"))
```

### 提示更新

重写 `_get_prompt_updates()` 以返回 `PromptUpdate` 实例的列表。

每个 `PromptUpdate` 实例指定由 HF 处理器执行的更新操作（例如：插入、替换）。

#### 基础示例：LLaVA

查看 HF 的 `LlavaProcessor`：

```plain
# https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llava/processing_llava.py#L167-L170
prompt_strings = []
for sample in text:
    sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
    prompt_strings.append(sample)
```

它只是将每个输入 `image_token` 重复与占位符特征 token 数量（`num_image_tokens`）相等的次数。基于此，我们重写 `_get_prompt_updates()` 如下：

```plain
def _get_prompt_updates(
    self,
    mm_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, object],
    out_mm_kwargs: MultiModalKwargs,
) -> Sequence[PromptUpdate]:
    hf_config = self.info.get_hf_config()
    image_token_id = hf_config.image_token_index


    def get_replacement(item_idx: int):
        images = mm_items.get_items("image", ImageProcessorItems)


        image_size = images.get_image_size(item_idx)
        num_image_tokens = self.info.get_num_image_tokens(
            image_width=image_size.width,
            image_height=image_size.height,
        )


        return [image_token_id] * num_image_tokens


    return [
        PromptReplacement(
            modality="image",
            target=[image_token_id],
            replacement=get_replacement,
        ),
    ]
```

#### Non-consecutive feature tokens: Fuyu

#### 非连续特征 token：Fuyu

回顾第 2 步中的特征 token 布局：

```plain
|SPEAKER||SPEAKER|...|SPEAKER||NEWLINE|
|SPEAKER||SPEAKER|...|SPEAKER||NEWLINE|
...
|SPEAKER||SPEAKER|...|SPEAKER||NEWLINE|
```

我们定义了一个辅助函数直接返回 `ncols` 和 `nrows`：

```plain
def get_image_feature_grid_size(
    self,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    image_processor = self.get_image_processor()
    target_width = image_processor.size["width"]
    target_height = image_processor.size["height"]
    patch_width = image_processor.patch_size["width"]
    patch_height = image_processor.patch_size["height"]


    if not (image_width <= target_width and image_height <= target_height):
        height_scale_factor = target_height / image_height
        width_scale_factor = target_width / image_width
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)


        image_height = int(image_height * optimal_scale_factor)
        image_width = int(image_width * optimal_scale_factor)


    ncols = math.ceil(image_width / patch_width)
    nrows = math.ceil(image_height / patch_height)
    return ncols, nrows
```

基于此，我们可以初步定义替换 token 为：

```plain
def get_replacement(item_idx: int):
    images = mm_items.get_items("image", ImageProcessorItems)
    image_size = images.get_image_size(item_idx)


    ncols, nrows = self.info.get_image_feature_grid_size(
        image_width=image_size.width,
        image_height=image_size.height,
    )


    # `_IMAGE_TOKEN_ID` 对应于 `|SPEAKER|`
    # `_NEWLINE_TOKEN_ID` 对应于 `|NEWLINE|`
    return ([_IMAGE_TOKEN_ID] * ncols + [_NEWLINE_TOKEN_ID]) * nrows
```

然而，这并不完全正确。在调用 `FuyuImageProcessor.preprocess_with_tokenizer_info` 后，BOS token（`<s>`）也会被添加到提示中：

```plain
# https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/fuyu/processing_fuyu.py#L417-L435
model_image_input = self.image_processor.preprocess_with_tokenizer_info(
    image_input=tensor_batch_images,
    image_present=image_present,
    image_unpadded_h=image_unpadded_heights,
    image_unpadded_w=image_unpadded_widths,
    image_placeholder_id=image_placeholder_id,
    image_newline_id=image_newline_id,
    variable_sized=True,
)
prompt_tokens, prompts_length = _tokenize_prompts_with_image_and_batch(
    tokenizer=self.tokenizer,
    prompts=prompts,
    scale_factors=scale_factors,
    max_tokens_to_generate=self.max_tokens_to_generate,
    max_position_embeddings=self.max_position_embeddings,
    add_BOS=True,
    add_beginning_of_answer_token=True,
)
```

为了适应这种情况，您可以返回一个 `PromptUpdateDetails` 实例，而不是字符串，其中包含不同的 `full` 和 `feature` 属性：

```plain
hf_config = self.info.get_hf_config()
bos_token_id = hf_config.bos_token_id  # `<s>`
assert isinstance(bos_token_id, int)


def get_replacement_fuyu(item_idx: int):
    images = mm_items.get_items("image", ImageProcessorItems)
    image_size = images.get_image_size(item_idx)


    ncols, nrows = self.info.get_image_feature_grid_size(
        image_width=image_size.width,
        image_height=image_size.height,
    )
    image_tokens = ([_IMAGE_TOKEN_ID] * ncols +
                    [_NEWLINE_TOKEN_ID]) * nrows


    return PromptUpdateDetails(
        full=image_tokens + [bos_token_id],
        features=image_tokens,
    )
```

最后，注意到 HF 处理器从分词后的提示中移除了 `|ENDOFTEXT|` token，我们可以搜索它以在字符串的开头进行替换：

```plain
def _get_prompt_updates(
    self,
    mm_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, object],
    out_mm_kwargs: MultiModalKwargs,
) -> Sequence[PromptUpdate]:
    hf_config = self.info.get_hf_config()
    bos_token_id = hf_config.bos_token_id
    assert isinstance(bos_token_id, int)


    tokenizer = self.info.get_tokenizer()
    eot_token_id = tokenizer.bos_token_id
    assert isinstance(eot_token_id, int)


    def get_replacement_fuyu(item_idx: int):
        images = mm_items.get_items("image", ImageProcessorItems)
        image_size = images.get_image_size(item_idx)


        ncols, nrows = self.info.get_image_feature_grid_size(
            image_width=image_size.width,
            image_height=image_size.height,
        )
        image_tokens = ([_IMAGE_TOKEN_ID] * ncols +
                        [_NEWLINE_TOKEN_ID]) * nrows


        return PromptUpdateDetails(
            full=image_tokens + [bos_token_id],
            features=image_tokens,
        )


    return [
        PromptReplacement(
            modality="image",
            target=[eot_token_id],
            replacement=get_replacement_fuyu,
        )
    ]
```

## 5. 注册处理器相关类

在定义了 `BaseProcessingInfo`（第 2 步）、`BaseDummyInputsBuilder`（第 3 步）和 `BaseMultiModalProcessor`（第 4 步）之后，使用 `MULTIMODAL_REGISTRY.register_processor` 装饰模型类，将它们注册到多模态注册表中：

```plain
  from vllm.model_executor.models.interfaces import SupportsMultiModal
+ from vllm.multimodal import MULTIMODAL_REGISTRY


+ @MULTIMODAL_REGISTRY.register_processor(YourMultiModalProcessor,
+                                         info=YourProcessingInfo,
+                                         dummy_inputs=YourDummyInputsBuilder)
  class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```

## 注意

### 插入特征 token 而不进行替换

一些 HF 处理器直接插入特征 token，而不替换原始提示中的任何内容。在这种情况下，您可以在 `_get_prompt_updates()` 中使用 `PromptInsertion` 而不是 `PromptReplacement`。

示例：

- BLIP-2（在提示开头插入）：[vllm/model_executor/models/blip2.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/blip2.py)
- Florence2（在提示开头插入）：[vllm/model_executor/models/florence2.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/florence2.py)
- Molmo（在 `<|endoftext|>` token 后插入）：[vllm/model_executor/models/molmo.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/molmo.py)

### 处理与多模态数据无关的提示更新

`_get_prompt_updates()` 假设每次应用提示更新都对应于一个多模态项。如果 HF 处理器执行额外的处理，无论有多少多模态项，您都应该重写 `_apply_hf_processor_tokens_only()`，以便处理后的 token 输入与在文本输入上应用 HF 处理器的结果一致。这是因为根据[我们的设计](https://docs.vllm.ai/en/latest/design/mm_processing.html#mm-processing)，token 输入会绕过 HF 处理器。

示例：

- Chameleon（附加 `sep_token`）：[vllm/model_executor/models/chameleon.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/chameleon.py)
- Fuyu（附加 `boa_token`）：[vllm/model_executor/models/fuyu.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/fuyu.py)
- Molmo（应用未在其他地方定义的聊天模板）：[vllm/model_executor/models/molmo.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/molmo.py)

### 自定义 HF 处理器

一些模型在 HF Hub 上没有定义 HF 处理器类。在这种情况下，您可以定义一个与 HF 处理器具有相同调用签名的自定义 HF 处理器，并将其传递给 `_call_hf_processor()`。

示例：

- DeepSeek-VL2: [vllm/model_executor/models/deepseek_vl2.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_vl2.py)
- InternVL: [vllm/model_executor/models/internvl.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/internvl.py)
- Qwen-VL: [vllm/model_executor/models/qwen_vl.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen_vl.py)
