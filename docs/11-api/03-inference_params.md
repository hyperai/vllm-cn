---

title: 推理参数

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


vLLM API 的推理参数。


## 采样参数

***class*****vllm.****SamplingParams****(*****n: int = 1*****,*****best_of: int | None = None*****,*****_real_n: int | None = None*****,*****presence_penalty: float = 0.0*****,*****frequency_penalty: float = 0.0*****,*****repetition_penalty: float = 1.0*****,*****temperature: float = 1.0*****,*****top_p: float = 1.0*****,*****top_k: int = -1*****,*****min_p: float = 0.0*****,*****seed: int | None = None*****,*****stop: str | list[str] | None = None*****,*****stop_token_ids: list[int] | None = None*****,*****ignore_eos: bool = False*****,*****max_tokens: int | None = 16*****,*****min_tokens: int = 0*****,*****logprobs: int | None = None*****,*****prompt_logprobs: int | None = None*****,*****detokenize: bool = True*****,*****skip_special_tokens: bool = True*****,*****spaces_between_special_tokens: bool = True*****,*****logits_processors: ~typing.Any | None = None*****,*****include_stop_str_in_output: bool = False*****,*****truncate_prompt_tokens: int | None = None*****,*****output_kind: ~vllm.sampling_params.RequestOutputKind = RequestOutputKind.CUMULATIVE*****,*****output_text_buffer_length: int = 0*****,*****_all_stop_token_ids: set[int] = <factory>*****,*****guided_decoding: ~vllm.sampling_params.GuidedDecodingParams | None = None*****,*****logit_bias: dict[int*****,*****float] | None = None*****,*****allowed_token_ids: list[int] | None = None*****,*****extra_args: dict[str*****,*****typing.Any] | None = None*****,*****bad_words: list[str] | None = None*****,*****_bad_words_token_ids: list[list[int]] = <factory>*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L108)


文本生成的采样参数。

总体而言，我们遵循 OpenAI 文本补全 API 的采样参数（[https://platform.openai.com/docs/api-reference/completions/create](https://platform.openai.com/docs/api-reference/completions/create)）。此外，我们还支持 OpenAI 不支持的束搜索（beam search）。


**参数:**

* **n** – 为给定提示返回的输出序列数量。
* **best_of** – 从提示生成的输出序列数量。从这些 `best_of` 序列中，返回前 `n` 个序列。`best_of` 必须大于或等于 `n`。默认情况下，`best_of` 设置为 `n`。注意：此功能仅在 V0 中支持。
* **presence_penalty** – 基于新生成的文本中是否出现新 token 的惩罚值。值 > 0 鼓励模型使用新 token，而值 < 0 鼓励模型重复 token。
* **frequency_penalty** – 基于新生成的文本中 token 频率的惩罚值。值 > 0 鼓励模型使用新 token，而值 < 0 鼓励模型重复 token。
* **repetition_penalty** – 基于提示和已生成文本中是否出现新 token 的惩罚值。值 > 1 鼓励模型使用新 token，而值 < 1 鼓励模型重复 token。
* **temperature** – 控制采样随机性的浮点数。较低的值使模型更具确定性，而较高的值使模型更具随机性。值为 0 表示贪婪采样。
* **top_p** – 控制要考虑的 top token 的累积概率的浮点数。必须在 (0, 1] 范围内。设置为 1 以考虑所有 token。
* **top_k** – 控制要考虑的 top token 数量的整数。设置为 -1 以考虑所有 token。
* **min_p** – 表示 token 被考虑的最小概率，相对于最可能 token 的概率。必须在 [0, 1] 范围内。设置为 0 以禁用此功能。
* **seed** – 用于生成的随机种子。
* **stop** – 生成时停止生成的字符串列表。返回的输出将不包含这些停止字符串。
* **stop_token_ids** – 生成时停止生成的 token 列表。返回的输出将包含停止 token，除非停止 token 是特殊 token。
* **bad_words** – 不允许生成的单词列表。更准确地说，只有当生成的 token 可以完成相应 token 序列时，才不允许生成该序列的最后一个 token。
* **include_stop_str_in_output** – 是否在输出文本中包含停止字符串。默认为 False。
* **ignore_eos** – 是否忽略 EOS token 并在生成 EOS token 后继续生成 token。
* **max_tokens** – 每个输出序列生成的最大 token 数量。
* **min_tokens** – 在生成 EOS 或 stop_token_ids 之前，每个输出序列生成的最小 token 数量。
* **logprobs** – 每个输出 token 返回的对数概率数量。设置为 None 时，不返回概率。如果设置为非 None 值，结果将包括指定数量的最可能 token 的对数概率，以及所选 token 的对数概率。注意，实现遵循 OpenAI API：API 将始终返回采样 token 的对数概率，因此响应中可能有多达 `logprobs+1` 个元素。
* **prompt_logprobs** – 每个提示 token 返回的对数概率数量。
* **detokenize** – 是否对输出进行反 tokenize。默认为 True。
* **skip_special_tokens** – 是否在输出中跳过特殊 token。
* **spaces_between_special_tokens** – 是否在输出中的特殊 token 之间添加空格。默认为 True。
* **logits_processors** – 基于先前生成的 token 修改 logits 的函数列表，可选地以提示 token 作为第一个参数。
* **truncate_prompt_tokens** – 如果设置为整数 k，则仅使用提示的最后 k 个 token（即左截断）。默认为 None（即不截断）。
* **guided_decoding** – 如果提供，引擎将从这些参数构建一个引导解码的 logits 处理器。默认为 None。
* **logit_bias** – 如果提供，引擎将构建一个应用这些 logit 偏差的 logits 处理器。默认为 None。
* **allowed_token_ids** – 如果提供，引擎将构建一个仅保留给定 token id 分数的 logits 处理器。默认为 None。
* **extra_args** – 任意附加参数，可用于自定义采样实现。不用于任何内置采样实现。


**clone****()****→**[SamplingParams](https://docs.vllm.ai/en/latest/api/inference_params.html#vllm.SamplingParams)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L529)

深拷贝，但可能不包括 LogitsProcessor 对象。

LogitsProcessor 对象可能包含大量数据，复制成本较高。然而，如果不复制，处理器需要支持多序列的并行解码。参见 [vllm-project/vllm#3087](https://github.com/vllm-project/vllm/issues/3087)。


**update_from_generation_config****(*****generation_config:***[dict](https://docs.python.org/3/library/stdtypes.html#dict)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[Any](https://docs.python.org/3/library/typing.html#typing.Any)***]*****,*****model_eos_token_id:***[int](https://docs.python.org/3/library/functions.html#int)***|***[None](https://docs.python.org/3/library/constants.html#None) ***=*** ***None*****)****→**[None](https://docs.python.org/3/library/constants.html#None)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L446)

如果 `generation_config` 中有非默认值，则更新采样参数。


## 池化参数

***class*****vllm.****PoolingParams****(*****additional_data:***[Any](https://docs.python.org/3/library/typing.html#typing.Any)***|***[None](https://docs.python.org/3/library/constants.html#None) ***=*** ***None*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/pooling_params.py#L8)

用于池化模型的 API 参数。目前这是一个占位符。


**additional_data**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/pooling_params.py#L8)

池化所需的任何额外数据。


>**类型****：**
>Any | None

**clone****()****→**[PoolingParams](https://docs.vllm.ai/en/latest/api/inference_params.html#vllm.PoolingParams)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/pooling_params.py#L19)

返回 PoolingParams 实例的深拷贝。

