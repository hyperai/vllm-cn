---
title: 采样参数
---

*class vllm.SamplingParams(n: int = 1, *best_of*: int | None = None, presence_penalty: float = 0.0, frequency_penalty: float = 0.0, repetition_penalty: float = 1.0, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1, min_p: float = 0.0, seed: int | None = None, *use_beam_search*: bool = False, length_penalty: float = 1.0, early_stopping: bool | str = False, stop: str | ~typing.List[str] | None = None, stop_token_ids: ~typing.List[int] | None = None, ignore_eos: bool = False, max_tokens: int | None = 16, min_tokens: int = 0, logprobs: int | None = None, prompt_logprobs: int | None = None, detokenize: bool = True, skip_special_tokens: bool = True, spaces_between_special_tokens: bool = True, logits_processors: ~typing.Any | None = None, include_stop_str_in_output: bool = False, truncate_prompt_tokens: int[int] | None = None, output_kind: ~vllm.sampling_params.RequestOutputKind = RequestOutputKind.CUMULATIVE, output_text_buffer_length: int = 0, _all_stop_token_ids: ~typing.Set[int] = <factory>)*

[[source]](https://docs.vllm.ai/en/latest/_modules/vllm/sampling_params.html#SamplingParams)

用于文本生成的采样参数。

总体而言，我们遵循 OpenAI 文本 completion API ([https://platform.openai.com/docs/api-reference/completions/create)](https://platform.openai.com/docs/api-reference/completions/create)) 中的采样参数。另外，我们还支持 OpenAI 不支持的束搜索。

## 参数：

- **n** – 针对给定提示要返回的输出序列数量。

- **best_of** – 从提示生成的输出序列的数量。从这些 _best_of_ 序列中返回前 n 个序列。 _best_of_ 必须大于或等于 n。当 _use_beam_search_ 为 True 时，该值被视为束宽度。默认 _best_of_ 设置为 _n_。

- **presence_penalty** – 基于新生成的标记是否已出现在目前生成的文本中进行惩罚的浮点数。值 > 0 鼓励模型使用新 token，而值 < 0 鼓励模型重复 token。

- **frequency_penalty** – 基于新生成的标记在目前生成的文本中的出现频率进行惩罚的浮点数。值 > 0 鼓励模型使用新 token，而值 < 0 鼓励模型重复 token。

- **repetition_penalty** – 基于新生成的标记是否已出现在提示和目前生成的文本中进行惩罚的浮点数。值 > 1 鼓励模型使用新 token，而值 < 1 鼓励模型重复 token。

- **temperature** – 控制采样随机性的浮点数。较低的值使模型更具确定性，而较高的值使模型更加随机。值为 0 意味着贪婪采样。

- **top_p** – 控制要考虑的顶级 token 的累积概率的浮点数。必须在 (0, 1] 区间内。设置为 1 则考虑所有 token。

- **top_k** – 控制要考虑的顶级 token 数量的整数。设置为 -1 则考虑所有 token。

- **min_p** – 表示一个 token 被考虑的最小概率（相对于最可能标记的概率）的浮点数。必须在 [0, 1] 区间内。设置为 0 则禁用此功能。

- **seed** – 用于生成的随机种子。

- **stop** – 当生成这些字符串时停止生成的字符串列表。返回的输出将不包含停止字符串。

- **stop_token_ids** – 当生成这些标记时停止生成的标记列表。除非停止标记是特殊标记，否则返回的输出将包含停止标记。

- **include_stop_str_in_output** – 是否在输出文本中包含停止字符串。默认为 False。

- **ignore_eos** – EOS 生成后是否忽略 EOS token 并继续生成 token。

- **max_tokens** – 每个输出序列生成的最大 token 数。

- **min_tokens** – 在生成 EOS 或 stop_token_ids 之前，每个输出序列生成的最小 token 数量。

- **logprobs** – 每个输出 token 返回的对数概率数。当设置为 None 时，不返回概率。如果设置为非 None 值，则结果包括指定数量的最可能 token 以及所选 token 的对数概率。请注意，该实现遵循 OpenAI API：API 将始终返回采样 token 的对数概率，因此响应中最多可能有 logprobs+1 个元素。

- **prompt_logprobs** – 每个提示 token 返回的日志概率数。

- **detokenize** – 是否对输出进行去 token 化。默认为 True。

- **skip_special_tokens** – 是否跳过输出中的特殊 token。

- **spaces_between_special_tokens** – 是否在输出中的特殊 token 之间添加空格。默认为 True。

- **logits_processors** – 根据先前生成的 token 修改 logits 的函数列表，并可选择提示 token 作为第一个参数。

- **truncate_prompt_tokens** – 如果设置为整数 k，则将仅使用提示中的最后 k 个 token （即左截断）。默认为无（即不截断）。

- **guided_decoding** - 如果提供，引擎将从这些参数构建一个引导解码对数概率处理器。默认为 None。

- **logit_bias** - 如果提供，引擎将构建一个应用这些对数概率偏差的对数概率处理器。默认为 None。

- **allowed_token_ids**- 如果提供，引擎将构建一个仅保留给定标记 ID 的分数的对数概率处理器。默认为 None。

> clone() -> [SamplingParams](#sampling-parameters)
> [[source]](https://docs.vllm.ai/en/latest/_modules/vllm/sampling_params.html#SamplingParams.clone)

深层复制不包括 LogitsProcessor 对象。

LogitsProcessor 对象被排除在外，因为它们可能包含任意的、大量的重要数据。[请参阅 vllm-project/vllm#3087](https://github.com/vllm-project/vllm/issues/3087)

> update_from_generation_config(generation_config: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], model_eos_token_id: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [None](https://docs.python.org/3/library/constants.html#None)
[[source]](https://docs.vllm.ai/en/latest/_modules/vllm/sampling_params.html#SamplingParams.update_from_generation_config)

如果 Generation_config 中存在非默认值，请进行更新操作。
