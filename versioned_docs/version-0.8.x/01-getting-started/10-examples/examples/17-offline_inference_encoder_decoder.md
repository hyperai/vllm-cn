---
title: 离线推理编码器-解码器
---


源代码: [vllm-project/vllm](https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference_encoder_decoder.py)

```python
'''
Demonstrate prompting of text-to-text
encoder/decoder models, specifically BART
展示如何对文本到文本的编码器/解码器模型进行提示，特别是 BART。

'''

from vllm import LLM, SamplingParams
from vllm.inputs import (ExplicitEncoderDecoderPrompt, TextPrompt,
                         TokensPrompt, zip_enc_dec_prompts)

dtype = "float"

# Create a BART encoder/decoder model instance
# 创建一个 BART encoder/decoder 模型 实例
llm = LLM(
    model="facebook/bart-large-cnn",
    dtype=dtype,
)

# Get BART tokenizer
# 获取 BART tokenizer

tokenizer = llm.llm_engine.get_tokenizer_group()

# Test prompts
# 测试提示
#
# This section shows all of the valid ways to prompt an
# encoder/decoder model.
# 这一部分展示了对 encoder/decoder 模型进行提示的所有有效方式。
#
# - Helpers for building prompts
# - Helpers 用于构建提示
text_prompt_raw = "Hello, my name is"
text_prompt = TextPrompt(prompt="The president of the United States is")
tokens_prompt = TokensPrompt(prompt_token_ids=tokenizer.encode(
    prompt="The capital of France is"))
# - Pass a single prompt to encoder/decoder model
# - 传递单个提示给 encoder/decoder 模型

#   (implicitly encoder input prompt);
#  （隐式传递 encoder 输入提示）
#   decoder input prompt is assumed to be None
#   decoder 输入提示假定为 None

single_text_prompt_raw = text_prompt_raw  # Pass a string directly # 直接传递一个字符串
single_text_prompt = text_prompt  # Pass a TextPrompt
single_tokens_prompt = tokens_prompt  # Pass a TokensPrompt

# - Pass explicit encoder and decoder input prompts within one data structure.
# - 在一个数据结构中传递显示的 encoder 和 decoder 输入提示

#   Encoder and decoder prompts can both independently be text or tokens, with
#   no requirement that they be the same prompt type. Some example prompt-type
#   combinations are shown below, note that these are not exhaustive.
#  encoder 和 decoder 提示都可以是文本或标记，且不要求它们是相同的提示类型。
#  下面展示了一些示例提示类型组合，注意这些并不是全部的组合。


enc_dec_prompt1 = ExplicitEncoderDecoderPrompt(
    # Pass encoder prompt string directly, &
    # pass decoder prompt tokens
    encoder_prompt=single_text_prompt_raw,
    decoder_prompt=single_tokens_prompt,
)
enc_dec_prompt2 = ExplicitEncoderDecoderPrompt(
    # Pass TextPrompt to encoder, and
    # pass decoder prompt string directly
    encoder_prompt=single_text_prompt,
    decoder_prompt=single_text_prompt_raw,
)
enc_dec_prompt3 = ExplicitEncoderDecoderPrompt(
    # Pass encoder prompt tokens directly, and
    # pass TextPrompt to decoder
    # 直接传递 encoder 提示 tokens 并传递 TextPrompt 给 decoder
    encoder_prompt=single_tokens_prompt,
    decoder_prompt=single_text_prompt,
)

# - Finally, here's a useful helper function for zipping encoder and
#   decoder prompts together into a list of ExplicitEncoderDecoderPrompt
#   instances
# - 最后，这里有一个有用的辅助函数，用于将 encoder 和 decoder 提示打包成
#   一系列 ExplicitEncoderDecoderPrompt 实例。
zipped_prompt_list = zip_enc_dec_prompts(
    ['An encoder prompt', 'Another encoder prompt'],
    ['A decoder prompt', 'Another decoder prompt'])

# - Let's put all of the above example prompts together into one list
#   which we will pass to the encoder/decoder LLM.
#   让我们将上述所有示例提示组合成一个列表，然后传递给 encoder/decoder LLM
prompts = [
    single_text_prompt_raw, single_text_prompt, single_tokens_prompt,
    enc_dec_prompt1, enc_dec_prompt2, enc_dec_prompt3
] + zipped_prompt_list

print(prompts)

# Create a sampling params object.
# 创建 sampling params 对象
sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    min_tokens=0,
    max_tokens=20,
)

# Generate output tokens from the prompts. The output is a list of
# RequestOutput objects that contain the prompt, generated
# text, and other information.
# 从提示中生成输出 tokens。输出是一个 RequestOutput 列表，包含提示、生成文本和其他信息

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
# 打印输出
for output in outputs:
    prompt = output.prompt
    encoder_prompt = output.encoder_prompt
    generated_text = output.outputs[0].text
    print(f"Encoder prompt: {encoder_prompt!r}, "
          f"Decoder prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")
```


