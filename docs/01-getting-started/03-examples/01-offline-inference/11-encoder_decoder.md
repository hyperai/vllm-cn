---
title: Encoder Decoder
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/encoder_decoder.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/encoder_decoder.py)

```python
# SPDX-License-Identifier: Apache-2.0
'''
Demonstrate prompting of text-to-text
encoder/decoder models, specifically BART
'''

from vllm import LLM, SamplingParams
from vllm.inputs import (ExplicitEncoderDecoderPrompt, TextPrompt,
                         TokensPrompt, zip_enc_dec_prompts)

dtype = "float"

# Create a BART encoder/decoder model instance
# 创建一个 BART 编码器/解码器模型实例
llm = LLM(
    model="facebook/bart-large-cnn",
    dtype=dtype,
)

# Get BART tokenizer
# 获取 BART tokenizer
tokenizer = llm.llm_engine.get_tokenizer_group()

# Test prompts
#
# This section shows all of the valid ways to prompt an
# encoder/decoder model.
#
# - Helpers for building prompts
# 测试提示
#
# 本节显示了提示的所有有效方法
# 编码器/解码器模型。
#
# - 构建提示的帮助
text_prompt_raw = "Hello, my name is"
text_prompt = TextPrompt(prompt="The president of the United States is")
tokens_prompt = TokensPrompt(prompt_token_ids=tokenizer.encode(
    prompt="The capital of France is"))
# - Pass a single prompt to encoder/decoder model
#   (implicitly encoder input prompt);
#   decoder input prompt is assumed to be None
#  - 将单个提示传递到编码器/解码器模型
#  (隐式编码器输入提示) ;
# 假定解码器输入提示符为 None

single_text_prompt_raw = text_prompt_raw  # Pass a string directly
single_text_prompt = text_prompt  # Pass a TextPrompt
single_tokens_prompt = tokens_prompt  # Pass a TokensPrompt

# - Pass explicit encoder and decoder input prompts within one data structure.
#   Encoder and decoder prompts can both independently be text or tokens, with
#   no requirement that they be the same prompt type. Some example prompt-type
#   combinations are shown below, note that these are not exhaustive.
#  - 在一个数据结构中传递显式编码器和解码器输入提示。
#    编码器和解码器提示可以独立地是文本或 token ，
#    不需要它们是相同的提示类型。一些示例及时类型
#    组合如下所示，请注意，这些并不详尽。

enc_dec_prompt1 = ExplicitEncoderDecoderPrompt(
    # Pass encoder prompt string directly, &
    # pass decoder prompt tokens
    # 直接传递编码器提示字符串，
    # 并传递解码器提示 token
    encoder_prompt=single_text_prompt_raw,
    decoder_prompt=single_tokens_prompt,
)
enc_dec_prompt2 = ExplicitEncoderDecoderPrompt(
    # Pass TextPrompt to encoder, and
    # pass decoder prompt string directly
    # 将 TextPrompt 传递给 encoder
    # 直接传递 decoder prompt 字符串
    encoder_prompt=single_text_prompt,
    decoder_prompt=single_text_prompt_raw,
)
enc_dec_prompt3 = ExplicitEncoderDecoderPrompt(
    # Pass encoder prompt tokens directly, and
    # pass TextPrompt to decoder
    # 通过直接传递编码器提示 token ，然后
    # 传递 TextPrompt 到解码器
    encoder_prompt=single_tokens_prompt,
    decoder_prompt=single_text_prompt,
)

# - Finally, here's a useful helper function for zipping encoder and
#   decoder prompts together into a list of ExplicitEncoderDecoderPrompt
#   instances
#  - 最后，这是用于 zipping 编码器的有用的助手功能，
#   解码器提示一起进入 explicitencoderdecoderprompt 的列表实例
zipped_prompt_list = zip_enc_dec_prompts(
    ['An encoder prompt', 'Another encoder prompt'],
    ['A decoder prompt', 'Another decoder prompt'])

# - Let's put all of the above example prompts together into one list
#   which we will pass to the encoder/decoder LLM.
#  - 让我们将上述所有示例提示放在一个列表中
#    我们将传递给编码器/解码器 LLM。
prompts = [
    single_text_prompt_raw, single_text_prompt, single_tokens_prompt,
    enc_dec_prompt1, enc_dec_prompt2, enc_dec_prompt3
] + zipped_prompt_list

print(prompts)

# Create a sampling params object.
# 创建一个采样参数对象。
sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    min_tokens=0,
    max_tokens=20,
)

# Generate output tokens from the prompts. The output is a list of
# RequestOutput objects that contain the prompt, generated
# text, and other information.
# 从提示中生成输出 token 。
# 输出是包含提示的对象，生成了文本和其他信息。
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
# 打印输出。
for output in outputs:
    prompt = output.prompt
    encoder_prompt = output.encoder_prompt
    generated_text = output.outputs[0].text
    print(f"Encoder prompt: {encoder_prompt!r}, "
          f"Decoder prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")

```
