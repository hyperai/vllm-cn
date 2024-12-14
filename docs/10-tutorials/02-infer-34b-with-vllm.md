---
title: 使用 vLLM 对 Qwen2.5 推理
---

[在线运行此教程](https://openbayes.com/console/hyperai-tutorials/containers/wmGLO8o5IiV)

该教程详细展示了如何完成一个 3B 参数的大语言模型的推理任务，包括模型的加载、数据的准备、推理过程的优化，以及结果的提取和评估。

## 目录
- [1.安装 vllm](#1.安装vllm)
- [2.使用vLLM加载Qwen量化模型](#2.使用vLLM加载Qwen量化模型)
- [3.加载测试数据](#3.加载测试数据)
- [4.提示工程](#4.提示工程)
- [5.Infer测试](#5.Infer测试)
- [6.提取推理概率](#6.提取推理概率)
- [7.创建提交CSV](#7.创建提交CSV)
- [8.计算CV分数](#8.计算CV分数)

## 1. 安装 vLLM

该教程基于 OpenBayes 云平台操作，该平台已完成 vllm==0.5.4 的安装。如果您在平台上操作，请跳过此步骤。如果您在本地部署，请按照以下步骤进行安装。

安装 vLLM 非常简单：

```
pip install vllm
```

本教程已经安装 vllm==0.5.4，如需更新 vllm 请取消下行注释

```
!pip install -U vllm
```

## 2. 使用 vLLM 加载 Qwen 量化模型
```
import os, math, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0"
```

```
# 我们将在此处加载并使用 Qwen2.5-3B-Instruct-AWQ

import vllm

llm = vllm.LLM(
    "/input0/Qwen2.5-3B-Instruct-AWQ",
    quantization="awq",
    tensor_parallel_size=1, 
    gpu_memory_utilization=0.95, 
    trust_remote_code=True,
    dtype="half", 
    enforce_eager=True,
    max_model_len=512,
    #distributed_executor_backend="ray",
)
tokenizer = llm.get_tokenizer()
```

## 3. 加载测试数据

在提交期间，我们加载 128 行 train 来计算 CV 分数，加载测试数据。

```
import pandas as pd
VALIDATE = 128

test = pd.read_csv("./lmsys-chatbot-arena/test.csv") 
if len(test)==3:
    test = pd.read_csv("./lmsys-chatbot-arena/train.csv")
    test = test.iloc[:VALIDATE]
print( test.shape )
test.head(1)
```

## 4. 提示工程
如果我们想提交零次 LLM，我们需要尝试不同的系统提示来提高 CV 分数。如果我们对模型进行微调，那么系统就不那么重要了，因为无论我们使用哪个系统提示，模型都会从目标中学习该做什么。

我们使用 logits 处理器强制模型输出我们感兴趣的 3 个标记。

```
from typing import Any, Dict, List
from transformers import LogitsProcessor
import torch

choices = ["A","B","tie"]

KEEP = []
for x in choices:
    c = tokenizer.encode(x,add_special_tokens=False)[0]
    KEEP.append(c)
print(f"Force predictions to be tokens {KEEP} which are {choices}.")

class DigitLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.allowed_ids = KEEP
        
    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        scores[self.allowed_ids] += 100
        return scores
```

```
sys_prompt = """Please read the following prompt and two responses. Determine which response is better.
If the responses are relatively the same, respond with 'tie'. Otherwise respond with 'A' or 'B' to indicate which is better."""
```

```
SS = "#"*25 + "\n"
```

```
all_prompts = []
for index,row in test.iterrows():
    
    a = " ".join(eval(row.prompt, {"null": ""}))
    b = " ".join(eval(row.response_a, {"null": ""}))
    c = " ".join(eval(row.response_b, {"null": ""}))
    
    prompt = f"{SS}PROMPT: "+a+f"\n\n{SS}RESPONSE A: "+b+f"\n\n{SS}RESPONSE B: "+c+"\n\n"
    
    formatted_sample = sys_prompt + "\n\n" + prompt
    
    all_prompts.append( formatted_sample )
```

## 5. Infer 测试
现在使用 vLLM 推断测试。我们要求 vLLM 输出第一个 Token 中被认为预测的前 5 个 Token 的概率。并将预测限制为 1 个 token，以提高推理速度。

根据推断 128 个训练样本所需的速度，可以推断出 25000 个测试样本需要多长时间。

```
from time import time
start = time()

logits_processors = [DigitLogitsProcessor(tokenizer)]
responses = llm.generate(
    all_prompts,
    vllm.SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0,  # randomness of the sampling
        seed=777, # Seed for reprodicibility
        skip_special_tokens=True,  # Whether to skip special tokens in the output.
        max_tokens=1,  # Maximum number of tokens to generate per output sequence.
        logits_processors=logits_processors,
        logprobs = 5
    ),
    use_tqdm = True
)

end = time()
elapsed = (end-start)/60. #minutes
print(f"Inference of {VALIDATE} samples took {elapsed} minutes!")
```

```
submit = 25_000 / 128 * elapsed / 60
print(f"Submit will take {submit} hours")
```

## 6. 提取推理概率
```
results = []
errors = 0

for i,response in enumerate(responses):
    try:
        x = response.outputs[0].logprobs[0]
        logprobs = []
        for k in KEEP:
            if k in x:
                logprobs.append( math.exp(x[k].logprob) )
            else:
                logprobs.append( 0 )
                print(f"bad logits {i}")
        logprobs = np.array( logprobs )
        logprobs /= logprobs.sum()
        results.append( logprobs )
    except:
        #print(f"error {i}")
        results.append( np.array([1/3., 1/3., 1/3.]) )
        errors += 1
        
print(f"There were {errors} inference errors out of {i+1} inferences")
results = np.vstack(results)
```

## 7. 创建提交 CSV
```
sub = pd.read_csv("./lmsys-chatbot-arena/sample_submission.csv")

if len(test)!=VALIDATE:
    sub[["winner_model_a","winner_model_b","winner_tie"]] = results
    
sub.to_csv("submission.csv",index=False)
sub.head()
```

## 8. 计算 CV 分数
```
if len(test)==VALIDATE:
    true = test[['winner_model_a','winner_model_b','winner_tie']].values
    print(true.shape)
```

```
if len(test)==VALIDATE:
    from sklearn.metrics import log_loss
    print(f"CV loglosss is {log_loss(true,results)}" )
```
