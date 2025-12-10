---
title: 使用 vLLM 对 Qwen2.5 推理
---

[<font face="黑体" color=DodgerBlue size=5>在线运行此教程</font>](https://app.hyper.ai/console/public/tutorials/IDcCqWcOKRD?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

本教程演示了如何在短短 5 小时内推理 LLM 3B 模型。

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

该教程基于 app.hyper.ai 云平台操作，该平台已完成 vllm==0.11.0 的安装。如果您在平台上操作，请跳过此步骤。如果您在本地部署，请按照以下步骤进行安装。

```
!pip install -U vllm -i https://pypi.org/simple
```

## 2. 使用 vLLM 加载 Qwen 量化模型

```
import os, math, numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["VLLM_USE_V1"]="0"
```

```
# 自定义逻辑处理器
from typing import Optional, List
import torch
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality
)
class ForceTokenLogitsProcessor(LogitsProcessor):
    """
    vLLM 风格 logits processor:
    - 可以在批处理中针对每个请求指定 target_token 或 allowed_ids
    - 在 apply() 中屏蔽其他 token 或提升 allowed token 权重
    """

    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool,
                 allowed_ids: Optional[List[int]] = None):
        """
        allowed_ids: 全局允许 token id 列表
        """
        self.req_info: dict[int, int] = {}  # 每个请求 target token
        self.allowed_ids = allowed_ids if allowed_ids is not None else []

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return

        # 处理新增请求
        for index, params, _, _ in batch_update.added:
            assert params is not None
            # 优先使用 SamplingParams.extra_args 中的 target_token
            if params.extra_args and (target_token := params.extra_args.get("target_token")):
                self.req_info[index] = target_token
            else:
                self.req_info.pop(index, None)

        if self.req_info:
            # 处理移除请求
            for index in batch_update.removed:
                self.req_info.pop(index, None)

            # 处理移动请求
            for adx, bdx, direct in batch_update.moved:
                a_val = self.req_info.pop(adx, None)
                b_val = self.req_info.pop(bdx, None)
                if a_val is not None:
                    self.req_info[bdx] = a_val
                if direct == MoveDirectionality.SWAP and b_val is not None:
                    self.req_info[adx] = b_val

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: shape (batch_size, vocab_size)
        逻辑:
        1. 如果 req_info 有 target_token，则只保留该 token
        2. 对 allowed_ids 全局 token，增加权重 +100
        """
        if not self.req_info and not self.allowed_ids:
            return logits

        # 1️⃣ 针对每个请求保留 target_token
        if self.req_info:
            rows = torch.tensor(list(self.req_info.keys()), dtype=torch.long, device=logits.device)
            cols = torch.tensor(list(self.req_info.values()), dtype=torch.long, device=logits.device)
            values_to_keep = logits[rows, cols].clone()
            logits[rows] = float('-inf')
            logits[rows, cols] = values_to_keep

        # 2️⃣ 对全局 allowed_ids token 提升权重
        if self.allowed_ids:
            idx = torch.tensor(self.allowed_ids, dtype=torch.long, device=logits.device)
            logits[:, idx] += 100

        return logits
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
    max_model_len=8192,
    # 新增自定义的逻辑处理器
    logits_processors = [ForceTokenLogitsProcessor]
    # num_scheduler_steps=1,
    #distributed_executor_backend="ray",
)
tokenizer = llm.get_tokenizer()
```

```
import vllm
print(f"当前加载的 vLLM 版本: {vllm.__version__}")
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
from abc import ABC, abstractmethod
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
choices = ["A", "B", "tie"]
KEEP = [tokenizer.encode(x, add_special_tokens=False)[0] for x in choices]
print("Allowed token ids:", KEEP)
print(type(KEEP))
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
    a = str(row.prompt) if row.prompt is not None else ""
    b = str(row.response_a) if row.response_a is not None else ""
    c = str(row.response_b) if row.response_b is not None else ""

    prompt = f"{SS}PROMPT: {a}\n\n{SS}RESPONSE A: {b}\n\n{SS}RESPONSE B: {c}\n\n"
    formatted_sample = sys_prompt + "\n\n" + prompt
    all_prompts.append(formatted_sample)
```

```
print(type(all_prompts))
```

```
all_prompts[0]
```

```
response = llm.generate(
    ["aaa","vvvv","aaaa"],
    vllm.SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0,  # randomness of the sampling
        seed=777, # Seed for reprodicibility
        skip_special_tokens=True,  # Whether to skip special tokens in the output.
        max_tokens=1,  # Maximum number of tokens to generate per output sequence.
        # 删除原来的，不支持了为单独的每个req设置logits_processors,需要在启动模型时自行设置中设置，然后使用SamplingParams单独设置
        #logits_processors=logits_processors,
        # logprobs = 5,
    ),
    use_tqdm = True
)
response
```


## 5. Infer 测试

现在使用 vLLM 推断测试。我们要求 vLLM 输出第一个 Token 中被认为预测的前 5 个 Token 的概率。并将预测限制为 1 个 token，以提高推理速度。

根据推断 128 个训练样本所需的速度，可以推断出 25000 个测试样本需要多长时间。

```
from time import time
start = time()

logits_processors = [DigitLogitsProcessor(tokenizer)]

results = []
batch_size = 4
for i in range(0, len(all_prompts), batch_size):
    batch_prompts = all_prompts[i:i+batch_size]
    response = llm.generate(
        batch_prompts,
        vllm.SamplingParams(
            n=1,  # Number of output sequences to return for each prompt.
            top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
            temperature=0,  # randomness of the sampling
            seed=777, # Seed for reprodicibility
            # skip_special_tokens=True,  # Whether to skip special tokens in the output.
            max_tokens=1,  # Maximum number of tokens to generate per output sequence.
            # 删除原来的，不支持了为单独的每个req设置logits_processors,需要在启动模型时自行设置中设置，然后使用SamplingParams单独设置
            #logits_processors=logits_processors,
            # logprobs = 5,
            extra_args = {"target_token": KEEP}
        ),
        use_tqdm = True
    )
    results.append(response)
responses = [item for batch in results for item in batch]
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
