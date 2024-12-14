---
title: 使用 vLLM 加载 AWQ 量化 Qwen2.5-3B-Instruct 进行少样本学习 (Few shot)
---

[在线运行此教程](https://openbayes.com/console/hyperai-tutorials/containers/1HFARLMLJXL)

该教程为在 RTX4090 上该教程为使用 vLLM 加载 Qwen2.5-3B-Instruct-AWQ 模型进行少样本学习。

- 对于每个测试问题，我们使用训练数据检索一组「支持」它的类似问题。
    - 考虑「construct」和「subject」等内容
- 使用一组类似的问题，我们创建了一个可以馈送到我们的模型的对话
    - 在对话中使用最近支持的 chat（） 功能
    - 生成温度略高的 n 个响应，以创建不同的输出

- 对于每个问题/答案对，我们现在有 n 个推断的误解，对于每个误解，我们使用 BGE 嵌入检索前 25 个误解。
- 对于每个问题/答案对的 n 个推断错误中的每一个的 25 个最接近的误解，现在可以使用 Borda Ranking 进行组合，这有点像最简单的集成形式。

## 目录
- [1. 导入相关的库](#1.导入相关的库)
- [2. 加载数据](#2.加载数据)
- [3. 使用 vLLM 启动 Qwen2.5-3B-Instruct-AWQ](#3.使用vLLM启动Qwen2.5-3B-Instruct-AWQ)
- [4. 后处理数据](#4.后处理数据)
- [5. 辅助函数](#5.辅助函数)
- [6. 使用llm.chat](#6.使用llm.chat)
- [7. 找到最相似的误解](#7.找到最相似的误解)
- [8. 提交](#8.提交)

## 1. 导入相关的库

```
import os
import gc
import ctypes
import numpy as np
import pandas as pd

from random import sample
from tqdm.auto import tqdm
from eedi_metrics import mapk, apk
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel
```


```
os.environ["CUDA_VISIBLE_DEVICES"]   = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()
```

## 2. 加载数据

```
k = 3

train_eval = True
n_train_eval_rows = 100

comp_dir  = './eedi-mining-misconceptions-in-mathematics'

llm_model_pth   = '/input0/Qwen2.5-3B-Instruct-AWQ'

embed_model_pth = '/input0/nomic-embed-text-v1.5'


if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    train_eval = False
```

```
if train_eval:
    test       = pd.read_csv(f'{comp_dir}/train.csv').sample(n_train_eval_rows, random_state=3)
    test       = test.sort_values(['QuestionId'], ascending=True).reset_index(drop=True)
else:
    test       = pd.read_csv(f'{comp_dir}/test.csv')

train          = pd.read_csv(f'{comp_dir}/train.csv')
sample_sub     = pd.read_csv(f'{comp_dir}/sample_submission.csv')
misconceptions = pd.read_csv(f'{comp_dir}/misconception_mapping.csv')

len(train), len(test), len(misconceptions)
```

## 3. 使用 vLLM 启动 Qwen2.5-3B-Instruct-AWQ

如果出现 OOM 错误，将 max_num_seqs减少到 4 或 8 甚至 1 可能会有所帮助（默认值为 256）。

```
llm = LLM(
    llm_model_pth,
    trust_remote_code=True,
    dtype="half", max_model_len=4096,
    tensor_parallel_size=1, gpu_memory_utilization=0.95, 
)

tokenizer = llm.get_tokenizer()
```

## 4. 后处理数据

```
answer_cols         = ["AnswerAText", "AnswerBText", "AnswerCText", "AnswerDText"]
misconception_cols  = ["MisconceptionAId", "MisconceptionBId", "MisconceptionCId", "MisconceptionDId"]

keep_cols           = ["QuestionId", "CorrectAnswer", "ConstructName", "SubjectName", "QuestionText" ]

def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    
    # Melt the answer columns
    answers_df = pd.melt(
        id_vars=keep_cols,
        frame=df[keep_cols + answer_cols],
        var_name='Answer', value_name='Value'
    ).sort_values(["QuestionId", "Answer"]).reset_index(drop=True)
    if misconception_cols[0] not in df.columns:  # If test set
        return answers_df
        
    # Melt the misconception columns
    misconceptions_df = pd.melt(
        id_vars=keep_cols,
        frame=df[keep_cols + misconception_cols],
        var_name='Misconception', value_name='MisconceptionId'
    ).sort_values(["QuestionId", "Misconception"]).reset_index(drop=True)

    answers_df[['Misconception', 'MisconceptionId']] = misconceptions_df[['Misconception', 'MisconceptionId']]
    
    return answers_df
test  = wide_to_long(test)
train = wide_to_long(train)

test['AnswerId']  = test.Answer.str.replace('Answer', '').str.replace('Text', '')
train['AnswerId'] = train.Answer.str.replace('Answer', '').str.replace('Text', '')

train = pd.merge(train, misconceptions, on='MisconceptionId', how='left')
if train_eval:
    test = pd.merge(test, misconceptions, on='MisconceptionId', how='left')
```


```
train.head(3)
```

```
test.head(3)
```


## 5. 辅助函数
### 在给定 subject 和 construct 的情况下获取最相似的 question_ids'

以下函数首先通过检查结构top_k subject 相似的问题来返回问题 ID 的数量。

如果这没有达到top_k，则选择具有相似主题或结构的问题。如果我们仍然缺少问题 ID'，我们会为剩余的 top_k 选择随机问题。


```
def get_topk_similar_rows(question_id: int, construct: str, subject: str, top_k: int) -> list[int]:
    """ Gets the top n ids of questions that most similar to the given construct and subject """
    
    # Rows with similar construct and subject
    similar_cs_rows = train[(train.ConstructName == construct) & (train.SubjectName == subject)]
    similar_cs_qids = list(set(similar_cs_rows.QuestionId.values.tolist()))
    
    if train_eval and question_id in similar_cs_qids:
        similar_cs_qids.remove(question_id)
        
    if len(similar_cs_qids) >= top_k:
        k_similar_cs_qids = sample(similar_cs_qids, top_k)
        return k_similar_cs_qids
    # Rows with similar construct or subject for remainder of top_k
    similar_s_rows = train[(train.ConstructName != construct) & (train.SubjectName == subject)]
    similar_c_rows = train[(train.ConstructName == construct) & (train.SubjectName != subject)]
    similar_c_or_s_qids = list(set(similar_s_rows.QuestionId.values.tolist() + similar_c_rows.QuestionId.values.tolist()))
    
    if train_eval and question_id in similar_c_or_s_qids:
        similar_c_or_s_qids.remove(question_id)
    
    if len(similar_c_or_s_qids) >= top_k - len(similar_cs_qids):
        n_similar_c_or_s_qids = sample(similar_c_or_s_qids, top_k - len(similar_cs_qids))
        return similar_cs_qids + n_similar_c_or_s_qids
        # Random rows for remainder of top_k
    not_so_similar_rows = train[(train.ConstructName != construct) & (train.SubjectName != subject)]
    not_so_similar_rows_qids = list(set(not_so_similar_rows.QuestionId.values.tolist()))
    
    if train_eval and question_id in not_so_similar_rows_qids:
        not_so_similar_rows_qids.remove(question_id)
    
    n_not_so_similar_rows_qids = sample(not_so_similar_rows_qids, top_k - len(similar_c_or_s_qids))
    return similar_c_or_s_qids + n_not_so_similar_rows_qids
```

### 获取每个问题的聊天对话


```
def get_conversation_msgs(question, correct_ans, incorrect_ans, misconception):
    msgs = [
        {'role': 'user',      'content': 'Question: ' + question.strip()},
        {'role': 'assistant', 'content': 'Provide me with the correct answer for a baseline.'},
        {'role': 'user',      'content': 'Correct Answer: ' + correct_ans.strip()},
        {'role': 'assistant', 'content': 'Now provide the incorrect answer and I will anaylze the difference to infer the misconception.'},
        {'role': 'user',      'content': 'Incorrect Answer: ' + incorrect_ans.strip()},
    ]
    
    if misconception is not None:
        msgs += [{'role': 'assistant', 'content': 'Misconception for incorrect answer: ' + misconception}]
        
    return msgs
```

## 6. 使用 llm.chat
注意：llm（） 是最近才推出的，仅在后续版本中可用

我们生成 n 个输出，使用更高的温度来创建输出的多样化表示，然后可以稍后用于对结果进行排名。

```
sampling_params = SamplingParams(
    n=10,                     # 对于每个提示，返回的输出序列数量。Number of output sequences to return for each prompt.
    # top_p=0.5,               # 控制考虑的顶部标记的累积概率的浮点数。Float that controls the cumulative probability of the top tokens to consider.
    temperature=0.7,          # 采样的随机性。randomness of the sampling
    seed=1,                   # 
用于可重复性的种子。Seed for reprodicibility
    skip_special_tokens=True, # 是否在输出中跳过特殊标记。Whether to skip special tokens in the output.
    max_tokens=64,            # 每个输出序列生成的最大标记数。Maximum number of tokens to generate per output sequence.
    stop=['\n\n', '. '],      # 当生成的文本中包含这些字符串时，将停止生成过程的字符串列表。List of strings that stop the generation when they are generated.
)
```

```
submission = []
for idx, row in tqdm(test.iterrows(), total=len(test)):
    
    if idx % 50:
        clean_memory()
        clean_memory()
    
    if row['CorrectAnswer'] == row['AnswerId']: continue
    if train_eval and not row['MisconceptionId'] >= 0: continue
        
    context_qids   = get_topk_similar_rows(row['QuestionId'], row['ConstructName'], row['SubjectName'], k)
    correct_answer = test[(test.QuestionId == row['QuestionId']) & (test.CorrectAnswer == test.AnswerId)].Value.tolist()[0]
    
    messages = []
    for qid in context_qids:
        correct_option = train[(train.QuestionId == qid) & (train.CorrectAnswer == train.AnswerId)]
        incorrect_options = train[(train.QuestionId == qid) & (train.CorrectAnswer != train.AnswerId)]
        for idx, incorrect_option in incorrect_options.iterrows():
            if type(incorrect_option['MisconceptionName']) == str: # Filter out NaNs
                messages += get_conversation_msgs(
                    question = correct_option.QuestionText.tolist()[0],
                    correct_ans = correct_option.Value.tolist()[0],
                    incorrect_ans = incorrect_option['Value'],
                    misconception = incorrect_option['MisconceptionName'],
                )
                
    # 对话对于错误答案以获取误解的原因。Coversation for Incorrect answer to get misconception for
    messages += get_conversation_msgs(
        question = row['QuestionText'],
        correct_ans = correct_answer,
        incorrect_ans = row['Value'],
        misconception = None,
    )
    
    output = llm.chat(messages, sampling_params, use_tqdm=False)
    inferred_misconceptions = [imc.text.split(':')[-1].strip() for imc in output[0].outputs]
    if not train_eval:
        submission.append([f"{row['QuestionId']}_{row['AnswerId']}", inferred_misconceptions])
    else:
        submission.append([
            f"{row['QuestionId']}_{row['AnswerId']}", 
            inferred_misconceptions, 
            context_qids,
            [int(row['MisconceptionId'])],
            row['MisconceptionName']
        ])
submission = pd.DataFrame(submission, columns=['QuestionId_Answer', 'InferredMisconception', 'TopKQuestionIDs', 
                                               'MisconceptionIdGT', 'MisconceptionNameGT'][:len(submission[0])])

len(submission)
```

```
submission.head()
```

## 7. 找到最相似的误解

删除模型并清理内存以加载嵌入模型
```
del llm

clean_memory(deep=True)
clean_memory(deep=True)
```

```
tokenizer   = AutoTokenizer.from_pretrained(embed_model_pth, trust_remote_code=True)
embed_model = AutoModel.from_pretrained(embed_model_pth, trust_remote_code=True).to("cuda:0")
```

```
def generate_embeddings(texts, batch_size=8):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=1024).to('cuda:0')
        with torch.no_grad():
            outputs = embed_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
        
    return np.concatenate(all_embeddings, axis=0)
```

```
all_ctx_vector  = generate_embeddings(list(misconceptions.MisconceptionName.values))

all_ctx_vector.shape
```

```
n_results = []

for results in tqdm(pd.DataFrame(submission.InferredMisconception.values.tolist()).T.values):
    all_text_vector = generate_embeddings(list(results))
    cosine_similarities = cosine_similarity(all_text_vector, all_ctx_vector)
    test_sorted_indices = np.argsort(-cosine_similarities, axis=1)
    n_results.append(test_sorted_indices)

n_results = np.array(n_results)
n_results.shape
```

```
n_results = np.transpose(n_results, (1, 0, 2))
n_results.shape
```

### 合并每个问题的每个生成输出的排名
Borda count 是一种非常简单的排名机制
```
def borda_count(rankings):
    scores = {}
    num_elements = len(next(iter(rankings)))
    
    for model_ranking in rankings:
        for idx, item in enumerate(model_ranking):
            points = num_elements - idx
            scores[item] = scores.get(item, 0) + points
            
    # 根据总分排序误解。Sort the misconceptions based on total points
    final_ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked_results = [r for r, score in final_ranking]
    return ranked_results

# 计算最终排名。Compute the final ranking
final_rankings = np.array([borda_count(result) for result in n_results])

final_rankings.shape
```

```
submission['MisconceptionId'] = final_rankings[:, :25].tolist()
```


## 8. 提交
```
if train_eval:
    submission['apk@25'] = submission.apply(lambda row: apk(row['MisconceptionIdGT'], row['MisconceptionId']), axis=1)
    submission.to_csv('submission_debug.csv', index=False)
    
    print(submission['apk@25'].mean())
```

```
submission["MisconceptionId"] = submission["MisconceptionId"].apply(lambda x: ' '.join(map(str, x)))
submission[['QuestionId_Answer', 'MisconceptionId']].to_csv('submission.csv', index=False)
```

```
submission.head(25)
```
