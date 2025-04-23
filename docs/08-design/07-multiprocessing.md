---
title: Python 多进程
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

## 调试

有关已知问题及其解决方法的信息，请参阅[故障排除](https://docs.vllm.ai/en/latest/getting_started/troubleshooting.html#troubleshooting-python-multiprocessing)页面。

## 介绍

> **重要\*\***提示\*\*
> 源代码引用的是 2024 年 12 月撰写时的代码状态。

在 vLLM 中使用 Python 多进程的复杂性在于：

- vLLM 作为库使用，无法控制使用 vLLM 的代码
- 多进程方法与 vLLM 依赖项之间存在不同程度的不兼容性

本文档描述了 vLLM 如何应对这些难题。

## 多进程方法

[Python 多进程方法](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)包括：

- `spawn` - 生成一个新的 Python 进程。这将是 Python 3.14 的默认方法。
- `fork` - 使用 `os.fork()` 来 fork Python 解释器。这是 Python 3.14 之前版本的默认方法。
- `forkserver` - 生成一个服务器进程，该进程将在请求时 fork 一个新进程。

### 权衡

`fork` 是最快的方法，但与使用线程的依赖项不兼容。

`spawn` 与依赖项的兼容性更好，但在 vLLM 作为库使用时可能会出现问题。如果使用代码没有使用 `__main__` 保护（`if __name__ == "__main__":`），当 vLLM 生成新进程时，代码将被无意中重新执行。这可能导致无限递归等问题。

`forkserver` 将生成一个新的服务器进程，该进程将按需 fork 新进程。不幸的是，当 vLLM 作为库使用时，这与 `spawn` 有相同的问题。服务器进程是作为生成的新进程创建的，它将重新执行未受 `__main__` 保护的代码。

对于 `spawn` 和 `forkserver`，进程不能依赖于继承任何全局状态，就像 `fork` 那样。

## 与依赖项的兼容性

多个 vLLM 依赖项表明它们更倾向于或要求使用 `spawn`：

- [https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing](https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing)
- [https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors](https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors)
- [https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders](https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders)

更准确地说，已知在初始化这些依赖项后使用 `fork` 会存在问题。

## 当前状态（v0）

环境变量 `VLLM_WORKER_MULTIPROC_METHOD` 可用于控制 vLLM 使用的方法。当前默认值为 `fork`。

- [vllm-project/vllm](https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L339-L342)

当我们根据使用了 `vllm` 命令而知道我们拥有该进程时，我们使用 `spawn`，因为它拥有最广的兼容性。

- [vllm-project/vllm](https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/scripts.py#L123-L140)

`multiproc_xpu_executor` 强制使用 `spawn`。

- [vllm-project/vllm](https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/executor/multiproc_xpu_executor.py#L14-L18)

还有其他一些地方硬编码了 `spawn` 的使用：

- [vllm-project/vllm](https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/distributed/device_communicators/custom_all_reduce_utils.py#L135)
- [vllm-project/vllm](https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/entrypoints/openai/api_server.py#L184)

相关 PR：

- [Pull Request #8823](https://github.com/vllm-project/vllm/pull/8823#)

## v1 中的先前状态

有一个环境变量 `VLLM_ENABLE_V1_MULTIPROCESSING` 用于控制 v1 引擎核心中是否使用多进程。默认情况下是关闭的。

- [vllm-project/vllm](https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L452-L454)

当启用时，v1 `LLMEngine` 将创建一个新进程来运行引擎核心。

- [vllm-project/vllm](https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L93-L95)
- [vllm-project/vllm](https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L70-L77)
- [vllm-project/vllm](https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/core_client.py#L44-L45)

由于上述所有原因——与依赖项和使用 vLLM 作为库的代码的兼容性，默认情况下它是关闭的。

### v1 中的更改

Python 的 `multiprocessing` 并没有一个简单的解决方案可以在所有地方都适用。作为第一步，我们可以将 v1 调整到一个状态，使其“尽力而为”地选择多进程方法以最大化兼容性。

- 默认使用 `fork`。
- 当我们知道我们控制主进程时（执行了 `vllm`），使用 `spawn`。
- 如果我们检测到 `cuda` 之前已经初始化，则强制使用 `spawn` 并发出警告。我们知道 `fork` 会失败，所以这是我们能做的最好的事情。

在这种情况下，已知仍然会失败的情况是使用 vLLM 作为库的代码在调用 vLLM 之前初始化了 `cuda`。我们发出的警告应指示用户添加 `__main__` 保护或禁用多进程。

如果发生这种已知的失败情况，用户将看到两条消息，解释发生了什么。首先，vLLM 的日志消息：

```go
WARNING 12-11 14:50:37 multiproc_worker_utils.py:281] CUDA was previously
    initialized. We must use the `spawn` multiprocessing start method. Setting
    VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See
    https://docs.vllm.ai/en/latest/getting_started/debugging.html#python-multiprocessing
    for more information.
```

其次，Python 本身将引发一个带有详细解释的异常：

```go
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.


        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:


            if __name__ == '__main__':
                freeze_support()
                ...


        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.


        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
```

## 考虑的替代方案

### 检测是否存在 `__main__` 保护

有人建议，如果我们能够检测到使用 vLLM 作为库的代码是否存在 `__main__` 保护，我们可以表现得更好。这篇 [stackoverflow 帖子](https://stackoverflow.com/questions/77220442/multiprocessing-pool-in-a-python-class-without-name-main-guard) 来自一个面临相同问题的库作者。

可以检测我们是否在原始的 `__main__` 进程中，或者是在后续生成的进程中。然而，检测代码中是否存在 `__main__` 保护似乎并不简单。

此选项已被认为不切实际而放弃。

### 使用 `forkserver`

乍一看，`forkserver` 似乎是一个很好的解决方案。然而，它的工作方式在 vLLM 作为库使用时与 `spawn` 存在相同的挑战。

### 始终强制使用 `spawn`

一种清理方法是始终强制使用 `spawn`，并记录在使用 vLLM 作为库时需要 `__main__` 保护。不幸的是，这将破坏现有代码，并使 vLLM 更难使用，违背了使 `LLM` 类尽可能易于使用的愿望。

我们不会将这个问题推给用户，而是保留复杂性，尽力使事情正常运行。

## 未来工作

- 我们未来可能会考虑采用不同的工作进程管理方法，以绕过这些挑战。
- 我们可以实现类似于 `forkserver` 的东西，但让进程管理器成为我们最初通过运行自己的子进程和自定义入口点来启动的东西（启动一个 `vllm-manager` 进程）。
- 我们可以探索其他可能更适合我们需求的库。可以考虑的示例：
- [joblib/loky](https://github.com/joblib/loky)
