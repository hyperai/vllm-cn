---

title: 为 vLLM 做贡献

---

[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


感谢您有兴趣为 vLLM 做贡献！我们的社区向所有人开放，欢迎各种规模的贡献，无论大小。您可以通过以下方式为项目做出贡献：

* 发现并报告问题或错误

* 请求或添加对新模型的支持

* 建议或实现新功能

* 改进文档或编写操作指南


我们相信社区支持的力量，所以回答问题、提供 PR 审查和帮助他人同样是备受推崇且有益的贡献方式。


最后，支持我们最具影响力的方式之一就是让更多人了解 vLLM。你可以在自己的博客文章中提到它，并分享它是如何助力你那些出色项目的。如果你正在使用 vLLM，也可以在社交媒体上表达你的支持；或者，简单地为我们项目仓库点个赞，也是一种很棒的支持方式！

## 

## 许可证

请参阅 [LICENSE](https://github.com/vllm-project/vllm/blob/main/LICENSE)。

## 

## 开发指南

根据您要进行的开发类型（例如 Python、CUDA），您可以选择是否编译构建 vLLM。详细信息请参阅[从源码构建](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-from-source)文档。

## 

## 测试

```plain
pip install -r requirements/dev.txt


# Linting, formatting and static type checking
# 代码检查、格式化和静态类型检查
pre-commit install --hook-type pre-commit --hook-type commit-msg


# You can manually run pre-commit with
# 手动运行 pre-commit
pre-commit run --all-files


# Unit tests
# 单元测试
pytest tests/
```


>**注意**
>当前仓库尚未完全通过 `mypy` 检查。
## 

## 问题报告

如果您遇到错误或有功能请求，请先[搜索现有问题](https://github.com/vllm-project/vllm/issues?q=is%253Aissue)，看看是否已经有人报告过。如果未找到相关记录，请[提交新问题](https://github.com/vllm-project/vllm/issues/new/choose)，并提供尽可能多的相关信息。


**重要提示**

如果您发现安全漏洞，请遵循[此处](https://github.com/vllm-project/vllm/blob/main/SECURITY.md)的说明操作。


## 拉取请求与代码审查

感谢您对 vLLM 的贡献！在提交 PR 前，请确保 PR 符合以下标准，这将帮助 vLLM 保持代码质量并提高审查效率。

### 

### DCO 与签署提交

当向本项目贡献变更时，您必须同意 [DCO](https://github.com/vllm-project/vllm/blob/main/DCO)。提交必须包含 `Signed-off-by:` 标头以证明您同意 DCO 条款。

使用 `git commit -s` 会自动添加此标头。


### PR 标题与分类

只有特定类型的 PR 会被审查。PR 标题需使用适当前缀标明变更类型。请选择以下前缀之一：

* `[Bugfix]` 错误修复

* `[CI/Build]` 构建或持续集成改进

* `[Doc]` 文档修正与改进

* `[Model]` 新增模型或改进现有模型（标题需包含模型名称）

* `[Frontend]` 前端变更（如 OpenAI API 服务器、`LLM` 类等）

* `[Kernel]` 影响 CUDA 内核或其他计算内核的变更

* `[Core]` 核心逻辑变更（如 `LLMEngine`、`AsyncLLMEngine`、`Scheduler` 等）

* `[Hardware][Vendor]` 硬件特定变更（需包含厂商名称，如 `[Hardware][AMD]`）

* `[Misc]` 其他类别（请谨慎使用）


>**注意**
>如果 PR 涉及多个类别，请包含所有相关前缀。
### 

### 代码质量要求

PR 需满足以下代码质量标准：

* 遵循 [Google Python 风格指南](https://google.github.io/styleguide/pyguide.html) 和 [Google C++ 风格指南](https://google.github.io/styleguide/cppguide.html)

* 通过所有代码检查工具，请使用 `pre-commit` 格式化代码（新用户参考 [pre-commit 使用指南](https://pre-commit.com/#usage)）

* 代码需充分注释以确保后续贡献者易于理解

* 包含充足的单元测试和集成测试

* 如果 PR 修改了用户可见行为，请在 `docs/source/` 中添加文档说明

### 

### 内核开发指南

每个自定义内核都需要一个模式 (schema) 和一个或多个实现，以便在 PyTorch 中进行注册：

* 遵循 PyTorch 指南注册自定义操作：

   * [自定义 C++/CUDA 操作教程](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial)

   * [自定义操作手册](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU)

* 返回 `Tensor` 的自定义操作需要元函数（meta-functions），应在 Python 中实现以处理动态维度

* 使用 [torch.library.opcheck()](https://pytorch.org/docs/stable/library.html#torch.library.opcheck) 测试函数注册和元函数（示例见 `tests/kernels`）

* 修改现有操作的 C++ 签名时需同步更新 schema

* 如需新自定义类型，请参考 [PT2 自定义类支持文档](https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA)

### 

### 大型变更的注意事项

请尽量保持变更简洁。对于重大架构变更（>500 行，不含内核/数据/配置/测试），我们期望先有 GitHub issue（RFC）讨论技术设计和合理性。否则将标记为 `rfc-required` 并可能不予审查。


### 审查流程说明

vLLM 团队致力于建立透明的审查机制，我们希望让每位贡献者都能清楚地了解评审过程，避免大家感到困惑，但团队规模有限，需要优先处理部分更重要的 PR。以下是审查流程说明：

* PR 提交后，该 PR 会被分配给一位审查员。每位审查员会根据自己的专业领域和时间安排来选择处理 PR。

* PR分配之后，审查员每 2-3 天更新一次评审进度。如果 PR 在 7 天内还没有得到审查，请随时提醒审查者或者 vLLM 团队。

* 审查结束后如需进行修改，审查员将在 PR 上标记 `action-required`的标签，贡献者处理后可请求重新审查。

* 请及时回应所有评论，如果某个评论不清晰，或者你不同意某个建议，随时可以要求澄清或者讨论这个建议。

* 请注意，由于计算资源限制，部分 CI 检查可能不会执行，`ready` 标签表示 PR 可合并或需要完整 CI 运行。

## 

## 致谢

最后，感谢您阅读这些指南并参与 vLLM 的贡献。您的所有努力都将帮助 vLLM 成为更优秀的工具和社区！

