# vLLM 中文文档

[中文文档](https://vllm.hyper.ai/)｜[了解更多](https://hyper.ai/)

vLLM 是一个高性能的大型语言模型推理引擎，采用创新的内存管理和执行架构，显著提升了大模型推理的速度和效率。它支持高度并发的请求处理，能够同时服务数千名用户，并且兼容多种深度学习框架，方便集成到现有的机器学习流程中。

鉴于目前 vLLM 的中文学习资料较为零散，不利于开发者系统性学习，我们在 GitHub 上创建了 vLLM 文档翻译项目。

随着 vLLM 官方文档的更新，中文文档也会同步修订，您可以：

- 学习 vLLM 中文文档，对于翻译不准确或存在歧义的地方，[提交 issue](https://github.com/hyperai/vllm-cn/issues) 或 [PR](https://github.com/hyperai/vllm-cn/pulls)
- 参与开源协作，跟踪文档更新，认领文档翻译，成为 vLLM 中文文档贡献者
- 加入 vLLM 中文社区，结识志同道合的伙伴，参与深入的讨论和交流

我们衷心希望通过这个项目，为 vLLM 中文社区的发展贡献一份力量。

## 参与贡献

本地开发服务器需要先安装 Node.js 和 [pnpm](https://pnpm.io/installation)。

```bash
pnpm install
pnpm start
```

## 创建新版本

如果当前版本为 `0.12.0`，想升到 `0.13.0`，那么你需要先保存当前版本

```bash
pnpm run docusaurus docs:version 0.12.0
```

然后编辑 `docusaurus.config.ts` 中 `versions.current.label` 为最新版本 `0.13.0`
