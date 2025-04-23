---
title: 日志配置说明
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/other/logging_configuration.md](https://github.com/vllm-project/vllm/blob/main/examples/other/logging_configuration.md).

vLLM 利用 Python 的 `logging.config.dictConfig` 功能，实现对 vLLM 所用各类日志记录器的健壮且灵活的配置。

vLLM 提供两个环境变量，可支持从简单不灵活到复杂灵活的各种日志配置方案：

- 禁用 vLLM 日志（简单不灵活）

  - 设置 `VLLM_CONFIGURE_LOGGING=0`（保持 `VLLM_LOGGING_CONFIG_PATH` 未设置）

- vLLM 默认日志配置（简单不灵活）

  - 保持 `VLLM_CONFIGURE_LOGGING` 未设置或设为 `VLLM_CONFIGURE_LOGGING=1`

- 细粒度自定义日志配置（更复杂，更灵活）

  - 保持 `VLLM_CONFIGURE_LOGGING` 未设置或设为 `VLLM_CONFIGURE_LOGGING=1`，同时设置 `VLLM_LOGGING_CONFIG_PATH=<日志配置文件路径.json>`

## 环境变量详解

### `VLLM_CONFIGURE_LOGGING`

`VLLM_CONFIGURE_LOGGING` 控制 vLLM 是否对其使用的日志记录器进行配置。此功能默认启用，但可通过在运行 vLLM 时设置 `VLLM_CONFIGURE_LOGGING=0` 来禁用。

若启用 `VLLM_CONFIGURE_LOGGING` 但未设置 `VLLM_LOGGING_CONFIG_PATH` 的值，vLLM 将使用内置默认配置来设置根 vLLM 日志记录器。默认情况下，其他 vLLM 日志记录器均不单独配置，因此所有 vLLM 日志记录器都将遵从根日志记录器做出日志决策。

若禁用 `VLLM_CONFIGURE_LOGGING` 却设置了 `VLLM_LOGGING_CONFIG_PATH` 的值，vLLM 将在启动时报错。

### `VLLM_LOGGING_CONFIG_PATH`

`VLLM_LOGGING_CONFIG_PATH` 允许用户指定一个 JSON 格式的自定义日志配置文件路径，该配置将替代 vLLM 内置的默认日志配置。日志配置文件需遵循 Python [日志配置字典模式](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details) 规定的 JSON 格式。

若指定了 `VLLM_LOGGING_CONFIG_PATH` 但禁用了 `VLLM_CONFIGURE_LOGGING`，vLLM 将在启动时报错。

## 示例

### 示例 1：自定义 vLLM 根日志记录器

在本示例中，我们将自定义 vLLM 根日志记录器，使用 `python-json-logger` （该组件已包含在容器镜像中），以 JSON 格式将 `INFO` 级别的日志输出到控制台 STDOUT。

首先，创建一个适当的 JSON 日志配置文件：

**/path/to/logging_config.json:**

```json
{
  "formatters": {
    "json": {
      "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
    }
  },
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "formatter": "json",
      "level": "INFO",
      "stream": "ext://sys.stdout"
    }
  },
  "loggers": {
    "vllm": {
      "handlers": ["console"],
      "level": "INFO",
      "propagate": false
    }
  },
  "version": 1
}
```

最后，运行 vLLM 并将 `VLLM_LOGGING_CONFIG_PATH` 环境变量设置为自定义日志配置 JSON 文件的路径：

```bash
VLLM_LOGGING_CONFIG_PATH=/path/to/logging_config.json \
    vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

### 示例 2：静默特定的 vLLM 日志记录器

要静默特定的 vLLM 日志记录器，需要为目标日志记录器提供自定义配置，使其日志消息不会传播到 vLLM 根日志记录器。

为任何日志记录器提供自定义配置时，还需同时配置 vLLM 根日志记录器，因为自定义配置会覆盖 vLLM 使用的内置默认日志配置。

首先，创建一个包含 vLLM 根日志记录器和目标静默日志记录器配置的 JSON 日志配置文件：

**/path/to/logging_config.json:**

```json
{
  "formatters": {
    "vllm": {
      "class": "vllm.logging_utils.NewLineFormatter",
      "datefmt": "%m-%d %H:%M:%S",
      "format": "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
    }
  },
  "handlers": {
    "vllm": {
      "class": "logging.StreamHandler",
      "formatter": "vllm",
      "level": "INFO",
      "stream": "ext://sys.stdout"
    }
  },
  "loggers": {
    "vllm": {
      "handlers": ["vllm"],
      "level": "DEBUG",
      "propagate": false
    },
    "vllm.example_noisy_logger": {
      "propagate": false
    }
  },
  "version": 1
}
```

最后，运行 vLLM 并将 `VLLM_LOGGING_CONFIG_PATH` 环境变量设置为自定义日志配置 JSON 文件的路径：

```bash
VLLM_LOGGING_CONFIG_PATH=/path/to/logging_config.json \
    vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

### 示例 3：禁用 vLLM 默认日志配置

要禁用 vLLM 的默认日志配置并静默所有 vLLM 日志记录器，只需在运行 vLLM 时设置 `VLLM_CONFIGURE_LOGGING=0`。这将阻止 vLLM 配置根日志记录器，从而静默所有其他 vLLM 日志记录器。

```bash
VLLM_CONFIGURE_LOGGING=0 \
    vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

## 附加资源

- `logging.config`[ 字典模式详情](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details)
