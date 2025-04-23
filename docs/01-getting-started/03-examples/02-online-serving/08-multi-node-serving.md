---
title: Multi-node-serving
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/multi-node-serving.sh](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/multi-node-serving.sh)

```bash
#!/bin/bash

subcommand=$1
shift

ray_port=6379
ray_init_timeout=300
declare -a start_params

case "$subcommand" in
  worker)
    ray_address=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --ray_address=*)
          ray_address="${1#*=}"
          ;;
        --ray_port=*)
          ray_port="${1#*=}"
          ;;
        --ray_init_timeout=*)
          ray_init_timeout="${1#*=}"
          ;;
        *)
          start_params+=("$1")
      esac
      shift
    done

    if [ -z "$ray_address" ]; then
      echo "Error: Missing argument --ray_address"
      exit 1
    fi

    for (( i=0; i < $ray_init_timeout; i+=5 )); do
      ray start --address=$ray_address:$ray_port --block "${start_params[@]}"
      if [ $? -eq 0 ]; then
        echo "Worker: Ray runtime started with head address $ray_address:$ray_port"
        exit 0
      fi
      echo "Waiting until the ray worker is active..."
      sleep 5s;
    done
    echo "Ray worker starts timeout, head address: $ray_address:$ray_port"
    exit 1
    ;;

  leader)
    ray_cluster_size=""
    while [ $# -gt 0 ]; do
          case "$1" in
            --ray_port=*)
              ray_port="${1#*=}"
              ;;
            --ray_cluster_size=*)
              ray_cluster_size="${1#*=}"
              ;;
            --ray_init_timeout=*)
              ray_init_timeout="${1#*=}"
              ;;
            *)
              start_params+=("$1")
          esac
          shift
    done

    if [ -z "$ray_cluster_size" ]; then
      echo "Error: Missing argument --ray_cluster_size"
      exit 1
    fi

    # start the ray daemon
    # 启动 ray 守护进程
    ray start --head --port=$ray_port "${start_params[@]}"

    # wait until all workers are active
    # 等待所有进程激活
    for (( i=0; i < $ray_init_timeout; i+=5 )); do
        active_nodes=`python3 -c 'import ray; ray.init(); print(sum(node["Alive"] for node in ray.nodes()))'`
        if [ $active_nodes -eq $ray_cluster_size ]; then
          echo "All ray workers are active and the ray cluster is initialized successfully."
          exit 0
        fi
        echo "Wait for all ray workers to be active. $active_nodes/$ray_cluster_size is active"
        sleep 5s;
    done

    echo "Waiting for all ray workers to be active timed out."
    exit 1
    ;;

  *)
    echo "unknown subcommand: $subcommand"
    exit 1
    ;;
esac

```
