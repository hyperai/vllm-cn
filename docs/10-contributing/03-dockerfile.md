---

title: Dockerfile

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


请参阅[此处](https://github.com/vllm-project/vllm/blob/main/Dockerfile)了解主要 Dockerfile，构建用于使用 vLLM 运行 OpenAI 兼容服务器的镜像。有关使用 Docker 进行部署的更多信息可以在[此处](https://docs.vllm.ai/en/stable/serving/deploying_with_docker.html)找到。


下面是多阶段 Dockerfile 的直观表示。构建图包含以下节点: 

* 所有构建阶段

* 默认构建目标（以灰色突出显示）

* 外部图像（带有虚线边框）


构建图的边代表：

* FROM ... dependencies （带有实线和完整箭头）

* COPY --from=... dependencies （带有虚线和空箭头）

* RUN --mount=(.*)from=... dependencies （带有虚线和空菱形箭头）

![图片](/img/docs/v1-Developer_Guide/03-dockerfile_1.png)
>使用: [https://github.com/patrickhoefler/dockerfilegraph](https://github.com/patrickhoefler/dockerfilegraph) 制作  
>
>重新生成构建图的命令 （确保从存在 dockerfile 的 vLLM 存储库的 *root* 目录运行它）:    
```bash
dockerfilegraph -o png --legend --dpi 200 --max-label-length 50 --filename Dockerfile
```
>   或者如果您想直接使用 docker 镜像运行它：
```bash
docker run \
   --rm \
   --user "$(id -u):$(id -g)" \
   --workdir /workspace \
   --volume "$(pwd)":/workspace \
   ghcr.io/patrickhoefler/dockerfilegraph:alpine \
   --output png \
   --dpi 200 \
   --max-label-length 50 \
   --filename Dockerfile \
   --legend
```
>（要使用不同的文件运行，您可以将不同的参数传递给标志 *--filename*。） 

