---
title: Helm 图表
---

[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/chart-helm](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/chart-helm)

本目录包含用于部署 vLLM 应用程序的 Helm 图表。该图表包含部署配置、自动扩缩容、资源管理及其他相关配置项。

## Files

* Chart.yaml：定义图表元数据，包括名称、版本和维护者信息

* ct.yaml：图表测试配置文件

* lintconf.yaml：YAML 文件的语法检查规则

* values.schema.json：用于验证 values.yaml 的 JSON 模式

* values.yaml：Helm 图表的默认配置值

* templates/_helpers.tpl：用于定义通用配置的辅助模板

* templates/configmap.yaml：创建 ConfigMap 的模板

* templates/custom-objects.yaml：自定义 Kubernetes 对象的模板

* templates/deployment.yaml：创建 Deployment 的模板

* templates/hpa.yaml：水平 Pod 自动扩缩容模板

* templates/job.yaml：Kubernetes Job 模板

* templates/poddisruptionbudget.yaml：Pod 中断预算模板

* templates/pvc.yaml：持久卷声明模板

* templates/secrets.yaml：Kubernetes Secret 模板

* templates/service.yaml：创建 Service 的模板

# Example materials

## .helmignore


```plain
*.png
.git/
ct.yaml
lintconf.yaml
values.schema.json
/workflows
```


## Chart.yaml


```plain
apiVersion: v2
name: chart-vllm
description: Chart vllm


# 图表类型可分为"应用型"或"库型"
# 
# 应用型图表是一组可打包成版本化归档文件
# 以供部署的模板集合
# 
# 库型图表为开发者提供实用工具函数，
# 它们作为应用图表的依赖项被引入，
# 将功能注入渲染流程。
# 库型图表不包含任何可部署模板，
# 因此无法直接部署。
type: application


# 图表版本号，每次修改图表内容
# 或模板时都应递增此版本号
# 包括应用版本更新时
# 版本号需遵循语义化版本规范(https://semver.org/)
version: 0.0.1


maintainers:
  - name: mfournioux
```


## ct.yaml


```plain
chart-dirs:
  - charts
validate-maintainers: false
```


## lintconf.yaml


```plain
---
rules:
  braces:
    min-spaces-inside: 0
    max-spaces-inside: 0
    min-spaces-inside-empty: -1
    max-spaces-inside-empty: -1
  brackets:
    min-spaces-inside: 0
    max-spaces-inside: 0
    min-spaces-inside-empty: -1
    max-spaces-inside-empty: -1
  colons:
    max-spaces-before: 0
    max-spaces-after: 1
  commas:
    max-spaces-before: 0
    min-spaces-after: 1
    max-spaces-after: 1
  comments:
    require-starting-space: true
    min-spaces-from-content: 2
  document-end: disable
  document-start: disable           # No --- to start a file  # 不 --- 启动文件
  empty-lines:
    max: 2
    max-start: 0
    max-end: 0
  hyphens:
    max-spaces-after: 1
  indentation:
    spaces: consistent
    indent-sequences: whatever      # - list indentation will handle both indentation and without
    indent-sequences: whatever      # - 列表缩进将会处理有无缩进
    check-multi-line-strings: false
  key-duplicates: enable
  line-length: disable              # Lines can be any length  # Lines 可以任意长度
  new-line-at-end-of-file: disable
  new-lines:
    type: unix
  trailing-spaces: enable
  truthy:
    level: warning
```


## templates/_helpers.tpl


```plain
{{/*
Define ports for the pods
定义 Pod 端口
*/}}
{{- define "chart.container-port" -}}
{{-  default "8000" .Values.containerPort }}
{{- end }}


{{/*
Define service name
定义服务名称
*/}}
{{- define "chart.service-name" -}}
{{-  if .Values.serviceName }}
{{-    .Values.serviceName | lower | trim }}
{{-  else }}
"{{ .Release.Name }}-service"
{{-  end }}
{{- end }}


{{/*
Define service port
定义服务端口
*/}}
{{- define "chart.service-port" -}}
{{-  if .Values.servicePort }}
{{-    .Values.servicePort }}
{{-  else }}
{{-    include "chart.container-port" . }}
{{-  end }}
{{- end }}


{{/*
Define service port name
定义服务端口名称
*/}}
{{- define "chart.service-port-name" -}}
"service-port"
{{- end }}


{{/*
Define container port name
定义容器端口名称
*/}}
{{- define "chart.container-port-name" -}}
"container-port"
{{- end }}


{{/*
Define deployment strategy
定义部署策略
*/}}
{{- define "chart.strategy" -}}
strategy:
{{-   if not .Values.deploymentStrategy }}
  rollingUpdate:
    maxSurge: 100%
    maxUnavailable: 0
{{-   else }}
{{      toYaml .Values.deploymentStrategy | indent 2 }}
{{-   end }}
{{- end }}


{{/*
Define additional ports
定义额外端口
*/}}
{{- define "chart.extraPorts" }}
{{-   with .Values.extraPorts }}
{{      toYaml . }}
{{-   end }}
{{- end }}


{{/*
Define chart external ConfigMaps and Secrets
定义外部配置映射和密钥
*/}}
{{- define "chart.externalConfigs" -}}
{{-   with .Values.externalConfigs -}}
{{      toYaml . }}
{{-   end }}
{{- end }}




{{/*
Define liveness et readiness probes
定义存活和就绪探针
*/}}
{{- define "chart.probes" -}}
{{-   if .Values.readinessProbe  }}
readinessProbe:
{{-     with .Values.readinessProbe }}
{{-       toYaml . | nindent 2 }}
{{-     end }}
{{-   end }}
{{-   if .Values.livenessProbe  }}
livenessProbe:
{{-     with .Values.livenessProbe }}
{{-       toYaml . | nindent 2 }}
{{-     end }}
{{-   end }}
{{- end }}


{{/*
Define resources
定义资源配额
*/}}
{{- define "chart.resources" -}}
requests:
  memory: {{ required "Value 'resources.requests.memory' must be defined !" .Values.resources.requests.memory | quote }}
  cpu: {{ required "Value 'resources.requests.cpu' must be defined !" .Values.resources.requests.cpu | quote }}
  {{- if and (gt (int (index .Values.resources.requests "nvidia.com/gpu")) 0) (gt (int (index .Values.resources.limits "nvidia.com/gpu")) 0) }}
  nvidia.com/gpu: {{ required "Value 'resources.requests.nvidia.com/gpu' must be defined !" (index .Values.resources.requests "nvidia.com/gpu") | quote }}
  {{- end }}
limits:
  memory: {{ required "Value 'resources.limits.memory' must be defined !" .Values.resources.limits.memory | quote }}
  cpu: {{ required "Value 'resources.limits.cpu' must be defined !" .Values.resources.limits.cpu | quote }}
  {{- if and (gt (int (index .Values.resources.requests "nvidia.com/gpu")) 0) (gt (int (index .Values.resources.limits "nvidia.com/gpu")) 0) }}
  nvidia.com/gpu: {{ required "Value 'resources.limits.nvidia.com/gpu' must be defined !" (index .Values.resources.limits "nvidia.com/gpu") | quote }}
  {{- end }}
{{- end }}




{{/*
Define User used for the main container
定义主容器运行用户
*/}}
{{- define "chart.user" }}
{{-   if .Values.image.runAsUser  }}
runAsUser: 
{{-     with .Values.runAsUser }}
{{-       toYaml . | nindent 2 }}
{{-     end }}
{{-   end }}
{{- end }}


{{- define "chart.extraInitImage" -}}
"amazon/aws-cli:2.6.4"
{{- end }}


{{- define "chart.extraInitEnv" -}}
- name: S3_ENDPOINT_URL
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secrets
      key: s3endpoint
- name: S3_BUCKET_NAME
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secrets
      key: s3bucketname
- name: AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secrets
      key: s3accesskeyid
- name: AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secrets
      key: s3accesskey
- name: S3_PATH
  value: "{{ .Values.extraInit.s3modelpath }}"
- name: AWS_EC2_METADATA_DISABLED
  value: "{{ .Values.extraInit.awsEc2MetadataDisabled }}"
{{- end }}


{{/*
  Define chart labels
定义图表标签
*/}}
{{- define "chart.labels" -}}
{{-   with .Values.labels -}}
{{      toYaml . }}
{{-   end }}
{{- end }}
```


## templates/configmap.yaml


```plain
{{- if .Values.configs -}}
apiVersion: v1
kind: ConfigMap
metadata:
  name: "{{ .Release.Name }}-configs"
  namespace: {{ .Release.Namespace }}
data:
  {{- with .Values.configs }}
  {{- toYaml . | nindent 2 }}
  {{- end }}
{{- end -}}
```


## templates/custom-objects.yaml


```plain
{{- if .Values.customObjects }}
{{- range .Values.customObjects }}
{{- tpl (. | toYaml) $ }}
---
{{- end }}
{{- end }}
```


## templates/deployment.yaml


```plain
apiVersion: apps/v1
kind: Deployment
metadata:
  name: "{{ .Release.Name }}-deployment-vllm"
  namespace: {{ .Release.Namespace }}
  labels:
  {{- include "chart.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  {{- include "chart.strategy" . | nindent 2 }}
  selector:                                                                                                                                  
    matchLabels:
      environment: "test"
      release: "test"
  progressDeadlineSeconds: 1200
  template:
    metadata:
      labels:
        environment: "test"
        release: "test"
    spec:
      containers:
        - name: "vllm"
          image: "{{ required "Required value 'image.repository' must be defined !" .Values.image.repository }}:{{ required "Required value 'image.tag' must be defined !" .Values.image.tag }}"
          {{- if .Values.image.command }}
          command :
            {{- with .Values.image.command }}
            {{- toYaml . | nindent 10 }}
            {{- end }}
          {{- end }}
          securityContext:
            {{- if .Values.image.securityContext }}
              {{- with .Values.image.securityContext }}
              {{- toYaml . | nindent 12 }}
              {{- end }}
            {{- else }}
            runAsNonRoot: false
              {{- include "chart.user" . | indent 12 }}
            {{- end }}
          imagePullPolicy: IfNotPresent
          {{- if .Values.image.env }}
          env :
            {{- with .Values.image.env }}
            {{- toYaml . | nindent 10 }}
            {{- end }}
          {{- else }}
          env: []
          {{- end }}
          {{- if or .Values.externalConfigs .Values.configs .Values.secrets }}
          envFrom:
            {{- if .Values.configs }}
            - configMapRef:
                name: "{{ .Release.Name }}-configs"
            {{- end }}
            {{- if .Values.secrets}}
            - secretRef:
                name: "{{ .Release.Name }}-secrets"
            {{- end }}
            {{- include "chart.externalConfigs" . | nindent 12 }}
          {{- end }}          
          ports:
            - name: {{ include "chart.container-port-name" . }}
              containerPort: {{ include "chart.container-port" . }}
            {{- include "chart.extraPorts" . | nindent 12 }}
          {{- include "chart.probes" . | indent 10 }}
          resources: {{- include "chart.resources" . | nindent 12 }}
          volumeMounts:
          - name: {{ .Release.Name }}-storage
            mountPath: /data


        {{- with .Values.extraContainers }}
        {{ toYaml . | nindent 8 }}
        {{- end }}


      {{-   if .Values.extraInit  }}
      initContainers:
      - name: wait-download-model
        image: {{ include "chart.extraInitImage" . }}
        command: 
          - /bin/bash
        args:
          - -eucx
          - while aws --endpoint-url $S3_ENDPOINT_URL s3 sync --dryrun s3://$S3_BUCKET_NAME/$S3_PATH /data | grep -q download; do sleep 10; done
        env: {{- include "chart.extraInitEnv" . | nindent 10 }}
        resources:
          requests:
            cpu: 200m
            memory: 1Gi
          limits:
            cpu: 500m
            memory: 2Gi
        volumeMounts:
        - name: {{ .Release.Name }}-storage
          mountPath: /data
      {{- end }}
      volumes:
        - name: {{ .Release.Name }}-storage
          persistentVolumeClaim:
            claimName: {{ .Release.Name }}-storage-claim     


      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- if and (gt (int (index .Values.resources.requests "nvidia.com/gpu")) 0) (gt (int (index .Values.resources.limits "nvidia.com/gpu")) 0) }}
      runtimeClassName: nvidia
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                - key: nvidia.com/gpu.product
                  operator: In
                  {{- with .Values.gpuModels }}
                  values:
                    {{- toYaml . | nindent 20 }}
                  {{- end }}
      {{- end }} 
```


## templates/hpa.yaml


```plain
{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: "{{ .Release.Name }}-hpa"
  namespace: {{ .Release.Namespace }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    {{- if .Values.autoscaling.targetCPUUtilizationPercentage }}
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
    {{- end }}
    {{- if .Values.autoscaling.targetMemoryUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}
    {{- end }}
{{- end }}
```


## templates/job.yaml


```plain
{{-   if .Values.extraInit  }}
apiVersion: batch/v1
kind: Job
metadata:
  name: "{{ .Release.Name }}-init-vllm"
  namespace: {{ .Release.Namespace }}
spec:
  ttlSecondsAfterFinished: 100
  template:
   metadata:
     name: init-vllm
   spec:
    containers:
    - name: job-download-model
      image: {{ include "chart.extraInitImage" . }}
      command: 
        - /bin/bash
      args:
        - -eucx
        - aws --endpoint-url $S3_ENDPOINT_URL s3 sync s3://$S3_BUCKET_NAME/$S3_PATH /data
      env: {{- include "chart.extraInitEnv" . | nindent 8 }}
      volumeMounts:
        - name: {{ .Release.Name }}-storage
          mountPath: /data
      resources:
        requests:
          cpu: 200m
          memory: 1Gi
        limits:
          cpu: 500m
          memory: 2Gi
    restartPolicy: OnFailure
    volumes:
    - name: {{ .Release.Name }}-storage
      persistentVolumeClaim:
        claimName: "{{ .Release.Name }}-storage-claim"
{{- end }}
```


## templates/poddisruptionbudget.yaml


```plain
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: "{{ .Release.Name }}-pdb"
  namespace: {{ .Release.Namespace }}
spec:
  maxUnavailable: {{ default 1 .Values.maxUnavailablePodDisruptionBudget }}
```


## templates/pvc.yaml


```plain
{{-   if .Values.extraInit  }}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: "{{ .Release.Name }}-storage-claim"
  namespace: {{ .Release.Namespace }}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: {{ .Values.extraInit.pvcStorage }}
{{- end }}
```


## templates/secrets.yaml


```plain
apiVersion: v1
kind: Secret
metadata:
  name: "{{ .Release.Name }}-secrets"
  namespace: {{ .Release.Namespace }}
type: Opaque
data:
  {{- range $key, $val := .Values.secrets }}
  {{ $key }}: {{ $val | b64enc | quote }}
  {{- end }}
```


## templates/service.yaml


```plain
apiVersion: v1
kind: Service
metadata:
  name: "{{ .Release.Name }}-service"
  namespace: {{ .Release.Namespace }}
spec:
  type: ClusterIP
  ports:
    - name: {{ include "chart.service-port-name" . }}
      port: {{ include "chart.service-port" . }}
      targetPort: {{ include "chart.container-port-name" . }}
      protocol: TCP
  selector:
  {{- include "chart.labels" . | nindent 4 }}
```


## values.schema.json


```plain
{
    "$schema": "http://json-schema.org/schema#",
    "type": "object",
    "properties": {
        "image": {
            "type": "object",
            "properties": {
                "repository": {
                    "type": "string"
                },
                "tag": {
                    "type": "string"
                },
                "command": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": [
                "command",
                "repository",
                "tag"
            ]
        },
        "containerPort": {
            "type": "integer"
        },
        "serviceName": {
            "type": "null"
        },
        "servicePort": {
            "type": "integer"
        },
        "extraPorts": {
            "type": "array"
        },
        "replicaCount": {
            "type": "integer"
        },
        "deploymentStrategy": {
            "type": "object"
        },
        "resources": {
            "type": "object",
            "properties": {
                "requests": {
                    "type": "object",
                    "properties": {
                        "cpu": {
                            "type": "integer"
                        },
                        "memory": {
                            "type": "string"
                        },
                        "nvidia.com/gpu": {
                            "type": "integer"
                        }
                    },
                    "required": [
                        "cpu",
                        "memory",
                        "nvidia.com/gpu"
                    ]
                },
                "limits": {
                    "type": "object",
                    "properties": {
                        "cpu": {
                            "type": "integer"
                        },
                        "memory": {
                            "type": "string"
                        },
                        "nvidia.com/gpu": {
                            "type": "integer"
                        }
                    },
                    "required": [
                        "cpu",
                        "memory",
                        "nvidia.com/gpu"
                    ]
                }
            },
            "required": [
                "limits",
                "requests"
            ]
        },
        "gpuModels": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "autoscaling": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean"
                },
                "minReplicas": {
                    "type": "integer"
                },
                "maxReplicas": {
                    "type": "integer"
                },
                "targetCPUUtilizationPercentage": {
                    "type": "integer"
                }
            },
            "required": [
                "enabled",
                "maxReplicas",
                "minReplicas",
                "targetCPUUtilizationPercentage"
            ]
        },
        "configs": {
            "type": "object"
        },
        "secrets": {
            "type": "object"
        },
        "externalConfigs": {
            "type": "array"
        },
        "customObjects": {
            "type": "array"
        },
        "maxUnavailablePodDisruptionBudget": {
            "type": "string"
        },
        "extraInit": {
            "type": "object",
            "properties": {
                "s3modelpath": {
                    "type": "string"
                },
                "pvcStorage": {
                    "type": "string"
                },
                "awsEc2MetadataDisabled": {
                    "type": "boolean"
                }
            },
            "required": [
                "pvcStorage",
                "s3modelpath",
                "awsEc2MetadataDisabled"
            ]
        },
        "extraContainers": {
            "type": "array"
        },
        "readinessProbe": {
            "type": "object",
            "properties": {
                "initialDelaySeconds": {
                    "type": "integer"
                },
                "periodSeconds": {
                    "type": "integer"
                },
                "failureThreshold": {
                    "type": "integer"
                },
                "httpGet": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string"
                        },
                        "port": {
                            "type": "integer"
                        }
                    },
                    "required": [
                        "path",
                        "port"
                    ]
                }
            },
            "required": [
                "failureThreshold",
                "httpGet",
                "initialDelaySeconds",
                "periodSeconds"
            ]
        },
        "livenessProbe": {
            "type": "object",
            "properties": {
                "initialDelaySeconds": {
                    "type": "integer"
                },
                "failureThreshold": {
                    "type": "integer"
                },
                "periodSeconds": {
                    "type": "integer"
                },
                "httpGet": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string"
                        },
                        "port": {
                            "type": "integer"
                        }
                    },
                    "required": [
                        "path",
                        "port"
                    ]
                }
            },
            "required": [
                "failureThreshold",
                "httpGet",
                "initialDelaySeconds",
                "periodSeconds"
            ]
        },
        "labels": {
            "type": "object",
            "properties": {
                "environment": {
                    "type": "string"
                },
                "release": {
                    "type": "string"
                }
            },
            "required": [
                "environment",
                "release"
            ]
        }
    },
    "required": [
        "autoscaling",
        "configs",
        "containerPort",
        "customObjects",
        "deploymentStrategy",
        "externalConfigs",
        "extraContainers",
        "extraInit",
        "extraPorts",
        "gpuModels",
        "image",
        "labels",
        "livenessProbe",
        "maxUnavailablePodDisruptionBudget",
        "readinessProbe",
        "replicaCount",
        "resources",
        "secrets",
        "servicePort"
    ]
}
```


## values.yaml


```plain
# -- Default values for chart vllm
# -- vLLM Chart 的默认值
# -- Declare variables to be passed into your templates.
# -- 声明要传递到模板中的变量


# -- Image configuration
# -- 镜像配置
image:
  # -- Image repository
  # -- 镜像仓库
  repository: "vllm/vllm-openai"
  # -- Image tag
  # -- 镜像标签
  tag: "latest"
  # -- Container launch command
  # -- 容器启动命令
  command: ["vllm", "serve", "/data/", "--served-model-name", "opt-125m", "--dtype", "bfloat16", "--host", "0.0.0.0", "--port", "8000"]


# -- Container port
# -- 容器端口
containerPort: 8000
# -- Service name
# -- 服务名称
serviceName:
# -- Service port
# -- 服务端口
servicePort: 80
# -- Additional ports configuration
# -- 额外端口配置
extraPorts: []


# -- Number of replicas
# -- 副本数量
replicaCount: 1


# -- Deployment strategy configuration
# -- 部署策略配置
deploymentStrategy: {}


# -- Resource configuration
# -- 资源配置
resources:
  requests:
    # -- Number of CPUs
    # -- CPU 数量
    cpu: 4
    # -- CPU memory configuration
     # -- CPU 内存配置
    memory: 16Gi
    # -- Number of gpus used
    # -- 使用的 GPU 数量
    nvidia.com/gpu: 1
  limits:
    # -- Number of CPUs
    # -- CPU 数量


    cpu: 4
    # -- CPU memory configuration
    # -- CPU memory configuration
    memory: 16Gi
    # -- Number of gpus used
    # -- 使用的 GPU 数量
    nvidia.com/gpu: 1


# -- Type of gpu used
# -- 使用的 GPU 类型
gpuModels:
  - "TYPE_GPU_USED"


# -- Autoscaling configuration
# -- 自动扩展配置
autoscaling:
  # -- Enable autoscaling
  # -- 是否启用自动扩展
  enabled: false
  # -- Minimum replicas
  # -- 最小副本数
  minReplicas: 1
  # -- Maximum replicas
  # -- 最大副本数
  maxReplicas: 100
  # -- Target CPU utilization for autoscaling
  # -- 自动扩展的目标 CPU 使用率
  targetCPUUtilizationPercentage: 80
  # targetMemoryUtilizationPercentage: 80
  # targetMemoryUtilizationPercentage: 80


# -- Configmap
# -- 配置映射
configs: {}


# -- Secrets configuration
# -- 机密配置
secrets: {}


# -- External configuration
# -- 外部配置
externalConfigs: []


# -- Custom Objects configuration
# -- 自定义对象配置
customObjects: []


# -- Disruption Budget Configuration
# -- 破坏预算配置
maxUnavailablePodDisruptionBudget: ""


# -- Additional configuration for the init container
# -- Init 容器的额外配置
extraInit:
   # -- Path of the model on the s3 which hosts model weights and config files
   # -- S3 上存储模型权重和配置文件的路径
  s3modelpath: "relative_s3_model_path/opt-125m"
   # -- Storage size of the s3
   # -- S3 的存储大小
  pvcStorage: "1Gi"
  awsEc2MetadataDisabled: true


# -- Additional containers configuration
# -- 额外的容器配置
extraContainers: []


# -- Readiness probe configuration
# -- 就绪探针（Readiness Probe）配置
readinessProbe:
  # -- Number of seconds after the container has started before readiness probe is initiated
  # -- 容器启动后执行就绪探针前的等待时间（秒）
  initialDelaySeconds: 5
  # -- How often (in seconds) to perform the readiness probe
  # -- 就绪探针执行的频率（秒）
  periodSeconds: 5
  # -- Number of times after which if a probe fails in a row, Kubernetes considers that the overall check has failed: the container is not ready
  # -- 如果连续失败的次数达到该值，则 Kubernetes 认为容器未就绪
  failureThreshold: 3
   # -- Configuration of the Kubelet http request on the server
   # -- Kubelet 在服务器上的 HTTP 请求配置
  httpGet:
    # -- Path to access on the HTTP server
    # -- HTTP 服务器上访问的路径
    path: /health
    # -- Name or number of the port to access on the container, on which the server is listening
    # -- 服务器监听的访问容器的端口或名字
    port: 8000


# -- Liveness probe configuration
# -- 存活探针（Liveness Probe）配置
livenessProbe:
 # -- Number of seconds after the container has started before liveness probe is initiated
 # -- 容器启动后执行存活探针前的等待时间（秒）
  initialDelaySeconds: 15
  # -- Number of times after which if a probe fails in a row, Kubernetes considers that the overall check has failed: the container is not alive
  # -- 如果连续失败的次数达到该值，则 Kubernetes 认为容器已崩溃
  failureThreshold: 3
  # -- How often (in seconds) to perform the liveness probe
  # -- 存活探针执行的频率（秒）
  periodSeconds: 10
  # -- Configuration of the Kubelet http request on the server
  # -- Kubelet 在服务器上的 HTTP 请求配置
  httpGet:
    # -- Path to access on the HTTP server
    # -- HTTP 服务器上访问的路径
    path: /health
    # -- Name or number of the port to access on the container, on which the server is listening
    # -- 服务器监听的访问容器的端口或名字
    port: 8000


labels:
  environment: "test"
  release: "test"
```


