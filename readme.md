# Vision Perception

车载驾驶视频智能分析平台。系统接收行车视频，使用 VLM 多智能体算法做交通场景挖掘，生成结构化标签、异常事件时间段和自然语言摘要，并通过 Qwen3-VL 多模态嵌入模型将文本、图片和视频帧统一映射到同一向量空间，写入 Milvus 后支持按文本、标签、图片和视觉语义检索视频。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 视频接入 | 支持上传视频、按 URL/原始数据 ID 导入视频，并存储到 MinIO |
| 场景挖掘 | 基于 Qwen3.5-Gemini-Distill VLM 和 LangGraph 多智能体 pipeline 分析行车场景 |
| 异常识别 | 输出弱势交通参与者异常、车辆轨迹冲突、极端高危事件等时间段 |
| 摘要生成 | 基于各类别分析结果生成可检索的视频摘要 |
| 多模态检索 | 支持 frame、summary、tags、visual、image 等检索模式 |
| 媒体代理 | 统一将 MinIO 资源映射为 `/media/<bucket>/<object>`，支持浏览器 Range 播放 |

## 架构概览

```text
Browser / Client
      |
Flask / Gunicorn  :30501
      |
      +-- UploadService       -> MinIO
      +-- AddVideoService
      |     +-- MiningVideoService  -> qwen-gemini-vlm :8576
      |     +-- SummaryVideoService -> qwen-gemini-vlm :8576
      |     +-- VideoFeatureService -> qwen3-vl-embedding :8575
      |
      +-- SearchVideoService  -> Milvus
```

当前推荐部署方式是 **Docker Compose**。`docker-compose.yml` 负责启动：

| 服务 | 作用 | 端口 |
| --- | --- | --- |
| `qwen-gemini-vlm` | 场景挖掘和摘要生成 VLM，OpenAI-compatible `/v1` 接口 | `127.0.0.1:8576` |
| `qwen3-vl-embedding` | Qwen3-VL 多模态嵌入服务，统一处理文本、图片和视频帧，提供 `/embed` 接口 | `127.0.0.1:8575` |
| `vision-perception` | Flask/Gunicorn 应用服务 | `0.0.0.0:30501` |

> 注意：当前 compose 文件不内置 Milvus 和 MinIO，需要使用已有服务，并在 `.env` 中配置连接信息。

## 模型下载

模型文件不提交到代码仓库，需要提前从 Hugging Face 下载并放到宿主机固定目录。

### 1. 场景挖掘 VLM

模型仓库：

```text
https://huggingface.co/Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill
```

下载到：

```text
/mnt/data/checkpoints/Qwen3_5-9B-Gemini-Distill
```

示例命令：

```bash
pip install -U huggingface_hub
huggingface-cli download \
  Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill \
  --local-dir /mnt/data/checkpoints/Qwen3_5-9B-Gemini-Distill
```

compose 中挂载为：

```text
/models/qwen-gemini
```

### 2. 多模态嵌入模型

当前嵌入模型已经切换为 **Qwen3-VL-Embedding-2B 多模态嵌入模型**，不再依赖原来的单独文本/图片嵌入模型。它用于：

1. 将 tags 和 summary 文本编码为文本向量。
2. 将图片查询编码为图片向量。
3. 将视频采样帧编码为视觉向量，并聚合成视频级全局视觉特征。
4. 让文本、图片、视频帧落在同一个 2048 维多模态语义空间，支持跨模态检索。

模型仓库：

```text
https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B
```

下载到：

```text
/mnt/data/checkpoints/Qwen3-VL-Embedding-2B
```

示例命令：

```bash
huggingface-cli download \
  Qwen/Qwen3-VL-Embedding-2B \
  --local-dir /mnt/data/checkpoints/Qwen3-VL-Embedding-2B
```

compose 中挂载为：

```text
/models/Qwen3-VL-Embedding-2B
```

## Docker Compose 启动

### 1. 准备环境变量

```bash
cp .env_sample .env
```

重点检查 `.env` 中以下配置：

```ini
# MinIO
OSS_ENDPOINT=xxx
OSS_ACCESS_KEY=xxx
OSS_SECRET_KEY=xxx
OSS_BUCKET_NAME=perception-mining
OSS_PUBLIC_BASE_URL=/media

# Milvus
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_DB_NAME=video_db
MILVUS_VIDEO_COLLECTION_NAME=videos

# Flask
SERVER_HOST=localhost
SERVER_PORT=30501

# 场景挖掘 VLM
SCENE_MINING_API_BASE_URL=http://localhost:8576/v1
SCENE_MINING_API_MODEL_NAME=/models/qwen-gemini
SCENE_MINING_CONFIG_PATH=app/algorithm/scene_mining/config-qwen-gemini.yaml
SCENE_MINING_OUTPUT_DIR=outputs/scene_mining
SCENE_MINING_VIDEO_CACHE_DIR=data/scene_mining_videos
SCENE_MINING_VIDEO_URL_PREFIX=file:///app/videos
SCENE_MINING_VLM_PORT=8576
SCENE_MINING_VLM_GPU_MEMORY_UTILIZATION=0.7773
SCENE_MINING_VLM_MAX_MODEL_LEN=40960

# Qwen3-VL 多模态嵌入服务
EMBEDDING_MODEL=qwen3-vl
QWEN3_VL_EMBEDDING_BASE_URL=http://localhost:8575
QWEN3_VL_EMBEDDING_PORT=8575
QWEN3_VL_EMBEDDING_DIM=2048
QWEN3_VL_GPU_MEMORY_UTILIZATION=0.1727

# 采样和运行期缓存
FRAME_SAMPLE_FPS=1
FRAME_SAMPLE_MAX_FRAMES=20
FRAME_SAMPLE_EVENT_FPS=1
FRAME_SAMPLE_EVENT_WEIGHT=1.5
MEDIA_CACHE_DIR=data/media_cache
TASK_STATUS_DIR=data/task_status
UPLOAD_LOCK_DIR=/tmp/vision_perception_locks
```

### 2. 确认视频目录

`docker-compose.yml` 默认把宿主机视频目录挂载到容器内：

```yaml
/mnt/data/ai-ground/dataset/videos:/app/videos:ro
```

如果实际视频目录不同，需要同步修改：

1. `docker-compose.yml` 中 `qwen-gemini-vlm` 和 `vision-perception` 的 volumes。
2. `.env` 中的 `SCENE_MINING_VIDEO_URL_PREFIX`，通常保持 `file:///app/videos`。
3. `app/algorithm/scene_mining/config-qwen-gemini.yaml` 中的 `paths.video_root`，本地算法调试时需要指向实际目录。

### 3. 启动

指定 GPU 后启动：

```bash
CUDA_VISIBLE_DEVICES=0 docker compose up --build
```

后台运行：

```bash
CUDA_VISIBLE_DEVICES=0 docker compose up --build -d
```

查看日志：

```bash
docker compose logs -f vision-perception
docker compose logs -f qwen-gemini-vlm
docker compose logs -f qwen3-vl-embedding
```

停止：

```bash
docker compose down
```

## 初始化数据库和特征集合

首次部署或 Milvus collection 变更后执行：

```bash
python -m app.scripts.create_database
```

如果已有历史视频，需要补齐新加的文本/视觉特征：

```bash
python -m app.scripts.backfill_features --limit 10
python -m app.scripts.backfill_features
```

只回填文本特征：

```bash
python -m app.scripts.backfill_features --skip-visual
```

## 视频处理链路

完整处理入口是 `AddVideoService.add()`：

```text
视频记录
  -> MiningVideoService
  -> SummaryVideoService
  -> VideoFeatureService
  -> VideoDAO.upsert_video
```

`action_type` 说明：

| action_type | 行为 |
| --- | --- |
| `1` | 只进行场景挖掘 |
| `2` | 场景挖掘 + 摘要 + 特征 |
| `3` | 默认完整流程，场景挖掘 + 摘要 + 特征 |

### 场景挖掘算法

算法目录：

```text
app/algorithm/scene_mining/
```

推荐配置：

```text
app/algorithm/scene_mining/config-qwen-gemini.yaml
```

算法基于 LangGraph，默认使用 DAG 模式：

```text
Supervisor 1
  -> Simple Workers: 时间、光照、天气、道路、特殊路段等基础类别
  -> Reducer
  -> Supervisor 2
  -> Complex Workers: 行人/非机动车异常、车辆轨迹冲突、极端高危事件
  -> Output Formatter
```

输出内容包括：

1. `pred`：每个类别的结构化标签。
2. `abnormal_event_times`：异常事件时间段。
3. `tags`：从 `pred` 中提取的检索标签。
4. `final_output`：完整类别输出。
5. `raw_outputs`：各 worker 原始输出，方便排查。

更详细的算法说明见：

```text
app/algorithm/scene_mining/README.md
```

## 检索能力

文本检索支持多种 `search_mode`：

| search_mode | 说明 |
| --- | --- |
| `frame` | 原有帧级检索 |
| `summary` | 按视频摘要语义检索 |
| `tags` / `tag_semantic` | 按场景标签语义检索 |
| `visual` | 将文本查询编码到多模态空间后，检索视频全局视觉向量 |

图片检索会将图片送入 Qwen3-VL 多模态嵌入模型，搜索 `video_visual_features`。视频视觉特征也来自同一个模型：系统先用 ffmpeg 抽取视频帧，再将帧图像编码为多模态向量并加权池化。

新增 Milvus collections：

| collection | 内容 |
| --- | --- |
| `video_text_features` | 每个视频两条文本特征：`tags` 和 `summary` |
| `video_visual_features` | 每个视频一条由采样帧加权池化得到的全局视觉特征 |

## 常用接口

### 异步处理视频

```bash
curl -X POST http://127.0.0.1:30501/api/add/task \
  -H 'Content-Type: application/json' \
  -d '{"video_url":"/media/perception-mining/example.mp4","action_type":3}'
```

查询任务：

```bash
curl http://127.0.0.1:30501/api/add/task/<task_id>
```

### 媒体代理健康检查

```bash
curl 'http://127.0.0.1:30501/api/media/health?url=/media/perception-mining/example.mp4'
```

预热缓存：

```bash
curl -X POST http://127.0.0.1:30501/api/media/health \
  -H 'Content-Type: application/json' \
  -d '{"urls":["/media/perception-mining/example.mp4"],"warm_cache":true}'
```

完整 API 见：

```text
docs/api.md
```

## 运行期清理

清理媒体缓存、任务状态、上传残留、上传锁和 scene mining 临时 clip：

```bash
python -m app.scripts.cleanup_runtime --dry-run
python -m app.scripts.cleanup_runtime --max-age-hours 24
```

默认不删除 `SCENE_MINING_VIDEO_CACHE_DIR`。确认不需要复用远程视频缓存时，可加：

```bash
python -m app.scripts.cleanup_runtime --max-age-hours 24 --include-video-cache
```

## 本地开发方式

本地直接运行 Flask 仍然可用，但需要你自行启动 Milvus、MinIO、VLM 和 Qwen3-VL 多模态嵌入服务：

```bash
conda create -y -p ./venv python=3.12
conda activate ./venv
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cp .env_sample .env
python app.py
```

生产环境建议使用 compose 中的 Gunicorn：

```bash
gunicorn --workers 2 --threads 8 \
  --bind 0.0.0.0:30501 \
  --timeout 3600 \
  wsgi:application
```

## 注意事项

1. `qwen-gemini-vlm` 和 `qwen3-vl-embedding` 默认只绑定 `127.0.0.1`，不暴露公网。
2. `vision-perception` 使用 host network，避免公网代理对 Docker bridge 端口转发不稳定。
3. 同卡部署时默认显存分配约为 VLM 0.7773、多模态嵌入服务 0.1727，可按机器显存调整。
4. ffmpeg/ffprobe 是视觉特征抽帧必需依赖，Docker 镜像内已安装。
5. 修改识别类别时，需要同步维护 `categories.json`、`skills/categories/` 和 `config-qwen-gemini.yaml` 的 `pipeline.category_order`。
