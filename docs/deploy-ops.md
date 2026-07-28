# 运维部署手册

本文档面向运维侧，说明如何基于 Docker Compose 构建和运行 `vision-perception` 应用镜像。

## 1. 部署范围

当前 `docker-compose.yml` 只构建和运行应用服务：

| 服务 | 镜像 | 说明 |
| --- | --- | --- |
| `vision-perception` | `vision-perception-app:latest` | Flask/Gunicorn 应用，端口 `30012` |

以下外部服务需要提前部署并保持可用：

| 依赖 | 默认地址 | 说明 |
| --- | --- | --- |
| one-api | `http://127.0.0.1:30049/v1` | 转发 `Qwen3-VL-32B-Instruct` 和 `Qwen3-VL-Embedding-8B` |
| Milvus | `127.0.0.1:19530` | 视频、文本、视觉向量检索 |
| MinIO | 由 `.env` 配置 | 视频上传和媒体文件存储 |

模型服务统一由外部 one-api/vLLM 提供，本项目不再通过 Docker Compose 启动大模型。

## 2. 前置条件

部署机器需要具备：

1. Docker Engine 和 Docker Compose v2。
2. 可访问 one-api、Milvus、MinIO。
3. 可访问宿主机视频目录，默认路径为 `/mnt/data/ai-ground/dataset/videos`。
4. 项目根目录存在 `.env`，并已配置生产环境密钥和服务地址。

项目根目录：

```bash
/mnt/data/ai-ground/projects/vision-perception-proj/vision-perception
```

## 3. 环境变量准备

首次部署可从模板复制：

```bash
cd /mnt/data/ai-ground/projects/vision-perception-proj/vision-perception
cp .env_sample .env
```

重点检查以下配置：

```ini
# 应用端口
SERVER_PORT=30012

# 浏览器 session 登录鉴权
SESSION_AUTH_ENABLED=true
SESSION_SECRET=change-me
SESSION_MAX_AGE_SECONDS=28800
SESSION_COOKIE_SECURE=false
ADMIN_USERNAME=admin
ADMIN_PASSWORD_HASH='pbkdf2_sha256$...'
BROWSER_ALLOWED_ORIGINS=http://localhost:30012,http://127.0.0.1:30012

# one-api
ONE_API_KEY=xxx
SCENE_MINING_API_BASE_URL=http://127.0.0.1:30049/v1
SCENE_MINING_API_MODEL_NAME=Qwen3-VL-32B-Instruct
QWEN3_VL_EMBEDDING_BASE_URL=http://127.0.0.1:30049/v1
QWEN3_VL_EMBEDDING_MODEL_NAME=Qwen3-VL-Embedding-8B
QWEN3_VL_EMBEDDING_DIM=4096

# Milvus
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530

# MinIO
OSS_ENDPOINT=xxx
OSS_ACCESS_KEY=xxx
OSS_SECRET_KEY=xxx
OSS_BUCKET_NAME=perception-mining

# 场景挖掘缓存和媒体 URL
SCENE_MINING_MEDIA_BASE_URL=http://127.0.0.1:30012
SCENE_MINING_MEDIA_SIGN_SECRET=change-me
SCENE_MINING_MEDIA_URL_TTL_SECONDS=7200

# 直接 URL 挖掘接口签名密钥
DIRECT_MINING_SIGN_SECRET=change-me
DIRECT_MINING_SIGN_WINDOW_SECONDS=300
```

`SCENE_MINING_MEDIA_BASE_URL` 必须配置成 vLLM 推理服务可访问的应用地址。当前 Compose 使用 `network_mode: host`，本机 vLLM 通常可使用 `http://127.0.0.1:30012`。

启用 session 登录前，先生成管理员密码哈希：

```bash
python app/utils/generate_password_hash.py
```

将输出写入 `.env` 的 `ADMIN_PASSWORD_HASH`，建议用单引号包裹整个哈希值，避免 `$` 被环境变量解析逻辑处理。`SESSION_SECRET` 必须使用生产环境随机字符串，并在多次重启间保持不变。

## 4. 构建镜像

Compose 已定义镜像构建方式：

```yaml
vision-perception:
  build:
    context: .
    dockerfile: app.Dockerfile
  image: vision-perception-app:latest
```

构建命令：

```bash
cd /mnt/data/ai-ground/projects/vision-perception-proj/vision-perception
docker compose build vision-perception
```

构建完成后确认镜像：

```bash
docker images | grep vision-perception-app
```

## 5. 启动服务

后台启动：

```bash
cd /mnt/data/ai-ground/projects/vision-perception-proj/vision-perception
docker compose up -d vision-perception
```

构建并启动可合并为：

```bash
docker compose up -d --build vision-perception
```

服务默认以 host 网络运行，应用监听：

```text
http://<部署机器IP>:30012
```

## 6. 查看状态和日志

查看容器状态：

```bash
docker compose ps
```

查看应用日志：

```bash
docker compose logs -f vision-perception
```

查看最近日志：

```bash
docker compose logs --tail=200 vision-perception
```

## 7. 健康检查

确认应用可访问：

```bash
curl -s http://127.0.0.1:30012/api/mining/tags | head
```

确认 API 文档可访问：

```bash
curl -I http://127.0.0.1:30012/docs
```

如果需要验证直接 URL 挖掘接口，请参考 `docs/direct-mining-url-stream-api.md` 生成签名请求。

## 8. 更新发布

代码或镜像内配置变更后执行：

```bash
cd /mnt/data/ai-ground/projects/vision-perception-proj/vision-perception
docker compose build vision-perception
docker compose up -d vision-perception
```

仅 `.env` 或 Compose 环境变量变更时，通常不需要重新构建镜像，直接重建容器即可：

```bash
docker compose up -d --force-recreate vision-perception
```

## 9. 停止和重启

重启应用：

```bash
docker compose restart vision-perception
```

停止应用：

```bash
docker compose stop vision-perception
```

停止并删除容器网络：

```bash
docker compose down
```

## 10. 数据目录

Compose 默认挂载：

| 宿主机路径 | 容器路径 | 说明 |
| --- | --- | --- |
| `/mnt/data/ai-ground/dataset/videos` | `/app/videos` | 只读视频目录 |
| `./outputs` | `/app/outputs` | 场景挖掘输出和切片 |
| `./data` | `/app/data` | 缓存、状态文件 |

直接 URL 挖掘下载的视频缓存默认位于：

```text
./data/scene_mining_videos
```

场景挖掘切片默认位于：

```text
./outputs/scene_mining/_tool_clips
```

## 11. 常见问题

### vLLM 无法读取视频

确认 `SCENE_MINING_MEDIA_BASE_URL` 是 vLLM 可访问的地址，并检查应用日志中是否有：

```text
GET /api/scene-mining/media/... 200
```

### one-api 调用失败

检查：

1. `SCENE_MINING_API_BASE_URL`
2. `SCENE_MINING_API_MODEL_NAME`
3. `QWEN3_VL_EMBEDDING_BASE_URL`
4. `QWEN3_VL_EMBEDDING_MODEL_NAME`
5. `ONE_API_KEY`

### 修改 `.env` 后未生效

重新创建容器：

```bash
docker compose up -d --force-recreate vision-perception
```
