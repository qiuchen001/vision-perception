# 使用公开 Python slim 镜像，并在镜像内安装 ffmpeg
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.12-slim

# 设置工作目录
WORKDIR /app

RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.huaweicloud.com/debian|g; s|http://security.debian.org/debian-security|https://mirrors.huaweicloud.com/debian-security|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 先复制requirements.txt
COPY ./requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY ./app ./app
COPY ./config ./config
COPY ./app.py ./app.py
COPY ./wsgi.py ./wsgi.py
COPY ./static ./static

# 设置环境变量
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV SCENE_MINING_CONFIG_PATH=/app/app/algorithm/scene_mining/config-qwen-gemini.yaml
ENV SCENE_MINING_OUTPUT_DIR=/app/outputs/scene_mining
ENV SCENE_MINING_VIDEO_CACHE_DIR=/app/data/scene_mining_videos
ENV QWEN3_VL_EMBEDDING_BASE_URL=http://qwen3-vl-embedding:8000
ENV EMBEDDING_MODEL=qwen3-vl
ENV MILVUS_VIDEO_TEXT_FEATURE_COLLECTION_NAME=video_text_features
ENV MILVUS_VIDEO_VISUAL_FEATURE_COLLECTION_NAME=video_visual_features

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["sh", "-c", "gunicorn --workers ${WEB_CONCURRENCY:-2} --threads ${GUNICORN_THREADS:-8} --bind ${SERVER_HOST:-0.0.0.0}:${SERVER_PORT:-5000} --timeout ${GUNICORN_TIMEOUT:-3600} --graceful-timeout 60 --access-logfile - --error-logfile - wsgi:application"]

# docker build -t images.51vr.local:5000/bdp/service/vision-perception-app:new -f app.Dockerfile .
# docker run -d -p 30500:5000 images.51vr.local:5000/bdp/service/vision-perception-app:new