# 使用包含ffmpeg的基础镜像
FROM images.51vr.local:5000/bdp/base/python3.10-slim-ffmpeg:latest

# 设置工作目录
WORKDIR /app

# 先复制requirements.txt
COPY ./requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY ./app ./app
COPY ./config ./config
COPY ./.env ./.env
COPY ./app.py ./app.py
COPY ./static ./static

# 设置环境变量
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "app.py"]

# docker build -t images.51vr.local:5000/bdp/service/vision-perception-app:new -f app.Dockerfile .
# docker run -d -p 30500:5000 images.51vr.local:5000/bdp/service/vision-perception-app:new