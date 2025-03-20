# 使用包含ffmpeg的基础镜像
FROM images.51vr.local:5000/bdp/base/python3.10-slim-ffmpeg:latest

# 设置用户
USER root

# 设置时区
ENV TZ Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 设置服务环境变量
ENV STORAGE_MANAGER_SERVICE http://10.66.12.37:30112

# 设置Python环境变量
ENV PYTHONUNBUFFERED 1

# 创建目录并设置权限
RUN mkdir -p /usr/lib/realtime_computing/flink && \
    chmod -R 777 /usr/lib/realtime_computing/flink

# 设置工作目录
WORKDIR /usr/lib/realtime_computing/flink

# 复制依赖文件
COPY requirements.txt ./
COPY ./custom_job_python_dependencies/* ./

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    pip install fastapi-0.74.1-py3-none-any.whl \
                uvicorn-0.17.5-py3-none-any.whl \
                urllib3-1.26.5-py2.py3-none-any.whl \
                redis-3.5.3-py2.py3-none-any.whl \
                bdp_api-2.0-py3-none-any.whl && \
    pip install --no-deps apache_skywalking-1.0.1-py3-none-any.whl

# 复制应用代码
COPY ./app ./app
COPY ./config ./config
COPY ./.env ./.env
COPY ./mining_wrapper.py ./custom_job_python.py

# 清理不需要的文件
RUN rm -rf *.whl && \
    rm -rf ~/.cache/pip/*

# docker build -t images.51vr.local:5000/bdp/default/rtflink/vision-perception-upload:new -f upload.Dockerfile .