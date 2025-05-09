# 项目文档

## 项目概述
本项目是一个基于 Flask 框架的 Web 应用程序，主要用于视频分析和处理。项目实现了视频上传、视频摘要生成、视频行为分析等功能，并将分析结果存储在 Milvus 数据库中，以便后续检索和查询。

## 文档
- [API 文档](docs/api.md)：详细的 API 接口说明
- [数据流图](docs/data_flow.md)：系统组件和数据流向说明

## 功能模块

### 1. 视频上传与处理
- **视频上传**：用户可以通过 API 上传视频文件，并将其存储在 MinIO 对象存储中。
- **视频处理**：上传的视频文件会被提取帧并转换为 Base64 编码，用于后续的分析和摘要生成。

### 2. 视频摘要生成
- **摘要生成**：通过调用 OpenAI 的 API，对视频内容进行分析并生成摘要。生成的摘要信息包括视频中的关键行为和时间范围。

### 3. 视频行为分析
- **行为分析**：通过分析视频帧，识别视频中的常见驾驶行为和其他交通参与者的行为，并将分析结果以 JSON 格式输出。

### 4. 数据库管理
- **Milvus 数据库**：使用 Milvus 数据库存储视频的元数据和分析结果，支持通过标签或文本进行视频检索。

## 技术栈
- **Flask**：Web 框架用于构建 API 和处理请求。
- **MinIO**：对象存储服务，用于存储视频文件。
- **OpenAI**：用于视频内容分析和摘要生成。
- **Milvus**：向量数据库，用于存储和检索视频分析结果。

## 安装与运行

### 1. 环境准备
确保已安装以下依赖：
- Python 3.8+
- Flask
- MinIO
- OpenAI
- Milvus

### 2. 下载模型
项目需要以下模型文件，请下载并放置在对应目录：
```
models/                      # 模型文件目录
├── embedding/              # 向量嵌入模型
│   ├── cn-clip/           # 中文CLIP模型
│   └── bge-small-zh-1.5/  # BGE中文向量模型
```
注意：模型文件较大，不包含在代码仓库中，请从以下地址下载：

- BGE 中文向量模型：[BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
  - 下载后放置在 `models/embedding/bge-small-zh-1.5/` 目录
- 中文 CLIP 模型：[OFA-Sys/chinese-clip-vit-large-patch14-336px](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14-336px)
  - 下载模型文件 `clip_cn_vit-l-14-336.pt`
  - 下载后放置在 `models/embedding/cn-clip/` 目录

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置环境变量
1. 复制环境变量示例文件：
```bash
cp .env_sample .env
```

2. 编辑 `.env` 文件，填写必要的环境变量：
```ini
# 服务器配置
SERVER_HOST=localhost        # 服务器主机地址
SERVER_PORT=30501           # 服务器端口

# API密钥
DASHSCOPE_API_KEY=your_api_key    # DashScope API密钥

# MinIO配置
OSS_BUCKET_NAME=your_bucket_name   # MinIO存储桶名称
```

注意：请将示例值替换为实际的配置值。

### 5. 启动应用
```bash
python run.py
```

## API 参考

详细的 API 文档请参考 [API 文档](docs/api.md)。

### 主要接口
- 视频上传：上传视频文件到系统
- 视频添加：添加视频并进行分析
- 视频挖掘：分析视频中的行为
- 摘要生成：生成视频内容摘要
- 视频搜索：基于文本搜索视频


## 开发说明

### 错误处理
项目使用统一的错误处理机制：
1. 参数验证错误通过 ValueError 抛出
2. 所有异常都会被转换为统一的 JSON 响应格式
3. 错误日志会自动记录到日志文件中

### 响应处理
- 使用 @api_handler 装饰器统一处理所有 API 响应
- 成功响应使用 api_response() 函数封装
- 错误响应使用 error_response() 函数封装
