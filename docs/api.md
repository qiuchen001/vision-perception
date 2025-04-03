# 视频处理系统API文档

## 目录
- [页面接口](#页面接口)
- [API接口](#api接口)
- [错误码说明](#错误码说明)
- [调用示例](#调用示例)

## 页面接口

### 1. 主页
- **路径:** `/`
- **方法:** `GET`
- **描述:** 返回系统主页
- **响应:** 返回index.html页面

### 2. 上传页面
- **路径:** `/upload`
- **方法:** `GET` 
- **描述:** 返回视频上传页面
- **响应:** 返回upload.html页面

### 3. 处理页面
- **路径:** `/process`
- **方法:** `GET`
- **描述:** 返回视频处理页面
- **响应:** 返回process.html页面

### 4. 搜索页面
- **路径:** `/search`
- **方法:** `GET`
- **描述:** 返回视频搜索页面
- **响应:** 返回search.html页面

## API接口

### 1. 上传视频
- **路径:** `/api/upload`
- **方法:** `POST`
- **Content-Type:** `multipart/form-data`
- **参数:**
  | 参数名 | 类型 | 必填 | 描述 |
  |--------|------|------|------|
  | file | File | 是 | 要上传的视频文件 |

- **响应格式:**
```json
{
    "status": "success",
    "data": {
        "file_name": "视频文件名",
        "video_url": "视频URL",
        "title": "视频标题"
    }
}
```

### 2. 处理视频
- **路径:** `/api/process`
- **方法:** `POST`
- **Content-Type:** `application/json`
- **请求体:**
```json
{
    "raw_id": "视频资源ID"
}
```
- **响应格式:**
```json
{
    "status": "success",
    "data": {
        "file_name": "视频文件名",
        "video_url": "视频URL",
        "title": "视频标题"
    }
}
```

### 3. 搜索视频
- **路径:** `/api/search`
- **方法:** `POST`
- **Content-Type:** `multipart/form-data`
- **参数:**
  | 参数名 | 类型 | 必填 | 描述 |
  |--------|------|------|------|
  | search_type | String | 是 | 搜索类型: smart(智能搜索)/text(文本搜索)/image(图片搜索)/tags(标签搜索) |
  | text_query | String | 否 | 文本搜索关键词 |
  | search_mode | String | 否 | 文本搜索模式: frame(帧搜索) |
  | image_file | File | 否 | 图片文件(图片搜索时使用) |
  | image_url | String | 否 | 图片URL(图片搜索时使用) |
  | tags | String | 否 | 标签列表(逗号分隔) |
  | page | Integer | 否 | 页码(默认1) |
  | page_size | Integer | 否 | 每页数量(默认6) |

- **响应格式:**
```json
{
    "status": "success",
    "data": [
        {
            "title": "视频标题",
            "video_url": "视频URL",
            "thumbnail_url": "缩略图URL",
            "tags": ["标签1", "标签2"],
            "summary": "视频摘要",
            "timestamp": 1234567890,
            "similarity": "0.8500"
        }
    ]
}
```

### 4. 添加视频
- **路径:** `/api/add`
- **方法:** `POST`
- **Content-Type:** `application/json`
- **请求体:**
```json
{
    "video_url": "视频URL",
    "action_type": 1
}
```
- **参数说明:**
  - action_type:
    - 1: 视频内容挖掘
    - 2: 视频内容总结
    - 3: 内容挖掘和总结

- **响应格式:**
```json
{
    "status": "success",
    "data": {
        "video_url": "视频URL",
        "action_type_desc": "处理类型描述",
        "m_id": "视频ID"
    }
}
```

## 错误码说明

所有接口在发生错误时会返回以下格式:
```json
{
    "status": "error",
    "message": "错误信息描述"
}
```

常见错误码:
- 400: 请求参数错误
- 404: 资源不存在
- 500: 服务器内部错误

## 调用示例

### 1. 上传视频

#### CURL示例
```bash
# 上传视频文件
curl -X POST http://localhost:5000/api/upload \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/video.mp4"
```

#### Python示例
```python
import requests

def upload_video(file_path):
    url = "http://localhost:5000/api/upload"
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"上传成功: {result}")
        return result
    else:
        print(f"上传失败: {response.text}")
        return None

# 调用示例
result = upload_video("/path/to/video.mp4")
```

### 2. 处理视频

#### CURL示例
```bash
# 处理视频
curl -X POST http://localhost:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{"raw_id": "your-video-id"}'
```

#### Python示例
```python
import requests

def process_video(raw_id):
    url = "http://localhost:5000/api/process"
    data = {"raw_id": raw_id}
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"处理成功: {result}")
        return result
    else:
        print(f"处理失败: {response.text}")
        return None

# 调用示例
result = process_video("your-video-id")
```

### 3. 搜索视频

#### CURL示例
```bash
# 文本搜索
curl -X POST http://localhost:5000/api/search \
  -F "search_type=text" \
  -F "text_query=关键词" \
  -F "page=1" \
  -F "page_size=6"

# 图片搜索
curl -X POST http://localhost:5000/api/search \
  -F "search_type=image" \
  -F "image_file=@/path/to/image.jpg"

# 标签搜索
curl -X POST http://localhost:5000/api/search \
  -F "search_type=tags" \
  -F "tags=标签1,标签2"
```

#### Python示例
```python
import requests

def search_by_text(query, page=1, page_size=6):
    url = "http://localhost:5000/api/search"
    data = {
        "search_type": "text",
        "text_query": query,
        "page": page,
        "page_size": page_size
    }
    
    response = requests.post(url, data=data)
    return response.json() if response.status_code == 200 else None

def search_by_image(image_path):
    url = "http://localhost:5000/api/search"
    
    with open(image_path, 'rb') as f:
        files = {'image_file': f}
        data = {"search_type": "image"}
        response = requests.post(url, data=data, files=files)
    
    return response.json() if response.status_code == 200 else None

def search_by_tags(tags):
    url = "http://localhost:5000/api/search"
    data = {
        "search_type": "tags",
        "tags": ",".join(tags)
    }
    
    response = requests.post(url, data=data)
    return response.json() if response.status_code == 200 else None

# 调用示例
text_results = search_by_text("搜索关键词")
image_results = search_by_image("/path/to/image.jpg")
tag_results = search_by_tags(["标签1", "标签2"])
```

### 4. 添加视频

#### CURL示例
```bash
# 添加视频
curl -X POST http://localhost:5000/api/add \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "http://example.com/video.mp4",
    "action_type": 1
  }'
```

#### Python示例
```python
import requests

def add_video(video_url, action_type):
    url = "http://localhost:5000/api/add"
    data = {
        "video_url": video_url,
        "action_type": action_type
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"添加成功: {result}")
        return result
    else:
        print(f"添加失败: {response.text}")
        return None

# 调用示例
result = add_video(
    video_url="http://example.com/video.mp4",
    action_type=1  # 1:内容挖掘, 2:内容总结, 3:挖掘和总结
)
```
