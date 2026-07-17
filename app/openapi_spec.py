from __future__ import annotations


OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "Vision Perception API",
        "version": "1.0.0",
        "description": "车载视频感知、场景挖掘、摘要生成与多模态检索接口文档。",
    },
    "servers": [
        {"url": "/", "description": "Current host"},
    ],
    "tags": [
        {"name": "Pages", "description": "静态页面"},
        {"name": "Upload", "description": "视频上传与媒体检查"},
        {"name": "Processing", "description": "视频处理与场景挖掘"},
        {"name": "Mining", "description": "场景挖掘标签配置"},
        {"name": "Search", "description": "视频检索"},
    ],
    "paths": {
        "/": {
            "get": {
                "tags": ["Pages"],
                "summary": "系统首页",
                "responses": {"200": {"description": "返回 index.html"}},
            }
        },
        "/upload": {
            "get": {
                "tags": ["Pages"],
                "summary": "上传页面",
                "responses": {"200": {"description": "返回 upload.html"}},
            }
        },
        "/process": {
            "get": {
                "tags": ["Pages"],
                "summary": "处理页面",
                "responses": {"200": {"description": "返回 process.html"}},
            }
        },
        "/search": {
            "get": {
                "tags": ["Pages"],
                "summary": "搜索页面",
                "responses": {"200": {"description": "返回 search.html"}},
            }
        },
        "/api/upload/config": {
            "get": {
                "tags": ["Upload"],
                "summary": "获取上传/处理配置",
                "responses": {
                    "200": {
                        "description": "当前最大并发和场景挖掘配置路径",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/UploadConfigResponse"}}},
                    }
                },
            }
        },
        "/api/upload": {
            "post": {
                "tags": ["Upload"],
                "summary": "上传视频文件",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "required": ["file"],
                                "properties": {
                                    "file": {"type": "string", "format": "binary", "description": "视频文件"},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {"description": "上传成功", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/UploadResponse"}}}},
                    "400": {"description": "参数错误"},
                    "500": {"description": "上传失败"},
                },
            }
        },
        "/api/upload/chunk": {
            "post": {
                "tags": ["Upload"],
                "summary": "分片上传视频",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "required": ["chunk", "chunk_index", "total_chunks", "file_name"],
                                "properties": {
                                    "chunk": {"type": "string", "format": "binary", "description": "分片文件，也兼容字段名 file"},
                                    "upload_id": {"type": "string", "description": "上传会话 ID；不传则自动生成"},
                                    "chunk_index": {"type": "integer", "minimum": 0, "description": "当前分片索引，从 0 开始"},
                                    "total_chunks": {"type": "integer", "minimum": 1, "description": "总分片数"},
                                    "file_name": {"type": "string", "description": "原始文件名"},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {"description": "分片接收中或合并上传成功"},
                    "400": {"description": "参数错误"},
                    "500": {"description": "处理失败"},
                },
            }
        },
        "/api/process": {
            "post": {
                "tags": ["Processing"],
                "summary": "按 raw_id 导入/处理视频",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["raw_id"],
                                "properties": {"raw_id": {"type": "string", "description": "原始数据 ID"}},
                            }
                        }
                    },
                },
                "responses": {
                    "200": {"description": "处理成功", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/UploadResponse"}}}},
                    "400": {"description": "参数错误"},
                    "500": {"description": "处理失败"},
                },
            }
        },
        "/api/mining/tags": {
            "get": {
                "tags": ["Mining"],
                "summary": "获取可挖掘标签列表",
                "description": "返回当前场景挖掘配置中的标签分类、扁平标签列表和统计信息。",
                "responses": {
                    "200": {
                        "description": "标签列表",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/MiningTagsResponse"}}},
                    },
                    "500": {"description": "读取标签配置失败", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                },
            }
        },
        "/api/mining/url/stream": {
            "post": {
                "tags": ["Mining"],
                "summary": "直接基于可下载视频 URL 流式执行标签挖掘",
                "description": "下载 http/https 视频到本地场景挖掘缓存后调用 VLM 分析，只返回挖掘结果，不上传 MinIO、不写入视频库或 Milvus。",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/DirectMiningUrlRequest"}}},
                },
                "parameters": [
                    {
                        "name": "X-Timestamp",
                        "in": "header",
                        "required": True,
                        "schema": {"type": "integer"},
                        "description": "秒级 Unix 时间戳，参与签名并用于过期校验",
                    },
                    {
                        "name": "X-Nonce",
                        "in": "header",
                        "required": True,
                        "schema": {"type": "string", "maxLength": 128},
                        "description": "随机 nonce，参与签名并用于防重放",
                    },
                    {
                        "name": "X-Content-SHA256",
                        "in": "header",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "原始请求体 UTF-8 字节的 SHA256 小写十六进制摘要",
                    },
                    {
                        "name": "X-Signature",
                        "in": "header",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "HMAC-SHA256 小写十六进制签名，签名原文为 HTTP_METHOD + 换行 + REQUEST_PATH + 换行 + X-Timestamp + 换行 + X-Nonce + 换行 + X-Content-SHA256",
                    },
                ],
                "responses": {
                    "200": {"description": "NDJSON 进度流；每行均为 code/msg/data 结构，最终 data.type=result 时返回 tags、pred 和 abnormal_event_times", "content": {"application/x-ndjson": {"schema": {"type": "string"}}}},
                    "400": {"description": "参数错误", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    "401": {"description": "签名错误、请求过期或 nonce 重复", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    "500": {"description": "服务端配置错误或挖掘过程失败", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                },
            }
        },
        "/api/add": {
            "post": {
                "tags": ["Processing"],
                "summary": "同步执行视频场景挖掘/摘要/特征处理",
                "description": "会执行耗时 VLM 和 embedding 流程；长视频建议使用 /api/add/task 或 /api/add/stream。",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/AddVideoRequest"}}},
                },
                "responses": {
                    "200": {"description": "处理成功", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/AddVideoResponse"}}}},
                    "400": {"description": "参数错误"},
                    "500": {"description": "处理失败"},
                },
            }
        },
        "/api/add/task": {
            "post": {
                "tags": ["Processing"],
                "summary": "提交异步视频处理任务",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/AddVideoRequest"}}},
                },
                "responses": {
                    "202": {"description": "任务已提交", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/AddTaskResponse"}}}},
                    "400": {"description": "参数错误"},
                },
            }
        },
        "/api/add/task/{task_id}": {
            "get": {
                "tags": ["Processing"],
                "summary": "查询异步视频处理任务状态",
                "parameters": [
                    {"name": "task_id", "in": "path", "required": True, "schema": {"type": "string"}},
                ],
                "responses": {
                    "200": {"description": "任务状态", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/TaskStatusResponse"}}}},
                    "404": {"description": "任务不存在"},
                },
            }
        },
        "/api/add/stream": {
            "post": {
                "tags": ["Processing"],
                "summary": "流式执行视频处理并返回 NDJSON 进度",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/AddVideoRequest"}}},
                },
                "responses": {
                    "200": {"description": "NDJSON 进度流", "content": {"application/x-ndjson": {"schema": {"type": "string"}}}},
                    "400": {"description": "参数错误"},
                },
            }
        },
        "/api/search": {
            "post": {
                "tags": ["Search"],
                "summary": "视频检索",
                "description": "支持 smart、text、image、tags、filter 五类检索。",
                "requestBody": {
                    "required": True,
                    "content": {"multipart/form-data": {"schema": {"$ref": "#/components/schemas/SearchRequest"}}},
                },
                "responses": {
                    "200": {"description": "检索结果", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SearchResponse"}}}},
                    "400": {"description": "参数错误"},
                    "500": {"description": "检索失败"},
                },
            }
        },
        "/api/media/health": {
            "get": {
                "tags": ["Upload"],
                "summary": "检查媒体对象是否存在",
                "parameters": [
                    {"name": "url", "in": "query", "required": True, "schema": {"type": "array", "items": {"type": "string"}}, "style": "form", "explode": True},
                    {"name": "warm_cache", "in": "query", "schema": {"type": "boolean", "default": False}},
                ],
                "responses": {"200": {"description": "媒体检查结果"}, "400": {"description": "缺少参数"}},
            },
            "post": {
                "tags": ["Upload"],
                "summary": "批量检查媒体对象是否存在",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "urls": {"type": "array", "items": {"type": "string"}},
                                    "warm_cache": {"type": "boolean", "default": False},
                                },
                            }
                        }
                    },
                },
                "responses": {"200": {"description": "媒体检查结果"}, "400": {"description": "缺少参数"}},
            },
        },
        "/media/{bucket_name}/{object_name}": {
            "get": {
                "tags": ["Upload"],
                "summary": "代理访问 MinIO 媒体文件",
                "parameters": [
                    {"name": "bucket_name", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "object_name", "in": "path", "required": True, "schema": {"type": "string"}},
                ],
                "responses": {"200": {"description": "媒体文件"}, "404": {"description": "对象不存在"}, "502": {"description": "缓存失败"}},
            }
        },
    },
    "components": {
        "schemas": {
            "UploadConfigResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "example": "success"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "max_concurrent_videos": {"type": "integer", "example": 10},
                            "source": {"type": "string"},
                        },
                    },
                },
            },
            "UploadResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "example": "success"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "file_name": {"type": "string"},
                            "video_url": {"type": "string"},
                            "title": {"type": "string"},
                        },
                    },
                },
            },
            "AddVideoRequest": {
                "type": "object",
                "required": ["video_url"],
                "properties": {
                    "video_url": {"type": "string", "description": "视频 URL，通常为 /media/<bucket>/<object>"},
                    "action_type": {
                        "type": "integer",
                        "enum": [1, 2, 3],
                        "default": 3,
                        "description": "1=视频内容挖掘，2=视频内容总结，3=内容挖掘和总结",
                    },
                },
            },
            "MiningTagsResponse": {
                "type": "object",
                "properties": {
                    "code": {"type": "integer", "example": 0},
                    "msg": {"type": "string", "example": "success"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "categories": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "category": {"type": "string", "example": "气象条件"},
                                        "tags": {"type": "array", "items": {"type": "string"}, "example": ["晴天", "阴天", "雨天"]},
                                        "count": {"type": "integer", "example": 6},
                                    },
                                },
                            },
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "category_count": {"type": "integer", "example": 10},
                            "tag_count": {"type": "integer", "example": 66},
                        },
                    },
                },
            },
            "ErrorResponse": {
                "type": "object",
                "properties": {
                    "code": {"type": "integer", "example": 400},
                    "msg": {"type": "string", "example": "参数错误"},
                    "data": {"nullable": True, "example": None},
                },
            },
            "DirectMiningUrlRequest": {
                "type": "object",
                "required": ["video_url"],
                "properties": {
                    "video_url": {
                        "type": "string",
                        "format": "uri",
                        "description": "可直接下载的视频 http/https 地址",
                        "example": "https://example.com/video.mp4",
                    },
                },
            },
            "AddVideoResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "example": "success"},
                    "data": {"$ref": "#/components/schemas/VideoResult"},
                },
            },
            "AddTaskResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "example": "success"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "status_url": {"type": "string"},
                        },
                    },
                },
            },
            "TaskStatusResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "example": "success"},
                    "data": {"type": "object"},
                },
            },
            "SearchRequest": {
                "type": "object",
                "required": ["search_type"],
                "properties": {
                    "search_type": {"type": "string", "enum": ["smart", "text", "image", "tags", "filter"]},
                    "text_query": {"type": "string", "description": "文本搜索关键词"},
                    "search_mode": {"type": "string", "enum": ["frame", "summary", "tags", "tag_semantic", "visual", "exact", "semantic"]},
                    "image_file": {"type": "string", "format": "binary", "description": "图片搜索文件"},
                    "image_url": {"type": "string", "description": "图片 URL"},
                    "tags": {"type": "string", "description": "逗号分隔标签"},
                    "page": {"type": "integer", "default": 1},
                    "page_size": {"type": "integer", "default": 6, "maximum": 50},
                    "top_k": {"type": "integer", "default": 10, "maximum": 100},
                    "vconfig_id": {"type": "string"},
                    "collect_start_time": {"type": "integer"},
                    "collect_end_time": {"type": "integer"},
                },
            },
            "SearchResponse": {
                "type": "object",
                "properties": {
                    "msg": {"type": "string", "example": "success"},
                    "code": {"type": "integer", "example": 0},
                    "data": {
                        "type": "object",
                        "properties": {
                            "total": {"type": "integer"},
                            "list": {"type": "array", "items": {"$ref": "#/components/schemas/VideoResult"}},
                        },
                    },
                },
            },
            "VideoResult": {
                "type": "object",
                "properties": {
                    "m_id": {"type": "string"},
                    "title": {"type": "string"},
                    "video_url": {"type": "string"},
                    "thumbnail_url": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"},
                    "timestamp": {"type": "integer"},
                    "similarity": {"type": "string"},
                    "vconfig_id": {"type": "string"},
                    "collect_start_time": {"type": "integer", "nullable": True},
                    "collect_end_time": {"type": "integer", "nullable": True},
                    "feature_type": {"type": "string", "nullable": True},
                    "sampled_seconds": {"type": "string", "nullable": True},
                    "sampled_frame_count": {"type": "integer", "nullable": True},
                },
            },
        }
    },
}
