"""
创建视频集合脚本。

该模块用于在 Milvus 中创建视频集合，定义集合的schema和字段。
包含视频ID、向量embedding、路径、缩略图、摘要和标签等信息。
"""

import os
from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient

# 加载环境变量
load_dotenv()

# 配置 Milvus 连接
SERVER_HOST = os.getenv("SERVER_HOST")
MILVUS_URI = f"http://{SERVER_HOST}:19530"
COLLECTION_NAME = "video_collection"

milvus_client = MilvusClient(
    uri=MILVUS_URI,
    db_name=os.getenv("DB_NAME")
)


def create_schema():
    """
    创建视频集合的schema定义。

    Returns:
        CollectionSchema: Milvus集合的schema对象
    """
    collection_schema = milvus_client.create_schema(
        auto_id=False,
        enable_dynamic_fields=True,
        description="视频检索集合：存储视频的向量特征和元数据信息，支持通过文本描述、标签等多种方式检索视频。"
    )

    collection_schema.add_field(
        field_name="m_id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=256,
        description="唯一ID"
    )
    
    collection_schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=512,
        description="视频详情embedding"
    )
    
    collection_schema.add_field(
        field_name="path",
        datatype=DataType.VARCHAR,
        max_length=256,
        description="视频地址"
    )
    
    collection_schema.add_field(
        field_name="thumbnail_path",
        datatype=DataType.VARCHAR,
        max_length=256,
        description="视频缩略图地址",
        nullable=True
    )
    
    collection_schema.add_field(
        field_name="summary_txt",
        datatype=DataType.VARCHAR,
        max_length=3072,
        description="视频详情",
        nullable=True
    )
    
    collection_schema.add_field(
        field_name="tags",
        datatype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=10,
        max_length=256,
        description="视频标签",
        nullable=True
    )

    return collection_schema


def create_collection(collection_schema):
    """
    使用指定的schema创建视频集合。

    Args:
        collection_schema (CollectionSchema): 集合的schema定义
    """
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=collection_schema,
    )


if __name__ == "__main__":
    schema = create_schema()
    create_collection(schema)