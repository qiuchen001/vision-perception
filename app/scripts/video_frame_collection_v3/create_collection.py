import os
from pymilvus import DataType, MilvusClient
from dotenv import load_dotenv

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
milvus_client = MilvusClient(uri=uri, db_name=os.getenv("MILVUS_DB_NAME"))
collection_name = "video_frame_vector_v3"


def create_schema():
    collection_schema = milvus_client.create_schema(
        auto_id=False,
        enable_dynamic_fields=True,
        description="video frame embedding search"
    )

    collection_schema.add_field(
        field_name="m_id",
        datatype=DataType.VARCHAR,
        is_primary=True, max_length=256,
        description="唯一ID"
    )
    collection_schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=1024,
        description="视频帧embedding"
    )
    collection_schema.add_field(
        field_name="video_id",
        datatype=DataType.VARCHAR,
        max_length=256,
        description="视频ID"
    )
    collection_schema.add_field(
        field_name="resource_id",
        datatype=DataType.VARCHAR,
        max_length=256,
        description="资源ID"
    )
    collection_schema.add_field(
        field_name="at_seconds",
        datatype=DataType.INT32,
        description="视频时间点(秒)"
    )
    collection_schema.add_field(
        field_name="frame_url",
        datatype=DataType.VARCHAR,
        max_length=1024,
        description="帧图片OSS地址"
    )
    collection_schema.add_field(
        field_name="vconfig_id",
        datatype=DataType.VARCHAR,
        max_length=256,
        description="车辆类型标识",
        nullable=True
    )
    collection_schema.add_field(
        field_name="collect_start_time",
        datatype=DataType.INT64,
        description="采集开始时间(毫秒时间戳)",
        nullable=True
    )
    collection_schema.add_field(
        field_name="collect_end_time", 
        datatype=DataType.INT64,
        description="采集结束时间(毫秒时间戳)",
        nullable=True
    )

    return collection_schema


def create_collection(collection_schema):
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=collection_schema,
        shards_num=2
    )


if __name__ == "__main__":
    schema = create_schema()
    create_collection(schema)
