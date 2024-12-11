import os
from pymilvus import DataType, MilvusClient
from dotenv import load_dotenv

load_dotenv()

SERVER_HOST = os.getenv("SERVER_HOST")
uri = f"http://{SERVER_HOST}:19530"
milvus_client = MilvusClient(uri=uri, db_name=os.getenv("DB_NAME"))
collection_name = "video_frame_vector"


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
        dim=768,
        description="视频帧embedding"
    )
    collection_schema.add_field(
        field_name="video_id", 
        datatype=DataType.VARCHAR, 
        max_length=256,
        description="视频ID"
    )
    collection_schema.add_field(
        field_name="at_seconds", 
        datatype=DataType.INT32,
        description="视频时间点(秒)"
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