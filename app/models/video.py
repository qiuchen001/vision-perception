from pymilvus import DataType, db
from flask import current_app
from ..utils.logger import logger


class Video:
    def __init__(self, m_id, embedding, path, thumbnail_path, summary_txt, tags):
        self.m_id = m_id
        self.embedding = embedding
        self.path, = path,
        self.thumbnail_path = thumbnail_path
        self.summary_txt = summary_txt
        self.tags = tags

    @staticmethod
    def create_database():
        milvus_client = current_app.config['MILVUS_CLIENT']
        milvus_client.create_database("video_db")
        logger.info("当前数据库列表: %s", db.list_database())

    @staticmethod
    def create_schema():
        milvus_client = current_app.config['MILVUS_CLIENT']
        schema = milvus_client.create_schema(
            auto_id=False,
            enable_dynamic_fields=True,
            description="text to video retrieval by tag or txt",
        )

        schema.add_field(field_name="m_id", datatype=DataType.VARCHAR, is_primary=True, max_length=256,
                         description="唯一ID")
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=512,
                         description="视频详情embedding")
        schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=256, description="视频地址")
        schema.add_field(field_name="thumbnail_path", datatype=DataType.VARCHAR, max_length=256,
                         description="视频缩略图地址", nullable=True)
        schema.add_field(field_name="summary_txt", datatype=DataType.VARCHAR, max_length=3072, description="视频详情",
                         nullable=True)
        schema.add_field(field_name="tags", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10,
                         max_length=256, description="视频标签", nullable=True)

        return schema

    @staticmethod
    def create_collection(collection_name, schema):
        milvus_client = current_app.config['MILVUS_CLIENT']
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
        )

    @staticmethod
    def create_index(collection_name):
        milvus_client = current_app.config['MILVUS_CLIENT']

        params = {"nlist": 1024}
        milvus_client.create_index(collection_name, "embedding", "IVF_FLAT", "IP", params)
