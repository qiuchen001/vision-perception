from pymilvus import MilvusClient
import os
from dotenv import load_dotenv

load_dotenv()

SERVER_HOST = os.getenv("SERVER_HOST")
uri = f"http://{SERVER_HOST}:19530"
milvus_client = MilvusClient(uri=uri, db_name=os.getenv("DB_NAME"))
collection_name = "video_collection"


def create_index():
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="IP",
        index_type="IVF_FLAT",
        index_name="vector_index",
        params={"nlist": 512}
    )

    milvus_client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )


if __name__ == "__main__":
    create_index()
