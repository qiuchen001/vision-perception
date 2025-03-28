from pymilvus import MilvusClient
import os
from dotenv import load_dotenv

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
milvus_client = MilvusClient(uri=uri, db_name=os.getenv("MILVUS_DB_NAME"))
collection_name = "video_collection"


def create_index():
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="IP",
        index_type="IVF_FLAT",
        index_name="embedding_index",
        params={"nlist": 1024}
    )

    index_params.add_index(
        field_name="summary_embedding",
        metric_type="IP",
        index_type="IVF_FLAT",
        index_name="summary_embedding_index",
        params={"nlist": 1024}
    )

    milvus_client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )


if __name__ == "__main__":
    create_index()
