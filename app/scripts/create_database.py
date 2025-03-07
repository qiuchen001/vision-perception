from pymilvus import connections, db
import os
from dotenv import load_dotenv
load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
DB_NAME = os.getenv("MILVUS_DB_NAME")

conn = connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
database = db.create_database(DB_NAME)
db.using_database(DB_NAME)
print(db.list_database())
