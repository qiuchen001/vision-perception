from sentence_transformers import SentenceTransformer
import os, json
from openai import OpenAI
from dotenv import load_dotenv
from app.utils.logger import logger
load_dotenv()


def embed_fn(text):
    """生成文本的embedding向量"""
    try:
        client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )

        response = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL_NAME"),
            input=text,
            dimensions=512,
            encoding_format="float"
        )

        response_json = response.model_dump_json()
        js = json.loads(response_json)

        embedding = js['data'][0]['embedding']
        logger.info(f"成功生成embedding,维度:{len(embedding)}")
        return embedding

    except Exception as e:
        logger.error(f"生成embedding失败:{str(e)}")
        raise e


