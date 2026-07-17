import base64
import io
import os
import time
from typing import List, Tuple

import requests
from PIL import Image

from app.utils.embedding.embedding_base import EmbeddingBase
from app.utils.logger import logger
from config.config import Config


class Qwen3VLEmbedding(EmbeddingBase):
    """HTTP client for Qwen3-VL-Embedding via one-api/OpenAI-compatible APIs."""

    def __init__(self):
        self.base_url = os.getenv("QWEN3_VL_EMBEDDING_BASE_URL", "http://localhost:8575").rstrip("/")
        self.model_name = os.getenv("QWEN3_VL_EMBEDDING_MODEL_NAME", "Qwen3-VL-Embedding-8B")
        self.api_key = (
            os.getenv("QWEN3_VL_EMBEDDING_API_KEY")
            or os.getenv("ONE_API_KEY")
            or os.getenv("SCENE_MINING_API_KEY")
            or os.getenv("API_KEY")
            or "EMPTY"
        )
        self.timeout = float(os.getenv("QWEN3_VL_EMBEDDING_TIMEOUT", "300"))
        self.retries = int(os.getenv("QWEN3_VL_EMBEDDING_RETRIES", "2"))
        self.retry_backoff = float(os.getenv("QWEN3_VL_EMBEDDING_RETRY_BACKOFF", "1.0"))
        self.expected_dim = int(os.getenv("QWEN3_VL_EMBEDDING_DIM", str(Config.QWEN3_VL_EMBEDDING_DIM)))

    def _embeddings_url(self) -> str:
        if self.base_url.endswith("/embeddings"):
            return self.base_url
        return f"{self.base_url}/embeddings"

    def _post_one(self, payload: dict) -> list[float]:
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(
                    self._embeddings_url(),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    raise RuntimeError(f"Qwen3-VL embedding 服务请求失败: {exc}") from exc
                sleep_seconds = self.retry_backoff * (2 ** attempt)
                logger.warning("Qwen3-VL embedding request failed, retrying in %.1fs: %s", sleep_seconds, exc)
                time.sleep(sleep_seconds)
        else:
            raise RuntimeError(f"Qwen3-VL embedding 服务请求失败: {last_error}")

        embeddings = data.get("data")
        if not isinstance(embeddings, list) or not embeddings:
            raise RuntimeError(f"Qwen3-VL embedding 服务返回异常: {data}")
        embedding = embeddings[0].get("embedding") if isinstance(embeddings[0], dict) else None
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(f"Qwen3-VL embedding 向量为空: {data}")
        if self.expected_dim > 0 and len(embedding) != self.expected_dim:
            raise RuntimeError(
                f"Qwen3-VL embedding 维度异常: expected={self.expected_dim}, actual={len(embedding)}"
            )
        return embedding

    def _post_text(self, text: str) -> list[float]:
        return self._post_one({
            "model": self.model_name,
            "input": text or "",
            "encoding_format": "float",
        })

    def _post_image(self, image_base64: str) -> list[float]:
        return self._post_one({
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Represent the user's input."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        }
                    ],
                },
            ],
            "encoding_format": "float",
        })

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def embedding_image(self, image: Image.Image) -> List[float]:
        return self._post_image(self._image_to_base64(image))

    def embedding_text(self, text: str) -> List[float]:
        return self._post_text(text)

    def embedding(self, image: Image.Image, text: str) -> Tuple[List[float], List[float]]:
        return self.embedding_image(image), self.embedding_text(text)
