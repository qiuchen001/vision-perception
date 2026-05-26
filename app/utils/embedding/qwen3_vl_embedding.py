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
    """HTTP client for the Qwen3-VL-Embedding vLLM sidecar service."""

    def __init__(self):
        self.base_url = os.getenv("QWEN3_VL_EMBEDDING_BASE_URL", "http://localhost:8575").rstrip("/")
        self.timeout = float(os.getenv("QWEN3_VL_EMBEDDING_TIMEOUT", "300"))
        self.retries = int(os.getenv("QWEN3_VL_EMBEDDING_RETRIES", "2"))
        self.retry_backoff = float(os.getenv("QWEN3_VL_EMBEDDING_RETRY_BACKOFF", "1.0"))
        self.expected_dim = int(os.getenv("QWEN3_VL_EMBEDDING_DIM", str(Config.QWEN3_VL_EMBEDDING_DIM)))

    def _post_embed(self, inputs: list[dict]) -> list[list[float]]:
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/embed",
                    json={"inputs": inputs, "normalize": True},
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

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise RuntimeError(f"Qwen3-VL embedding 服务返回异常: {data}")
        for index, embedding in enumerate(embeddings):
            if not isinstance(embedding, list) or not embedding:
                raise RuntimeError(f"Qwen3-VL embedding 第 {index} 个向量为空: {data}")
            if self.expected_dim > 0 and len(embedding) != self.expected_dim:
                raise RuntimeError(
                    f"Qwen3-VL embedding 维度异常: expected={self.expected_dim}, actual={len(embedding)}"
                )
        return embeddings

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def embedding_image(self, image: Image.Image) -> List[float]:
        return self._post_embed([{"image_base64": self._image_to_base64(image)}])[0]

    def embedding_text(self, text: str) -> List[float]:
        return self._post_embed([{"text": text or ""}])[0]

    def embedding(self, image: Image.Image, text: str) -> Tuple[List[float], List[float]]:
        embeddings = self._post_embed([
            {"image_base64": self._image_to_base64(image)},
            {"text": text or ""},
        ])
        return embeddings[0], embeddings[1]
