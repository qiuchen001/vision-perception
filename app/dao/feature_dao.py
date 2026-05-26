import os
import time
import uuid
from typing import Any, Dict, List, Optional

from pymilvus import DataType, MilvusClient

from config.config import Config


class FeatureDAO:
    TEXT_MAX_LENGTH = 8192

    def __init__(self):
        milvus_host = os.getenv("MILVUS_HOST")
        milvus_port = os.getenv("MILVUS_PORT")
        self.client = MilvusClient(
            uri=f"http://{milvus_host}:{milvus_port}",
            db_name=os.getenv("MILVUS_DB_NAME"),
        )
        self.text_collection = Config.MILVUS_VIDEO_TEXT_FEATURE_COLLECTION_NAME
        self.visual_collection = Config.MILVUS_VIDEO_VISUAL_FEATURE_COLLECTION_NAME
        self.dim = Config.QWEN3_VL_EMBEDDING_DIM

    def ensure_collections(self) -> None:
        if not self.client.has_collection(self.text_collection):
            self._create_text_collection()
        self.client.load_collection(self.text_collection)

        if not self.client.has_collection(self.visual_collection):
            self._create_visual_collection()
        self.client.load_collection(self.visual_collection)

    def _create_text_collection(self) -> None:
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_fields=True,
            description="视频文本特征集合：保存 tags 与 summary 两类语义向量。",
        )
        schema.add_field("feature_id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("m_id", DataType.VARCHAR, max_length=256)
        schema.add_field("resource_id", DataType.VARCHAR, max_length=256, nullable=True)
        schema.add_field("path", DataType.VARCHAR, max_length=1024)
        schema.add_field("feature_type", DataType.VARCHAR, max_length=32)
        schema.add_field("text", DataType.VARCHAR, max_length=8192, nullable=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("vconfig_id", DataType.VARCHAR, max_length=256, nullable=True)
        schema.add_field("collect_start_time", DataType.INT64, nullable=True)
        schema.add_field("collect_end_time", DataType.INT64, nullable=True)
        schema.add_field("created_at", DataType.INT64)
        self.client.create_collection(self.text_collection, schema=schema)
        self._create_vector_index(self.text_collection, "embedding", "text_embedding_index")

    def _create_visual_collection(self) -> None:
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_fields=True,
            description="视频全局视觉特征集合：每个视频一条由采样帧聚合得到的向量。",
        )
        schema.add_field("visual_feature_id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("m_id", DataType.VARCHAR, max_length=256)
        schema.add_field("resource_id", DataType.VARCHAR, max_length=256, nullable=True)
        schema.add_field("path", DataType.VARCHAR, max_length=1024)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("sampling_policy", DataType.VARCHAR, max_length=1024, nullable=True)
        schema.add_field("sampled_seconds", DataType.VARCHAR, max_length=8192, nullable=True)
        schema.add_field("sampled_frame_count", DataType.INT64)
        schema.add_field("vconfig_id", DataType.VARCHAR, max_length=256, nullable=True)
        schema.add_field("collect_start_time", DataType.INT64, nullable=True)
        schema.add_field("collect_end_time", DataType.INT64, nullable=True)
        schema.add_field("created_at", DataType.INT64)
        self.client.create_collection(self.visual_collection, schema=schema)
        self._create_vector_index(self.visual_collection, "embedding", "visual_embedding_index")

    def _create_vector_index(self, collection_name: str, field_name: str, index_name: str) -> None:
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name=field_name,
            metric_type="COSINE",
            index_type="FLAT",
            index_name=index_name,
            params={},
        )
        self.client.create_index(collection_name=collection_name, index_params=index_params)

    def upsert_text_features(self, video: Dict[str, Any], tags_text: str, summary_text: str, tags_embedding: List[float], summary_embedding: List[float]) -> None:
        now = int(time.time() * 1000)
        common = self._common_fields(video, now)
        data = [
            {
                **common,
                "feature_id": self._text_feature_id(video["m_id"], "tags"),
                "feature_type": "tags",
                "text": self._truncate_varchar(tags_text, self.TEXT_MAX_LENGTH),
                "embedding": tags_embedding,
            },
            {
                **common,
                "feature_id": self._text_feature_id(video["m_id"], "summary"),
                "feature_type": "summary",
                "text": self._truncate_varchar(summary_text, self.TEXT_MAX_LENGTH),
                "embedding": summary_embedding,
            },
        ]
        self.client.upsert(self.text_collection, data)

    def upsert_visual_feature(self, video: Dict[str, Any], embedding: List[float], sampling_policy: str, sampled_seconds: str, sampled_frame_count: int) -> None:
        now = int(time.time() * 1000)
        data = {
            **self._common_fields(video, now),
            "visual_feature_id": self._visual_feature_id(video["m_id"]),
            "embedding": embedding,
            "sampling_policy": sampling_policy,
            "sampled_seconds": sampled_seconds,
            "sampled_frame_count": sampled_frame_count,
        }
        self.client.upsert(self.visual_collection, [data])

    def search_text_features(self, embedding: List[float], feature_type: str, limit: int, **filter_params) -> List[Dict[str, Any]]:
        expr_parts = [f'feature_type == "{feature_type}"']
        extra = self._build_filter_expression(filter_params)
        if extra:
            expr_parts.append(extra)
        return self._search(self.text_collection, embedding, limit, " and ".join(expr_parts), [
            "feature_id", "m_id", "resource_id", "path", "feature_type", "text",
            "vconfig_id", "collect_start_time", "collect_end_time",
        ])

    def search_visual_features(self, embedding: List[float], limit: int, **filter_params) -> List[Dict[str, Any]]:
        expr = self._build_filter_expression(filter_params)
        return self._search(self.visual_collection, embedding, limit, expr, [
            "visual_feature_id", "m_id", "resource_id", "path", "sampling_policy",
            "sampled_seconds", "sampled_frame_count", "vconfig_id",
            "collect_start_time", "collect_end_time",
        ])

    def _search(self, collection_name: str, embedding: List[float], limit: int, filter_expr: str, output_fields: List[str]) -> List[Dict[str, Any]]:
        search_params = {"metric_type": "COSINE", "params": {}}
        result = self.client.search(
            collection_name=collection_name,
            anns_field="embedding",
            data=[embedding],
            limit=limit,
            filter=filter_expr or "",
            search_params=search_params,
            output_fields=output_fields,
            consistency_level="Strong",
        )
        items: List[Dict[str, Any]] = []
        for hit in result[0] if result else []:
            entity = dict(hit.get("entity", {}))
            entity["similarity"] = f"{hit.get('distance', 0):.4f}"
            items.append(entity)
        return items

    @staticmethod
    def _text_feature_id(m_id: str, feature_type: str) -> str:
        return f"{m_id}:{feature_type}"

    @staticmethod
    def _visual_feature_id(m_id: str) -> str:
        return f"{m_id}:visual"

    @staticmethod
    def _truncate_varchar(value: str, max_length: int) -> str:
        text = str(value or "")
        encoded = text.encode("utf-8")
        if len(encoded) <= max_length:
            return text
        return encoded[:max_length].decode("utf-8", errors="ignore")

    @staticmethod
    def _common_fields(video: Dict[str, Any], created_at: int) -> Dict[str, Any]:
        return {
            "m_id": video["m_id"],
            "resource_id": video.get("resource_id") or "",
            "path": video["path"],
            "vconfig_id": video.get("vconfig_id") or "",
            "collect_start_time": video.get("collect_start_time"),
            "collect_end_time": video.get("collect_end_time"),
            "created_at": created_at,
        }

    @staticmethod
    def _build_filter_expression(filter_params: Dict[str, Any]) -> str:
        conditions = []
        if filter_params.get("vconfig_id"):
            conditions.append(f'vconfig_id == "{filter_params["vconfig_id"]}"')
        if filter_params.get("collect_start_time") is not None:
            conditions.append(f'collect_start_time >= {filter_params["collect_start_time"]}')
        if filter_params.get("collect_end_time") is not None:
            conditions.append(f'collect_end_time <= {filter_params["collect_end_time"]}')
        if filter_params.get("resource_id"):
            conditions.append(f'resource_id == "{filter_params["resource_id"]}"')
        return " and ".join(conditions)


def ensure_feature_collections() -> None:
    FeatureDAO().ensure_collections()
