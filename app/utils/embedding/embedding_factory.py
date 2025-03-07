from typing import Optional, Dict
from app.utils.embedding.embedding_base import EmbeddingBase
from app.utils.embedding.multimodal_embedding import MultiModalEmbedding
from app.utils.embedding.embedding_types import EmbeddingType
from config.config import Config


class EmbeddingFactory:
    """Embedding模型工厂类"""

    # 使用字典存储实例
    _instances: Dict[EmbeddingType, Optional[EmbeddingBase]] = {
        EmbeddingType.CLIP: None,
        EmbeddingType.MULTIMODAL: None
    }

    @classmethod
    def create_embedding(cls, model_type: Optional[EmbeddingType] = None) -> EmbeddingBase:
        """
        创建Embedding实例
        Args:
            model_type: 模型类型,如果为None则从配置获取
        Returns:
            EmbeddingBase实例
        """
        # 如果未指定类型,从配置获取
        if model_type is None:
            model_type = Config.get_embedding_model_type()

        # 如果实例不存在,创建新实例
        if cls._instances[model_type] is None:
            if model_type == EmbeddingType.MULTIMODAL:
                cls._instances[model_type] = MultiModalEmbedding()

        return cls._instances[model_type]
