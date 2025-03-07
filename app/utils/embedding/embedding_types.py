from enum import Enum


class EmbeddingType(Enum):
    """Embedding模型类型枚举"""
    CLIP = 'clip'
    MULTIMODAL = 'multimodal'
