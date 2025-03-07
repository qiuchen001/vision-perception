import os
from dotenv import load_dotenv
from app.utils.embedding.embedding_types import EmbeddingType

load_dotenv()


class Config:
    # 服务器配置
    SERVER_HOST = os.getenv('SERVER_HOST', 'localhost')
    SERVER_PORT = int(os.getenv('SERVER_PORT', '30501'))

    # 视频处理配置
    VIDEO_FRAME_INTERVAL = int(os.getenv('VIDEO_FRAME_INTERVAL', '30'))  # 视频抽帧间隔
    VIDEO_FRAME_BATCH_SIZE = int(os.getenv('VIDEO_FRAME_BATCH_SIZE', '50'))  # 批处理大小

    # 模型配置
    MODEL_BASE_DIR = os.getenv('MODEL_BASE_DIR', 'models')
    CN_CLIP_MODEL_PATH = os.path.join(
        MODEL_BASE_DIR,
        'embedding',
        'cn-clip',
        'clip_cn_vit-l-14-336.pt'
    )

    # 默认使用CLIP模型
    DEFAULT_EMBEDDING_MODEL = EmbeddingType.CLIP
    
    # 从环境变量获取模型类型
    @classmethod
    def get_embedding_model_type(cls) -> EmbeddingType:
        model_type = os.getenv('EMBEDDING_MODEL', cls.DEFAULT_EMBEDDING_MODEL.value)
        try:
            return EmbeddingType(model_type.lower())
        except ValueError:
            print(f"警告:不支持的模型类型 {model_type},使用默认模型 {cls.DEFAULT_EMBEDDING_MODEL.value}")
            return cls.DEFAULT_EMBEDDING_MODEL

    # ... 其他配置 ...
