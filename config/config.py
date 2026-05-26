import os
from dotenv import load_dotenv
from app.utils.embedding.embedding_types import EmbeddingType

load_dotenv()


class Config:
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

    # 场景挖掘算法配置
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    _COPIED_SCENE_MINING_CONFIG = os.path.join(
        _PROJECT_ROOT,
        'app',
        'algorithm',
        'scene_mining',
        'config-qwen-gemini.yaml'
    )
    SCENE_MINING_CONFIG_PATH = os.getenv(
        'SCENE_MINING_CONFIG_PATH',
        _COPIED_SCENE_MINING_CONFIG
    )
    SCENE_MINING_OUTPUT_DIR = os.getenv(
        'SCENE_MINING_OUTPUT_DIR',
        os.path.join(_PROJECT_ROOT, 'outputs', 'scene_mining')
    )
    SCENE_MINING_VIDEO_CACHE_DIR = os.getenv(
        'SCENE_MINING_VIDEO_CACHE_DIR',
        os.path.join(_PROJECT_ROOT, 'data', 'scene_mining_videos')
    )
    SCENE_MINING_VIDEO_URL_PREFIX = os.getenv(
        'SCENE_MINING_VIDEO_URL_PREFIX',
        'file:///app/videos'
    )

    QWEN3_VL_EMBEDDING_BASE_URL = os.getenv('QWEN3_VL_EMBEDDING_BASE_URL', 'http://localhost:8575')
    QWEN3_VL_EMBEDDING_DIM = int(os.getenv('QWEN3_VL_EMBEDDING_DIM', '2048'))
    MILVUS_VIDEO_TEXT_FEATURE_COLLECTION_NAME = os.getenv(
        'MILVUS_VIDEO_TEXT_FEATURE_COLLECTION_NAME',
        'video_text_features'
    )
    MILVUS_VIDEO_VISUAL_FEATURE_COLLECTION_NAME = os.getenv(
        'MILVUS_VIDEO_VISUAL_FEATURE_COLLECTION_NAME',
        'video_visual_features'
    )
    FRAME_SAMPLE_FPS = float(os.getenv('FRAME_SAMPLE_FPS', '1'))
    FRAME_SAMPLE_MAX_FRAMES = int(os.getenv('FRAME_SAMPLE_MAX_FRAMES', '20'))
    FRAME_SAMPLE_EVENT_FPS = float(os.getenv('FRAME_SAMPLE_EVENT_FPS', '1'))
    FRAME_SAMPLE_EVENT_WEIGHT = float(os.getenv('FRAME_SAMPLE_EVENT_WEIGHT', '1.5'))
    RAWDATA_REQUEST_TIMEOUT = float(os.getenv('RAWDATA_REQUEST_TIMEOUT', '3'))
    
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
