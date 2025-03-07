from abc import ABC, abstractmethod
from typing import Tuple, List
from PIL import Image


class EmbeddingBase(ABC):
    """多模态向量化基类"""
    
    @abstractmethod
    def embedding_image(self, image: Image.Image) -> List[float]:
        """生成图片embedding向量"""
        pass
        
    @abstractmethod 
    def embedding_text(self, text: str) -> List[float]:
        """生成文本embedding向量"""
        pass
        
    @abstractmethod
    def embedding(self, image: Image.Image, text: str) -> Tuple[List[float], List[float]]:
        """生成图文联合embedding向量"""
        pass 