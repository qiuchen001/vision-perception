import os
import base64
from PIL import Image
import io
import openai
from typing import List, Tuple, Dict
import numpy as np
from app.utils.embedding.embedding_base import EmbeddingBase
from app.utils.logger import logger

class JinaClipEmbedding(EmbeddingBase):
    """Jina CLIP v2 多模态embedding实现"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """初始化配置"""
        self.client = openai.Client(
            api_key=api_key or "cannot be empty",
            base_url=base_url or "http://14.103.238.131:9997/v1"
        )
        
    def prepare_image(self, image: Image.Image) -> Image.Image:
        """预处理图片"""
        try:
            if not isinstance(image, Image.Image):
                raise ValueError("输入必须是PIL Image对象")
                
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            MAX_IMAGE_SIZE = 1024
            if image.size[0] > MAX_IMAGE_SIZE or image.size[1] > MAX_IMAGE_SIZE:
                ratio = MAX_IMAGE_SIZE / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"调整图片尺寸到: {new_size}")
                
            return image
            
        except Exception as e:
            logger.error(f"图片预处理失败: {str(e)}")
            raise ValueError(f"图片预处理失败: {str(e)}")
            
    def _image_to_base64(self, image: Image.Image) -> str:
        """将PIL Image转换为base64编码"""
        try:
            if not isinstance(image, Image.Image):
                raise ValueError("输入必须是PIL Image对象")
                
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"图片转base64失败: {str(e)}")
            raise ValueError(f"图片转base64失败: {str(e)}")

    def embedding_image(self, image: Image.Image) -> List[float]:
        """生成图片的embedding向量"""
        try:
            processed_image = self.prepare_image(image)
            image_b64 = self._image_to_base64(processed_image)
            
            completion = self.client.embeddings.create(
                model="jina-clip-v2",
                input=[{"image": image_b64}]
            )
            
            return completion.data[0].embedding
            
        except Exception as e:
            logger.error(f"生成图片embedding失败:{str(e)}")
            raise e

    def embedding_text(self, text: str) -> List[float]:
        """生成文本的embedding向量"""
        try:
            completion = self.client.embeddings.create(
                model="jina-clip-v2", 
                input=[{"text": text}]
            )
            
            return completion.data[0].embedding
            
        except Exception as e:
            logger.error(f"生成文本embedding失败:{str(e)}")
            raise e

    def embedding(self, image: Image.Image, text: str) -> Tuple[List[float], List[float]]:
        """生成图文联合embedding向量"""
        img_emb = self.embedding_image(image)
        txt_emb = self.embedding_text(text)
        return img_emb, txt_emb
        
    def match(self, image: Image.Image, texts: List[str]) -> Dict[str, float]:
        """计算图片与多个文本的匹配度"""
        try:
            img_emb = np.array(self.embedding_image(image))
            
            results = {}
            for text in texts:
                txt_emb = np.array(self.embedding_text(text))
                
                similarity = np.dot(img_emb, txt_emb) / (
                    np.linalg.norm(img_emb) * np.linalg.norm(txt_emb)
                )
                probability = float(max(0, min(1, (similarity + 1) / 2)))
                
                results[text] = probability
                
            return results
            
        except Exception as e:
            logger.error(f"计算匹配度失败: {str(e)}")
            raise e

# 创建全局实例            
jina_clip_embedding = JinaClipEmbedding()


if __name__ == "__main__":
    # 创建实例
    embedding = JinaClipEmbedding()

    # 生成向量
    image = Image.open("first_frame.png")
    # image = Image.open("pokemon.jpeg")

    # 计算匹配度
    texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘", "白天汽车行驶在道路上"]
    match_results = embedding.match(image, texts)

    print("\n图片与文本的匹配度:")
    for text, score in match_results.items():
        print(f"'{text}' 的匹配度: {score:.4f}")