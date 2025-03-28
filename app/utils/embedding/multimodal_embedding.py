import os
import base64
from PIL import Image
import io
import dashscope
from http import HTTPStatus
from app.utils.embedding.embedding_base import EmbeddingBase
from app.utils.logger import logger
from typing import List, Tuple, Dict
import numpy as np


class MultiModalEmbedding(EmbeddingBase):
    """通义千问多模态embedding实现"""

    def __init__(self):
        """初始化DashScope配置"""
        dashscope.api_key = os.getenv("API_KEY")

    def prepare_image(self, image: Image.Image) -> Image.Image:
        """
        准备图片用于embedding。
        
        处理步骤:
        1. 验证输入图片的有效性
        2. 确保图片为RGB模式
        3. 如果图片尺寸过大,进行等比例缩放
        
        Args:
            image: PIL Image对象
            
        Returns:
            Image.Image: 处理后的PIL Image对象
            
        Raises:
            ValueError: 当输入无效或处理失败时
        """
        try:
            # 1. 验证输入
            if not isinstance(image, Image.Image):
                raise ValueError("输入必须是PIL Image对象")
                
            # 2. 确保RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 3. 尺寸调整
            MAX_IMAGE_SIZE = 1024  # 图片最大尺寸限制
            if image.size[0] > MAX_IMAGE_SIZE or image.size[1] > MAX_IMAGE_SIZE:
                ratio = MAX_IMAGE_SIZE / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"调整图片尺寸到: {new_size}")
                
            logger.debug(f"图片处理完成: size={image.size}, mode={image.mode}")
            return image
            
        except Exception as e:
            logger.error(f"图片预处理失败: {str(e)}")
            raise ValueError(f"图片预处理失败: {str(e)}")

    def _image_to_base64(self, image: Image.Image) -> str:
        """将PIL Image转换为base64编码"""
        try:
            # 验证输入
            if not isinstance(image, Image.Image):
                raise ValueError("输入必须是PIL Image对象")
                
            # 确保图片是RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # 保存为JPEG格式(更稳定,文件更小)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"图片转base64失败: {str(e)}")
            raise ValueError(f"图片转base64失败: {str(e)}")

    def embedding_image(self, image: Image.Image) -> List[float]:
        """生成图片的embedding向量"""
        try:
            # 预处理图片
            processed_image = self.prepare_image(image)
            
            # 将图片转换为base64
            image_b64 = self._image_to_base64(processed_image)
            image_data = f"data:image/jpeg;base64,{image_b64}"

            # 调用DashScope API
            resp = dashscope.MultiModalEmbedding.call(
                model="multimodal-embedding-v1",
                input=[{'image': image_data}]
            )

            if resp.status_code == HTTPStatus.OK:
                return resp.output['embeddings'][0]['embedding']
            else:
                raise Exception(f"API调用失败: {resp.code}, {resp.message}")

        except Exception as e:
            logger.error(f"生成图片embedding失败:{str(e)}")
            raise e

    def embedding_text(self, text: str) -> List[float]:
        """生成文本的embedding向量"""
        try:
            # 调用DashScope API
            resp = dashscope.MultiModalEmbedding.call(
                model="multimodal-embedding-v1",
                input=[{'text': text}]
            )

            if resp.status_code == HTTPStatus.OK:
                return resp.output['embeddings'][0]['embedding']
            else:
                raise Exception(f"API调用失败: {resp.code}, {resp.message}")

        except Exception as e:
            logger.error(f"生成文本embedding失败:{str(e)}")
            raise e

    def embedding(self, image: Image.Image, text: str) -> Tuple[List[float], List[float]]:
        """生成图文联合embedding向量"""
        img_emb = self.embedding_image(image)
        txt_emb = self.embedding_text(text)
        return img_emb, txt_emb

    def match(self, image: Image.Image, texts: List) -> Dict:
        """计算图片与多个文本的匹配度 (返回概率值)

        Args:
            image: PIL Image对象
            texts: 文本列表

        Returns:
            Dict: 文本及其对应的匹配概率 (0-1之间)
        """
        try:
            # 获取图片embedding
            img_emb = np.array(self.embedding_image(image))

            # 计算每个文本的相似度
            results = {}
            for text in texts:
                # 获取文本embedding
                txt_emb = np.array(self.embedding_text(text))

                # 计算余弦相似度
                similarity = np.dot(img_emb, txt_emb) / (np.linalg.norm(img_emb) * np.linalg.norm(txt_emb))
                # 确保相似度在0-1之间，并作为概率值
                probability = float(max(0, min(1, (similarity + 1) / 2)))

                results.update({text: probability})

            return results

        except Exception as e:
            logger.error(f"计算匹配度失败: {str(e)}")
            raise e


# 创建全局实例
multimodal_embedding = MultiModalEmbedding()

if __name__ == "__main__":
    # 测试代码
    image_path = "first_frame.png"
    pil_image = Image.open(image_path)

    # # 测试图片embedding
    # image_embeddings = multimodal_embedding.embedding_image(pil_image)
    # print(f"图片embedding维度:{len(image_embeddings)}")
    #
    # # 测试文本embedding
    # text_embeddings = multimodal_embedding.embedding_text("这是一张测试图片")
    # print(f"文本embedding维度:{len(text_embeddings)}")
    #
    # # 测试图文联合embedding
    # img_emb, txt_emb = multimodal_embedding.embedding(pil_image, "这是一张测试图片")
    # print(f"图文embedding维度:{len(img_emb)}, {len(txt_emb)}")

    # 测试图片与多个文本的匹配度
    texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘", "白天汽车行驶在道路上"]
    match_results = multimodal_embedding.match(pil_image, texts)
    
    print("\n图片与文本的匹配度:")
    for text, score in match_results.items():
        print(f"'{text}' 的匹配度: {score:.4f}")

    # # 使用环境变量中配置的模型
    # embedding = EmbeddingFactory.create_embedding()
    # image = Image.open('test.jpg')
    # img_emb, txt_emb = embedding.embedding(image, '测试文本')
    #
    # # 也可以显式指定模型类型
    # clip_embedding = EmbeddingFactory.create_embedding(EmbeddingType.CLIP)
