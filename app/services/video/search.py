from typing import Optional, List, Dict, Any, Union, Tuple
from werkzeug.datastructures import FileStorage
from PIL import Image
import requests
from io import BytesIO

from app.dao.video_dao import VideoDAO
from app.services.video.video_frame_search import image_to_frame, text_to_frame
from app.utils.embedding.text_embedding import embed_fn
from app.utils.logger import logger
from app.utils.milvus_operator import video_frame_operator
from app.utils.embedding.embedding_factory import EmbeddingFactory


class SearchVideoService:
    def __init__(self):
        self.video_dao = VideoDAO()

    def search_by_text(self, txt: str, page: int = 1, page_size: int = 6, search_mode: str = "frame") -> List[Dict[str, Any]]:
        """
        通过文本搜索视频。

        Args:
            txt: 搜索文本
            page: 页码
            page_size: 每页数量
            search_mode: 搜索模式
                - "frame": 先搜索视频帧,再获取视频信息(默认)
                - "summary": 直接搜索视频摘要
        Returns:
            List[Dict[str, Any]]: 视频列表
        """
        try:
            if search_mode == "frame":
                # 使用新的方法获取帧图片URL
                video_paths, timestamps, frame_urls = self.text_to_frame_with_url(txt)
                return self._get_video_details_with_frame(video_paths, timestamps, frame_urls, page, page_size)
            else:
                # 直接搜索视频摘要
                summary_embedding = embed_fn(txt)  # 使用文本embedding函数
                return self.video_dao.search_video(
                    summary_embedding=summary_embedding,
                    page=page,
                    page_size=page_size
                )
            
        except Exception as e:
            logger.error(f"文本搜索失败: {str(e)}")
            return []

    def search_by_image(
            self,
            image_file: Optional[Union[FileStorage, Image.Image]] = None,
            image_url: Optional[str] = None,
            page: int = 1,
            page_size: int = 6
    ) -> List[Dict[str, Any]]:
        """
        通过图片搜索视频。

        Args:
            image_file: 上传的图片文件或PIL Image对象
            image_url: 图片URL
            page: 页码
            page_size: 每页数量

        Returns:
            List[Dict[str, Any]]: 视频列表
        """
        try:
            # 处理图片输入
            if image_file:
                if isinstance(image_file, Image.Image):
                    image = image_file  # 直接使用PIL Image对象
                else:
                    image = Image.open(image_file).convert('RGB')
            elif image_url:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                raise ValueError("No image provided")

            # 使用图片搜索视频帧,获取frame_url
            video_paths, timestamps, frame_urls = self.image_to_frame_with_url(image)
            
            # 获取视频详细信息
            return self._get_video_details_with_frame(video_paths, timestamps, frame_urls, page, page_size)
            
        except Exception as e:
            logger.error(f"图片搜索失败: {str(e)}")
            return []

    def _get_video_details(
            self,
            video_paths: List[str],
            timestamps: List[int],
            page: int,
            page_size: int
    ) -> List[Dict[str, Any]]:
        """获取视频详细信息"""
        try:
            # 计算分页
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            # 获取当前页的视频路径和时间戳
            page_paths = video_paths[start_idx:end_idx]
            page_timestamps = timestamps[start_idx:end_idx]
            
            # 获取视频详细信息
            video_list = []
            for video_path, timestamp in zip(page_paths, page_timestamps):
                video_info = self.video_dao.get_by_path(video_path)
                if video_info:
                    video_data = video_info[0].copy()  # 创建副本避免修改原始数据
                    # 确保所有数值类型都是 Python 原生类型
                    video_data['timestamp'] = int(timestamp)  # 转换时间戳为整数

                    # # 处理可能的 numpy 类型
                    # if 'embedding' in video_data:
                    #     if hasattr(video_data['embedding'], 'tolist'):
                    #         video_data['embedding'] = video_data['embedding'].tolist()

                    # 移除 embedding 字段
                    video_data.pop('embedding', None)
                    
                    # 处理其他可能的特殊类型字段
                    for key, value in video_data.items():
                        if hasattr(value, 'item'):  # 处理 numpy 标量类型
                            video_data[key] = value.item()
                        elif hasattr(value, 'tolist'):  # 处理 numpy 数组类型
                            video_data[key] = value.tolist()
                    
                    video_list.append(video_data)
                    
            return video_list
            
        except Exception as e:
            logger.error(f"获取视频详情失败: {str(e)}")
            return []

    def search_by_tags(self, tags: Union[str, List[str]], page: int = 1, page_size: int = 6) -> List[Dict[str, Any]]:
        """
        通过标签搜索视频。

        Args:
            tags: 单个标签字符串或标签列表
            page: 页码
            page_size: 每页数量

        Returns:
            List[Dict[str, Any]]: 视频列表
        """
        try:
            # 将单个标签转换为列表
            if isinstance(tags, str):
                tags = [tags]

            # 调用DAO层进行搜索
            return self.video_dao.search_by_tags(
                tags=tags,
                page=page,
                page_size=page_size
            )

        except Exception as e:
            logger.error(f"标签搜索失败: {str(e)}")
            return []

    def text_to_frame_with_url(self, txt: str) -> Tuple[List[str], List[int], List[str]]:
        """
        通过文本搜索视频帧，返回视频路径、时间戳和帧图片URL。
        
        Args:
            txt: 搜索文本
            
        Returns:
            Tuple[List[str], List[int], List[str]]: 视频路径列表、时间戳列表和帧图片URL列表
        """
        try:
            # 获取文本embedding
            embedding = embed_fn(txt)
            
            # 搜索相似的视频帧
            results = video_frame_operator.search_frame(embedding)
            
            if not results:
                return [], [], []
            
            # 提取视频路径、时间戳和帧图片URL
            video_paths = []
            timestamps = []
            frame_urls = []
            
            # 设置相似度阈值 (IP距离越大表示越相似)
            SIMILARITY_THRESHOLD = 0.01
            
            for result in results:
                # 获取相似度值(IP距离)
                similarity = result.get('distance', float('-inf'))
                
                # 保留相似度高于阈值的结果
                if similarity >= SIMILARITY_THRESHOLD:
                    video_paths.append(result.get('video_id', ''))
                    timestamps.append(result.get('at_seconds', 0))
                    frame_urls.append(result.get('frame_url', ''))
                    logger.info(f"匹配结果 - 图片: {result.get('frame_url', '')}, 相似度: {similarity:.4f}")
                else:
                    logger.info(f"过滤掉低相似度结果 - 视频: {result.get('video_id', '')}, 相似度: {similarity:.4f}")
            
            return video_paths, timestamps, frame_urls
            
        except Exception as e:
            logger.error(f"文本到帧搜索失败: {str(e)}")
            return [], [], []

    def _get_video_details_with_frame(self, video_paths: List[str], timestamps: List[int], frame_urls: List[str], page: int = 1, page_size: int = 6) -> List[Dict[str, Any]]:
        """
        获取视频详细信息，使用帧图片作为封面。
        
        Args:
            video_paths: 视频路径列表
            timestamps: 时间戳列表
            frame_urls: 帧图片URL列表
            page: 页码
            page_size: 每页数量
            
        Returns:
            List[Dict[str, Any]]: 视频详细信息列表
        """
        if not video_paths:
            return []
        
        # 计算分页
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # 获取当前页的数据
        current_paths = video_paths[start_idx:end_idx]
        current_timestamps = timestamps[start_idx:end_idx]
        current_frame_urls = frame_urls[start_idx:end_idx]
        
        results = []
        for path, timestamp, frame_url in zip(current_paths, current_timestamps, current_frame_urls):
            # 获取视频信息
            video_info = self.video_dao.get_by_path(path)
            if video_info and len(video_info) > 0:  # 确保有返回结果
                video_data = video_info[0]  # 获取第一个结果
                result = {
                    'title': video_data.get('title', '未知'),
                    'path': path,
                    'thumbnail_path': frame_url,  # 使用帧图片URL作为封面
                    'tags': video_data.get('tags', []),
                    'summary_txt': video_data.get('summary_txt', ''),
                    'timestamp': timestamp
                }
                results.append(result)
            
        return results

    def image_to_frame_with_url(self, image: Image.Image) -> Tuple[List[str], List[int], List[str]]:
        """
        通过图片搜索视频帧，返回视频路径、时间戳和帧图片URL。
        
        Args:
            image: PIL Image对象
            
        Returns:
            Tuple[List[str], List[int], List[str]]: 视频路径列表、时间戳列表和帧图片URL列表
        """
        try:
            # 获取图片embedding
            embedding_model = EmbeddingFactory.create_embedding()
            embedding = embedding_model.embedding_image(image)
            
            # 搜索相似的视频帧
            results = video_frame_operator.search_frame(embedding)
            
            if not results:
                return [], [], []
            
            # 提取视频路径、时间戳和帧图片URL
            video_paths = []
            timestamps = []
            frame_urls = []
            
            # 设置相似度阈值 (IP距离越大表示越相似)
            SIMILARITY_THRESHOLD = 0.01
            
            for result in results:
                # 获取相似度值(IP距离)
                similarity = result.get('distance', float('-inf'))
                
                # 保留相似度高于阈值的结果
                if similarity >= SIMILARITY_THRESHOLD:
                    video_paths.append(result.get('video_id', ''))
                    timestamps.append(result.get('at_seconds', 0))
                    frame_urls.append(result.get('frame_url', ''))
                    logger.info(f"匹配结果 - 图片: {result.get('frame_url', '')}, 相似度: {similarity:.4f}")
                else:
                    logger.info(f"过滤掉低相似度结果 - 视频: {result.get('video_id', '')}, 相似度: {similarity:.4f}")
            
            return video_paths, timestamps, frame_urls
            
        except Exception as e:
            logger.error(f"图片到帧搜索失败: {str(e)}")
            return [], [], []


if __name__ == "__main__":
    search_service = SearchVideoService()
    # 使用单个标签搜索
    results = search_service.search_by_tags("晚上")
    print(results)

    # # 使用多个标签搜索
    # results = search_service.search_by_tags(["教育", "科技"])
    #
    # # 带分页的搜索
    # results = search_service.search_by_tags(["教育", "科技"], page=2, page_size=10)
    #
    # results = search_service.search_by_text("主干道")
    # print(results)

