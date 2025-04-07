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

    def search_by_text(self, txt: str, page: int = 1, page_size: int = 6, search_mode: str = "frame", **filter_params) -> tuple[List[Dict[str, Any]], int]:
        """
        通过文本搜索视频。

        Args:
            txt: 搜索文本
            page: 页码
            page_size: 每页数量
            search_mode: 搜索模式
                - "frame": 先搜索视频帧,再获取视频信息(默认)
                - "summary": 直接搜索视频摘要
            **filter_params: 附加过滤条件
                - vconfig_id: 车辆类型标识
                - collect_start_time: 采集开始时间
                - collect_end_time: 采集结束时间
        Returns:
            Tuple[List[Dict[str, Any]], int]: 视频列表和总数
        """
        try:
            if search_mode == "frame":
                # 使用新的方法获取帧图片URL
                video_paths, timestamps, frame_urls, similarities = self.text_to_frame_with_url(txt)
                results = self._get_video_details_with_frame(video_paths, timestamps, frame_urls, similarities, page, page_size, **filter_params)
                total = len(video_paths)  # 总数为匹配的帧数
                return results, total
            else:
                # 直接搜索视频摘要
                summary_embedding = embed_fn(txt)  # 使用文本embedding函数
                return self.video_dao.search_video(
                    summary_embedding=summary_embedding,
                    page=page,
                    page_size=page_size,
                    **filter_params
                )
            
        except Exception as e:
            logger.error(f"文本搜索失败: {str(e)}")
            return [], 0

    def search_by_image(
            self,
            image_file: Optional[Union[FileStorage, Image.Image]] = None,
            image_url: Optional[str] = None,
            page: int = 1,
            page_size: int = 6,
            **filter_params
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        通过图片搜索视频。

        Args:
            image_file: 上传的图片文件或PIL Image对象
            image_url: 图片URL
            page: 页码
            page_size: 每页数量
            **filter_params: 附加过滤条件
                - vconfig_id: 车辆类型标识
                - collect_start_time: 采集开始时间
                - collect_end_time: 采集结束时间

        Returns:
            Tuple[List[Dict[str, Any]], int]: 视频列表和总数
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
            video_paths, timestamps, frame_urls, similarities = self.image_to_frame_with_url(image)
            
            # 获取视频详细信息
            results = self._get_video_details_with_frame(video_paths, timestamps, frame_urls, similarities, page, page_size, **filter_params)
            total = len(video_paths)  # 总数为匹配的帧数
            return results, total
            
        except Exception as e:
            logger.error(f"图片搜索失败: {str(e)}")
            return [], 0

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

    def search_by_tags(self, tags: Union[str, List[str]], page: int = 1, page_size: int = 6, **filter_params) -> tuple[List[Dict[str, Any]], int]:
        """
        通过标签搜索视频。

        Args:
            tags: 单个标签字符串或标签列表
            page: 页码
            page_size: 每页数量
            **filter_params: 附加过滤条件
                - vconfig_id: 车辆类型标识
                - collect_start_time: 采集开始时间
                - collect_end_time: 采集结束时间

        Returns:
            Tuple[List[Dict[str, Any]], int]: 视频列表和总数
        """
        try:
            # 将单个标签转换为列表
            if isinstance(tags, str):
                tags = [tags]

            # 调用DAO层进行搜索
            return self.video_dao.search_by_tags(
                tags=tags,
                page=page,
                page_size=page_size,
                **filter_params
            )

        except Exception as e:
            logger.error(f"标签搜索失败: {str(e)}")
            return [], 0

    def text_to_frame_with_url(self, txt: str) -> Tuple[List[str], List[int], List[str], List[float]]:
        """
        通过文本搜索视频帧，返回视频路径、时间戳、帧图片URL和相似度分数。
        结果按相似度从大到小排序。
        
        Args:
            txt: 搜索文本
            
        Returns:
            Tuple[List[str], List[int], List[str], List[float]]: 视频路径列表、时间戳列表、帧图片URL列表和相似度分数列表
        """
        try:
            # 获取文本embedding
            # embedding = embed_fn(txt)
            embedObj = EmbeddingFactory.create_embedding()
            embedding = embedObj.embedding_text(txt)
            
            # 搜索相似的视频帧
            results = video_frame_operator.search_frame(embedding)
            
            if not results:
                return [], [], [], []
            
            # 设置相似度阈值 (IP距离越大表示越相似)
            SIMILARITY_THRESHOLD = 0.01
            
            # 收集满足阈值的结果
            valid_results = []
            for result in results:
                similarity = result.get('distance', float('-inf'))
                if similarity >= SIMILARITY_THRESHOLD:
                    # 将结果和相似度一起保存
                    valid_results.append({
                        'video_id': result.get('video_id', ''),
                        'at_seconds': result.get('at_seconds', 0),
                        'frame_url': result.get('frame_url', ''),
                        'similarity': similarity
                    })
                    logger.info(f"匹配结果 - 视频: {result.get('video_id', '')}, 相似度: {similarity:.4f}")
                else:
                    logger.info(f"过滤掉低相似度结果 - 视频: {result.get('video_id', '')}, 相似度: {similarity:.4f}")
            
            # 按相似度降序排序
            sorted_results = sorted(valid_results, key=lambda x: x['similarity'], reverse=True)
            
            # 分离排序后的结果
            video_paths = [r['video_id'] for r in sorted_results]
            timestamps = [r['at_seconds'] for r in sorted_results]
            frame_urls = [r['frame_url'] for r in sorted_results]
            similarities = [r['similarity'] for r in sorted_results]
            
            return video_paths, timestamps, frame_urls, similarities
            
        except Exception as e:
            logger.error(f"文本到帧搜索失败: {str(e)}")
            return [], [], [], []

    def _get_video_details_with_frame(
            self,
            video_paths: List[str],
            timestamps: List[int],
            frame_urls: List[str],
            similarities: List[float],
            page: int = 1,
            page_size: int = 6,
            **filter_params
    ) -> List[Dict[str, Any]]:
        """
        获取视频详细信息，使用帧图片作为封面。
        
        Args:
            video_paths: 视频路径列表
            timestamps: 时间戳列表
            frame_urls: 帧图片URL列表
            similarities: 相似度分数列表
            page: 页码
            page_size: 每页数量
            **filter_params: 附加过滤条件
                - vconfig_id: 车辆类型标识
                - collect_start_time: 采集开始时间
                - collect_end_time: 采集结束时间
            
        Returns:
            List[Dict[str, Any]]: 视频详细信息列表，包含相似度分数
        """
        if not video_paths:
            return []
        
        # 应用筛选条件 - 只保留符合条件的视频
        filtered_paths = []
        filtered_timestamps = []
        filtered_frame_urls = []
        filtered_similarities = []
        
        if filter_params:
            # 需要分别获取每个视频的完整信息进行过滤
            for path, timestamp, frame_url, similarity in zip(video_paths, timestamps, frame_urls, similarities):
                video_info = self.video_dao.get_by_path(path)
                if not video_info or len(video_info) == 0:
                    continue
                
                video_data = video_info[0]
                
                # 检查是否符合过滤条件
                if self._match_filter_params(video_data, filter_params):
                    filtered_paths.append(path)
                    filtered_timestamps.append(timestamp)
                    filtered_frame_urls.append(frame_url)
                    filtered_similarities.append(similarity)
            
            # 使用过滤后的列表
            video_paths = filtered_paths
            timestamps = filtered_timestamps
            frame_urls = filtered_frame_urls
            similarities = filtered_similarities
            
            # 如果过滤后没有结果，则返回空列表
            if not video_paths:
                return []
        
        # 计算分页
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # 获取当前页的数据
        current_paths = video_paths[start_idx:end_idx]
        current_timestamps = timestamps[start_idx:end_idx]
        current_frame_urls = frame_urls[start_idx:end_idx]
        current_similarities = similarities[start_idx:end_idx]
        
        results = []
        for path, timestamp, frame_url, similarity in zip(current_paths, current_timestamps, current_frame_urls, current_similarities):
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
                    'timestamp': timestamp,
                    'similarity': f"{similarity:.4f}",  # 添加相似度分数，保留4位小数
                    'vconfig_id': video_data.get('vconfig_id', ''),
                    'collect_start_time': video_data.get('collect_start_time'),
                    'collect_end_time': video_data.get('collect_end_time')
                }
                results.append(result)
            
        return results
        
    def _match_filter_params(self, video_data: Dict[str, Any], filter_params: Dict[str, Any]) -> bool:
        """
        检查视频数据是否匹配过滤条件
        
        Args:
            video_data: 视频数据
            filter_params: 过滤条件
            
        Returns:
            bool: 是否匹配
        """
        # 检查vconfig_id
        if 'vconfig_id' in filter_params and filter_params['vconfig_id']:
            if not video_data.get('vconfig_id') or video_data.get('vconfig_id') != filter_params['vconfig_id']:
                return False
        
        # 检查collect_start_time
        if 'collect_start_time' in filter_params and filter_params['collect_start_time'] is not None:
            if not video_data.get('collect_start_time') or video_data.get('collect_start_time') < filter_params['collect_start_time']:
                return False
        
        # 检查collect_end_time
        if 'collect_end_time' in filter_params and filter_params['collect_end_time'] is not None:
            if not video_data.get('collect_end_time') or video_data.get('collect_end_time') > filter_params['collect_end_time']:
                return False
        
        return True

    def image_to_frame_with_url(self, image: Image.Image) -> Tuple[List[str], List[int], List[str], List[float]]:
        """
        通过图片搜索视频帧，返回视频路径、时间戳、帧图片URL和相似度分数。
        结果按相似度从大到小排序。
        
        Args:
            image: PIL Image对象
            
        Returns:
            Tuple[List[str], List[int], List[str], List[float]]: 视频路径列表、时间戳列表、帧图片URL列表和相似度分数列表
        """
        try:
            # 获取图片embedding
            embedding_model = EmbeddingFactory.create_embedding()
            embedding = embedding_model.embedding_image(image)
            
            # 搜索相似的视频帧
            results = video_frame_operator.search_frame(embedding)
            
            if not results:
                return [], [], [], []
            
            # 设置相似度阈值 (IP距离越大表示越相似)
            SIMILARITY_THRESHOLD = 0.01
            
            # 收集满足阈值的结果
            valid_results = []
            for result in results:
                similarity = result.get('distance', float('-inf'))
                if similarity >= SIMILARITY_THRESHOLD:
                    # 将结果和相似度一起保存
                    valid_results.append({
                        'video_id': result.get('video_id', ''),
                        'at_seconds': result.get('at_seconds', 0),
                        'frame_url': result.get('frame_url', ''),
                        'similarity': similarity
                    })
                    logger.info(f"匹配结果 - 视频: {result.get('video_id', '')}, 相似度: {similarity:.4f}")
                else:
                    logger.info(f"过滤掉低相似度结果 - 视频: {result.get('video_id', '')}, 相似度: {similarity:.4f}")
            
            # 按相似度降序排序
            sorted_results = sorted(valid_results, key=lambda x: x['similarity'], reverse=True)
            
            # 分离排序后的结果
            video_paths = [r['video_id'] for r in sorted_results]
            timestamps = [r['at_seconds'] for r in sorted_results]
            frame_urls = [r['frame_url'] for r in sorted_results]
            similarities = [r['similarity'] for r in sorted_results]
            
            return video_paths, timestamps, frame_urls, similarities
            
        except Exception as e:
            logger.error(f"图片到帧搜索失败: {str(e)}")
            return [], [], [], []

    def search_by_filter(self, page: int = 1, page_size: int = 6, **filter_params) -> tuple[List[Dict[str, Any]], int]:
        """
        仅使用过滤条件搜索视频。

        Args:
            page: 页码
            page_size: 每页数量
            **filter_params: 过滤条件
                - vconfig_id: 车辆类型标识
                - collect_start_time: 采集开始时间
                - collect_end_time: 采集结束时间

        Returns:
            Tuple[List[Dict[str, Any]], int]: 视频列表和总数
        """
        try:
            # 如果没有提供任何过滤条件，则返回空结果
            if not filter_params:
                return [], 0

            # 调用DAO层进行搜索
            return self.video_dao.search_by_filter(
                page=page,
                page_size=page_size,
                **filter_params
            )

        except Exception as e:
            logger.error(f"过滤搜索失败: {str(e)}")
            return [], 0


if __name__ == "__main__":
    search_service = SearchVideoService()
    # 使用单个标签搜索
    results, total = search_service.search_by_tags("晚上")
    print(results)
    print(f"总数: {total}")

    # # 使用多个标签搜索
    # results, total = search_service.search_by_tags(["教育", "科技"])
    #
    # # 带分页的搜索
    # results, total = search_service.search_by_tags(["教育", "科技"], page=2, page_size=10)
    #
    # results, total = search_service.search_by_text("主干道")
    # print(results)
    # print(f"总数: {total}")

