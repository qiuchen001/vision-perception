from typing import List, Tuple, Union
from PIL import Image
import os
import requests
from io import BytesIO
import re
import numpy as np

from app.utils.embedding.embedding_factory import EmbeddingFactory
from app.utils.milvus_operator import video_frame_operator


def _is_valid_url(url: str) -> bool:
    """
    检查是否为有效的URL。

    Args:
        url: 要检查的URL字符串

    Returns:
        bool: 是否为有效URL
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// 或 https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # 域名
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP地址
        r'(?::\d+)?'  # 可选端口
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(url))


def _load_image_from_url(url: str) -> Image.Image:
    """
    从URL加载图片。

    Args:
        url: 图片URL

    Returns:
        Image.Image: PIL图片对象

    Raises:
        Exception: 当图片下载或处理失败时
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        raise Exception(f"从URL加载图片失败: {str(e)}")


def _process_search_results(results) -> Tuple[List[str], List[int], List[str]]:
    """
    处理搜索结果。

    Args:
        results: Milvus 搜索返回的结果

    Returns:
        Tuple[List[str], List[int], List[str]]: 视频路径、时间戳和帧图片URL列表
    """
    video_paths = []
    at_seconds = []
    frame_urls = []
    for result in results:
        video_id = result.get('video_id')
        timestamp = result.get('at_seconds')
        frame_url = result.get('frame_url', '')
        if video_id is not None and timestamp is not None:
            video_paths.append(video_id)
            at_seconds.append(timestamp)
            frame_urls.append(frame_url)
    return video_paths, at_seconds, frame_urls


def video_frame_search(query: Union[str, Image.Image]) -> Tuple[List[str], List[int], List[str]]:
    """
    通过文本或图片搜索视频帧。

    Args:
        query: 可以是:
            - 文本字符串
            - PIL.Image 对象
            - 本地图片路径
            - 在线图片URL

    Returns:
        Tuple[List[str], List[int], List[str]]: 返回视频路径列表、时间戳列表和帧图片URL列表
    """
    if query is None:
        print("没有任何输入！")
        return [], [], []

    try:
        # 获取embedding实例
        embedding = EmbeddingFactory.create_embedding()
        
        # 根据输入类型生成向量
        if isinstance(query, str):
            if _is_valid_url(query):  # 检查是否为URL
                try:
                    image = _load_image_from_url(query)
                    input_embedding = embedding.embedding_image(image)
                except Exception as e:
                    print(f"处理在线图片失败: {str(e)}")
                    return [], [], []
            elif os.path.isfile(query):  # 本地图片路径
                try:
                    image = Image.open(query).convert('RGB')
                    input_embedding = embedding.embedding_image(image)
                except Exception as e:
                    print(f"读取本地图片失败: {str(e)}")
                    return [], [], []
            else:  # 文本查询
                input_embedding = embedding.embedding_text(query)
        elif isinstance(query, Image.Image):
            input_embedding = embedding.embedding_image(query)
        else:
            print(f"不支持的查询类型: {type(query)}")
            return [], [], []

        if input_embedding is None or len(input_embedding) == 0:
            print("无法生成查询向量")
            return [], [], []

        # 转换向量格式为numpy数组
        input_embedding = np.array(input_embedding, dtype='float32')

        print("input_embedding shape:", input_embedding.shape)

        # 执行搜索
        results = video_frame_operator.search_data(
            embedding=input_embedding,
            output_fields=['video_id', 'at_seconds', 'frame_url']
        )

        print("找到结果数量:", len(results))

        return _process_search_results(results)

    except Exception as e:
        print(f"搜索失败: {str(e)}")
        return [], [], []


def image_to_frame(image_source: Union[str, Image.Image]) -> Tuple[List[str], List[int], List[str]]:
    """
    通过图片搜索视频帧。

    Args:
        image_source: 可以是:
            - 本地图片路径
            - 在线图片URL
            - PIL.Image 对象

    Returns:
        Tuple[List[str], List[int], List[str]]: 返回视频路径列表、时间戳列表和帧图片URL列表
    """
    return video_frame_search(image_source)


def text_to_frame(text: str) -> Tuple[List[str], List[int], List[str]]:
    """
    通过文本搜索视频帧。

    Args:
        text: 搜索文本

    Returns:
        Tuple[List[str], List[int], List[str]]: 返回视频路径列表、时间戳列表和帧图片URL列表
    """
    return video_frame_search(text)


if __name__ == "__main__":
    # 文本搜索示例
    print("\n=== 文本搜索测试 ===")
    video_paths, at_seconds, frame_urls = text_to_frame("高速路")
    print("文本搜索结果:")
    print("视频路径:", video_paths)
    print("时间戳:", at_seconds)
    print("帧图片URL:", frame_urls)

    # 本地图片搜索示例
    print("\n=== 本地图片搜索测试 ===")
    local_image_path = r"E:\workspace\ai-ground\dataset\traffic\CAM_BACK\1537295813887.jpg"
    if os.path.exists(local_image_path):
        video_paths, at_seconds, frame_urls = image_to_frame(local_image_path)
        print("本地图片搜索结果:")
        print("视频路径:", video_paths)
        print("时间戳:", at_seconds)
        print("帧图片URL:", frame_urls)

    # 在线图片搜索示例
    print("\n=== 在线图片搜索测试 ===")
    online_image_url = "http://10.66.12.37:30946/perception-mining/b7ec1001240181ceb5ec3e448c7f9b78.mp4_t_0.jpg"
    try:
        video_paths, at_seconds, frame_urls = image_to_frame(online_image_url)
        print("在线图片搜索结果:")
        print("视频路径:", video_paths)
        print("时间戳:", at_seconds)
        print("帧图片URL:", frame_urls)
    except Exception as e:
        print(f"在线图片搜索失败: {e}") 