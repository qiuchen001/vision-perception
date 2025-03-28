from typing import Iterator, List, Dict, Any
import os
from dotenv import load_dotenv
from pymilvus import MilvusClient
import logging

# 加载环境变量
load_dotenv()


class VideoCollectionPaginator:
    """视频集合分页迭代器 - Milvus版本"""

    def __init__(self, client: MilvusClient, collection_name: str, page_size: int = 100):
        """
        初始化分页迭代器
        
        Args:
            client: MilvusClient实例
            collection_name: 集合名称
            page_size: 每页数据量,默认100
        """
        self.client = client
        self.collection_name = collection_name
        self.page_size = page_size
        self.current_offset = 0
        self.logger = logging.getLogger(__name__)

        # 获取总数
        self.total = self.client.get_collection_stats(collection_name)["row_count"]

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        """实现迭代器接口"""
        return self

    def __next__(self) -> List[Dict[str, Any]]:
        """获取下一页数据"""
        if self.current_offset >= self.total:
            raise StopIteration

        try:
            # 定义要查询的输出字段
            output_fields = [
                "m_id", "path", "thumbnail_path",
                "title", "summary_txt", "tags"
            ]

            # 执行查询
            results = self.client.query(
                collection_name=self.collection_name,
                filter="",  # 无过滤条件
                output_fields=output_fields,
                limit=self.page_size,
                offset=self.current_offset
            )

            if not results:
                raise StopIteration

            self.current_offset += self.page_size
            return results

        except Exception as e:
            self.logger.error(f"获取数据失败(offset={self.current_offset}): {str(e)}")
            raise


def get_video_paginator(
        host: str = None,
        db_name: str = None,
        collection_name: str = "video_collection_v2",
        page_size: int = 100
) -> VideoCollectionPaginator:
    """
    创建视频集合分页迭代器
    
    Args:
        host: Milvus服务器地址,默认从环境变量获取
        db_name: 数据库名,默认从环境变量获取
        collection_name: 集合名称
        page_size: 每页数据量
        
    Returns:
        VideoCollectionPaginator对象
    """
    try:
        # 如果未指定host和db_name,从环境变量获取
        if not host:
            host = f"http://{os.getenv('SERVER_HOST')}:19530"
        if not db_name:
            db_name = os.getenv("DB_NAME")

        client = MilvusClient(
            uri=host,
            db_name=db_name
        )
        return VideoCollectionPaginator(client, collection_name, page_size)

    except Exception as e:
        logging.error(f"创建分页迭代器失败: {str(e)}")
        raise


# 使用示例:
"""
# 创建分页迭代器
paginator = get_video_paginator()

# 遍历所有视频数据
for page in paginator:
    for video in page:
        # 处理每个视频数据
        process_video(video)
"""

# 创建分页迭代器
paginator = get_video_paginator()

# # 遍历所有视频数据
# for page in paginator:
#     for video in page:
#         print("video:", video)
        # 处理每个视频数据
        # process_video(video)
