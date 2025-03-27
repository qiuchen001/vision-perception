from typing import List, Dict, Any

from pymilvus import MilvusClient
from ..models.video import Video
from ..utils.logger import logger
import uuid
from flask import current_app
import os
import json


class VideoDAO:
    def __init__(self):
        MILVUS_HOST = os.getenv("MILVUS_HOST")
        MILVUS_PORT = os.getenv("MILVUS_PORT")
        self.milvus_client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
                                          db_name=os.getenv("MILVUS_DB_NAME"))
        # self.milvus_client = current_app.config['MILVUS_CLIENT']
        self.collection_name = "video_collection"

    # def init_video(self):
    #     Video.create_database()
    #     schema = Video.create_schema()
    #     Video.create_collection(self.collection_name, schema)
    #     Video.create_index(self.collection_name)

    def get_all_videos(self):
        logger.info(f"Querying all users from collection: {self.collection_name}")
        return self.milvus_client.query(self.collection_name, filter="", limit=6)

    def search_all_videos(self, page=1, page_size=10):
        offset = (page - 1) * page_size
        limit = page_size
        search_params = {
            "metric_type": "IP",  # 指定相似度度量类型，IP表示内积（Inner Product）
            "offset": offset,
            "limit": limit
        }
        logger.info(f"Searching all videos from collection: {self.collection_name} with params: {search_params}")
        return self.milvus_client.search(self.collection_name, filter="", **search_params)

    def insert_video(self, user):
        user_data = {
            "m_id": user.m_id,
            "embedding": user.embedding,
            "path": user.path,
            "thumbnail_path": user.thumbnail_path,
            "summary_txt": user.summary_txt,
            "tags": str(user.tags)  # 将数组转换为字符串
        }
        self.milvus_client.insert(self.collection_name, [user_data])

    def check_url_exists(self, url):
        # 检查URL是否存在
        # 返回True或False
        query_result = self.milvus_client.query(self.collection_name, filter=f"path == '{url}'", limit=1)
        return len(query_result) > 0

    def get_by_path(self, url):
        query_result = self.milvus_client.query(self.collection_name, filter=f"path == '{url}'", limit=1)
        return query_result

    def init_video(self, url, embedding, summary_embedding, thumbnail_oss_url, title, resource_id):
        # 插入URL到数据库
        video_data = {
            "m_id": str(uuid.uuid4()),
            "embedding": embedding,
            "summary_embedding": summary_embedding,
            "path": url,
            "thumbnail_path": thumbnail_oss_url,
            "title": title,
            "summary_txt": None,
            "tags": None,  # 保留tags字段
            "mining_results": None,  # 添加mining_results字段
            "resource_id": resource_id
        }
        res = self.milvus_client.insert(self.collection_name, [video_data])
        return res

    def upsert_video(self, video):
        # 从mining_results中提取behaviourName作为tags
        mining_results = video.get('mining_results', [])
        tags = list(set([
            result['behaviour']['behaviourName']
            for result in mining_results
            if result.get('behaviour', {}).get('behaviourName')
        ])) if mining_results else []

        user_data = {
            "m_id": video['m_id'],
            "embedding": video['embedding'],
            "summary_embedding": video['summary_embedding'],
            "path": video['path'],
            "thumbnail_path": video['thumbnail_path'],
            "title": video['title'],
            "summary_txt": video['summary_txt'],
            "tags": tags,  # 更新tags字段
            "mining_results": json.dumps(mining_results, ensure_ascii=False),  # 不转义中文字符
            "resource_id": video['resource_id']
        }
        return self.milvus_client.upsert(self.collection_name, [user_data])

    def search_video(self, summary_embedding=None, page=1, page_size=6):
        offset = (page - 1) * page_size
        limit = page_size

        search_params = {
            "metric_type": "IP",
            "offset": offset,
            "ignore_growing": False,
            "params": {"nprobe": 16}
        }

        if summary_embedding is not None:
            # 设置相似度阈值
            SIMILARITY_THRESHOLD = 0.01

            result = self.milvus_client.search(
                collection_name=self.collection_name,
                anns_field="summary_embedding",
                data=[summary_embedding],
                limit=limit,
                search_params=search_params,
                output_fields=['m_id', 'path', 'thumbnail_path', 'summary_txt', 'tags', 'title'],
                consistency_level="Strong"
            )

            new_result_list = []
            if result[0] is not None:
                for hit in result[0]:
                    similarity = hit.get("distance", 0)  # 获取相似度分数
                    if similarity >= SIMILARITY_THRESHOLD:  # 过滤低相似度结果
                        entity = hit.get("entity", {})
                        if entity:
                            entity['timestamp'] = 0
                            entity['similarity'] = f"{similarity:.4f}"  # 添加相似度分数，保留4位小数
                            new_result_list.append(entity)
                
                # 按相似度降序排序
                new_result_list.sort(key=lambda x: float(x['similarity']), reverse=True)
            return new_result_list

        else:
            result = self.milvus_client.query(
                self.collection_name,
                filter="",
                offset=offset,
                limit=limit,
                output_fields=['m_id', 'path', 'thumbnail_path', 'summary_txt', 'tags', 'title']
            )
            for item in result:
                item['timestamp'] = 0
            return result

    def search_by_tags(self, tags: List[str], page: int = 1, page_size: int = 6) -> List[Dict[str, Any]]:
        """
        根据标签列表搜索视频，使用ARRAY_CONTAINS操作符查询tags字段

        Args:
            tags: 标签列表
            page: 页码
            page_size: 每页数量

        Returns:
            List[Dict[str, Any]]: 匹配的视频列表
        """
        offset = (page - 1) * page_size

        # 构建标签过滤条件
        tag_filters = []
        for tag in tags:
            # 使用ARRAY_CONTAINS操作符
            tag_filters.append(f'ARRAY_CONTAINS(tags, "{tag}")')

        # 组合多个标签的过滤条件(使用OR连接)
        filter_expr = " or ".join(tag_filters)
        
        logger.info(f"Generated filter expression: {filter_expr}")  # 添加日志记录

        # 执行查询
        result = self.milvus_client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            offset=offset,
            limit=page_size,
            output_fields=['m_id', 'path', 'thumbnail_path', 'summary_txt', 'tags', 'mining_results', 'title']
        )

        # 处理结果
        for item in result:
            item['timestamp'] = 0
            if item.get('mining_results') is None:
                item['mining_results'] = []
            else:
                # 确保mining_results是JSON对象而不是字符串
                if isinstance(item['mining_results'], str):
                    item['mining_results'] = json.loads(item['mining_results'])

        return result
