from typing import List, Dict, Any

from pymilvus import DataType, MilvusClient
from ..models.video import Video
from ..utils.logger import logger
import uuid
from flask import current_app
import os
import json
import ast
import requests
from urllib.parse import urlparse
from config.config import Config


class VideoDAO:
    SUMMARY_MAX_LENGTH = 8192
    MINING_RESULTS_MAX_LENGTH = 65535
    DETAIL_OUTPUT_FIELDS = [
        'm_id', 'resource_id', 'path', 'thumbnail_path', 'title', 'summary_txt',
        'tags', 'mining_results', 'embedding', 'summary_embedding',
        'vconfig_id', 'collect_start_time', 'collect_end_time'
    ]

    def __init__(self):
        MILVUS_HOST = os.getenv("MILVUS_HOST")
        MILVUS_PORT = os.getenv("MILVUS_PORT")
        self.milvus_client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
                                          db_name=os.getenv("MILVUS_DB_NAME"))
        # self.milvus_client = current_app.config['MILVUS_CLIENT']
        self.collection_name = os.getenv("MILVUS_VIDEO_COLLECTION_NAME")
        self.ensure_collection()

    @staticmethod
    def _normalize_tags(tags: Any) -> List[str]:
        if isinstance(tags, list):
            return [str(tag) for tag in tags if tag is not None]
        if not tags:
            return []
        if isinstance(tags, str):
            try:
                parsed = json.loads(tags)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(tags)
                except (SyntaxError, ValueError):
                    logger.warning("Failed to parse tags safely: %s", tags[:200])
                    return []
            if isinstance(parsed, list):
                return [str(tag) for tag in parsed if tag is not None]
        return []

    # def init_video(self):
    #     Video.create_database()
    #     schema = Video.create_schema()
    #     Video.create_collection(self.collection_name, schema)
    #     Video.create_index(self.collection_name)

    def ensure_collection(self):
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.load_collection(self.collection_name)
            return

        schema = self.milvus_client.create_schema(
            auto_id=False,
            enable_dynamic_fields=True,
            description="视频主集合：保存视频基础信息、摘要和标签。",
        )
        dim = Config.QWEN3_VL_EMBEDDING_DIM
        schema.add_field("m_id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("resource_id", DataType.VARCHAR, max_length=256)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("summary_embedding", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("path", DataType.VARCHAR, max_length=1024)
        schema.add_field("thumbnail_path", DataType.VARCHAR, max_length=1024, nullable=True)
        schema.add_field("title", DataType.VARCHAR, max_length=512, nullable=True)
        schema.add_field("summary_txt", DataType.VARCHAR, max_length=8192, nullable=True)
        schema.add_field(
            "tags",
            DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=10,
            max_length=256,
            nullable=True,
        )
        schema.add_field("mining_results", DataType.VARCHAR, max_length=65535, nullable=True)
        schema.add_field("vconfig_id", DataType.VARCHAR, max_length=256, nullable=True)
        schema.add_field("collect_start_time", DataType.INT64, nullable=True)
        schema.add_field("collect_end_time", DataType.INT64, nullable=True)
        self.milvus_client.create_collection(self.collection_name, schema=schema)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="FLAT",
            index_name="embedding_index",
            params={},
        )
        index_params.add_index(
            field_name="summary_embedding",
            metric_type="COSINE",
            index_type="FLAT",
            index_name="summary_embedding_index",
            params={},
        )
        self.milvus_client.create_index(self.collection_name, index_params=index_params)
        self.milvus_client.load_collection(self.collection_name)

    def get_all_videos(self):
        logger.info(f"Querying all users from collection: {self.collection_name}")
        return self.milvus_client.query(self.collection_name, filter="", limit=6)

    def search_all_videos(self, page=1, page_size=10):
        offset = (page - 1) * page_size
        limit = page_size
        search_params = {
            "metric_type": "COSINE",  # 指定相似度度量类型，IP表示内积（Inner Product）
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
            "tags": self._normalize_tags(user.tags)
        }
        self.milvus_client.insert(self.collection_name, [user_data])

    def check_url_exists(self, url):
        # 检查URL是否存在
        # 返回True或False
        query_result = self.milvus_client.query(self.collection_name, filter=f"path == '{url}'", limit=1)
        return len(query_result) > 0

    def get_by_path(self, url):
        for candidate in self._path_candidates(url):
            query_result = self.milvus_client.query(
                self.collection_name,
                filter=f"path == '{candidate}'",
                limit=1,
                output_fields=self.DETAIL_OUTPUT_FIELDS,
            )
            if query_result:
                return query_result
        return []

    def get_by_resource_id(self, resource_id):
        query_result = self.milvus_client.query(
            self.collection_name,
            filter=f"resource_id == '{resource_id}'",
            limit=1,
            output_fields=self.DETAIL_OUTPUT_FIELDS,
        )
        return query_result

    def get_by_m_id(self, m_id):
        return self.milvus_client.query(
            self.collection_name,
            filter=f'm_id == "{m_id}"',
            limit=1,
            output_fields=self.DETAIL_OUTPUT_FIELDS,
        )

    @staticmethod
    def _path_candidates(url):
        candidates = []
        raw_url = str(url or "").strip()
        if raw_url:
            candidates.append(raw_url)

        bucket_name = os.getenv("OSS_BUCKET_NAME", "").strip()
        oss_endpoint = os.getenv("OSS_ENDPOINT", "").strip()
        parsed = urlparse(raw_url)
        if bucket_name:
            media_prefix = f"/media/{bucket_name}/"
            bucket_prefix = f"/{bucket_name}/"
            if raw_url.startswith(media_prefix) and oss_endpoint:
                object_name = raw_url[len(media_prefix):]
                candidates.append(f"http://{oss_endpoint}/{bucket_name}/{object_name}")
            elif parsed.scheme in {"http", "https"} and parsed.path.startswith(bucket_prefix):
                candidates.append(f"/media{parsed.path}")

        deduped = []
        for candidate in candidates:
            if candidate and candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def init_video(self, url, embedding, summary_embedding, thumbnail_oss_url, title, resource_id, fetch_metadata=True, m_id=None):
        # 从元数据接口获取数据
        metadata = {}
        if fetch_metadata:
            try:
                rawdata_service_base_url = os.getenv("RAWDATA_SERVICE_BASE_URL", "http://10.66.12.37:31557")
                response = requests.get(
                    f"{rawdata_service_base_url}/dataplatform/rawdata/{resource_id}",
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=Config.RAWDATA_REQUEST_TIMEOUT,
                )

                if response.status_code != 200:
                    logger.error(f"获取元数据失败: {response.text}")
                else:
                    data = response.json()
                    metadata = data.get("rawdata", {})
            except Exception as e:
                logger.error(f"调用元数据接口失败: {str(e)}")
        else:
            logger.debug("跳过 rawdata 元数据查询: 快速上传路径不依赖外部资源元数据")

        # 插入URL到数据库
        video_data = {
            "m_id": m_id or str(uuid.uuid4()),
            "embedding": embedding,
            "summary_embedding": summary_embedding,
            "path": url,
            "thumbnail_path": thumbnail_oss_url,
            "title": title,
            "summary_txt": None,
            "tags": None,  # 保留tags字段
            "mining_results": None,  # 添加mining_results字段
            "resource_id": resource_id,
            # 添加新字段，从元数据中获取
            "vconfig_id": metadata.get("vconfigId"),
            "collect_start_time": metadata.get("collectStartTime"),
            "collect_end_time": metadata.get("collectEndTime")
        }
        res = self.milvus_client.upsert(self.collection_name, [video_data])
        self.milvus_client.flush(self.collection_name)
        return res

    def upsert_video(self, video):
        # 新算法直接写入 tags；旧 behaviour 结构仍保留兼容解析。
        mining_results = video.get('mining_results', [])
        video_tags = video.get('tags')
        if isinstance(video_tags, list):
            tags = []
            for tag in video_tags:
                tag_text = str(tag).strip()
                if tag_text and tag_text not in tags:
                    tags.append(tag_text)
            tags = tags[:10]
        elif isinstance(mining_results, list):
            tags = list(set([
                result['behaviour']['behaviourName']
                for result in mining_results
                if isinstance(result, dict) and result.get('behaviour', {}).get('behaviourName')
            ]))
        else:
            tags = []

        if isinstance(mining_results, str):
            mining_results_json = mining_results
        else:
            mining_results_json = json.dumps(mining_results, ensure_ascii=False)

        user_data = {
            "m_id": video['m_id'],
            "embedding": video['embedding'],
            "summary_embedding": video['summary_embedding'],
            "path": video['path'],
            "thumbnail_path": video['thumbnail_path'],
            "title": video['title'],
            "summary_txt": self._truncate_varchar(video.get('summary_txt'), self.SUMMARY_MAX_LENGTH),
            "tags": tags,  # 更新tags字段
            "mining_results": self._truncate_varchar(mining_results_json, self.MINING_RESULTS_MAX_LENGTH),  # 不转义中文字符
            "resource_id": video['resource_id'],
            "vconfig_id": video.get('vconfig_id'),
            "collect_start_time": video.get('collect_start_time'),
            "collect_end_time": video.get('collect_end_time')
        }
        res = self.milvus_client.upsert(self.collection_name, [user_data])
        self.milvus_client.flush(self.collection_name)
        return res

    @staticmethod
    def _truncate_varchar(value, max_length: int):
        text = "" if value is None else str(value)
        encoded = text.encode("utf-8")
        if len(encoded) <= max_length:
            return text
        return encoded[:max_length].decode("utf-8", errors="ignore")

    def get_total_count(self, filter_expr: str = "") -> int:
        """
        获取符合条件的记录总数

        Args:
            filter_expr: 过滤条件表达式

        Returns:
            int: 记录总数
        """
        try:
            return self.milvus_client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=['m_id']
            ).__len__()
        except Exception as e:
            logger.error(f"获取总数失败: {str(e)}")
            return 0

    def search_video(self, summary_embedding=None, page=1, page_size=6, top_k=10, **filter_params):
        offset = (page - 1) * page_size
        limit = page_size

        # 构建过滤条件表达式
        filter_expr = self._build_filter_expression(filter_params)

        if summary_embedding is not None:
            # 设置相似度阈值
            SIMILARITY_THRESHOLD = 0.01

            # 获取所有结果时不需要offset
            search_params = {
                "metric_type": "COSINE",
                "ignore_growing": False,
                "params": {"nprobe": 16}
            }

            # 应用过滤条件（如果有）
            if filter_expr:
                search_params["filter"] = filter_expr

            result = self.milvus_client.search(
                collection_name=self.collection_name,
                anns_field="summary_embedding",
                data=[summary_embedding],
                limit=top_k or 1000,
                search_params=search_params,
                output_fields=['m_id', 'path', 'thumbnail_path', 'summary_txt', 'tags', 'title', 'vconfig_id', 'collect_start_time', 'collect_end_time'],
                consistency_level="Strong"
            )

            new_result_list = []
            total = 0
            if result[0] is not None:
                for hit in result[0]:
                    similarity = hit.get("distance", 0)  # 获取相似度分数
                    if similarity >= SIMILARITY_THRESHOLD:  # 过滤低相似度结果
                        total += 1
                        entity = hit.get("entity", {})
                        if entity:
                            entity['timestamp'] = 0
                            entity['similarity'] = f"{similarity:.4f}"  # 添加相似度分数，保留4位小数
                            new_result_list.append(entity)
                            if top_k and total >= top_k:
                                break
                
                # 按相似度降序排序
                new_result_list.sort(key=lambda x: float(x['similarity']), reverse=True)
                
                # 分页
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                new_result_list = new_result_list[start_idx:end_idx]
                
            return new_result_list, total

        else:
            # 获取总数
            total = self.get_total_count(filter_expr) if filter_expr else self.get_total_count()
            if top_k:
                total = min(total, top_k)
                if offset >= total:
                    return [], total
                limit = min(limit, total - offset)
            
            # 获取分页数据
            result = self.milvus_client.query(
                self.collection_name,
                filter=filter_expr if filter_expr else "",
                offset=offset,
                limit=limit,
                output_fields=['m_id', 'path', 'thumbnail_path', 'summary_txt', 'tags', 'title', 'vconfig_id', 'collect_start_time', 'collect_end_time']
            )
            for item in result:
                item['timestamp'] = 0
            return result, total

    def search_by_tags(self, tags: List[str], page: int = 1, page_size: int = 6, top_k: int = 10, **filter_params) -> tuple[List[Dict[str, Any]], int]:
        """
        根据标签列表搜索视频，使用ARRAY_CONTAINS操作符查询tags字段

        Args:
            tags: 标签列表
            page: 页码
            page_size: 每页数量
            **filter_params: 附加过滤条件
                - vconfig_id: 车辆类型标识
                - collect_start_time: 采集开始时间
                - collect_end_time: 采集结束时间

        Returns:
            Tuple[List[Dict[str, Any]], int]: 匹配的视频列表和总数
        """
        offset = (page - 1) * page_size

        # 构建标签过滤条件
        tag_filters = []
        for tag in tags:
            # 使用ARRAY_CONTAINS操作符
            tag_filters.append(f'ARRAY_CONTAINS(tags, "{tag}")')

        # 组合多个标签的过滤条件(使用OR连接)
        filter_expr = " or ".join(tag_filters)
        
        # 添加额外过滤条件
        extra_filter = self._build_filter_expression(filter_params)
        if extra_filter:
            filter_expr = f"({filter_expr}) and {extra_filter}"
        
        logger.info(f"Generated filter expression: {filter_expr}")  # 添加日志记录

        # 获取总数
        total = self.get_total_count(filter_expr)
        if top_k:
            total = min(total, top_k)
            if offset >= total:
                return [], total
            page_size = min(page_size, total - offset)

        # 执行查询
        result = self.milvus_client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            offset=offset,
            limit=page_size,
            output_fields=['m_id', 'path', 'thumbnail_path', 'summary_txt', 'tags', 'title', 'vconfig_id', 'collect_start_time', 'collect_end_time']
        )

        # 为结果添加额外信息
        for video in result:
            video['timestamp'] = 0  # 添加默认时间戳
            video['tags'] = self._normalize_tags(video.get('tags'))

        return result, total
        
    def search_by_filter(self, page: int = 1, page_size: int = 6, top_k: int = 10, **filter_params) -> tuple[List[Dict[str, Any]], int]:
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
            Tuple[List[Dict[str, Any]], int]: 匹配的视频列表和总数
        """
        offset = (page - 1) * page_size

        # 构建过滤条件表达式
        filter_expr = self._build_filter_expression(filter_params)
        
        # 如果没有过滤条件，则返回空结果
        if not filter_expr:
            return [], 0
            
        logger.info(f"Generated filter expression: {filter_expr}")  # 添加日志记录

        # 获取总数
        total = self.get_total_count(filter_expr)
        if top_k:
            total = min(total, top_k)
            if offset >= total:
                return [], total
            page_size = min(page_size, total - offset)

        # 执行查询
        result = self.milvus_client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            offset=offset,
            limit=page_size,
            output_fields=['m_id', 'path', 'thumbnail_path', 'summary_txt', 'tags', 'title', 'vconfig_id', 'collect_start_time', 'collect_end_time']
        )

        # 为结果添加额外信息
        for video in result:
            video['timestamp'] = 0  # 添加默认时间戳
            video['tags'] = self._normalize_tags(video.get('tags'))

        return result, total
    
    def _build_filter_expression(self, filter_params: Dict[str, Any]) -> str:
        """
        根据过滤参数构建Milvus过滤表达式

        Args:
            filter_params: 过滤参数
                - vconfig_id: 车辆类型标识
                - collect_start_time: 采集开始时间
                - collect_end_time: 采集结束时间

        Returns:
            str: Milvus过滤表达式
        """
        conditions = []
        
        # 添加vconfig_id过滤条件
        if 'vconfig_id' in filter_params and filter_params['vconfig_id']:
            conditions.append(f'vconfig_id == "{filter_params["vconfig_id"]}"')
        
        # 添加collect_start_time过滤条件
        if 'collect_start_time' in filter_params and filter_params['collect_start_time'] is not None:
            conditions.append(f'collect_start_time >= {filter_params["collect_start_time"]}')
        
        # 添加collect_end_time过滤条件
        if 'collect_end_time' in filter_params and filter_params['collect_end_time'] is not None:
            conditions.append(f'collect_end_time <= {filter_params["collect_end_time"]}')
        
        # 组合所有条件（使用AND连接）
        if conditions:
            return " and ".join(conditions)
        
        return ""
