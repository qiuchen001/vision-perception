import os
from typing import List, Dict, Any, Union
from app.services.video.intent import IntentService
from app.services.video.search import SearchVideoService
from app.utils.logger import logger


class IntegratedSearchService:
    """集成搜索服务类"""

    def __init__(self):
        """初始化服务"""
        self.intent_service = IntentService()
        self.search_service = SearchVideoService()

    def _extract_search_params(self, intent_result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从意图识别结果中提取搜索参数。

        Args:
            intent_result: 意图识别结果

        Returns:
            Dict[str, Any]: 包含tags和text的字典
        """
        search_params = {
            "tags": [],
            "text": []
        }

        for item in intent_result:
            if item["type"] == "tag":
                search_params["tags"].extend(item["list"])
            elif item["type"] == "text":
                search_params["text"].extend(item["list"])

        return search_params

    def _merge_search_results(self, results1: List[Dict[str, Any]], results2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并两个搜索结果列表,去除重复项。

        Args:
            results1: 第一个结果列表
            results2: 第二个结果列表

        Returns:
            List[Dict[str, Any]]: 合并后的结果列表
        """
        # 使用视频路径作为唯一标识
        seen_paths = set()
        merged_results = []

        for result in results1 + results2:
            if result.get("thumbnail_path") not in seen_paths:
                seen_paths.add(result.get("thumbnail_path"))
                # seen_paths.add(result.get("path"))
                merged_results.append(result)

        return merged_results

    def search(self, query: str, page: int = 1, page_size: int = 6, **filter_params) -> tuple[List[Dict[str, Any]], int]:
        """
        根据用户查询进行集成搜索。

        Args:
            query: 用户查询文本
            page: 页码
            page_size: 每页数量
            **filter_params: 附加过滤条件
                - vconfig_id: 车辆类型标识
                - collect_start_time: 采集开始时间
                - collect_end_time: 采集结束时间

        Returns:
            Tuple[List[Dict[str, Any]], int]: 搜索结果列表和总数
        """
        try:
            # 1. 进行意图识别
            intent_result = self.intent_service.recognize_intent(query)
            if not intent_result:
                logger.warning(f"意图识别结果为空: {query}")
                return [], 0
            logger.info(f"意图识别结果: {intent_result}")

            # 2. 提取搜索参数
            search_params = self._extract_search_params(intent_result)
            
            # 3. 根据意图类型执行搜索
            has_tags = bool(search_params["tags"])
            has_text = bool(search_params["text"])

            if has_tags and has_text:
                # 混合搜索
                tag_results, tag_total = self.search_service.search_by_tags(
                    tags=search_params["tags"],
                    page=1,  # 先获取所有结果再合并
                    page_size=100,
                    **filter_params  # 传递过滤参数
                )
                # 为标签搜索结果添加相似度分数
                for result in tag_results:
                    result['similarity'] = '1.0000'  # 标签完全匹配设为1.0

                text_results = []
                text_total = 0
                for text in search_params["text"]:
                    results, total = self.search_service.search_by_text(
                        txt=text,
                        page=1,  # 先获取所有结果再合并
                        page_size=100,
                        **filter_params  # 传递过滤参数
                    )
                    text_results.extend(results)
                    text_total += total
                
                # 合并结果并分页
                all_results = self._merge_search_results(tag_results, text_results)
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                return all_results[start_idx:end_idx], len(all_results)

            elif has_tags:
                # 纯标签搜索
                results, total = self.search_service.search_by_tags(
                    tags=search_params["tags"],
                    page=page,
                    page_size=page_size,
                    **filter_params  # 传递过滤参数
                )
                # 为标签搜索结果添加相似度分数
                for result in results:
                    result['similarity'] = '1.0000'  # 标签完全匹配设为1.0
                return results, total

            elif has_text:
                # 纯文本搜索
                results = []
                total = 0
                for text in search_params["text"]:
                    text_results, text_total = self.search_service.search_by_text(
                        txt=text,
                        page=page,
                        page_size=page_size,
                        **filter_params  # 传递过滤参数
                    )
                    results.extend(text_results)
                    total += text_total
                return results[:page_size], total  # 限制返回数量

            else:
                logger.warning(f"未识别到有效的搜索意图: {query}")
                return [], 0

        except Exception as e:
            logger.error(f"集成搜索失败: {str(e)}")
            return [], 0


# 使用示例
if __name__ == "__main__":
    # 创建服务实例
    search_service = IntegratedSearchService()

    # 测试不同类型的搜索
    queries = [
        # "找一下标签是急刹车的视频",  # 纯标签搜索
        # "搜索视频中有人闯红灯的画面",  # 纯文本搜索
        # "查找标签是急刹车且内容包含闯红灯的视频"  # 混合搜索

        "视频记录了一段城市道路的行车情况，天气为多云，光线条件适中。车辆在城市道路上行驶"
    ]

    for query in queries:
        print(f"\n搜索查询: {query}")
        results, total = search_service.search(query)
        print(f"搜索结果数量: {len(results)}")
        for result in results:
            print(f"- {result.get('path')}") 