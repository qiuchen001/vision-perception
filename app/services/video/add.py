from app.dao.video_dao import VideoDAO
from app.services.video.features import VideoFeatureService
from app.services.video.mining import MiningVideoService
from app.services.video.summary import SummaryVideoService
from app.utils.embedding.text_embedding import *


class AddVideoService:
    def __init__(self):
        self.video_dao = VideoDAO()
        self.feature_service = VideoFeatureService()

    @staticmethod
    def parse_mining_result(mining_results):
        if isinstance(mining_results, dict):
            tags = mining_results.get("tags", [])
            return tags if isinstance(tags, list) else []
        tags = []
        for item in mining_results:
            tag = item["behaviour"]["behaviourName"]
            tags.append(tag)
        return tags

    @staticmethod
    def parse_summary_result(summary_result):
        return summary_result['summary']

    @staticmethod
    def _emit_progress(progress_callback, stage, message, detail=None):
        if progress_callback:
            progress_callback(stage, message, detail or {})

    def add(self, video_url, action_type=3, process_video_url=None, progress_callback=None):
        self._emit_progress(progress_callback, "processing", "处理中...")
        video_info = self.video_dao.get_by_path(video_url)
        if len(video_info) == 0:
            raise ValueError("Video not found")

        video = video_info[0]
        process_video_url = process_video_url or video_url
        try:
            action_type = int(action_type or 3)
        except ValueError:
            raise ValueError("action_type must be an integer")

        mining_results = None
        if action_type == 1:
            mining_results = self.process_mining(video, process_video_url, progress_callback)
        elif action_type == 2:
            mining_results = self.process_mining(video, process_video_url, progress_callback)
            self.process_summary(video, process_video_url, mining_results, progress_callback)
            self.process_features(video, progress_callback)
        elif action_type == 3:
            mining_results = self.process_mining(video, process_video_url, progress_callback)
            self.process_summary(video, process_video_url, mining_results, progress_callback)
            self.process_features(video, progress_callback)
        else:
            raise ValueError("无效的操作类型")

        self._emit_progress(progress_callback, "saving", "保存结果中...")
        upsert_res = self.video_dao.upsert_video(video)
        print("upsert_res:", upsert_res)
        self._emit_progress(progress_callback, "complete", "处理完成")

        return video['m_id']

    def process_mining(self, video, video_url, progress_callback=None):
        self._emit_progress(progress_callback, "vlm", "VLM分析中...")
        mining_service = MiningVideoService()
        mining_results = mining_service.mining(video_url, progress_callback=progress_callback)
        tags = self.parse_mining_result(mining_results)
        video['tags'] = tags
        video['mining_results'] = mining_results
        return mining_results

    def process_summary(self, video, video_url, mining_results=None, progress_callback=None):
        self._emit_progress(progress_callback, "summary", "摘要生成中...")
        summary_service = SummaryVideoService()
        summary_result = summary_service.summary(video_url, mining_results)
        summary_txt = self.parse_summary_result(summary_result)
        video['summary_txt'] = summary_txt
        video['summary_embedding'] = embed_fn(summary_txt)

    def process_features(self, video, progress_callback=None):
        self._emit_progress(progress_callback, "features", "特征提取中...")
        self.feature_service.upsert_text_features(video)
        self.feature_service.upsert_visual_feature(video)
