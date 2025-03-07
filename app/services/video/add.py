from app.dao.video_dao import VideoDAO
from app.services.video.mining import MiningVideoService
from app.services.video.summary import SummaryVideoService
from app.utils.embedding.text_embedding import *


class AddVideoService:
    def __init__(self):
        self.video_dao = VideoDAO()

    @staticmethod
    def parse_mining_result(mining_results):
        tags = []
        for item in mining_results:
            tag = item["behaviour"]["behaviourName"]
            tags.append(tag)
        return tags

    @staticmethod
    def parse_summary_result(summary_result):
        return summary_result['summary']

    def add(self, video_url, action_type):
        video_info = self.video_dao.get_by_path(video_url)
        if len(video_info) == 0:
            raise ValueError("Video not found")

        video = video_info[0]

        if action_type == 1:
            self.process_mining(video, video_url)
        elif action_type == 2:
            self.process_summary(video, video_url)
        elif action_type == 3:
            self.process_mining(video, video_url)
            self.process_summary(video, video_url)
        else:
            raise ValueError("无效的操作类型")

        self.video_dao.upsert_video(video)
        return video['m_id']

    def process_mining(self, video, video_url):
        mining_service = MiningVideoService()
        mining_result = mining_service.mining(video_url)
        tags = self.parse_mining_result(mining_result)
        video['tags'] = tags

    def process_summary(self, video, video_url):
        summary_service = SummaryVideoService()
        summary_result = summary_service.summary(video_url)
        summary_txt = self.parse_summary_result(summary_result)
        video['summary_txt'] = summary_txt
        video['summary_embedding'] = embed_fn(summary_txt)
