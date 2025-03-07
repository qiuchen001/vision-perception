from PIL import Image
import uuid
from typing import Dict, Any, List

from app.dao.video_dao import VideoDAO
from app.utils.common import *
from app.utils.embedding.text_embedding import *
from app.utils.minio_uploader import MinioFileUploader
from app.utils.embedding.embedding_factory import EmbeddingFactory
from app.utils.milvus_operator import video_frame_operator
from werkzeug.utils import secure_filename
from config.config import Config
from app.utils.video_processor import VideoProcessor
from app.prompt.title import system_instruction, prompt


class UploadVideoService:
    def __init__(self):
        self.video_dao = VideoDAO()
        self.minioFileUploader = MinioFileUploader()
        self.frame_interval = Config.VIDEO_FRAME_INTERVAL
        self.batch_size = Config.VIDEO_FRAME_BATCH_SIZE
        self.video_processor = VideoProcessor()

    def upload(self, video_file) -> Dict[str, Any]:
        """
        上传视频并处理。
        
        Args:
            video_file: 上传的视频文件
            
        Returns:
            Dict[str, Any]: 包含视频URL和处理结果的字典
        """
        # 保存临时文件
        filename = secure_filename(video_file.filename)
        video_file_path = os.path.join('/tmp', filename)
        video_file.save(video_file_path)

        result = {
            "frame_count": 0,
            "processed_frames": 0
        }

        try:
            # 上传视频到OSS
            video_oss_url = upload_thumbnail_to_oss(filename, video_file_path)
            thumbnail_oss_url = self.minioFileUploader.generate_video_thumbnail_url(video_oss_url)
            
            # 处理视频帧
            frames = self._extract_frames(video_file_path)
            result["frame_count"] = len(frames)
            
            if frames:
                self._process_frames(video_oss_url, frames)
                result["processed_frames"] = len(frames)

            # 生成并更新标题
            title = self.generate_title(video_file_path)

            # 添加视频信息到数据库
            if not self.video_dao.check_url_exists(video_oss_url):
                embedding = embed_fn(" ")
                summary_embedding = embed_fn(" ")
                self.video_dao.init_video(video_oss_url, embedding, summary_embedding, thumbnail_oss_url, title)
            
            result.update({
                "file_name": video_oss_url,
                "video_url": video_oss_url,
                "title": title
            })

        except Exception as e:
            logger.error(f"处理视频失败: {str(e)}")
            raise
        finally:
            # 清理临时文件
            os.remove(video_file_path)
            logger.debug(f"Deleted temporary file: {video_file_path}")

        return result

    def _extract_frames(self, video_path: str) -> List[Image.Image]:
        """提取视频帧"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % self.frame_interval == 0:
                    # 转换为PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                    
                frame_count += 1
        finally:
            cap.release()
            
        return frames

    def _process_frames(self, video_url: str, frames: List[Image.Image]) -> None:
        """
        处理视频帧并存入向量数据库。
        
        Args:
            video_url: 视频文件URL
            frames: 提取的视频帧列表
        """
        m_ids, embeddings, paths, at_seconds = [], [], [], []

        # 获取视频的FPS
        cap = cv2.VideoCapture(video_url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        for idx, frame in enumerate(frames):
            try:
                # 获取embedding实例
                embedding = EmbeddingFactory.create_embedding()

                if embedding is None:
                    continue

                # 准备数据
                m_ids.append(str(uuid.uuid4()))
                # embeddings.append(embedding[0].detach().cpu().numpy().tolist())
                embeddings.append(embedding)
                paths.append(video_url)

                # 正确计算时间戳（秒）
                # 当前帧实际的帧号 = 索引 * 帧间隔
                # 时间戳 = 帧号 / FPS
                frame_number = idx * self.frame_interval
                timestamp = int(frame_number / fps)
                at_seconds.append(timestamp)

                # 使用配置的批处理大小
                if len(m_ids) >= self.batch_size:
                    video_frame_operator.insert_data([m_ids, embeddings, paths, at_seconds])
                    logger.info(f"批量插入 {len(m_ids)} 帧，时间戳范围: {at_seconds[0]}-{at_seconds[-1]}秒")
                    m_ids, embeddings, paths, at_seconds = [], [], [], []

            except Exception as e:
                logger.error(f"处理帧 {idx} 失败: {str(e)}")
                continue

        # 处理剩余的帧
        if m_ids:
            video_frame_operator.insert_data([m_ids, embeddings, paths, at_seconds])
            logger.info(f"批量插入剩余 {len(m_ids)} 帧，时间戳范围: {at_seconds[0]}-{at_seconds[-1]}秒")

    def generate_title(self, video_path):
        """生成视频标题"""
        # 1. 提取关键帧
        frame_urls = self.video_processor.extract_key_frames(video_path)

        # 2. 调用通义千问VL模型
        client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )

        messages = [{
            "role": "system",
            "content": system_instruction
        }, {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frame_urls
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]

        response = client.chat.completions.create(
            model=os.getenv("VISION_MODEL_NAME"),
            messages=messages,
            response_format={"type": "json_object"}
        )

        response_json = response.model_dump_json()
        js = json.loads(response_json)
        content = js['choices'][0]['message']['content']
        title_json = json.loads(content)
        return title_json["title"]