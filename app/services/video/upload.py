from PIL import Image
import uuid
from typing import Dict, Any, List, Optional, Tuple
from werkzeug.datastructures import FileStorage
import cv2
import os
import json
import requests
from openai import OpenAI
import tempfile
import numpy as np

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
        self.storage_service_base_url = os.getenv("STORAGE_SERVICE_BASE_URL", "http://10.66.12.37:30112")
        self.rawdata_service_base_url = os.getenv("RAWDATA_SERVICE_BASE_URL", "http://10.66.12.37:31557")

    def upload(self, video_file: FileStorage) -> Dict[str, Any]:
        """
        上传视频并处理。
        
        Args:
            video_file: 上传的视频文件，类型为FileStorage
            
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
                resource_id = str(uuid.uuid4())
                self._process_frames(video_oss_url, frames, resource_id)
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
        """
        提取视频帧。
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List[Image.Image]: 提取的视频帧列表
        """
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

    def _process_frames(self, video_url: str, frames: List[Image.Image], resource_id: str) -> None:
        """
        处理视频帧并存入向量数据库。
        
        Args:
            video_url: 视频文件URL
            frames: 提取的视频帧列表
            resource_id: 资源ID，用于关联原始数据
        """
        m_ids: List[str] = []
        embeddings: List[List[float]] = []
        paths: List[str] = []
        at_seconds: List[int] = []
        resource_ids: List[str] = []

        # 获取视频的FPS
        cap = cv2.VideoCapture(video_url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        for idx, frame in enumerate(frames):
            try:
                # 获取embedding实例
                embedding_model = EmbeddingFactory.create_embedding()
                embedding = embedding_model.embedding_image(frame)
                if embedding is None:
                    continue

                # 准备数据
                m_ids.append(str(uuid.uuid4()))
                embeddings.append(embedding)
                paths.append(video_url)
                resource_ids.append(resource_id)

                # 正确计算时间戳（秒）
                # 当前帧实际的帧号 = 索引 * 帧间隔
                # 时间戳 = 帧号 / FPS
                frame_number = idx * self.frame_interval
                timestamp = int(frame_number / fps)
                at_seconds.append(timestamp)

                # 使用配置的批处理大小
                if len(m_ids) >= self.batch_size:
                    video_frame_operator.insert_data([m_ids, embeddings, paths, at_seconds, resource_ids])
                    logger.info(f"批量插入 {len(m_ids)} 帧，时间戳范围: {at_seconds[0]}-{at_seconds[-1]}秒")
                    m_ids, embeddings, paths, at_seconds, resource_ids = [], [], [], [], []

            except Exception as e:
                logger.error(f"处理帧 {idx} 失败: {str(e)}")
                continue

        # 处理剩余的帧
        if m_ids:
            video_frame_operator.insert_data([m_ids, embeddings, paths, at_seconds, resource_ids])
            logger.info(f"批量插入剩余 {len(m_ids)} 帧，时间戳范围: {at_seconds[0]}-{at_seconds[-1]}秒")

    def generate_title(self, video_path: str) -> str:
        """
        生成视频标题。
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            str: 生成的视频标题
        """
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

    def process_data_path(self, data_path: str, raw_id: Optional[str] = None) -> Dict[str, Any]:
        """
        处理数据路径并生成视频。
        
        Args:
            data_path: 格式为 collection:path 的数据路径
            raw_id: 原始数据ID
            
        Returns:
            Dict[str, Any]: 包含视频URL和处理结果的字典
        """
        # 解析data_path
        collection, prefix = self._parse_data_path(data_path)

        # 获取所有图片文件
        image_files = self._get_all_files(collection, prefix)
        if not image_files:
            raise ValueError(f"未找到图片文件: {data_path}")

        # 创建临时视频文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            video_path = temp_video.name

        try:
            # 下载图片并生成视频
            self._create_video_from_images(collection, image_files, video_path)

            # 复用upload方法的处理逻辑
            result = {
                "frame_count": 0,
                "processed_frames": 0
            }

            # 上传视频到OSS
            video_filename = f"{uuid.uuid4()}.mp4"
            video_oss_url = upload_thumbnail_to_oss(video_filename, video_path)
            thumbnail_oss_url = self.minioFileUploader.generate_video_thumbnail_url(video_oss_url)

            # 处理视频帧
            frames = self._extract_frames(video_path)
            result["frame_count"] = len(frames)

            if frames:
                resource_id = raw_id if raw_id else str(uuid.uuid4())
                self._process_frames(video_oss_url, frames, resource_id)
                result["processed_frames"] = len(frames)

            # 生成并更新标题
            title = self.generate_title(video_path)

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

            return result

        except Exception as e:
            logger.error(f"处理数据路径失败: {str(e)}")
            raise
        finally:
            # 清理临时文件
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.debug(f"Deleted temporary video file: {video_path}")

    def _parse_data_path(self, data_path: str) -> Tuple[str, str]:
        """
        解析数据路径为collection和prefix，并添加avm-front子目录
        
        Args:
            data_path: 格式为 collection:path 的数据路径
            
        Returns:
            Tuple[str, str]: collection和处理后的prefix
        """
        parts = data_path.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"无效的数据路径格式: {data_path}")
        
        collection, prefix = parts
        
        # 确保prefix以/结尾
        if not prefix.endswith('/'):
            prefix += '/'
        
        # 添加avm-front子目录
        prefix += 'avm-front/'
        
        logger.info(f"解析数据路径: collection={collection}, prefix={prefix}")
        return collection, prefix

    def _get_all_files(self, collection: str, prefix: str) -> List[Dict[str, Any]]:
        """获取所有图片文件信息"""
        all_files = []
        page = 1

        while True:
            try:
                logger.info(f"获取文件列表: collection={collection}, prefix={prefix}, page={page}")
                # 调用获取文件列表API
                response = requests.get(
                    f"{self.storage_service_base_url}/filestore/{collection}/files",
                    params={
                        "prefix": prefix,
                        "page": page,
                        "page-size": 10,
                        "keyword": ""
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )

                if response.status_code != 200:
                    raise Exception(f"获取文件列表失败: {response.text}")

                data = response.json()
                files = data.get("files", [])

                # 过滤出图片文件并按时间戳排序
                image_files = [f for f in files if f["filename"].lower().endswith(('.jpg', '.jpeg', '.png'))]
                all_files.extend(image_files)

                logger.info(f"获取到 {len(image_files)} 个图片文件")

                # 检查是否需要继续翻页
                if len(files) < 10:
                    break

                page += 1

            except Exception as e:
                logger.error(f"获取文件列表失败: {str(e)}")
                raise

        if not all_files:
            logger.warning(f"未找到任何图片文件: collection={collection}, prefix={prefix}")
            return []

        # 按文件名排序（假设文件名包含时间戳）
        all_files.sort(key=lambda x: x["filename"])
        logger.info(f"总共获取到 {len(all_files)} 个图片文件")
        return all_files

    def _create_video_from_images(self, collection: str, image_files: List[Dict[str, Any]], output_path: str):
        """从图片序列创建视频"""
        if not image_files:
            raise ValueError("没有图片文件可以处理")

        try:
            logger.info("开始生成视频...")
            # 下载第一张图片来获取尺寸
            first_image = self._download_image(collection, image_files[0]["filename"])
            height, width = first_image.shape[:2]

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

            try:
                # 处理所有图片
                total = len(image_files)
                for i, file_info in enumerate(image_files, 1):
                    logger.debug(f"处理第 {i}/{total} 张图片: {file_info['filename']}")
                    image = self._download_image(collection, file_info["filename"])
                    out.write(image)

            finally:
                out.release()

            logger.info(f"视频生成完成: {output_path}")

        except Exception as e:
            logger.error(f"生成视频失败: {str(e)}")
            raise

    def _download_image(self, collection: str, filename: str) -> np.ndarray:
        """下载单个图片文件"""
        try:
            response = requests.get(
                f"{self.storage_service_base_url}/filestore/{collection}",
                params={"filename": filename},
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            if response.status_code != 200:
                raise Exception(f"下载图片失败: {filename}, 状态码: {response.status_code}")

            # 将响应内容转换为图片
            nparr = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise Exception(f"无法解码图片: {filename}")

            return image
        
        except Exception as e:
            error_msg = str(e)
            if hasattr(e, '__cause__') and e.__cause__:
                error_msg += f" (caused by: {str(e.__cause__)})"
            logger.error(f"下载图片失败: {filename} - {error_msg}")
            raise Exception(f"下载图片失败: {filename} - {error_msg}")

    def process_by_raw_id(self, raw_id: str) -> Dict[str, Any]:
        """
        通过rawId处理数据并生成视频。
        
        Args:
            raw_id: 原始数据ID
            
        Returns:
            Dict[str, Any]: 包含视频URL和处理结果的字典
        """
        try:
            # 调用获取合规数据API
            response = requests.get(
                f"{self.rawdata_service_base_url}/dataplatform/rawdata/{raw_id}",
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            if response.status_code != 200:
                raise Exception(f"获取数据失败: {response.text}")

            data = response.json()
            
            # 检查响应格式和状态
            if data.get("result") != "success" or "rawdata" not in data:
                raise Exception(f"无效的响应数据: {data}")
            
            # 获取dataPath
            data_path = data["rawdata"].get("dataPath")
            if not data_path:
                raise Exception(f"数据路径为空: rawId={raw_id}")
            
            logger.info(f"获取到数据路径: {data_path}")
            
            # 调用已有的process_data_path方法
            return self.process_data_path(data_path, raw_id)
            
        except Exception as e:
            logger.error(f"处理rawId失败: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        service = UploadVideoService()
        result = service.process_by_raw_id("1e9f6957-4097-4a20-a9cf-f07d91e44cf8")
        print(f"视频处理完成: {result['video_url']}")
    except Exception as e:
        print(f"处理失败: {str(e)}")
