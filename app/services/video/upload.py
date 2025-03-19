import pdb

from PIL import Image
import uuid
from typing import Dict, Any, List, Optional, Tuple
from werkzeug.datastructures import FileStorage
import tempfile
import numpy as np
import subprocess
import shutil

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

            # 生成resource_id
            resource_id = str(uuid.uuid4())

            if frames:
                self._process_frames(video_oss_url, frames, resource_id)
                result["processed_frames"] = len(frames)

            # 生成并更新标题
            title = self.generate_title(video_file_path)

            # 添加视频信息到数据库
            if not self.video_dao.check_url_exists(video_oss_url):
                embedding = embed_fn(" ")
                summary_embedding = embed_fn(" ")
                self.video_dao.init_video(
                    video_oss_url,
                    embedding,
                    summary_embedding,
                    thumbnail_oss_url,
                    title,
                    resource_id  # 传入resource_id
                )

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

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        提取视频帧。

        Args:
            video_path: 视频文件路径

        Returns:
            List[np.ndarray]: 提取的视频帧列表，BGR格式的numpy数组
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
                    # 直接使用BGR格式的frame
                    if frame is None or frame.size == 0:
                        logger.warning(f"跳过空帧: frame_count={frame_count}")
                        continue
                        
                    logger.debug(f"提取帧: frame_count={frame_count}, shape={frame.shape}, dtype={frame.dtype}")
                    frames.append(frame)

                frame_count += 1
            
            logger.info(f"总共提取了 {len(frames)} 帧")
            
        finally:
            cap.release()

        if not frames:
            raise ValueError("没有提取到任何有效帧")
        
        return frames

    def _process_frames(self, video_url: str, frames: List[np.ndarray], resource_id: str) -> None:
        """
        处理视频帧并存入向量数据库。

        Args:
            video_url: 视频文件URL
            frames: 提取的视频帧列表，BGR格式的numpy数组
            resource_id: 资源ID，用于关联原始数据
        """
        batch_data = {
            'm_ids': [],
            'embeddings': [],
            'paths': [],
            'resource_id': [],
            'at_seconds': []
        }

        cap = cv2.VideoCapture(video_url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        def insert_batch(data: Dict[str, List]):
            if not data['m_ids']:
                return

            try:
                insert_data = [
                    data['m_ids'],
                    data['embeddings'],
                    data['paths'],
                    data['resource_id'],
                    data['at_seconds']
                ]
                video_frame_operator.insert_data(insert_data)
                logger.info(f"批量插入 {len(data['m_ids'])} 帧")
            except Exception as e:
                logger.error(f"插入数据失败: {str(e)}")
                raise

        def clear_batch(data: Dict[str, List]):
            for key in data:
                data[key] = []

        def prepare_image_for_embedding(frame: np.ndarray) -> Optional[Image.Image]:
            """准备图片用于embedding"""
            try:
                # 确保frame是有效的numpy数组
                if frame is None or frame.size == 0:
                    return None
                    
                # 转换为RGB格式
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 转换为PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # 检查图片模式
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # 检查图片尺寸，如果太大则调整大小
                max_size = 1024  # 设置最大尺寸
                if pil_image.size[0] > max_size or pil_image.size[1] > max_size:
                    ratio = max_size / max(pil_image.size)
                    new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                    
                logger.debug(f"准备好的图片: size={pil_image.size}, mode={pil_image.mode}")
                return pil_image
                
            except Exception as e:
                logger.error(f"准备图片失败: {str(e)}")
                return None

        for idx, frame in enumerate(frames):
            try:
                # 准备图片
                pil_image = prepare_image_for_embedding(frame)
                if pil_image is None:
                    logger.warning(f"跳过无效帧: idx={idx}")
                    continue
                
                # 获取embedding
                embedding_model = EmbeddingFactory.create_embedding()
                embedding = embedding_model.embedding_image(pil_image)
                
                if embedding is None:
                    logger.warning(f"跳过无embedding的帧: idx={idx}")
                    continue

                batch_data['m_ids'].append(str(uuid.uuid4()))
                batch_data['embeddings'].append(embedding)
                batch_data['paths'].append(video_url)
                batch_data['resource_id'].append(str(resource_id))

                frame_number = idx * self.frame_interval
                timestamp = int(frame_number / fps)
                batch_data['at_seconds'].append(timestamp)

                if len(batch_data['m_ids']) >= self.batch_size:
                    insert_batch(batch_data)
                    clear_batch(batch_data)

            except Exception as e:
                logger.error(f"处理帧 {idx} 失败: {str(e)}")
                continue

        if batch_data['m_ids']:
            insert_batch(batch_data)

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

            # 使用传入的raw_id或生成新的resource_id
            resource_id = raw_id if raw_id else str(uuid.uuid4())

            if frames:
                self._process_frames(video_oss_url, frames, resource_id)
                result["processed_frames"] = len(frames)

            # 生成并更新标题
            title = self.generate_title(video_path)

            # 添加视频信息到数据库
            if not self.video_dao.check_url_exists(video_oss_url):
                embedding = embed_fn(" ")
                summary_embedding = embed_fn(" ")
                self.video_dao.init_video(
                    video_oss_url,
                    embedding,
                    summary_embedding,
                    thumbnail_oss_url,
                    title,
                    resource_id  # 传入resource_id
                )

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

    def _preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """预处理图片，确保尺寸和格式正确"""
        try:
            # 检查图片是否为空
            if image is None:
                raise ValueError("输入图片为空")

            # 打印图片信息
            logger.info(f"原始图片信息: shape={image.shape}, dtype={image.dtype}")

            # 确保图片是BGR格式(OpenCV默认格式)
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"图片格式不正确: shape={image.shape}")

            # 调整图片尺寸
            height, width = target_size
            if image.shape[:2] != (height, width):
                logger.info(f"调整图片尺寸从 {image.shape[:2]} 到 {(height, width)}")
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

            # 检查像素值范围
            logger.info(f"图片像素值范围: min={image.min()}, max={image.max()}")
            return image

        except Exception as e:
            logger.error(f"图片预处理失败: {str(e)}")
            raise

    def _create_video_from_images(self, collection: str, image_files: List[Dict[str, Any]], output_path: str):
        """从图片序列创建视频"""
        if not image_files:
            raise ValueError("没有图片文件可以处理")

        try:
            logger.info("开始生成视频...")
            # 创建临时目录存放图片
            temp_dir = tempfile.mkdtemp()
            
            try:
                # 下载并保存所有图片
                for i, file_info in enumerate(image_files):
                    image = self._download_image(collection, file_info["filename"])
                    cv2.imwrite(os.path.join(temp_dir, f"frame_{i:04d}.jpg"), image)
                
                # 使用ffmpeg生成视频
                cmd = f'ffmpeg -y -framerate 30 -i "{temp_dir}/frame_%04d.jpg" -c:v libx264 -preset medium -crf 23 "{output_path}"'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg生成视频失败: {result.stderr}")
                    
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    raise RuntimeError("生成的视频文件无效")
                    
            finally:
                # 清理临时目录
                shutil.rmtree(temp_dir)
                
        except Exception as e:
            logger.error(f"生成视频失败: {str(e)}")
            raise

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
