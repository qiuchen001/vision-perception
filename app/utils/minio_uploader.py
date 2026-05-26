import os
import json

from minio import Minio
from minio.error import S3Error
from urllib.parse import urljoin
import mimetypes
import ffmpeg
from ..utils.logger import logger
from dotenv import load_dotenv
load_dotenv()


class MinioFileUploader:
    _public_policy_configured = set()

    def __init__(self):
        """
        初始化 MinIO 客户端
        """
        self.minio_client = Minio(
            os.getenv('OSS_ENDPOINT'),  # MinIO 服务端点
            access_key=os.getenv('OSS_ACCESS_KEY'),  # 访问密钥
            secret_key=os.getenv('OSS_SECRET_KEY'),  # 秘密密钥
            secure=False  # 如果你的 Minio 实例没有启用 SSL，请将 secure 参数设置为 False
        )

    def _get_bucket_name(self):
        bucket_name = os.getenv('OSS_BUCKET_NAME')
        if not bucket_name:
            raise ValueError("OSS_BUCKET_NAME 未配置")
        return bucket_name

    def _ensure_bucket_ready(self, bucket_name):
        found = self.minio_client.bucket_exists(bucket_name)
        if not found:
            self.minio_client.make_bucket(bucket_name)
            print(f"桶 {bucket_name} 已创建")
        else:
            print(f"桶 {bucket_name} 已存在")

        if bucket_name in self._public_policy_configured:
            return

        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
                }
            ],
        }
        self.minio_client.set_bucket_policy(bucket_name, json.dumps(policy))
        self._public_policy_configured.add(bucket_name)

    def upload_file(self, object_name, file_path):
        """
        上传文件到 MinIO
        :param object_name: 对象名（包含路径）
        :param file_path: 本地文件路径
        """
        bucket_name = self._get_bucket_name()
        self._ensure_bucket_ready(bucket_name)

        try:
            # 获取文件的 MIME 类型
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                content_type = "application/octet-stream"  # 默认 MIME 类型

            # 上传文件
            self.minio_client.fput_object(bucket_name, object_name, file_path, content_type=content_type)
            print(f"文件 {file_path} 已上传到 {bucket_name}/{object_name}")
        except S3Error as e:
            raise RuntimeError(f"上传文件到 MinIO 失败: {e}") from e

        public_base_url = os.getenv('OSS_PUBLIC_BASE_URL', '/media').rstrip('/')
        return f"{public_base_url}/{bucket_name}/{object_name}"

    def generate_thumbnail_from_video(self, video_url, thumbnail_path, time_seconds):
        if not video_url:
            raise ValueError("视频URL不能为空")
        try:
            (
                ffmpeg
                .input(video_url, ss=time_seconds)
                .output(thumbnail_path, vframes=1)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            stderr = e.stderr.decode(errors='replace') if e.stderr else str(e)
            raise RuntimeError(f"生成视频缩略图失败: {stderr}") from e

    def upload_thumbnail_to_oss(self, object_name, file_path):
        # 创建 MinioFileUploader 实例
        uploader = MinioFileUploader()
        return uploader.upload_file(object_name, file_path)


    def generate_video_thumbnail_url(self, video_url):
        start_time = 0
        thumbnail_file_name = os.path.basename(video_url) + "_t_" + str(start_time) + ".jpg"
        thumbnail_local_path = os.path.join('/tmp', thumbnail_file_name)
        self.generate_thumbnail_from_video(video_url, thumbnail_local_path, start_time)
        thumbnail_oss_url = self.upload_thumbnail_to_oss(thumbnail_file_name, thumbnail_local_path)
        print(f"thumbnail_oss_url:{thumbnail_oss_url}")
        os.remove(thumbnail_local_path)
        logger.debug(f"Deleted temporary file: {thumbnail_local_path}")
        return thumbnail_oss_url


# 示例用法
if __name__ == "__main__":
    # 创建 MinioFileUploader 实例
    uploader = MinioFileUploader()

    # 上传文件
    # bucket_name = os.getenv('OSS_BUCKET_NAME')
    # object_name = "path/to/your/object.txt"
    # file_path = "local/path/to/your/file.txt"

    object_name = "b7ec1001240181ceb5ec3e448c7f9b78.mp4"
    file_path = r"E:\workspace\ai-ground\videos\mining-well\b7ec1001240181ceb5ec3e448c7f9b78.mp4"

    uploader.upload_file(object_name, file_path)