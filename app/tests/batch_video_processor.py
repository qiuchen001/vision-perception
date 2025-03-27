import os
import sys
import time
import requests
from pathlib import Path
import logging
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_processing.log')
    ]
)

class BatchVideoProcessor:
    """批量视频处理器"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.supported_formats = {'.mp4', '.avi', '.mkv', '.mov', '.wmv'}
        
    def scan_directory(self, directory: str) -> List[Path]:
        """
        扫描目录获取视频文件列表
        
        Args:
            directory: 要扫描的目录路径
            
        Returns:
            List[Path]: 视频文件路径列表
        """
        video_files = []
        try:
            for file in Path(directory).rglob('*'):
                if file.suffix.lower() in self.supported_formats:
                    video_files.append(file)
            logging.info(f"找到 {len(video_files)} 个视频文件")
            return video_files
        except Exception as e:
            logging.error(f"扫描目录失败: {str(e)}")
            return []
            
    def upload_video(self, video_path: Path) -> Dict[str, Any]:
        """
        上传视频文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            Dict[str, Any]: 上传响应数据
        """
        try:
            with open(video_path, 'rb') as f:
                files = {'file': (video_path.name, f, 'video/mp4')}
                response = requests.post(
                    f"{self.base_url}/api/upload",
                    files=files
                )
                response.raise_for_status()
                result = response.json()
                
                if result.get('status') == 'success':
                    data = result.get('data', {})
                    # 确保返回必要的字段
                    if 'video_url' not in data:
                        raise Exception('上传响应中缺少video_url字段')
                        
                    # 设置默认值
                    data.setdefault('title', video_path.stem)
                    data.setdefault('tags', [])
                    data.setdefault('summary_txt', '')
                    
                    logging.info(f"上传成功: {video_path.name}")
                    return data
                else:
                    raise Exception(result.get('message', '上传失败'))
                    
        except Exception as e:
            logging.error(f"上传视频失败 {video_path.name}: {str(e)}")
            raise
            
    def process_video(self, video_data: Dict[str, Any]) -> bool:
        """
        处理视频
        
        Args:
            video_data: 视频数据,包含video_url等信息
            
        Returns:
            bool: 处理是否成功
        """
        try:
            # 构建请求数据
            request_data = {
                "video_url": video_data.get('video_url'),
                "title": video_data.get('title', os.path.basename(video_data.get('video_url', ''))),
                "tags": video_data.get('tags', []),
                "summary_txt": video_data.get('summary_txt', ''),
                "thumbnail_url": video_data.get('thumbnail_url', ''),
                "action_type": 3
            }
            
            # 发送请求
            response = requests.post(
                f"{self.base_url}/api/add",
                json=request_data
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get('status') == 'success':
                logging.info(f"处理成功: {video_data.get('video_url')}")
                return True
            else:
                raise Exception(result.get('message', '处理失败'))
                
        except Exception as e:
            logging.error(f"处理视频失败 {video_data.get('video_url')}: {str(e)}")
            return False
            
    def process_directory(self, directory: str, max_retries: int = 3) -> None:
        """
        处理目录中的所有视频
        
        Args:
            directory: 视频目录路径
            max_retries: 最大重试次数
        """
        video_files = self.scan_directory(directory)
        if not video_files:
            logging.warning("没有找到视频文件")
            return
            
        for video_path in video_files:
            retries = 0
            while retries < max_retries:
                try:
                    # 上传视频
                    upload_result = self.upload_video(video_path)
                    
                    # 处理视频
                    if self.process_video(upload_result):
                        break
                    
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        logging.error(f"处理视频失败,已达到最大重试次数: {video_path.name}")
                    else:
                        logging.warning(f"处理失败,将重试 ({retries}/{max_retries}): {video_path.name}")
                        time.sleep(2)  # 重试前等待
                        
    def process_videos(self, directory: str, max_retries: int = 3) -> None:
        """
        处理指定目录中的所有视频
        
        Args:
            directory: 视频目录路径
            max_retries: 最大重试次数
        """
        if not os.path.isdir(directory):
            logging.error(f"目录不存在: {directory}")
            return
            
        self.process_directory(directory, max_retries)

def main():
    """主函数"""
    # 支持两种使用方式:
    # 1. 命令行参数指定目录
    # 2. 代码中直接指定目录
    
    # 示例1: 通过命令行参数
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        processor = BatchVideoProcessor()
        processor.process_videos(directory)
        return
        
    # 示例2: 直接在代码中指定目录
    video_directory = "E:/workspace/ai-ground/removed-video-mark-clear"  # 在这里指定你的视频目录
    processor = BatchVideoProcessor()
    processor.process_videos(video_directory)
    
if __name__ == "__main__":
    main()

# python batch_video_processor.py E:\workspace\ai-ground\videos-new