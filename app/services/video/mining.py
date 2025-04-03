from app.dao.video_dao import VideoDAO
from app.utils.common import *
import os
import json
from openai import OpenAI
from app.prompt import mining
from dotenv import load_dotenv
from pymilvus import MilvusClient
from app.utils.logger import logger
import time
import functools
from typing import TypeVar, Callable, Any

# 加载环境变量
load_dotenv()

# Milvus配置
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
COLLECTION_NAME = os.getenv("MILVUS_VIDEO_COLLECTION_NAME")

T = TypeVar("T")

def with_retry(
    max_retries: int = 3,
    initial_delay: float = 0.5,
    exceptions: tuple = (Exception,),
    on_retry: Callable[[int, Exception], None] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """重试装饰器
    
    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟时间(秒)
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数,参数为(重试次数,异常)
        
    Example:
        @with_retry(max_retries=3, initial_delay=1)
        def my_function():
            pass
            
        @with_retry(exceptions=(ValueError, TypeError))
        def another_function():
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    
                    # 对于查询操作,空结果也需要重试
                    if result or attempt == max_retries - 1:
                        return result
                        
                    delay = initial_delay * (2 ** attempt)
                    logger.info(f"{func.__name__}: Empty result, retry {attempt + 1}/{max_retries} after {delay}s")
                    time.sleep(delay)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries - 1:
                        raise
                        
                    delay = initial_delay * (2 ** attempt)
                    
                    if on_retry:
                        on_retry(attempt + 1, e)
                    else:
                        logger.warning(
                            f"{func.__name__} failed: {str(e)}, "
                            f"retry {attempt + 1}/{max_retries} after {delay}s"
                        )
                        
                    time.sleep(delay)
                    
            if last_exception:
                raise last_exception
                
            return None  # type: ignore
            
        return wrapper
        
    return decorator


def parse_json_string(json_str):
    cleaned_str = json_str.replace('\\n', '').replace('\\"', '"')
    cleaned_str = cleaned_str.strip('```json')
    parsed_data = json.loads(cleaned_str)
    return parsed_data


def time_to_seconds(time_str):
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("时间格式不正确，应为 '0:13' 或 '1:23:45'")


def seconds_to_time_format(total_seconds):
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02}:{seconds:02}"


def time_to_standard_format(time_range_str):
    start_time_str, end_time_str = time_range_str.split('-')
    start_seconds = time_to_seconds(start_time_str)
    end_seconds = time_to_seconds(end_time_str)
    start_time_formatted = seconds_to_time_format(start_seconds)
    end_time_formatted = seconds_to_time_format(end_seconds)
    return start_time_formatted, end_time_formatted


def format_mining_result(mining_result, video_url):
    mining_result_new = []
    for item in mining_result:
        if item['behaviour'] is None or item['behaviour'] == {} or \
                not isinstance(item['behaviour'], dict) or \
                item['behaviour'].get('behaviourId') is None or \
                item['behaviour'].get('behaviourName') is None:
            continue

        if len(item['behaviour']['timeRange'].split('-')) < 2:
            continue

        start_time_formatted, end_time_formatted = time_to_standard_format(item['behaviour']['timeRange'])
        time_range_str = f"{start_time_formatted}-{end_time_formatted}"
        item['behaviour']['timeRange'] = time_range_str
        start_time = time_to_seconds(start_time_formatted)
        thumbnail_file_name = os.path.basename(video_url) + "_t_" + str(start_time) + ".jpg"
        thumbnail_local_path = os.path.join('/tmp', thumbnail_file_name)
        generate_thumbnail_from_video(video_url, thumbnail_local_path, start_time)
        item['thumbnail_url'] = upload_thumbnail_to_oss(thumbnail_file_name, thumbnail_local_path)
        mining_result_new.append(item)
        os.remove(thumbnail_local_path)
    return mining_result_new


class MiningVideoService:
    def __init__(self):
        self.video_dao = VideoDAO()
        uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
        self.milvus_client = MilvusClient(
            uri=uri,
            db_name=os.getenv("MILVUS_DB_NAME")
        )

    def mining(self, video_url):
        mining_result = self.mining_video_handler(video_url)
        js = json.loads(mining_result)
        content = js['choices'][0]['message']['content']
        mining_json = parse_json_string(content)
        return format_mining_result(mining_json, video_url)

    @staticmethod
    def mining_video_handler(video_url):
        model_name = os.getenv("VISION_MODEL_NAME")

        client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )

        base64_images = extract_frames_and_convert_to_base64(video_url)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": base64_images
                    },
                    {
                        "type": "text",
                        "text": mining.system_instruction + "\n" + mining.prompt
                    }
                ]
            }
        ]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"}
        )
        return response.model_dump_json()

    @with_retry(max_retries=3, initial_delay=0.5)
    def mining_by_raw_id(self, json_data):
        """
        根据raw_id进行视频挖掘
        
        Args:
            json_data: 包含raw_id的json数据
            
        Returns:
            mining结果
            
        Raises:
            ValueError: 当raw_id不存在或找不到对应视频时
        """
        # 获取raw_id
        raw_id = json_data.get("raw_id")
        if not raw_id:
            raise ValueError("raw_id is required")

        # 查询视频path
        res = self.milvus_client.query(
            collection_name=COLLECTION_NAME,
            filter=f'resource_id == "{raw_id}"',
            output_fields=["path"]
        )

        if not res:
            raise ValueError(f"No video found for raw_id: {raw_id}")

        video_path = res[0]["path"]

        # 调用原有mining方法
        return self.mining(video_path)


if __name__ == "__main__":
    mining_video_service = MiningVideoService()
    mining_result = mining_video_service.mining_by_raw_id({"raw_id": "1e9f6957-4097-4a20-a9cf-f07d91e44cf8"})
    print(mining_result)
