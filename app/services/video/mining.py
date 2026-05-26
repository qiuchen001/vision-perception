import functools
import os
import time
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv
from pymilvus import MilvusClient

from app.algorithm.scene_mining.adapter import analyze_video, extract_tags
from app.dao.video_dao import VideoDAO
from app.utils.logger import logger

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
COLLECTION_NAME = os.getenv("MILVUS_VIDEO_COLLECTION_NAME")

T = TypeVar("T")


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 0.5,
    exceptions: tuple = (Exception,),
    on_retry: Callable[[int, Exception], None] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    if result or attempt == max_retries - 1:
                        return result
                    delay = initial_delay * (2 ** attempt)
                    logger.info("%s: empty result, retry %s/%s after %.1fs", func.__name__, attempt + 1, max_retries, delay)
                    time.sleep(delay)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        raise
                    delay = initial_delay * (2 ** attempt)
                    if on_retry:
                        on_retry(attempt + 1, e)
                    else:
                        logger.warning("%s failed: %s, retry %s/%s after %.1fs", func.__name__, e, attempt + 1, max_retries, delay)
                    time.sleep(delay)
            if last_exception:
                raise last_exception
            return None  # type: ignore

        return wrapper

    return decorator


class MiningVideoService:
    def __init__(self):
        self.video_dao = VideoDAO()
        uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
        self.milvus_client = MilvusClient(
            uri=uri,
            db_name=os.getenv("MILVUS_DB_NAME"),
        )

    def mining(self, video_url: str, progress_callback=None) -> dict[str, Any]:
        scene_result = analyze_video(video_url, progress_callback=progress_callback)
        tags = extract_tags(scene_result.summary_item)[:10]
        return {
            "pred": scene_result.summary_item.get("pred", {}),
            "abnormal_event_times": scene_result.summary_item.get("abnormal_event_times", []),
            "tags": tags,
            "final_output": scene_result.final_output,
            "raw_outputs": scene_result.raw_outputs,
            "output_dir": scene_result.output_dir,
            "video_path": scene_result.video_path,
            "local_video_path": scene_result.local_video_path,
        }

    @with_retry(max_retries=3, initial_delay=0.5)
    def mining_by_raw_id(self, json_data):
        raw_id = json_data.get("raw_id")
        if not raw_id:
            raise ValueError("raw_id is required")

        res = self.milvus_client.query(
            collection_name=COLLECTION_NAME,
            filter=f'resource_id == "{raw_id}"',
            output_fields=["path"],
        )
        if not res:
            raise ValueError(f"No video found for raw_id: {raw_id}")

        video_path = res[0]["path"]
        return self.mining(video_path)


if __name__ == "__main__":
    mining_video_service = MiningVideoService()
    mining_result = mining_video_service.mining_by_raw_id({"raw_id": "1e9f6957-4097-4a20-a9cf-f07d91e44cf8"})
    print(mining_result)
