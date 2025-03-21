import json
from app.services.video.mining import MiningVideoService


def flink_job_init():
    return True


def flink_job_cleanup():
    return True


def flink_job_execute(inputMessageStr):
    global result
    inputMessage = json.loads(inputMessageStr)

    taskid = inputMessage.get("taskid")
    raw_id = inputMessage.get("raw_id")

    try:
        service = MiningVideoService()
        result = service.mining_by_raw_id({"raw_id": raw_id})
        print(f"视频挖掘完成,共发现 {len(result)} 个行为片段, result: {result}")

        message = {
            "public-vision-perception-mining-success": [
                result
            ]
        }

        return json.dumps(message)
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return None


if __name__ == "__main__":
    inputMessage = {
        "taskid": "9cf99967-aa69-44cd-b4f2-d22d886817fb",
        # "raw_id": "1e9f6957-4097-4a20-a9cf-f07d91e44cf8"
        "raw_id": "9a8afc7e-19de-4e13-8b3c-44794ccb49c6"
    }
    flink_job_execute(json.dumps(inputMessage))
