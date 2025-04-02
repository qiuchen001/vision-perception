import json
from app.services.video.mining import MiningVideoService
from app.dao.video_dao import VideoDAO


def flink_job_init():
    return True


def flink_job_cleanup():
    return True


def parse_mining_result(mining_results):
    tags = []
    for item in mining_results:
        tag = item["behaviour"]["behaviourName"]
        tags.append(tag)
    return tags


def process_mining_result(raw_id, mining_results):
    video_dao = VideoDAO()
    video_info = video_dao.get_by_resource_id(raw_id)
    video = video_info[0]

    tags = parse_mining_result(mining_results)
    video['tags'] = tags
    video['mining_results'] = mining_results

    upsert_res = video_dao.upsert_video(video)
    print("upsert_res:", upsert_res)


def flink_job_execute(inputMessageStr):
    global result
    inputMessage = json.loads(inputMessageStr)

    taskid = inputMessage.get("taskid")
    raw_id = inputMessage.get("raw_id")

    try:
        service = MiningVideoService()
        result = service.mining_by_raw_id({"raw_id": raw_id})
        print(f"视频挖掘完成,共发现 {len(result)} 个行为片段, result: {result}")

        process_mining_result(raw_id, result)

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
        # "raw_id": "9a8afc7e-19de-4e13-8b3c-44794ccb49c6"
        # "raw_id": "9a8afc7e-19de-4e13-8b3c-44794ccb49c6"
        # "raw_id": "c87a2a25-bbf6-4d07-94c3-acbde311fea6"
        "raw_id": "3eba0cc6-d627-4b7e-b80b-3e21e077b4e7"
    }
    flink_job_execute(json.dumps(inputMessage))
