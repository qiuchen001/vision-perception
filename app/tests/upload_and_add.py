import os
import time

import requests
import json


def upload_and_insert_videos():
    video_dir = r"E:\playground\ai\datasets\videos\360-camera\videos"
    upload_url = "http://127.0.0.1:30012/vision-analyze/video/upload"
    insert_url = "http://127.0.0.1:30012/vision-analyze/video/add"

    for filename in os.listdir(video_dir):
        if filename.endswith(".mp4"):
            file_path = os.path.join(video_dir, filename)
            with open(file_path, 'rb') as video_file:
                upload_response = requests.post(upload_url, files={'video': video_file})
                upload_data = upload_response.json()
                print(f"上传结果: {upload_data}")

                time.sleep(1)

                if upload_response.status_code == 200:
                    insert_data = {
                        "video_url": upload_data['data']['video_url'],
                        "action_type": 3
                    }
                    insert_response = requests.post(insert_url, data=insert_data)
                    insert_result = insert_response.json()
                    print(f"插入结果: {insert_result}")
                else:
                    print(f"上传失败: {filename}")
                print("*" * 50)


upload_and_insert_videos()
