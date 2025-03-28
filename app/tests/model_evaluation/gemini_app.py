import os
import time
import json
import shortuuid
import ffmpeg
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import logging
from werkzeug.utils import secure_filename
# from utils.minio_uploader import MinioFileUploader

load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

# 添加文件日志处理器
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
app.logger.addHandler(file_handler)

# 添加控制台日志处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
app.logger.addHandler(console_handler)


def get_uuid():
    # 生成一个随机的短 UUID
    unique_id = shortuuid.uuid()
    return str.lower(unique_id)


# 将在线视频地址下载到本地
# 如：http://10.66.12.37:30946/perception-mining/ad_09d93c10-e39d-43fd-bda2-7e797dce9220.mp4
def download_video(video_url, save_dir='videos'):
    """
    Download video from URL to local directory
    Args:
        video_url: URL of the video 
        save_dir: Directory to save the video, defaults to 'videos'
    Returns:
        str: Path to downloaded video file
    Raises:
        FFmpegError: If ffmpeg download fails
        OSError: If file operations fail
        Exception: For other errors
    """
    temp_path = None
    try:
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Generate temp filename
        temp_filename = f"temp_{get_uuid()}.mp4"
        temp_path = os.path.join(save_dir, temp_filename)
        
        # Generate final filename
        final_filename = f"video_{get_uuid()}.mp4"
        final_path = os.path.join(save_dir, final_filename)

        app.logger.debug(f"Downloading video from {video_url} to {temp_path}")

        # Download to temp file first
        stream = ffmpeg.input(video_url)
        stream = ffmpeg.output(stream, temp_path, acodec='copy', vcodec='copy')
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

        # Verify file exists and has size > 0
        if not os.path.exists(temp_path):
            raise OSError("Download failed - output file not found")
        if os.path.getsize(temp_path) == 0:
            raise OSError("Download failed - output file is empty")
            
        # Rename temp file to final name
        os.rename(temp_path, final_path)
        app.logger.debug(f"Video downloaded successfully to {final_path}")
        
        return final_path

    except Exception as e:
        app.logger.error(f"Error downloading video: {str(e)}")
        # Clean up temp file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                app.logger.error(f"Error cleaning up temp file: {str(cleanup_error)}")
        raise




def upload_to_gemini(file_path, mime_type=None):
    """Uploads the given file to Gemini."""
    app.logger.debug(f"Uploading file...")
    file = genai.upload_file(path=file_path, mime_type=mime_type, name=get_uuid())
    app.logger.debug(f"Completed upload: {file.uri}\n")
    return file


def wait_for_files_active(video_file):
    """Waits for the given files to be active."""
    app.logger.debug("Waiting for file processing...")
    while video_file.state.name == "PROCESSING":
        app.logger.debug('.')
        time.sleep(1)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name != "ACTIVE":
        raise Exception(f"File {video_file.name} failed to process")

    app.logger.debug("...video_file ready\n")
    print()


def time_to_standard_format(time_range_str):
    """将 '0:13-0:14' 或 '1:23:45-1:23:46' 格式的时间范围统一转换为 '1:23:45-1:23:46' 格式"""

    # 分割时间范围字符串
    start_time_str, end_time_str = time_range_str.split('-')

    # 将开始时间和结束时间分别转换为秒
    start_seconds = time_to_seconds(start_time_str)
    end_seconds = time_to_seconds(end_time_str)

    # 将秒转换为 '1:23:45' 格式
    start_time_formatted = seconds_to_time_format(start_seconds)
    end_time_formatted = seconds_to_time_format(end_seconds)

    # 返回格式化后的时间范围
    # return f"{start_time_formatted}-{end_time_formatted}"
    return start_time_formatted, end_time_formatted


# 辅助函数：将秒转换为 '1:23:45' 格式
def seconds_to_time_format(total_seconds):
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02}:{seconds:02}"


def time_to_seconds(time_str):
    """将 '0:13' 或 '1:23:45' 格式的时间转换为秒"""
    parts = list(map(int, time_str.split(':')))

    if len(parts) == 2:
        # 格式为 '0:13'
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        # 格式为 '1:23:45'
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("时间格式不正确，应为 '0:13' 或 '1:23:45'")


def get_thumbnail(video_path, thumbnail_path, time_seconds):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    (
        ffmpeg
        .input(video_path, ss=time_seconds)  # ss参数指定时间点
        .output(thumbnail_path, vframes=1)  # 只输出一帧
        .overwrite_output()  # 使用overwrite_output方法来覆盖输出文件
        .run()
    )


def upload_thumbnail_to_oss(object_name, file_path):
    # object_name = "b7ec1001240181ceb5ec3e448c7f9b78.mp4_t_11.jpg"
    # file_path = r"E:\playground\ai\projects\gemini-vision-perception\b7ec1001240181ceb5ec3e448c7f9b78.mp4_t_10.jpg"

    # 创建 MinioFileUploader 实例
    uploader = MinioFileUploader()
    return uploader.upload_file(object_name, file_path)


def format_mining_result(mining_result, video_file):
    video_file_path = os.path.join('/tmp', video_file.display_name)

    mining_result_new = []
    for item in mining_result:
        if item['behaviour']['behaviourId'] is None or item['behaviour']['behaviourName'] is None or item['behaviour'][
            'timeRange'] is None:
            continue

        # time_range_str = time_to_standard_format(item['behaviour']['timeRange'])
        start_time_formatted, end_time_formatted = time_to_standard_format(item['behaviour']['timeRange'])
        time_range_str = f"{start_time_formatted}-{end_time_formatted}"
        item['behaviour']['timeRange'] = time_range_str

        start_time = time_to_seconds(start_time_formatted)

        thumbnail_file_name = video_file.display_name + "_t_" + str(start_time) + ".jpg"

        thumbnail_path = os.path.join('/tmp', thumbnail_file_name)
        get_thumbnail(video_file_path, thumbnail_path, start_time)

        thumbnail_oss_url = upload_thumbnail_to_oss(thumbnail_file_name, thumbnail_path)
        print(thumbnail_oss_url)

        item['thumbnail_url'] = thumbnail_oss_url
        mining_result_new.append(item)
    return mining_result_new


def main(video_file_name):
    # video_file = upload_to_gemini(video_file_path)
    # wait_for_files_active(video_file)

    video_file = genai.get_file(video_file_name)

    system_instruction = """
    你是一名聪明、敏感且经验丰富的驾驶助理，负责分析驾驶场景视频。
    我将会向你提供常见的驾驶客观因素，你的任务是观察视频中是否出现这些驾驶客观因素。
    你的分析结果非常重要，因为它将会影响主车驾驶员的驾驶决策。
    """

    prompt = """
以下是常见的驾驶客观因素：
车辆行为
    B1: 车辆急刹： 行驶道路上车辆突然刹车。
    B2: 车辆逆行： 行驶道路上车辆沿着道路方向逆向行驶。
    B3: 车辆变道： 行驶道路上车辆变更车道。
    B4: 连续变道： 行驶道路上车辆进行变道，连续变更多个车道。
    B5: 车辆压线： 行驶道路上车辆行驶中持续大于2秒以上压线行驶。
    B6: 实线变道： 行驶道路上车辆跨越实线进行变道。
    B7: 车辆碰撞： 行驶道路上车辆发生碰撞。
    B8: 未开车灯： 夜间行驶车辆未开车灯。
    B9: 未打信号灯： 行驶道路上车辆转弯或变道未开启信号灯。

其他交通参与者行为
    B10: 非机动车乱窜： 行驶道路上有非机动车在横穿行驶。
    B11: 行人横穿： 行驶道路上有行人横穿马路。
    B12: 行人闯红灯： 行驶道路上行人闯红灯过马路。

道路环境
    B13: 自行车： 行驶道路上发现静止的自行车。

行驶环境
    B14: 高速路： 车辆行驶在高速路上。
    B15: 雨天： 车辆行驶中天空中在下雨。
    B16: 夜间： 车辆处于夜间行驶。

仔细观察视频中的内容，分析上述的驾驶客观因素是否在视频中出现，并使用分配的 ID 进行识别。

以如下JSON的格式输出：
[
  {
    "analysis": "对视频场景的详细分析...",  # 这里翻译成中文输出
    "behaviour": {
      "behaviourId": "B1",
      "behaviourName": "车辆急刹",
      "timeRange": "00:00:11-00:00:12" #  客观因素发生的时间范围
    }
  },
  {
    "analysis": "对视频场景的详细分析...",
    "behaviour": {
      "behaviourId": "B14",
      "behaviourName": "高速路",
      "timeRange": "00:00:11-00:00:12" #  客观因素发生的时间范围
    }
  }
  ...
] 
    """

    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    generation_config = genai.GenerationConfig(**generation_config)

    model = genai.GenerativeModel(model_name=os.getenv('VISION_MODEL'), generation_config=generation_config,
                                  system_instruction=system_instruction)

    app.logger.debug("Making LLM inference request...")
    response = model.generate_content([video_file, prompt],
                                      request_options={"timeout": 600})

    app.logger.debug(response.text)

    mining_result = json.loads(response.text)

    return format_mining_result(mining_result, video_file)


@app.route('/vision-analyze/video/mining', methods=['POST'])
def mining_video():
    # if 'video' not in request.files:
    #     return jsonify({"error": "No video file provided"}), 400
    #
    # video_file = request.files['video']
    # filename = secure_filename(video_file.filename)
    # video_file_path = os.path.join('/tmp', filename)
    # video_file.save(video_file_path)

    file_name = request.form.get('file_name')

    try:
        result = main(file_name)

        response = {
            "msg": "success",
            "code": 0,
            "data": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    video_url = 'http://10.66.12.37:30946/perception-mining/ad_09d93c10-e39d-43fd-bda2-7e797dce9220.mp4'
    final_path = download_video(video_url)
    video_file = upload_to_gemini(final_path)
    print(video_file)
