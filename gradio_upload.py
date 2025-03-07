import gradio as gr
from app.services.video.upload import UploadVideoService
import tempfile
import os
from app import create_app
from werkzeug.datastructures import FileStorage
import time

# 从环境变量获取配置名称,默认使用development
# config_name = os.getenv('FLASK_CONFIG', 'development')
config_name = "config.config.Config"

# 创建Flask应用实例
app = create_app(config_name)


def process_video(video_file):
    """处理上传的视频文件"""
    if video_file is None:
        return "请选择要上传的视频文件"

    try:
        # 添加调试信息
        debug_info = f"""
        文件对象类型: {type(video_file)}
        文件对象属性: {dir(video_file)}
        是否为字典: {isinstance(video_file, dict)}
        是否有orig_name: {hasattr(video_file, 'orig_name')}
        是否有name: {hasattr(video_file, 'name')}
        """
        print(debug_info)  # 打印调试信息

        # 在Flask应用上下文中执行
        with app.app_context():
            # 从文件对象获取信息
            if isinstance(video_file, dict):
                print("进入dict分支")
                filename = video_file.get('name', 'unknown.mp4')
            elif hasattr(video_file, 'orig_name'):
                print("进入orig_name分支")
                filename = video_file.orig_name
            elif hasattr(video_file, 'name'):
                print("进入name分支")
                if not isinstance(video_file.name, bytes):
                    filename = video_file.name
                else:
                    filename = f"video_{int(time.time())}.mp4"
            else:
                print("进入else分支")
                filename = f"video_{int(time.time())}.mp4"

            print(f"最终使用的文件名: {filename}")  # 打印最终文件名

            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
            try:
                # 如果是二进制数据,直接写入
                if isinstance(video_file, (bytes, bytearray)):
                    print("写入二进制数据")
                    temp_file.write(video_file)
                elif hasattr(video_file, 'read'):
                    print("写入文件对象")
                    temp_file.write(video_file.read())
                else:
                    print("写入文件路径")
                    with open(str(video_file), 'rb') as src:
                        temp_file.write(src.read())
                temp_file.flush()

                # 创建FileStorage对象
                with open(temp_file.name, 'rb') as f:
                    file_storage = FileStorage(
                        stream=f,
                        filename=os.path.basename(filename),
                        content_type='video/mp4'
                    )

                    # 创建服务实例并处理视频
                    upload_service = UploadVideoService()
                    result = upload_service.upload(file_storage)

                    # 检查返回结果
                    if not result:
                        return """
                        处理失败: 视频上传服务返回空结果
                        请检查服务日志获取详细错误信息
                        """
                    
                    # 检查必要的字段
                    if not all(key in result for key in ['file_name', 'video_url', 'title']):
                        return f"""
                        处理结果格式异常: {result}
                        缺少必要的返回字段
                        """

                    # 格式化输出结果
                    output = f"""
                    处理完成!
                    
                    文件名: {result.get('file_name', '未知')}
                    视频URL: {result.get('video_url', '未知')}
                    标题: {result.get('title', '未知')}
                    处理帧数: {result.get('processed_frames', 0)}/{result.get('frame_count', 0)}
                    """

                    return output
            finally:
                # 清理临时文件
                temp_file.close()
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)

    except Exception as e:
        return f"""
        处理失败: {str(e)}
        文件类型: {type(video_file)}
        文件属性: {dir(video_file) if hasattr(video_file, '__dir__') else '无法获取属性'}
        """


# 创建Gradio界面
def create_interface():
    iface = gr.Interface(
        fn=process_video,
        inputs=[
            gr.File(
                label="上传视频",
                type="binary",  # 使用binary类型
                file_types=[".mp4", ".avi", ".mov", ".mkv"],
                file_count="single"
            )
        ],
        outputs=gr.Textbox(label="处理结果"),
        title="视频上传处理系统",
        description="上传视频文件进行处理。支持MP4、AVI、MOV、MKV格式。"
    )
    return iface


# 启动服务
if __name__ == "__main__":
    iface = create_interface()
    iface.launch(server_name="0.0.0.0", server_port=7860)
