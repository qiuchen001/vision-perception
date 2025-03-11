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

def process_by_raw_id(raw_id: str) -> str:
    """处理raw_id"""
    if not raw_id or not raw_id.strip():
        return "请输入有效的raw_id"
    
    try:
        with app.app_context():
            upload_service = UploadVideoService()
            result = upload_service.process_by_raw_id(raw_id.strip())
            
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
            
    except Exception as e:
        return f"处理失败: {str(e)}"

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

def process_input(choice: str, video_file: str = None, raw_id: str = None) -> str:
    """根据用户选择处理输入"""
    if choice == "上传视频":
        if video_file is None:
            return "请选择要上传的视频文件"
        return process_video(video_file)
    else:  # raw_id处理
        if not raw_id:
            return "请输入raw_id"
        return process_by_raw_id(raw_id)

def update_visible(choice: str):
    """更新组件可见性"""
    if choice == "上传视频":
        return (
            gr.update(visible=True, value=None),  # 重置File组件的值并显示
            gr.update(visible=False, value="")    # 清空并隐藏Textbox
        )
    else:
        return (
            gr.update(visible=False, value=None), # 重置File组件的值并隐藏
            gr.update(visible=True, value="")     # 清空并显示Textbox
        )

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="视频上传处理系统") as iface:
        gr.Markdown("# 视频上传处理系统")
        gr.Markdown("支持上传本地视频或通过raw_id处理数据")
        
        with gr.Row():
            choice = gr.Radio(
                choices=["上传视频", "输入raw_id"],
                value="上传视频",
                label="处理方式"
            )
        
        with gr.Row():
            # 视频上传组件
            video_file = gr.File(
                label="上传视频",
                type="binary",
                file_types=[".mp4", ".avi", ".mov", ".mkv"],
                file_count="single",
                visible=True
            )
            
            # raw_id输入组件
            raw_id = gr.Textbox(
                label="输入raw_id",
                placeholder="请输入raw_id",
                visible=False
            )
        
        # 处理结果显示
        output = gr.Textbox(label="处理结果")
        
        # 提交按钮
        submit_btn = gr.Button("开始处理")
        
        # 根据选择显示/隐藏组件
        choice.change(
            fn=update_visible,
            inputs=[choice],
            outputs=[video_file, raw_id]
        )
        
        # 处理提交
        submit_btn.click(
            fn=process_input,
            inputs=[choice, video_file, raw_id],
            outputs=[output]
        )
        
    return iface

# 启动服务
if __name__ == "__main__":
    iface = create_interface()
    iface.launch(server_name="0.0.0.0", server_port=7860)
