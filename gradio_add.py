import gradio as gr
from app.services.video.add import AddVideoService
from app import create_app

# 创建Flask应用实例
config_name = "config.config.Config"
app = create_app(config_name)


def process_add(video_url, action_type):
    """处理视频添加请求"""
    if not video_url:
        return "请输入视频URL"

    try:
        with app.app_context():
            # 创建服务实例
            add_service = AddVideoService()

            # 处理视频
            m_id = add_service.add(video_url, int(action_type))

            # 格式化输出结果
            output = f"""
            处理完成!

            视频URL: {video_url}
            处理类型: {get_action_type_desc(int(action_type))}
            视频ID: {m_id}
            """

            return output

    except ValueError as ve:
        return f"处理失败: {str(ve)}"
    except Exception as e:
        return f"处理失败: {str(e)}"


def get_action_type_desc(action_type):
    """获取操作类型描述"""
    types = {
        1: "视频内容挖掘",
        2: "视频内容总结",
        3: "内容挖掘和总结"
    }
    return types.get(action_type, "未知操作")


# 创建Gradio界面
def create_interface():
    iface = gr.Interface(
        fn=process_add,
        inputs=[
            gr.Textbox(
                label="视频URL",
                placeholder="请输入视频URL",
                lines=1
            ),
            gr.Radio(
                choices=["1", "2", "3"],
                label="处理类型",
                value="1",
                info="1:内容挖掘 2:内容总结 3:两者都做"
            )
        ],
        outputs=gr.Textbox(label="处理结果"),
        title="视频处理系统",
        description="""
        请输入要处理的视频URL，并选择处理类型：
        - 内容挖掘：分析视频内容并提取标签
        - 内容总结：生成视频内容的文字总结
        - 两者都做：同时进行内容挖掘和总结
        """
    )
    return iface


# 启动服务
if __name__ == "__main__":
    iface = create_interface()
    iface.launch(server_name="0.0.0.0", server_port=7861)  # 使用不同的端口
