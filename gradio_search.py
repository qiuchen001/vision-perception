import logging

import gradio as gr
from app.services.video.search import SearchVideoService
from app.services.video.integrated_search import IntegratedSearchService
from app import create_app
from PIL import Image
import json

# 创建Flask应用实例
config_name = "config.config.Config"
app = create_app(config_name)


def format_video_info(video):
    """格式化视频信息为HTML"""
    video_url = video.get('path', '')
    title = video.get('title', '未知')

    info = "<div style='color: #666; line-height: 1.6;'>"
    
    # 添加标签信息
    if 'tags' in video and video['tags']:
        tags_html = ', '.join([f'<span style="background: #f0f0f0; padding: 2px 8px; border-radius: 12px; margin-right: 8px;">{tag}</span>' for tag in video['tags']])
        info += f"""
        <div style="margin-bottom: 12px;">
            <strong style="color: #444;">标签:</strong><br>
            <div style="margin-top: 6px;">{tags_html}</div>
        </div>
        """

    # 添加时间戳信息
    if 'timestamp' in video:
        info += f"""
        <div style="margin-bottom: 12px;">
            <strong style="color: #444;">时间戳:</strong> {video['timestamp']}秒
        </div>
        """

    # 添加摘要信息
    if 'summary_txt' in video and video['summary_txt']:
        info += f"""
        <div style="margin-bottom: 12px;">
            <strong style="color: #444;">摘要:</strong><br>
            <div style="margin-top: 6px; padding: 10px; background: #f9f9f9; border-radius: 4px;">
                {video['summary_txt']}
            </div>
        </div>
        """

    info += "</div>"
    return info, video_url, title


def create_video_player_html(video_url, title, video_info):
    """创建视频播放器和信息展示的HTML"""
    try:
        # 确保所有输入都是字符串类型
        video_url = str(video_url) if video_url else ''
        title = str(title) if title else ''
        video_info = str(video_info) if video_info else ''
        
        # 处理HTML转义字符
        video_info = (video_info
            .replace('\\n', '\n')
            .replace("\\'", "'")
            .replace('&quot;', '"')
            .replace('&amp;', '&')
            .replace('&lt;', '<')
            .replace('&gt;', '>')
        )
        
        # 检查视频URL是否有效
        if not video_url.startswith('http'):
            raise ValueError("Invalid video URL")
            
        # 通过代理服务访问视频
        proxy_url = f"http://127.0.0.1:30501/vision-analyze/video/proxy/{video_url}"
            
        print(f"Original URL: {video_url}")
        print(f"Proxy URL: {proxy_url}")
            
        # 构建HTML内容
        html = f"""
        <div class="video-container" style="height: 100%; background: #f5f5f5; border-radius: 8px;">
            <!-- 视频播放器区域 -->
            <div style="padding: 20px 20px 0 20px;">
                <div style="position: relative; width: 100%; padding-top: 56.25%; background: black; border-radius: 8px; overflow: hidden;">
                    <video 
                        id="video-player"
                        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" 
                        controls
                        autoplay
                        preload="auto"
                        controlsList="nodownload"
                    >
                        <source src="{proxy_url}" type="video/mp4">
                        <p style="color: white; text-align: center;">您的浏览器不支持视频播放</p>
                    </video>
                </div>
            </div>

            <!-- 视频信息区域 -->
            <div style="padding: 20px;">
                <div class="video-info" style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="margin-top: 0; margin-bottom: 15px; color: #333; font-size: 18px;">{title}</h3>
                    <div style="max-height: 300px; overflow-y: auto;">
                        {video_info}
                    </div>
                </div>
            </div>
        </div>
        """

        html = f"""
        <div class="video-container" style="height: 100%; background: #f5f5f5; border-radius: 8px;">
            <!-- 视频播放器区域 -->
            <div style="padding: 20px 20px 0 20px;">
                <div style="position: relative; width: 100%; padding-top: 56.25%; background: black; border-radius: 8px; overflow: hidden;">
                    <video 
                        id="video-player"
                        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" 
                        controls
                        autoplay
                        preload="auto"
                        controlsList="nodownload"
                    >
                        <source src="{proxy_url}" type="video/mp4">
                        <p style="color: white; text-align: center;">您的浏览器不支持视频播放</p>
                    </video>
                </div>
            </div>
            
            <!-- 视频信息区域 -->
            <div style="padding: 20px;">
                <h3 style="margin-top: 0; margin-bottom: 15px; color: #333; font-size: 18px;">{title}</h3>
                <div style="max-height: 300px; overflow-y: auto;">
                    {video_info}
                </div>
            </div>
        </div>
        """
        
        print(f"Video URL: {video_url}")
        print(f"Proxy URL: {proxy_url}")
        print(f"Generated HTML length: {len(html)}")
        return html
        
    except Exception as e:
        print(f"Error in create_video_player_html: {e}")
        return create_error_html(f"视频播放器创建失败: {str(e)}")


def format_gallery_html(gallery_data):
    """创建自定义网格布局的HTML"""
    html = """
    <style>
    .custom-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 20px;
        padding: 20px;
        width: 100%;
        box-sizing: border-box;
    }
    .video-card {
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        overflow: hidden;
        cursor: pointer;
        transition: transform 0.2s;
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .video-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .video-thumbnail {
        width: 100%;
        aspect-ratio: 16/9;
        object-fit: cover;
    }
    .video-info {
        padding: 15px;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    .video-title {
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 8px;
        color: #333;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .video-tags {
        color: #666;
        font-size: 14px;
        margin-top: auto;
    }
    </style>

    <div class="custom-gallery">
    """
    
    # 添加视频卡片
    for idx, (thumbnail_url, title, tags) in enumerate(gallery_data):
        html += f"""
        <div class="video-card">
            <img class="video-thumbnail" src="{thumbnail_url}" alt="{title}">
            <div class="video-info">
                <div class="video-title">{title}</div>
                <div class="video-tags">{tags}</div>
            </div>
        </div>
        """
    
    html += "</div>"
    return html


def search_videos(
        search_type,
        text_query="",
        image_file=None,
        image_url="",
        search_mode="frame",
        page=1,
        page_size=6,
        tags_input=""
):
    """处理视频搜索请求"""
    try:
        with app.app_context():
            search_service = SearchVideoService()
            integrated_service = IntegratedSearchService()

            # 根据搜索类型调用不同的搜索方法
            if search_type == "智能搜索":
                if not text_query.strip():
                    return [], None, "请输入搜索文本"
                results = integrated_service.search(
                    query=text_query,
                    page=page,
                    page_size=page_size
                )
            elif search_type == "标签搜索":
                if not tags_input.strip():
                    return [], None, "请输入搜索标签"
                    
                # 将输入的标签字符串转换为列表
                tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
                if not tags:
                    return [], None, "请输入有效的标签"
                    
                results = search_service.search_by_tags(
                    tags=tags,
                    page=page,
                    page_size=page_size
                )
            elif search_type == "文本搜索":
                if not text_query.strip():
                    return [], None, "请输入搜索文本"
                results = search_service.search_by_text(
                    text_query,
                    page=page,
                    page_size=page_size,
                    search_mode=search_mode
                )
            else:  # 图片搜索
                if image_file is None and not image_url.strip():
                    return [], None, "请上传图片或输入图片URL"
                
                try:
                    if image_url.strip():
                        results = search_service.search_by_image(
                            image_file=None,
                            image_url=image_url,
                            page=page,
                            page_size=page_size
                        )
                    elif isinstance(image_file, Image.Image):
                        results = search_service.search_by_image(
                            image_file=image_file,
                            image_url=None,
                            page=page,
                            page_size=page_size
                        )
                    else:
                        return [], None, f"不支持的图片格式: {type(image_file)}"
                except Exception as e:
                    return [], None, f"图片处理失败: {str(e)}"

            # 格式化输出结果
            if not results:
                return [], None, "未找到匹配的视频"

            # 准备Gallery数据
            gallery_data = []
            video_data = {}

            for idx, video in enumerate(results):
                thumbnail_url = video.get('thumbnail_path', '')
                if not thumbnail_url:
                    thumbnail_url = "path/to/default/thumbnail.jpg"

                video_data[str(idx)] = {
                    'url': video.get('path', ''),
                    'title': video.get('title', '未知'),
                    'info': format_video_info(video)[0],
                    'raw_data': video
                }

                gallery_data.append((
                    thumbnail_url,
                    video.get('title', '未知')
                ))

            global _video_data
            _video_data = video_data

            return gallery_data, gr.HTML(value=""), "搜索完成"

    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        return [], gr.HTML(value=""), f"搜索失败: {str(e)}"


def on_select(evt: gr.SelectData):
    """处理视选择事件"""
    try:
        global _video_data
        if hasattr(evt, 'index'):
            video_info = _video_data.get(str(evt.index))
            if video_info:
                html_content = create_video_player_html(
                    video_info['url'],
                    video_info['title'],
                    video_info['info']  # 传递完整的视频信息
                )
                return gr.HTML(value=html_content)

        return gr.HTML(value="<p>无法播放视频</p>")
    except Exception as e:
        print(f"Error in on_select: {e}")
        return gr.HTML(value=f"<p>播放错误: {str(e)}</p>")


def update_input_visibility(search_type):
    """更新输入组件的可见性"""
    if search_type == "智能搜索":
        return (
            gr.update(visible=True),   # 文本输入
            gr.update(visible=False),  # 搜索模式
            gr.update(visible=False),  # 图片上传
            gr.update(visible=False),  # 图片URL
            gr.update(visible=False)   # 标签输入
        )
    elif search_type == "文本搜索":
        return (
            gr.update(visible=True),   # 文本输入
            gr.update(visible=True),   # 搜索模式
            gr.update(visible=False),  # 图片上传
            gr.update(visible=False),  # 图片URL
            gr.update(visible=False)   # 标签输入
        )
    elif search_type == "图片搜索":
        return (
            gr.update(visible=False),  # 文本输入
            gr.update(visible=False),  # 搜索模式
            gr.update(visible=True),   # 图片上传
            gr.update(visible=True),   # 图片URL
            gr.update(visible=False)   # 标签输入
        )
    else:  # 标签搜索
        return (
            gr.update(visible=False),  # 文本输入
            gr.update(visible=False),  # 搜索模式
            gr.update(visible=False),  # 图片上传
            gr.update(visible=False),  # 图片URL
            gr.update(visible=True)    # 标签输入
        )


def handle_video_select(index):
    """处理视频选择"""
    try:
        print(f"Handling video select for index: {index}")  # 调试日志
        global _video_data
        if index is not None:
            video_info = _video_data.get(str(index))
            if video_info:
                print(f"Found video info: {video_info}")  # 调试日志
                html_content = create_video_player_html(
                    video_info['url'],
                    video_info['title'],
                    video_info['info']  # 传递完整的视频信息
                )
                return html_content  # 直接返回HTML字符串
        return "<p>无法播放视频</p>"
    except Exception as e:
        print(f"Error in handle_video_select: {e}")  # 调试日志
        return f"<p>播放错误: {str(e)}</p>"


# 创建Gradio界面
def create_interface():
    css = """
    #container {
        min-height: calc(100vh - 100px);
        padding: 1rem;
    }
    .search-controls {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .results-container {
        display: flex;
        gap: 2rem;
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        width: 100%;
        box-sizing: border-box;
    }
    .gallery-area {
        flex: 2;  /* 增加gallery区域的比例 */
        min-width: 0;
        width: 100%;
    }
    .video-area {
        flex: 1;
        min-width: 300px;
        position: sticky;
        top: 1rem;
        max-height: calc(100vh - 120px);
        overflow-y: auto;
    }
    .status-box {
        margin-top: 1rem;
        padding: 0.75rem;
        border-radius: 6px;
        background: #f8f9fa;
        border: 1px solid #dee2e6;
    }
    .search-button {
        background: #007bff;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: background 0.3s;
    }
    .search-button:hover {
        background: #0056b3;
    }
    /* Gallery式布局 */
    .gallery-grid {
        display: grid !important;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)) !important;
        gap: 1.5rem !important;
        width: 100% !important;
        padding: 1rem !important;
        box-sizing: border-box !important;
    }
    
    .gallery-grid > div {
        display: flex !important;
        flex-direction: column !important;
        border-radius: 8px !important;
        background: white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        height: 100% !important;
        overflow: hidden !important;
        transition: transform 0.2s !important;
    }
    
    .gallery-grid > div:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    
    .gallery-grid > div > img {
        width: 100% !important;
        aspect-ratio: 16/9 !important;
        object-fit: cover !important;
        border-radius: 8px 8px 0 0 !important;
        margin-bottom: 0 !important;
    }
    
    .gallery-grid > div > .caption {
        padding: 1rem !important;
        background: white !important;
        border-radius: 0 0 8px 8px !important;
        border-top: 1px solid #eee !important;
        flex-grow: 1 !important;
    }
    
    .gallery-grid .title {
        font-weight: bold !important;
        font-size: 1rem !important;
        color: #333 !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.4 !important;
        display: -webkit-box !important;
        -webkit-line-clamp: 2 !important;
        -webkit-box-orient: vertical !important;
        overflow: hidden !important;
    }
    
    .gallery-grid .tags {
        color: #666 !important;
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
    }
    """

    with gr.Blocks(title="视频搜索系统", css=css) as iface:
        gr.Markdown("# 视频搜索系统")

        # 首先创建隐藏的索引输入框
        selected_index = gr.State(value=None)  # 改用 gr.State 来存储选中的索引

        # 搜索控制区域
        with gr.Column(elem_classes="search-controls"):
            gr.Markdown("## 搜索条件")
            with gr.Row():
                search_type = gr.Radio(
                    choices=["智能搜索", "文本搜索", "图片搜索", "标签搜索"],
                    label="搜索类型",
                    value="智能搜索",
                    container=False
                )

            with gr.Row():
                # 文本搜索相关组件
                text_query = gr.Textbox(
                    label="搜索文本",
                    placeholder="请输入搜索关键词",
                    lines=2,
                    visible=True,
                    container=True
                )
                search_mode = gr.Radio(
                    choices=["frame", "summary"],
                    label="搜索模式",
                    value="frame",
                    visible=True,
                    info="frame: 搜索视频帧 | summary: 搜索视频摘要",
                    container=True
                )

            # 图片搜索相关组件
            with gr.Row() as image_search_row:
                image_file = gr.Image(
                    label="上传图片",
                    type="pil",
                    container=True,
                    visible=False
                )
                image_url = gr.Textbox(
                    label="图片URL",
                    placeholder="请输入图片URL",
                    container=True,
                    visible=False
                )
                
            # 标签搜索相关组件
            with gr.Row() as tag_search_row:
                tags_input = gr.Textbox(
                    label="标签搜索",
                    placeholder="请输入标签，多个标签用逗号分隔",
                    lines=1,
                    visible=False,
                    container=True
                )

            with gr.Row():
                with gr.Column(scale=1):
                    page = gr.Number(
                        label="页码",
                        value=1,
                        minimum=1,
                        step=1
                    )
                with gr.Column(scale=1):
                    page_size = gr.Number(
                        label="每页数量",
                        value=6,
                        minimum=1,
                        maximum=20,
                        step=1
                    )

            with gr.Row():
                search_button = gr.Button("搜索", elem_classes="search-button")

        # 搜索结果和视频播放区域
        with gr.Row(elem_classes="results-container"):
            # 左侧搜索结果
            with gr.Column(scale=1, elem_classes="gallery-area"):
                gr.Markdown("## 搜索结果")
                gallery = gr.Gallery(
                    label="搜索结果",
                    show_label=False,
                    elem_id="gallery",
                    columns=2,
                    object_fit="contain",
                    height="auto",
                    allow_preview=False,
                    elem_classes="custom-gallery"
                )
                status = gr.Textbox(label="状态", interactive=False)

            # 右侧视频播放区域
            with gr.Column(scale=1, elem_classes="video-area"):
                gr.Markdown("## 视频播放")
                video_area = gr.HTML(
                    value="""
                    <div style="height: 100%; display: flex; align-items: center; justify-content: center; 
                         background: white; border-radius: 10px; padding: 2rem; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
                        <div style="text-align: center;">
                            <svg style="width:64px;height:64px;margin-bottom:1.5rem;color:#6c757d" viewBox="0 0 24 24">
                                <path fill="currentColor" d="M17,10.5V7A1,1 0 0,0 16,6H4A1,1 0 0,0 3,7V17A1,1 0 0,0 4,18H16A1,1 0 0,0 17,17V13.5L21,17.5V6.5L17,10.5Z" />
                            </svg>
                            <div style="color:#6c757d;font-size:1.1rem;">请选择要播放的视频</div>
                        </div>
                    </div>
                    """,
                    elem_classes="video-player"
                )

        # 添加自定义CSS
        css = """
        .results-container {
            gap: 20px;
            padding: 20px;
        }
        .gallery-area, .video-area {
            min-height: 600px;
        }
        .custom-gallery {
            min-height: 300px !important;
            border-radius: 8px !important;
            background: white !important;
            padding: 20px !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
        }
        .custom-gallery img {
            border-radius: 8px !important;
            transition: transform 0.2s !important;
            cursor: pointer !important;
        }
        .custom-gallery img:hover {
            transform: scale(1.02) !important;
        }
        .video-player {
            height: 100% !important;
            min-height: 400px !important;
        }
        .video-info {
            margin-top: 20px;
        }
        """

        # 修改事件处理函数
        def on_gallery_select(evt: gr.SelectData):
            """处理 Gallery 选择事件"""
            try:
                global _video_data
                print(f"Selected index: {evt.index}")
                print(f"Available video data: {list(_video_data.keys())}")
                
                video_info = _video_data.get(str(evt.index))
                if not video_info:
                    print(f"No video info found for index {evt.index}")
                    return create_error_html("无法加载视频信息")
                    
                print(f"Found video info: {video_info}")
                
                # 确保所有必要的字段都存在
                required_fields = ['url', 'title', 'info']
                if not all(key in video_info for key in required_fields):
                    missing_fields = [field for field in required_fields if field not in video_info]
                    print(f"Missing required fields: {missing_fields}")
                    return create_error_html("视频信息不完整")
                    
                # 检查URL是否有效
                if not video_info['url'] or not isinstance(video_info['url'], str):
                    print(f"Invalid video URL: {video_info['url']}")
                    return create_error_html("无效的视频地址")
                    
                # 创建视频播放器HTML
                html_content = create_video_player_html(
                    video_info['url'],
                    video_info['title'],
                    video_info['info']
                )
                
                if not html_content:
                    return create_error_html("视频播放器创建失败")
                    
                return html_content
                
            except Exception as e:
                print(f"Error in on_gallery_select: {e}")
                import traceback
                traceback.print_exc()
                return create_error_html(f"播放错误: {str(e)}")

        # 绑定 Gallery 选择事件
        gallery.select(
            fn=on_gallery_select,
            outputs=video_area,
            show_progress=True
        )

        # 添加图片输入组件互斥处理
        def clear_other_input(value, is_url):
            """当一个输入有值时，清除另一个输入"""
            if is_url and value:  # URL输入有值时
                return gr.update(value=None), gr.update()
            elif not is_url and value is not None:  # 图片上传有值时
                return gr.update(), gr.update(value="")
            return gr.update(), gr.update()

        # 事件绑定
        search_type.change(
            fn=update_input_visibility,
            inputs=[search_type],
            outputs=[text_query, search_mode, image_file, image_url, tags_input]
        )

        # 添加图片输入互斥处理
        image_file.change(
            fn=lambda x: clear_other_input(x, False),
            inputs=[image_file],
            outputs=[image_file, image_url]
        )

        image_url.change(
            fn=lambda x: clear_other_input(x, True),
            inputs=[image_url],
            outputs=[image_file, image_url]
        )

        search_button.click(
            fn=search_videos,
            inputs=[
                search_type,
                text_query,
                image_file,
                image_url,
                search_mode,
                page,
                page_size,
                tags_input  # 添加标签输入
            ],
            outputs=[gallery, video_area, status]
        )

    return iface


# 初始化全局变量
_video_data = {}

# 启动服务
if __name__ == "__main__":
    iface = create_interface()
    iface.launch(server_name="0.0.0.0", server_port=7862)  # 使用7862端口

def create_error_html(message):
    """创建错误提示的HTML"""
    return f"""
    <div style="height: 100%; display: flex; align-items: center; justify-content: center; 
         background: white; border-radius: 10px; padding: 2rem; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
        <div style="text-align: center;">
            <div style="color:#dc3545;font-size:1.1rem;">{message}</div>
        </div>
    </div>
    """