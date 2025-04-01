from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
from flask_cors import CORS
from app.services.video.upload import UploadVideoService
from app.services.video.search import SearchVideoService
from app.services.video.integrated_search import IntegratedSearchService
import tempfile
import os
from werkzeug.datastructures import FileStorage
from PIL import Image
import io

# 获取当前文件所在目录的绝对路径
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# 确保static目录存在
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app = Flask(__name__,
            static_url_path='',
            static_folder='static'
            )
CORS(app)  # 启用CORS支持

# 打印调试信息
print(f"Base Directory: {BASE_DIR}")
print(f"Static Directory: {STATIC_DIR}")


@app.route('/')
def index():
    return redirect(url_for('upload'))


@app.route('/upload')
def upload():
    """返回上传页面"""
    try:
        print(f"Trying to serve index.html from {STATIC_DIR}")
        if not os.path.exists(os.path.join(STATIC_DIR, 'upload.html')):
            print("Warning: index.html not found!")
            return "Error: index.html not found", 404
        return send_from_directory(STATIC_DIR, 'upload.html')
    except Exception as e:
        print(f"Error serving index.html: {str(e)}")
        return str(e), 500


@app.route('/process')
def process():
    """返回添加页面"""
    try:
        print(f"Trying to serve add.html from {STATIC_DIR}")
        if not os.path.exists(os.path.join(STATIC_DIR, 'process.html')):
            print("Warning: add.html not found!")
            return "Error: add.html not found", 404
        return send_from_directory(STATIC_DIR, 'process.html')
    except Exception as e:
        print(f"Error serving add.html: {str(e)}")
        return str(e), 500


@app.route('/search')
def search():
    """返回搜索页面"""
    try:
        print(f"Trying to serve search.html from {STATIC_DIR}")
        if not os.path.exists(os.path.join(STATIC_DIR, 'search.html')):
            print("Warning: search.html not found!")
            return "Error: search.html not found", 404
        return send_from_directory(STATIC_DIR, 'search.html')
    except Exception as e:
        print(f"Error serving search.html: {str(e)}")
        return str(e), 500


# 添加静态文件路由
@app.route('/<path:path>')
def serve_static(path):
    """服务静态文件"""
    print(f"Requested path: {path}")
    return send_from_directory(STATIC_DIR, path)


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """处理视频上传"""
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': '未找到上传文件'
        }), 400

    video_file = request.files['file']
    if not video_file:
        return jsonify({
            'status': 'error',
            'message': '请选择要上传的视频文件'
        }), 400

    try:
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1])
        try:
            video_file.save(temp_file.name)
            temp_file.flush()

            # 创建FileStorage对象
            with open(temp_file.name, 'rb') as f:
                file_storage = FileStorage(
                    stream=f,
                    filename=video_file.filename,
                    content_type=video_file.content_type
                )

                # 处理视频
                upload_service = UploadVideoService()
                result = upload_service.upload(file_storage)

                if not result:
                    return jsonify({
                        'status': 'error',
                        'message': '视频上传服务返回空结果'
                    }), 500

                if not all(key in result for key in ['file_name', 'video_url', 'title']):
                    return jsonify({
                        'status': 'error',
                        'message': f'处理结果格式异常: {result}'
                    }), 500

                return jsonify({
                    'status': 'success',
                    'data': result
                })

        finally:
            # 清理临时文件
            temp_file.close()
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/process', methods=['POST'])
def process_raw_id():
    """处理raw_id"""
    data = request.get_json()
    if not data or 'raw_id' not in data:
        return jsonify({
            'status': 'error',
            'message': '请提供raw_id'
        }), 400

    raw_id = data['raw_id']
    if not raw_id or not raw_id.strip():
        return jsonify({
            'status': 'error',
            'message': '请输入有效的raw_id'
        }), 400

    try:
        # 处理raw_id
        upload_service = UploadVideoService()
        result = upload_service.process_by_raw_id(raw_id.strip())

        if not result:
            return jsonify({
                'status': 'error',
                'message': '视频上传服务返回空结果'
            }), 500

        if not all(key in result for key in ['file_name', 'video_url', 'title']):
            return jsonify({
                'status': 'error',
                'message': f'处理结果格式异常: {result}'
            }), 500

        return jsonify({
            'status': 'success',
            'data': result
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/search', methods=['POST'])
def search_videos():
    """处理视频搜索"""
    try:
        search_type = request.form.get('search_type')
        page = int(request.form.get('page', 1))
        page_size = int(request.form.get('page_size', 6))

        search_service = SearchVideoService()
        integrated_service = IntegratedSearchService()

        if search_type == 'smart':
            text_query = request.form.get('text_query', '').strip()
            if not text_query:
                return jsonify({
                    'status': 'error',
                    'message': '请输入搜索关键词'
                }), 400
            results = integrated_service.search(
                query=text_query,
                page=page,
                page_size=page_size
            )

        elif search_type == 'text':
            text_query = request.form.get('text_query', '').strip()
            search_mode = request.form.get('search_mode', 'frame')
            if not text_query:
                return jsonify({
                    'status': 'error',
                    'message': '请输入搜索关键词'
                }), 400

            try:
                results = search_service.search_by_text(
                    text_query,
                    page=page,
                    page_size=page_size,
                    search_mode=search_mode
                )

                # 如果结果为None，返回空列表
                if results is None:
                    results = []

                # 确保结果可以被JSON序列化
                if results:
                    results = [{
                        'title': str(video.get('title', '未知')),
                        'path': str(video.get('path', '')),
                        'thumbnail_path': str(video.get('thumbnail_path', '')),
                        'tags': list(video.get('tags', [])) if video.get('tags') else [],
                        'summary_txt': str(video.get('summary_txt', '')),
                        'timestamp': video.get('timestamp', 0),
                        'similarity': str(video.get('similarity', '0.0000')),
                    } for video in results]
            except Exception as e:
                print(f"Text search error: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'文本搜索失败: {str(e)}'
                }), 500

        elif search_type == 'image':
            image_file = request.files.get('image_file')
            image_url = request.form.get('image_url', '').strip()

            if not image_file and not image_url:
                return jsonify({
                    'status': 'error',
                    'message': '请上传图片或输入图片URL'
                }), 400

            if image_file:
                # 将文件内容转换为PIL Image对象
                image_data = image_file.read()
                image = Image.open(io.BytesIO(image_data))
            else:
                image = None

            results = search_service.search_by_image(
                image_file=image,
                image_url=image_url,
                page=page,
                page_size=page_size
            )

        elif search_type == 'tags':
            tags_input = request.form.get('tags', '').strip()
            if not tags_input:
                return jsonify({
                    'status': 'error',
                    'message': '请输入搜索标签'
                }), 400

            tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
            if not tags:
                return jsonify({
                    'status': 'error',
                    'message': '请输入有效的标签'
                }), 400

            results = search_service.search_by_tags(
                tags=tags,
                page=page,
                page_size=page_size
            )

        else:
            return jsonify({
                'status': 'error',
                'message': '不支持的搜索类型'
            }), 400

        if not results:
            return jsonify({
                'status': 'success',
                'data': []
            })

        # 格式化返回结果
        formatted_results = []
        for video in results:
            formatted_video = {
                'title': video.get('title', '未知'),
                'video_url': video.get('path', ''),
                'thumbnail_url': video.get('thumbnail_path', ''),
                'tags': video.get('tags', []),
                'summary': video.get('summary_txt', ''),
                'timestamp': video.get('timestamp', 0),
                'similarity': str(video.get('similarity', '0.0000')),
            }
            formatted_results.append(formatted_video)

        return jsonify({
            'status': 'success',
            'data': formatted_results
        })

    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'参数错误: {str(e)}'
        }), 400
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'搜索失败: {str(e)}'
        }), 500


@app.route('/api/add', methods=['POST'])
def add_video():
    """处理视频添加"""
    data = request.get_json()
    if not data or 'video_url' not in data or 'action_type' not in data:
        return jsonify({
            'status': 'error',
            'message': '请提供视频URL和处理类型'
        }), 400

    video_url = data['video_url']
    action_type = data['action_type']

    if not video_url or not video_url.strip():
        return jsonify({
            'status': 'error',
            'message': '请输入有效的视频URL'
        }), 400

    try:
        # 创建服务实例
        from app.services.video.add import AddVideoService
        add_service = AddVideoService()

        # 处理视频
        m_id = add_service.add(video_url, action_type)

        # 获取处理类型描述
        action_type_desc = {
            1: "视频内容挖掘",
            2: "视频内容总结",
            3: "内容挖掘和总结"
        }.get(action_type, "未知操作")

        return jsonify({
            'status': 'success',
            'data': {
                'video_url': video_url,
                'action_type_desc': action_type_desc,
                'm_id': m_id
            }
        })

    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': str(ve)
        }), 400
    except Exception as e:
        print(f"Add video error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'处理失败: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
