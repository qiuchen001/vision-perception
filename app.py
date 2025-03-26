from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
from flask_cors import CORS
from app.services.video.upload import UploadVideoService
import tempfile
import os
from werkzeug.datastructures import FileStorage

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
        if not os.path.exists(os.path.join(STATIC_DIR, 'index.html')):
            print("Warning: index.html not found!")
            return "Error: index.html not found", 404
        return send_from_directory(STATIC_DIR, 'index.html')
    except Exception as e:
        print(f"Error serving index.html: {str(e)}")
        return str(e), 500

@app.route('/add')
def add():
    """返回添加页面"""
    try:
        print(f"Trying to serve add.html from {STATIC_DIR}")
        if not os.path.exists(os.path.join(STATIC_DIR, 'add.html')):
            print("Warning: add.html not found!")
            return "Error: add.html not found", 404
        return send_from_directory(STATIC_DIR, 'add.html')
    except Exception as e:
        print(f"Error serving add.html: {str(e)}")
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 