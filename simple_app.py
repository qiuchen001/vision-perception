from flask import Flask, send_from_directory, redirect, url_for
import os

# 获取当前文件所在目录的绝对路径
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# 确保static目录存在
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app = Flask(__name__)

@app.route('/')
def index():
    return redirect(url_for('upload'))

@app.route('/upload')
def upload():
    """返回上传页面"""
    try:
        index_path = os.path.join(STATIC_DIR, 'index.html')
        print(f"Trying to serve index.html from: {index_path}")
        if not os.path.exists(index_path):
            print("Warning: index.html not found!")
            return "Error: index.html not found", 404
        return send_from_directory(STATIC_DIR, 'index.html')
    except Exception as e:
        print(f"Error serving index.html: {str(e)}")
        return str(e), 500

if __name__ == '__main__':
    print(f"Static Directory: {STATIC_DIR}")
    app.run(host='0.0.0.0', port=5000, debug=True) 