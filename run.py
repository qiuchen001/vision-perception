from app import create_app

app = create_app('config.config.Config')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30501, debug=True)