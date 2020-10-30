from flask import Flask
app = Flask(__name__)
@app.route('/')
def index():
    return 'hello world'
if __name__ == '_main_':
    app.run(host="0.0.0.0", port=2333, debug=True)
    # 可以指定运行的主机IP地址，端口，是否开启调试模式
