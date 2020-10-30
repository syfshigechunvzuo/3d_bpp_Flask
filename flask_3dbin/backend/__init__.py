#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Created by liaoyangyang1 on 2018/8/21 下午2:41.
"""
# http://10.103.241.91:10101/account/post_json
from flask import Flask
from config.config import config
from backend.urls import register
from flask_cors import *
def create_app():

    #初始化项目实例
    app = Flask(__name__)
    # CORS(app, supports_credentials=True)
    app.secret_key = app.config['SECRET_KEY']

    #导入配置项
    app.config.from_object(config)
    # 注册路由
    register(app)

    # 钩子 在请求执行之前
    @app.before_request
    def before_request():
        pass
       # print('hi')



    return app


