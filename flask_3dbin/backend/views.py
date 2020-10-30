#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
# http://10.103.241.91:10101/account/return_json
from flask import Blueprint, jsonify, app, request
# from .reactt import*
import numpy as np
import matplotlib.pyplot as plt
from .all_test import*
from flask_cors import *



    # 账户的蓝图  访问http://host:port/account 这个链接的子链接，都会跳到这里
account = Blueprint('/account', __name__)
CORS(account, supports_credentials=True)


    # 访问http://host:port/account/return_json 这个链接，就会跳到这里

@account.route('/post_json', methods=['POST'])
# @cross_origin()
    # 上面的链接，绑定的就是这个方法，我们给浏览器或者接口请求 一个json格式的返回
    #http://127.0.0.1:10101/account/post_json
    # http://10.103.241.91/account/return_json

def post_json():
    if request.method == 'POST':
        request_data = request.get_data().decode('utf-8')
        # 解析json前进行编码.不然出来的结果并不是中文的
        all_run(json.loads(request_data))

        return 'post_json'
    return 'post_json'

@account.route('/return_json')
# @cross_origin()
def return_json():
    path = "./data/output.json"
    with open(path, "r", encoding="utf-8") as f:
        load_dict = json.load(f)
        # print(load_dict)
    # return jsonify(load_dict)
    return jsonify({'code': 0, 'content': 'hi flask'})

