#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math
import numpy as np
from decimal import Decimal
import decimal
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from backend.py3dbp.constants import Task
from backend.py3dbp import Bin, Item, Packer
from backend.py3dbp import bin_items_show
from tqdm import tqdm
import json


from flask_script import Manager, Server
from backend import create_app

app = create_app()

app.debug = app.config["DEBUG"]
# 获取根目录config.py的配置项
host = app.config["HOST"]
port = app.config["PORT"]

# Init manager object via app object
manager = Manager(app)

# Create a new commands: server
# This command will be run the Flask development_env server
manager.add_command("runserver", Server(host=host, port=port, threaded=True))

app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"  # 指定浏览器渲染的文件类型，和解码格式；


@manager.shell
def make_shell_context():
    """Create a python CLI.

    return: Default import object
    type: `Dict`
    """
    # 确保有导入 Flask app object，否则启动的 CLI 上下文中仍然没有 app 对象
    return dict(app=app)


if __name__ == '__main__':
    manager.run()




