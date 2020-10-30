#!/usr/bin/python
# -*- coding: UTF-8 -*-

from backend.views import account

# 蓝图注册
def register(app):
    app.register_blueprint(account, url_prefix='/account', strict_slashes=False)