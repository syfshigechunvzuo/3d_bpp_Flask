from .py3dbp.constants import Task
from .py3dbp import Bin, Item, Packer
from .py3dbp import bin_items_show

import json
import decimal
# python dict类型转换为json字符串时，需要把decimal类型转换成float类型




class DecimalEncoder(json.JSONEncoder):
    # python dict类型转换为json字符串时，需要把decimal类型转换成float类型
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        super(DecimalEncoder, self).default(o)

def init_data(json_data):

    Task.Used_bins = []
    Task.Bins = []
    Task.Packers = []
    Task.TradeId = ''
    Task.msg = ''
    Task.Not_fit_num = 0
    Task.TradeId = json_data['tradeId']
    print(type(json_data['tradeId']))
    print(Task.TradeId)
    BinPickingOrder = json_data['orderList']
    BinPickingVehicleModel = json_data['vehicleModelList']

    for jsbin in BinPickingVehicleModel:  # 遍历货箱列表
        bin_num = jsbin['qty']
        if bin_num == 0:
            bin_num == 300
        for i in range(bin_num):
            bin = Bin(jsbin['modelCode'], jsbin['weight']*1000, jsbin['length']*0.001, jsbin['width']*0.001,
                      jsbin['height']*0.001)
            Task.Bins.append(bin)

    for jspacker in BinPickingOrder:  # 遍历订单
        packer_name = jspacker['unloadingSequence']
        # 卸货顺序对应原来的1号订单，2号订单
        packer_id = jspacker['orderCode']
        jsitems = jspacker['goodList']

        packer = Packer(packer_name, packer_id)

        for jsitem in jsitems:
            dirct_limit = []
            load_or_not = [0, 0, 0, 0, 0, 0]
            load_limit = [20, 20, 20, 20, 20, 20]
            stack_limit = [0, 0, 0, 0, 0, 0]
            BinPickingRestriction = jsitem['restrictionList']
            for limit in BinPickingRestriction:
                if limit['flag'] == '1':
                    dirct_limit.append(0)
                    load_limit[0] = limit['bearLevel']
                    if limit['isBear']:
                        load_or_not[0] = 1
                    if limit['isStack']:
                        stack_limit[0] = limit['stackLevel']
                    else:
                        stack_limit[0] = 100
                elif limit['flag'] == '2':
                    dirct_limit.append(1)
                    load_limit[1] = limit['bearLevel']
                    if limit['isBear']:
                        load_or_not[1] = 1
                    if limit['isStack']:
                        stack_limit[1] = limit['stackLevel']
                    else:
                        stack_limit[1] = 100
                elif limit['flag'] == '3':
                    dirct_limit.append(3)
                    load_limit[3] = limit['bearLevel']
                    if limit['isBear']:
                        load_or_not[3] = 1
                    if limit['isStack']:
                        stack_limit[3] = limit['stackLevel']
                    else:
                        stack_limit[3] = 100
                elif limit['flag'] == '4':
                    dirct_limit.append(2)
                    load_limit[2] = limit['bearLevel']
                    if limit['isBear']:
                        load_or_not[2] = 1
                    if limit['isStack']:
                        stack_limit[2] = limit['stackLevel']
                    else:
                        stack_limit[2] = 100
                elif limit['flag'] == '5':
                    dirct_limit.append(5)
                    load_limit[5] = limit['bearLevel']
                    if limit['isBear'] :
                        load_or_not[5] = 1
                    if limit['isStack']:
                        stack_limit[5] = limit['stackLevel']
                    else:
                        stack_limit[5] = 100
                elif limit['flag'] == '6':
                    dirct_limit.append(4)
                    load_limit[4] = limit['bearLevel']
                    if limit['isBear']:
                        load_or_not[4] = 1
                    if limit['isStack']:
                        stack_limit[4] = limit['stackLevel']
                    else:
                        stack_limit[4] = 100
                else:
                    Task.msg = '没有摆放方向，原始数据出错'
                    print(Task.msg)
            item_num = jsitem['qty']
            for i in range(item_num):
                vol = jsitem['length']*jsitem['width']*jsitem['height']
                item = Item(packer_name, jsitem['setCode'], jsitem['materialCode'], jsitem['weight'],
                        vol,
                        jsitem['length']*0.001, jsitem['width']*0.001, jsitem['height']*0.001, dirct_limit, load_limit, stack_limit, load_or_not)
                packer.items.append(item)

        Task.Packers.append(packer)
        del packer


