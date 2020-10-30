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
    packerlist = json_data['订单列表']
    binlist = json_data['货箱列表']

    for jsbin in binlist:        # 遍历货箱列表
        bin_attr = jsbin['货箱属性']
        bin_num = jsbin['可用数量']
        bin = Bin(bin_attr['货箱型号'], bin_attr['货箱类型'], eval(bin_attr['重量']), eval(bin_attr['长']), eval(bin_attr['宽']),
                  eval(bin_attr['高']))
        for i in range(eval(bin_num)):
            Task.Bins.append(bin)

    for jspacker in packerlist:          # 遍历订单
        packer_name = jspacker['订单序号']
        packer_id = jspacker['订单ID']
        jsitems = jspacker['货物集合']

        packer = Packer(packer_name, packer_id)

        for jsitem in jsitems:
            load_limits = jsitem['装箱限制']
            dirct_limit = []
            load_or_not = [0, 0, 0, 0, 0, 0]
            load_limit = [20, 20, 20, 20, 20, 20]
            stack_limit = [0, 0, 0, 0, 0, 0]
            for limit in load_limits:
                if limit['摆放方向'] == '立放正向':
                    dirct_limit.append(0)
                    load_limit[0] = eval(limit['承重级别'])
                    if limit['是否承重面'] == '是':
                        load_or_not[0] = 1
                    if limit['堆码限制'] == '是':
                        stack_limit[0] = eval(limit['堆码层数'])
                elif limit['摆放方向'] == '立放横向':
                    dirct_limit.append(1)
                    load_limit[1] = eval(limit['承重级别'])
                    if limit['是否承重面'] == '是':
                        load_or_not[1] = 1
                    if limit['堆码限制'] == '是':
                        stack_limit[1] = eval(limit['堆码层数'])
                elif limit['摆放方向'] == '侧放正向':
                    dirct_limit.append(3)
                    load_limit[3] = eval(limit['承重级别'])
                    if limit['是否承重面'] == '是':
                        load_or_not[3] = 1
                    if limit['堆码限制'] == '是':
                        stack_limit[3] = eval(limit['堆码层数'])
                elif limit['摆放方向'] == '侧放横向':
                    dirct_limit.append(2)
                    load_limit[2] = eval(limit['承重级别'])
                    if limit['是否承重面'] == '是':
                        load_or_not[2] = 1
                    if limit['堆码限制'] == '是':
                        stack_limit[2] = eval(limit['堆码层数'])
                elif limit['摆放方向'] == '卧放正向':
                    dirct_limit.append(5)
                    load_limit[5] = eval(limit['承重级别'])
                    if limit['是否承重面'] == '是':
                        load_or_not[5] = 1
                    if limit['堆码限制'] == '是':
                        stack_limit[5] = eval(limit['堆码层数'])
                elif limit['摆放方向'] == '卧放正向':
                    dirct_limit.append(4)
                    load_limit[4] = eval(limit['承重级别'])
                    if limit['是否承重面'] == '是':
                        load_or_not[4] = 1
                    if limit['堆码限制'] == '是':
                        stack_limit[4] = eval(limit['堆码层数'])

            item = Item(packer_name, jsitem['产品名'], jsitem['套机编码'], jsitem['货物编码'], jsitem['货物型号'], jsitem['重量'],
                        jsitem['体积'],
                        jsitem['长'], jsitem['宽'], jsitem['高'], dirct_limit, load_limit, stack_limit, load_or_not)

            packer.items.append(item)

        Task.Packers.append(packer)
        del packer


