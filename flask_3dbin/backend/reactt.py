from .py3dbp.constants import Task
from .py3dbp import Bin, Item, Packer
from .py3dbp import bin_items_show
import time
import json
import decimal


start = time.perf_counter()

class DecimalEncoder(json.JSONEncoder):
    # python dict类型转换为json字符串时，需要把decimal类型转换成float类型
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        super(DecimalEncoder, self).default(o)

def react_post_json(json_data):
# with open('test_json(2)/test_json/5output_test.json', 'r', encoding='GBK') as fp:
#     json_data = json.load(fp)

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


    num_bin = len(Task.Bins)

    for packer in Task.Packers:
        for i in range(num_bin):
            if packer.pack(i+1,False,False) == 0:
                break


    # 结果输出
    i = 1   # 装箱步骤计数
    output = {
                '装箱步骤': []
            }
    output_3d = {
                '箱子': []
            }
    for bin in Task.Used_bins:
        hua = []
        totalvol = bin.get_total_vol()
        print('货箱利用率')
        print(totalvol / bin.get_volume())
        bin.items.sort(key=lambda x: (x.position[0], x.position[1], x.position[2]))
        itemsinfo = []
        for item in bin.items:
            item_info = []
            info = {
                '装箱步骤': '步骤'+str(i),
                '订单序号': item.packer_name,
                '货箱序号': bin.name,
                '货物坐标': item.position
            }
            output['装箱步骤'].append(info)
            i += 1

            pos = (float(item.position[0]), float(item.position[1]), float(item.position[2]))
            item_info.append(pos)
            dimension = item.get_dimension()
            item_info.append(float(dimension[0]))
            item_info.append(float(dimension[1]))
            item_info.append(float(dimension[2]))

            item_info = tuple(item_info)
            itemsinfo.append(item_info)
        info_3d = {
            '货箱长': bin.width,
            '货箱宽': bin.depth,
            '货箱高': bin.height,
            '货箱货物': itemsinfo
        }
        output_3d['箱子'].append(info_3d)


        # bin_items_show.item_show(bin.width, bin.depth, bin.height, itemsinfo)


    jsondata = json.dumps(output, cls=DecimalEncoder, ensure_ascii=False)
    with open('output1.json', 'w', encoding='utf-8') as fp:
        fp.write(jsondata)
        fp.close()


    jsondata3d = json.dumps(output_3d, cls=DecimalEncoder, ensure_ascii=False)
    with open('output_3d.json', 'w', encoding='utf-8') as fp:
        fp.write(jsondata3d)
        fp.close()

    end = time.perf_counter()
    print('程序运行时间：%s' % (end-start))


    # return hua







