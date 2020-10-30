from .constants import RotationType, Axis, Task
from .auxiliary_methods import intersect, set_to_decimal
DEFAULT_NUMBER_OF_DECIMALS = 3
START_POSITION = [0, 0, 0]

class Item:
    def __init__(self, packer_name, name, suite_id, id, type, weight, vol, width, height, depth, limit_dirct=[], limit_load=[], limit_stack=[],load_or_not=[]):
        self.name = name
        self.suite_id = suite_id
        self.id = id
        self.type = type
        self.vol = vol
        self.width = width
        self.height = height
        self.depth = depth
        self.weight = weight
        self.rotation_type = 0
        self.position = START_POSITION
        self.number_of_decimals = DEFAULT_NUMBER_OF_DECIMALS
        self.limit_dirct = limit_dirct  # 方向限制
        self.limit_load = limit_load  # 承重限制
        self.limit_stack = limit_stack  # 堆码限制
        self.packer_name = packer_name  # 记录所属的订单
        self.load_or_not = load_or_not

    def format_numbers(self, number_of_decimals):
        self.width = set_to_decimal(self.width, number_of_decimals)
        self.height = set_to_decimal(self.height, number_of_decimals)
        self.depth = set_to_decimal(self.depth, number_of_decimals)
        self.weight = set_to_decimal(self.weight, number_of_decimals)
        self.number_of_decimals = number_of_decimals

    def string(self):
        return "%s(%sx%sx%s, weight: %s) pos(%s) rt(%s) vol(%s) L_d(%s) L_l(%s)" % (
            self.name, self.width, self.height, self.depth, self.weight,
            self.position, self.rotation_type, self.get_volume(), self.limit_dirct, self.limit_load
        )

    def get_volume(self):
        return set_to_decimal(
            self.width * self.height * self.depth, self.number_of_decimals
        )


    def get_dimension(self):
        if self.rotation_type == RotationType.RT_WHD:
            dimension = [self.width, self.height, self.depth]
        elif self.rotation_type == RotationType.RT_HWD:
            dimension = [self.height, self.width, self.depth]
        elif self.rotation_type == RotationType.RT_HDW:
            dimension = [self.height, self.depth, self.width]
        elif self.rotation_type == RotationType.RT_DHW:
            dimension = [self.depth, self.height, self.width]
        elif self.rotation_type == RotationType.RT_DWH:
            dimension = [self.depth, self.width, self.height]
        elif self.rotation_type == RotationType.RT_WDH:
            dimension = [self.width, self.depth, self.height]
        else:
            dimension = []

        return dimension


class Bin:
    def __init__(self, name, type, weight, width, height, depth):
        self.name = name
        self.type = type
        self.width = width
        self.height = height
        self.depth = depth
        self.weight = weight
        self.items = []
        self.unfitted_items = []
        self.number_of_decimals = DEFAULT_NUMBER_OF_DECIMALS

    def format_numbers(self, number_of_decimals):
        self.width = set_to_decimal(self.width, number_of_decimals)
        self.height = set_to_decimal(self.height, number_of_decimals)
        self.depth = set_to_decimal(self.depth, number_of_decimals)
        self.weight = set_to_decimal(self.weight, number_of_decimals)
        self.number_of_decimals = number_of_decimals

    def string(self):
        return "%s(%sx%sx%s, max_weight:%s) vol(%s)" % (
            self.name, self.width, self.height, self.depth, self.weight,
            self.get_volume()
        )

    def get_volume(self):
        return set_to_decimal(
            self.width * self.height * self.depth, self.number_of_decimals
        )

    def get_total_volume(self):
        total_volume = 0

        for item in self.items:
            total_volume += item.get_volume()

        return set_to_decimal(total_volume, self.number_of_decimals)

    def get_total_weight(self):
        total_weight = 0

        for item in self.items:
            total_weight += item.weight

        return set_to_decimal(total_weight, self.number_of_decimals)

    def get_total_vol(self):
        total_vol = 0

        for item in self.items:
            total_vol += item.get_volume()

        return set_to_decimal(total_vol, self.number_of_decimals)

    def put_item1(self, item, pivot):
        fit = False
        valid_item_position = item.position
        item.position = pivot

        for i in range(0, len(RotationType.ALL)):
            item.rotation_type = i
            dimension = item.get_dimension()
            if (
                self.width < pivot[0] + dimension[0] or
                self.height < pivot[1] + dimension[1] or
                self.depth < pivot[2] + dimension[2]
            ):
                continue

            fit = True

            for current_item_in_bin in self.items:
                if intersect(current_item_in_bin, item):
                    fit = False
                    break

            if fit:
                if self.get_total_weight() + item.weight > self.weight:
                    # 超重、朝向错误
                    fit = False
                    return fit
                if item.rotation_type not in item.limit_dirct:
                    fit = False
                if fit:
                    self.items.append(item)
                    return fit

            if not fit:
                item.position = valid_item_position

            #return fit
            #如果每次都return 还要for干什么,不过他是item1

        if not fit:
            item.position = valid_item_position

        return fit

    def put_item2(self, item, pivot, axis, ib):
        fit = False
        valid_item_position = item.position
        item.position = pivot
        temp_limit_stack = item.limit_stack[item.rotation_type]
        for i in range(0, len(RotationType.ALL)):
            item.rotation_type = i
            dimension = item.get_dimension()
            if (
                self.width < pivot[0] + dimension[0] or
                self.height < pivot[1] + dimension[1] or
                self.depth < pivot[2] + dimension[2]
            ):
                continue
            fit = True
            xuankong_ornot = False
            # 错在这里了居然，item的position居然等于【0，0，0】
            # temp_limit_stack = item.limit_stack[item.rotation_type]
            for current_item_in_bin in self.items:
                if intersect(current_item_in_bin, item):
                    fit = False
                    break
                if item.position[2] == current_item_in_bin.position[2] + current_item_in_bin.depth:
                    xuankong_ornot = True
                    if current_item_in_bin.load_or_not[current_item_in_bin.rotation_type] == 1:  # 是否承重面
                        if current_item_in_bin.limit_load[current_item_in_bin.rotation_type] > item.limit_load[
                            item.rotation_type]:
                            fit = False
                        if current_item_in_bin.limit_stack[current_item_in_bin.rotation_type] == 0:
                            fit = False
                        else:
                            if (current_item_in_bin.limit_stack[current_item_in_bin.rotation_type] - 1) < \
                                    item.limit_stack[item.rotation_type]:
                                item.limit_stack[item.rotation_type] = current_item_in_bin.limit_stack[
                                                                           current_item_in_bin.rotation_type] - 1
                    else:
                        fit = False
            if item.position[2] == 0:
                xuankong_ornot = True

            if fit and xuankong_ornot:
                if self.get_total_weight() + item.weight > self.weight:
                    # 超重
                    fit = False
                    item.limit_stack[item.rotation_type] = temp_limit_stack
                    return fit and xuankong_ornot
                if item.rotation_type not in item.limit_dirct:  # 朝向错误
                    fit = False

            if fit and xuankong_ornot:
                self.items.append(item)
                return fit and xuankong_ornot
            # if not fit:
            #     item.position = valid_item_position

        if not fit or not xuankong_ornot:
            item.position = valid_item_position
            item.limit_stack[item.rotation_type] = temp_limit_stack
        return fit and xuankong_ornot


class Packer:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.items = []
        self.unfit_items = []
        self.total_items = 0
        self.bins = []

    '''def add_bin(self, bin):
        return self.bins.append(bin)

    def add_item(self, item):
        self.total_items = len(self.items) + 1

        return self.items.append(item)'''

    def add_item(self, item):
        return self.items.append(item)

    def pack_to_bin(self, bin, item):
        fitted = False

        if not bin.items:       # 放第一个箱子
            if bin.put_item1(item, START_POSITION):
                fitted = True
            else:
                print("shi'zhe")
                bin.unfitted_items.append(item)
                fitted = False
            return fitted

        for axis in range(0, 3):
            items_in_bin = bin.items
            items_in_bin.sort(key=lambda x: x.position[2])

            for ib in items_in_bin:
                pivot = [0, 0, 0]
                w, h, d = ib.get_dimension()
                if axis == Axis.HEIGHT:
                    pivot = [
                        ib.position[0],
                        ib.position[1] + h,
                        ib.position[2]
                    ]
                elif axis == Axis.WIDTH:
                    pivot = [
                        ib.position[0] + w,
                        ib.position[1],
                        ib.position[2]
                    ]
                elif axis == Axis.DEPTH:
                    pivot = [
                        ib.position[0],
                        ib.position[1],
                        ib.position[2] + d
                    ]

                if bin.put_item2(item, pivot, axis, ib):
                    fitted = True
                    break
            if fitted:
                break
        if not fitted:
            bin.unfitted_items.append(item)

        return fitted

    def pack(
        self, bin_num, is_reward, is_all_bins, bigger_first=True,
        number_of_decimals=DEFAULT_NUMBER_OF_DECIMALS
    ):
        if is_reward:
            for bin in Task.Bins_for_reward:
                bin.format_numbers(number_of_decimals)

            Task.Bins_for_reward.sort(
                key=lambda bin: bin.get_volume(), reverse=bigger_first
            )
            Bins = Task.Bins_for_reward[:bin_num]
        else:
            for bin in Task.Bins:
                bin.format_numbers(number_of_decimals)

            Task.Bins.sort(
                key=lambda bin: bin.get_volume(), reverse=bigger_first
            )  # reverse = True 降序， reverse = False 升序
            Bins = Task.Bins[:bin_num]
        for item in self.items:
            item.format_numbers(number_of_decimals)
            # reverse = True 降序， reverse = False 升序

        '''self.items.sort(
            key=lambda item: item.geolume(), reverse=bigger_first
        )'''


        Items = self.items.copy()
        print("====> 这次一共有几个箱子参与装箱", len(Bins))
        # for bin in Bins:
        #     print("====> 这是一个新箱子", bin.string())
        #     print("属于一个套机这个箱子里目前含有的货物")
        #     for item in bin.items:
        #         print("====> ", item.string())
        #     print("货物结束")
        for bin in Bins:

            '''for item in self.items:
                self.pack_to_bin(bin, item)'''
            # print("====> 这是一个新箱子", bin.string())
            # print("属于一个套机这个箱子里目前含有的货物")
            # for item in bin.items:
            #     print("====> ", item.string())
            # print("货物结束")
            i = 0
            # print(len(self.items))
            while i < len(self.items):

                item = self.items[i]
                # print("====> ", item.string())
                if self.pack_to_bin(bin, item):
                    # for binn in Bins:
                    #     print("====> 这是一个新箱子之放入一个货物之后", binn.string())
                    #     print("属于一个套机这个箱子里目前含有的货物")
                    #     for item in binn.items:
                    #         print("====> ", item.string())
                    #     print("货物结束")
                    # print("一个新的套机装进去了")
                    # print("====>装入物品的位置是 ", item.string())
                    if (item.suite_id != ''and is_reward == False) or (item.suite_id != 0 and is_reward == True):
                        # print('套机编码：%s #' %item.suite_id)
                        j = i
                        i += 1
                        if i < len(self.items):
                            while self.items[i].suite_id == item.suite_id:  # 套机限制,默认套机连续排列
                                # print(i)
                                item = self.items[i]
                                if self.pack_to_bin(bin, item):
                                    # print("====>装入物品的位置是 ", item.string())
                                    # print("属于一个套机这个箱子里目前含有的货物")
                                    # for item in bin.items:
                                    #     print("====> ", item.string())
                                    # print("货物结束")
                                    i += 1
                                    if i >= len(self.items):
                                        break
                                else:
                                    # print('从多少到多少是一个套机的')
                                    # print(j)
                                    # print(i)
                                    # print("这个箱子里目前含有的货物")
                                    # for item in bin.items:
                                    #     print("====> ", item.string())
                                    # print("货物结束")
                                    for k in range(j, i):
                                        # print(k)
                                        bin.items.remove(self.items[k])
                                    while self.items[i].suite_id == item.suite_id:
                                        i = i + 1
                                        if i >= len(self.items):
                                            break
                                    # print("除去一个套机的订单后这个箱子里目前含有的货物")
                                    # for item in bin.items:
                                    #     print("====> ", item.string())
                                    # print("货物结束")
                                    # i = i+1
                                    #排序不是这个订单的货物
                                    break
                                    #这里break的有问题，应该连续两个break直接换箱子
                                if i >= len(self.items):
                                    break

                    else:
                        i += 1
                else:
                    if (item.suite_id != '' and is_reward == False) or (item.suite_id != 0 and is_reward == True):
                        i = i + 1
                        if i < len(self.items):
                            while self.items[i].suite_id == item.suite_id:
                                i = i + 1
                                if i >= len(self.items):
                                    break
                    else:
                        i = i+1


            for item in bin.items:
                # print("====> ", item.string())已经装过箱的要移走
                try:
                    self.items.remove(item)
                except:
                    pass

        if self.items:
            # print("====> 全部物体", len(Items),"剩余物体", len(self.items))
            notfit_number = len(self.items)
            self.items = Items.copy()
            if not is_all_bins:
                for item in Items:
                    for bin in Bins:
                        try:
                            bin.items.remove(item)
                        except:
                            pass
            else:
                for bin in Bins:
                    if bin.items:
                        Task.Used_bins_for_reward.append(bin)
            return notfit_number
        else:
            for bin in Bins:
                if is_reward:
                    Task.Used_bins_for_reward.append(bin)
                    self.bins.append(bin)
                    # totalvol = bin.get_total_vol()
                    # if totalvol / bin.get_volume() >= 80:
                    #     Task.Bins_for_reward.remove(bin)
                else:
                    if bin.items:
                        if (bin in Task.Used_bins):
                            Task.Used_bins.remove(bin)
                            Task.Used_bins.append(bin)
                        else:
                            Task.Used_bins.append(bin)
                        print('加入一个箱子', len(Task.Used_bins))
                        self.bins.append(bin)
                        totalvol = bin.get_total_vol()
                        if totalvol / bin.get_volume() >= 80:
                            Task.Bins.remove(bin)

        return 0
    '''优先选择最大的箱子，填充率大于%80默认为装满，不继续装，防止重复被选影响系统效率'''

