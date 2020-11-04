class RotationType:
    RT_WHD = 0
    RT_HWD = 1
    RT_HDW = 2
    RT_DHW = 3
    RT_DWH = 4
    RT_WDH = 5

    ALL = [RT_WHD, RT_HWD, RT_HDW, RT_DHW, RT_DWH, RT_WDH]


class Axis:
    WIDTH = 1
    HEIGHT = 0
    DEPTH = 2

    ALL = [WIDTH, HEIGHT, DEPTH]

class Task:
    Bins = []
    Used_bins = []
    Packers = []
    Bins_for_reward = []
    Used_bins_for_reward = []
    TradeId = ''
    Not_fit_num = 0
    msg = ''

    #tradeId str

