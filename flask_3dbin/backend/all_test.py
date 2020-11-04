from .py3dbp.constants import Task
from .py3dbp import Bin, Item, Packer
from .datainit import*
from .py3dbp.auxiliary_methods import intersect, set_to_decimal
import numpy as np
import time
import torch
import json
import decimal
import sys
import os

import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

from decimal import Decimal
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import seaborn as sns
import random
import json
import copy
#第一次摆放作为baseline,没有这步了学弟不要baseline
#每个订单调用Ptr获取顺序
#先不考虑bin有东西
#输出值就是下标编号
#输入：[batch_size, input_size_item, seq_len_items]
#输出：probs：[seq_len_items*, batch_size, seq_len_items]，idxs：[seq_len_items*, batch_size]
'''1：套机编码（没有设为0）
2-5：长，宽，高，重量
6-9：第1个朝向是否成立（0-1），第1个承重是否成立（0-1）,承重（0-5），堆码（0-5）
10-13：第2个朝向是否成立（0-1），第2个承重是否成立（0-1）,承重（0-5），堆码（0-5）
14-17：第3个朝向是否成立（0-1），第3个承重是否成立（0-1）,承重（0-5），堆码（0-5）
18-21：第4个朝向是否成立（0-1），第4个承重是否成立（0-1）,承重（0-5），堆码（0-5）
12-25：第5个朝向是否成立（0-1），第5个承重是否成立（0-1）,承重（0-5），堆码（0-5）
26-29：第6个朝向是否成立（0-1），第6个承重是否成立（0-1）,承重（0-5），堆码（0-5）
'''

# !/usr/bin/env python
# coding: utf-8

# In[1]:




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Embedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        """
        Args:
            input_size:输入维度
            embedding_size:嵌入维度
        """

        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size))  # [input_size, embedding_size]
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def forward(self, inputs):
        """
        Args:
            inputs:[batch_size, input_size, seq_len]
        """

        batch_size = inputs.size(0)
        seq_len = inputs.size(2)
        embedding = self.embedding.repeat(batch_size, 1, 1)  # [batch_size, input_size, embedding_size]
        embedded = []
        inputs = inputs.unsqueeze(1)  # [batch_size, 1, input_size, seq_len]

        for i in range(seq_len):
            embedded.append(torch.bmm(inputs[:, :, :, i].float(), embedding))  # [batch_size, 1, embedding_size]
        embedded = torch.cat(embedded, 1)  # [batch_size, seq_len, embedding_size]
        return embedded  # [batch_size, seq_len, embedding_size]


# In[3]:


class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10):
        """
        Args:
            hidden_size:隐藏层中h的维数
        """

        super(Attention, self).__init__()
        self.C = C
        self.use_tanh = use_tanh
        self.W_hd = nn.Linear(hidden_size, hidden_size)  # 输入/输出[batch_size, hidden_size]
        self.W_he = nn.Conv1d(hidden_size, hidden_size, 1, 1)  # 输入/输出[batch_size, hidden_size, seq_len]
        self.V = nn.Parameter(torch.FloatTensor(hidden_size))  # [hidden_size]
        self.V.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

    def forward(self, hidden_d, hidden_e):
        """
        Args:
            hidden_d:[batch_size, hidden_size] decoder的hidden向量
            hidden_e:[batch_size, seq_len, hidden_size] encoder的hidden向量序列
        """

        batch_size = hidden_e.size(0)
        seq_len = hidden_e.size(1)

        hidden_e = hidden_e.permute(0, 2, 1)
        hidden_d = self.W_hd(hidden_d).unsqueeze(2)  # [batch_size, hidden_size, 1]
        hidden_e = self.W_he(hidden_e)  # [batch_size, hidden_size, seq_len]
        expanded_hidden_d = hidden_d.repeat(1, 1, seq_len)  # [batch_size, hidden_size, seq_len]
        V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, 1, hidden_size]
        logits = torch.bmm(V, torch.tanh(expanded_hidden_d + hidden_e)).squeeze(1)  # [batch_size, seq_len]

        if self.use_tanh:
            logits = self.C * torch.tanh(logits)
        else:
            logits = logits

        return hidden_e, logits  # [batch_size, hidden_size, seq_len] [batch_size, seq_len]


# In[4]:


class PointerNet(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 hidden_size,
                 n_glimpses,
                 use_tanh,
                 tanh_exploration):
        """
        Args:
            input_size:输入维度
            embedding_size:嵌入维度
            hidden_size:隐藏层中h的维数
            n_glimpses:注意力迭代次数
        """

        super(PointerNet, self).__init__()
        self.n_glimpses = n_glimpses

        # 输入[batch_size, input_size, seq_len]
        # 输出[batch_size, seq_len, embedding_szie]
        self.embedding = Embedding(input_size, embedding_size)

        # 输入[batch_size, seq_len, embedding_size]
        # 输出[batch_size, seq_len, hidden_size], ([1, batch_size, hidden_size], [1, batch_size, hidden_size])
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)

        # 输入[batch_szie, hidden_size] [batch_size, seq_len, hidden_size]
        # 输出[batch_size, hidden_size, seq_len] [batch_size, seq_len]
        self.pointer = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration)
        self.glimpse = Attention(hidden_size, use_tanh=False)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))  # [embedding_size]
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def apply_mask_to_logits(self, logits, mask, idxs):
        """
        Args:
            logits:[batch_size, seq_len] 注意力分布
            mask:[batch_size, seq_len] 掩码tensor
            idxs:[batch_size] 选择item索引
        """

        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask  # [batch_size, seq_len], [batch_size, seq_len]

    def forward(self, inputs_items):
        """
        Args:
            inputs_items:[batch_size, input_size_item, seq_len_items]
        """

        batch_size = inputs_items.size(0)
        seq_len = inputs_items.size(2)

        # [batch_size, seq_len_items, embedding_szie]
        embedded = self.embedding(inputs_items)

        # encoder_outputs:[batch_size, seq_len_items, hidden_size]
        # hidden:[1, batch_size, hidden_size]
        # context:[1, batch_size, hidden_size]
        encoder_outputs, (hidden, context) = self.encoder(embedded)

        prev_probs = []  # [seq_len*, batch_size, seq_len_items]
        prev_idxs = []  # [seq_len*, batch_size]
        mask = torch.zeros(batch_size, seq_len).bool().to(device)
        idxs = None

        # [batch_size, embedding_size]
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        for i in range(seq_len):
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            hidden_d = hidden.squeeze(0)  # [batch_size, hidden_size]
            for i in range(self.n_glimpses):
                hidden_e, logits = self.glimpse(hidden_d, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                hidden_d = torch.bmm(hidden_e, F.softmax(logits, dim=1).unsqueeze(2)).squeeze(2)

            _, logits = self.pointer(hidden_d, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            probs = F.softmax(logits, dim=1)  # [batch_size, seq_len_items]

            idxs = probs.multinomial(1).squeeze(1)  # [batch_size]
            temp = 1
            while True:
                for old_idxs in prev_idxs:
                    if old_idxs.eq(idxs).data.any():
                        temp = 0
                        break
                if temp == 1:
                    break
                else:
                    idxs = probs.multinomial(1).squeeze(1)
                    temp = 1

            # [batch_size, embedding_size]
            decoder_input = embedded[[x for x in range(batch_size)], idxs, :]

            prev_probs.append(probs)
            prev_idxs.append(idxs)

        return prev_probs, prev_idxs
        # [seq_len_items*, batch_size, seq_len_items]
        # [seq_len_items*, batch_size]


# In[10]:





# In[ ]:

def app_path():
    if hasattr(sys, 'frozen'):
        return os.path.dirname(sys.executable)
    return os.path.dirname(__file__)

def all_run(json_data):
    # device = torch.device("cpu")

    # In[2]:

    #
    #

    start = time.perf_counter()
    # with open('test_json/5output_test.json', 'r', encoding='GBK') as fp:
    #     json_data = json.load(fp)
    print(json_data)
    init_data(json_data)
    print('使用过的箱子数', len(Task.Used_bins))
    num_bin = len(Task.Bins)
    input_size_item = 29
    print(len(Task.Packers))
    for packer in Task.Packers:
        ptr_input = [[0] * len(packer.items) for n in range(input_size_item)]
        i=0
        # suite.id要改
        for item in packer.items:

            strsuit = item.suite_id
            print(strsuit)
            if not strsuit:
                # print('here')
                strsuit = '0'
                m = 1
                # print(int(str[0:]))
            else:
                # print('here',)
                for n in range(len(strsuit)):

                    if (strsuit[n] == 'T'or strsuit[n] == '0'):
                        n = n+1
                    else:
                        strsuit = strsuit[n:]
                        # print(int(str[i:]))
                        break
            item.suite_id = strsuit
            ptr_input[0][i] = float(item.suite_id)
            ptr_input[1][i] = float(item.width)
            ptr_input[2][i] = float(item.height)
            # print(item.width)
            # print(ptr_input[1][i])
            ptr_input[3][i] = float(item.depth)
            ptr_input[4][i] = float(item.weight)/10
            for m in range(6):
                if m in item.limit_dirct:
                    ptr_input[m * 4 + 5][i] = 1
                else:
                    ptr_input[m * 4 + 5][i] = 0

                ptr_input[m * 4 + 6][i] = float(item.load_or_not[m])
                ptr_input[m * 4 + 7][i] = float(item.limit_load[m])
                ptr_input[m * 4 + 8][i] = float(item.limit_stack[m])
            # print('ptr-inout', ptr_input)
            i = i + 1
        y = np.expand_dims(np.array(ptr_input), axis=0)
        print('here', y)
        y = torch.FloatTensor(y)

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_size_item = 29
        embedding_size = 128
        hidden_size = 128
        n_glimpses = 1
        use_tanh = True
        tanh_exploration = 10

        model = PointerNet(input_size_item,
                           embedding_size,
                           hidden_size,
                           n_glimpses,
                           use_tanh,
                           tanh_exploration).to(device)
        model.load_state_dict(torch.load('pointer.pkl'))
        model.eval()
        # model = torch.load('pointer.pkl').actor
        probs, idxs = model(y.to(device))
        # idxs = idxs.numpy().tolist()
        print(type(idxs))
        # idxs：[seq_len_items *, batch_size]
        packer_item_change = []
        for m in range(len(idxs)):
            packer_item_change.append(packer.items[idxs[m][0]])
        #更新packer重新装item
        packer.items = packer_item_change

        for i in range(num_bin):
            #如果这个箱子装满了这些订单 break运行下一个订单，否则就上下一个箱子
            Not_fit_num = packer.pack(i+1, False, i == num_bin-1)
            if Not_fit_num == 0:
                break
            elif i == (num_bin-1):
                Task.Not_fit_num = Not_fit_num






    i = 1   # 装箱步骤计数
    # output = {
    #             '装箱步骤': []
    #         }
    output_3d = {
        '箱子': []
    }
    print(len(Task.Used_bins))
    goodNum = 0
    totalCapacity = 0
    totalWeight = 0
    trainList = []
    for bin in Task.Used_bins:
        totalvol = bin.get_total_vol()
        totalCapacity = totalCapacity + totalvol
        totalWeight = totalWeight + bin.get_total_weight()
        print('货箱利用率')
        print(totalvol / bin.get_volume())
        bin.items.sort(key=lambda x: (x.position[0], x.position[1], x.position[2]))
        itemsinfo = []
        goodNum_bin = len(bin.items)
        goodNum = goodNum + goodNum_bin
        m = 1
        stepList = []
        for item in bin.items:
            item_info = []
            BinPickingResultGood = {
                'materialCode': item.id,
                'restrictionFlag': str(item.rotation_type),
                'x': float(item.position[0]),
                'y': float(item.position[1]),
                'z': float(item.position[2]),
                'trainIndex': int(m)
            }
            goodList_step = []
            goodList_step.append(BinPickingResultGood)
            BinPickingResultStep = {
                'step': int(m),
                'qty': 1,
                'directionNum':'2*2*1',
                'orderCode': item.packer_name,
                'goodList': goodList_step
            }
            m += 1
            stepList.append(BinPickingResultStep)
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
        BinPickingResultTrain = {
            'train': int(i),
            'modelCode': bin.type,
            'good_Num': goodNum_bin,
            'totalCapacity': totalvol,
            'totalWeight': bin.get_total_weight() * (set_to_decimal(0.001, 3)),
            'packingRate': totalvol / bin.get_volume(),
            'stepList': stepList
        }
        i = i + 1
        trainList.append(BinPickingResultTrain)
        output_3d['箱子'].append(info_3d)



    # tradeId = str(Task.TradeId),
    print(Task.TradeId)
    print(Task.Not_fit_num)
    if(Task.Not_fit_num == 0):
        status = 0
    else:
        status = 1
    msg = Task.msg
    carNum = len(Task.Used_bins)
    BinPickingResult = {
        'tradeId': Task.TradeId,
        'status' : status,
        'msg' : msg,
        'carNum' : carNum,
        'goodNum' : goodNum,
        'totalCapacity' : totalCapacity,
        'totalWeight' : totalWeight * (set_to_decimal(0.001, 3)),
        'trainList' : trainList
    }
    Pathout = os.path.join(app_path(), 'output.json')
    Pathout3d = os.path.join(app_path(), 'output_3d.json')
    if not os.path.exists(Pathout):
        # print('不存在pathout')
        file = open(Pathout, 'w')
        file.close()
    if not os.path.exists(Pathout3d):
        # print('不存在pathout')
        file = open(Pathout3d, 'w')
        file.close()

    print(Pathout)
    print(Pathout3d)
    jsondata = json.dumps(BinPickingResult, cls=DecimalEncoder, ensure_ascii=False)
    with open(Pathout, 'w', encoding='utf-8') as fp:
        fp.write(jsondata)
        fp.close()

    jsondata3d = json.dumps(output_3d, cls=DecimalEncoder, ensure_ascii=False)
    with open(Pathout3d, 'w', encoding='utf-8') as fp:
        fp.write(jsondata3d)
        fp.close()

    end = time.perf_counter()
    print('程序运行时间：%s' % (end - start))


















