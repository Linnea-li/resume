from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F

def sk_model(model_num):
    if model_num == 0:
        model = LinearRegression()
    elif model_num == 1:
        model = BayesianRidge(compute_score = True)
    elif model_num == 2:
        model = ElasticNetCV(alphas = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], \
                                    l1_ratio = [.01, .1, .5, .9, .99],  max_iter = 5000)
    elif model_num == 3:
        model = RandomForestRegressor()
    elif model_num == 4:
        model = BaggingRegressor()
    elif model_num == 5:
        model = DecisionTreeRegressor()

    return model


# Define model architecture
class SetConv(nn.Module):
    # 都是一些维度的大小参数
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units):
        super(SetConv, self).__init__()

        # nn.Linear 用于设置网络中的全连接层，第一个是in, 第二个是out
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)

        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)

        #针对表连接的条件
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)

        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1) #输出层结果

    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]
        # num_joins???


        # F.relu 非线性激活函数 参数是input
        # Tensor按元素应用整流线性单位函数

        #这些应该就是在网络中一层一层往下算
        hid_sample = F.relu(self.sample_mlp1(samples)) #输入
        hid_sample = F.relu(self.sample_mlp2(hid_sample)) #上一层的结果作为下一层的输入

        hid_sample = hid_sample * sample_mask  # Mask
        hid_sample = torch.sum(hid_sample, dim = 1, keepdim = False)
        sample_norm = sample_mask.sum(1, keepdim = False)
        hid_sample = hid_sample / sample_norm  # Calculate average only over non-masked parts

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim = 1, keepdim = False)
        predicate_norm = predicate_mask.sum(1, keepdim = False)
        hid_predicate = hid_predicate / predicate_norm

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim = 1, keepdim = False)
        join_norm = join_mask.sum(1, keepdim = False)
        hid_join = hid_join / join_norm

        hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        hid = F.relu(self.out_mlp1(hid))
        
        #最终结果
        out = torch.sigmoid(self.out_mlp2(hid))
        return out