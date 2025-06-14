####################################################################################################
# 功能：投毒防御
# 作者：洪建华
# 版本：创建——20241115_1526
#       编写MESAS防御的代码——20241115_1526
#       MESAS防御增加Dixon's Q检验——20241120_1413
#       优化代码，统一格式，增加注释——20241121_1610
#       增加Multi-Krum鲁棒聚合防御的代码——20241127_1733
#       增加FLTrust防御的代码——20250104_1953
#       增加ERR防御的代码——20250113_0048
#       增加FedGT防御的代码——20250401_1535
#       增加FedNorAvg防御的代码——20250413_1139
####################################################################################################
import numpy as np
from scipy.stats import ttest_ind, levene, ks_2samp
import torch
import copy
from concurrent.futures import ThreadPoolExecutor
from FLFunction import test, fedavg


# MESAS防御
class MESAS:
    pass


# Multi-Krum鲁棒聚合
class MultiKrum:
    pass


# FLTrust防御
class FLTrust:
    pass


# ERR防御
class ERR:
    pass


# FedGT防御
class FedGT:
    pass


# FedGT防御
class FedNorAvg:
    pass


####################################################################################################
####################################################################################################
