####################################################################################################
# 功能：投毒攻击
# 作者：洪建华
# 版本：创建——20241013_1350
#       编写添加触发器函数和MyAttack类的代码——20241013_1456
#       修复myattck投毒攻击无法执行的BUG——20241014_1607
#       优化数据集分配和myattack投毒训练——20241014_1634
#       优化myattack投毒训练损失函数，增加裁剪范围缩放的功能——20241101_1023
#       添加FedIMP投毒攻击——20241115_1514
#       myattack不动态修改学习率——20241204_1439
#       添加PatternBackdoor图案后门攻击——20241205_1628
#       添加CMP投毒攻击——20241206_1912
#       修复FedIMP投毒攻击提取参数BUG——20241211_1138
#       添加InvertedGradient反向梯度攻击——20250117_1540
#       添加CSA余弦相似度攻击——20250121_1659
#       优化addTrigger函数使适用于CIFAR10和MNIST数据集——20250123_2145
#       优化addTrigger函数使适用于GTSRB数据集——20250207_2243
#       添加标签翻转函数和FL-MMR投毒攻击——20250319_1957
#       添加FedGhost投毒攻击——20250406_1818
####################################################################################################
import torch
import torch.nn as nn
import numpy as np
import ot
import copy
from FLFunction import train, test
from Defense import MultiKrum


####################################################################################################
# 功能：在数据集中添加触发器
# 输入：dataset：数据集
#       target_to：将添加触发器的数据标签改为target_to
#       rate：添加触发器的数据比例
#       dataset_name：数据集名称
# 输出：dataset：投毒数据集
####################################################################################################
def addTrigger(dataset, target_to, rate, dataset_name):
    trigger_indices = [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3)]
    if isinstance(dataset.data, torch.Tensor):
        trigger_data = dataset.data.clone()
    else:
        trigger_data = torch.tensor(dataset.data).clone()
    trigger_targets = torch.tensor(dataset.targets).clone()
    poison_data_num = 0
    if dataset_name == 'CIFAR10' or dataset_name == 'MNIST':
        for idx in range(len(trigger_data)):
            if poison_data_num / (idx + 1) < rate:
                poison_data_num += 1
                for i, j in trigger_indices:
                    trigger_data[idx][i][j] = 255
                trigger_targets[idx] = target_to
    elif dataset_name == 'GTSRB':
        for idx in range(len(trigger_data)):
            if poison_data_num / (idx + 1) < rate:
                poison_data_num += 1
                for i, j in trigger_indices:
                    trigger_data[idx][0][i][j] = 255
                    trigger_data[idx][1][i][j] = 255
                    trigger_data[idx][2][i][j] = 255
                trigger_targets[idx] = target_to
    if isinstance(dataset.data, torch.Tensor):
        dataset.data = trigger_data
    else:
        dataset.data = trigger_data.numpy()
    dataset.targets = trigger_targets.tolist()


####################################################################################################
# 功能：对数据集标签翻转
# 输入：dataset：数据集
#       flip_label_map：如何翻转标签，如{0: 1, 1: 0}表示0类和1类互换
# 输出：poisoned_dataset：投毒数据集
####################################################################################################
def flipLabel(dataset, flip_label_map):
    poisoned_dataset = []
    for img, label in dataset:
        if label in flip_label_map:
            label = flip_label_map[label]
        poisoned_dataset.append((img, label))
    return poisoned_dataset


# 我的投毒攻击：使恶意模型参数在良性模型参数范围内
class MyAttack:
    ################################################################################################
    # 功能：投毒模型训练
    # 输入：epoch_size：训练轮数
    #       poison_model：投毒模型
    #       global_model：全局模型
    #       train_loader：训练数据迭代器
    #       test_loader：测试数据迭代器
    #       criterion：损失函数
    #       fedprox_miu：FedProx正则化项的强度
    #       optimizer：优化器
    #       device：设备
    # 输出：poison_model：投毒模型
    ################################################################################################
    def poisonTrain(self, epoch_size, poison_model, global_model, train_loader, test_loader, criterion, fedprox_miu, optimizer, device):
        for epoch in range(epoch_size):
            train_loss = train(poison_model, global_model, train_loader, criterion, fedprox_miu, optimizer, device)
            print('poison model train epoch:{}, trainloss:{:.4f}'.format(epoch, train_loss))


    ################################################################################################
    # 功能：修剪投毒模型参数在良性模型参数范围内
    # 输入：poison_model：投毒模型
    #       benign_models：良性模型
    #       threshold_multipl：阈值倍数
    # 输出：poison_model：投毒模型
    ################################################################################################
    def clip(self, poison_model, benign_models, threshold_multipl):
        poison_model_params = list(poison_model.parameters())
        benign_models_params = [list(model.parameters()) for model in benign_models]
        for i, p_poison in enumerate(poison_model_params):
            benign_params_i = [params[i].data for params in benign_models_params]
            min_vals = torch.min(torch.stack(benign_params_i), dim=0).values
            max_vals = torch.max(torch.stack(benign_params_i), dim=0).values
            min_vals_multipl = threshold_multipl * min_vals - (threshold_multipl - 1) * max_vals
            max_vals_multipl = threshold_multipl * max_vals - (threshold_multipl - 1) * min_vals
            p_poison.data = torch.max(torch.min(p_poison.data, max_vals_multipl), min_vals_multipl)


# FedIMP攻击
class FedIMP:
    pass

# PatternBackdoor图案后门攻击
class PatternBackdoor:
    pass


# CMP攻击
class CMP:
    pass


# InvertedGradient反向梯度攻击
class InvertedGradient:
    pass


# CSA余弦相似度攻击
class CSA:
    pass


# FL-MMR攻击
class FLMMR:
    pass


# FedGhost攻击
class FedGhost:
    pass


####################################################################################################
####################################################################################################
