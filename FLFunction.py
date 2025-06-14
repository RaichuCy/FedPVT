####################################################################################################
# 功能：联邦学习中用到的一些函数
# 作者：洪建华
# 版本：创建——20241007_1951
#       编写模型训练、模型测试、fedavg聚合、Dirichlet划分函数代码——20241008_1726
#       优化模型训练损失函数——20241101_1028
#       添加设置随机数种子的功能——20241115_1426
#       编写权重聚合函数代码——20250104_1953
#       编写初始化优化器函数——20250210_2210
####################################################################################################
import torch
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Subset


####################################################################################################
# 功能：模型训练
# 输入：model：模型
#       global_model：全局模型
#       data_loader：数据迭代器
#       criterion：损失函数
#       fedprox_miu：fedprox正则化强度
#       optimizer：优化器
#       device：设备
# 输出：model：模型
#       train_loss：训练过程中的损失值总和
####################################################################################################
def train(model, global_model, data_loader, criterion, fedprox_miu, optimizer, device):
    model.train()
    train_loss = 0.0
    global_model_params = list(global_model.parameters())
    local_model_params = list(model.parameters())
    for _, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 添加FedProx的正则化项
        fedprox_reg = 0.0
        for param_local, param_global in zip(local_model_params, global_model_params):
            fedprox_reg += torch.norm(param_local - param_global) ** 2
        loss += 0.5 * fedprox_miu * fedprox_reg

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计训练过程中的损失值总和
        train_loss += loss.item()
    return train_loss


####################################################################################################
# 功能：模型测试
# 输入：model：模型
#       data_loader：数据迭代器
#       device：设备
# 输出：accuracy：预测准确率
####################################################################################################
def test(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = correct / total
    return accuracy


####################################################################################################
# 功能：fedavg聚合
# 输入：global_model：全局模型
#       local_models：本地模型
# 输出：global_model：全局模型
####################################################################################################
def fedavg(global_model, local_models):
    num_users = len(local_models)
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = 0
        for i in range(num_users):
            global_dict[k] += local_models[i].state_dict()[k].float()
        global_dict[k] /= num_users
    global_model.load_state_dict(global_dict)


####################################################################################################
# 功能：权重聚合
# 输入：global_model：全局模型
#       local_models：本地模型
#       weights：本地模型权重
# 输出：global_model：全局模型
####################################################################################################
def aggregation(global_model, local_models, weights):
    num_users = len(local_models)
    total_weight = 0
    for i in range(num_users):
        total_weight += weights[i]
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = 0
        for i in range(num_users):
            global_dict[k] += local_models[i].state_dict()[k].float() * weights[i]
        global_dict[k] /= total_weight
    global_model.load_state_dict(global_dict)


####################################################################################################
# 功能：使用Dirichlet分布进行非IID数据划分
# 输入：dataset：被划分的数据集
#       divisions_num：划分份数
#       alpha：Dirichlet分布alpha值
# 输出：dataset_subs：划分后的子数据集
####################################################################################################
def dirichletSplit(dataset, divisions_num, alpha):

    # 按类别划分数据索引
    num_classes = len(np.unique([dataset[i][1] for i in range(len(dataset))]))
    data_indices = [[] for _ in range(num_classes)]
    for idx, (img, label) in enumerate(dataset):
        data_indices[label].append(idx)

    # 使用Dirichlet分布进行划分
    data_sub_indices = [[] for _ in range(divisions_num)]
    class_sub_counts = [np.zeros(num_classes) for _ in range(divisions_num)]
    for cls in range(num_classes):
        cls_indices = data_indices[cls]
        np.random.shuffle(cls_indices)
        proportions = np.random.dirichlet([alpha] * divisions_num)
        split_indices = np.array_split(cls_indices, (proportions[:-1].cumsum() * len(cls_indices)).astype(int))
        for id, indices in enumerate(split_indices):
            data_sub_indices[id].extend(indices)
            class_sub_counts[id][cls] += len(indices)
    data_sub_indices = [np.array(indices) for indices in data_sub_indices]
    dataset_subs = [Subset(dataset, data_sub_indices[i]) for i in range(divisions_num)]

    # 输出划分结果
    for id, counts in enumerate(class_sub_counts):
        print(f'子数据集{id}: ' + ', '.join([f'Label {cls}: {int(count)}' for cls, count in enumerate(counts)]))

    return dataset_subs


####################################################################################################
# 功能：初始化优化器
# 输入：dataset_name：数据集名称
#       model：模型
#       lr：学习率
#       sgd_momentum：SGD优化器动量因子
#       sgd_weight_decay：SGD优化器权重衰减强度
# 输出：optimizer：优化器
####################################################################################################
def optimInit(dataset_name, model, lr, sgd_momentum = 0, sgd_weight_decay = 0):
    if dataset_name == 'CIFAR10' or dataset_name == 'MNIST':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=sgd_momentum, weight_decay=sgd_weight_decay)
    elif dataset_name == 'GTSRB':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer


####################################################################################################
# 功能：设置随机数种子
# 输入：seed：随机数种子
# 输出：无
####################################################################################################
def setRandomSeed(seed):
    # 设置 Python 随机数种子
    random.seed(seed)
    # 设置 NumPy 随机数种子
    np.random.seed(seed)
    # 设置 PyTorch 随机数种子
    torch.manual_seed(seed)  # 对于 CPU 张量
    torch.cuda.manual_seed(seed)  # 对于当前 GPU 张量
    torch.cuda.manual_seed_all(seed)  # 对于所有 GPU（如果使用多 GPU）
    # 设置 CuDNN 相关的参数（提高可重现性）
    torch.backends.cudnn.deterministic = True  # 确保卷积运算是确定的
    torch.backends.cudnn.benchmark = False  # 禁用优化的算法来保持一致性


####################################################################################################
####################################################################################################
