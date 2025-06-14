####################################################################################################
# 功能：联邦学习主函数
# 作者：洪建华
# 版本：创建——20241010_1636
#       编写联邦学习代码（使用cifar10数据集，10客户端，Dirichlet分布alpha为0.1，resnet18模型，学习率
#           0.01，无调度器，轮数100轮）——20241010_1725
#       改为动态调整学习率，用以提高预测准确率——20241012_1442
#       添加myattck投毒攻击功能——20241013_1613
#       修复myattck投毒攻击无法执行的BUG——20241014_1607
#       优化数据集分配和myattack投毒训练——20241014_1634
#       修改代码以适应改进后的myattack投毒，不动态调整学习率——20241101_1038
#       添加设置随机数种子，FedIMP投毒攻击，MESAS防御的功能——20241115_1538
#       添加投毒攻击数据集大小的设置功能，myattack设置每轮都训练的功能，将添加触发器由倒数比例设置改
#           为小数比例设置，添加MultiKrum投毒防御功能——20241204_1446
#       添加PatternBackdoor投毒攻击功能——20241205_1640
#       添加CMP投毒攻击功能——20241206_1925
#       添加FLTrust投毒防御功能——20250104_1953
#       增加ERR防御的代码——20250113_0054
#       添加InvertedGradient反向梯度攻击——20250117_1623
#       添加CSA余弦相似度攻击——20250121_1707
#       添加数据集和模型选择功能，添加MNIST数据集和CNN模型——20250123——1605
#       添加GTSRB数据集和SqueezeNet模型——20250207_2023
#       修改SqueezeNet模型训练时的参数和优化器以提高预测准确率——20250210_2212
#       添加FL-MMR投毒攻击功能——20250319_2018
#       添加FedGT投毒防御功能——20250401_1550
#       添加FedGhost投毒攻击功能——20250406_1824
#       添加FedNorAvg投毒防御功能——20250413_1143
####################################################################################################
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import copy
import random
from FLFunction import train, test, fedavg, aggregation, dirichletSplit, optimInit, setRandomSeed
from ResNet18 import ResNet18
from CNN import CNN3
from SqueezeNet import SqueezeNet1
from Cifar10 import loadCifar10
from Mnist import loadMnist
from GTSRB import loadGTSRB
from Attack import addTrigger, flipLabel, MyAttack, FedIMP, PatternBackdoor, CMP, InvertedGradient, CSA, FLMMR, FedGhost
from Defense import MESAS, MultiKrum, FLTrust, ERR, FedGT, FedNorAvg
from c2net.context import prepare
from c2net.context import upload_output

# 参数设置
if True:
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_set_random_seed', type=bool, default=True, help='是否设置随机数种子')
    parser.add_argument('--random_seed', type=int, default=42, help='随机数种子')
    parser.add_argument('--num_users', type=int, default=10, help='客户端数量')
    parser.add_argument('--data_set', type=str, default='CIFAR10', help='数据集')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5, help='dirichlet分布alpha参数值')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    parser.add_argument('--dataloader_num_workers', type=int, default=2, help='数据迭代器加载数据时的进程数')
    parser.add_argument('--epoch_size', type=int, default=100, help='训练轮数')
    parser.add_argument('--fedprox_miu', type=float, default=0.1, help='FedProx正则化项的强度')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='SGD动量')
    parser.add_argument('--sgd_weight_decay', type=float, default=1e-5, help='SGD权重衰减')
    parser.add_argument('--poison_type', type=str, default='myAttack', help='投毒攻击类型')
    parser.add_argument('--poison_num_users', type=int, default=1, help='恶意客户端数量')
    parser.add_argument('--poison_dataset_rate', type=float, default=0.1, help='恶意客户端数据集大小(占比)')
    parser.add_argument('--poison_target_to', type=int, default=0, help='将添加触发器的数据标签改为target_to')
    parser.add_argument('--poison_train_rate', type=float, default=0.5, help='添加触发器的训练数据比例')
    parser.add_argument('--poison_test_rate', type=float, default=0.5, help='添加触发器的测试数据比例')
    parser.add_argument('--myattack_train_all_epoch', type=bool, default=True, help='myattack投毒模型是否每轮都训练')
    parser.add_argument('--myattack_train_epochs', type=int, nargs='+', default=[0], help='myattack投毒模型被训练的轮次')
    parser.add_argument('--myattack_train_epoch_size', type=int, default=1, help='myattack投毒模型每次训练的轮数')
    parser.add_argument('--myattack_fedprox_miu', type=float, default=0, help='myattack投毒模型训练时FedProx正则化项的强度')
    parser.add_argument('--myattack_threshold_multipl', type=float, default=10000, help='myattack投毒模型裁剪阈值倍数')
    parser.add_argument('--FedIMP_delta_max', type=float, default=1000, help='FedIMP最大delta值')
    parser.add_argument('--defense_type', type=str, default='none', help='投毒防御类型')
    parser.add_argument('--MultiKrum_m', type=int, default=5, help='MultiKrum聚合的模型数量')

if __name__ == '__main__':
    print("begin running!!!!!!!")

    # 选择设备（CPU或GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载parser参数
    if True:
        args, unknown = parser.parse_known_args()
        is_set_random_seed = args.is_set_random_seed
        random_seed = args.random_seed
        num_users = args.num_users
        data_set = args.data_set
        dirichlet_alpha = args.dirichlet_alpha
        batch_size = args.batch_size
        dataloader_num_workers = args.dataloader_num_workers
        epoch_size = args.epoch_size
        fedprox_miu = args.fedprox_miu
        lr = args.lr
        sgd_momentum = args.sgd_momentum
        sgd_weight_decay = args.sgd_weight_decay
        poison_type = args.poison_type
        poison_num_users = args.poison_num_users
        poison_dataset_rate = args.poison_dataset_rate
        poison_target_to = args.poison_target_to
        poison_train_rate = args.poison_train_rate
        poison_test_rate = args.poison_test_rate
        myattack_train_all_epoch = args.myattack_train_all_epoch
        myattack_train_epochs = args.myattack_train_epochs
        myattack_train_epoch_size = args.myattack_train_epoch_size
        myattack_fedprox_miu = args.myattack_fedprox_miu
        myattack_threshold_multipl = args.myattack_threshold_multipl
        FedIMP_delta_max = args.FedIMP_delta_max
        defense_type = args.defense_type
        MultiKrum_m = args.MultiKrum_m

    # 设置随机数种子
    if is_set_random_seed:
        setRandomSeed(random_seed)

    # 初始化导入数据集和预训练模型到容器内
    c2net_context = prepare()

    # 导入数据集
    if data_set == 'CIFAR10':
        dataset_torch_path = c2net_context.dataset_path+"/CIFAR10Dataset_torch"
        train_dataset, test_dataset = loadCifar10(dataset_torch_path)
        dataset_num_classes = 10
    elif data_set == 'MNIST':
        dataset_torch_path = c2net_context.dataset_path+"/MNISTDataset_torch"
        train_dataset, test_dataset = loadMnist(dataset_torch_path)
        dataset_num_classes = 10
    elif data_set == 'GTSRB':
        dataset_torch_path = c2net_context.dataset_path+"/GTSRBDataset_torch"
        train_dataset, test_dataset = loadGTSRB(dataset_torch_path)
        dataset_num_classes = 43

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_num_workers)

    # 将训练数据集拆分成非IID，分配给良性客户端
    if poison_type == 'none':
        users_dataset = dirichletSplit(train_dataset, num_users, dirichlet_alpha)
        local_train_loaders = [DataLoader(users_dataset[i], batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers) for i in range(num_users)]
    else:
        users_dataset = dirichletSplit(train_dataset, num_users, dirichlet_alpha)
        local_train_loaders = [DataLoader(users_dataset[i], batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers) for i in range(num_users - poison_num_users)]

    # 制作恶意客户端数据集
    if poison_type == 'none' or poison_type == 'myAttack' or poison_type == 'PatternBackdoor' or poison_type == 'CMP':

        # 制作后门投毒训练数据集
        poison_train_dataset = copy.deepcopy(train_dataset)
        addTrigger(poison_train_dataset, poison_target_to, poison_train_rate, data_set)
        original_length = len(poison_train_dataset)
        reduced_length = int(original_length * poison_dataset_rate)
        indices = random.sample(range(original_length), reduced_length)
        poison_train_dataset = torch.utils.data.Subset(poison_train_dataset, indices)
        poison_train_loader = DataLoader(poison_train_dataset, batch_size=batch_size, shuffle=True)

        # 制作后门投毒测试数据集
        poison_test_dataset = copy.deepcopy(test_dataset)
        addTrigger(poison_test_dataset, poison_target_to, poison_test_rate, data_set)
        original_length = len(poison_test_dataset)
        reduced_length = int(original_length * poison_dataset_rate)
        indices = random.sample(range(original_length), reduced_length)
        poison_test_dataset = torch.utils.data.Subset(poison_test_dataset, indices)
        poison_test_loader = DataLoader(poison_test_dataset, batch_size=batch_size, shuffle=True)

        # 制作后门投毒成功率测试数据集
        poison_acc_test_dataset = copy.deepcopy(test_dataset)
        addTrigger(poison_acc_test_dataset, poison_target_to, 1, data_set)
        poison_acc_test_loader = DataLoader(poison_acc_test_dataset, batch_size=batch_size, shuffle=True)
    elif poison_type == 'FLMMR':
        
        # 制作标签翻转训练数据集
        poison_train_dataset = copy.deepcopy(train_dataset)
        poison_train_dataset = flipLabel(poison_train_dataset, {i: (i + 1) % dataset_num_classes for i in range(dataset_num_classes)})
        poison_train_loader = DataLoader(poison_train_dataset, batch_size=batch_size, shuffle=True)
        original_length = len(poison_train_dataset)
        reduced_length = int(original_length * poison_dataset_rate)
        indices = random.sample(range(original_length), reduced_length)
        poison_train_dataset = torch.utils.data.Subset(poison_train_dataset, indices)
        poison_train_loader = DataLoader(poison_train_dataset, batch_size=batch_size, shuffle=True)

        # 模拟良性数据集
        benign_train_dataset = copy.deepcopy(train_dataset)
        original_length = len(benign_train_dataset)
        reduced_length = int(original_length * poison_dataset_rate)
        indices = random.sample(range(original_length), reduced_length)
        benign_train_dataset = torch.utils.data.Subset(benign_train_dataset, indices)
        benign_train_loader = DataLoader(benign_train_dataset, batch_size=batch_size, shuffle=True)
    elif poison_type == 'FedIMP' or poison_type == 'InvertedGradient':
        poison_train_dataset = copy.deepcopy(train_dataset)
        original_length = len(poison_train_dataset)
        reduced_length = int(original_length * poison_dataset_rate)
        indices = random.sample(range(original_length), reduced_length)
        poison_train_dataset = torch.utils.data.Subset(poison_train_dataset, indices)
        poison_train_loader = DataLoader(poison_train_dataset, batch_size=batch_size, shuffle=True)

    # 加载模型
    if data_set == 'CIFAR10':
        Model = ResNet18
    elif data_set == 'MNIST':
        Model = CNN3
    elif data_set == 'GTSRB':
        Model = SqueezeNet1
    global_model = Model().to(device)
    local_model = Model().to(device)
    if data_set == 'GTSRB':
        pretrain_model_path = c2net_context.pretrain_model_path+"/SqueezeNet/acc10.pth"
        global_model.load_state_dict(torch.load(pretrain_model_path))

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optimInit(data_set, local_model, lr, sgd_momentum, sgd_weight_decay)

    # 投毒准备
    if poison_type == 'myAttack':
        Attack = MyAttack()
        target_model = Model().to(device)
        poison_model = Model().to(device)
        target_optimizer = optimInit(data_set, target_model, lr, sgd_momentum, sgd_weight_decay)
    elif poison_type == 'FedIMP':
        Attack = FedIMP()
        poison_model = Model().to(device)
    elif poison_type == 'PatternBackdoor':
        Attack = PatternBackdoor()
        target_model = Model().to(device)
        poison_model = Model().to(device)
        target_optimizer = optimInit(data_set, target_model, lr, sgd_momentum, sgd_weight_decay)
    elif poison_type == 'CMP':
        Attack = CMP()
        target_model = Model().to(device)
        poison_model = Model().to(device)
        target_optimizer = optimInit(data_set, target_model, lr, sgd_momentum, sgd_weight_decay)
    elif poison_type == 'InvertedGradient':
        Attack = InvertedGradient()
        malicious_model = Model().to(device)
        poison_model = Model().to(device)
        malicious_optimizer = optimInit(data_set, malicious_model, lr, sgd_momentum, sgd_weight_decay)
        poison_optimizer = optimInit(data_set, poison_model, lr, sgd_momentum, sgd_weight_decay)
    elif poison_type == 'CSA':
        Attack = CSA()
        proxy_model = Model().to(device)
        poison_model = Model().to(device)
        proxy_optimizer = optimInit(data_set, proxy_model, lr, sgd_momentum, sgd_weight_decay)
        poison_optimizer = optimInit(data_set, poison_model, lr, sgd_momentum, sgd_weight_decay)
    elif poison_type == 'FLMMR':
        Attack = FLMMR()
        benign_model = Model().to(device)
        target_model = Model().to(device)
        benign_optimizer = optimInit(data_set, benign_model, lr, sgd_momentum, sgd_weight_decay)
        target_optimizer = optimInit(data_set, target_model, lr, sgd_momentum, sgd_weight_decay)
    elif poison_type == 'FedGhost':
        Attack = FedGhost()
        poison_model = Model().to(device)

    # 防御准备
    if defense_type == 'MESAS':
        Defense = MESAS()
    elif defense_type == 'MultiKrum':
        Defense = MultiKrum()
    elif defense_type == 'FLTrust':
        Defense = FLTrust()
        reference_model = Model().to(device)
        reference_optimizer = optimInit(data_set, reference_model, lr, sgd_momentum, sgd_weight_decay)
    elif defense_type == 'ERR':
        Defense = ERR()
    elif defense_type == 'FedGT':
        Defense = FedGT()
    elif defense_type == 'FedNorAvg':
        Defense = FedNorAvg()

    # 训练和测试
    sign = 0
    for epoch in range(epoch_size):
        print('myattack_threshold_multipl:{}, myattack_fedprox_miu:{}'.format(myattack_threshold_multipl, myattack_fedprox_miu))
        local_models = []

        # 本地训练
        if poison_type == 'none':
            for user_id in range(num_users):

                # 获取全局模型
                local_model.load_state_dict(global_model.state_dict())

                # 训练本地模型
                user_loader = local_train_loaders[user_id]
                train_loss = train(local_model, global_model, user_loader, criterion, fedprox_miu, optimizer, device)
                print('epoch:{}, userid:{}, trainloss:{:.4f}'.format(epoch, user_id, train_loss))
                local_models.append(copy.deepcopy(local_model))
        elif poison_type == 'myAttack':

            # 良性训练
            for user_id in range(num_users - poison_num_users):
                local_model.load_state_dict(global_model.state_dict())
                user_loader = local_train_loaders[user_id]
                train_loss = train(local_model, global_model, user_loader, criterion, fedprox_miu, optimizer, device)
                # print('epoch:{}, userid:{}, trainloss:{:.4f}'.format(epoch, user_id, train_loss))
                local_models.append(copy.deepcopy(local_model))

            # 投毒训练
            if epoch == 0:
                target_model.load_state_dict(global_model.state_dict())
            if myattack_train_all_epoch or epoch in myattack_train_epochs:
                target_model.load_state_dict(global_model.state_dict())
                Attack.poisonTrain(myattack_train_epoch_size, target_model, global_model, poison_train_loader, poison_test_loader, criterion, myattack_fedprox_miu, target_optimizer, device)
                accuracy = test(target_model, test_loader, device)
                print('epoch:{}, target_model accuracy:{:.4f}'.format(epoch, accuracy))
                accuracy = test(target_model, poison_acc_test_loader, device)
                print('epoch:{}, target_model poison_accuracy:{:.4f}'.format(epoch, accuracy))

            # 加载并裁剪
            poison_model.load_state_dict(target_model.state_dict())
            Attack.clip(poison_model, local_models, myattack_threshold_multipl)

            # 测试
            accuracy = test(poison_model, test_loader, device)
            print('epoch:{}, poison_user, accuracy:{:.4f}'.format(epoch, accuracy))
            accuracy = test(poison_model, poison_acc_test_loader, device)
            print('epoch:{}, poison_user, poison_accuracy:{:.4f}'.format(epoch, accuracy))
            for user_id in range(num_users - poison_num_users, num_users):
                local_models.append(copy.deepcopy(poison_model))

        elif poison_type == 'FedIMP':
            fisher_list = []
            data_sizes = [len(users_dataset[i]) for i in range(num_users - poison_num_users)]  # 计算良性数据集大小

            # 良性客户端训练并计算 Fisher 信息
            for user_id in range(num_users - poison_num_users):
                local_model.load_state_dict(global_model.state_dict())
                train_loss = train(local_model, global_model, local_train_loaders[user_id], criterion, fedprox_miu, optimizer, device)
                print('epoch:{}, userid:{}, trainloss:{:.4f}'.format(epoch, user_id, train_loss))
                fisher_info = Attack.calcFisherInfo(local_model, local_train_loaders[user_id], device)
                fisher_list.append(fisher_info)
                local_models.append(copy.deepcopy(local_model))

            # 计算加权平均 Fisher 信息
            avg_fisher = Attack.weightAvgFisher(fisher_list, data_sizes)
            binary_mask = Attack.createBinaryMask(avg_fisher)

            # 生成恶意更新
            mean_update, std_update = Attack.calcMeanAndStd(local_models)
            boosting_coefficient = Attack.binarySearchBoostingCoefficient(mean_update, std_update, binary_mask, local_models, FedIMP_delta_max)
            # boosting_coefficient = 1000
            print('boosting_coefficient:{:.4f}'.format(boosting_coefficient))
            malicious_update = Attack.genMaliciousUpdate(mean_update, std_update, binary_mask, boosting_coefficient)
            poison_model_state_dict = Attack.reshapeModel(malicious_update, poison_model)

            poison_model.load_state_dict(poison_model_state_dict)
            for user_id in range(num_users - poison_num_users, num_users):
                local_models.append(copy.deepcopy(poison_model))  # 将恶意模型添加到本地模型列表中
        elif poison_type == 'PatternBackdoor':

            # 良性训练
            for user_id in range(num_users - poison_num_users):
                local_model.load_state_dict(global_model.state_dict())
                user_loader = local_train_loaders[user_id]
                train_loss = train(local_model, global_model, user_loader, criterion, fedprox_miu, optimizer, device)
                local_models.append(copy.deepcopy(local_model))

            # 目标模型训练
            if True:
                target_model.load_state_dict(global_model.state_dict())
                Attack.poisonTargetTrain(target_model, poison_train_loader, poison_test_loader, criterion, target_optimizer, device)
                accuracy = test(target_model, test_loader, device)
                print('epoch:{}, target_model accuracy:{:.4f}'.format(epoch, accuracy))
                accuracy = test(target_model, poison_acc_test_loader, device)
                print('epoch:{}, target_model poison_accuracy:{:.4f}'.format(epoch, accuracy))

            # 恶意模型制作
            poison_model.load_state_dict(target_model.state_dict())
            for user_id in range(num_users - poison_num_users, num_users):
                local_models.append(copy.deepcopy(poison_model))
        elif poison_type == 'CMP':

            # 良性训练
            for user_id in range(num_users - poison_num_users):
                local_model.load_state_dict(global_model.state_dict())
                user_loader = local_train_loaders[user_id]
                train_loss = train(local_model, global_model, user_loader, criterion, fedprox_miu, optimizer, device)
                local_models.append(copy.deepcopy(local_model))

            # 目标模型训练
            if True:
                target_model.load_state_dict(global_model.state_dict())
                Attack.poisonTargetTrain(target_model, poison_train_loader, poison_test_loader, criterion, target_optimizer, device)
                accuracy = test(target_model, test_loader, device)
                print('epoch:{}, target_model accuracy:{:.4f}'.format(epoch, accuracy))
                accuracy = test(target_model, poison_acc_test_loader, device)
                print('epoch:{}, target_model poison_accuracy:{:.4f}'.format(epoch, accuracy))

            # 恶意模型制作
            Attack.createPoisonModel(global_model, local_models, target_model, poison_model, poison_num_users, MultiKrum_m)
            accuracy = test(poison_model, test_loader, device)
            print('epoch:{}, poison_model accuracy:{:.4f}'.format(epoch, accuracy))
            accuracy = test(poison_model, poison_acc_test_loader, device)
            print('epoch:{}, poison_model poison_accuracy:{:.4f}'.format(epoch, accuracy))
            for user_id in range(num_users - poison_num_users, num_users):
                local_models.append(copy.deepcopy(poison_model))
        elif poison_type == 'InvertedGradient':

            # 良性训练
            for user_id in range(num_users - poison_num_users):
                local_model.load_state_dict(global_model.state_dict())
                user_loader = local_train_loaders[user_id]
                train_loss = train(local_model, global_model, user_loader, criterion, fedprox_miu, optimizer, device)
                local_models.append(copy.deepcopy(local_model))

            # 制作投毒数据集
            if True:
                malicious_model.load_state_dict(global_model.state_dict())
                poison_train_loader = Attack.createPoisonDataset(malicious_model, poison_train_loader, criterion, malicious_optimizer, device)

            # 恶意训练
            for user_id in range(num_users - poison_num_users, num_users):
                poison_model.load_state_dict(global_model.state_dict())
                Attack.poisonTargetTrain(poison_model, poison_train_loader, criterion, poison_optimizer, device)
                accuracy = test(poison_model, test_loader, device)
                print('epoch:{}, poison_model accuracy:{:.4f}'.format(epoch, accuracy))
                local_models.append(copy.deepcopy(poison_model))
        elif poison_type == 'CSA':

            # 良性训练
            for user_id in range(num_users - poison_num_users):
                local_model.load_state_dict(global_model.state_dict())
                user_loader = local_train_loaders[user_id]
                train_loss = train(local_model, global_model, user_loader, criterion, fedprox_miu, optimizer, device)
                local_models.append(copy.deepcopy(local_model))

            # 恶意训练
            proxy_model.load_state_dict(global_model.state_dict())
            train(proxy_model, global_model, test_loader, criterion, fedprox_miu, proxy_optimizer, device)
            Attack.createPoisonModel(poison_model, global_model, local_models, proxy_model)
            for user_id in range(num_users - poison_num_users, num_users):
                local_models.append(copy.deepcopy(poison_model))
        elif poison_type == 'FLMMR':
            # 良性训练
            for user_id in range(num_users - poison_num_users):
                local_model.load_state_dict(global_model.state_dict())
                user_loader = local_train_loaders[user_id]
                train_loss = train(local_model, global_model, user_loader, criterion, fedprox_miu, optimizer, device)
                local_models.append(copy.deepcopy(local_model))

            # 目标模型训练
            if epoch == 0:
                target_model.load_state_dict(global_model.state_dict())
                Attack.poisonTargetTrain(target_model, poison_train_loader, criterion, target_optimizer, device)
                accuracy = test(target_model, test_loader, device)
                print('epoch:{}, target_model accuracy:{:.4f}'.format(epoch, accuracy))

            # 模拟良性训练
            # if True:
            #     benign_model.load_state_dict(global_model.state_dict())
            #     train_loss = train(benign_model, global_model, benign_train_loader, criterion, fedprox_miu, benign_optimizer, device)
            #     accuracy = test(benign_model, test_loader, device)
            #     print('epoch:{}, benign_model accuracy:{:.4f}'.format(epoch, accuracy))
            fedavg(benign_model, local_models)

            # 恶意模型制作
            poison_model = Attack.createPoisonModel(benign_model, target_model, epoch, num_users)
            accuracy = test(poison_model, test_loader, device)
            print('epoch:{}, poison_model accuracy:{:.4f}'.format(epoch, accuracy))
            for user_id in range(num_users - poison_num_users, num_users):
                local_models.append(copy.deepcopy(poison_model))
        elif poison_type == 'FedGhost':

            # 良性训练
            for user_id in range(num_users - poison_num_users):
                local_model.load_state_dict(global_model.state_dict())
                user_loader = local_train_loaders[user_id]
                train_loss = train(local_model, global_model, user_loader, criterion, fedprox_miu, optimizer, device)
                local_models.append(copy.deepcopy(local_model))

            # 恶意训练
            Attack.createPoisonModel(poison_model, global_model, epoch)
            if epoch < 10:
                poison_model.load_state_dict(local_models[0].state_dict())
            accuracy = test(poison_model, test_loader, device)
            print('epoch:{}, poison_model accuracy:{:.4f}'.format(epoch, accuracy))
            for user_id in range(num_users - poison_num_users, num_users):
                local_models.append(copy.deepcopy(poison_model))

        # 服务器聚合全局模型并测试
        if defense_type == 'none':
            fedavg(global_model, local_models)
        elif defense_type == 'MESAS':
            local_models, benign_indices = Defense.runDefense(global_model, local_models)
            if all(i < num_users - poison_num_users for i in benign_indices):
                if sign == 0:
                    if myattack_threshold_multipl > 5.5:
                        myattack_threshold_multipl = 5.5
                    else:
                        myattack_threshold_multipl = max(myattack_threshold_multipl * 0.95, 0.5)
                    myattack_fedprox_miu = min(2 * 15 ** (-myattack_threshold_multipl + 3), 2)
                    if data_set == 'CIFAR10':
                        myattack_fedprox_miu = myattack_fedprox_miu*39.72427043*2
                    elif data_set == 'MNIST':
                        myattack_fedprox_miu = myattack_fedprox_miu*6.113030049*2
                    elif data_set == 'GTSRB':
                        myattack_fedprox_miu = myattack_fedprox_miu*63.55617282*2
                else:
                    sign -= 1
            else:
                sign = 2
            fedavg(global_model, local_models)
        elif defense_type == 'MultiKrum':
            local_models, indexs = Defense.runDefense(local_models, MultiKrum_m)
            if all(i < num_users - poison_num_users for i in indexs):
                if sign == 0:
                    if myattack_threshold_multipl > 5.5:
                        myattack_threshold_multipl = 5.5
                    else:
                        myattack_threshold_multipl = max(myattack_threshold_multipl * 0.95, 0.5)
                    myattack_fedprox_miu = min(2 * 15 ** (-myattack_threshold_multipl + 3), 2)
                    if data_set == 'CIFAR10':
                        myattack_fedprox_miu = myattack_fedprox_miu*39.72427043*2
                    elif data_set == 'MNIST':
                        myattack_fedprox_miu = myattack_fedprox_miu*6.113030049*2
                    elif data_set == 'GTSRB':
                        myattack_fedprox_miu = myattack_fedprox_miu*63.55617282*2
                else:
                    sign -= 1
            else:
                sign = 2
            fedavg(global_model, local_models)
        elif defense_type == 'FLTrust':
            reference_model.load_state_dict(global_model.state_dict())
            train(reference_model, global_model, test_loader, criterion, fedprox_miu, reference_optimizer, device)
            # accuracy = test(reference_model, test_loader, device)
            # print('epoch:{}, reference_model accuracy:{:.4f}'.format(epoch, accuracy))
            # accuracy = test(reference_model, poison_acc_test_loader, device)
            # print('epoch:{}, reference_model poison_accuracy:{:.4f}'.format(epoch, accuracy))
            weights = Defense.runDefense(global_model, local_models, reference_model)
            if all(i < 0.01 for i in weights[num_users - poison_num_users:]):
                if sign == 0:
                    if myattack_threshold_multipl > 5.5:
                        myattack_threshold_multipl = 5.5
                    else:
                        myattack_threshold_multipl = max(myattack_threshold_multipl * 0.95, 0.5)
                    myattack_fedprox_miu = min(2 * 15 ** (-myattack_threshold_multipl + 3), 2)
                    if data_set == 'CIFAR10':
                        myattack_fedprox_miu = myattack_fedprox_miu*39.72427043*2
                    elif data_set == 'MNIST':
                        myattack_fedprox_miu = myattack_fedprox_miu*6.113030049*2
                    elif data_set == 'GTSRB':
                        myattack_fedprox_miu = myattack_fedprox_miu*63.55617282*2
                else:
                    sign -= 1
            else:
                sign = 2
            aggregation(global_model, local_models, weights)
        elif defense_type == 'ERR':
            local_models,eliminate_index = Defense.runDefense(global_model, local_models, 1, test_loader, device)
            if all(i in eliminate_index for i in range(num_users - poison_num_users, num_users)):
                if sign == 0:
                    if myattack_threshold_multipl > 5.5:
                        myattack_threshold_multipl = 5.5
                    else:
                        myattack_threshold_multipl = max(myattack_threshold_multipl * 0.95, 0.5)
                    myattack_fedprox_miu = min(2 * 15 ** (-myattack_threshold_multipl + 3), 2)
                    if data_set == 'CIFAR10':
                        myattack_fedprox_miu = myattack_fedprox_miu*39.72427043*2
                    elif data_set == 'MNIST':
                        myattack_fedprox_miu = myattack_fedprox_miu*6.113030049*2
                    elif data_set == 'GTSRB':
                        myattack_fedprox_miu = myattack_fedprox_miu*63.55617282*2
                else:
                    sign -= 1
            else:
                sign = 2
            fedavg(global_model, local_models)
        elif defense_type == 'FedGT':
            eliminate_index = Defense.runDefense(local_models, poison_num_users, test_loader, device)
            if all(i in eliminate_index for i in range(num_users - poison_num_users, num_users)):
                if sign == 0:
                    if myattack_threshold_multipl > 5.5:
                        myattack_threshold_multipl = 5.5
                    else:
                        myattack_threshold_multipl = max(myattack_threshold_multipl * 0.95, 0.5)
                    myattack_fedprox_miu = min(2 * 15 ** (-myattack_threshold_multipl + 3), 2)
                    if data_set == 'CIFAR10':
                        myattack_fedprox_miu = myattack_fedprox_miu*39.72427043*2
                    elif data_set == 'MNIST':
                        myattack_fedprox_miu = myattack_fedprox_miu*6.113030049*2
                    elif data_set == 'GTSRB':
                        myattack_fedprox_miu = myattack_fedprox_miu*63.55617282*2
                else:
                    sign -= 1
            else:
                sign = 2
            local_models = [model for i, model in enumerate(local_models) if i not in eliminate_index]
            fedavg(global_model, local_models)
        elif defense_type == 'FedNorAvg':
            Defense.fednoravg(global_model, local_models, data_set)
        global_accuracy = test(global_model, test_loader, device)
        print('epoch:{}, global_model accuracy:{:.4f}'.format(epoch, global_accuracy))
        if poison_type == 'none' or poison_type == 'myAttack' or poison_type == 'PatternBackdoor' or poison_type == 'CMP':
            accuracy = test(global_model, poison_acc_test_loader, device)
            print('epoch:{}, global_model poison_accuracy:{:.4f}'.format(epoch, accuracy))

    upload_output()


####################################################################################################
####################################################################################################
