####################################################################################################
# 功能：使用Cifar10数据集
# 作者：洪建华
# 版本：创建——20241008_1739
#       编写下载并加载数据集函数代码——20241008_1751
####################################################################################################
import torchvision
import torchvision.transforms as transforms
import os


####################################################################################################
# 功能：下载并加载CIFAR-10数据集
# 输入：dataset_path：数据集存放路径
# 输出：trainset：训练集
#       testset：测试集
####################################################################################################
def loadCifar10(dataset_path):

    # 定义数据增强和数据加载
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 下载并加载CIFAR-10数据集
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(dataset_path, "train"), train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=os.path.join(dataset_path, "test"), train=False, download=True, transform=transform_test)

    return trainset, testset


####################################################################################################
####################################################################################################