####################################################################################################
# 功能：使用Mnist数据集
# 作者：洪建华
# 版本：创建——20250123_1517
#       编写下载并加载数据集函数代码——20250123_1528
####################################################################################################
import torchvision
import torchvision.transforms as transforms
import os


####################################################################################################
# 功能：下载并加载MNIST数据集
# 输入：dataset_path：数据集存放路径
# 输出：trainset：训练集
#       testset：测试集
####################################################################################################
def loadMnist(dataset_path):

    # 定义数据增强和数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]区间
    ])

    # 下载并加载MNIST数据集
    trainset = torchvision.datasets.MNIST(root=os.path.join(dataset_path, "train"), train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=os.path.join(dataset_path, "test"), train=False, download=True, transform=transform)


    return trainset, testset


####################################################################################################
####################################################################################################