####################################################################################################
# 功能：SqueezeNet模型
# 作者：洪建华
# 版本：创建——20250207_2007
#       搬运SqueezeNet模型代码，来源：https://blog.csdn.net/weixin_45084253/article/details/12455571
#       4?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522504ac481ab5a32065fb93630438b6d52%2
#       522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=504ac481ab5a32065fb
#       93630438b6d52&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_posit
#       ive~default-1-124555714-null-null.142^v101^pc_search_result_base6&utm_term=SqueezeNet&spm=10
#       18.2226.3001.4187——20250207_2017
####################################################################################################
import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()
        self.in_channels = in_channels
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        e1 = self.expand1x1_activation(self.expand1x1(x))
        e2 = self.expand3x3_activation(self.expand3x3(x))
        out = torch.cat([e1, e2], 1)
        return out


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=43):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        out = torch.flatten(x, 1)
        return out


def SqueezeNet1():
    return SqueezeNet()


####################################################################################################
####################################################################################################