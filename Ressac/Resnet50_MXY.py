import torch
import torch.nn as nn
import torchvision.models as models

# # 定义编码器的输入形状
# input_shape = (1, 557, 557)  # 单通道的输入


# 创建一个自定义编码器
class CustomEncoder(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomEncoder, self).__init__()

        # 加载预训练的ResNet-50模型
        self.resnet = models.resnet50(pretrained=True)
        # self.resnet = models.resnet34(pretrained=True)

        # 修改ResNet-50的输入通道数为1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 移除ResNet-50的最后一层全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # 添加一个自定义的全连接层来产生输出
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # 将输入传递给ResNet-50模型
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # 展平输出以供全连接层处理
        x = self.fc(x)  # 通过全连接层产生输出
        return x

#
# # 创建编码器实例
# encoder = CustomEncoder(num_classes=10)
#
# # 打印编码器的结构
# print(encoder)


def main():
    # blk = ResBlk(64, 128, stride=1)
    tmp = torch.randn(1, 1, 557, 557)
    encoder = CustomEncoder(num_classes=10)
    out = encoder(tmp)
    print("block:", out.shape)

    # # x = torch.randn(2, 3, 32, 32)
    # x = torch.randn(1, 1, 106, 106)
    # model = ResNet18()
    # out = model(x)
    # print("resnet:", out.shape)

if __name__=='__main__':
    main()
