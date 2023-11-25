import torch
import torch.nn as nn
import torchvision.models as models

# # 定义编码器的输入形状
# input_shape = (1, 557, 557)  # 单通道的输入


# 创建一个自定义编码器
class CustomEncoder(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomEncoder, self).__init__()

        # 加载预训练的ResNet-34模型
        # self.resnet = models.resnet34(pretrained=True)
        self.resnet = models.resnet18(pretrained=True)

        # 修改ResNet-34的输入通道数为1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 移除ResNet-34的最后一层全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # 添加一个自定义的全连接层来产生输出
        # self.fc = nn.Linear(512, num_classes)  # 注意，ResNet-34的最后一层特征图大小为512

        self.sample = GaussianSample(512, num_classes)

    def forward(self, x):
        # 将输入传递给ResNet-34模型
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # 展平输出以供全连接层处理
        # x = self.fc(x)  # 通过全连接层产生输出
        x = self.sample(x)
        return x

class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """
    def reparametrize(self, mu, logvar):
        epsilon = torch.randn(mu.size(), requires_grad=False, device=mu.device)
        std = logvar.mul(0.5).exp_()
#         std = torch.clamp(logvar.mul(0.5).exp_(), -5, 5)
        z = mu.addcmul(std, epsilon)

        return z

class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)

        return self.reparametrize(mu, log_var), mu, log_var


def build_mlp(layers, activation=nn.ReLU(), bn=False, dropout=0):
    """
    Build multilayer linear perceptron
    """
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if bn:
            net.append(nn.BatchNorm1d(layers[i]))
        net.append(activation)
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)
class Decoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0, output_activation=nn.Sigmoid()):
        # dims: 是一个包含三个整数的列表，表示潜在空间维度(z_dim)、隐藏层维度(h_dim)和输出维度(x_dim)。
        # bn: 是一个布尔值，表示是否在隐藏层中使用批量归一化(Batch Normalization)。默认为False，即不使用批量归一化。
        # dropout: 是一个浮点数，表示在隐藏层中使用的dropout比例。默认为0，即不使用dropout。
        # output_activation: 是一个PyTorch的激活函数实例，表示输出层的激活函数。默认为nn.Sigmoid()，即使用Sigmoid函数作为输出层的激活函数。
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(Decoder, self).__init__()

        [z_dim, h_dim, x_dim] = dims
        # [z_dim, *h_dim]将潜在空间维度与隐藏层维度拼接成一个列表，
        # 作为build_mlp函数的输入，从而构建了一个包含多个隐藏层的MLP，并将其赋值给self.hidden，作为这个解码器的隐藏层。
        self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)
        # self.hidden = build_mlp([z_dim]+h_dim, bn=bn, dropout=dropout)
        # 创建了一个线性层(nn.Linear)，用于将隐藏层的输出映射到输出层。
        # [z_dim, *h_dim][-1]表示将潜在空间维度与隐藏层维度拼接成一个列表，并取最后一个元素，即MLP输出层的维度。
        # 然后，用MLP输出层的维度和输出维度x_dim作为输入，创建了一个线性层，并将其赋值给self.reconstruction。
        self.reconstruction = nn.Linear([z_dim, *h_dim][-1], x_dim)
        # self.reconstruction = nn.Linear(([z_dim]+h_dim)[-1], x_dim)
        # 将传入的输出激活函数实例赋值给self.output_activation，表示这个解码器的输出层使用指定的激活函数。
        self.output_activation = output_activation

    def forward(self, x):
        x = self.hidden(x)
        # x = x.view(-1, 1, self.input_shape, self.input_shape)
        if self.output_activation is not None:
            return self.output_activation(self.reconstruction(x))
        else:
            return self.reconstruction(x)


# 定义解码器
class CustomDecoder(nn.Module):
    def __init__(self):
        super(CustomDecoder, self).__init__()

        # 创建多层反卷积网络
        # self.deconv_net = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, kernel_size=4, stride=4, padding=0),  # 4x upsampling
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4, padding=0),  # 16x upsampling
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4, padding=0),  # 64x upsampling
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),  # 128x upsampling
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=13),  # 256x upsampling
        # )
        self.deconv_net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=8, stride=8, padding=0),  # 4x upsampling
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=7, padding=0),  # 16x upsampling
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2, padding=0),  # 64x upsampling
        )


    def forward(self, x):
            return self.deconv_net(x)

# # 创建解码器
# decoder = Decoder()
#
# # 输入数据的示例，形状为 (32, 512, 1, 1)
# input_data = torch.randn(32, 512, 1, 1)
#
# # 使用解码器进行转换
# output_data = decoder(input_data)
#
# # 输出数据的形状
# print(output_data.shape)  # 应该是 (32, 1, 106, 106)
#
#
#
# # 创建编码器实例
# encoder = CustomEncoder(num_classes=10)
#
# # 打印编码器的结构
# print(encoder)


# def main():
#     # blk = ResBlk(64, 128, stride=1)
#     # tmp = torch.randn(1, 1, 557, 557)
#     # encoder = CustomEncoder(num_classes=10)
#     # out = encoder(tmp)
#     # print("block:", out.shape)
#
#     # tmp = torch.randn(3, 10)
# #     # decode_dim = []
# #     # decoder = Decoder([10, decode_dim, 32])
# #     # out = decoder(tmp)
# #     # print("block:", out.shape)
#
#     # 输入的形状为 (32, 10)
#
#     # 输入的通道数为 10
#
#     tmp = torch.randn(3, 10)
#
#     fc4 = torch.nn.Linear(10, 512)
#     tmp=fc4(tmp)
#
#     tmp = tmp.view(3, 512, 1, 1)
#     # 创建解码器
#     decoder = CustomDecoder()
#
#     # 使用解码器进行转换
#     out = decoder(tmp)
#     print("block:", out.shape)
#
#     # # x = torch.randn(2, 3, 32, 32)
#     # x = torch.randn(1, 1, 106, 106)
#     # model = ResNet18()
#     # out = model(x)
#     # print("resnet:", out.shape)
#
# if __name__=='__main__':
#     main()
