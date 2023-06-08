import torch.nn as tnn


def conv_layer(chann_in, chann_out, k_size, p_size):#将一个卷积层、BN层和非线性层封装在一起
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):#将上述封装的卷积层按照指定的参数组装在一起，最后再加上一个最大池化层
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):#将一个线性层、BN层和非线性层封装在一起
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


class VGG16(tnn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()

        # near infrared image feature extraction近红外图像特征提取
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)#指定每个VGG块的参数
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)

        # color image feature extraction彩色图像特征提取
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # classifier分类
        self.layer6 = vgg_fc_layer(7 * 7 * 512, 4096)#过度全连接层
        self.layer7 = vgg_fc_layer(4096, 4096)
        self.layer8 = tnn.Linear(4096, n_classes)#最后的全连接层

    def forward(self, x):
        out = self.layer1(x)#两层卷积
        out = self.layer2(out)#两层卷积
        out = self.layer3(out)#三层卷积
        out = self.layer4(out)#三层卷积
        vgg16_features = self.layer5(out)#三次卷积
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out