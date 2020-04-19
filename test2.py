# 训练+测试


import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import cv2

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# 超参数
EPOCH = 1  # 训练整批数据的次数
BATCH_SIZE = 50
LR = 0.001  # 学习率
DOWNLOAD_MNIST = True  # 表示还没有下载数据集，如果数据集下载好了就写False

# 下载mnist手写数据集
train_data = torchvision.datasets.MNIST(
    root='./data/',  # 保存或提取的位置  会放在当前文件夹中
    train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
    transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray

    download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
)

test_data = torchvision.datasets.MNIST(
    root='./data/',
    train=False  # 表明是测试集
)

# 批训练 50个samples， 1  channel，28x28 (50,1,28,28)
# Torch中的DataLoader是用来包装数据的工具，它能帮我们有效迭代数据，这样就可以进行批训练
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True  # 是否打乱数据，一般都打乱
)

# 进行测试
# 为节约时间，测试时只测试前2000个
#
test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)[:2000] / 255
# torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
# 图像的pixel本来是0到255之间，除以255对图像进行归一化使取值范围在(0,1)
test_y = test_data.test_labels[:2000]


# 用class类来建立CNN模型
# CNN流程：卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        展平多维的卷积成的特征图->接入全连接层(Linear)->输出

class CNN(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self):
        super(CNN, self).__init__()
        # 建立第一个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv1 = nn.Sequential(
            # 第一个卷积con2d
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels=1,  # 输入图片的高度，因为minist数据集是灰度图像只有一个通道
                out_channels=16,  # n_filters 卷积核的高度
                kernel_size=5,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),  # 输出图像大小(16,28,28)
            # 激活函数
            nn.ReLU(),
            # 池化，下采样
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
            # 输出图像大小(16,14,14)
        )
        # 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv2 = nn.Sequential(
            # 输入图像大小(16,14,14)
            nn.Conv2d(  # 也可以直接简化写成nn.Conv2d(16,32,5,1,2)
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出图像大小(32,7,7)
        )
        # 建立全卷积连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输出是10个类

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.conv1(x)  # x先通过conv1
        x = self.conv2(x)  # 再通过conv2
        # 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)

# 训练
# 把x和y 都放入Variable中，然后放入cnn中计算output，最后再计算误差

# 优化器选择Adam
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 损失函数
loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted

# 开始训练
# for epoch in range(EPOCH):
#     for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch data
#         output = cnn(b_x)  # 先将数据放到cnn中计算output
#         loss = loss_func(output, b_y)  # 输出和真实标签的loss，二者位置不可颠倒
#         optimizer.zero_grad()  # 清除之前学到的梯度的参数
#         loss.backward()  # 反向传播，计算梯度
#         optimizer.step()  # 应用梯度
#
#         if step % 50 == 0:
#             test_output = cnn(test_x)
#             pred_y = torch.max(test_output, 1)[1].data.numpy()
#             accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
#
# torch.save(cnn.state_dict(), 'cnn2.pkl')#保存模型

# 加载模型，调用时需将前面训练及保存模型的代码注释掉，否则会再训练一遍
cnn.load_state_dict(torch.load('cnn2.pkl'))
cnn.eval()
# print 10 predictions from test data
inputs = test_x[:32]  # 测试32个数据
test_output = cnn(inputs)
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')  # 打印识别后的数字
# print(test_y[:10].numpy(), 'real number')

img = torchvision.utils.make_grid(inputs)
img = img.numpy().transpose(1, 2, 0)

# 下面三行为改变图片的亮度
# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
# img = img * std + mean
cv2.imshow('win', img)  # opencv显示需要识别的数据图片
key_pressed = cv2.waitKey(0)
