import torchvision.datasets
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10(root='./DATASET', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./DATASET', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络


class ANN(nn.Module):

    def __init__(self):
        super(ANN, self).__init__()

        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


ann = ANN()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
optim = torch.optim.SGD(ann.parameters(), lr=0.01)

# 设置训练网络的参数
# 训练次数
total_train_step = 0
# 测试次数
total_test_step = 0
# 训练论述
epoch = 10

writer = SummaryWriter('train')


for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+1))
    ann.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = ann(imgs)
        loss = loss_fn(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}， loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)
    # 测试步骤
    ann.eval()
    total_test_loss = 0
    total_test_acc = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = ann(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            acc = (outputs.argmax(1) == targets).sum()
            total_test_acc += acc.item()

    print("整体数据集上loss：{}".format(total_test_loss))
    print("整体数据集上正确率：{}".format(total_test_acc/test_data_size))

    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_acc', total_test_acc/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(ann, "ann_{}.pth".format(i))
    # torch.save(ann.state_dict(), "ann_{}.pth".format(i))
    print("模型已保存")

writer.close()
