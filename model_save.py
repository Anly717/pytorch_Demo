import torchvision.datasets
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./DATASET', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)


vgg_false = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT')


# 保存方式1
torch.save(vgg_false, './vgg_false.pth')

# 保存方式2(模型参数)
torch.save(vgg_false.state_dict(), './vgg_false_state_dict.pth')




