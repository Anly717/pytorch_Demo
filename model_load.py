import torchvision.datasets
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 保存方式1 加载模型
model1 = torch.load("vgg_false.pth", weights_only=False)

# 保存方式2 加载模型
vgg_false = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT')
vgg_false.load_state_dict(torch.load("vgg_false_state_dict.pth"))
# model2 = torch.load("vgg_false_state_dict.pth")

print(vgg_false)
