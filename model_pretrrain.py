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


vgg_false = torchvision.models.vgg16(pretrained=False)
vgg_true = torchvision.models.vgg16(pretrained=True)

vgg_true.classifier.add_module('add_Linear', nn.Linear(in_features=1000, out_features=10))
vgg_true.classifier[6] = nn.Linear(in_features=1000, out_features=10)
print(vgg_true)
