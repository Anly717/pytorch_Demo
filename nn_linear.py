import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./DATASET', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, x):
        x = self.linear1(x)
        return x

ann = ANN()

for data in dataloader:
    imgs, targets = data

    print(imgs.shape)

    output = torch.flatten(imgs)
    print(output.shape)
    output = ann(output)
    print(output.shape)

