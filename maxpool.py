import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./DATASET', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# kernel = torch.tensor([[1, 2, 1],
#                        [0, 1, 0],
#                        [2, 1, 0]])

# input = torch.reshape(input, (-1, 1, 5, 5))
# kernel = torch.reshape(kernel, (-1, 1, 3, 3))
# print(input.shape)
# print(kernel.shape)

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

ann = ANN()

writer = SummaryWriter("pool")
step = 0
for data in dataloader:
    imgs, targets = data
    # 注意！！！add_images  要加s
    writer.add_images("input", imgs, step)
    outputs = ann(imgs)
    writer.add_images("output", outputs, step)
    step += 1

writer.close()

