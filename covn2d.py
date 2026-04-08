import torchvision
from torch import nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root='./DATASET', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)

class Ann(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)


    def forward(self, input):
        input = self.conv1(input)
        return input

ann = Ann()

for data in test_loader:
    imgs, targets = data
    output = ann(imgs)
    print(output.shape)
    print(imgs.shape)
