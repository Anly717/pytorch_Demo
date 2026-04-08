import torchvision.datasets
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

from torchvision import transforms

img_path = 'hymenoptera_data/dog.png'

imgs = Image.open(img_path)
print(imgs)

imgs = imgs.convert('RGB')

transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
imgs = transform(imgs)
print(imgs.shape)


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


model = torch.load('ann_9.pth', weights_only=False)
imgs = torch.reshape(imgs, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    imgs = imgs.cuda()
    output = model(imgs)
print(output)
output = output.argmax()
print(output)


