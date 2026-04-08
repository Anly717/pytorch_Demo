import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        # self.conv1 = Conv2d(3, 32, 5, padding=2)
        # self.pool1 = nn.MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        # self.pool2 = nn.MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.pool3 = nn.MaxPool2d(2)
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(1024, 64)
        # self.fc2 = nn.Linear(64, 10)

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
        # x = self.conv1(x)
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.pool2(x)
        # x = self.conv3(x)
        # x = self.pool3(x)
        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.model1(x)
        return x

ann = ANN()

print(ann)
input = torch.ones((64, 3, 32, 32))
output = ann(input)
print(output.shape)

writer = SummaryWriter("logs_model")
writer.add_graph(ann, input)
writer.close()