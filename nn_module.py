from typing import Any
import torch

from torch import nn


class Ann(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


ann1 = Ann()
x = torch.tensor(1.0)
output = ann1(x)
print(output)
