'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AllDNet(nn.Module):
    def __init__(self):
        super(AllDNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv2 = nn.Linear(6*14*14, 16*10*10)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        activations = []
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        # out = out.view(out.size(0), -1)
        # activations.append(out)
        out = F.relu(self.conv2(out))
        # out = out.view(out.size(0), 16, 10, -1)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        activations.append(out)
        out = F.relu(self.fc1(out))
        activations.append(out)
        out = F.relu(self.fc2(out))
        activations.append(out)
        out = self.fc3(out)
        return out, activations

