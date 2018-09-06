'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1   = nn.Linear(16*5*5, 120)
        self.fc1   = nn.Linear(576, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

        self.adapter = nn.Conv2d(6, 64, 1, 1, 0, bias=False)

    def forward(self, x, manifold_mixup=False, mixup_batches=None, mixup_lambda=None):
        out_intermediate = F.relu(self.conv1(x))
        if manifold_mixup:
            out_intermediate = mixup_lambda * out_intermediate[mixup_batches, ...] + (1-mixup_lambda) * out_intermediate

        out = F.max_pool2d(out_intermediate, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # import pdb; pdb.set_trace()
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        out_intermediate = self.adapter(out_intermediate)
        return out, out_intermediate
