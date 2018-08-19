"""
@Title: DenseNets on PyTorch for CIFAR-10

@References:

[1] Gao Huang, Zhuang Liu, Laurens van der Maaten
    Densely Connected Deep Convolutional Networks. arXiv:1512.03385
    
[2] PyTorch Open Source Repository
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py   
"""


import math
import torch
import torch.nn as nn


class SingleLayer(nn.Module):
    
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out



class Bottleneck(nn.Module):
    
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, 4*growthRate, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(4*growthRate)
        self.conv2 = nn.Conv2d(4*growthRate, growthRate, kernel_size=3, padding=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.avgpool(out)
        return out


class DenseNet(nn.Module):
    
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()
        
        compression = True if reduction < 1 else False  # Determine if DenseNet-C
        
        nDenseLayers = (depth-4) // 3
        if bottleneck:
            nDenseLayers //= 2
            
        nChannels = 2 * growthRate if compression and bottleneck else 16
        
        # First convolution
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
        
        # Dense Block 1 
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseLayers, bottleneck)
        nChannels += nDenseLayers*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        
        # Transition Block 1
        self.trans1 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        
        # Dense Block 2
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseLayers, bottleneck)
        nChannels += nDenseLayers*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        
        # Transition Block 2
        self.trans2 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        
        # Dense Block 3
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseLayers, bottleneck)
        
        # Transition Block 3
        nChannels += nDenseLayers*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        
        # Dense Layer
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        def flatten(x): return x.view(x.size(0), -1)
        self.fc = nn.Linear(nChannels, nClasses)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseLayers, bottleneck):
        ''' Function to build a Dense Block '''
        layers = []
        for i in range(int(nDenseLayers)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)                     # 32x32
        out = self.trans1(self.dense1(out))     # 16x16
        out = self.trans2(self.dense2(out))     # 8x8
        out = self.dense3(out)                  # 8x8
        out = self.avgpool(out)                 # 1x1
        out = self.flatten(out)
        out = self.fc(out)
        return out



def denseNet_40_12():
    return DenseNet(12, 40, 1, 10, bottleneck=False)

def denseNet_100_12():
    return DenseNet(12, 100, 1, 10, bottleneck=False)

def denseNet_100_24():
    return DenseNet(24, 100, 1, 10, bottleneck=False)

def denseNetBC_100_12():
    return DenseNet(12, 100, 0.5, 10, bottleneck=True)

def denseNetBC_250_24():
    return DenseNet(24, 250, 0.5, 10, bottleneck=True)

def denseNetBC_190_40():
    return DenseNet(40, 190, 0.5, 10, bottleneck=True)



if __name__ == '__main__':

    from utils import count_parameters
    from beautifultable import BeautifulTable as BT

    densenet_40_12 = denseNet_40_12()
    densenet_100_12 = denseNet_100_12()
    densenet_100_24 = denseNet_100_24()
    densenetBC_100_12 = denseNetBC_100_12() 
    densenetBC_250_24 = denseNetBC_250_24()
    densenetBC_190_40 = denseNetBC_190_40()
    
    
    table = BT()
    table.append_row(['Model', 'Growth Rate', 'Depth', 'M. of Params'])
    table.append_row(['DenseNet', 12, 40, count_parameters(densenet_40_12)*1e-6])
    table.append_row(['DenseNet', 12, 100, count_parameters(densenet_100_12)*1e-6])
    table.append_row(['DenseNet', 24, 100, count_parameters(densenet_100_24)*1e-6])
    table.append_row(['DenseNet-BC', 12, 100, count_parameters(densenetBC_100_12)*1e-6])
    table.append_row(['DenseNet-BC', 24, 250, count_parameters(densenetBC_250_24)*1e-6])
    table.append_row(['DenseNet-BC', 40, 190, count_parameters(densenetBC_190_40)*1e-6])
    print(table)
        
    
    '''
    DenseNets implemented on the paper <https://arxiv.org/pdf/1608.06993.pdf>
    
    +-------------+-------------+-------+--------------+
    |    Model    | Growth Rate | Depth | M. of Params |
    +-------------+-------------+-------+--------------+
    |  DenseNet   |     12      |  40   |     1.02     |
    +-------------+-------------+-------+--------------+
    |  DenseNet   |     12      |  100  |     6.98     |
    +-------------+-------------+-------+--------------+
    |  DenseNet   |     24      |  100  |    27.249    |
    +-------------+-------------+-------+--------------+
    | DenseNet-BC |     12      |  100  |    0.769     |
    +-------------+-------------+-------+--------------+
    | DenseNet-BC |     24      |  250  |    15.324    |
    +-------------+-------------+-------+--------------+
    | DenseNet-BC |     40      |  190  |    25.624    |
    +-------------+-------------+-------+--------------+
    
    '''
