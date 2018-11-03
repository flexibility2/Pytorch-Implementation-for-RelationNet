# Reference "https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py"

import torch
import torch.nn as nn
from torch.autograd import Variable
import math


def conv3_3(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_channels=in_planes,out_channels=out_planes,stride=stride,kernel_size=3,padding=1,bias=False)  #??

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,in_planes,planes,stride=1,downsample=None):                                #  downsample 的条件
        super(BasicBlock, self).__init__()
        self.conv1 = conv3_3(in_planes,planes,stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)                 # 另一位用 sigmod ?
        self.conv2 = conv3_3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample                  #  downsample -> nn, 考虑维度

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,block,layers):
        super(ResNet,self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,padding=3,stride=2,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels     # m.kaiming
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def _make_layer(self,block,planes,blocks,stride = 1):
        downsample = None
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        out= self.conv1(x)
        out= self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out= self.layer1(out)
        out= self.layer2(out)
        out= self.layer3(out)

        return out

def resnet18():
    model = ResNet(BasicBlock,[3,4,6])
    return model

def test():
    x = Variable(torch.randn(5,3,224,224)).cuda()
    net = resnet18().cuda()
    y =net(x)
    print(y.size())

if __name__=="__main__":
    test()


