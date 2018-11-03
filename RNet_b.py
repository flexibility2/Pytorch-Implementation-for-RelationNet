# Reference "https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py"

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F



def conv3_3(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_channels=in_planes,out_channels=out_planes,stride=stride,kernel_size=3,padding=1,bias=False)  #??

# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self,in_planes,planes,stride=1,downsample=None):                                #  downsample 的条件
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3_3(in_planes,planes,stride=stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)                 # 另一位用 sigmod ?
#         self.conv2 = conv3_3(planes,planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample                  #  downsample -> nn, 考虑维度
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,in_planes,planes,stride=1,downsample=None):                                #  downsample 的条件
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3_3(planes,planes,stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)                 # 另一位用 sigmod ?
        self.conv3 = nn.Conv2d(planes,planes*4,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu =nn.ReLU(inplace=True)
        self.downsample = downsample                  #  downsample -> nn, 考虑维度

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RelationNetWork(nn.Module):
    def __init__(self):
        super(RelationNetWork,self).__init__()
        # self.inplanes = 64
        self.inplanes = 2*256
        # self.conv1 = nn.Conv2d(3,64,kernel_size=7,padding=3,stride=2,bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layer(Bottleneck,128,4,stride=2)
        self.layer2 = self._make_layer(Bottleneck,64,3,stride=2)
        self.avg = nn.AvgPool2d(kernel_size=4,stride=1)

        self.fc = nn.Sequential(
            nn.Linear(256,64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64,1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels     # m.kaiming
                m.weight.data.normal_(0,math.sqrt(2./n))
                # nn.init.kaiming_normal(m.weight.data,mode='fan_out')
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # nn.init.constant(m.weight.data,1)
                # nn.init.constant(m.bias.data,0)



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
        out = self.layer1(x)
        out = self.layer2(out)

        # print(out.size())

        out = self.avg(out) # N,256,1,1
        # print(out.size())
        out = out.view(out.size(0),-1)  # N ,256
        # print(out.size())
        out = self.fc(out)

        return out    # 256,1

# def resnet18():
#     model = ResNet(BasicBlock,[3,4,6])
#     return model

def test():
    x = Variable(torch.randn(5,256*2,14,14)).cuda()
    net = RelationNetWork().cuda()
    y =net(x)
    print(y.size())
    print(y)

if __name__=="__main__":
    test()


