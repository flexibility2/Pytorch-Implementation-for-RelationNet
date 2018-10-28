import torch
import torch.nn as nn
from MiniImagenet2 import MiniImagenet

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,3),
            nn.BatchNorm2d(64),                 # 作者moment设为1
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,64,stride=1,padding=1,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64,64,stride=1,padding=1,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self,x):
        x = self.layer1(x)
        # print("layer 1",x.size())
        x = self.layer2(x)
        # print("layer 2",x.size())
        x = self.layer3(x)
        # print("layer 3",x.size())

        x = self.layer4(x)

        return x                        # [bz * (way*(s+q)), 64, 19,19]




from torch.autograd import Variable

if __name__ == "__main__":
    mini = MiniImagenet(root='./mini-imagenet/', mode='train', batchsz=100, n_way=5, k_shot=5, k_query=5, resize=84, startidx=0)
    for i,m in enumerate(mini):
        support_x, support_y, query_x, query_y = m
        print(i,support_x.size())
        support_x = Variable(support_x).cuda()
        net = CNNEncoder().cuda()
        ans = net(support_x)
        print(ans.size())
        print("--------")