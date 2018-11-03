import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from MiniImagenet2 import MiniImagenet

class RelationNetWork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RelationNetWork,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(256*2,128,kernel_size=3,stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )


        self.layer2 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1),
            nn.BatchNorm2d(64),                      # affine = 1 ?
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(input_size*2*2,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)          # [bh*set1*set2,64,3,3]
        # print(x.size())
        x = x.view(x.size(0),-1)   # [ bh*set1*set2,64*3*3]
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))   # [bh*set1*set2,1]

        return x



if __name__=='__main__':
    x = Variable(torch.randn(5*3,256*2,14,14)).cuda()
    # mini = MiniImagenet('./mini-imagenet/', mode='train', n_way=5, k_shot=1, k_query=15, batchsz=30, resize=84)
    net = RelationNetWork(64, 8).cuda()
    # for i,m in enumerate(mini):
    #     support_x, support_y, query_x, query_y = m
    #     print(i,support_x.size())
    #     support_x = Variable(support_x).cuda()
    #     ans = net(support_x)
    #     print(ans)
    # mdfile = './ckpy/%d-way-%d-shot.pkl'%(2,3)
    # way = 5
    # shot = 1

    # if os.path.exists(mdfile):
    #     print("exit")
    #     net.load_state_dict(torch.load(mdfile))

    # torch.save(net.state_dict(),mdfile)

    y = net(x)
    print(y.size())
    print(y)