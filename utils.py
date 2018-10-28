import torch
import torch.nn as nn
import math

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv")!=-1:
        n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
        m.weight.data.normal_(0,math.sqrt(2./n))
        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find("BatchNorm")!=-1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    elif classname.find("Linear")!=-1:
        m.weight.data.normal_(0,0.01)
        m.bias.data = torch.ones(m.bias.data.size()).cuda()