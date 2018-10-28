import torch
import torch.nn as nn
from MiniImagenet2 import MiniImagenet
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from RNet import RelationNetWork
from embed import CNNEncoder
import numpy as np
import scipy as sp
import scipy.stats



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h


def main():
    n_way = 5
    k_shot = 1
    k_query = 15
    batchsz = 5

    mdfile1 = './ckpy/feature-%d-way-%d-shot.pkl' %(n_way,k_shot)
    mdfile2 = './ckpy/relation-%d-way-%d-shot.pkl' %(n_way,k_shot)

    mini = MiniImagenet('./mini-imagenet/', mode='test', n_way=n_way, k_shot=k_shot, k_query=k_query, batchsz=2000, resize=84)   # 训练是，batchsz = 200
    db = DataLoader(mini,batch_size=batchsz,num_workers=0,pin_memory=False)

    feature_embed = CNNEncoder().cuda()
    Relation_score = RelationNetWork(64, 8).cuda()  # relation_dim == 8 ??


    if os.path.exists(mdfile1):
        print("file1-feature exit...")
        feature_embed.load_state_dict(torch.load(mdfile1))
    if os.path.exists(mdfile2):
        print("f2-relation exit...")
        Relation_score.load_state_dict(torch.load(mdfile2))



    for ts in range(3):
        correct = 0
        total = 0
        accuarcy = 0
        accuarcies = []

        for i,batch in enumerate(db):
            support_x = Variable(batch[0]).cuda()   # [batch_size, n_way*k_shot, c , h , w]
            support_y = Variable(batch[1]).cuda()
            query_x = Variable(batch[2]).cuda()
            query_y = Variable(batch[3]).cuda()     # [b, n_way * q ]

            bh,set1,c,h,w = support_x.size()
            set2 = query_x.size(1)

            support_xf = feature_embed(support_x.view(bh*set1,c,h,w)).view(bh,set1,64,19,19)
            query_xf = feature_embed(query_x.view(bh*set2,c,h,w)).view(bh,set2,64,19,19)

            support_xf = support_xf.unsqueeze(1).expand(bh,set2,set1,64,19,19)
            query_xf = query_xf.unsqueeze(2).expand(bh,set2,set1,64,19,19)

            comb = torch.cat((support_xf,query_xf),dim=3)

            score = Relation_score(comb.view(bh*set2*set1,64*2,19,19)).view(bh,set2,set1)

            score_np = score.cpu().data.numpy()
            support_y_np = support_y.cpu().data.numpy()

            pred = []
            for ii,bb in enumerate(score_np):
                for jj,bset in enumerate(bb):
                    sim = []
                    for way in range(n_way):
                        sim.append(np.sum(bset[way*k_shot:(way+1)*k_shot]))
                    idx = np.array(sim).argmax()
                    pred.append(support_y_np[ii,k_shot*idx])
            pred = Variable(torch.from_numpy(np.array(pred).reshape(bh,set2))).cuda()

            correct += torch.eq(pred,query_y).sum()
            total += query_y.size(0)*query_y.size(1)
            accuarcy = correct/total
            print("epoch",ts,"i-batch",i,"acc:",accuarcy)
            accuarcies.append(accuarcy)

        test_accuracy, h = mean_confidence_interval(accuracies)
        print("test accuracy:", test_accuracy, "h:", h)







