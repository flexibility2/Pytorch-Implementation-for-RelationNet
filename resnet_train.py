import torch
import torch.nn as nn
from utils import weight_init
from RNet_resnet import RelationNetWork
#from embed import CNNEncoder
from MiniImagenet2 import MiniImagenet
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import numpy as np
from logger import Logger
from ResNet_feature import resnet18
from torch.optim.lr_scheduler import StepLR

LOG_DIR = './log'
logger = Logger(LOG_DIR)

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
def main():

    n_way = 5
    k_shot = 1
    k_query = 5    #  5-way-5shot
    batchsz = 3
    best_acc = 0
    mdfile1 = './ckpy/res_feature-%d-way-%d-shot.pkl' %(n_way,k_shot)
    mdfile2 = './ckpy/res_relation-%d-way-%d-shot.pkl' %(n_way,k_shot)
    # feature_embed = CNNEncoder().cuda()
    # print(torch.cuda.is_available())
    feature_embed = resnet18().cuda()

    Relation_score = RelationNetWork(64, 8).cuda()  # relation_dim == 8 ??

    Relation_score.apply(weight_init)

    feature_optim = torch.optim.Adam(feature_embed.parameters(), lr=0.001)
    relation_opim = torch.optim.Adam(Relation_score.parameters(), lr=0.001)

    feature_optim_scheduler = StepLR(feature_optim,step_size=10,gamma=0.5)    # 1-shot 1w , 5-shot 5k
    relation_opim_scheduler = StepLR(relation_opim,step_size=10,gamma=0.5)

    loss_fn = torch.nn.MSELoss().cuda()

    if os.path.exists(mdfile1):
         print("load mdfile1...")
         feature_embed.load_state_dict(torch.load(mdfile1))
    if os.path.exists(mdfile2):
         print("load mdfile2...")
         Relation_score.load_state_dict(torch.load(mdfile2))

    for epoch in range(100):
        feature_optim_scheduler.step(epoch)                     #  降低学习率
        relation_opim_scheduler.step(epoch)

        mini = MiniImagenet('./mini-imagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query, batchsz=6000, resize=224)  #38400
        db = DataLoader(mini,batch_size=batchsz,shuffle=True,num_workers=0,pin_memory=False)  # 64 , 5*(1+15) , c, h, w
        mini_val = MiniImagenet('./mini-imagenet/', mode='val', n_way=n_way, k_shot=k_shot, k_query=k_query, batchsz=200, resize=224)   #9600
        db_val = DataLoader(mini_val,batch_size=batchsz,shuffle=True,num_workers=0,pin_memory=False)


        for step,batch in enumerate(db):
            support_x = Variable(batch[0]).cuda()   # [batch_size, n_way*(k_shot+k_query), c , h , w]
            support_y = Variable(batch[1]).cuda()
            query_x = Variable(batch[2]).cuda()
            query_y = Variable(batch[3]).cuda()

            bh,set1,c,h,w = support_x.size()
            set2 = query_x.size(1)

            feature_embed.train()
            Relation_score.train()

            # support_xf = feature_embed(support_x.view(bh*set1,c,h,w)).view(bh,set1,64,19,19)                 # 在 test 的 时候 重复
            support_xf = feature_embed(support_x.view(bh * set1, c, h, w)).view(bh, set1, 256, 14, 14)
            # query_xf = feature_embed(query_x.view(bh*set2,c,h,w)).view(bh,set2,64,19,19)
            query_xf = feature_embed(query_x.view(bh*set2,c,h,w)).view(bh,set2,256,14,14)


            # print("query_f:", query_xf.size())

            # support_xf = support_xf.unsqueeze(1).expand(bh,set2,set1,64,19,19)
            support_xf = support_xf.unsqueeze(1).expand(bh,set2,set1,256,14,14)

            query_xf = query_xf.unsqueeze(2).expand(bh,set2,set1,256,14,14)


            comb = torch.cat((support_xf,query_xf),dim=3)       # bh,set2,set1,2c,h,w
            # print(comb.is_cuda)
            # print(comb.view(bh*set2*set1,2*64,19,19).is_cuda)
            # print(comb.size())
            score = Relation_score(comb.view(bh*set2*set1,2*256,14,14)).view(bh,set2,set1,1).squeeze(3)

            support_yf = support_y.unsqueeze(1).expand(bh,set2,set1)
            query_yf = query_y.unsqueeze(2).expand(bh,set2,set1)
            label = torch.eq(support_yf,query_yf).float()

            feature_optim.zero_grad()
            relation_opim.zero_grad()

            loss = loss_fn(score,label)
            loss.backward()

            torch.nn.utils.clip_grad_norm(feature_embed.parameters(),0.5)             # 梯度裁剪？ 降低学习率？
            torch.nn.utils.clip_grad_norm(Relation_score.parameters(),0.5)

            feature_optim.step()
            relation_opim.step()

            # if step%100==0:
            #     print("step:",epoch+1,"train_loss: ",loss.data[0])
            logger.log_value('resnet_{}-way-{}-shot loss：'.format(n_way, k_shot),loss.data[0])

            if step%200==0:
                print("---------test--------")

                total_correct = 0
                total_num = 0
                accuracy = 0
                for j,batch_test in enumerate(db_val):
                    # if (j%100==0):
                    #     print(j,'-------------')
                    support_x = Variable(batch_test[0]).cuda()
                    support_y = Variable(batch_test[1]).cuda()
                    query_x = Variable(batch_test[2]).cuda()
                    query_y = Variable(batch_test[3]).cuda()

                    bh,set1,c,h,w = support_x.size()
                    set2 = query_x.size(1)

                    feature_embed.eval()
                    Relation_score.eval()

                    support_xf = feature_embed(support_x.view(bh*set1,c,h,w)).view(bh,set1,256,14,14)                 # 在 test 的 时候 重复
                    query_xf = feature_embed(query_x.view(bh*set2,c,h,w)).view(bh,set2,256,14,14)

                    support_xf = support_xf.unsqueeze(1).expand(bh,set2,set1,256,14,14)
                    query_xf = query_xf.unsqueeze(2).expand(bh,set2,set1,256,14,14)

                    comb = torch.cat((support_xf,query_xf),dim=3)       # bh,set2,set1,2c,h,w
                    score = Relation_score(comb.view(bh*set2*set1,2*256,14,14)).view(bh,set2,set1,1).squeeze(3)

                    rn_score_np = score.cpu().data.numpy()                                                      # 转numpy cpu
                    pred = []
                    support_y_np = support_y.cpu().data.numpy()

                    for ii,tb in enumerate(rn_score_np):
                        for jj,tset in enumerate(tb):
                            sim = []
                            for way in range(n_way):
                                sim.append(np.sum(tset[way*k_shot:(way+1)*k_shot]))

                            idx = np.array(sim).argmax()
                            pred.append(support_y_np[ii,idx*k_shot])                 # 同一个类标签相同 ，注意还有batch维度
                                                                                     # ×k_shot是因为，上一个步用sum将k_shot压缩了

                    #此时的pred.size = [b.set2]
                    #print("pred.size=", np.array(pred).shape)
                    pred = Variable(torch.from_numpy(np.array(pred).reshape(bh,set2))).cuda()
                    correct = torch.eq(pred,query_y).sum()

                    total_correct += correct.data[0]
                    total_num += query_y.size(0)*query_y.size(1)

                accuracy = total_correct/total_num
                logger.log_value('acc : ', accuracy)

                print("epoch:",epoch,"acc:",accuracy)
                if accuracy>best_acc:
                    print("-------------------epoch",epoch,"step:",step,"acc:",accuracy,"---------------------------------------")
                    best_acc = accuracy
                    torch.save(feature_embed.state_dict(),mdfile1)
                    torch.save(Relation_score.state_dict(),mdfile2)
            logger.step()
            #if step% == 0 and step != 0:
             #   print("%d-way %d-shot %d batch | epoch:%d step:%d, loss:%f" %(n_way,k_shot,batchsz,epoch,step,loss.cpu().data[0]))
            #logger.step()
if __name__=='__main__':
    main()
