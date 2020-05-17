import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch.nn.functional as F
import torch

input_size = 310

class MFSAN(nn.Module):
    def __init__(self, num_classes, domain_num):
        super(MFSAN, self).__init__()
        self.sharenet = nn.Sequential(
                nn.Linear(310,310),
                nn.LeakyReLU(),
            )
        self.sonnet = []
        for i in range(domain_num):
            net = nn.Sequential(
                nn.Linear(310,310),
                nn.LeakyReLU(),
                nn.Linear(310,310),
                nn.LeakyReLU(),
                nn.Linear(310,310),
                nn.LeakyReLU(),
                nn.Linear(310,310),
                nn.LeakyReLU(),
            ).cuda()
            self.sonnet.append(net)
        
        self.cls_fc_son = []
        for i in range(domain_num):
            net = nn.Linear(310*2, num_classes).cuda()
            self.cls_fc_son.append(net)

        #self.w1 = torch.tensor([1], requires_grad=True, dtype=torch.float32)
        #self.w2 = torch.tensor([1], requires_grad=True, dtype=torch.float32)

    def forward(self, data_src, data_tgt = 0, label_src = 0, mark = 1, domain_num = 3):
        mmd_loss = 0
        #print(self.w1.item(),self.w2.item())
        if mark!=-1:#self.training == True:
            data_tgt_share = self.sharenet(data_tgt)
            data_src_share = self.sharenet(data_src)

            similarity_loss = torch.mean( torch.abs(data_tgt_share-data_src_share))
            #print(similarity_loss)
            data_tgt_son = []
            pred_tgt_son = []
            for i in range(domain_num):
                data_tgt_son_tmp = self.sonnet[i](data_tgt)
                #print(data_tgt_son_tmp.size(),data_tgt_share.size())
                #pred_tgt_son_tmp = self.cls_fc_son[i](self.w1*data_tgt_son_tmp + self.w2*data_tgt_share)
                pred_tgt_son_tmp = self.cls_fc_son[i](torch.cat((data_tgt_son_tmp,data_tgt_share),1))
                data_tgt_son.append(data_tgt_son_tmp)
                pred_tgt_son.append(pred_tgt_son_tmp)

            
            for j in range(domain_num):
                if mark == j+1:
                    #######
                    data_src = self.sonnet[j](data_src)
                    diff_loss = torch.mean( torch.abs(data_src-data_src_share))
                    #print("diff-sim",diff_loss-similarity_loss)
                    mmd_loss += mmd.mmd(data_src, data_tgt_son[j])


                    l1_loss = 0
                    for t in range(domain_num):
                        l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son[j],dim=1)
                                              -torch.nn.functional.softmax(data_tgt_son[t], dim=1)) )
                    #pred_src = self.cls_fc_son[j](self.w1*data_src + self.w2*data_src_share)
                    pred_src = self.cls_fc_son[j](torch.cat((data_src,data_src_share),1))
                    cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src,reduction='sum')#-10*diff_loss+10*similarity_loss
                    return cls_loss, mmd_loss, l1_loss / 2

        else:
            data = self.sharenet(data_src)
            fea_son = []
            pred = []
            for i in range(domain_num):
                fea_son_tmp = self.sonnet[i](data_src)
                #pred_tmp = self.cls_fc_son[i](self.w1*fea_son_tmp + self.w2*data)
                pred_tmp = self.cls_fc_son[i](torch.cat((fea_son_tmp,data),1))
                fea_son.append(fea_son_tmp)
                pred.append(pred_tmp)

            return pred
