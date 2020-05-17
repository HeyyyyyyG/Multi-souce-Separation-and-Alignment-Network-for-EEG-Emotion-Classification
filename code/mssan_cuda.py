from __future__ import print_function
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import numpy as np
import model_cuda as models
import scipy.io as scio
from torch.utils.data import TensorDataset,DataLoader
from sklearn import preprocessing

# Training settings
input_size=310
batch_size = 64
iteration = 3000
lr = 0.002
momentum = 0.9
cuda = True
seed = 8
log_interval = 30
l2_decay = 5e-4
class_num = 4
cls_p = 0.8

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
output = open('data.csv', 'w')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)

source = []

for i in range(11,14):
    #DataFile = '../../../../../2020/test/'+str(i)+'.mat'
    DataFile = '../2020/test/'+str(i)+'.mat'
    xi = (scio.loadmat(DataFile)['de_feature'])
    yi = scio.loadmat(DataFile)['label']
    xi = preprocessing.scale(xi)
    if i==11:
        x=xi
        y=yi
    else:
        x=np.concatenate((x,xi))
        y=np.concatenate((y,yi))
x = torch.Tensor(x)
x = torch.squeeze(x).float()
y = torch.Tensor(y)
y = torch.squeeze(y).long()
print(x.size(),y.size())
myset = TensorDataset(x,y)
target_train_loader = DataLoader(dataset=myset, batch_size=batch_size, drop_last=True, shuffle=False)
target_test_loader = DataLoader(dataset=myset, batch_size=batch_size, drop_last=False, shuffle=False)

def source_load(domain_num):
    if domain_num == 1:
        for i in range(1, 11):
            #DataFile = '../../../../../2020/train/'+str(i)+'.mat'
            DataFile = '../2020/train/'+str(i)+'.mat'
            xi = (scio.loadmat(DataFile)['de_feature'])
            yi = scio.loadmat(DataFile)['label']
            xi = preprocessing.scale(xi)
            if i==1:
                x=xi
                y=yi
            else:
                x=np.concatenate((x,xi))
                y=np.concatenate((y,yi))
        x = preprocessing.scale(x)
        x = torch.Tensor(x).float()
        x = torch.squeeze(x)
        y = torch.Tensor(y).long()
        y = torch.squeeze(y)
        print(x.size(),y.size())
        myset = TensorDataset(x,y)
        myloader = DataLoader(dataset=myset, batch_size=batch_size,drop_last=True, shuffle=False)
        source.append(myloader)
        
    if domain_num == 2:
        for i in range(1, 6):
            #DataFile = '../../../../../2020/train/'+str(i)+'.mat'
            DataFile = '../2020/train/'+str(i)+'.mat'
            xi = (scio.loadmat(DataFile)['de_feature'])
            yi = scio.loadmat(DataFile)['label']
            xi = preprocessing.scale(xi)
            if i==1:
                x=xi
                y=yi
            else:
                x=np.concatenate((x,xi))
                y=np.concatenate((y,yi))
        x = preprocessing.scale(x)
        x = torch.Tensor(x).float()
        x = torch.squeeze(x)
        y = torch.Tensor(y).long()
        y = torch.squeeze(y)
        print(x.size(),y.size())
        myset = TensorDataset(x,y)
        myloader = DataLoader(dataset=myset, batch_size=batch_size,drop_last=True, shuffle=False)
        source.append(myloader)
        
        for i in range(6, 11):
            #DataFile = '../../../../../2020/train/'+str(i)+'.mat'
            DataFile = '../2020/train/'+str(i)+'.mat'
            xi = (scio.loadmat(DataFile)['de_feature'])
            yi = scio.loadmat(DataFile)['label']
            xi = preprocessing.scale(xi)
            if i==6:
                x=xi
                y=yi
            else:
                x=np.concatenate((x,xi))
                y=np.concatenate((y,yi))
        x = preprocessing.scale(x)
        x = torch.Tensor(x).float()
        x = torch.squeeze(x)
        y = torch.Tensor(y).long()
        y = torch.squeeze(y)        
        print(x.size(),y.size())
        myset = TensorDataset(x,y)
        myloader = DataLoader(dataset=myset, batch_size=batch_size,drop_last=True, shuffle=False)
        source.append(myloader)
        
    if domain_num == 3:
        for i in range(1, 4):
            #DataFile = '../../../../../2020/train/'+str(i)+'.mat'
            DataFile = '../2020/train/'+str(i)+'.mat'
            xi = (scio.loadmat(DataFile)['de_feature'])
            yi = scio.loadmat(DataFile)['label']
            xi = preprocessing.scale(xi)
            if i==1:
                x=xi
                y=yi
            else:
                x=np.concatenate((x,xi))
                y=np.concatenate((y,yi))
        x = preprocessing.scale(x)
        x = torch.Tensor(x).float()
        x = torch.squeeze(x)
        y = torch.Tensor(y).long()
        y = torch.squeeze(y)
        print(x.size(),y.size())
        myset = TensorDataset(x,y)
        myloader = DataLoader(dataset=myset, batch_size=batch_size,drop_last=True, shuffle=False)
        source.append(myloader)
        
        for i in range(4, 7):
            #DataFile = '../../../../../2020/train/'+str(i)+'.mat'
            DataFile = '../2020/train/'+str(i)+'.mat'
            xi = (scio.loadmat(DataFile)['de_feature'])
            yi = scio.loadmat(DataFile)['label']
            xi = preprocessing.scale(xi)
            if i==4:
                x=xi
                y=yi
            else:
                x=np.concatenate((x,xi))
                y=np.concatenate((y,yi))
        x = preprocessing.scale(x)
        x = torch.Tensor(x).float()
        x = torch.squeeze(x)
        y = torch.Tensor(y).long()
        y = torch.squeeze(y)       
        print(x.size(),y.size())
        myset = TensorDataset(x,y)
        myloader = DataLoader(dataset=myset, batch_size=batch_size,drop_last=True, shuffle=False)
        source.append(myloader)
        
        for i in range(7, 11):
            #DataFile = '../../../../../2020/train/'+str(i)+'.mat'
            DataFile = '../2020/train/'+str(i)+'.mat'
            xi = (scio.loadmat(DataFile)['de_feature'])
            yi = scio.loadmat(DataFile)['label']
            xi = preprocessing.scale(xi)
            if i==7:
                x=xi
                y=yi
            else:
                x=np.concatenate((x,xi))
                y=np.concatenate((y,yi))
        x = preprocessing.scale(x)
        x = torch.Tensor(x).float()
        x = torch.squeeze(x)
        y = torch.Tensor(y).long()
        y = torch.squeeze(y)
        print(x.size(),y.size())
        myset = TensorDataset(x,y)
        myloader = DataLoader(dataset=myset, batch_size=batch_size,drop_last=True, shuffle=False)
        source.append(myloader)
        
    if domain_num == 5:
        for i in range(1, 3):
            #DataFile = '../../../../../2020/train/'+str(i)+'.mat'
            DataFile = '../2020/train/'+str(i)+'.mat'
            xi = (scio.loadmat(DataFile)['de_feature'])
            yi = scio.loadmat(DataFile)['label']
            xi = preprocessing.scale(xi)
            if i==1:
                x=xi
                y=yi
            else:
                x=np.concatenate((x,xi))
                y=np.concatenate((y,yi))
        x = preprocessing.scale(x)
        x = torch.Tensor(x).float()
        x = torch.squeeze(x)
        y = torch.Tensor(y).long()
        y = torch.squeeze(y)
        print(x.size(),y.size())
        myset = TensorDataset(x,y)
        myloader = DataLoader(dataset=myset, batch_size=batch_size,drop_last=True, shuffle=False)
        source.append(myloader)
        
        for i in range(3, 5):
            #DataFile = '../../../../../2020/train/'+str(i)+'.mat'
            DataFile = '../2020/train/'+str(i)+'.mat'
            xi = (scio.loadmat(DataFile)['de_feature'])
            yi = scio.loadmat(DataFile)['label']
            xi = preprocessing.scale(xi)
            if i==3:
                x=xi
                y=yi
            else:
                x=np.concatenate((x,xi))
                y=np.concatenate((y,yi))
        x = preprocessing.scale(x)
        x = torch.Tensor(x).float()
        x = torch.squeeze(x)
        y = torch.Tensor(y).long()
        y = torch.squeeze(y)       
        print(x.size(),y.size())
        myset = TensorDataset(x,y)
        myloader = DataLoader(dataset=myset, batch_size=batch_size,drop_last=True, shuffle=False)
        source.append(myloader)
        
        for i in range(5, 7):
            #DataFile = '../../../../../2020/train/'+str(i)+'.mat'
            DataFile = '../2020/train/'+str(i)+'.mat'
            xi = (scio.loadmat(DataFile)['de_feature'])
            yi = scio.loadmat(DataFile)['label']
            xi = preprocessing.scale(xi)
            if i==5:
                x=xi
                y=yi
            else:
                x=np.concatenate((x,xi))
                y=np.concatenate((y,yi))
        x = preprocessing.scale(x)
        x = torch.Tensor(x).float()
        x = torch.squeeze(x)
        y = torch.Tensor(y).long()
        y = torch.squeeze(y)
        print(x.size(),y.size())
        myset = TensorDataset(x,y)
        myloader = DataLoader(dataset=myset, batch_size=batch_size,drop_last=True, shuffle=False)
        source.append(myloader)
        
        for i in range(7, 9):
            #DataFile = '../../../../../2020/train/'+str(i)+'.mat'
            DataFile = '../2020/train/'+str(i)+'.mat'
            xi = (scio.loadmat(DataFile)['de_feature'])
            yi = scio.loadmat(DataFile)['label']
            xi = preprocessing.scale(xi)
            if i==7:
                x=xi
                y=yi
            else:
                x=np.concatenate((x,xi))
                y=np.concatenate((y,yi))
        x = preprocessing.scale(x)
        x = torch.Tensor(x).float()
        x = torch.squeeze(x)
        y = torch.Tensor(y).long()
        y = torch.squeeze(y)
        print(x.size(),y.size())
        myset = TensorDataset(x,y)
        myloader = DataLoader(dataset=myset, batch_size=batch_size,drop_last=True, shuffle=False)
        source.append(myloader)
        
        for i in range(9, 11):
            #DataFile = '../../../../../2020/train/'+str(i)+'.mat'
            DataFile = '../2020/train/'+str(i)+'.mat'
            xi = (scio.loadmat(DataFile)['de_feature'])
            yi = scio.loadmat(DataFile)['label']
            xi = preprocessing.scale(xi)
            if i==9:
                x=xi
                y=yi
            else:
                x=np.concatenate((x,xi))
                y=np.concatenate((y,yi))
        x = preprocessing.scale(x)
        x = torch.Tensor(x).float()
        x = torch.squeeze(x)
        y = torch.Tensor(y).long()
        y = torch.squeeze(y)
        print(x.size(),y.size())
        myset = TensorDataset(x,y)
        myloader = DataLoader(dataset=myset, batch_size=batch_size,drop_last=True, shuffle=False)
        source.append(myloader)
        
    if domain_num == 10:
        for i in range(1, 11):
            #DataFile = '../../../../2020/train/'+str(i)+'.mat'
            DataFile = '../2020/train/'+str(i)+'.mat'
            x = (scio.loadmat(DataFile)['de_feature'])
            y = scio.loadmat(DataFile)['label']
            x = preprocessing.scale(x)
            x = torch.Tensor(x).float()
            x = torch.squeeze(x)
            y = torch.Tensor(y).long()
            y = torch.squeeze(y)
            print(x.size(),y.size())
            myset = TensorDataset(x,y)
            myloader = DataLoader(dataset=myset, batch_size=batch_size,drop_last=True, shuffle=False)
            source.append(myloader)
            
def train(model,domain_num = 3):
    source_iter = []
    for i in range(domain_num):
        source_iter_tmp = iter(source[i])
        source_iter.append(source_iter_tmp)
    
    target_iter = iter(target_train_loader)
    correct = 0

    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate：", LEARNING_RATE)
        optimizer_params = []
        optimizer_params.append({'params': model.sharenet.parameters(),'lr': LEARNING_RATE})
        
        for j in range(domain_num):
            optimizer_params.append({'params': model.cls_fc_son[j].parameters(), 'lr': LEARNING_RATE})
            optimizer_params.append({'params': model.sonnet[j].parameters(), 'lr': LEARNING_RATE})
        optimizer = torch.optim.SGD(optimizer_params, lr=LEARNING_RATE , momentum=momentum, weight_decay=l2_decay)

        for j in range(domain_num):
            try:
                source_data, source_label = source_iter[j].next()
            except Exception as err:
                source_iter[j] = iter(source[j])
                source_data, source_label = source_iter[j].next()
            try:
                target_data, __ = target_iter.next()
            except Exception as err:
                target_iter = iter(target_train_loader)
                target_data, __ = target_iter.next()
            if cuda:
                source_data, source_label = source_data.to(device), source_label.to(device)
                target_data = target_data.to(device)
            source_data, source_label = Variable(source_data), Variable(source_label)
            target_data = Variable(target_data)
            optimizer.zero_grad()

            cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, mark=j+1, domain_num = domain_num)
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
            l1_loss = 10000*l1_loss
            mmd_loss = 10*mmd_loss
            cls_loss = cls_p *cls_loss
            loss = cls_loss + gamma * (mmd_loss + l1_loss)
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                print('Train source {} iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}\tgamma: {:.6f}'.format(
                    j+1, i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item(),gamma))

        if i % log_interval == 0:
            t_correct = test(model, domain_num)
            if t_correct > correct:
                correct = t_correct
                torch.save(model,'best_model_'+'iter='+str(i)+'_acc='+str(correct.item()/2553)+'.pth')
            print("number of source", domain_num, "max correct:", correct.item(), "\n")
            output.write(str(i)+','+str(t_correct.item()/2553)+'\n')

def test(model,domain_num):
    model.eval()
    test_loss = 0
    correct = 0
    correct_source = []
    correct2 = 0
    correct_source2 = []
    for i in range(domain_num):
        correct_source_tmp = 0
        correct_source.append(correct_source_tmp)
        correct_source2.append(correct_source_tmp)

    with torch.no_grad():
        for data, target in target_test_loader:
            '''
            mmd_loss1=0
            mmd_loss2=0
            mmd_loss3=0
            for i,(source1,source_label1) in enumerate(source[0]):
                cls_loss, mmd_loss, l1_loss = model(source1, data, source_label1, mark=1)
                mmd_loss1+=mmd_loss
                if i>=10:
                    break
            for i,(source2,source_label2) in enumerate(source[1]):
                cls_loss, mmd_loss, l1_loss = model(source2, data, source_label2, mark=2)
                mmd_loss2+=mmd_loss
                if i>=10:
                    break
            for i,(source3,source_label3) in enumerate(source[2]):
                cls_loss, mmd_loss, l1_loss = model(source3, data, source_label3, mark=3)
                mmd_loss3+=mmd_loss
                if i>=10:
                    break
            mmd_loss1,mmd_loss2,mmd_loss3 = 20/mmd_loss1, 20/mmd_loss2, 20/mmd_loss3
            weight = torch.nn.functional.softmax(torch.Tensor([mmd_loss1,mmd_loss2,mmd_loss3]),dim=0).numpy()
            print(weight)
            '''
            if cuda:
                data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            pred_source = model(data_src=data, mark=-1, domain_num=domain_num)

            target = target.squeeze(dim=-1)
            
            pred = 0
            pred2=0
            for j in range(domain_num):
                pred_source[j] = torch.nn.functional.softmax(pred_source[j], dim=1)
                pred += pred_source[j]#*weight[j]
                #pred2 +=pred_source[j]
            
            pred = pred/domain_num
            #print(pred)
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target,reduction='sum').item()  # sum up batch loss
            pred = pred.data.max(1)[1]  # get the index of the max log-probability
            #pred2 = pred2.data.max(1)[1]
            #print(pred)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            #correct2 += pred2.eq(target.data.view_as(pred2)).cpu().sum()

            for j in range(domain_num):
                pred = pred_source[j].data.max(1)[1]  # get the index of the max log-probability
                correct_source[j] += pred.eq(target.data.view_as(pred)).cpu().sum()


        test_loss /= len(target_test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) )\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print("correct_source",correct_source)
    return correct

if __name__ == '__main__':
    source_num = 2
    lr = lr/math.sqrt(source_num+1)
    cls_p = cls_p/math.sqrt(source_num+1)
    source_load(source_num)
    model = models.MFSAN(num_classes = class_num, domain_num = source_num)
    if cuda:
        model.cuda(device)
    print(model)
    s_time = time.time()
    train(model, domain_num = source_num)
    e_time = time.time()
    print("time:",(e_time-s_time))
    output.close()