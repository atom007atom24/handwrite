
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import mymodel
import numpy as np
from preprocess import findBorderContours,transMNIST,showResults
from add_data import load_mnist,Mydataset
import logging


#定义一些超参数，
batch_size = 64
learning_rate = 1e-2
num_epoches = 20

#先定义数据预处理
def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化
    x = x.reshape((-1,)) # 拉平
    x = torch.from_numpy(x)
    return x
   
def train(kfold):
    # 定义logging输出的格式和内容
    logging.basicConfig(filename='log.txt',
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="[%a %d %Y %H:%M:%S]"
                   )
    imgs,labels=load_mnist()
    for j in range(kfold):

        logging.info('Kfold={}'.format(j))

        train_dataset=Mydataset(imgs,labels,transform=data_tf,k=kfold,ki=j,typ='train')
        test_dataset=Mydataset(imgs,labels,transform=data_tf,k=kfold,ki=j,typ='test')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False)


#导入网络，定义损失函数和优化方法，模型已在net.py里面定义过了
        model = mymodel.MLP_3Layer(28*28, 300, 100, 10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# 开始训练
        for n in range(num_epoches):
            train_loss = 0
            train_acc = 0
            eval_loss = 0
            eval_acc = 0
            model.train()
            for im, label in train_loader: 
                im = Variable(im)
                label = Variable(label)
        # 前向传播
                out = model(im)
                loss = criterion(out, label.long())
        # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # 记录误差
                train_loss += loss.item()#对张量对象可以用item得到元素值
        # 计算分类的准确率
                _, pred = out.max(1)
                num_correct = (pred == label).sum().item()       #对张量对象可以用item得到元素值
                acc = num_correct / im.shape[0]    #预测正确数/总数，对于这个程序由于小批量设置的是64，所以总数为64
                train_acc += acc                   #计算总的正确率，以便求平均值
        
            logging.info('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}'
          .format(n, train_loss / len(train_loader), train_acc / len(train_loader)))

          

    #进入测试阶段
            model.eval()
            for im, label in test_loader:
        # print(im.shape, im.size)
                im = Variable(im)
                label = Variable(label)
                out = model(im)
                loss = criterion(out, label.long())
        # 记录误差
                eval_loss += loss.item()
        # 记录准确率
                _, pred = out.max(1)
                num_correct = (pred == label).sum().item()
                acc = num_correct / im.shape[0]
                eval_acc += acc

            logging.info('epoch: {}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(n, eval_loss / len(test_loader), eval_acc / len(test_loader)))

        #保存模型
            torch.save(model.state_dict(), 'model\\model_mnist_{}.pth'.format(j))

# 预测手写数字
def predict(imgData):
    model = mymodel.MLP_3Layer(28*28, 300, 100, 10)
    model.load_state_dict(torch.load('model\\model_mnist_0.pth'))
    result_number = []
    model.eval()
    img = data_tf(imgData)
    img = img.reshape(-1,784)
    print(img.shape)
    test_xmw = DataLoader(img)
    for a in test_xmw:
        img = Variable(a)
        out = model(img)
        _,n = out.max(1)
        result_number.append(n.numpy())
    return result_number

def shom_result():
    path = 'test\\test3.png'          #获取图像地址
    re_path='test\\result3.png'
    # print(path)
    borders = findBorderContours(path)    #获取数字边界并截取成单个数字图像
    imgData = transMNIST(path, borders)   #转变成mnist格式图像
    results = predict(imgData)                #进行预测
    showResults(path, borders, re_path,results)   #图像展示
# 结构体
if __name__ =="__main__":
    # train(5)
    shom_result()