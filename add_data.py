from preprocess import findBorderContours,transMNIST
import numpy as np
import torch
import os
from torch.utils.data import Dataset

import numpy as np

def load_mnist():

    """Load MNIST data from `path`"""
    labels_path = 'data\\MNIST\\raw\\train-labels-idx1-ubyte'
    images_path = 'data\\MNIST\\raw\\train-images-idx3-ubyte'

    with open(labels_path, 'rb') as lbpath:
        # magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)[8:]

    with open(images_path, 'rb') as imgpath:
        # magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8)[16:].reshape((-1,28,28))

    labels2_path = 'data\\MNIST\\raw\\t10k-labels-idx1-ubyte'
    images2_path = 'data\\MNIST\\raw\\t10k-images-idx3-ubyte'

    with open(labels2_path, 'rb') as lb2path:
        labels = np.concatenate((labels,np.fromfile(lb2path,dtype=np.uint8)[8:]),axis=0)

    with open(images2_path, 'rb') as img2path:
        images=np.concatenate((images,np.fromfile(img2path,dtype=np.uint8)[16:].reshape((-1,28,28))),axis=0)
        
    return images, labels

class Mydataset(Dataset):
    def __init__(self,minst_img,label,transform,k,ki,typ) -> None:
        super().__init__()
        self.imgs=minst_img
        self.label=label
        for i in range(1):
            path='data\\data{}.png'.format(i)
            # path='test1.png'
            borders = findBorderContours(path)    
            imgdata = transMNIST(path, borders).reshape((-1,28,28)) 
            for k in range(10):
                self.imgs=np.concatenate((self.imgs,imgdata),axis=0)
                self.label=np.concatenate((self.label,np.array([i]*imgdata.shape[0])))
                
        leng = self.imgs.shape[0]
        every_z_len = leng // k
        if typ == 'test':
            self.trimage = self.imgs[every_z_len * ki : every_z_len * (ki+1)]
            self.trlabel = self.label[every_z_len * ki : every_z_len * (ki+1)]
        elif typ == 'train':
            self.trimage = np.concatenate((self.imgs[: every_z_len * ki],self.imgs[every_z_len * (ki+1) :]),axis=0)
            self.trlabel = np.concatenate((self.label[: every_z_len * ki],self.label[every_z_len * (ki+1) :]),axis=0)

        self.transform=transform

    def __getitem__(self, idx):
        return self.transform(self.imgs[idx]),self.label[idx]

    def __len__(self):
        return self.imgs.shape[0]
if __name__ == '__main__':

    img,label=load_mnist('data\\MNIST\\raw')
    data=Mydataset(img,label)
    print(len(data))


# def add_dataset(train_dataset,test_dataset):
#     train_data,test_data=train_dataset.data.numpy(),test_dataset.data.numpy()
#     train_label,test_label=list(train_dataset.targets.numpy()),list(test_dataset.targets.numpy())
#     for i in range(10):
#         trpath='trdata{}'.format(i)
#         # trpath='test1.png'
#         trborders = findBorderContours(trpath)    
#         trimgdata = transMNIST(trpath, trborders).reshape((-1,28,28)) 
#         tepath='tedata{}'.format(i)
#         # tepath='test1.png'
#         teborders = findBorderContours(tepath)    
#         teimgdata = transMNIST(tepath, teborders).reshape((-1,28,28)) 

#         train_data=np.concatenate((train_data,trimgdata),axis=0)
#         test_data=np.concatenate((test_data,teimgdata),axis=0)
#         trlabel=[i]*trimgdata.shape[0]
#         telabel=[i]*teimgdata.shape[0]
#         train_label+=trlabel
#         test_label+=telabel
#     train_dataset.data=torch.from_numpy(train_data)
#     test_dataset.data=torch.from_numpy(test_data)
#     train_dataset.targets=torch.from_numpy(np.array(train_label))
#     test_dataset.targets=torch.from_numpy(np.array(test_label))
#     print(len(train_dataset))
#     return train_dataset,test_dataset


        





