import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from datetime import datetime
from PIL import Image
import sys
from torchvision import models, transforms,datasets
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import glob
from natsort import natsorted
import unicodedata
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import SubsetRandomSampler
import random
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torch.utils.data.dataset import Subset

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self,x):
        return x.view(self.shape)

def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.Tanh(),
            nn.Flatten(),
            nn.Linear(32*4*4, 128), nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 32*4*4), nn.Tanh(),
            Reshape(-1, 32, 4, 4),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Tanh(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Tanh(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        # for layer in self.encoder:
        #     x=layer(x)
        #     print(x.shape,layer)
        z = self.encoder(x)     
        x_pred = self.decoder(z)
        return x_pred, z
    
def Train(model,train_dataloader,device,optimizer,loss_fn):
    # savepoint読み出し
    # stdict_img = 'SegNet04.pth'
    # if os.path.isfile(stdict_img)==True:
    #     model.load_state_dict(torch.load(stdict_img))
    model.train()        
    train_loss_ae = 0.0

    for j, (data,label)in tqdm(enumerate(train_dataloader, 1),total=len(train_dataloader)):
        #dataには画像データが、labelにはそれに対応したラベルが含まれている
        images=data.to(device)
        optimizer.zero_grad()
        



        output,_ = model(images)


        loss_ae = loss_fn(output, images)

        loss_ae.backward()
        optimizer.step()


        train_loss_ae += loss_ae.item()

 

        # print('Loss at {} mini-batch: {}'.format(j, loss.item()/train_dataloader.batch_size))
    all_train_loss_ae=train_loss_ae/j

    return model,all_train_loss_ae

        
def Validation(model,valid_dataloader,device,loss_fn):
    model.eval()
    valid_loss_ae=0
    with torch.no_grad():
        for j,(data,label) in tqdm(enumerate(valid_dataloader,1),total=len(valid_dataloader)):
            images=data.to(device)
            


            output,image_hidden = model(images)
            loss = loss_fn(output, images)

            valid_loss_ae += loss.item() #1バッチのトータルのLossを加算していく




        # writer.add_scalars('loss/valid',{'valid':all_valid_loss},epoch)

    all_valid_loss_ae=valid_loss_ae/j #1epoch平均のLoss

    return model,all_valid_loss_ae





  
class ImageDataset:
    def __init__( self,dir,transform ): #呼び出されたとき、最初に行うこと
        self.transform = transform
        self.data = [] #画像が入ってる
        target_dir = os.path.join(dir, "*")
        for path in glob.glob(target_dir):
            self.data.append(path) 
    def __len__(self):
        """data の行数を返す。
        """
        return len(self.data)
    def __getitem__(self, index): #dataloaderを作ったときにできるやつ
        """サンプルを返す。
        """
        img_path = self.data[index] #適当な画像を取ってくる
        img = Image.open(img_path).convert("RGB") #img_pathの画像を開く
        img = self.transform(img) #transformする

        return img, img_path
def main():
    #dataset作成
    
    size=32
    transform = transforms.Compose([transforms.Resize((size, size)),transforms.RandomResizedCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()])

    # ,
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    train_dataset = ImageDataset( "dataset/image/train",transform)
    valid_dataset = ImageDataset("dataset/image/valid",transform)
    imagenet_train_dataset=datasets.ImageFolder("../../../../Pictures/tiny-imagenet-200/train", transform=transform) #ImageNetのデータセット（容量がでかいから研究室PCのローカルに存在している）
    imagenet_valid_dataset=datasets.ImageFolder("../../../../Pictures/tiny-imagenet-200/val", transform=transform)
    #dataloader作成


    # データセットの長さを取得
    dataset_length = len(imagenet_train_dataset)

    # サンプル数を計算
    sample_size = dataset_length // 10

    # データセットからサンプルするインデックスをランダムに選択
    indices = list(range(dataset_length))
    random.shuffle(indices)
    sampled_indices = indices[:sample_size]

    # サンプルされたインデックスを使ってサブセットデータセットを作成
    sampled_dataset = torch.utils.data.Subset(imagenet_train_dataset, sampled_indices)

    # サンプルされたデータセットを使用してデータローダを作成
    sampled_dataloadert = DataLoader(sampled_dataset, batch_size=128, shuffle=True, drop_last=True)


    dataset_length = len(imagenet_valid_dataset)
    sample_size = dataset_length // 10
    indices = list(range(dataset_length))
    random.shuffle(indices)
    sampled_indices = indices[:sample_size]
    sampled_dataset = torch.utils.data.Subset(imagenet_valid_dataset, sampled_indices)
    sampled_dataloaderv = DataLoader(sampled_dataset, batch_size=128, shuffle=True, drop_last=True)
    # train_dataloader = DataLoader(imagenet_train_dataset, batch_size=128, shuffle=True,drop_last=True)
    # valid_dataloader = DataLoader(imagenet_valid_dataset, batch_size=128, shuffle=True,drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=True,drop_last=True)
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    n_sample = len(trainset)
    train_size = int(n_sample * 0.8)
    # print("train data: {}, validation data: {}".format(train_size, n_sample-train_size))
    subset1_indices = list(range(0,train_size))
    subset2_indices = list(range(train_size,n_sample))
    train_dataset = Subset(trainset, subset1_indices)
    val_dataset   = Subset(trainset, subset2_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model=CAE().to(device)
    imgfile='model/cae2'
    model.load_state_dict(torch.load(imgfile))
    optimizer = optim.Adam(model.parameters())
    # スケジューラーを定義
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) 
    criterion=nn.BCELoss()
    loss_fn=nn.MSELoss()
    run_epoch = 1000
    val_ae_save=10000
    # val_gan_save=300000
    # savepoint読み出し
    # stdict_img = 'model/Imagenetmodel'
    # if os.path.isfile(stdict_img)==True:
    #     model.load_state_dict(torch.load(stdict_img))
    writer = SummaryWriter(log_dir="log/mycae")
    for epoch in range(1, run_epoch + 1):
        model,all_train_ae=Train(model,train_dataloader,device,optimizer,loss_fn)
        model,all_valid_ae=Validation(model,valid_dataloader,device,loss_fn)   
        writer.add_scalars('loss/ae/',{'train':all_train_ae,'valid':all_valid_ae} ,epoch)
        # writer.add_scalars('loss/gan/',{'train':all_train_gan,'valid':all_valid_gan} ,epoch)
        print('Train_Loss_ae Epoch{}: {}'.format(epoch,all_train_ae))
        print('Validation_Loss_ae Epoch{}:{}'.format(epoch,all_valid_ae))
        # print('Train_Loss_ae Epoch{}: {}'.format(epoch,all_train_ae))
        # print('Validation_Loss_ae Epoch{}:{}'.format(epoch,all_valid_ae))
        if(val_ae_save>=all_valid_ae):
            torch.save(model.state_dict(), "model/mycae")
            val_ae_save=all_valid_ae
            print("＿＿＿＿＿＿＿model更新＿＿＿＿＿＿＿＿＿")
        torch.save(model.state_dict(),"model/mycaecheckpoint")
        scheduler.step()
    writer.close()

    

if __name__ == '__main__':
    main()