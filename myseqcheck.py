import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import sys
import pandas as pd
from PIL import Image
from natsort import natsorted
import unicodedata
import glob
import os
import torchvision
from torchvision import models, transforms,datasets
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torch.utils.data.dataset import Subset
import japanize_matplotlib  # 日本語をサポートするためのライブラリ
from sklearn.decomposition import PCA
import math
import matplotlib.collections as mc

# from myseq import Lang
# from myseq import Encoder
# from myseq import Decoder
# from myseq import tensorFromSentence

# from segnet import SegNet
# from segnet import Reshape
# from segnet import ConvAutoencoder
# from vgg16autoencoder import VGG16Autoencoder
# from aegan import AE_GAN
from cae import CAE

from myseq import ImageLang
from seq2seq import Encoder
from seq2seq import Decoder
from seq2seq import tensorFromSentence

def calcAcc(word,phoneme):
    acc=0
    total=0
    word=[x.replace("<EOS>","") for x in word]
    word=''.join(word)

    onomatope=phoneme.replace(" ","")

    if word==onomatope:
        acc +=1
    print(f"predict is {word} answer is {onomatope}")
    total+=1

    return acc/total

def img_to_ono(img,decoder):
    
    decoder_input      = torch.tensor([[SOS_token]], device=device)  # SOS
    decoder_hidden     = img
    decoded_words      = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden )
            
        topv, topi = decoder_output.data.topk(1)
        if topi.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[topi.item()])

        decoder_input = topi.squeeze().detach()
    return decoded_words
def ono_to_ono(sentence,encoder,decoder):
    
    input_tensor   = tensorFromSentence(lang, sentence)
    input_length   = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden()
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input      = torch.tensor([[SOS_token]], device=device)  # SOS
    decoder_hidden     = encoder_hidden
    decoded_words      = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden )
            
        topv, topi = decoder_output.data.topk(1)
        if topi.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[topi.item()])

        decoder_input = topi.squeeze().detach()
    return decoded_words,encoder_hidden
def generate_and_save_image(pic, epoch,num,label):
    fig = plt.figure(figsize=(4, 4))
    for i in range(1):
        plt.subplot(1, 1, i + 1)
        print(pic.shape,"picdesu")
        plt.imshow(pic.cpu().data[i, :, :, :].permute(1, 2, 0))
        name=os.path.basename(label[i])
        name,_=os.path.splitext(name)
        plt.title(name)
        plt.axis('off')
    if num == 0:
        plt.savefig("./output/single/original{}.jpg".format(epoch))
    else:
        plt.savefig('./output/single/image{}.png'.format(epoch))

def generate_and_save_images(pic, epoch,num,label):
    fig = plt.figure(figsize=(4, 4))
    for i in range(8):
        plt.subplot(4, 2, i + 1)
        plt.imshow(pic.cpu().data[i, :, :, :].permute(1, 2, 0))

        plt.title(label[i])
        plt.axis('off')
    if num == 0:
        plt.savefig("./output/batch/original{}.jpg".format(epoch))
    elif num ==1:
        plt.savefig('./output/batch/image{}.png'.format(epoch))
    else:
        plt.savefig('./output/batch/onomatope{}.png'.format(epoch))
if __name__ == '__main__':
    with torch.no_grad():
        SOS_token = 0
        EOS_token = 1
        device = "cuda" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size =8
        num=40 #入出力として使える音素の数=データセット内の.n_wordsに等しい
        embedding_size = 128
        hidden_size   = 128
        max_length=20
        size=32
        
        #データセットの準備
        transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
        lang  = ImageLang( 'dataset/onomatope.csv',"dataset/image/train",transform)
        valid_lang  = ImageLang( 'dataset/onomatope.csv',"dataset/image/valid",transform)

        train_dataloader = DataLoader(lang, batch_size=batch_size, shuffle=False,drop_last=True) #drop_lastをtruenにすると最後の中途半端に入っているミニバッチを排除してくれる
        valid_dataloader=DataLoader(valid_lang,batch_size=batch_size, shuffle=False,drop_last=True)

        #モデルの準備
        encoder           = Encoder( num, embedding_size, hidden_size ).to( device )
        decoder           = Decoder( hidden_size, embedding_size, num ).to( device )
        image_model=CAE().to(device)
        enfile = "model/firstencoder" #学習済みのエンコーダモデル
        defile = "model/firstdecoder" #学習済みのデコーダモデル
        imgfile="model/imgmodel"
        encoder.load_state_dict( torch.load( enfile ) ) #読み込み
        decoder.load_state_dict( torch.load( defile ) )
        image_model.load_state_dict(torch.load(imgfile))
        encoder.eval()
        decoder.eval()
        image_model.eval()
        criterion=nn.MSELoss()
        sentence="g o r i g o r i"
        ono_word,encoder_hidden=ono_to_ono(sentence,encoder,decoder)
        print(ono_word,"enc")
        sys.exit()
        ono_list=[]
        word_list=[]
        # img_torch=torch.zeros(len(train_dataloader),embedding_size) #imgの埋め込みベクトルのリスト
        # ono_torch=torch.zeros(len(train_dataloader),embedding_size) #onomatopeの埋め込みベクトルのリスト
        
        dataloader=valid_dataloader
        img_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,128))
        ono_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,128))
        # for batch_num,(img,path,ono,phoneme) in tqdm.tqdm(enumerate(train_dataloader),total=len(train_dataloader)): #プログレスバーあり
        for batch_num,(img,path,ono,phoneme) in enumerate(dataloader):
            img=img.to(device)
            output_batch,_=image_model(img)

            
            # sys.exit()
            img_numpy=np.zeros((0,128))
            ono_numpy=np.zeros((0,128))    
            for data_num in range(dataloader.batch_size):           
                img_input=img[data_num].view(-1,3,size,size).to(device)
                output,img_hidden=image_model(img_input)   
                img_hidden=img_hidden.view(1,1,128) #画像の特徴ベクトルをオノマトペの隠れベクトルの大きさに合わせる
                
 

                img_word=img_to_ono(img_hidden,decoder) #画像の隠れベクトルからオノマトペを生成
                ono_word,encoder_hidden=ono_to_ono(phoneme[data_num],encoder,decoder)
                encoder_hidden=encoder_hidden.view(1,1,128)
                loss=criterion(encoder_hidden,img_hidden)
                # print(loss) #Lossの表示
                
                encoder_hidden=encoder_hidden.view(-1,128) #オノマトペの特徴ベクトルを画像の隠れベクトルのサイズに合わせる
                ono_output=image_model.decoder(encoder_hidden) #画像のデコーダにオノマトペの特徴ベクトルを渡す
                ono_list.append(ono_output) #オノマトペから生成された画像をリストにアペンドしていく（のちにバッチサイズ分のトーチのリストとなる）
                word_list.append(ono[data_num]) #オノマトペの単語をリストにアペンド（主成分分析のラベルとして使う）
                img_hidden=img_hidden.view(1,128).to('cpu').numpy()    
                ono_hidden=encoder_hidden.view(1,128).to('cpu').numpy()           
                img_numpy=np.concatenate((img_numpy,img_hidden),axis=0)
                ono_numpy=np.concatenate((ono_numpy,ono_hidden),axis=0)

                # print(img_word,ono[data_num],ono_word,phoneme[data_num]) #画像特徴からオノマトペの音素とオノマトペの音素からオノマトペの音素の結果を表示
            
            ono_output_batch=torch.cat(ono_list,dim=0) #エンコーダの隠れベクトルをバッチサイズ分の大きさにする

            # generate_and_save_images(img,batch_num,0,ono) #オリジナルの画像
            # generate_and_save_images(output_batch,batch_num,1,ono) #画像特徴から画像
            # generate_and_save_images(ono_output_batch,batch_num,2,phoneme) #オノマトペの音素から画像
            img_batch_numpy[batch_num]=img_numpy
            ono_batch_numpy[batch_num]=ono_numpy

   


    #主成分分析する
    

    pca = PCA()
    img_batch_numpy=img_batch_numpy.reshape(-1,128)
    ono_batch_numpy=ono_batch_numpy.reshape(-1,128)
    p1 = pca.fit(img_batch_numpy)
    p2 = pca.fit(ono_batch_numpy)
    # 分析結果を元にデータセットを主成分に変換する

    transformed1 = p1.fit_transform(img_batch_numpy)
    transformed2 = p2.fit_transform(ono_batch_numpy)

    markers1 = ["$x$","$\\alpha$", "$A$","$B$","$C$","$D$","$E$","$F$","$G$","$H$","$J$","$K$", "$L$","$M$","$N$","$O$","$P$","$Q$","$R$","$S$","$T$","$U$","$V$","$W$", ",", "o", "v", "^", "p", "*", "D","d"]
    color1 =["black","gray","silver","rosybrown","firebrick",
            "red","darksalmon","sienna","sandybrown","tan",
            "gold","olivedrab","chartreuse","palegreen",
            "darkgreen","lightseagreen","paleturquoise",
            "deepskyblue","blue","pink","orange","crimson",
            "mediumvioletred","plum","darkorchid","mediumpurple",
            "navy","chocolate","peru","yellow","y","aqua","lightsteelblue","linen","teal"]


    fig = plt.figure()
    ax=fig.add_subplot(aspect='1')

    print(len(word_list))
    print(len(transformed1))
    for i in range(len(transformed1)):
        lines=[[(transformed1[i,0],transformed1[i,1]),(transformed2[i,0],transformed2[i,1])]for i in range(len(transformed1))] 
        linedistance=math.sqrt((transformed2[i,0]-transformed1[i,0])**2+(transformed2[i,1]-transformed1[i,1])**2)
        plt.text(transformed1[i, 0], transformed1[i, 1], word_list[i], fontsize=4)
        plt.text(transformed2[i, 0], transformed2[i, 1], word_list[i], fontsize=4)

        # print(linedistance,"line")
    lc=mc.LineCollection(lines,colors="k",linewidths=0.5)
    plt.axis([-4,8,-4,8])
    plt.scatter(transformed1[:len(transformed1), 0], transformed1[:len(transformed1), 1],c="red")#画像
    plt.scatter(transformed2[:len(transformed2), 0], transformed2[:len(transformed2), 1],c="blue",marker="$x$")#オノマトペ
    ax.add_collection(lc)

    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    fig.savefig("figure/(train).png")
    plt.show()