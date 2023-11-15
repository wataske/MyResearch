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
from seq2seq import Lang
from texture import TextureNet,style_loss_and_diffs

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1) #s2の音素数＋1をpreviousにする
    for i, c1 in enumerate(s1): #s1の各音素がどうなるかs2の各音素でチェックする
        current_row = [i + 1]

        for j, c2 in enumerate(s2): #s1の音素（C1)に対してs2は何をしたらどれくらいコストがかかるのか表示する
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2) #Trueなら1をFalseなら0を返す

            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def is_close_match(s1, s2, tolerance=2):
    return levenshtein_distance(s1, s2) <= tolerance

def calc_ono2ono_accu(encoder,decoder,dataloader,lang):
    score=0
    for batch_num,(img,img_path,ono,phoneme) in enumerate(dataloader):  
        for data_num in range(dataloader.batch_size):           
            ono_word,encoder_hidden=ono_to_ono(phoneme[data_num],encoder,decoder,lang)
            
            word=[x.replace("<EOS>","") for x in ono_word]
            word=[x+' 'for x in word] #1音素ずつに半角の空白を追加
            word[-1]=word[-1].strip() #最後の音素の後ろの空白だけ消す
            word=''.join(word) #リストになってたものを１つの単語にする

            if is_close_match(phoneme[data_num],word):
                score+=1
    return (score/len(dataloader.dataset))

def calc_img2ono_accu(image_model,decoder,dataloader,lang):
    score=0
    for batch_num,(img,_,_,phoneme) in enumerate(dataloader):  
        for data_num in range(dataloader.batch_size):           
            img_word,_=img_to_ono(img[data_num],image_model,decoder,lang)
            
            word=[x.replace("<EOS>","") for x in img_word]
            word=[x+' 'for x in word] #1音素ずつに半角の空白を追加
            word[-1]=word[-1].strip() #最後の音素の後ろの空白だけ消す
            word=''.join(word) #リストになってたものを１つの単語にする

            if is_close_match(phoneme[data_num],word):
                score+=1
    return (score/len(dataloader.dataset))
def img_to_ono(img,image_model,decoder,lang):
    img_input=img.view(-1,3,size,size).to(device)
    output,img_hidden=image_model(img_input)   
    img_hidden=img_hidden.view(1,1,128) #画像の特徴ベクトルをオノマトペの隠れベクトルの大きさに合わせる
    decoder_input      = torch.tensor([[SOS_token]], device=device)  # SOS
    decoder_hidden     = img_hidden
    decoded_words      = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden )
        topv, topi = decoder_output.data.topk(1) #topiはアウトプットの中から最も確率の高いラベル（音素のインデックス番号）取り出す
        if topi.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[topi.item()]) 

        decoder_input = topi.squeeze().detach()
    return decoded_words,img_hidden
def ono_to_ono(sentence,encoder,decoder,lang):
    
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
        plt.savefig("/workspace/mycode/Documents/output/single/onomatope{}.jpg".format(epoch))
    else:
        plt.savefig('/workspace/mycode/Documents/output/single/image{}.png'.format(epoch))

def generate_and_save_images(pic, epoch,num,label):
    fig = plt.figure(figsize=(4, 4))
    for i in range(8):
        plt.subplot(4, 2, i + 1)
        plt.imshow(pic.cpu().data[i, :, :, :].permute(1, 2, 0))

        plt.title(label[i])
        plt.axis('off')
    if num == 0:
        plt.savefig("/workspace/mycode/Documents/output/batch/original{}.jpg".format(epoch))
    elif num ==1:
        plt.savefig('/workspace/mycode/Documents/output/batch/image{}.png'.format(epoch))
    else:
        plt.savefig('/workspace/mycode/Documents/output/batch/onomatope{}.png'.format(epoch))

if __name__ == '__main__':
    with torch.no_grad():
        SOS_token = 0
        EOS_token = 1
        device = "cuda" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size =16
        num=40 #入出力として使える音素の数=データセット内の.n_wordsに等しい
        embedding_size = 128
        hidden_size   = 128
        max_length=20
        size=64
        
        #データセットの準備
        transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
        # transform = transforms.Compose([transforms.Resize((size, size)),
        # transforms.RandomResizedCrop(size),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor()])
        lang=Lang('dataset/dictionary.csv')
        train_lang  = ImageLang( 'dataset/onomatope.csv',"dataset/image/train",transform)
        valid_lang  = ImageLang( 'dataset/onomatope.csv',"dataset/image/valid",transform)
        
        train_dataloader = DataLoader(train_lang, batch_size=batch_size, shuffle=False,drop_last=True) #drop_lastをtruenにすると最後の中途半端に入っているミニバッチを排除してくれる
        valid_dataloader=DataLoader(valid_lang,batch_size=batch_size, shuffle=False,drop_last=True)
        #モデルの準備
        encoder           = Encoder( num, embedding_size, hidden_size ).to( device )
        decoder           = Decoder( hidden_size, embedding_size, num ).to( device )
        image_model=TextureNet(feature_layers=["0", "5", "10"]).to(device)
        # enfile = "../../../M1 春セメ/作業用/github/model/encoder0928" #学習済みのエンコーダモデル
        # defile = "../../../M1 春セメ/作業用/github/model/decoder0928" #学習済みのデコーダモデル
        # imgfile="../../../M1 春セメ/作業用/github/model/imgmodel0928"
        enfile="model/encodercheckcosineseparatecross"
        defile="model/decodercheckcosineseparatecross"
        imgfile="model/imgmodelcheckcosineseparatecross"
        encoder.load_state_dict( torch.load( enfile ) ) #読み込み
        decoder.load_state_dict( torch.load( defile ) )
        image_model.load_state_dict(torch.load(imgfile))
        encoder.eval()
        decoder.eval()
        image_model.eval()
        criterion=nn.MSELoss()
        cos=nn.CosineEmbeddingLoss()
        sentence="a m i a m i"
        ono_word,encoder_hidden=ono_to_ono(sentence,encoder,decoder,lang)

        ono_list=[]
        word_list=[]
        hist_list=[]
        # img_torch=torch.zeros(len(train_dataloader),embedding_size) #imgの埋め込みベクトルのリスト
        # ono_torch=torch.zeros(len(train_dataloader),embedding_size) #onomatopeの埋め込みベクトルのリスト
        
        dataloader=train_dataloader
        img_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,128))
        ono_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,128))
     
        img_hidden_list=[]
        for batch_num,(img,path,ono,phoneme) in tqdm.tqdm(enumerate(dataloader),total=len(dataloader)): #プログレスバーあり
        # for batch_num,(img,path,ono,phoneme) in enumerate(dataloader):
            img=img.to(device)
            output_batch,_=image_model(img)
            
            # sys.exit()
            img_numpy=np.zeros((0,128))
            ono_numpy=np.zeros((0,128))    
            for data_num in range(dataloader.batch_size):      
                add_hidden=[]     
                img_word,img_hidden=img_to_ono(img[data_num],image_model,decoder,lang) #画像の隠れベクトルからオノマトペを生成
                ono_word,encoder_hidden=ono_to_ono(phoneme[data_num],encoder,decoder,lang)

                add_hidden.append(img_hidden)
                img_hidden_list.append(img_hidden)
                encoder_hidden=encoder_hidden.squeeze(1)
                img_hidden=img_hidden.squeeze(1)
                # print(encoder_hidden)
                # print(img_hidden)
                cos2=nn.CosineSimilarity()
                hist_list.append(cos2(encoder_hidden,img_hidden))
                loss=cos(encoder_hidden,img_hidden,torch.tensor([1.0]).to(device))

                print(loss) #Lossの表示
                encoder_hidden=encoder_hidden.view(-1,128) #オノマトペの特徴ベクトルを画像の隠れベクトルのサイズに合わせる

                ono_output=image_model.decoder(encoder_hidden) #画像のデコーダにオノマトペの特徴ベクトルを渡す
                ono_list.append(ono_output) #オノマトペから生成された画像をリストにアペンドしていく（のちにバッチサイズ分のトーチのリストとなる）
                word_list.append(ono[data_num]) #オノマトペの単語をリストにアペンド（主成分分析のラベルとして使う）
                img_hidden=img_hidden.view(1,128).to('cpu').numpy()    
                ono_hidden=encoder_hidden.view(1,128).to('cpu').numpy()  

                img_numpy=np.concatenate((img_numpy,img_hidden),axis=0)
                ono_numpy=np.concatenate((ono_numpy,ono_hidden),axis=0)

                print(img_word,ono[data_num],ono_word,phoneme[data_num]) #画像特徴からオノマトペの音素とオノマトペの音素からオノマトペの音素の結果を表示
            
            ono_output_batch=torch.cat(ono_list,dim=0) #エンコーダの隠れベクトルをバッチサイズ分の大きさにする

            generate_and_save_images(img,batch_num,0,ono) #オリジナルの画像
            generate_and_save_images(output_batch,batch_num,1,ono) #画像特徴から画像
            generate_and_save_images(ono_output_batch,batch_num,2,phoneme) #オノマトペの音素から画像
            print(criterion(img,ono_output_batch))
            sys.exit()
            img_batch_numpy[batch_num]=img_numpy
            ono_batch_numpy[batch_num]=ono_numpy
        # print(calc_ono2ono_accu(encoder,decoder,dataloader,lang))
        # print(calc_img2ono_accu(image_model,decoder,dataloader,lang))
        # img1=img_hidden_list[0]
        # img1=torch.tensor(img1)
        # img2=img_hidden_list[400]
        # img2=torch.tensor(img2)
        # loss=criterion(img1,img2) 
        



    #主成分分析する
    from sklearn.preprocessing import StandardScaler

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances
    fig=plt.figure()
    numpy_arrays = [tensor.cpu().numpy() for tensor in hist_list]

    plt.hist(numpy_arrays,edgecolor='black',linewidth=1.5)
    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    fig.savefig("figure/cosine_histgram2.png")
    plt.show()
    sys.exit()
    scaler=StandardScaler()
    # sys.exit()

    
    img_batch_numpy=img_batch_numpy.reshape(-1,128)
    ono_batch_numpy=ono_batch_numpy.reshape(-1,128)
    print(len(img_batch_numpy))
    print(len(ono_batch_numpy))
    combined_numpy = np.concatenate((img_batch_numpy, ono_batch_numpy), axis=0)
    # dfimg=pd.DataFrame(img_batch_numpy)
    # dfono=pd.DataFrame(ono_batch_numpy)
    # print(dfimg.describe(),"img")
    # print(dfono.describe(),"ono")
    # img_batch_numpy=scaler.fit_transform(img_batch_numpy)
    # ono_batch_numpy=scaler.fit_transform(ono_batch_numpy)
    # dissimilarities_img = pairwise_distances(img_batch_numpy)
    # dissimilarities_ono = pairwise_distances(ono_batch_numpy)

    pca=PCA()
    # print(len(combined_numpy))
    p1 = pca.fit(img_batch_numpy)
    p2 = pca.fit(ono_batch_numpy)
    p3=pca.fit(combined_numpy)
    # 分析結果を元にデータセットを主成分に変換する

    mds = MDS(n_components=2, metric=False, dissimilarity='precomputed', random_state=42)

    # projected_img = mds.fit_transform(dissimilarities_img)
    # projected_ono = mds.fit_transform(dissimilarities_ono)
    transformed1 = p1.fit_transform(img_batch_numpy)
    transformed2 = p2.fit_transform(ono_batch_numpy)
    transformed3=p3.fit_transform(combined_numpy)
    eigenvalues_combined = pca.explained_variance_

    # 固有値を出力
    # print("Eigenvalues of combined_numpy:", eigenvalues_combined)
    # sys.exit()

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


    img_norm=0
    phoneme_norm=0
    for i in range(len(transformed1)):
        tensor1=torch.tensor(transformed1[i],dtype=torch.float32)
        # print(tensor1)
        # print(img_batch_numpy[i])
        img_norm +=torch.norm(tensor1)
        # print(transformed1[i])


   
        tensor2=torch.tensor(transformed2[i],dtype=torch.float32)
        # print(tensor2)
        # print(ono_batch_numpy[i])
        phoneme_norm +=torch.norm(tensor2)
        # print(transformed2[i])
        # MSELossを使用して二つのテンソル間の損失を計算
        criterion = torch.nn.MSELoss()
        loss = criterion(tensor1, tensor2)

        # print(loss.item())
    # print(img_norm/len(transformed1))
    # print(phoneme_norm/len(transformed2))
    #     lines=[[(transformed1[i,0],transformed1[i,1]),(transformed2[i,0],transformed2[i,1])]for i in range(200)] 
    #     linedistance=math.sqrt((transformed2[i,0]-transformed1[i,0])**2+(transformed2[i,1]-transformed1[i,1])**2)
    #     plt.text(transformed1[i, 0], transformed1[i, 1], word_list[i], fontsize=4)
    #     plt.text(transformed2[i, 0], transformed2[i, 1], word_list[i], fontsize=4)

    #     print(linedistance,"line",word_list[i])
    # lc=mc.LineCollection(lines,colors="k",linewidths=0.5)
    print(len(transformed3))
    plt.scatter(transformed3[:1392, 0], transformed3[:1392, 1],c="red")#画像
    plt.scatter(transformed3[1392:, 0], transformed3[1392:, 1],c="blue",marker="$x$")#オノマトペ
    # ax.add_collection(lc)

    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    fig.savefig("figure/PCA.png")
    plt.show()

    # fig = plt.figure()
    # ax=fig.add_subplot(aspect='1')

    # print(len(word_list))
    # print(transformed1[0])
    # print(len(transformed2))

    # for i in range(len(transformed1)):
    #     lines=[[(transformed1[i,0],transformed1[i,1]),(transformed2[i,0],transformed2[i,1])]for i in range(200)] 
    #     linedistance=math.sqrt((transformed2[i,0]-transformed1[i,0])**2+(transformed2[i,1]-transformed1[i,1])**2)
    #     plt.text(transformed1[i, 0], transformed1[i, 1], word_list[i], fontsize=4)
    #     plt.text(transformed2[i, 0], transformed2[i, 1], word_list[i], fontsize=4)

    #     print(linedistance,"line")
    # lc=mc.LineCollection(lines,colors="k",linewidths=0.5)
    # plt.axis([-0.5,0.5,-0.5,0.5])
    # plt.scatter(transformed1[:len(transformed1), 0], transformed1[:len(transformed1), 1],c="red")#画像
    # plt.scatter(transformed2[:len(transformed2), 0], transformed2[:len(transformed2), 1],c="blue",marker="$x$")#オノマトペ
    # plt.scatter(transformed1[:, 0], transformed1[:, 1],c="red")#画像
    # plt.scatter(transformed2[:, 0], transformed2[:, 1],c="blue",marker="$x$")#オノマトペ
    # ax.add_collection(lc)

    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    fig.savefig("figure/PCA.png")
    plt.show()

    # fig = plt.figure()
    # ax=fig.add_subplot(aspect='1')
    # for i in range(200):
    #     lines=[[(projected_img[i,0],projected_img[i,1]),(projected_ono[i,0],projected_ono[i,1])]for i in range(200)] 
    #     linedistance=math.sqrt((projected_ono[i,0]-projected_img[i,0])**2+(projected_ono[i,1]-projected_img[i,1])**2)
    #     plt.text(projected_img[i, 0], projected_img[i, 1], word_list[i], fontsize=4)
    #     plt.text(projected_ono[i, 0], projected_ono[i, 1], word_list[i], fontsize=4)

    #     print(linedistance,"line")
    # lc=mc.LineCollection(lines,colors="k",linewidths=0.5)


    # plt.scatter(projected_img[:200, 0], projected_img[:200, 1],c="red")#画像
    # plt.scatter(projected_ono[:200, 0], projected_ono[:200, 1],c="blue",marker="$x$")#オノマトペ
    # ax.add_collection(lc)


    # plt.title('principal component')
    # plt.xlabel('pc1')
    # plt.ylabel('pc2')
    # fig.savefig("figure/MDS.png")
    # plt.show()