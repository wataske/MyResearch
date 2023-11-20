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

# def calc_ono2ono_accu(encoder,decoder,dataloader,lang):
#     score=0
#     for batch_num,(img,img_path,ono,phoneme) in enumerate(dataloader):  
#         for data_num in range(dataloader.batch_size):           
#             ono_word,encoder_hidden=ono_to_ono(phoneme[data_num],encoder,decoder,lang)
            
#             word=[x.replace("<EOS>","") for x in ono_word]
#             word=[x+' 'for x in word] #1音素ずつに半角の空白を追加
#             word[-1]=word[-1].strip() #最後の音素の後ろの空白だけ消す
#             word=''.join(word) #リストになってたものを１つの単語にする

#             if is_close_match(phoneme[data_num],word):
#                 score+=1
#     return (score/len(dataloader.dataset))
def calc_accu(encoder,decoder,dataloader,lang):
    score=0
    count=0
    for batch_num,(ono,phoneme) in enumerate(dataloader):  
        for data_num in range(dataloader.batch_size):           
            ono_word,encoder_hidden=ono_to_ono(phoneme[data_num],encoder,decoder,lang)

            word=[x.replace("<EOS>","") for x in ono_word]
            word=[x+' 'for x in word] #1音素ずつに半角の空白を追加
            word[-1]=word[-1].strip() #最後の音素の後ろの空白だけ消す
            word=''.join(word) #リストになってたものを１つの単語にする
 
            if is_close_match(phoneme[data_num],word):
                score+=1

            count+=1
    return (score/count)

def calc_img2ono_accu(image_model,decoder,dataloader,lang):
    score=0
    count=0
    for batch_num,(img,_,_,phoneme) in enumerate(dataloader):  
        for data_num in range(dataloader.batch_size):           
            img_word,_=img_to_ono(img[data_num],image_model,decoder,lang)
            
            word=[x.replace("<EOS>","") for x in img_word]
            word=[x+' 'for x in word] #1音素ずつに半角の空白を追加
            word[-1]=word[-1].strip() #最後の音素の後ろの空白だけ消す
            word=''.join(word) #リストになってたものを１つの単語にする

            if is_close_match(phoneme[data_num],word):
                score+=1
            count+=1
    return (score/count)
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
def generate_ono_image(pic, epoch,label):
    fig = plt.figure(figsize=(4, 4))

    plt.subplot(1, 1, 1)
    print(pic.shape,"picdesu")
    plt.imshow(pic.cpu().data[:, :, :].permute(1, 2, 0))
    name=os.path.basename(label)
    name,_=os.path.splitext(name)
    plt.title(name)
    plt.axis('off')
    plt.savefig("/workspace/mycode/Documents/output/batch/{}.jpg".format(label))


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

def generate_bar_graph(dictionary):
    graph_dict={}
    
    for label,values in dictionary.items():
        total=sum(values)
        count=len(values)
        graph_dict[label]=total/count
    
    graph_dict = dict(sorted(graph_dict.items(), key=lambda item: item[1], reverse=True))
    # 辞書からラベルと値を取り出す

    labels = list(graph_dict.keys())
    values = list(graph_dict.values())

    # 棒グラフの生成
    plt.figure(figsize=(15,5))
    bars=plt.bar(labels, values)
    # 各棒の上に値を表示
    for bar in bars:
        print(bar)
        yval = bar.get_height()
        print(yval)
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, '{:.2f}'.format(yval), va='bottom', ha='center')
    # グラフのタイトルと軸ラベルの追加
    plt.title('Cosine Similarity')
    plt.xlabel('onomatope')
    plt.xticks(rotation=30,fontsize=10)
    plt.ylabel('cosine')
    plt.ylim(0,1)
    plt.savefig("figure/cosine_bar_graph.jpg")
    # グラフの表示
    plt.show()


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
        cosine_dict={}
        #データセットの準備
        transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
  
        lang=Lang('dataset/onomatope/dictionary.csv')
        train_lang  = ImageLang( 'dataset/imageono/onomatope/train/train_image_onomatope.csv',"dataset/imageono/image/train",transform)
        valid_lang  = ImageLang( 'dataset/imageono/onomatope/train/train_image_onomatope.csv',"dataset/imageono/image/valid",transform)
        
        train_dataloader = DataLoader(train_lang, batch_size=batch_size, shuffle=False,drop_last=True) #drop_lastをtruenにすると最後の中途半端に入っているミニバッチを排除してくれる
        valid_dataloader=DataLoader(valid_lang,batch_size=batch_size, shuffle=False,drop_last=True)

        ono_train_dataloader=DataLoader(lang,batch_size=batch_size,shuffle=False,drop_last=True)    
        ono_valid_dataset=Lang('dataset/onomatope/onomatopeunknown.csv')
        ono_valid_dataloader=DataLoader(ono_valid_dataset,batch_size=batch_size,shuffle=False,drop_last=True)    
        #モデルの準備
        encoder           = Encoder( num, embedding_size, hidden_size ).to( device )
        decoder           = Decoder( hidden_size, embedding_size, num ).to( device )
        image_model=TextureNet(feature_layers=["0", "5", "10"]).to(device)

        enfile="model/encodercosine2"
        defile="model/decodercosine2"
        imgfile="model/imgmodelcosine2"
        encoder.load_state_dict( torch.load( enfile ) ) #読み込み
        decoder.load_state_dict( torch.load( defile ) )
        image_model.load_state_dict(torch.load(imgfile))
        encoder.eval()
        decoder.eval()
        image_model.eval()
        criterion=nn.MSELoss()
        cos=nn.CosineEmbeddingLoss()
        cos2=nn.CosineSimilarity()
        sentence="p u r i p u r i"
        ono_word,encoder_hidden=ono_to_ono(sentence,encoder,decoder,lang)

        word_list=[]

        
        dataloader=train_dataloader
        img_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,128))
        ono_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,128))
     
        img_hidden_list=[]
        for batch_num,(img,path,ono,phoneme) in tqdm.tqdm(enumerate(dataloader),total=len(dataloader)): #プログレスバーあり
        # for batch_num,(img,path,ono,phoneme) in enumerate(dataloader):
            img_numpy=np.zeros((0,128))
            ono_numpy=np.zeros((0,128))    
            img_list=[]
            ono_list=[]
            img_output_batch=torch.zeros(dataloader.batch_size,3,size,size)
            ono_output_batch=torch.zeros(dataloader.batch_size,3,size,size)
            for data_num in range(dataloader.batch_size):      
                add_hidden=[]

                img_data=img[data_num].to(device)
                img_data=img_data.unsqueeze(0)
                output_batch,_=image_model(img_data)
                img_list.append(output_batch)     

                img_word,img_hidden=img_to_ono(img[data_num],image_model,decoder,lang) #画像の隠れベクトルからオノマトペを生成
                ono_word,encoder_hidden=ono_to_ono(phoneme[data_num],encoder,decoder,lang)
                add_hidden.append(img_hidden)
                img_hidden_list.append(img_hidden)
                encoder_hidden=encoder_hidden.squeeze(1)
                img_hidden=img_hidden.squeeze(1)
                


                #-------------------------------------------- cosine類似度を計算
                loss=cos(encoder_hidden,img_hidden,torch.tensor([1.0]).to(device))
                similar=cos2(encoder_hidden,img_hidden)
                #---------------------------------------
                encoder_hidden=encoder_hidden.view(-1,128) #オノマトペの特徴ベクトルを画像の隠れベクトルのサイズに合わせる

                ono_output=image_model.decoder(encoder_hidden) #画像のデコーダにオノマトペの特徴ベクトルを渡す
                ono_list.append(ono_output) #オノマトペから生成された画像をリストにアペンドしていく（のちにバッチサイズ分のトーチのリストとなる）
                word_list.append(ono[data_num]) #オノマトペの単語をリストにアペンド（主成分分析のラベルとして使う）
                img_hidden=img_hidden.view(1,128).to('cpu').numpy()    
                ono_hidden=encoder_hidden.view(1,128).to('cpu').numpy()  

                img_numpy=np.concatenate((img_numpy,img_hidden),axis=0)
                ono_numpy=np.concatenate((ono_numpy,ono_hidden),axis=0)
                print(similar.item(),"↓のコサイン類似度")
                print(img_word,ono[data_num],ono_word,phoneme[data_num]) #画像特徴からオノマトペの音素とオノマトペの音素からオノマトペの音素の結果を表示

                #--------------------------------------
                if ono[data_num] not in cosine_dict:
                    cosine_dict[ono[data_num]]=[]
                    generate_ono_image(ono_output[0],batch_num,phoneme[data_num])#オノマトペの音素から画像
                cosine_dict[ono[data_num]].append(similar.item())
                #-------------------------------------------
            img_output_batch=torch.cat(img_list,dim=0)
            ono_output_batch=torch.cat(ono_list,dim=0) #エンコーダの隠れベクトルをバッチサイズ分の大きさにする

            # generate_and_save_images(img,batch_num,0,ono) #オリジナルの画像
            # generate_and_save_images(img_output_batch,batch_num,1,ono) #画像特徴から画像

    
            img_batch_numpy[batch_num]=img_numpy
            ono_batch_numpy[batch_num]=ono_numpy


        # generate_bar_graph(cosine_dict)

        # print(calc_accu(encoder,decoder,ono_valid_dataloader,lang))
        # print(calc_img2ono_accu(image_model,decoder,dataloader,lang))
        



    #主成分分析する
    from sklearn.preprocessing import StandardScaler

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances

    scaler=StandardScaler()

    
    img_batch_numpy=img_batch_numpy.reshape(-1,128)
    ono_batch_numpy=ono_batch_numpy.reshape(-1,128)

    combined_numpy = np.concatenate((img_batch_numpy, ono_batch_numpy), axis=0)

    pca=PCA()
    img_pca = pca.fit(img_batch_numpy)
    ono_pca = pca.fit(ono_batch_numpy)
    imgono_pca=pca.fit(combined_numpy)
    # 分析結果を元にデータセットを主成分に変換する



    transformed1 = img_pca.fit_transform(img_batch_numpy)
    transformed2 = ono_pca.fit_transform(ono_batch_numpy)
    transformed3=imgono_pca.fit_transform(combined_numpy)
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

    print(len(transformed3))
    num=int(len(transformed3)/2)
    plt.scatter(transformed3[:num, 0], transformed3[:num, 1],c="red")#画像
    plt.scatter(transformed3[num:, 0], transformed3[num:, 1],c="blue",marker="$x$")#オノマトペ
    for i in range(int(len(transformed3)/2)): #ラベルをプロットしていく
        plt.text(transformed3[i+int(len(transformed3)/2), 0], transformed3[i+int(len(transformed3)/2), 1],word_list[i],fontsize=8)
    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    fig.savefig("figure/PCA.png")
    plt.show()
