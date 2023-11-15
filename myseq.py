#自分なりにseq2seqを書いてみた
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
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from seq2seq import Lang
from seq2seq import Encoder
from seq2seq import Decoder
from seq2seq import tensorFromSentence
from texture import TextureNet,style_loss_and_diffs

SOS_token = 0
EOS_token = 1

device = "cuda" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageLang:
    def __init__( self, filename,dir,transform ): #呼び出されたとき、最初に行うこと
        self.transform = transform
        max_length    = 20
        self.filename = filename
        self.data = [] #画像が入ってる
        self.labels = [] #画像のラベルを入れてる
        self.ono=[]    
        self.word2index = {}
        self.word2count = {}
        self.sentences = []
        self.index2word = { 0: "SOS", 1: "EOS" }
        self.n_words = 2  # Count SOS and EOS
        name_to_label={}
        df = pd.read_csv(filename)
        num=df.shape[0] #csvファイルの行数を取得（データセットの数）
        for i in range(num): #264種類のラベルを作る
            word=df.iloc[i,2] #単語(3列目)
            phoneme = df.iloc[i, 1]  # 音素(2列目)
            ono_labels = df.iloc[i, 0].astype(str)  # ラベル (1列目)
            dict={word:i}
            i=i+1
            self.ono.append(word)
            self.sentences.append(phoneme)
            name_to_label.update(dict)
        self.allow_list = [ True ] * len( self.sentences )


        target_dir = os.path.join(dir, "*")
        #フォルダではなくファイル名でラベルを振る場合----------------------------------------------------------------
        # for path in glob.glob(target_dir):

        #     name = os.path.splitext(os.path.basename(path))[0]
        #     name=unicodedata.normalize("NFKC",name)
        #     label = name_to_label[name]
        #     self.data.append(path)
        #     self.labels.append(label) 


        #----------------------------------------------------------------
        #フォルダにラベル名を振る場合のコード----------------------------------------------------------------
        for path in glob.glob(target_dir):

            name = os.path.splitext(os.path.basename(path))[0]
            name=unicodedata.normalize("NFKC",name)
            label = name_to_label[name]
            
             
            for data in glob.glob(os.path.join(path,"*")):
                self.labels.append(label) 
                self.data.append(data)
        #----------------------------------------------------------------
        allow_list  = self.get_allow_list( max_length )#maxlength以下の長さの音素数をallowlistに入れる（長すぎる音素は省かれる)
        self.load_file(allow_list)#max以下の長さである単語を引数に渡す（今回の場合は264こ）
    def get_sentences( self ):
        return self.sentences[ :: ] 

    def get_sentence( self, index ):
        return self.sentences[ index ]

    def choice( self ): #ランダムに配列をピックする
        while True:
            index = random.randint( 0, len( self.allow_list ) - 1 )
            if self.allow_list[ index ]:
                break
        return self.sentences[ index ], index

    def get_allow_list( self, max_length ):
        allow_list = []
        for sentence in self.sentences:
            if len( sentence.split() ) < max_length:
                allow_list.append( True )
            else:
                allow_list.append( False )
        return allow_list
                    
    def load_file( self, allow_list = [] ):
        if allow_list: #allow_listが空でなければ実行
            self.allow_list = [x and y for (x,y) in zip( self.allow_list, allow_list ) ] #自分のTrueの情報を与えられたデータセットに合わせる(max_lengthより長い音素は全てFalseに変わる)
        self.target_sentences = []
        for i, sentence in enumerate( self.sentences ):
            if self.allow_list[i]: #i番目がtrueであったら行う、Falseならスルー
                self.addSentence( sentence ) #単語に含まれている音素が初見であればリストに加えていく
                self.target_sentences.append( sentence ) #音素をtarget_sentencesに加えていく
                    
    def addSentence( self, sentence ): #一単語に含まれている音素を全てリストに入れていく
        for word in sentence.split():
            self.addWord(word)
            

    def addWord( self, word ): #wordを数値化する
      
        
        if word not in self.word2index:
            self.word2index[ word ] = self.n_words #word2indexに音素を入力すると数字が出てくる
            
            self.word2count[ word ] = 1 #ある音素が出現した数をカウントしてる
            self.index2word[ self.n_words ] = word #index2wordに数字を入力すると音素が出てくる
            self.n_words += 1 #リストに入っている音素の総数(かぶっているものは除外される)語彙数みたいなもん
        else:
            self.word2count[word] += 1
    #ここから自分



    def __len__(self):
        """data の行数を返す。
        """
        return len(self.data)
    def __getitem__(self, index):
#Tripletの場合------------------------------------------------------------------------------------------------------------
        # """Tripletを返す。
        # """
        
        # # アンカーのデータを取得
        # anchor_img_path = self.data[index]
        # anchor_label = self.labels[index]
        # anchor_ono = self.ono[anchor_label]
        # anchor_phoneme = self.sentences[anchor_label]
        # anchor_img = Image.open(anchor_img_path).convert("RGB")
        # anchor_img = self.transform(anchor_img)
        
        # # ポジティブのデータを取得
        # # ここでは、同じラベルの別の画像をランダムに選択します
        # positive_indices = [i for i, label in enumerate(self.labels) if label == anchor_label and i != index]
        # positive_index = random.choice(positive_indices)
        # positive_img_path = self.data[positive_index]
        # positive_img = Image.open(positive_img_path).convert("RGB")
        # positive_img = self.transform(positive_img)
        
        # # ネガティブのデータを取得
        # # 異なるラベルの画像をランダムに選択します
        # negative_indices = [i for i, label in enumerate(self.labels) if label != anchor_label]
        # negative_index = random.choice(negative_indices)
        # negative_img_path = self.data[negative_index]
        # negative_img = Image.open(negative_img_path).convert("RGB")
        # negative_img = self.transform(negative_img)
        
        # return anchor_img, anchor_img_path, anchor_ono, anchor_phoneme, positive_img, negative_img
#------------------------------------------------------------------------------------------------------------

        img_path = self.data[index] #適当な画像を取ってくる
        img_label = self.labels[index] #その画像のラベル番号は何番か取ってくる
        ono=self.ono[img_label] #取ってきたラベル番号のオノマトペを取ってくる（オノマトペのラベル番号とリストの番号は一致している）
        
        phoneme = self.sentences[img_label] #同上
    
        
        img = Image.open(img_path).convert("RGB") #img_pathの画像を開く
        img = self.transform(img) #transformする
        return img, img_path, ono, phoneme       
# def l2_norm_difference_loss(v, w):
#     return (v.norm(p=2, dim=1) - w.norm(p=2, dim=1)).pow(2).mean()

# def regularization_loss(z):
#     return (z.norm(p=2, dim=1) - 1).pow(2).mean()

def Train(encoder,decoder,image_model,lang,imageono_dataloader,ono_dataloader):
    encoder.train()
    decoder.train()
    image_model.train()
    train_ono_loss=0
    train_img_loss=0
    train_recon_loss=0
    train_s_loss=0
    train_imgono_loss=0
    train_total_loss=0
    train_img2ono_loss=0
    train_ono2img_loss=0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    learning_rate = 1e-6 #10e-6→10e-8

    image_weight=1#画像のLossにかける重み 1
    diff_weight=1e-1 #2つの値を近づける重み1
    ono_weight=1e-2 #オノマトペの重み e-2
    recon_weight=1e-4 #画像の復元部分におけるLossの重み

    size=64
    criterion      = nn.CrossEntropyLoss() #これをバッチサイズ分繰り返してそれをエポック回数分まわす？
    mse=nn.MSELoss()
    cos=nn.CosineEmbeddingLoss()
    # TripletLossの定義
    # triplet_loss = nn.TripletMarginLoss(margin=1.0)



    encoder_optimizer = optim.Adam( encoder.parameters(), lr=learning_rate )
    decoder_optimizer = optim.Adam( decoder.parameters(), lr=learning_rate )
    image_optimizer=optim.Adam(image_model.parameters(), lr=learning_rate) #学習率は標準で10e-4
    for batch_num,(ono,phoneme) in tqdm.tqdm(enumerate(ono_dataloader),total=len(ono_dataloader)): #音素単体の復元
            batch_ono_loss=0 #1バッチ分のLossをここに入れていく
            for data_num in range(ono_dataloader.batch_size):
    
                ono2_tensor=tensorFromSentence(lang,phoneme[data_num]).to(device) #一つ一つの音素をインデックス番号に変換する

                encoder_hidden              = encoder.initHidden()
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                input_length  = ono2_tensor.size(0)
        
                for i in range( input_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                    encoder_output, encoder_hidden = encoder( ono2_tensor[ i ], encoder_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！  
                # Decoder phese
                loss_ono = 0 #seq2seqのloss
                decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to( device )
                decoder_hidden = encoder_hidden
                for i in range( input_length ):
                    decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) 
                        
                    decoder_input = ono2_tensor[ i ] #次の音素（インデックス）をセットしておく
                    if random.random() < 0.5: 
                        topv, topi                     = decoder_output.topk( 1 )
                        decoder_input                  = topi.squeeze().detach() # detach from history as input
                    loss_ono += criterion( decoder_output, ono2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
                    topv, topi = decoder_output.data.topk(1)

                    if topi.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了
    
                loss_ono.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                #1バッチ分(８個のデータ)のLossになるように加算していく        
                batch_ono_loss +=loss_ono
            # print(batch_loss/dataloader.batch_size) #１バッチにおける1個のデータあたりのLossを計算する
            train_ono_loss +=batch_ono_loss/ono_dataloader.batch_size

        #(total_loss)をバッチの個数分で割ることで１データローダにおけるLossがわかる   
    train_ono_loss=train_ono_loss/len(ono_dataloader) 

    for batch_num, (IMG,PATH,_,_) in enumerate(tqdm.tqdm(imageono_dataloader)): #画像単体での復元
            batch_img_loss=0
            batch_recon_loss=0
            batch_s_loss=0


            for data_num in range(imageono_dataloader.batch_size): 

                IMG_tensor=IMG[data_num].to(device) #画像のテンソル
                IMG_input=IMG_tensor.view(-1,3,size,size)

                image_optimizer.zero_grad()


                img_output,hidden=image_model(IMG_input)       


                loss_img=0 #画像のloss
  
                loss_recon =mse(img_output,IMG_input) #画像 Loss
                loss_reconw=loss_recon*recon_weight #重みをかける
                s_loss, gram_diffs = style_loss_and_diffs(img_output, IMG_input, image_model)
                # losses = {}
                # for name, diff in gram_diffs.items():
                #     writer.add_scalars('GramDiffs/', {f"{name}" + "train":diff.item()}, epoch)
                # writer.add_scalars('recon_loss/',{"train":loss_recon.item()},epoch)
                loss_img=loss_reconw+s_loss
                loss_img=loss_img

                loss_img.backward()
                image_optimizer.step()
                batch_recon_loss +=loss_recon
                batch_s_loss +=s_loss
                batch_img_loss +=loss_img
            train_img_loss +=batch_img_loss/imageono_dataloader.batch_size
            train_recon_loss +=batch_recon_loss/imageono_dataloader.batch_size
            train_s_loss +=batch_s_loss/imageono_dataloader.batch_size
    for batch_num, (IMG,_,_,PHONEME) in enumerate(tqdm.tqdm(imageono_dataloader)): #画像と音素を近づける
            batch_imgono_loss=0

            for data_num in range(imageono_dataloader.batch_size): 

                IMG_tensor=IMG[data_num].to(device) #画像のテンソル
                IMG_input=IMG_tensor.view(-1,3,size,size)
                ONO2_tensor=tensorFromSentence(lang,PHONEME[data_num]).to(device) #一つ一つの音素をインデックス番号に変換する

                encoder_hidden              = encoder.initHidden()
                ENCODER_hidden=encoder.initHidden()
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                image_optimizer.zero_grad()

                INPUT_length=ONO2_tensor.size(0)

                for i in range( INPUT_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                    encoder_output, ENCODER_hidden = encoder( ONO2_tensor[ i ], ENCODER_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！   


                img_output,hidden=image_model(IMG_input)       


                img_hidden=hidden.to(device)
                IMG_hidden=img_hidden.view(-1,1,128)


                loss_imgono=0 #特徴ベクトルを近づけるloss


                ENCODER_hidden = ENCODER_hidden.squeeze(1)
                IMG_hidden = IMG_hidden.squeeze(1)
                loss_imgono=cos(ENCODER_hidden,IMG_hidden,torch.tensor([1.0]).to(device))
            

                loss_imgono=loss_imgono

                loss_imgono.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                image_optimizer.step()
                batch_imgono_loss +=loss_imgono
            train_imgono_loss +=batch_imgono_loss/imageono_dataloader.batch_size

    for batch_num, (IMG,_,_,PHONEME) in enumerate(tqdm.tqdm(imageono_dataloader)): #互いのモーダルで復元
            batch_img2ono_loss=0
            batch_ono2img_loss=0
            max_length=20
            for data_num in range(imageono_dataloader.batch_size): 

                IMG_tensor=IMG[data_num].to(device) #画像のテンソル
                IMG_input=IMG_tensor.view(-1,3,size,size)
                ONO2_tensor=tensorFromSentence(lang,PHONEME[data_num]).to(device) #一つ一つの音素をインデックス番号に変換する

                ENCODER_hidden=encoder.initHidden()

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                image_optimizer.zero_grad()

                INPUT_length=ONO2_tensor.size(0)

                for i in range( INPUT_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                    encoder_output, ENCODER_hidden = encoder( ONO2_tensor[ i ], ENCODER_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！   


                _,hidden=image_model(IMG_input)     #img_hidden→[1,128]  ENCODER_hidden→[1,1,128]

                img_hidden=hidden.to(device)


                ENCODER_hidden = ENCODER_hidden.squeeze(1)
                IMG_hidden = img_hidden.unsqueeze(1)
                loss_img2ono=0
                decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to( device )
                decoder_hidden = IMG_hidden
                for i in range( INPUT_length ):
                    decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) 
                        
                    decoder_input = ONO2_tensor[ i ] #次の音素（インデックス）をセットしておく
                    if random.random() < 0.5: 
                        topv, topi                     = decoder_output.topk( 1 )
                        decoder_input                  = topi.squeeze().detach() # detach from history as input
                    loss_img2ono += criterion( decoder_output, ONO2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
                    topv, topi = decoder_output.data.topk(1)

                    if topi.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了
                loss_img2ono.backward()
              
                ono_output=image_model.decoder(ENCODER_hidden)
                loss_ono2img=mse(ono_output,IMG_input)
                loss_ono2img.backward()

                encoder_optimizer.step()
                decoder_optimizer.step()
                image_optimizer.step()
                




                batch_img2ono_loss +=loss_img2ono
                batch_ono2img_loss +=loss_ono2img
            train_img2ono_loss +=batch_img2ono_loss/imageono_dataloader.batch_size
            train_ono2img_loss +=batch_ono2img_loss/imageono_dataloader.batch_size
    # for batch_num, (IMG,PATH,_,PHONEME) in enumerate(tqdm.tqdm(imageono_dataloader)):
    #     # batch_ono_loss=0
    #     batch_img_loss=0
    #     batch_recon_loss=0
    #     batch_s_loss=0
    #     batch_imgono_loss=0
    #     batch_total_loss=0

    #     for data_num in range(imageono_dataloader.batch_size): 
    #         #データセットの確認--------------------------------
    #         # plt.imshow(IMG.cpu().data[0, :, :, :].permute(1, 2, 0))
    #         # plt.savefig("./output/test.jpg")
    #         # print(PATH[0])
    #         # sys.exit()
    #         #-------------------------------------------------
    #         # img_tensor=img[data_num].to(device) #画像のテンソル
    #         # img_input=img_tensor.view(-1,3,size,size)
    #         IMG_tensor=IMG[data_num].to(device) #画像のテンソル
    #         IMG_input=IMG_tensor.view(-1,3,size,size)

    #         # #Triplet追加用
    #         # PIMG_tensor=PIMG[data_num].to(device) #画像のテンソル
    #         # PIMG_input=PIMG_tensor.view(-1,3,size,size)
    #         # NIMG_tensor=NIMG[data_num].to(device) #画像のテンソル
    #         # NIMG_input=NIMG_tensor.view(-1,3,size,size)


    #         ONO2_tensor=tensorFromSentence(lang,PHONEME[data_num]).to(device) #一つ一つの音素をインデックス番号に変換する

    #         encoder_hidden              = encoder.initHidden()
    #         ENCODER_hidden=encoder.initHidden()
    #         encoder_optimizer.zero_grad()
    #         decoder_optimizer.zero_grad()
    #         image_optimizer.zero_grad()

    #         INPUT_length=ONO2_tensor.size(0)

    #         for i in range( INPUT_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
    #             encoder_output, ENCODER_hidden = encoder( ONO2_tensor[ i ], ENCODER_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！   


    #         img_output,hidden=image_model(IMG_input)       

          
    #         # #Triplet用追加
    #         # _,Phidden=image_model(PIMG_input)
    #         # _,Nhidden=image_model(NIMG_input)

    #         img_hidden=hidden.to(device)
    #         IMG_hidden=img_hidden.view(-1,1,128)

    #         #triplet用追加
    #         # Pimg_hidden=Phidden.to(device)
    #         # PIMG_hidden=Pimg_hidden.view(-1,1,128)
    #         # Nimg_hidden=Nhidden.to(device)
    #         # NIMG_hidden=Nimg_hidden.view(-1,1,128)

    #         # Decoder phese
    #         # loss_ono = 0 #音素のlosss
    #         loss_img=0 #画像のloss
    #         loss_imgono=0 #特徴ベクトルを近づけるloss

    #         # decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to( device )
    #         # decoder_hidden = encoder_hidden
    #         # for i in range( input_length ):
    #         #     decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) 
                    
    #         #     decoder_input = ONO_tensor[ i ] #次の音素（インデックス）をセットしておく
    #         #     if random.random() < 0.5: 
    #         #         topv, topi                     = decoder_output.topk( 1 )

    #         #         decoder_input                  = topi.squeeze().detach() # detach from history as input
    #         #     loss_ono += criterion( decoder_output, ONO_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
    #         #     topv, topi = decoder_output.data.topk(1)

    #         #     if topi.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了



    #         loss_recon =mse(img_output,IMG_input) #画像 Loss
    #         loss_reconw=loss_recon*recon_weight #重みをかける
    #         s_loss, gram_diffs = style_loss_and_diffs(img_output, IMG_input, image_model)
    #         losses = {}
    #         # for name, diff in gram_diffs.items():
    #         #     writer.add_scalars('GramDiffs/', {f"{name}" + "train":diff.item()}, epoch)
    #         # writer.add_scalars('recon_loss/',{"train":loss_recon.item()},epoch)
    #         loss_img=loss_reconw+s_loss
    #         loss_img=loss_img*image_weight
    #         ENCODER_hidden = ENCODER_hidden.squeeze(1)
    #         IMG_hidden = IMG_hidden.squeeze(1)
    #         loss_imgono=cos(ENCODER_hidden,IMG_hidden,torch.tensor([1.0]).to(device))
           

    #         loss_imgono=loss_imgono*diff_weight
    #         loss_ono_w=loss_ono*ono_weight



            
    #         loss=loss_img + loss_imgono
    #         loss.backward()
    #         encoder_optimizer.step()
    #         decoder_optimizer.step()
    #         image_optimizer.step()

            #1バッチ分(８個のデータ)のLossになるように加算していく        
            # batch_ono_loss +=loss_ono 

            # batch_total_loss +=loss
     
        # print(batch_loss/dataloader.batch_size) #１バッチにおける1個のデータあたりのLossを計算する

        # train_ono_loss +=batch_ono_loss/imageono_dataloader.batch_size

        

        # train_total_loss +=batch_total_loss/imageono_dataloader.batch_size

    #(total_loss)をバッチの個数分で割ることで１データローダにおけるLossがわかる   
    # train_ono_loss=train_ono_loss/len(imageono_dataloader) 
    train_img_loss=train_img_loss/len(imageono_dataloader) 
    train_recon_loss =train_recon_loss/len(imageono_dataloader) 
    train_s_loss =train_s_loss/len(imageono_dataloader) 
    train_imgono_loss=train_imgono_loss/len(imageono_dataloader) 
    train_total_loss=train_total_loss/len(imageono_dataloader) +train_ono_loss
    train_img2ono_loss=train_img2ono_loss/len(imageono_dataloader)
    train_ono2img_loss=train_ono2img_loss/len(imageono_dataloader)

    return train_ono_loss,train_img_loss,train_recon_loss,train_s_loss,train_imgono_loss,train_img2ono_loss,train_ono2img_loss,encoder,decoder,image_model

def Validation(encoder,decoder,img_model,lang,imageono_dataloader,ono_dataloader):
    encoder.eval()
    decoder.eval()
    img_model.eval()
    with torch.no_grad():
        valid_ono_loss=0
        valid_img_loss=0
        valid_recon_loss=0
        valid_s_loss=0
        valid_imgono_loss=0
        valid_total_loss=0
        valid_ono2img_loss=0
        valid_img2ono_loss=0
        image_weight=1 #segnetにかける重み 1
        recon_weight=1e-4 #画像復元の重みe-4
        weight=1e-1 #2つの値を近づける重みe-2
        ono_weight=1e-2 #オノマトペの重み e-3
        size=64
        max_length=20

        criterion=nn.CrossEntropyLoss()
        mse=nn.MSELoss()
        cos=nn.CosineEmbeddingLoss()
        # TripletLossの定義
        # triplet_loss = nn.TripletMarginLoss(margin=1.0)

        for batch_num,(ono,phoneme) in enumerate(tqdm.tqdm(ono_dataloader)):
            batch_ono_loss=0 #1バッチ分のLossをここに入れていく

            for data_num in range(ono_dataloader.batch_size):
                ono2_tensor=tensorFromSentence(lang,phoneme[data_num]).to(device) #一つ一つの音素をインデックス番号に変換する
                encoder_hidden              = encoder.initHidden()
                input_length  = ono2_tensor.size(0)
                for i in range( input_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                    encoder_output, encoder_hidden = encoder( ono2_tensor[ i ], encoder_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！  
                loss_ono = 0 #seq2seqのloss
                decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to( device )
                decoder_hidden = encoder_hidden
                for i in range( input_length ):
                    decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) 
                        
                    decoder_input = ono2_tensor[ i ] #次の音素（インデックス）をセットしておく
                    if random.random() < 0.5: 
                        topv, topi                     = decoder_output.topk( 1 )
                    
                        decoder_input                  = topi.squeeze().detach() # detach from history as input
                    loss_ono += criterion( decoder_output, ono2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
                    topv, topi = decoder_output.data.topk(1)
                    # print(loss_ono,"lossono"+str(i),ono2_tensor[i],topi.item())
              

                    if topi.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了
                    loss=loss_ono
                batch_ono_loss +=loss
            valid_ono_loss+=batch_ono_loss/ono_dataloader.batch_size

        for batch_num, (IMG,PATH,_,PHONEME) in enumerate(tqdm.tqdm(imageono_dataloader)):
            #データセットの確認--------------------------------
            # plt.imshow(IMG.cpu().data[0, :, :, :].permute(1, 2, 0))
            # plt.savefig("./output/test.jpg")
            # print(PATH[0])
            # sys.exit()
            #-------------------------------------------------            
            batch_img_loss=0
            batch_recon_loss=0
            batch_s_loss=0
            batch_imgono_loss=0
            batch_total_loss=0
            batch_ono2img_loss=0
            batch_img2ono_loss=0
            for data_num in range(imageono_dataloader.batch_size): 
                
                # img_input=img_tensor.view(-1,3,size,size)
                IMG_tensor=IMG[data_num].to(device) #画像のテンソル
                IMG_input=IMG_tensor.view(-1,3,size,size)

                # #Triplet追加用
                # PIMG_tensor=PIMG[data_num].to(device) #画像のテンソル
                # PIMG_input=PIMG_tensor.view(-1,3,size,size)
                # NIMG_tensor=NIMG[data_num].to(device) #画像のテンソル
                # NIMG_input=NIMG_tensor.view(-1,3,size,size)

                # ono2_tensor=tensorFromSentence(lang,phoneme[data_num]).to(device) #一つ一つの音素をインデックス番号に変換する
                ONO2_tensor=tensorFromSentence(lang,PHONEME[data_num]).to(device) #一つ一つの音素をインデックス番号に変換する
                encoder_hidden              = encoder.initHidden()
                ENCODER_hidden=encoder.initHidden()
                # input_length  = ono2_tensor.size(0)
                INPUT_length=ONO2_tensor.size(0)
                # for i in range( input_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                #     encoder_output, encoder_hidden = encoder( ono2_tensor[ i ], encoder_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！  
                for i in range( INPUT_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                    encoder_output, ENCODER_hidden = encoder( ONO2_tensor[ i ], ENCODER_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！   

                img_output,hidden=img_model(IMG_input)

                # #Triplet用追加
                # _,Phidden=img_model(PIMG_input)
                # _,Nhidden=img_model(NIMG_input)

                img_hidden=hidden.to(device)
                IMG_hidden=img_hidden.view(-1,1,128)

                # #triplet用追加
                # Pimg_hidden=Phidden.to(device)
                # PIMG_hidden=Pimg_hidden.view(-1,1,128)
                # Nimg_hidden=Nhidden.to(device)
                # NIMG_hidden=Nimg_hidden.view(-1,1,128)


                # Decoder phese
                loss_img=0 #画像のLoss
                loss_imgono=0 #特徴ベクトルを近づけるloss

                loss_recon =mse(img_output,IMG_input) #画像 Loss
                loss_reconw=loss_recon*recon_weight #重みをかける
                s_loss, gram_diffs = style_loss_and_diffs(img_output, IMG_input, img_model)
                loss_img=loss_reconw+s_loss #再構成とスタイルLossを足し合わせる
                loss_img=loss_img*image_weight #画像のLossに重みをかける

                ENCODER_hidden = ENCODER_hidden.squeeze(1).to(device)
                IMG_hidden = IMG_hidden.squeeze(1).to(device)

                loss_imgono=cos(ENCODER_hidden,IMG_hidden,torch.tensor([1.0]).to(device))
                loss_imgono=loss_imgono
            

                
                # loss=loss_img + loss_imgono

                ENCODER_hidden=ENCODER_hidden.view(-1,128)
                IMG_hidden=IMG_hidden.view(-1,1,128)

                ono_output=img_model.decoder(ENCODER_hidden)
                loss_ono2img=mse(ono_output,IMG_input)
                
   

                loss_img2ono = 0 #seq2seqのloss
                decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to( device )
                decoder_hidden = IMG_hidden
                for i in range( INPUT_length ):
                    decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) 
                        
                    decoder_input = ONO2_tensor[ i ] #次の音素（インデックス）をセットしておく
                    if random.random() < 0.5: 
                        topv, topi                     = decoder_output.topk( 1 )
                    
                        decoder_input                  = topi.squeeze().detach() # detach from history as input
                    loss_img2ono += criterion( decoder_output, ONO2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
                    topv, topi = decoder_output.data.topk(1)
                    # print(loss_ono,"lossono"+str(i),ono2_tensor[i],topi.item())
              

                    if topi.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了
                
                            

                #1バッチ分(８個のデータ)のLossになるように加算していく        
                
                
                batch_img_loss +=loss_img
                batch_recon_loss +=loss_recon
                batch_s_loss +=s_loss
                batch_imgono_loss +=loss_imgono
                # batch_total_loss +=loss
                batch_ono2img_loss+=loss_ono2img
                batch_img2ono_loss +=loss_img2ono    
            
            valid_img_loss+=batch_img_loss/imageono_dataloader.batch_size
            valid_recon_loss+=batch_recon_loss/imageono_dataloader.batch_size
            valid_s_loss+=batch_s_loss/imageono_dataloader.batch_size
            valid_imgono_loss+=batch_imgono_loss/imageono_dataloader.batch_size
            # valid_total_loss+=batch_total_loss/imageono_dataloader.batch_size
            valid_ono2img_loss+=batch_ono2img_loss/imageono_dataloader.batch_size
            valid_img2ono_loss+=batch_img2ono_loss/imageono_dataloader.batch_size
        #(total_loss)をバッチの個数分で割ることで１データローダにおけるLossがわかる      
        valid_ono_loss=valid_ono_loss/len(ono_dataloader) 
        valid_img_loss=valid_img_loss/len(imageono_dataloader)
        valid_recon_loss=valid_recon_loss/len(imageono_dataloader)
        valid_s_loss=valid_s_loss/len(imageono_dataloader) 
        valid_imgono_loss=valid_imgono_loss/len(imageono_dataloader) 
        # valid_total_loss=valid_total_loss/len(imageono_dataloader) 
        # valid_total_loss+=valid_ono_loss
        valid_ono2img_loss=valid_ono2img_loss/len(imageono_dataloader)
        valid_img2ono_loss=valid_img2ono_loss/len(imageono_dataloader)
        
        valid_total_loss=(valid_ono_loss*1e-5)+valid_img_loss+(valid_imgono_loss*1e-2)
        return valid_ono_loss,valid_img_loss,valid_recon_loss,valid_s_loss,valid_imgono_loss,valid_ono2img_loss,valid_img2ono_loss,valid_total_loss
    

def main():
    embedding_size = 128
    hidden_size   = 128
    num=40 #入出力として使える音素の数=データセット内の.n_wordsに等しい
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_model=TextureNet(feature_layers=["0", "5", "10"]).to(device)
    stdict_img = "model/texture_net64.pth"
    image_model.load_state_dict(torch.load(stdict_img))
    encoder           = Encoder( num, embedding_size, hidden_size ).to( device )
    decoder           = Decoder( hidden_size, embedding_size, num ).to( device )
    epochs=10000
    save_loss=3000
    size=64
    batch_size=16
    augment=True
    enfile = "model/firstencoder" #学習済みのエンコーダモデル
    defile = "model/firstdecoder" #学習済みのデコーダモデル
    encoder.load_state_dict( torch.load( enfile ) ) #読み込み
    decoder.load_state_dict( torch.load( defile ) )
    transform = transforms.Compose([transforms.Resize((size, size)),transforms.ToTensor()])

    lang  = Lang( 'dataset/dictionary.csv')


    imageono_train_dataset=ImageLang('dataset/onomatope.csv',"dataset/image/train",transform)
    imageono_train_dataloader = DataLoader(imageono_train_dataset, batch_size=batch_size, shuffle=False,drop_last=True) #drop_lastをtruenにすると最後の中途半端に入っているミニバッチを排除してくれる

    ono_train_dataset  = Lang( 'dataset/dictionary.csv',augment)
    ono_train_dataloader=DataLoader(ono_train_dataset,batch_size=batch_size, shuffle=True,drop_last=True)
    ono_valid_dataset=Lang('dataset/onomatopeunknown.csv')
    ono_valid_dataloader=DataLoader(ono_valid_dataset,batch_size=batch_size,shuffle=False,drop_last=True)    

    imageono_valid_dataset=ImageLang('dataset/onomatope.csv',"dataset/image/valid",transform)
    imageono_valid_dataloader=DataLoader(imageono_valid_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    for epoch in range(epochs):
        writer=SummaryWriter(log_dir="/workspace/mycode/Documents/log/crossmodalcosineseparatecross")
        train_ono,train_img,train_recon,train_s,train_imgono,train_img2ono,train_ono2img,encoder,decoder,image_model=Train(encoder,decoder,image_model,lang,imageono_train_dataloader,ono_train_dataloader)
        valid_ono,valid_img,valid_recon,valid_s,valid_imgono,valid_ono2img,valid_img2ono,valid_total=Validation(encoder,decoder,image_model,lang,imageono_valid_dataloader,ono_valid_dataloader)
        print( "[epoch num %d ] [ train_cosine: %f]" % ( epoch+1, train_imgono) )
        print( "[epoch num %d ] [ valid_cosine: %f]" % ( epoch+1, valid_imgono ) )

        writer.add_scalars('loss/ono',{'train':train_ono,'valid':valid_ono} ,epoch+1)
        
        writer.add_scalars('loss/img',{'train':train_img,'valid':valid_img} ,epoch+1)
        writer.add_scalars('loss/recon',{'train':train_recon,'valid':valid_recon} ,epoch+1)
        writer.add_scalars('loss/s',{'train':train_s,'valid':valid_s} ,epoch+1)
        writer.add_scalars('loss/imgono',{'train':train_imgono,'valid':valid_imgono} ,epoch+1)
        writer.add_scalars('loss/ono2img',{'train':train_ono2img,'valid':valid_ono2img} ,epoch+1)
        writer.add_scalars('loss/img2ono',{'train':train_img2ono,'valid':valid_img2ono} ,epoch+1)


        writer.close()
        if (save_loss >= valid_total):
            torch.save(encoder.state_dict(), 'model/encodercosineseparatecross')
            torch.save(decoder.state_dict(), 'model/decodercosineseparatecross')
            torch.save(image_model.state_dict(),'model/imgmodelcosineseparatecross')
            save_loss=valid_total
            print("-------model 更新---------")
        torch.save(encoder.state_dict(), 'model/encodercheckcosineseparatecross')
        torch.save(decoder.state_dict(), 'model/decodercheckcosineseparatecross')
        torch.save(image_model.state_dict(),'model/imgmodelcheckcosineseparatecross')
if __name__ == '__main__':
    main()
    #追加で書き込み
    