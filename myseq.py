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

        img_path = self.data[index] #適当な画像を取ってくる
        img_label = self.labels[index] #その画像のラベル番号は何番か取ってくる
        ono=self.ono[img_label] #取ってきたラベル番号のオノマトペを取ってくる（オノマトペのラベル番号とリストの番号は一致している）
        
        phoneme = self.sentences[img_label] #同上
    
        
        img = Image.open(img_path).convert("RGB") #img_pathの画像を開く
        img = self.transform(img) #transformする
        return img, img_path, ono, phoneme       

def Train(encoder,decoder,image_model,lang,imageono_dataloader,ono_dataloader):
    encoder.train()
    decoder.train()
    image_model.train()
    train_ono_loss=0
    train_ONO_loss=0
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
    imgono_weight=1e-3 #2つの値を近づける重みe-3
    ono_weight=1e-5 #オノマトペの重み e-5

    recon_weight=1e-4 #画像の復元部分におけるLossの重み

    size=64 #画像のサイズ
    criterion= nn.CrossEntropyLoss() #これをバッチサイズ分繰り返してそれをエポック回数分まわす？
    mse=nn.MSELoss()
    cos=nn.CosineEmbeddingLoss()


    encoder_optimizer = optim.Adam( encoder.parameters(), lr=learning_rate )
    decoder_optimizer = optim.Adam( decoder.parameters(), lr=learning_rate )
    image_optimizer=optim.Adam(image_model.parameters(), lr=learning_rate) #学習率は標準で10e-4

    phoneme_iterator=iter(ono_dataloader) #オノマトペの音素のみが詰まったイテレータ
    imageono_iterator=iter(imageono_dataloader) #画像と音素がバインドされたデータセットのイテレータ
    max_iter=max(len(phoneme_iterator),len(imageono_iterator))

    for _ in tqdm.tqdm(range(max_iter)):
        for j in range(imageono_dataloader.batch_size):
#--------------------------------------------------------------------------音素復元単体            
            try:
                _,phoneme=next(phoneme_iterator)
                phoneme2_tensor=tensorFromSentence(lang,phoneme[j]).to(device)
                encoder_hidden=encoder.initHidden()
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                input_length  = phoneme2_tensor.size(0)  

                for i in range( input_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                    encoder_output, encoder_hidden = encoder( phoneme2_tensor[ i ], encoder_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！  
                # Decoder phese
                loss_ono = 0 #seq2seqのloss
                decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to( device )
                decoder_hidden = encoder_hidden
                for i in range( input_length ):
                    decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) 
                        
                    decoder_input = phoneme2_tensor[ i ] #次の音素（インデックス）をセットしておく
                    if random.random() < 0.5: 
                        topv, topi                     = decoder_output.topk( 1 )
                        decoder_input                  = topi.squeeze().detach() # detach from history as input
                    loss_ono += criterion( decoder_output, phoneme2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
                    topv, topi = decoder_output.data.topk(1)

                    if topi.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了
                loss_ono.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                train_ono_loss += loss_ono
            except StopIteration:
                phoneme_iterator=iter(ono_dataloader)
    #-----------------------------------------------------------      画像復元  
            try:
                IMG,PATH,ONO,PHONEME=next(imageono_iterator)
                IMG_tensor=IMG[j].to(device) #画像のテンソル
                IMG_input=IMG_tensor.view(-1,3,size,size)

                image_optimizer.zero_grad()
                img_output,hidden=image_model(IMG_input)
                


                loss_img=0 #画像のloss

                loss_recon =mse(img_output,IMG_input) #画像 Loss
                loss_reconw=loss_recon*recon_weight #重みをかける
                s_loss, _ = style_loss_and_diffs(img_output, IMG_input, image_model)
                loss_img=loss_reconw+s_loss
                train_s_loss +=s_loss
                train_recon_loss +=loss_recon
                train_img_loss +=loss_img
                loss_img=loss_img *image_weight
    #----------------------------------------------------------　音素復元
                PHONEME2_tensor=tensorFromSentence(lang,PHONEME[j]).to(device)
                ENCODER_hidden=encoder.initHidden()
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                INPUT_length  = PHONEME2_tensor.size(0)  

                for i in range( INPUT_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                    encoder_output, ENCODER_hidden = encoder( PHONEME2_tensor[ i ], ENCODER_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！  
                # Decoder phese
                loss_ONO = 0 #seq2seqのloss
                decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to( device )
                decoder_hidden = ENCODER_hidden
                for i in range( input_length ):
                    decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) 
                        
                    decoder_input = PHONEME2_tensor[ i ] #次の音素（インデックス）をセットしておく
                
                    if random.random() < 0.5: 
                        topv, topi                     = decoder_output.topk( 1 )
                        decoder_input                  = topi.squeeze().detach() # detach from history as input
                    loss_ONO += criterion( decoder_output, PHONEME2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
                    topv, topi = decoder_output.data.topk(1)

                    if topi.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了
                train_ONO_loss += loss_ONO
                loss_ONO=loss_ONO*ono_weight
    #----------------------------------------------------------------　２つを近づける
                img_hidden=hidden.to(device)
                IMG_hidden=img_hidden.view(-1,1,128)
                loss_imgono=0 #特徴ベクトルを近づけるloss
                ENCODER_hidden = ENCODER_hidden.squeeze(1)
                IMG_hidden = IMG_hidden.squeeze(1)

                loss_imgono=cos(ENCODER_hidden,IMG_hidden,torch.tensor([1.0]).to(device))
                
                train_imgono_loss += loss_imgono
                loss_imgono=loss_imgono*imgono_weight
    #-------------------------------------------------------------------
                loss=loss_ONO+loss_img+loss_imgono
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                image_optimizer.step()

                total_loss=loss+(loss_ono*ono_weight)
                train_total_loss += total_loss
            except StopIteration:
                imageono_iterator=iter(imageono_dataloader)

    train_ono_loss=train_ono_loss/(max_iter*(imageono_dataloader.batch_size))
    train_ONO_loss=train_ONO_loss/(max_iter*(imageono_dataloader.batch_size))
    train_img_loss=train_img_loss/(max_iter*(imageono_dataloader.batch_size))
    train_s_loss=train_s_loss/(max_iter*(imageono_dataloader.batch_size))
    train_recon_loss=train_recon_loss/(max_iter*(imageono_dataloader.batch_size))
    train_imgono_loss=train_imgono_loss/(max_iter*(imageono_dataloader.batch_size))
    train_total_loss=train_total_loss/(max_iter*(imageono_dataloader.batch_size))

   


    return train_ono_loss,train_ONO_loss,train_img_loss,train_recon_loss,train_s_loss,train_imgono_loss,train_total_loss,encoder,decoder,image_model

def Validation(encoder,decoder,image_model,lang,imageono_dataloader,ono_dataloader):
    encoder.eval()
    decoder.eval()
    image_model.eval()
    with torch.no_grad():
        valid_ono_loss=0
        valid_ONO_loss=0
        valid_img_loss=0
        valid_recon_loss=0
        valid_s_loss=0
        valid_imgono_loss=0
        valid_total_loss=0
        valid_ono2img_loss=0
        valid_img2ono_loss=0



        size=64

        image_weight=1#画像のLossにかける重み 1
        imgono_weight=1e-3 #2つの値を近づける重みe-3
        ono_weight=1e-5 #オノマトペの重み e-5

        recon_weight=1e-4 #画像の復元部分におけるLossの重み

        criterion=nn.CrossEntropyLoss()
        mse=nn.MSELoss()
        cos=nn.CosineEmbeddingLoss()
        phoneme_iterator=iter(ono_dataloader) #オノマトペの音素のみが詰まったイテレータ
        imageono_iterator=iter(imageono_dataloader) #画像と音素がバインドされたデータセットのイテレータ
        max_iter=max(len(phoneme_iterator),len(imageono_iterator))

        for _ in tqdm.tqdm(range(max_iter)):
            for j in range(imageono_dataloader.batch_size):
    #--------------------------------------------------------------------------音素復元単体            
                try:
                    _,phoneme=next(phoneme_iterator)
                    phoneme2_tensor=tensorFromSentence(lang,phoneme[j]).to(device)
                    encoder_hidden=encoder.initHidden()
                    input_length  = phoneme2_tensor.size(0)  

                    for i in range( input_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                        encoder_output, encoder_hidden = encoder( phoneme2_tensor[ i ], encoder_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！  
                    # Decoder phese
                    loss_ono = 0 #seq2seqのloss
                    decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to( device )
                    decoder_hidden = encoder_hidden
                    for i in range( input_length ):
                        decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) 
                            
                        decoder_input = phoneme2_tensor[ i ] #次の音素（インデックス）をセットしておく
                        if random.random() < 0.5: 
                            topv, topi                     = decoder_output.topk( 1 )
                            decoder_input                  = topi.squeeze().detach() # detach from history as input
                        loss_ono += criterion( decoder_output, phoneme2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
                        topv, topi = decoder_output.data.topk(1)

                        if topi.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了
                    valid_ono_loss += loss_ono
                except StopIteration:
                    phoneme_iterator=iter(ono_dataloader)
        #-----------------------------------------------------------      画像復元  
                try:
                    IMG,PATH,ONO,PHONEME=next(imageono_iterator)
                    IMG_tensor=IMG[j].to(device) #画像のテンソル
                    IMG_input=IMG_tensor.view(-1,3,size,size)

                    img_output,hidden=image_model(IMG_input)
                    


                    loss_img=0 #画像のloss

                    loss_recon =mse(img_output,IMG_input) #画像 Loss
                    loss_reconw=loss_recon*recon_weight #重みをかける
                    s_loss, _ = style_loss_and_diffs(img_output, IMG_input, image_model)
                    loss_img=loss_reconw+s_loss
                    valid_s_loss +=s_loss
                    valid_recon_loss +=loss_recon
                    valid_img_loss +=loss_img
                    loss_img=loss_img *image_weight
        #----------------------------------------------------------　音素復元
                    PHONEME2_tensor=tensorFromSentence(lang,PHONEME[j]).to(device)
                    ENCODER_hidden=encoder.initHidden()

                    INPUT_length  = PHONEME2_tensor.size(0)  

                    for i in range( INPUT_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                        encoder_output, ENCODER_hidden = encoder( PHONEME2_tensor[ i ], ENCODER_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！  
                    # Decoder phese
                    loss_ONO = 0 #seq2seqのloss
                    decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to( device )
                    decoder_hidden = ENCODER_hidden
                    for i in range( input_length ):
                        decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) 
                            
                        decoder_input = PHONEME2_tensor[ i ] #次の音素（インデックス）をセットしておく
                    
                        if random.random() < 0.5: 
                            topv, topi                     = decoder_output.topk( 1 )
                            decoder_input                  = topi.squeeze().detach() # detach from history as input
                        loss_ONO += criterion( decoder_output, PHONEME2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
                        topv, topi = decoder_output.data.topk(1)

                        if topi.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了
                    valid_ONO_loss += loss_ONO

                    loss_ONO=loss_ONO*ono_weight
        #----------------------------------------------------------------　２つを近づける
                    img_hidden=hidden.to(device)
                    IMG_hidden=img_hidden.view(-1,1,128)
                    loss_imgono=0 #特徴ベクトルを近づけるloss
                    ENCODER_hidden = ENCODER_hidden.squeeze(1)
                    IMG_hidden = IMG_hidden.squeeze(1)

                    loss_imgono=cos(ENCODER_hidden,IMG_hidden,torch.tensor([1.0]).to(device))
                    valid_imgono_loss += loss_imgono
                    loss_imgono=loss_imgono*imgono_weight
        #-------------------------------------------------------------------
                    loss=loss_ONO+loss_img+loss_imgono


                    total_loss=loss+(loss_ono*ono_weight)
                    valid_total_loss += total_loss
                except StopIteration:
                    imageono_iterator=iter(imageono_dataloader)

        valid_ono_loss=valid_ono_loss/(max_iter*(imageono_dataloader.batch_size))
        valid_ONO_loss=valid_ONO_loss/(max_iter*(imageono_dataloader.batch_size))
        valid_img_loss=valid_img_loss/(max_iter*(imageono_dataloader.batch_size))
        valid_s_loss=valid_s_loss/(max_iter*(imageono_dataloader.batch_size))
        valid_recon_loss=valid_recon_loss/(max_iter*(imageono_dataloader.batch_size))
        valid_imgono_loss=valid_imgono_loss/(max_iter*(imageono_dataloader.batch_size))
        valid_total_loss=valid_total_loss/(max_iter*(imageono_dataloader.batch_size))
        return valid_ono_loss,valid_ONO_loss,valid_img_loss,valid_recon_loss,valid_s_loss,valid_imgono_loss,valid_total_loss
    

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
    imageono_train_dataloader = DataLoader(imageono_train_dataset, batch_size=batch_size, shuffle=True,drop_last=True) #drop_lastをtruenにすると最後の中途半端に入っているミニバッチを排除してくれる
    imageono_valid_dataset=ImageLang('dataset/onomatope.csv',"dataset/image/valid",transform)
    imageono_valid_dataloader=DataLoader(imageono_valid_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

    
    ono_train_dataset  = Lang( 'dataset/dictionary.csv',augment)
    ono_train_dataloader=DataLoader(ono_train_dataset,batch_size=batch_size, shuffle=True,drop_last=True)
    ono_valid_dataset=Lang('dataset/onomatopeunknown.csv')
    ono_valid_dataloader=DataLoader(ono_valid_dataset,batch_size=batch_size,shuffle=False,drop_last=True)    


    for epoch in range(epochs):
        writer=SummaryWriter(log_dir="/workspace/mycode/Documents/log/crossmodalcosine2")
        train_ono,train_ONO,train_img,train_recon,train_s,train_imgono,train_total,encoder,decoder,image_model=Train(encoder,decoder,image_model,lang,imageono_train_dataloader,ono_train_dataloader)
        valid_ono,valid_ONO,valid_img,valid_recon,valid_s,valid_imgono,valid_total=Validation(encoder,decoder,image_model,lang,imageono_valid_dataloader,ono_valid_dataloader)
        print( "[epoch num %d ] [ train_cosine: %f]" % ( epoch+1, train_imgono) )
        print( "[epoch num %d ] [ valid_cosine: %f]" % ( epoch+1, valid_imgono ) )

        writer.add_scalars('loss/ono',{'train':train_ono,'valid':valid_ono} ,epoch+1)
        writer.add_scalars('loss/img',{'train':train_img,'valid':valid_img} ,epoch+1)
        writer.add_scalars('loss/recon',{'train':train_recon,'valid':valid_recon} ,epoch+1)
        writer.add_scalars('loss/s',{'train':train_s,'valid':valid_s} ,epoch+1)
        writer.add_scalars('loss/imgono',{'train':train_imgono,'valid':valid_imgono} ,epoch+1)
        writer.add_scalars('loss/total',{'train':train_total,'valid':valid_total} ,epoch+1)

        # writer.add_scalars('loss/ono2img',{'train':train_ono2img,'valid':valid_ono2img} ,epoch+1)
        # writer.add_scalars('loss/img2ono',{'train':train_img2ono,'valid':valid_img2ono} ,epoch+1)


        writer.close()
        if (save_loss >= valid_total):
            torch.save(encoder.state_dict(), 'model/encodercosine2')
            torch.save(decoder.state_dict(), 'model/decodercosine2')
            torch.save(image_model.state_dict(),'model/imgmodelcosine2')
            save_loss=valid_total
            print("-------model 更新---------")
        torch.save(encoder.state_dict(), 'model/encodercheckcosine2')
        torch.save(decoder.state_dict(), 'model/decodercheckcosine2')
        torch.save(image_model.state_dict(),'model/imgmodelcheckcosine2')
if __name__ == '__main__':
    main()
    