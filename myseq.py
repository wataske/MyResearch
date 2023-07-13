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

from seq2seq import Lang
from seq2seq import Encoder
from seq2seq import Decoder
from seq2seq import tensorFromSentence
from cae import CAE


SOS_token = 0
EOS_token = 1

device = "cuda" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageLang:
    def __init__( self, filename,dir,transform ): #呼び出されたとき、最初に行うこと
        max_length    = 20
        self.transform = transform
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
        # self.target_sentences = self.sentences[ :: ]

        target_dir = os.path.join(dir, "*")
        # path=glob.glob(target_dir)
        for path in glob.glob(target_dir):

            name = os.path.splitext(os.path.basename(path))[0]
            name=unicodedata.normalize("NFKC",name)
            label = name_to_label[name]
            self.data.append(path)
            self.labels.append(label)  
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
    def __getitem__(self, index): #dataloaderを作ったときにできるやつ
        """サンプルを返す。
        """
        
        img_path = self.data[index] #適当な画像を取ってくる
        img_label = self.labels[index] #その画像のラベル番号は何番か取ってくる
        
        ono=self.ono[img_label] #取ってきたラベル番号のオノマトペを取ってくる（オノマトペのラベル番号とリストの番号は一致している）
        phoneme = self.sentences[img_label] #同上

        img = Image.open(img_path).convert("RGB") #img_pathの画像を開く

        img = self.transform(img) #transformする

        return img, img_path, ono, phoneme


mm = preprocessing.MinMaxScaler()
# Start core part

def Train(encoder,decoder,image_model,lang,dataloader):
    encoder.train()
    decoder.train()
    image_model.train()
    train_ono_loss=0
    train_img_loss=0
    train_imgono_loss=0
    train_total_loss=0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    learning_rate = 1e-5 #10e-6
    batch_size =8
    seg_weight=1 #segnetにかける重み 10e-5
    weight=1 #2つの値を近づける重み1
    size=32
    criterion      = nn.CrossEntropyLoss() #これをバッチサイズ分繰り返してそれをエポック回数分まわす？
    mse=nn.MSELoss()




    encoder_optimizer = optim.Adam( encoder.parameters(), lr=learning_rate )
    decoder_optimizer = optim.Adam( decoder.parameters(), lr=learning_rate )
    segnet_optimizer=optim.Adam(image_model.parameters()) #学習率は標準で10e-4
    for batch_num,(img,path,ono,phoneme) in tqdm.tqdm(enumerate(dataloader),total=len(dataloader)):
        batch_ono_loss=0 #1バッチ分のLossをここに入れていく
        batch_img_loss=0
        batch_imgono_loss=0
        batch_total_loss=0
        for data_num in range(dataloader.batch_size): 
            img_tensor=img[data_num].to(device) #画像のテンソル
            img_input=img_tensor.view(-1,3,size,size)
            ono2_tensor=tensorFromSentence(lang,phoneme[data_num]).to(device) #一つ一つの音素をインデックス番号に変換する
            encoder_hidden              = encoder.initHidden()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            segnet_optimizer.zero_grad()
            input_length  = ono2_tensor.size(0)
            for i in range( input_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                encoder_output, encoder_hidden = encoder( ono2_tensor[ i ], encoder_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！  

            img_output,hidden=image_model(img_input) #segnetモデルに渡す

            img_hidden=hidden.to(device)
            img_hidden=img_hidden.view(-1,1,128)
            # Decoder phese
            loss_ono = 0 #seq2seqのloss
            loss2=0 #segnetのloss
            loss3=0 #特徴ベクトルを近づけるloss
            decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to( device )
            decoder_hidden = encoder_hidden
            for i in range( input_length ):
                decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) 
                    
                decoder_input = ono2_tensor[ i ] #次の音素（インデックス）をセットしておく
                if random.random() < 0.5: 
                    topv, topi                     = decoder_output.topk( 1 )
                    # print(topi)
                    # sys.exit()
                    decoder_input                  = topi.squeeze().detach() # detach from history as input
                loss_ono += criterion( decoder_output, ono2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
                topv, topi = decoder_output.data.topk(1)

                if decoder_input.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了

            loss2 =mse(img_output,img_input) #画像と単語を近づけるLoss
            loss_img=loss2*seg_weight #画像と単語を近づけるLossに重みをかける
            loss3=mse(img_hidden,encoder_hidden)
            loss_imgono=loss3*weight


            
            loss=loss_ono + loss_img + loss_imgono
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            segnet_optimizer.step()

            #1バッチ分(８個のデータ)のLossになるように加算していく        
            batch_ono_loss +=loss_ono 
            batch_img_loss +=loss2
            batch_imgono_loss +=loss3
            batch_total_loss +=loss
        # print(batch_loss/dataloader.batch_size) #１バッチにおける1個のデータあたりのLossを計算する

        train_ono_loss +=batch_ono_loss/dataloader.batch_size
        train_img_loss +=batch_img_loss/dataloader.batch_size
        train_imgono_loss +=batch_imgono_loss/dataloader.batch_size
        train_total_loss +=batch_total_loss/dataloader.batch_size

    #(total_loss)をバッチの個数分で割ることで１データローダにおけるLossがわかる   
    train_ono_loss=train_ono_loss/len(dataloader) 
    train_img_loss=train_img_loss/len(dataloader) 
    train_imgono_loss=train_imgono_loss/len(dataloader) 
    train_total_loss=train_total_loss/len(dataloader) 

    return train_ono_loss,train_img_loss,train_imgono_loss,train_total_loss,encoder,decoder,image_model

def Validation(encoder,decoder,img_model,lang,dataloader,ono_dataloader):
    encoder.eval()
    decoder.eval()
    img_model.eval()
    with torch.no_grad():
        batch_size =8 
        valid_ono_loss=0
        valid_img_loss=0
        valid_imgono_loss=0
        valid_total_loss=0
        seg_weight=1 #segnetにかける重み 10e-3
        weight=1 #2つの値を近づける重み10e-3
        # transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        # lang  = Lang( 'dataset/onomatope.csv',"dataset/image/train copy",transform)
        # valid_lang  = Lang( 'dataset/onomatopev.csv',"dataset/image/valid",transform)
        # valid_dataloader = DataLoader(valid_lang, batch_size=batch_size, shuffle=True,drop_last=True) #drop_lastをtruenにすると最後の中途半端に入っているミニバッチを排除してくれる
        criterion=nn.CrossEntropyLoss()
        mse=nn.MSELoss()
        for batch_num,(ono,phoneme) in tqdm.tqdm(enumerate(ono_dataloader),total=len(ono_dataloader)):
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
                    decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) #
                        
                    decoder_input = ono2_tensor[ i ] #次の音素（インデックス）をセットしておく
                    if random.random() < 0.5: 
                        topv, topi                     = decoder_output.topk( 1 )
                        # print(topi)
                        # sys.exit()
                        decoder_input                  = topi.squeeze().detach() # detach from history as input
                    loss_ono += criterion( decoder_output, ono2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
                    topv, topi = decoder_output.data.topk(1)
                    if decoder_input.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了

                batch_ono_loss +=loss_ono 
            valid_ono_loss+=batch_ono_loss/ono_dataloader.batch_size
        for batch_num,(img,path,ono,phoneme) in tqdm.tqdm(enumerate(dataloader),total=len(dataloader)):
            batch_ono_loss=0 #1バッチ分のLossをここに入れていく
            batch_img_loss=0
            batch_imgono_loss=0
            batch_total_loss=0
            for data_num in range(dataloader.batch_size): 
                img_tensor=img[data_num].to(device) #画像のテンソル
                img_input=img_tensor.view(-1,3,32,32)
                img_output,hidden=img_model(img_input) #segnetモデルに渡す
                img_hidden=hidden.to(device)
                img_hidden=img_hidden.view(-1,1,128)                

                ono2_tensor=tensorFromSentence(lang,phoneme[data_num]).to(device) #一つ一つの音素をインデックス番号に変換する
                encoder_hidden              = encoder.initHidden()
                input_length  = ono2_tensor.size(0)
                for i in range( input_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                    encoder_output, encoder_hidden = encoder( ono2_tensor[ i ], encoder_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！           
                loss2=0 #segnetのloss
                loss3=0 #特徴ベクトルを近づけるloss

                loss2 =mse(img_output,img_input) #画像と単語を近づけるLoss
                loss_img=loss2*seg_weight #画像と単語を近づけるLossに重みをかける
                loss3=mse(img_hidden,encoder_hidden)
                loss_imgono=loss3*weight


                
                loss=loss_img + loss_imgono

                #1バッチ分(８個のデータ)のLossになるように加算していく        
                
                batch_img_loss +=loss2
                batch_imgono_loss +=loss3
                batch_total_loss +=loss
            
            
            valid_img_loss+=batch_img_loss/dataloader.batch_size
            valid_imgono_loss+=batch_imgono_loss/dataloader.batch_size
            valid_total_loss+=batch_total_loss/dataloader.batch_size

        #(total_loss)をバッチの個数分で割ることで１データローダにおけるLossがわかる      
        valid_ono_loss=valid_ono_loss/len(ono_dataloader) 
        valid_img_loss=valid_img_loss/len(dataloader) 
        valid_imgono_loss=valid_imgono_loss/len(dataloader) 
        valid_total_loss=valid_total_loss/len(dataloader) 

        return valid_ono_loss,valid_img_loss,valid_imgono_loss,valid_total_loss
    

def main():
    embedding_size = 128
    hidden_size   = 128
    num=40 #入出力として使える音素の数=データセット内の.n_wordsに等しい
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_model=CAE().to(device)
    stdict_img = "model/firstimgmodel"
    image_model.load_state_dict(torch.load(stdict_img))
    encoder           = Encoder( num, embedding_size, hidden_size ).to( device )
    decoder           = Decoder( hidden_size, embedding_size, num ).to( device )
    epochs=10000
    save_loss=3000
    size=32
    batch_size=8
    enfile = "model/firstencoder" #学習済みのエンコーダモデル
    defile = "model/firstdecoder" #学習済みのデコーダモデル
    encoder.load_state_dict( torch.load( enfile ) ) #読み込み
    decoder.load_state_dict( torch.load( defile ) )
    # transform = transforms.Compose([transforms.Resize((size, size)),transforms.ToTensor()])
    transform = transforms.Compose([transforms.Resize((size, size)),
    transforms.RandomHorizontalFlip(),  # 水平方向にランダムに反転
      # ランダムに-15度から15度まで回転
    transforms.ToTensor()
])
    lang  = ImageLang( 'dataset/onomatope.csv',"dataset/image/train",transform)
    train_dataset=ImageLang('dataset/onomatope.csv',"dataset/image/train",transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True) #drop_lastをtruenにすると最後の中途半端に入っているミニバッチを排除してくれる
    valid_dataset=ImageLang('dataset/onomatopevalidation.csv',"dataset/image/valid",transform)
    valid_dataloader=DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    ono_valid_dataset=Lang('dataset/onomatopeunknown.csv')
    ono_valid_dataloader=DataLoader(ono_valid_dataset,batch_size=batch_size,shuffle=False,drop_last=True)
    
    
    for epoch in range(epochs):
        writer=SummaryWriter(log_dir="log/crossmodal")
        train_ono,train_img,train_imgono,train_total,encoder,decoder,image_model=Train(encoder,decoder,image_model,lang,train_dataloader)
        valid_ono,valid_img,valid_imgono,valid_total=Validation(encoder,decoder,image_model,lang,valid_dataloader,ono_valid_dataloader)
        print( "[epoch num %d ] [ train: %f]" % ( epoch+1, train_total ) )
        print( "[epoch num %d ] [ valid: %f]" % ( epoch+1, valid_total ) )
        writer.add_scalars('loss/ono/train',{'train':train_ono} ,epoch+1)
        writer.add_scalars('loss/ono/valid',{'valid':valid_ono} ,epoch+1)
        writer.add_scalars('loss/img',{'train':train_img,'valid':valid_img} ,epoch+1)
        writer.add_scalars('loss/imgono',{'train':train_imgono,'valid':valid_imgono} ,epoch+1)
        writer.add_scalars('loss/total',{'train':train_total,'valid':valid_total} ,epoch+1)
        # writer.add_scalars('loss/ono/',{'train':train_ono} ,epoch+1)
        # writer.add_scalars('loss/img/',{'train':train_img} ,epoch+1)
        # writer.add_scalars('loss/imgono/',{'train':train_imgono} ,epoch+1)
        # writer.add_scalars('loss/total/',{'train':train_total} ,epoch+1)
        writer.close()
        if (save_loss >= valid_total):
            torch.save(encoder.state_dict(), 'model/encoder')
            torch.save(decoder.state_dict(), 'model/decoder')
            torch.save(image_model.state_dict(),'model/imgmodel')
            save_loss=valid_total
            print("-------model 更新---------")
        torch.save(encoder.state_dict(), 'model/encodercheck')
        torch.save(decoder.state_dict(), 'model/decodercheck')
        torch.save(image_model.state_dict(),'model/imgmodelcheck')
if __name__ == '__main__':
    main()
    #追加で書き込み
    