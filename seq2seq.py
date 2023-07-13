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

SOS_token = 0
EOS_token = 1

device = "cuda" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lang:
    def __init__( self, filename ): #呼び出されたとき、最初に行うこと
        max_length    = 20
        self.filename = filename
        self.ono=[]    
        self.word2index = {}
        self.word2count = {}
        self.sentences = []
        self.index2word = { 0: "SOS", 1: "EOS" }
        self.n_words = 2  # Count SOS and EOS
        df = pd.read_csv(filename)
        num=df.shape[0] #csvファイルの行数を取得（データセットの数）
        
        for i in range(num): #264種類のラベルを作る
            word=df.iloc[i,2] #単語(3列目)
            phoneme = df.iloc[i, 1]  # 音素(2列目)
            i=i+1
            self.ono.append(word)
            self.sentences.append(phoneme)
        self.allow_list = [ True ] * len( self.sentences )
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
        return len(self.ono)
    def __getitem__(self, index): #dataloaderを作ったときにできるやつ
        """サンプルを返す。
        """
        
        ono=self.ono[index] #取ってきたラベル番号のオノマトペを取ってくる（オノマトペのラベル番号とリストの番号は一致している）
        phoneme = self.sentences[index] #同上

        return ono, phoneme


mm = preprocessing.MinMaxScaler()
# Start core part
class Encoder( nn.Module ):
    def __init__( self, input_size, embedding_size, hidden_size ):
        super().__init__()
        self.hidden_size = hidden_size
        # 単語をベクトル化する。1単語はembedding_sie次元のベクトルとなる
        self.embedding   = nn.Embedding( input_size, embedding_size )
        # GRUに依る実装. 
        self.gru         = nn.GRU( embedding_size, hidden_size )
        self.sigmoid=nn.Sigmoid()

    def initHidden( self ):
        return torch.zeros( 1, 1, self.hidden_size ).to( device )

    def forward( self, _input, hidden ):
        # 単語のベクトル化
        embedded        = self.embedding( _input ).view( 1, 1, -1 )
        out, new_hidden = self.gru( embedded, hidden )
        # new_hidden=self.sigmoid(new_hidden)
        return out, new_hidden

class Decoder( nn.Module ):
    def __init__( self, hidden_size, embedding_size, output_size ):
        super().__init__()
        self.hidden_size = hidden_size
        # 単語をベクトル化する。1単語はembedding_sie次元のベクトルとなる
        self.embedding   = nn.Embedding( output_size, embedding_size )
        # GRUによる実装（RNN素子の一種）
        self.gru         = nn.GRU( embedding_size, hidden_size )
        # 全結合して１層のネットワークにする
        self.linear         = nn.Linear( hidden_size, output_size )
        # softmaxのLogバージョン。dim=1で行方向を確率変換する(dim=0で列方向となる)
        self.softmax     = nn.LogSoftmax( dim = 1 )
        
    def forward( self, _input, hidden ):
        # 単語のベクトル化。GRUの入力に合わせ三次元テンソルにして渡す。
        embedded           = self.embedding( _input ).view( 1, 1, -1 )
        # relu活性化関数に突っ込む( 3次元のテンソル）
        relu_embedded      = F.relu( embedded )
        # GRU関数( 入力は３次元のテンソル )
        gru_output, hidden = self.gru( relu_embedded, hidden )
        # softmax関数の適用。outputは３次元のテンソルなので２次元のテンソルを渡す
        result             = self.softmax( self.linear( gru_output[ 0 ] ) )
        return result, hidden
    
    def initHidden( self ):
        return torch.zeros( 1, 1, self.hidden_size ).to( device )



def tensorFromSentence( lang, sentence ): #sentenceをインデックス番号に変換したtensor配列にする
    indexes = [ lang.word2index[ word ] for word in sentence.split(' ') ] #sentenceの音素をインデックス番号に変換してリストにする
    indexes.append( EOS_token )
    

    return torch.tensor( indexes, dtype=torch.long ).to( device ).view(-1, 1)


def Train(encoder,decoder,lang,dataloader):
    train_total_loss=0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    learning_rate = 1e-5 #10e-6
    batch_size =8
    criterion      = nn.CrossEntropyLoss() #これをバッチサイズ分繰り返してそれをエポック回数分まわす？


    encoder_optimizer = optim.Adam( encoder.parameters(), lr=learning_rate )
    decoder_optimizer = optim.Adam( decoder.parameters(), lr=learning_rate )
    for batch_num,(ono,phoneme) in tqdm.tqdm(enumerate(dataloader),total=len(dataloader)):
        batch_ono_loss=0 #1バッチ分のLossをここに入れていく
        batch_total_loss=0
        for data_num in range(dataloader.batch_size): 
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

                if decoder_input.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了


            
            loss=loss_ono
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            #1バッチ分(８個のデータ)のLossになるように加算していく        
            batch_total_loss +=loss
        # print(batch_loss/dataloader.batch_size) #１バッチにおける1個のデータあたりのLossを計算する
        train_total_loss +=batch_total_loss/dataloader.batch_size

    #(total_loss)をバッチの個数分で割ることで１データローダにおけるLossがわかる   
    train_total_loss=train_total_loss/len(dataloader) 

    return train_total_loss,encoder,decoder

def Validation(encoder,decoder,lang,dataloader):
    with torch.no_grad():
        valid_total_loss=0

        criterion=nn.CrossEntropyLoss()
        mse=nn.MSELoss()
        for batch_num,(ono,phoneme) in tqdm.tqdm(enumerate(dataloader),total=len(dataloader)):
            batch_total_loss=0
            for data_num in range(dataloader.batch_size): 

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

                loss=loss_ono

                #1バッチ分(８個のデータ)のLossになるように加算していく        
                batch_total_loss +=loss
        
            valid_total_loss+=batch_total_loss/dataloader.batch_size

        #(total_loss)をバッチの個数分で割ることで１データローダにおけるLossがわかる      
        valid_total_loss=valid_total_loss/len(dataloader) 

        return valid_total_loss
def main():
    embedding_size = 128
    hidden_size   = 128
    num=40 #入出力として使える音素の数=データセット内の.n_wordsに等しい
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder           = Encoder( num, embedding_size, hidden_size ).to( device )
    decoder           = Decoder( hidden_size, embedding_size, num ).to( device )
    epochs=10000
    save_loss=50000
    batch_size=8
    lang  = Lang( 'dataset/onomatope.csv') #word2indexを作るための辞書
    dataloader = DataLoader(lang, batch_size=batch_size, shuffle=True,drop_last=True) #drop_lastをtruenにすると最後の中途半端に入っているミニバッチを排除してくれる
    train_dataset  = Lang( 'dataset/onomatope.csv')
    valid_dataset  = Lang( 'dataset/onomatopeunknown.csv')
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size, shuffle=True,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,drop_last=True) #drop_lastをtruenにすると最後の中途半端に入っているミニバッチを排除してくれる

    enfile = "model/encoder0704" #学習済みのエンコーダモデル
    defile = "model/decoder0704" #学習済みのデコーダモデル
    encoder.load_state_dict( torch.load( enfile ) ) #読み込み
    decoder.load_state_dict( torch.load( defile ) )
    for epoch in range(epochs):
        writer=SummaryWriter(log_dir="log/seq2seq07042")
        train_total,encoder,decoder=Train(encoder,decoder,lang,train_dataloader)
        valid_total=Validation(encoder,decoder,lang,valid_dataloader)
        print( "[epoch num %d ] [ train: %f]" % ( epoch+1, train_total ) )
        print( "[epoch num %d ] [ valid: %f]" % ( epoch+1, valid_total ) )
        writer.add_scalars('loss/total',{'train':train_total,'valid':valid_total} ,epoch+1)
        # writer.add_scalars('loss/ono/',{'train':train_ono} ,epoch+1)
        # writer.add_scalars('loss/img/',{'train':train_img} ,epoch+1)
        # writer.add_scalars('loss/imgono/',{'train':train_imgono} ,epoch+1)
        # writer.add_scalars('loss/total/',{'train':train_total} ,epoch+1)
        writer.close()
        if (save_loss >= train_total):
            torch.save(encoder.state_dict(), 'model/encoder07042')
            torch.save(decoder.state_dict(), 'model/decoder07042')
            save_loss=train_total
            print("-------model 更新---------")
        torch.save(encoder.state_dict(), 'model/encodercheckpoint2')
        torch.save(decoder.state_dict(), 'model/decodercheckpoint2')
if __name__ == '__main__':
    main()
    #追加で書き込み
    