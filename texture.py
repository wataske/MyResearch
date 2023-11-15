import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from torch.utils.data import DataLoader
from PIL import Image
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import sys


# データセット定義
class ImageDataset:
    def __init__( self,dir,transform ): #呼び出されたとき、最初に行うこと
        self.transform = transform
        self.data = [] #画像が入ってる
        target_dir = os.path.join(dir, "*/*")
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
    
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self,x):
        return x.view(self.shape)
    
# # VGG16から特徴を抽出
# class VGGIntermediate(nn.Module):
#     def __init__(self, requested=[]):
#         super(VGGIntermediate, self).__init__()
#         self.intermediates = {}
#         self.vgg = models.vgg16(pretrained=True).features
#         for name, module in self.vgg.named_children():
#             if name in requested:
#                 def curry(i):
#                     def hook(module, input, output):
#                         self.intermediates[i] = output
#                     return hook
#                 module.register_forward_hook(curry(name))

#     def forward(self, x):
#         self.vgg(x)
#         return self.intermediates
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.layers = {'0': 'conv1_1',
                       '5': 'conv2_1',
                       '10': 'conv3_1',
        }
                    #    '19': 'conv4_1',
                    #    '21': 'conv4_2',
                    #    '28': 'conv5_1'}
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features
# グラム行列計算
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
class PrintShape(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x

# ネットワーク定義
class TextureNet(nn.Module):
    def __init__(self, feature_layers, fc_out_dim=128, img_size=64, img_channels=3):
        super(TextureNet, self).__init__()
        self.img_size = img_size
        self.extractor =VGGFeatures()#VGG16で特徴を抽出する部分
        self.fc = nn.Linear(86016, fc_out_dim) #抽出した複数の特徴を1つのベクトルに圧縮してfc_out_dim=128次元に圧縮

        self.decoder = nn.Sequential(
        nn.Linear(128, 32*8*8), nn.Tanh(),
        Reshape(-1, 32, 8, 8),
        nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),nn.Tanh(),
        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Tanh(),
        nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
    )

    def forward(self, x):
        features = self.extractor(x)
        gram_features = [gram_matrix(f).view(f.size(0), -1) for f in features.values()]
        concatenated_features = torch.cat(gram_features, dim=1)
        compressed = self.fc(concatenated_features)
        compressed=compressed/(torch.norm(compressed))
        output = self.decoder(compressed)

        return output,compressed
def style_loss_and_diffs(recon, orig, model):
    recon_features = model.extractor(recon)
    orig_features = model.extractor(orig)
    
    loss = 0
    diffs = {}
    criterion=nn.MSELoss()
    for layer, (rf, of) in enumerate(zip(recon_features.values(), orig_features.values())):

        diff=criterion(gram_matrix(rf),gram_matrix(of))
        diffs[f'Layer_{layer}_diff'] = diff
        loss += diff
    return loss, diffs
# 学習部分
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    train_dataset = ImageDataset(dir="./dataset/image/train", transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataset = ImageDataset(dir="./dataset/image/valid", transform=transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextureNet(feature_layers=["0", "5", "10"]).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    

    model_save_path = "texture_net64.pth"
    epochs = 1000
    i=0
    w=0.0001 #重み
    save_loss=1
    for epoch in range(epochs):
            writer = SummaryWriter(log_dir="/workspace/mycode/Documents/log/texture64")
            for batch_idx, (imgs, _) in enumerate(train_dataloader):
                model.train()
                imgs = imgs.to(device)
                optimizer.zero_grad()
                outputs,_= model(imgs)
          

                orig_features = model.extractor(imgs)
                recon_features = model.extractor(outputs)
                s_loss, gram_diffs = style_loss_and_diffs(outputs, imgs, model)
                recon_loss=criterion(outputs,imgs)
                for name, diff in gram_diffs.items():
                    writer.add_scalars('GramDiffs/', {f"{name}" + "train":diff.item()}, epoch)
                writer.add_scalars('recon_loss/',{"train":recon_loss.item()},epoch)
                recon_loss=recon_loss*w
                
                train_loss = recon_loss + s_loss
                writer.add_scalars('total_loss/',{"train":train_loss},epoch)
                train_loss.backward()
                optimizer.step()
            for batch_idx, (imgs, _) in enumerate(valid_dataloader):
                model.eval()
                imgs=imgs.to(device)
                outputs,_=model(imgs)

                orig_features = model.extractor(imgs)
                recon_features = model.extractor(outputs)
                s_loss, gram_diffs = style_loss_and_diffs(outputs, imgs, model)
                recon_loss=criterion(outputs,imgs)
                for name, diff in gram_diffs.items():
                    writer.add_scalars('GramDiffs/', {f"{name}" + "valid":diff.item()}, epoch)
                    i +=1
                writer.add_scalars('recon_loss/',{"valid":recon_loss.item()},epoch)
                recon_loss=recon_loss*w
                
                valid_loss = recon_loss + s_loss
                writer.add_scalars('total_loss/',{"valid":valid_loss},epoch)               

            print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_loss.item()}")
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {valid_loss.item()}")

            # モデルの保存
            if save_loss >= valid_loss: # 10エポックごとにモデルを保存
                torch.save(model.state_dict(), model_save_path)
                save_loss=valid_loss
                print(f"Model saved to {model_save_path}")

            writer.close()
