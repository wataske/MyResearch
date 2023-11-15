import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import sys
import matplotlib.pyplot as plt
from texture import TextureNet,ImageDataset


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

        # plt.title(label[i])
        plt.axis('off')
    if num == 0:
        plt.savefig("/workspace/mycode/Documents/output/batch/original{}.jpg".format(epoch))
    elif num ==1:
        plt.savefig('/workspace/mycode/Documents/output/batch/image{}.png'.format(epoch))
    else:
        plt.savefig('/workspace/mycode/Documents/output/batch/onomatope{}.png'.format(epoch))

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = ImageDataset(dir="./dataset/image/train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextureNet(feature_layers=["0", "5", "10"]).to(device)

    criterion = nn.MSELoss()

    

    # model_save_path = "model/texture_net64.pth"
    model_save_path = "model/texture_net64.pth"
    model.load_state_dict(torch.load(model_save_path))

    for batch_idx, (imgs, label) in enumerate(dataloader):
        imgs = imgs.to(device)
        _,hidden = model(imgs)
        outputs=model.decoder(hidden)
        generate_and_save_images(imgs,batch_idx,0,label)
        generate_and_save_images(outputs,batch_idx,1,label)
    



