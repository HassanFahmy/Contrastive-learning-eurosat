from collections import defaultdict
from statistics import mean
from torch import nn
from torch import optim
import argparse
import clip
import itertools
import numpy as np
import os
import random
import torch
import torch.multiprocessing
import torch.nn.functional as F
import wandb
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import rasterio

clip_version = 'ViT-B/16' 
device = "cuda:4"
image_dir = "/data2/hassanf/Eurosat/Images"
sat_dir = "/data2/hassanf/Eurosat/spectral_data/"
batch_size = 100
num_workers=2
num_epochs = 2


def l2_normalize(X):
  return F.normalize(X, p=2, dim=1)


print('Loading clip model...')
clip_model, image_preprocess = clip.load(clip_version, jit=False, device=device)
print('Done.')


print('Setting up Sattelite encoder...')
resnet = models.resnet50(pretrained=True).to(device)
modules=list(resnet.children())[:-1]
modules.append(nn.Flatten().to(device))
modules.append(nn.Linear(2048, 512, bias=True).to(device))
#print (modules)
resnet2=nn.Sequential(*modules)
optimizer = optim.Adam(resnet2.parameters())
print('Done.')


print('Loading Data...')

res_preprocess = transforms.Compose([
  transforms.Resize(224),                                                           
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def data_loader (path):
  image = Image.open(path)
  image = image_preprocess(image)
  sat_path = path.replace("Images","spectral_data").replace("jpg","tiff")
  sat_data = rasterio.open(sat_path).read()
  s = torch.tensor(sat_data.tolist()).float()
  s = s [[4,5,6]]
  s = res_preprocess(s).float()
  return(image,s )
  #return image


full_data = datasets.ImageFolder(image_dir,loader = data_loader)

train_data, val_data = torch.utils.data.random_split(full_data, [26000,1000])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last = True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=250, shuffle=True, num_workers=num_workers, drop_last = True)

#classes = sat_data.classes
classes = ['a centered satellite photo of annual crop land', 'a centered satellite photo of a forest', 'a centered satellite photo of brushland or shrubland', 'a centered satellite photo of a highway or road', 'a centered satellite photo of industrial buildings or commercial buildings', 'a centered satellite photo of pasture land', 'a centered satellite photo of permanent crop land', 'a centered satellite photo of residential buildings or homes or apartments', 'a centered satellite photo of a river', 'a centered satellite photo of a sea or a lake']

print('Done.')


text = clip.tokenize(classes).to(device)
text_embeddings = l2_normalize(clip_model.encode_text(text).float()).to(device)


def loss_fn(logits):
  labels = torch.arange(logits.shape[0]).to(device)
  loss_image = F.cross_entropy(logits, labels)
  loss_sat = F.cross_entropy(logits.T, labels)
  loss = (loss_image + loss_sat) / 2
  return loss

def run_val():
  image_accs = []
  text_accs = []
  im_text_accs = []

  for images, labels in val_loader:
    with torch.no_grad():
      rgb_images = images[0].to(device)
      sat_images = images[1].to(device)
      labels = labels.to(device)
      image_embeddings = l2_normalize(clip_model.encode_image(rgb_images).float())
      sat_embeddings = resnet2(sat_images).float()
      
      logits = torch.mm(image_embeddings, sat_embeddings.T)
      con_labels = torch.arange(logits.shape[0]).to(device)
      preds = torch.argmax(logits, 1)
      image_accs.append (torch.mean((preds == con_labels).type(torch.float32)).item())

      logits = torch.mm(sat_embeddings, text_embeddings.T)
      preds = torch.argmax(logits, 1)
      text_accs.append(torch.mean((preds == labels).float()).item())

      logits = torch.mm(image_embeddings, text_embeddings.T)
      preds = torch.argmax(logits, 1)
      im_text_accs.append(torch.mean((preds == labels).float()).item())
      wandb.log({"imgs": [wandb.Image( rgb_images[0], caption="rgb"),wandb.Image( sat_images[0], caption="infrared")]})
  wandb.log({  'val/sat text acc': mean(text_accs), 'val/sat image acc': mean(image_accs), 'val/text image acc': mean(im_text_accs), })


wandb.init(project='clip-experimentation', entity='hassanf')
for epoch in range(num_epochs):
  print('Epoch: {} / {}'.format(epoch, num_epochs))
  i = 0
  for images, labels in train_loader:
    i +=1
    if (i%50 == 0):
      run_val()
    rgb_images = images[0].to(device)
    sat_images = images[1].to(device)
    labels = labels.to(device)
    with torch.no_grad():
      image_embeddings = l2_normalize(clip_model.encode_image(rgb_images).float())

    optimizer.zero_grad()
    sat_embeddings = resnet2(sat_images).float()
    logits = torch.mm(image_embeddings, sat_embeddings.T)
    loss = loss_fn(logits)
    loss.backward()
    optimizer.step()


    con_labels = torch.arange(logits.shape[0]).to(device)
    preds = torch.argmax(logits, 1)
    image_acc = torch.mean((preds == con_labels).type(torch.float32)).item()
    logits = torch.mm(sat_embeddings, text_embeddings.T)
    preds = torch.argmax(logits, 1)
    text_acc = torch.mean((preds == labels).float()).item()
    wandb.log({ 'train/loss':loss.item() , 'train/sat text acc': text_acc, 'train/sat image acc': image_acc, })
    #torch.save(audio_encoder.state_dict(), /data2/hassanf/model.pt)
