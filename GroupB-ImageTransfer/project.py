import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import copy
#device
device=torch.device("cuda")
#vgg19
model=models.vgg19(pretrained=True).features.to(device).eval()
#size
imsize = 512
#loader
def image_loader(path):
    image=Image.open(path)
    image=loader(image).unsqueeze(0)
    return image.to(device,torch.float)

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])

#image
style_img = image_loader("images/picasso.jpg")
content_img = image_loader("images/dancing.jpg")

generated_image=content_img.clone().requires_grad_(True)

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.req_features= ['0','5','10','19','28'] 
        self.model=models.vgg19(pretrained=True).features[:29]
    def forward(self,x):
        features=[]
        for layer_num,layer in enumerate(self.model):
            x=layer(x)
            if (str(layer_num) in self.req_features):
                features.append(x)
        return features

#def content loss
def calc_content_loss(gen_feat,orig_feat):
    content_l=torch.mean((gen_feat-orig_feat)**2)#*0.5
    return content_l

#def style loss
def calc_style_loss(gen,style):
    batch_size,channel,height,width=gen.shape

    G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
    A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())
    style_l=torch.mean((G-A)**2)#/(4*channel*(height*width)**2)
    return style_l

#calc loss
def calculate_loss(gen_features, orig_feautes, style_featues):
    style_loss=content_loss=0
    for gen,cont,style in zip(gen_features,orig_feautes,style_featues):
        content_loss+=calc_content_loss(gen,cont)
        style_loss+=calc_style_loss(gen,style)
    
    total_loss=alpha*content_loss + beta*style_loss 
    return total_loss

model=VGG().to(device).eval() 

#main
epoch=7000
lr=0.004
alpha=8
beta=70

optimizer=optim.Adam([generated_image],lr=lr)

for e in range (epoch):
    gen_features=model(generated_image)
    orig_feautes=model(content_img)
    style_featues=model(style_img)
    
    total_loss=calculate_loss(gen_features, orig_feautes, style_featues)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if(not (e%100)):
        print(total_loss)
        
        save_image(generated_image,"gen.jpg")