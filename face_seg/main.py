#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'face_seg'))
from PIL import Image

import torch
import torchvision.transforms as transforms

from model import BiSeNet
from utils.util import convert_img_type

n_classes = 19
net = BiSeNet(n_classes=n_classes)
net.cuda()
net.load_state_dict(torch.load('./face_seg/ptnn/79999_iter.pth'))
net.eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def face_parsing(pil_image):
    image = convert_img_type(pil_image,'pil')
    with torch.no_grad():
        #try:
        image = image.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image).unsqueeze(0).cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

    return parsing
    
