import os
import sys
sys.path.append(os.path.join(os.getcwd(),'head_pose'))

import torch
from torch.autograd import Variable
from torchvision import transforms

import torchvision
import torch.nn.functional as F

import packages.head_pose.hopenet as hopenet
from packages.utils.util import convert_img_type
from packages.utils.model_util import download_weight

"""
input
    - image : pillow Image
output
    - degrees numpy
"""

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

file_PATH = './packages/head_pose/ptnn/hopenet.pkl'
if not os.path.isfile(file_PATH):
    download_weight('head_pose')

saved_state_dict = torch.load(file_PATH)
model.load_state_dict(saved_state_dict)
model.cuda(device)
model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

transformations = transforms.Compose([transforms.Resize(224),
transforms.CenterCrop(224), transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
print('>>> Ready to deep head pose.')

def do_head_pose(pil_img):
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(device)

    # Transform
    img = convert_img_type(pil_img,'pil')
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img).cuda(device)

    yaw, pitch, roll = model(img)

    yaw_predicted = F.softmax(yaw,dim=-1)
    pitch_predicted = F.softmax(pitch,dim=-)
    roll_predicted = F.softmax(roll,dim=-1)
    # Get continuous predictions in degrees.
    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    yaw_score = yaw_predicted.detach().cpu().numpy()
    pitch_score = pitch_predicted.detach().cpu().numpy()
    roll_score = roll_predicted.detach().cpu().numpy()

    return yaw_score, pitch_score, roll_score
    
