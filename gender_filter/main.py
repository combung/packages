import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from .nets import regressor
import warnings; warnings.filterwarnings('ignore')

cwd = os.path.dirname(os.path.realpath(__file__))

class GenderFilter(nn.Module):
    def __init__(self, ckpt_path = 'ckpt/gender.pt'):
        super(GenderFilter, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gender_filter = regressor().to(self.device).eval()
        ckpt = torch.load(os.path.join(cwd, ckpt_path), map_location=self.device)
        self.gender_filter.load_state_dict(ckpt)
        for param in self.gender_filter.parameters():
            param.requires_grad = False
        del ckpt
        
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])    
        
    def forward(self, PIL_image):
        tensor = self.transforms(PIL_image).unsqueeze(0).to(self.device)
        score = self.gender_filter(tensor)
        return score