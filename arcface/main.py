import torch
import os
from torchvision import transforms
import torch.nn.functional as F

from packages.arcface.nets import Backbone
from packages.utils.util import convert_img_type
from packages.utils.model_util import download_weight

device = "cuda" if torch.cuda.is_available() else "cpu"
file_PATH = './packages/arcface/ptnn/arcface.pth'

arcface = Backbone(50, 0.6, 'ir_se').to(device)
if not os.path.isfile(file_PATH):
    download_weight('arcface')

arcface.load_state_dict(torch.load(file_PATH, map_location=device))
arcface.eval()

def get_id(face):
    image = convert_img_type(face,'pil')
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    aligned_source_face_ = transform(image).unsqueeze(0).to(device).float()
    with torch.no_grad():
        source_id = arcface(F.interpolate(aligned_source_face_[:, :, 16:240, 16:240], (112, 112), mode='bilinear', align_corners=True))

    return source_id