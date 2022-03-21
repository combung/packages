import os
import torch
import torchvision.transforms as transforms

from packages.gfpgan.net import GFPGANv1Clean

from packages.utils.util import convert_img_type
from packages.utils.model_util import download_weight

# initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize the GFP-GAN
gfpganNET = GFPGANv1Clean(
    out_size=512,
    num_style_feat=512,
    channel_multiplier=2,
    decoder_load_path=None,
    fix_decoder=False,
    num_mlp=8,
    input_is_latent=True,
    different_w=True,
    narrow=1,
    sft_half=True)

file_PATH="./packages/gfpgan/ptnn/GFPGANv1.3.pth"
if not os.path.isfile(file_PATH):
    download_weight('gfpgan')

loadnet = torch.load(file_PATH)
if 'params_ema' in loadnet:
    keyname = 'params_ema'
else:
    keyname = 'params'
gfpganNET.load_state_dict(loadnet[keyname], strict=True)
gfpganNET.eval()
gfpganNET = gfpganNET.to(device)
    

def do_gfpgan(pil_image):
    pil_image = convert_img_type(pil_image,'pil')

    with torch.no_grad():
        output = gfpganNET(transforms.ToTensor()(pil_image.resize((512,512))).unsqueeze(0).to("cuda"), return_rgb=False)[0]
        return (output.permute(0, 2, 3, 1) * 255).clamp(0, 255).squeeze().detach().cpu().numpy()
            
