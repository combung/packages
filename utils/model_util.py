import gdown
import os

# ex) job name : https://drive.google.com/uc?id={공유 link}

weight_dic = {
    'face_seg': 'https://drive.google.com/uc?id=10G_lqmYCP7piDKsJYGKU9_6mXHGDRP6j',
}

save_path = {
    'face_seg':'./face_seg/ptnn/79999_iter.pth',
}

def download_weight(job):
    gdown.download(weight_dic[job], output=save_path[job], quiet=False)
