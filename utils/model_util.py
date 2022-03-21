import gdown
import os

# ex) job name : https://drive.google.com/uc?id={공유 link}

weight_dic = {
    'face_seg': 'https://drive.google.com/uc?id=166yC9tK7VXRZJGhBk8I66ewl1Fa3L6_3',
    'head_pose': 'https://drive.google.com/uc?id=1AH9cJCPAbov1OxtXPaihqhgveZIGjF7U',
    'swinIR': 'https://drive.google.com/uc?id=1hQSl6iwmlhD6cR5qoMHc7TMqSxRZwHS6',
    'gfpgan': 'https://drive.google.com/uc?id=1fyXxJzQzh00WJbZEvUj_utFXIZt29_4Y',

}

save_path = {
    'face_seg': './packages/face_seg/ptnn/face_seg.pth',
    'head_pose': './packages/head_pose/ptnn/hopenet_robust_alpha1.pkl',
    'swinIR': './packages/swinIR/ptnn/swinIR_large.pth',
    'gfpgan': './packages/gfpgan/ptnn/GFPGANv1.3.pth',

}

def download_weight(job):
    gdown.download(weight_dic[job], output=save_path[job], quiet=False)
