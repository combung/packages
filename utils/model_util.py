import gdown
import os

# ex) job name : https://drive.google.com/uc?id={공유 link}

weight_dic = {'gfpgan': 'https://drive.google.com/file/d/14OnzO4QWaAytKXVqcfWo_o2MzoR4ygnr/',
}

save_path = {
    'gfpgan':'./packages/gfpgan/ptnn/model.pth',

}

def download_weight(job):
    gdown.download(weight_dic[job], output=save_path[job], quiet=False)
