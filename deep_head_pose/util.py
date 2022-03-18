from PIL import Image
import cv2
import numpy as np

def convert_img_type(img, target_type):
    # check input image type
    try:
        if img.filename:
            img_type = 'pil'
    except:
        img_type = 'cv2'

    if img_type == target_type:
        return img
    else:
        # cv2 -> pil
        if target_type == 'pil':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            return pil_img
        # PIL -> cv2
        elif target_type == 'cv2':
            img = np.array(img)
            opencv_image=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return opencv_image


def convert_3ch(np_img):
    _, _, c = np.array(np_img).shape
    if c == 4:
        return np.array(np_img)[:,:,2]
    elif c == 1:
        return np.expand_dims(np_img,-1).repeat(3,axis=-1)


