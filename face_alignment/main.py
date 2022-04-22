import numpy as np
import os
import glob
from PIL import Image
import cv2
from packages.utils.util import convert_img_type
from facenet_pytorch import MTCNN 
from face_alignment.utils import align_image

mtcnn = MTCNN()

def do_align(image, output_size):
    pil_image = convert_img_type(image,'pil')
    lms = mtcnn.detect(pil_image, landmarks=True)[2][0]
    aligned_image = align_image(pil_image, lms, output_size)
    return aligned_image

