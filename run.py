import os
import cv2
import sys
sys.path.append("../")
import glob
from PIL import Image
from facenet_pytorch import MTCNN 
from face_alignment import do_align

mtcnn = MTCNN()

if __name__ == "__main__":
    image_paths = glob.glob("samples/*.*")
    for image_path in image_paths:
        image_name = os.path.split(image_path)[1][:-4]
        image = Image.open(image_path)
        aligned_iamge = do_align(image, output_size=256)[0]
        cv2.imwrite(f"outputs/{image_name}.jpg", aligned_iamge[:, :, ::-1])
