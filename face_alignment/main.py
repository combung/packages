import numpy as np
import os
import glob
from PIL import Image
import cv2
from packages.utils.util import convert_img_type
from face_alignment.utils import align_image

from facenet_pytorch import MTCNN 
mtcnn = MTCNN()

from insightface.app import FaceAnalysis
app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'landmark_2d_106'])
app.prepare(ctx_id=0, det_size=(256, 256))

def do_align(image, output_size, detector="insight", size_thres=512):
    pil_image = convert_img_type(image,'pil')

    if detector == "mtcnn":
        boxes, _, lms = mtcnn.detect(pil_image, landmarks=True)[2]

        if lms is None: 
            return None

        aligned_images = []

        for i, lm in enumerate(lms):

            # size condition of bounding box
            if (boxes[i][2]-boxes[i][0]) + (boxes[i][3]-boxes[i][1]) > size_thres:

                aligned_image = align_image(pil_image, lm, output_size)[0]
                aligned_images.append(aligned_image)
    
    if detector == "insight":
        lms = []
        faces = app.get(image)
        for face in faces:
            
            # size condition of bounding box
            if (face.bbox[2]-face.bbox[0]) + (face.bbox[3]-face.bbox[1]) > size_thres:  
                
                lm_106 = face.landmark_2d_106
                lms.append(np.array([lm_106[38], lm_106[88], lm_106[86], lm_106[52], lm_106[61]]))
                
        if len(lms) == 0: 
            return None

        aligned_images = []

        for lm in lms:
            aligned_image = align_image(pil_image, lm, output_size)[0]
            aligned_images.append(aligned_image)
    return aligned_images

