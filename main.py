import head_pose
import face_seg
from PIL import Image
import numpy as np
import swinIR
import cv2
import gfpgan

img = Image.open('./tmp.png')

result = head_pose.cal_head_pose(img)
print("head pose:", result)

result = face_seg.face_parsing(img)
cv2.imwrite("seg_result.png", result[:, :]*255)

result = swinIR.do_SR(img)
cv2.imwrite("swinIR_result.png", result[:, :, ::-1])

result = gfpgan.do_gfpgan(img)
cv2.imwrite("gfpgan_result.png", result[:, :, ::-1])