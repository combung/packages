import deep_head_pose
import face_seg
from PIL import Image
import numpy as np

img = Image.open('./tmp.png')
# result = deep_head_pose.cal_head_pose(img)
result = face_seg.face_parsing(img)
# result_pil = 
print(result)