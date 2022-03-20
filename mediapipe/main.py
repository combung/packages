import cv2
import math
import numpy as np
import mediapipe as mp
from packages.utils.util import convert_img_type

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def resize(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

    return img
     
def get_468_lms(cv_image,get_draw_img=False,save_path=None,name=None):
    landmarks = []
    cv_image  = convert_img_type(cv_image,'cv2')
    ori_h, ori_w = cv_image.shape[:2]
    image_ = resize(cv_image)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=2,
        min_detection_confidence=0.5) as face_mesh:

        lms = face_mesh.process(cv2.cvtColor(image_, cv2.COLOR_BGR2RGB))
        for face_landmarks in lms.multi_face_landmarks:
            # for i in range(468):
            for i in range(len(face_landmarks.landmark)):
                x, y, _ = face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, face_landmarks.landmark[i].z
                ori_x, ori_y = x*ori_w, y*ori_h
                cv2.waitKey(0)
                landmarks.append([ori_x,ori_y])

            landmarks = np.array(landmarks,dtype=np.int32)


        if get_draw_img:
            annotated_image = cv_image.copy()
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())

            canvas = np.zeros_like(cv_image)
            mp_drawing.draw_landmarks(
                image=canvas,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(
                image=canvas,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=canvas,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
            cv2.imwrite(f'{save_path}/{name}.png',annotated_image)
            cv2.imwrite(f'{save_path}/{name}_canvas.png',canvas)

        return landmarks
