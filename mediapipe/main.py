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
     
def get_478_lms(cv_image,get_draw_img=False,save_path=None,name=None):
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

def get_pupil_lms(cv_image):
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
            try:
                for i in range(len(face_landmarks.landmark)):
                    x, y, _ = face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, face_landmarks.landmark[i].z
                    ori_x, ori_y = x*ori_w, y*ori_h
                    landmarks.append([ori_x,ori_y])

                landmarks = np.array(landmarks,dtype=np.int32)
            except:
                continue

        # right eye index : 474, 475, 476, 477
        # left eye index : 469, 470, 471, 472
        right_eye_mid = (landmarks[474] + landmarks[475] + landmarks[476] + landmarks[477])//4
        left_eye_mid = (landmarks[469] + landmarks[470] + landmarks[471] + landmarks[472])//4
        right_eye_lms = np.array([landmarks[474],landmarks[475],landmarks[476],landmarks[477]],dtype=np.int32)
        left_eye_lms = np.array([landmarks[469],landmarks[470],landmarks[471],landmarks[472]],dtype=np.int32)
        
        #   2
        # 1   3
        #   0
        def get_2_vectors(eye_mid, eye_lms, idx1,idx2):
            v1 = eye_mid-eye_lms[idx1]
            v2 = eye_mid-eye_lms[idx2]

            return eye_mid-v1-v2

        new_right_eye_lms = []
        new_right_eye_lms.append(get_2_vectors(right_eye_mid,right_eye_lms,1,2))
        new_right_eye_lms.append(get_2_vectors(right_eye_mid,right_eye_lms,2,3))
        new_right_eye_lms.append(get_2_vectors(right_eye_mid,right_eye_lms,0,3))
        new_right_eye_lms.append(get_2_vectors(right_eye_mid,right_eye_lms,1,0))
        new_right_eye_lms = np.array(new_right_eye_lms,dtype=np.int32)
        
        new_left_eye_lms = []
        new_left_eye_lms.append(get_2_vectors(left_eye_mid,left_eye_lms,1,2))
        new_left_eye_lms.append(get_2_vectors(left_eye_mid,left_eye_lms,2,3))
        new_left_eye_lms.append(get_2_vectors(left_eye_mid,left_eye_lms,0,3))
        new_left_eye_lms.append(get_2_vectors(left_eye_mid,left_eye_lms,1,0))
        new_left_eye_lms = np.array(new_left_eye_lms,dtype=np.int32)

        right_size_vector = np.sqrt(np.square(right_eye_lms[1]-right_eye_lms[3]).sum())
        if right_size_vector//2 == 1:
            right_size_vector += 1

        left_size_vector = np.sqrt(np.square(left_eye_lms[1]-left_eye_lms[3]).sum())
        if left_size_vector//2 == 1:
            left_size_vector += 1
            
        eye_size = np.array([right_size_vector, left_size_vector],dtype=np.int32)

        return eye_size, new_right_eye_lms, new_left_eye_lms

def get_5_lms(cv_image):
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
            for i in range(len(face_landmarks.landmark)):
                x, y, _ = face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, face_landmarks.landmark[i].z
                ori_x, ori_y = x*ori_w, y*ori_h
                landmarks.append([ori_x,ori_y])

            landmarks = np.array(landmarks,dtype=np.int32)

    right_eye_mid = (landmarks[3] + landmarks[7] + landmarks[163] + landmarks[144] + landmarks[145] + landmarks[153]\
         + landmarks[154] + landmarks[155] + landmarks[133] + landmarks[173] + landmarks[157] + landmarks[158] + landmarks[159]\
              + landmarks[160] + landmarks[161] + landmarks[246])//16
    left_eye_mid = (landmarks[263] + landmarks[249] + landmarks[390] + landmarks[373] + landmarks[374] + landmarks[380]\
         + landmarks[381] + landmarks[382] + landmarks[362] + landmarks[398] + landmarks[384] + landmarks[385] + landmarks[386]\
              + landmarks[387] + landmarks[388] + landmarks[468])//16
    nose = landmarks[19]
    right_mouth = landmarks[62]
    left_mouth = landmarks[292]

    return np.array([right_eye_mid, left_eye_mid, nose, right_mouth, left_mouth])