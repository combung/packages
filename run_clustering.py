
import sys
sys.path.append("../")
import os
import cv2
import glob
from PIL import Image
from arcface import get_id
from clustering import cluster
import os
import cv2
import sys
sys.path.append("../")
import glob
from face_alignment import do_align

if __name__ == "__main__":
    root = "../../../../media/deep3090/새 볼륨/4Kdownload" # 동영상이 모여있는 폴더들이 있는 위치
    # dirs = os.listdir(root) # 동영상이 들어있는 폴더들의 이름을 모아놓은 리스트 ex - ["딩고뮤직", "딩고프리스타일"]
    dirs = ["딩고뮤직2"]
    frame_freq = 20 # 몇 프레임마다 쓸건지? (인접한 프레임은 얼굴이 똑같을거라서 띄엄띄엄 추출)

    # directory 하나씩 처리
    for dir in dirs:
        video_paths = sorted(glob.glob(f"{root}/{dir}/*.mp4"))
        total_id_count = 0

        # 비디오 하나씩 처리
        for video_index, video_path in enumerate(video_paths):
            print(f" > > > > > > processing {video_path}...")

            # frame_freq 에 따라서 frame 추출
            videoObj = cv2.VideoCapture(video_path)            
            frame_list = [] 
            frame_count = 0
            while True:
                try:
                    ret, frame = videoObj.read()

                    if not ret:
                        break
                    
                    if frame is None: 
                        break

                    if frame_count % frame_freq:
                        frame_list.append(frame)

                    frame_count += 1
                    
                except:
                    continue

            print(f" > > > > > > {len(frame_list)} frames extracted with frame_freq of {frame_freq}...")

            # 얼굴 추출 및 정렬
            aligned_faces = []
            frame_count = 0 
            for frame in frame_list:
                tmp = do_align(frame, output_size=256)
                if tmp is None: 
                    continue
                aligned_faces += tmp
                
            print(f" > > > > > > {len(aligned_faces)} faces are recognized in the video...")

            if len(aligned_faces) < 20: # 얼굴 수 많이 없으면 다음 비디오로 넘어감
                print(f" > > > > > > not enough, jump to next video...")
                continue
                
            print(f" > > > > > > Embedding aligned faces...")

            # 얼굴 특징 벡터 추출
            embedding_vectors = []
            for aligned_face in aligned_faces:
                embedding_vector = get_id(aligned_face).squeeze().detach().cpu().numpy()
                embedding_vectors.append(embedding_vector)

            # 클러스터링
            indexes_list = cluster(embedding_vectors)
            
            print(f" > > > > > > detected {len(indexes_list)} identites, now saving faces...")

            # id 별로 얼굴 이미지 저장
            os.makedirs(f"{root}/{dir}_faces", exist_ok=True)
            for id_num, indexes in enumerate(indexes_list):
                if id_num == 0:
                    continue
                total_id_count += 1
                dir_name = f"{root}/{dir}_faces/{dir}_id_{str(total_id_count).zfill(8)}"
                os.makedirs(dir_name, exist_ok=True) 

                count = 0
                for index in indexes:
                    pathname = os.path.join(dir_name, f"{str(count).zfill(8)}.jpg")
                    cv2.imwrite(pathname, aligned_faces[index][:, :, ::-1])
                    count += 1

            print(f" > > > > > > finished!")
