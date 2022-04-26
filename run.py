import os
import cv2
import sys
sys.path.append("../")
import glob
from face_alignment import do_align

if __name__ == "__main__":
    root = "../../../../media/deep3090/새 볼륨/4Kdownload" # 동영상이 모여있는 폴더들이 있는 위치
    dirs = os.listdir(root) # 동영상이 들어있는 폴더들의 이름을 모아놓은 리스트 ex - ["딩고뮤직", "딩고프리스타일"]
    save_freq = 10 # 몇 프레임마다 쓸건지 (인접한 프레임은 얼굴이 똑같을거라서 띄엄띄엄 추출)
    log_freq = 100 # 몇 프레임마다 로그 출력?

    for dir in dirs:
        video_paths = glob.glob(f"{root}/{dir}/*.mp4")

        for video_path in video_paths:
            print(f">>>>>>>>>> processing {video_path}")
            video_root = video_path[:-4]

            os.makedirs(video_root, exist_ok=True) # 프레임 저장하는 폴더
            os.makedirs(f"{video_root}_face", exist_ok=True) # 얼굴 저장하는 폴더
            
            videoObj = cv2.VideoCapture(video_path)
            ret = 1
            frameCount = 0
            faceCount = 0
            
            ret, frame = videoObj.read()
            while ret:
                frame_num = int(round(videoObj.get(1))) #current frame number
                ret, frame = videoObj.read()

                if frame_num % save_freq == 0:

                    # 프레임 저장
                    save_path = f"{video_root}/{str(frameCount).zfill(8)}.png"
                    cv2.imwrite(save_path, frame)
                    frameCount += 1

                    # 얼굴 추출 및 정렬
                    aligned_images = do_align(frame, output_size=256)

                    if aligned_images is None:
                        continue
                    
                    # 정렬된 얼굴들 저장
                    for aligned_image in aligned_images:       
                        cv2.imwrite(f"{video_root}_face/{str(faceCount).zfill(8)}.jpg", aligned_image[:, :, ::-1]) 
                        faceCount += 1

                if frame_num % log_freq == 0:
                    print(f">>>>>> {frameCount}th frame has been processed, and {faceCount} faces are saved.")

