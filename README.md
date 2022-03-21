# Usage
## import
`
from packages import {name}
`

> from packages import face_seg
## use

`
{package_name}.{function_name}(parameters)
`

> result = face_seg.do_parsing(image)

# Add packages
package를 추가하고 싶을 때 읽어보면 된다.

## 1. _ _ init _ _.py
추가할 package에 _ _ init _ _ .py 를 만든다.

`
from .main import {function_name}
`

이렇게 해야 외부에서 {package_name}.{function_name}로 바로 사용가능하다.  
위 같이 하지 않으면 {package_name}.main.{function_name}로 사용 가능함. -> 귀찮...

> from .main import get_478_lms, get_5_lms, get_pupil_lms

## 2. main.py
모델을 가져오고, 모델을 이용한 작업을 함수화 한다.

### 2.1 downloads pretrained model
utils/model_util.py 를 참조한다.  

weight_dic 에는 innerverz 공유 드라이브에 올려놓은 pretrain model의 링크를 id={link id} 에 적어놓고, save_path는 저장할 경로를 적어준다.  

다운로드할 때는 os.path.isfile을 이용하여 모델이 있는지 체크하는 방식을 사용한다
import 할때마다 덮어씌우는 다운로드 작업을 피하자.  

## 3. network.py
main.py 에는 작업 관련 함수를 넣어주고, network 관련 코드는 여기다 넣어준다.  
main.py 에서 불러오도록 하자.

# ETC
## utils/util.py
### convert_img_type
종종 input image 형식(PIL or cv2)을 맞춰야 할 필요가 있다.
input 형식에 구애받지 않고 함수 내에서 이미지 형식을 알아서 바꿔주고 싶을 때 이용하면 된다.