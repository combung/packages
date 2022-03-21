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

## 2. read pretrained model