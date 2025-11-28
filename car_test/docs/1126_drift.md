## Blender Drift to Genesis
* RBC Pro 사용

![](../res/path_point.png)

### Youtube 영상 보며 Guide Path 동작 구현
youtube reference: `https://www.youtube.com/watch?v=rrI6wzFquhU` 
* Bezier Curve 를 자동차가 따라가며 auto drive




https://github.com/i1uvmango/Genesis_ai_graphicstudy/issues/9#issue-3669513138


![](../res/path_follow.gif)



## Blender Script 사용시 문제점 정의
### "핸들러 잔존(Handler Persistence)" 및 "전역 변수 오염(Global State Contamination)" 문제

1. 문제 정의 (Problem Statement)
* 현상: 스크립트를 재실행하거나 다른 .blend 파일을 로드했음에도 불구하고, 이전 시뮬레이션의 움직임 데이터가 출력되거나, 데이터가 비정상적으로 튀는 현상(Ghosting) 발생.

* 사용자 관찰: "이전 파일의 움직임이 그대로 나온다" 혹은 "핸들러 캐시 문제"로 인지됨.

#### 문제 
1. 핸들러의 영속성 (Handler Persistence):

bpy.app.handlers.frame_change_post에 등록된 함수는 사용자가 명시적으로 제거(remove)하거나 블렌더 프로그램을 완전히 끄기 전까지 메모리에 계속 상주합니다.

2. 애플리케이션 메모리 공유:

블렌더 내의 Python 인터프리터는 씬(Scene)이 바뀌어도 메모리를 공유합니다. 다른 파일을 열어도 변수명이 같다면 이전 메모리 주소를 참조할 위험이 있습니다.
스크립트를 다시 실행할 때 이전 핸들러를 제대로 삭제하지 않으면, 똑같은 함수가 2개, 3개 중복되어 등록됩니다.

이 경우, 한 번의 프레임 변경에 스크립트가 여러 번 실행되면서 CSV에 데이터를 중복 기록하거나 덮어쓰게 됩니다.

## Guide Path Driving 영상
![](../res/drive_square.gif)
* 사각형 auto drive
  
![](../res/drive8.gif)  
* 8자 형태 auto drive

### 공유 데이터 추출 코드
[shared_data_extracter](../src/shared_data_extracter.py)
* 연구생 공동 사용하는 data_extracting code
### On/Off 할 수 있는 데이터 추출 코드
[button_extract_data](../src/extract_data_blender.py)
* 스크립트 실행시 항상 데이터가 csv 파일에 저장되는걸 방지하기 위해 ON/OFF 로직 추가