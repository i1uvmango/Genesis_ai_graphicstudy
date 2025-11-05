# Adjusting GenesisAI Engine to Blender

## 연구실 terrain blender 적용
[![terrain_drive](../res/terraindrive_mesh.mp4)](https://github.com/user-attachments/assets/1bcbc266-c78b-4341-a654-860d5592b4b7)
* terrain: passive collision, mesh 표면
    * convex hull: 직육면체로 표현 &rarr; 단순한 계산
    * mesh: terrain 표현 하나하나를 mesh에 따라 계산 &rarr; 연산량 증가, 정교한 표현
      

## 물리 법칙 모사
Blender는 물리 엔진을 외부에 제공하지 않음
* Bullet Physics Library를 기반함(C++)
* 이를 모사하기 위해 Pybullet을 Genesis에서 사용  


### Bullet Engine 의 문제점
* Pro: Genesis에서도 동일한 움직임을 구현할 수 있다
* Con: CPU 기반 연산이기에 매우 느리고, Genesis에서 돌리는 이유가 사라진다
    * GPU 기반의 빠른 연산 & 렌더링이 필요하기에 PyBullet은 탈락 


## Blender Car to Genesis
[![terrain_drive](../res/car_genesis.mp4)](https://github.com/user-attachments/assets/1ee96045-1d38-4925-888c-2870c0f73916)  
[![terrain_drive](../res/car_genesis2.mp4)](https://github.com/user-attachments/assets/ba70947a-7ed5-459c-bf7f-274ecad34938)  
* 아직 미완성 상태
#### 오류 정리
* ㅁ
* ㅁ
* ㅁ
### 선택 방법
* Blender Simulation에서 직접 parameter를 추출하여 Genesis의 솔버를 학습시키는 방법
* 시뮬레이션 실행하며 매 frame 마다 parameter을 추출 
---
## Extracting Data

### rigid body data
```
name, mass, friction, restitution, linear damping, angular damping, collision shape, origin[x,y,z], origin[rpy] 
```
* 이름
* 질량
* 마찰 계수
* 반발계수
* 선형감쇠
* 각 감쇠
* 충돌 형상
* 위치
* 회전(rpy)
* 부모링크
* 자식링크
  
### joint data
```
name, type, object1, object2, origin["xyz"], origin["rpy"], axis_world, motor.velocity, motor.max_impulse
```
* 이름
* 타입
* 부모링크
* 자식링크
* 위치
* 회전(rpy)
* 축 방향
* 모터정보  

위 parameter 추출 후 URDF 로 변환
  
Physics Engine Calibration (물리파라미터 보정법)
    * Blender Simulation 프레임별로 `bpy` 로 물리 parameter 추출 : JSON 으로 저장
        * 코드: (~~)
    * JSON으로 URDF 작성 &rarr; Genesis 적용


* 데이터 추출할때 주행 중심으로 데이터를 추출해야하나?
* 아니면 동역학 중심으로 데이터를 추출해야하나?


---


---
### Code: Extracting Parameters from Blender
* Blender to JSON : ([../src/blendertoJson.py](https://github.com/i1uvmango/Genesis_ai_graphicstudy/blob/main/car_test/src/blendertoJson.py))
* JSON to URDF: (https://github.com/i1uvmango/Genesis_ai_graphicstudy/blob/main/car_test/src/jsontoURDF.py)
