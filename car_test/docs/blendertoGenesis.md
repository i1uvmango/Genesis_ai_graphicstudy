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

## 직접 URDF 만들어서 엔진 학습
#### URDF 변환 하며 생긴 문제 정리
* 시도한 것: blender에서 mesh+bone을 parenting으로 연결하여 한번에 genesis에 import 하려했음  

##### 왜 다음과 같이 하면 안되는가?  

* parenting 해체 필요: Blender에서 Export 시 parenting이 되어있으면 x,y,z 위치가 부모에 종속적이어서 이상한 위치, 회전된 상태로 생성 될 수 있음
    * URDF는 절대 좌표계 기준으로만 동작, Blender의 local transform 정보를 인식하지 못함
* bone 불필요: Genesis에서 URDF안에서 joint를 정의할 수 있어서 bone은 또 다른 rigid body가 됨 &rarr; mesh 만 필요
    * Genesis는 joint로 물리 연결을 정의하기 때문에 bone은 하나의 객체로 인식함, parenting 도 설정되지 않음
* mesh (0,0,0) origin 설정: 원점으로 기본 좌표계를 설정해주지 않으면 joint origin(urdf setting)과 mesh의 offset이 이중으로 적용되어서 멀리 떨어져 생성될 수 있음


## Blender Car to Genesis
https://github.com/user-attachments/assets/1ee96045-1d38-4925-888c-2870c0f73916 

https://github.com/user-attachments/assets/ba70947a-7ed5-459c-bf7f-274ecad34938  

* 아직 미완성 상태  
* body, wheel_fl, wheel_fr, wheel_rl, wheel_rr: 5개의 dae 파일
    * dae 를 사용한 이유: Blender, Genesis 에서 geometry + transform + scale까지 완전 보존
* 모두 parenting 해제 후 (0,0,0)로 좌표 설정 후 `.dae` 로 export
* URDF에서 5개의 `.dae` 파일 읽은 후 Genesis에 로딩

#### Export 시 다음과 같은 설정을 따름
![export_setting](../res/export_setting.png)
* Selection Only : 마우스 선택한 mesh 만 export
* Include children(x) : children 포함하지 않고 export
* Global Orientation : 위치, 회전, 스케일을 유지하고 export
    * apply transform은 blender 4.0 이후로 global orientation에 포함

* forward: x, up: z 설정 필요(Genesis 좌표계)
    * 수정 필요

#### 현재 오류 정리
* **car_body** : z축 기준으로 180도 회전된 상태로 생성
* **wheel** : dof 가 너무 큼, 뒷바퀴는 dof 0으로 고정 시키기기
    

# Training : PPO
### 선택 방법
* Blender Simulation에서 직접 parameter를 추출하여 Genesis의 솔버를 학습시키는 방법
* 주행 시뮬레이션 실행하며 매 frame 마다 parameter을 추출

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


##############################
# 틀린 부분
* 데이터를 뽑아낸다 라고해서 동역학의 데이터 라고 생각했는데 train을 시키려면 train에 필요한 데이터가 필요하다는 걸 깨달음
  
* PPO(자율주행,강화학습 기반 제어) 로 할꺼면 위 데이터로 하면 안됨.
    * 주행 데이터, 앞에 어떤 물체가 있는지, 속도, 각속도 등의 데이터가 필요함 &rarr; 코드 수정 필요
####################



* 데이터 추출할때 주행 중심으로 데이터를 추출해야하나?
* 아니면 동역학 중심으로 데이터를 추출해야하나?


---
### Code: Extracting Parameters from Blender
* Blender to JSON : ([../src/blendertoJson.py](https://github.com/i1uvmango/Genesis_ai_graphicstudy/blob/main/car_test/src/blendertoJson.py))
* JSON to URDF: (https://github.com/i1uvmango/Genesis_ai_graphicstudy/blob/main/car_test/src/jsontoURDF.py)
