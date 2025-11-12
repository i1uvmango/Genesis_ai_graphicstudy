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
* URDF 변환 시 주의사항
    * parenting 해체: parenting이 되어있으면 x,y,z 위치가 부모에 종속적이어서 이상한 위치에 생성될 수 있음
    * bone(blender) 불필요: Genesis에서 URDF안에서 joint를 정의할 수 있어서 bone은 또 다른 rigid body가 됨 &rarr; mesh 만 필요

## Blender Car to Genesis
[![terrain_drive](../res/car_genesis2.mp4)](https://github.com/user-attachments/assets/1ee96045-1d38-4925-888c-2870c0f73916)  
[![terrain_drive](../res/car_genesis1.mp4)](https://github.com/user-attachments/assets/ba70947a-7ed5-459c-bf7f-274ecad34938)    

* 아직 미완성 상태  
* body, wheel_fl, wheel_fr, wheel_rl, wheel_rr: 5개의 dae 파일
* 모두 parenting 해제 후 (0,0,0)로 좌표 설정 후 dae export
* URDF에서 5개의 dae 파일 읽은 후 Genesis에 entity 생성


#### 오류 정리
* ㅁ
* ㅁ
* ㅁ

# Genesis 에 URDF 로딩
문제: 
* 공중에 날라감
    * Plane 충돌 방지를 위해 공중 생성
    * constraints 안정되지 않았을때 suspension의 spring을 피거나 or damping이 너무 작을때 스프링에 의해 힘을 위로 받아서 날라감

## 문제 1: 바퀴 회전 축 및 구조 문제
### 문제 상황
- 바퀴가 전혀 회전하지 않음
- 뒷바퀴만 회전 조인트가 있고, 앞바퀴에는 조향 조인트만 있음
- 바퀴 회전 축이 잘못 설정됨 (Y축이 아닌 X축으로 설정되어 있었음)

### 원인
1. **회전 축 문제**: Genesis에서는 Y축(0, 1, 0) 회전이 전방 이동을 의미함 
    * x방향이 전방을 의미 &rarr; y축 방향의 회전이 들어가야 바퀴가 굴러감
    * x축을 돌려서 차가 움직이지 않았음
2. **앞바퀴 회전 조인트 부재**: 앞바퀴에 회전 조인트가 없어서 바퀴가 회전하지 않음
3. **조인트 구조 문제**: 조향과 회전이 하나의 조인트로 결합되어 있음

### 해결 방법
1. 모든 바퀴에 회전 조인트 추가 (Y축 회전)
2. 앞바퀴에 조향 링크 추가하여 조향과 회전 분리
3. 바퀴 회전 조인트를 `continuous` 타입으로 설정
#### 변경된 구조
```
기존: car_body → susp → wheel (조향만 가능)
수정: car_body → susp → steer_link → wheel (조향 + 회전 모두 가능)
```
  
## 문제 2: 차량이 하늘로 솟구치는 문제

### 문제 상황
- 차량이 스폰될 때 하늘로 솟구침
- 초기 위치가 너무 높게 설정되어 있거나, 바퀴가 지면과 겹침
### 원인
1. **초기 위치 계산 오류**: 차체 중심 위치와 바퀴 위치 관계를 정확히 계산하지 못함
2. **바퀴-지면 간섭**: 바퀴가 지면 아래로 들어가서 물리 엔진이 튕김
3. **서스펜션 조인트 범위**: 서스펜션 조인트의 limit이 넓어서 불안정함

### 영향
- 차량이 안정적으로 스폰됨
- 하늘로 살짝 솟구침 (서스펜션 확인을 위해)
- 바퀴가 지면에 안정적으로 닿음

## 문제 3: 서스펜션 튕김 및 뒤집힘 문제

### 문제 상황
- 서스펜션이 튕겨서 차량이 뒤로 뒤집어짐
- 서스펜션 조인트가 너무 자유롭게 움직임

### 원인
1. **서스펜션 링크 질량**: 5.0kg으로 너무 무거워서 불안정
2. **서스펜션 조인트 범위**: -0.15 ~ 0.15로 너무 넓음
3. **서스펜션 조인트 타입**: `prismatic` 타입이 damping이 없어서 튕김
    * `prismatic` 은 축 고정 &rarr; z축으로만 서스펜션이 작동
    * damping 은 충격을 흡수하여 진동을 줄이는 힘(저항) 

### 해결 방법
1. 서스펜션 조인트를 `fixed` 타입으로 변경 (test용)
2. `prismatic`을 z축 고정?

#### 실제 자동차 구조
```
실제 자동차:
┌─────────────┐
│   차체      │
└──────┬──────┘
       │
    [스프링] ← 탄성 (진동)
       │
    [댐퍼]   ← 댐핑 (진동 감쇠) ← 이게 없으면 튕김!
       │
    [바퀴]
```

### test 영상

[![terrain_drive](../res/car_genesis_drive.mp4)](https://github.com/user-attachments/assets/6556f3a5-b75f-49a8-a252-b733c86edb08)
------


  


# 일단 여기까지 
---
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


##############################
# 틀린 부분
* PPO(자율주행,강화학습 기반 제어) 로 할꺼면 위 데이터로 하면 안됨.

####################
  


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
