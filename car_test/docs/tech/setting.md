## 1. 차량 설정 (Blender)

### RBC addon rigged car
  
![](../../res/0115/car.png)  

* car rig
![](../../res/0115/car3.png)
  
* mesh + armature &rarr; car 형상 + 차량 제어 constraints 로 구성

![](../../res/0115/bone.png)
* bone 에 걸리는 물리력을 python script를 통해 데이터를 추출함


![](../../res/0115/car4.png)
* 자동차 세팅 조절 (bone &rarr; constraints / physics &rarr; 세팅 변경)

## 2. 경로 설정 (Blender)
* bezier curve로 경로를 그린 뒤, path follow 클릭 후 bezier curve 선택
<video controls src="../../res/0115/car.mp4" title="Title"></video>

## 3. 구동계 설정
![](../../res/0115/car2.png)
* 바퀴:50kg * 4개
* 차체: 1000kg
* 총 : 1200kg

### wheel constraints
![](../../res/0115/wheel.png)
* 후륜 구동
* 앞바퀴 조향만

### suspension constraints
![](../../res/0115/sus.png)
* 서스펜션 : 1 (interface 기준)
* Spring stiffness: 50 (interface 기준)
* Damping: 2(interface 기준)
### weight / inertia
![](../../res/0115/weight.png)
* 중앙 기준
* weight ratio : 0.15%

### engine constraint
![](../../res/0115/engine.png)
* basic(default setting)

### driving constraint
![](../../res/0115/driving.png)
* traction control : 1 (interface 기준)
    *  바퀴 슬립을 감지해 구동 토크를 제한함으로써 접지력을 유지하는 제어 시스템
    * 마찰을 넘지 않게 토크를 관리하는 시스템
    ![](../../res/0115/traction.png)



## 4. 주행 경로 및 마찰 계수 설정 (Variable Friction)

### 주행경로에 따로 마찰계수를 주는 방법
![](../../res/0115/road.png)
### 1) 도로를 구간별 Mesh로 분할
- 예: `Road_Dry`, `Road_Wet`, `Road_Ice`
- 방법:
  - Edit Mode → Edge 선택
  - `P` → Selection 으로 분리
  - 또는 처음부터 타일형 도로 설계

### 2) 각 도로 구간에 Rigid Body 설정
- Type: **Passive** (충돌 x)
- Shape: **Mesh** (입자 하나하나 계산하여 계산량은 증가하지만 정확한 계산)
- Animated: x

### 3) 마찰 계수 설정


1. mesh 클릭 후 우측 아래 physics properties 로 이동

![](../../res/0115/constraints.png)

2. surface response → friction 로 마찰계수를 조정 가능



#### 예시
| 구간 | Friction | 용도 |
|------|----------|------|
| Dry  | 0.8 ~ 1.0 | 마른 아스팔트 |
| Wet  | 0.3 ~ 0.5 | 빗길 |
| Ice  | 0.05 ~ 0.15 | 빙판 |



#### 하나의 terrain 에서 차량이 움직이는 경우
![](../../res/0115/road2.png)
* 산악 지형: 연속적 &rarr; genesis 에서는 object 로 불러와서 terrain field 로 계산하는 것이 유리하다고 함



### 5) 차량 바퀴 설정
- Rigid Body: **Active**(충돌 가능)
- Shape: **Cylinder**
- Friction: **1.0**

## 5. 주행 데이터 획득 방법
![](../../res/0115/script.png)
* scripting 으로 python 을 통한 제어 가능

![](../../res/0115/script2.png)
* import bpy 를 해야 python이 작동함
* 우측 상단 run 버튼을 눌러 실행
* 좌측 하단 콘솔창


## 6. 구현 코드 검증
(내용 작성 예정)

## 7. Genesis Real2Sim Physics 학습 방법
(내용 작성 예정)

