# Steering & Throttle 통일

* 앞바퀴 조향을 하나의 바퀴 기준으로 통일
* Throttle 도 뒷바퀴의 평균값으로 통일  
&rarr; 수치 안정성 확보

### 데이터 추출 코드 : [blender_data_extract](../src/on_off_data_blender_data.py)
* 정규화 `steer_norm`, `throttle_l` 
* 클리핑 `steer_norm = max(-1.0, min(1.0, steer_norm))` 을 통해 outlier 제거
* AI behavior cloning에 유리하도록 함

### 데이터 전처리 오류
![](../res/데이터전처리_오류.png)
* Steering(정상) : FL Wheel 기준으로 steering 통일
* Throttle(오류) : 뒷바퀴의 평균 값 구한 뒤 정규화 (50km 기준)
    * 하지만 csv 파일에서 +,-,+,-가 반복됨
    ![](../res/데이터전처리_오류2.png)


### 오류 원인 예상
1. 왼쪽 바퀴(spin_rl)와 오른쪽 바퀴(spin_rr)를 똑같은 공식으로 계산
    * 왼쪽 +0.5, 오른쪽 -0.5 , 합 0(이러면 모든 구간 0이어야함)
    
2. 쿼터니언의 Double Cover 현상(q,-q 가 동일한 회전을 나타냄)
    * 하지만 이것도 다음 로직으로 부호 통일 시킴
    * 이게 문제라면 절댓값이 연속적인 throttle 값이 나와야함

#### throttle 안정화
* 다음과 같이 값을 안정화 시킴
* 원인은 미분에 의한 노이즈였음

1. deadzone 적용: V_LONG_EPS = 0.10 (m/s)
* 종속도가 0 근처일 때는, 위치 미분 기반 속도(v_B)가 프레임 노이즈 때문에 부호가 흔들릴 수 있습니다.
* 그래서 |v_long| < 0.10이면 전/후진 판정을 하지 않고 throttle을 0으로 고정합니다.
* *즉, 정지·저속에서 “throttle이 +였다가 −였다가” 튀는 걸 막는 장치입니다.

2. 크기는 abs(spin_R)
* 휠 회전이 빠를수록 가속 입력이 크다고 “추정”합니다.
* 부호를 제거하고 순수 크기만 씁니다.

3. 부호는 sign(v_long)로 결정
* 전진이면 +, 후진이면 −
* 즉, “전진 throttle”과 “후진 throttle”을 구분하려는 목적입니다
  ![](../res/1217데이터.png)
  * 정상적인 throttle 수치

  
### 좌표계 정렬부터 다시 시행
* Blender ↔ Genesis 좌표계 정렬 해결
* 문제 상황 : 차량이 학습 후 시뮬레이션에서:
```
우회전 명령을 전혀 예측하지 못함
속도 감소 후 정지: 점진적으로 감속하여 완전히 멈춤
```

* Blender 좌표계  
Forward: -Y 방향  
Right: +X 방향  
Up: +Z 방향  

* Genesis 좌표계  
Forward: +X 방향  
Left: +Y 방향  
Up: +Z 방향  

재정렬 후 다시 시도 해봄
* y축 방향으로 바퀴축이 굴러가는것 까지 확인


![](../res/1217.gif)
* kickstart 이후 감속하며 차가 멈춤 
    * 차의 절대적인 속도가 낮아서 deadzone이 영향을 줬을거라 예상
* 우회전 나타나지 않음
    * 아직 원인을 찾지 못함



#### 코드
블렌더 데이터 추출: [data_blender](../src/on_off_data_blender_data.py)  

BC training 코드: [train_bc.py](../src/train_bc.py)  

BC 추론 및 실행 코드: [test_bc.py](../src/test_bc.py)  
