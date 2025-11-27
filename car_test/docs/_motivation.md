grapicar



# 최상위 목적 (End Goal)
AI 에이전트가 Genesis에서 학습한 제어 능력을
현실(Real world)에 Zero-shot 또는 최소 튜닝으로 그대로 수행하도록 만드는 것.

이를 위해 필수 조건은 다음 두 가지:

1. 가상 환경(Genesis)의 물리 행동이 현실 세계와 최대한 일치
2. 학습 데이터(Blender/Real sensor data 포함)가 현실 행동을 정확히 반영



## Blender의 역할
#### Blender 목적: Data-driven Seamless Simulation Physics
* 현실과 최대한 비슷한 학습용 물리 데이터 생성기

#### Blender에서 하는 일

1. 차량의 실제 움직임과 가장 유사한 물리 결과값 생성
* 바퀴 토크
* 차체에 걸리는 힘
* 속도, 각속도, 조향량
* 경로에 따른 motion pattern

2. 가상 센서를 만들고, 매 프레임마다 핵심 물리량을 추출

* g_lin_vx, g_ang_vz 등 6DoF state
* torque, steering angle
* wheel rotation

3. Genesis 제어와 동일한 인터페이스 유지

* Blender에서도 “조향 + 가속 입력”으로 움직임 생성
* Genesis에서도 동일 양식으로 행동 생성 
    → 두 환경을 “API 단위”에서 맞추는 것이 핵심
```
Blender는
현실을 근사한 data-driven 시뮬레이션 환경 → 데이터 수집기 역할
Genesis가 학습하는 데 필요한 데이터를 공급하는 “ground-truth physics generator”
```


## Genesis 의 역할
Genesis 목적: Goal-driven Autonomous Control Physics & Reinforcement Learning
  
* 즉, Genesis는 단순 시뮬레이터가 아닌
목표 기반 제어(Goal-driven control policy)를 강화학습/모방학습을 통해 학습시키는 환경

Genesis에서 하는 일
1. 조향(angle) + 가속(throttle) 을 매 프레임 policy가 직접 생성
2. 그 action으로 물리 simulation step 진행
3. 경로 목표(goal trajectory)를 보고 policy가 스스로 조정
4. 학습 후 checkpoints를 이용해 다양한 경로에서 Generalization 검증
5. 최종적으로 Real-world에 전이(Transfer) 가능한 형태로 정책(policy) 확보


## Blender &rarr; Genesis 데이터 파이프라인

Blender가 제공해야 하는 것 = Genesis가 필요한 것

Blender로부터 Genesis가 받는 데이터:
* 선속도 벡터 (g_lin_vx, g_lin_vy, g_lin_vz)
* 각속도 벡터 (g_ang_vx, g_ang_vy, g_ang_vz)
* 바퀴 토크 (virtual torque sensor)
* steering angle, throttle

※ “Genesis는 조향/가속 입력만 받으므로”, Blender에서도 같은 포맷을 유지해야 학습과 전이가 가능함.

### 왜 인터페이스 맞추기가 중요한가?

Real → Blender → Genesis → Real 전이 구조에서  
action/state 포맷이 일치하지 않으면 policy가 완전히 무너짐  
그래서 Blender는 Genesis의 제어 인터페이스에 맞춘 “matching simulator” 역할을 해야 한다.




### Overfitting vs Real Generalization
#### Overfitting 상태

* 8자만 따라가고
* 네모에서는 완전히 이상하게 움직임
* Blender motion trace만 흉내 내는 모사 수준

#### 우리가 원하는 것

* 8자로 학습되었지만
* 사각형, S-curve, arbitrary waypoint에서도
* steering/throttle 조합이 자연스럽게 전개됨 → 이것이 Data-driven simulation physics → Goal-driven RL control의 성공


### Generative Physics가 왜 중요한가?

Generative Physics =
“현실 데이터를 기반으로 가상 물리를 자동으로 보정하는 AI 기반 물리 추정 기술”


* 현실의 움직임(Real motion trace)
* Blender의 가상 센서 데이터
* Genesis의 simulation state
비교해가며 **가상세계 물리가 현실과 가까워지도록 자동 조정**

### Generative Physics의 이점
1. 현실 데이터를 주면 가상으로 물리 자동 보정
2. 튜닝 비용 급감
3. 가상의 테스트의 다양성과 안정성 확보
4. QA 테스트를 정확하게 시뮬레이션
    * 고속 회피
    * 급조향 미끄럼
    * 화물 무게 변환
    * 롤오버 조건 &rarr; 위험한 상황도 가상에서 수천 번 생성 가능


### 최종 결과

* Generative Physics가 있으면
* 현실-가상을 매우 가깝게 만들고
* 그 위에서 학습된 정책(policy)은 곧바로 현실에서 동작한다.

    * 이게 테슬라가 말하는 Zero-Shot Transfer의 핵심 원리.








