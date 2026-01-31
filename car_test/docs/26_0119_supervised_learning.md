# Supervised Learning with UKMAC(Unicycle Kinematic Model for Acceleration and Curvature)


* **목표:** 속도/조향(Steer)의 목표는 Blender.
* **지도 학습:** 속도 / 조향.
* **현재 이해:** PPO(우리 모델) + Blender 데이터 정답값 -> 지도 학습으로 MLP 생성.
* **솔버 입력:** Blender와 Genesis의 솔버 입력값이 다름 -> 따라서 솔버의 입력은 Blender와 경로가 동일해지는 입력이 필요
* 바퀴 마찰 계수 등 고려 필요.

> "두 솔버(Solver)가 공통으로 해석 가능한 상태(State) 공간이 필요하다."





## Unicycle Kinematic Model for Acceleration and Curvature(UKMAC)

![](../res/0119/input.png)
> 두 솔버가 공유하는 state
* solver가 공유하는 state 이므로, (a,k) 기반으로 두 솔버의 입력 state를 동기화해야함
---

## Env Sync (Stage 1)

> 목적: 시뮬레이션의 기본 물리법칙만으로는 학습되지 않는 `마찰력`, `공기저항`, `중력 방향`, `타이어 슬립` 등 환경적 요소를 Residual Learning 으로 시뮬레이션의 정확도를 높인다.

* 쿼터니언 &rarr; 중력 방향 
* v_long 속도 &rarr; 공기저항 (속도의 제곱에 비례)
* v_long 속도 + 이전 step 속도 &rarr; 질량과 회전관성
* v_lat(횡방향 속도) + yaw_rate(회전 각속도) &rarr; 타이어 슬립 
* roll, pitch &rarr; 차량의 자세 변화에 따른 동역학에 주는 영향 학습



### 입력 데이터(Raw Data)
* `v_long`: 차량의 전진 속도 (Longitudinal Velocity)
* `v_lat`: 차량의 횡방향(미끄러짐) 속도 (Lateral Velocity)
* `yaw_rate`: 차량의 회전 각속도 (Yaw Rate)
* `g_qw`, `g_qx`, `g_qy`, `g_qz`: 차량의 자세를 나타내는 쿼터니언(Quaternion) 정보

### mlp 구조
#### input
* car state(5) + env state(6) : 11 dim
#### hidden layer
* `(128, 128, 64)`
#### output 
* delta_v_long: 종방향 속도 보정값 &rarr; 공기저항 고려 속도 차이
* delta_v_lat: 횡방향 속도 보정값 &rarr; 타이어 슬립/마찰력 고려 속도 차이
* delta_yaw_rate: 요(Yaw) 각속도 보정값 &rarr; 회전 관성 고려 각속도 차이


#### 체크포인트로 저장되어 stage3 ground truth 생성에 사용됨
* blender와 genesis의 환경적 편차를 MLP가 학습하고,  이에 따른 ground truth 데이터를 생성함

![stage1_loss](../res/0119/stage1_loss.png)

### 요약
이 코드는 "8.csv"라는 주행 데이터를 입력받아, 차량이 도로 위에서 어떻게 움직이는지 물리 법칙을 스스로 학습한 뒤, 그 정보를 담은 "env_best_8.pth" 파일을 생성합니다.

---

## Residual Dynamics (Stage 2)


Stage 1이 "환경(경사로, 중력)"을 배웠다면, Stage 2는 **"순수한 차량의 고유한 움직임(핸들을 꺾었을 때 얼마나 확 도는지, 타이어가 얼마나 미끄러지는지)"**을 학습하여 시뮬레이션을 완성하는 단계입니다.

> Bicycle Kinematics 를 사용하여 stage1 에서 배운 환경을 고려한 차량의 움직임을 학습

```
Input: "지금 속도는 10m/s고, 핸들은 0.5만큼 꺾었어. 차는 2도 정도 기울어져 있고..."

Baseline: "이론상으론 횡방향 속도가 1.2가 나와야 해."

Env Sync: "거기에 환경 저항을 고려하면 1.15가 되겠네."

Target (정답): 그런데 실제 데이터(drive_8.csv)를 보니 1.0이 찍혀 있네?

Stage 2 MLP의 임무: "아하! -0.15만큼의 차이가 생기는구나. 이 -0.15가 바로 우리 차(URDF)의 고유한 동역학적 특성이네! 이걸 외워야지."

```
* URDF 자체의 슬립 , 속도 등의 동역학적 특성을 Residual Learning 으로 학습


### 모델: ResidualDynamicsMLP
* **입력(14개 피처) $\rightarrow$ 출력(횡방향 속도 잔차, 회전각속도 잔차)**
* 이 모델은 **"핸들을 꺾었을 때 차가 실제로 얼마나 반응하는지"**에 대한 정밀한 물리 엔진 역할을 합니다.
* 가장 바깥 껍질인 **"마찰(Baseline)"**을 벗겨내고, 두 번째 껍질인 **"환경(Stage 1)"**을 벗겨낸 뒤, 가장 안쪽에 남은 **"차량의 진짜 성능(Dynamics)"**만을 학습하여 저장합니다.

### 구조 (Architecture)
```python
nn.Linear(14, 128), nn.ReLU(),
nn.Linear(128, 128), nn.ReLU(),
nn.Linear(128, 64), nn.ReLU(),
nn.Linear(64, 2)
```

### 입력/출력

### 1. 입력 변수 (Input Features - 11차원)
모델이 "지금 상황이 이렇구나!"라고 판단하는 근거입니다. 크게 세 그룹으로 나뉩니다.

#### 1. 현재 운동 상태 (Current State)
차량이 현재 어느 정도의 속도로 어떻게 움직이고 있는지를 나타냅니다.
* $v$ (Longitudinal Velocity): 차량의 전진 속도. 타이어 마찰력과 공기 저항을 결정하는 가장 핵심 변수입니다.
* $v_{lat}$ (Lateral Velocity): 옆으로 미끄러지는 속도. 이 값이 클수록 차가 통제력을 잃기 직전임을 의미합니다.
* $yaw\_rate$ (Yaw Rate): 차체가 수직축을 중심으로 회전하는 속도.

#### 2. 운전자 명령 및 의도 (Control Commands)
UKMAC의 $a$와 $k$가 여기서 '힌트' 역할을 합니다.
* $a$ (Acceleration): 가속도 명령. 엔진이 휠에 전달하는 힘의 크기를 대변합니다.
* $k$ (Curvature): 곡률 명령. 스티어링 휠을 얼마나 꺾었는지를 나타냅니다.

#### 3. 물리적 환경 및 자세 (Physical Context)
차량의 자세가 접지력에 주는 영향을 설명합니다.
* $\beta$ (Side Slip Angle): 차체가 바라보는 방향과 실제 이동 방향 사이의 각도. ($\arctan2(v_{lat}, v)$)
* $pitch$ / $roll$: 차체가 앞뒤/좌우로 기울어진 정도. 하중 이동(Load Transfer)을 파악하여 타이어 접지력을 추론합니다.
* $v^2$ (Velocity Squared): 속도의 제곱. 고속 주행 시 급격히 커지는 공기 저항과 동적 안정성을 위한 비선형 힌트입니다.
* $prev\_v_{lat}$ / $prev\_yaw$: 직전 타임스텝의 상태. 차량의 관성(Inertia)이 현재 거동에 주는 영향을 학습하기 위한 시계열 데이터입니다.

### 2. 출력 변수 (Output Variables - 2차원)
모델이 "수학 공식(Bicycle)이랑 실제 제네시스(URDF)는 이만큼 다르네!"라고 답하는 결과값입니다.

#### 1. $\Delta v_{lat}$ (Lateral Velocity Residual)
* **의미:** "이론상으론 옆으로 안 밀려야 하는데(Kinematic), 실제 제네시스에서는 타이어가 미끄러져서 이만큼 옆으로 더 밀리네?"를 나타내는 보정값입니다.
* **역할:** 타이어의 코너링 포스(Cornering Force) 한계를 모델링합니다.

#### 2. $\Delta \omega$ (Yaw Rate Residual)
* **의미:** "이론상으론 10도 돌아야 하는데, 실제 차는 무게 중심과 관성 때문에 9도만 돌거나(Understeer), 뒤가 털려서 11도 도네(Oversteer)?"를 나타내는 보정값입니다.
* **역할:** 차량의 회전 관성(Moment of Inertia) 특성을 모델링합니다.


### 동일한 변수들을 사용하는데 Stage1 과 어떻게 독립인지?
* Stage 2를 학습할 때 Stage 1 모델은 **'수정 불가능한 상식'**이 됩니다. 즉, Stage 2 모델이 아무리 노력해도 Stage 1이 담당하는 영역(중력, 공기 저항)을 뺏어오거나 대신 학습할 수 없도록 설계

* $Actual - (Kinematic + Stage1)$ 로 학습
* Stage 1(env sync)가 설명하지 못한 나머지 에러에 집중


### UKMAC 방법 사용
단순히 속도 자체를 배우는 것이 아니라, 물리적 변화량인 $a$와 $k$를 학습 타겟으로 삼아 모델이 차량의 동역학적 특성을 훨씬 정밀하게 파악하도록 합니다.

```python
# target_a: 가속도, target_k: 곡률 (yaw_rate / v)
target_a = (v_next - v_curr) / dt
target_k = yr_curr / max(abs(v_curr), 0.5) 

# 모델은 기본 물리(Baseline)가 설명 못 하는 '나머지'를 예측
# Output: [delta_v_lat, delta_yaw_rate] -> 실제 a, k를 맞추기 위한 보정치
residual_target = (target_a - base_a, target_k - base_k)
```
![stage2](../res/0119/stage2.png)
---



## Ground Truth (Stage 3)

```
Inverse Dynamics MLP를 학습시키기 위해 Genesis 세계에 맞는 데이터들을 생성해주는 과정
```


![](../res/0119/gt.png)

* 이전 단계(Stage 1 & 2)에서 학습한 모델들을 사용하여, 원본 데이터의 불완전한 사람 입력을 "시뮬레이터에서 실제로 작동하는 완벽한 정답(Ground Truth)"으로 교정해주는 과정입니다.

> 목표: "Blender 와 똑같은 가속도(a)와 곡률(k)를 내고싶다"

#### a,k 가 blender/genesis 솔버를 잇는 유일한 state 값이므로, 이를 동일하게 하는 input state (T,S)를 찾는 과정
* 수치 최적화(optimization)을 통해서 a,k가 동일해지는 (Throttle, Steer)값을 찾는다
* 이렇게 찾아진 데이터를 ($T_{gt}$, $S_{gt}$)로 정의 : _gt는 ground truth를 의미


### Objective Function
* 단순 computing이 아니라 아래 objective function을 최소화하는 방향으로 최적화(optimization)를 수행


  
$$(T_{gt}, S_{gt}) = \arg\min_{T, S} \left[ \left( a_{genesis}(T, S) - a_{blender}^* \right)^2 + 5 \cdot \left( k_{genesis}(T, S) - k_{blender}^* \right)^2 \right]$$


(단 a*,k* 은 stage1,2 에서 학습된 보정치)


$$ a_{blender}^* = a_{blender} + \text{Residual}_{Env}(a) + \text{Residual}_{Dyn}(a) $$

$$ k_{blender}^* = k_{blender} + \text{Residual}_{Env}(k) + \text{Residual}_{Dyn}(k) $$

* 경로 추종이 더 중요하기 때문에 1:5(weight)로 설정




## Filtering (Stage 3.5)

> Garbage in Garbage out

Ground Truth 를 통해 생성된 데이터를 `물리적인 한계` , `계산상 오류` 등을 다음과 같은 기준으로 필터링

* loss : 
    * Loss < 1.0~2.0: Genesis 솔버가 "이 정도 핸들($S$)과 스로틀($T$)이면 Blender랑 거의 똑같이 움직일 수 있어!"라고 자신 있게 찾아낸 모범 답안입니다.
    * "아무리 밟고 꺾어봐도 Blender처럼 움직일 수가 없어..."라고 포기한 오답입니다.
* 최적화 미수렴 값:
    * 최적화가 수렴하지 않아 튀어버린 값들을 제거하여 학습 데이터의 **S/N비(신호 대 잡음비)**를 높입니다.
* 물리적 불가능성:
    * 가속(`t_opt`)이나 조향(`s_opt`)이 물리적 한계치(99% 이상)에 도달한 데이터는 제외 (제어 모델의 유연성 저하 방지).


![alt text](../res/0119/3_filter.png)

* **90%가 들어옴:** 정답 Labeling이 매우 잘됨.
* **정밀도 필터링:** 최적화 과정에서 오차(loss)가 0.5 이상 발생한 불확실한 데이터는 삭제.
* **포화 상태 제거:** 가속(`t_opt`)이나 조향(`s_opt`)이 물리적 한계치(99% 이상)에 도달한 데이터는 제외 (제어 모델의 유연성 저하 방지).
* *참고: "정답지는 정확해야 함."*


![inference](../res/0119/inference.png)



---

## Inverse Dynamics MLP (Stage 4)

> Genesis 의 `Rigid Solver`는 `differentiable 하지 않음` 이에 따라 최종 목표인 `Sim2Sim Calibration`을 위해 Genesis의 Simulator를 흉내내는 미분가능한 MLP 모델을 학습

#### 작동 원리
* stage1,2 를 통해 Blender와 Genesis의 `환경` 및 `동역학`을 동기화 시켰고
* 이를 사용해 stage3 에서 GroundTruth 를 통한 지도학습의 정답 label 을 생성
* 정답값은 정확해야 하므로, stage3.5 를 통해 outlier 들을 필터링

### GT-based Supervised Trained Differentiable Inverse Dynamics MLP(policy)
* Differentiable Physics-informed Learning
* Learned Inverse Dynamics Controller
* Neural Inverse Mapping Policy



### Objective Function
```python
# L = (1.0 * L_throttle) + (1.0 * L_steer)
loss = (cfg.throttle_weight * loss_throttle) + (cfg.steer_weight * loss_steer)
```
* *참고: 스티어링(Steering)에 가중치를 주려 했으나 스로틀(Throttle)이 낮아서 오차가 누적됨.*




### Input Features (4 dim)

| 분류 | 변수명 | 설명 |
| :--- | :--- | :--- |
| State | $v_{long}$, $\omega_{yaw}$ | 현재 차량의 속도와 회전 빠르기 |
| Action | $T_{gt}$, $S_{gt}$ | Genesis에 맞게 변환된 Blender 제어량 (Ground Truth) |

### Hidden Layer
* **입력** : $(v_{long}, \omega_{yaw}, T_{gt}, S_{gt})$
* **은닉층 구조:** $128 \rightarrow 128 \rightarrow 64$
* **출력:** $(T^*, S^*)$



### Open Loop & Closed Loop 

### Open Loop

![](../res/0119/open.png)

* 오픈 루프는 입력이 출력에 영향을 주지만, 출력이 다시 입력에 영향을 주지 않는 방식
> "일단 명령을 내렸으면 결과가 어떻든 상관하지 않는다"는 쿨한(하지만 위험한) 방식

* **특징:**
    * 피드백(Feedback)이 없습니다.
    * 외부 방해(외란)나 환경 변화에 대응하지 못합니다.
    * 구조가 단순하고 비용이 저렴합니다.

결과 : <video controls src="../res/0119/openloop.mp4" title="Open Loop"></video>
* 차량이 전혀 제어가 안됨 (이유 찾지 못함)

### Closed Loop

* Closed Loop는 출력 결과를 센서로 측정하여 다시 입력으로 보내는(Feedback) 방식입니다. 목표값과 현재값의 차이(오차)를 계산해서 실시간으로 보정합니다.
* **특징:**
    * 피드백 루프가 존재합니다.
    * 오차(Error)를 줄이는 방향으로 계속 수정하므로 정확도가 높습니다.
    * 외부 방해에 강합니다 (예: 바람이 불어도 자동차가 차선을 유지함).
    * 구조가 복잡하고 설계 비용이 높습니다.
> open loop 방식으로는 도저히 결과가 나오지 않았음

## Closed Loop 방식으로 수정

 MLP input 에 `CTE`, `Heading Error`를 추가하여 Closed Loop 방식으로 수정
### term
 * `CTE` : 차량과 경로 사이의 상대적 오차
 * `Heading Error` : 차량의 회전 각도와 경로의 회전 각도의 차이


> MLP가 매 step `CTE`, `Heading Error`를 입력으로 받아 다음 step 의 `$T^*`, `$S^*$`를 출력

### Stage 3 Ground Truth 도 CTE , HE 를 계산하도록 해야함
#### 수정된 Ground Truth Output(stage3) 


| 항목 구분 | 컬럼명 (예시) | 의미 | 활용처 |
| :--- | :--- | :--- | :--- |
| **정답 (Labels)** | $t_{opt}, s_{opt}$ | Genesis에서 목표를 달성하기 위한 최적의 조작 | Stage 4의 Target($y$) |
| **상황 (Features)** | $CTE, HE$ | 그 조작을 결정해야 했던 당시의 오차 상황 | Stage 4의 Input($X$) |
| **상황 (Features)** | $la\_cte, la\_he$ | 5스텝 뒤에 발생할 미래의 오차 상황 | Stage 4의 Input($X$) |
| **성적 (Score)** | $loss$ | 이 정답이 얼마나 믿을만한가? (필터링용) | 데이터 정제용 |

#### 수정된 GT Objective Function
$$\mathcal{L} = \underbrace{(a_{gen} - a^*)^2 + 5(k_{gen} - k^*)^2}_{\text{Motion Matching}} + \underbrace{\beta_1 \cdot CTE^2 + \beta_2 \cdot HE^2}_{\text{Path Alignment}}$$

* `CTE` , `HE` 에 대해 penalty 항을 부여하여 closed loop로 재정의된 목적함수 설계



### 수정된 Input Features (8 dim)
$$Input = [v_{long}, \omega, T^*, S^*, CTE, HE, la\_CTE, la\_HE]$$

* State : [v_long, \omega]
* Action : [T^*, S^*]
* Feedback : [CTE, HE]
* Future Lookahead : [la\_CTE, la\_HE] (lookahead step 만큼 뒤에서 벌어질 오차)

### MLP Architecture
#### Input
$$Input (8D) = [v_{long}, \omega, T^*, S^*, CTE, HE, la\_CTE, la\_HE]$$
* State : [v_long, \omega]
* Action : [T^*, S^*]
* Feedback : [CTE, HE]
* Future Lookahead : [la\_CTE, la\_HE] (lookahead step 만큼 뒤에서 벌어질 오차)


#### Hidden Layer


$$ H_1 = \text{Dropout}(\text{ReLU}(W_1 X_{in} + b_1), p=0.2) $$
$$ H_2 = \text{Dropout}(\text{ReLU}(W_2 H_1 + b_2), p=0.2) $$
$$ H_3 = \text{Dropout}(\text{ReLU}(W_3 H_2 + b_3), p=0.2) $$
$$ Y_{out} = \text{Tanh}(W_4 H_3 + b_4) = [T^*, S^*] $$


#### Output
$$Output (2D) = [T^*, S^*]$$
* updated (T*,S*)



#### 의문점
> Closed loop로 하는게 잘못된 방법이 아닌지? Inverse Dynamics를 통한 sim2sim Calibration으로 접근해야하는데 cte , he 를넣는 순간 이게 RL에 의한 움직임 제어가 되는건 아닐까?
* 위와 같은 우려가 생김


##### 강화학습과의 차이점
* RL (Reinforcement Learning): 정답이 없는 상태에서 보상(Reward)을 통해 맨땅에 헤딩하며 행동 전략(Policy)을 창조
* 우리의 방식 (Inverse Dynamics): Blender의 궤적이 이미 존재합니다. 우리는 그 궤적을 Genesis 에서 재현하기 위한 **'수학적 변환 함수'**를 학습하는 것입니다. $CTE$를 넣는 것은 모델이 "지금 상황에서 블렌더의 의도를 제네시스 물리로 어떻게 번역해야 가장 정확한지"를 더 잘 이해하게 돕는 **추가 정보(Context)**일 뿐

##### 'Calibration'의 정의: 점(Point)이 아닌 선(Path)의 일치
* 순수 역동역학($f^{-1}$)은 완벽한 모델을 가정하지만, 현실(Sim2Sim)에서는 모델 오차가 반드시 존재합니다. $CTE$와 $HE$는 그 모델 오차를 실시간으로 상쇄하기 위한 파라미터로 작동
* 동적인 상황(Dynamic Context)까지 고려한 더 고차원적인 캘리브레이션

---


## 5. Test/Inference (Stage 5)

### for every step  
* **입력 (8D):** State(`V_long`, `omega`), Action(`$T^*$`, `$S^*$`), Feedback(`CTE`, `HE`), Future Lookahead(`la_CTE`, `la_HE`)
* MLP Model
* **출력 (2D):** `t_opt` (가속), `s_opt` (조향) -> Tanh (-1 ~ 1)를 거쳐 제어(Control)로 입력됨.


$$(T, S)_{final} = \text{MLP}(v, \omega, t_b, s_b, \mathbf{CTE}, \mathbf{HE}, \dots)$$



| 구분 | Stage 4 (Training) | Stage 5 (Inference/Testing) |
| :--- | :--- | :--- |
| **데이터 출처** | 이미 저장된 CSV 파일 | Genesis 실시간 센서 |
| **CTE/HE** | 미리 계산된 정적(Static) 값 | 매 스텝 새로 계산되는 동적(Dynamic) 값 |
| **정답(Target)** | 있음 ($T_{gt}, S_{gt}$) | 없음 (모델이 직접 생성) |
| **목표** | 오차와 정답 사이의 패턴 학습 | 성공적인 주행 및 경로 완주 |

### 주행 비교
