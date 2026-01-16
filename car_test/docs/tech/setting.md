## 1. 차량 설정 (Blender)

### RBC addon rigged car
  
![](../../res/0115/car.png)  



* RBC addon rigged car 사용


![](../../res/0115/car3.png)
  
* mesh + armature &rarr; car 형상 + 차량 제어 constraints 로 구성

![](../../res/0115/bone.png)
* bone 에 걸리는 물리력을 python script를 통해 데이터를 추출함


![](../../res/0115/car4.png)
* 자동차 세팅 조절 (bone &rarr; constraints / physics &rarr; 세팅 변경)

## 2. 경로 설정 (Blender)
* bezier curve로 경로를 그린 뒤, path follow 클릭 후 bezier curve 선택

<video controls src="../../res/0115/car.mp4" title="Title"></video>
비디오 링크: https://drive.google.com/file/d/1iwlGOhXwCR_1VFKzuBZLOm6HuUWURWuv/view?usp=sharing 

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
### 4.1) 도로를 구간별 Mesh로 분할
- 예: `Road_Dry`, `Road_Wet`, `Road_Ice`
- 방법:
  - Edit Mode → Edge 선택
  - `P` → Selection 으로 분리
  - 또는 처음부터 타일형 도로 설계

### 4.2) 각 도로 구간에 Rigid Body 설정
- Type: **Passive** (충돌 x)
- Shape: **Mesh** (입자 하나하나 계산하여 계산량은 증가하지만 정확한 계산)
- Animated: x

### 4.3) 마찰 계수 설정


1. mesh 클릭 후 우측 아래 physics properties 로 이동

![](../../res/0115/constraints.png)

2. surface response → friction 로 마찰계수를 조정 가능



#### 마찰계수 예시
| 구간 | Friction | 용도 |
|------|----------|------|
| Dry  | 0.8 ~ 1.0 | 마른 아스팔트 |
| Wet  | 0.3 ~ 0.5 | 빗길 |
| Ice  | 0.05 ~ 0.15 | 빙판 |



#### 하나의 terrain 에서 차량이 움직이는 경우
![](../../res/0115/road2.png)
* 산악 지형: 연속적 &rarr; genesis 에서는 object 로 불러와서 terrain field 로 계산하는 것이 유리하다고 함



### 4.4) 차량 바퀴 설정
- Rigid Body: **Active**(충돌 가능)
- Shape: **Cylinder**
- Friction: **1.0**

## 5 주행 데이터 획득 방법
![](../../res/0115/script.png)
* scripting 으로 python 을 통한 제어 가능

![](../../res/0115/script2.png)
* import bpy 를 해야 python이 작동함
* 우측 상단 run 버튼을 눌러 실행
* 좌측 하단 콘솔창

아래 코드를 사용하여 데이터 추출
데이터 추출 코드 링크: [data_extracter_blender](../../src/on_off_data_blender_data.py)


## 6 구현 코드 검증
* train 코드 : [train_ppo](../../src/train_ppo.py)
* inference/실행 코드 : [test_ppo](../../src/test_ppo.py)
* blender 데이터 추출 코드 : [data_extracter_blender](../../src/on_off_data_blender_data.py)


## 6.1 PPO Direct Control: MLP 구조 상세

> **목표**: 경로를 따라가는 차량의 **조향(Steer)**과 **가속/제동(Throttle/Brake)**을 MLP가 직접 출력

---

### 6.1.1 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────────────┐
│  World State (Simulation)                                               │
│  • 차량 위치, 자세, 속도, 각속도                                           │
│  • 경로 Waypoints                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Observation (18차원)                                                    │
│  • 목표점 상대좌표, 속도, 각속도, 중력, 접선방향, 슬립, 이전행동            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Actor Network (Policy MLP)                                             │
│  FC(18→128) → ReLU → FC(128→128) → ReLU → FC(128→2)                     │
│  + log_std (learnable parameter)                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Action (2차원)                                                          │
│  • accel_brake ∈ [-1, 1]: 양수=가속, 음수=제동                           │
│  • steer ∈ [-1, 1]: 양수=우회전, 음수=좌회전                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Physics Simulation → Reward → PPO Update                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 6.1.2 Input: Observation (18차원)

#### 변수 목록

| Index | 변수 | 차원 | 의미 | 좌표계 |
|:---:|:---|:---:|:---|:---|
| 0-2 | `target_rel` | 3 | 목표점 상대위치 $(x_t, y_t, z_t)$ | Body |
| 3-5 | `v_body` | 3 | 속도 $(v_x, v_y, v_z)$ | Body |
| 6-8 | `ω_body` | 3 | 각속도 $(ω_x, ω_y, ω_z)$ | Body |
| 9-11 | `g_body` | 3 | 중력 방향 $(g_x, g_y, g_z)$ | Body |
| 12-14 | `tangent_rel` | 3 | 경로 접선 방향 | Body |
| 15 | `slip_proxy` | 1 | 슬립 지표 | - |
| 16-17 | `prev_action` | 2 | 이전 행동 | - |

#### 주요 변수 수식

**목표점 상대위치** (Arc-Length Lookahead):
$$
\vec{p}_{target}^{body} = R_{car}^T \cdot (\vec{p}_{target}^{world} - \vec{p}_{car}^{world})
$$

- $s_{target} = s_{nearest} + L$ (L = 5.0m)
- target_idx = $\arg\min_i |s_i - s_{target}|$

**속도 변환**:
$$
\vec{v}_{body} = R_{car}^T \cdot \vec{v}_{world}
$$

**슬립 지표**:
$$
\text{slip} = \frac{a_{prev} - v_x}{v_{max}}
$$

---

### 6.1.3 Actor Network (Policy MLP)

#### 구조

```
Input: obs ∈ ℝ^18
        ↓
    ┌───────────────────┐
    │  Linear(18, 128)  │  W₁ ∈ ℝ^(128×18), b₁ ∈ ℝ^128
    └───────────────────┘
        ↓
    ┌───────────────────┐
    │      ReLU         │  max(0, x)
    └───────────────────┘
        ↓
    ┌───────────────────┐
    │  Linear(128, 128) │  W₂ ∈ ℝ^(128×128), b₂ ∈ ℝ^128
    └───────────────────┘
        ↓
    ┌───────────────────┐
    │      ReLU         │
    └───────────────────┘
        ↓
    ┌───────────────────┐
    │  Linear(128, 2)   │  W₃ ∈ ℝ^(2×128), b₃ ∈ ℝ^2
    └───────────────────┘
        ↓
    mean ∈ ℝ^2  (μ_throttle, μ_steer)
        +
    log_std ∈ ℝ^2  (학습 가능한 파라미터)
        ↓
    ┌───────────────────┐
    │  Gaussian Dist    │  π(a|s) = N(μ, σ²)
    └───────────────────┘
        ↓
    ┌───────────────────┐
    │      tanh         │  Squash to [-1, 1]
    └───────────────────┘
        ↓
Output: action ∈ [-1, 1]²
```

#### 수식

**Forward Pass**:
$$
\begin{align}
h_1 &= \text{ReLU}(W_1 \cdot \text{obs} + b_1) \\
h_2 &= \text{ReLU}(W_2 \cdot h_1 + b_2) \\
\mu &= W_3 \cdot h_2 + b_3 \\
\sigma &= \exp(\log\_std)
\end{align}
$$

**Action Sampling (Stochastic)**:
$$
a_{raw} \sim \mathcal{N}(\mu, \sigma^2) \\
a = \tanh(a_{raw})
$$

**Log Probability (Squashed Gaussian)**:
$$
\log \pi(a|s) = \log \mathcal{N}(a_{raw}|\mu, \sigma^2) - \sum_i \log(1 - \tanh^2(a_{raw,i}))
$$

---

### 6.1.4 Critic Network (Value MLP)

#### 구조

```
Input: obs ∈ ℝ^18
        ↓
    Linear(18, 128) → ReLU
        ↓
    Linear(128, 128) → ReLU
        ↓
    Linear(128, 1)
        ↓
Output: V(s) ∈ ℝ  (상태 가치)
```

#### 수식

$$
V(s) = W_v \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot \text{obs} + b_1) + b_2) + b_v
$$

---

### 6.1.5 Output: Action (2차원)

| Index | 변수 | 범위 | 물리적 의미 |
|:---:|:---|:---:|:---|
| 0 | `accel_brake` | [-1, 1] | 가속(+) / 제동(-) |
| 1 | `steer` | [-1, 1] | 우(+) / 좌(-) 조향 |

#### Action → 물리 제어 변환

**Throttle/Brake**:
$$
\begin{align}
\tau_{engine} &= \max(a_0, 0) \cdot \tau_{max}^{engine} \\
\tau_{brake} &= \max(-a_0, 0) \cdot \tau_{max}^{brake} \\
\tau_{drive} &= \max(\tau_{engine} - \tau_{brake}, 0) \\
\omega_{target} &= \frac{\tau_{drive} / \tau_{max}^{engine} \cdot v_{target}}{r_{wheel}}
\end{align}
$$

**Steering**:
$$
\delta = a_1 \cdot \delta_{max} \quad \text{(Position Control)}
$$

---

### 6.1.6 Learning: PPO Algorithm

#### Reward Function

$$
R = \underbrace{R_{align} + 2 \cdot R_{recover}}_{\text{Steering}} + \underbrace{R_{proj} + R_{arc}}_{\text{Progress}} + R_{forward} - \underbrace{P_{steer} - P_{rate} - P_{speed} - P_{stuck}}_{\text{Penalties}}
$$

| 항목 | 수식 | 의미 |
|:---|:---|:---|
| $R_{align}$ | $\text{clamp}(y_{target} \cdot steer, -0.5, 0.5)$ | 목표 방향 조향 |
| $R_{recover}$ | $\text{clamp}(\|e_{prev}\| - \|e_{curr}\|, -0.2, 0.2)$ | 오차 감소 |
| $R_{proj}$ | $\text{clamp}(v_x \cdot t_x + v_y \cdot t_y, -0.2, 0.5)$ | 경로 방향 속도 |
| $R_{arc}$ | $\text{clamp}(s_{curr} - s_{prev}, 0, 0.5)$ | 진행 거리 |

#### PPO Update

**Advantage (GAE)**:
$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**Policy Loss (Clipped)**:
$$
L^{CLIP} = -\mathbb{E}\left[\min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \hat{A}, \text{clip}(\cdot, 1-\epsilon, 1+\epsilon) \hat{A}\right)\right]
$$

**Value Loss**:
$$
L^{VF} = \mathbb{E}\left[(V_\theta(s) - R_t)^2\right]
$$

**Total Loss**:
$$
L = L^{CLIP} + c_1 L^{VF} - c_2 H[\pi_\theta]
$$

---

### 6.1.7 파라미터 요약

| 항목 | 값 |
|:---|:---:|
| Observation Dim | 18 |
| Action Dim | 2 |
| Hidden Layers | [128, 128] |
| Activation | ReLU |
| Output Activation | tanh |
| Learning Rate | 3e-4 |
| Discount (γ) | 0.99 |
| GAE Lambda (λ) | 0.95 |
| Clip Epsilon (ε) | 0.2 |
| Entropy Coef (c₂) | 0.01 |
| Value Coef (c₁) | 0.5 |

---

### 6.1.8 도식 요약

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PPO Direct Control Pipeline                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────┐     │
│  │ World   │ →  │  Observation     │ →  │ Actor MLP   │ →  │ Action   │     │
│  │ State   │    │  (18-dim)        │    │ (128,128)   │    │ (2-dim)  │     │
│  └─────────┘    └──────────────────┘    └─────────────┘    └──────────┘     │
│       ↑                                        ↓                 ↓          │
│       │                                 ┌─────────────┐    ┌──────────┐     │
│       │                                 │ Critic MLP  │    │ Physics  │     │
│       │                                 │ → V(s)      │    │ Sim      │     │
│       │                                 └─────────────┘    └──────────┘     │
│       │                                        ↓                 ↓          │
│       │                                 ┌─────────────────────────┐         │
│       └─────────────────────────────────│      PPO Update         │         │
│                                         │  (Policy + Value Loss)  │         │
│                                         └─────────────────────────┘         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```
---



## 7. Genesis Real2Sim Physics 학습 방법
(내용 작성 예정)

```
Real Data (input)
 (경로 / 마찰 / 지형)
      ↓
Parameter Mapping (AI training)
      ↓
Simulation (Genesis)
      ↓
Policy Output (by trained AI model)
 (steer / throttle)
      ↓
Real Control Interface 

```

## 필요한 데이터
* “그래서 우리가 뭘 주고, 뭘 받으면 되지?”

#### Input (from company side : 현재는 blender 대체)

1. 경로 데이터 (waypoints)

2. 환경 파라미터 (마찰, 경사, 노면 구분)

3. 차량 기본 제원 (wheelbase, limits)


#### Output (from Genesis)

1. Steering command

2. Throttle / acceleration command

3. Optional(아직 미구현): 상태 로그 (lat error, speed, slip)