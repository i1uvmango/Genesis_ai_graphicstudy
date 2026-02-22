# Implementation Plan: Path-Error Based MLP Controller

> **핵심 목표 (한 문장 정의)**:
> 이 모델은 행동을 그대로 따라 하는 것이 아니라,
> **차량이 블렌더 기준 경로 안에서 동일한 속력과 방향으로 주행하도록
> 조향과 쓰로틀을 조정하는 경로 추종 제어기**이다.

---

## 1. 설계 개요

### 1.1 핵심 컨셉
| 항목 | 기존 (Behavior Cloning) | 현행 (Pure Pursuit + PPO) |
|:---|:---|:---|
| **학습 방식** | Supervised Learning (지도학습) | Reinforcement Learning (강화학습) |
| **최적화 대상** | Expert Action (조향/가속) 모방 | **주행 오차 (e_lat, e_head) 보상 최적화** |
| **제어기 구조** | Black-box Neural Network | **Hybrid (Pure Pursuit + Residual MLP)** |

### 1.2 MLP의 역할 명확화 

> **MLP는 조향·쓰로틀을 직접 결정하는 것이 아니라,
> 현재 제어값과 차량 상태를 입력으로 받아
> 경로를 따라가기 위해 필요한 보정값을 출력하는 제어기이다.**

```
┌──────────────────────────────────────────────────┐
│  MLP는 "보정기(Corrector)"                        │
├──────────────────────────────────────────────────┤
│                                                  │
│  Input:                                          │
│    - 현재 steering (지금 핸들이 꺾인 정도)          │
│    - 현재 throttle (지금 페달이 밟힌 정도)          │
│    - 차량 상태 (속도, 방향, 관성 등 가상 센서 값)    │
│                                                  │
│  Output:                                         │
│    - 조정(보정)된 steering                        │
│    - 조정(보정)된 throttle                        │
│                                                  │
│  적용: steering_new = steering + Δsteering       │
│        throttle_new = throttle + Δthrottle       │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 1.3 MLP 구조 선택

**설계**: Pure Pursuit 기반 조향 + 분리 MLP

| 구성 | 입력 | 출력 |
|:---|:---|:---|
| **Pure Pursuit** | 현재 위치, Lookahead Point | `steer_pp` (기본 조향) |
| **SteerCorrectionMLP** | `[steer_pp, e_lat, e_head, speed]` | `Δsteer` (보정값) |
| **ThrottleMLP** | `[e_lat, e_head, speed, κ]` | `throttle` (0~1) |

**아키텍처**:
```
[Steering]
┌─────────────────────────────────────────┐
│  Lookahead Point → Pure Pursuit → steer_pp │
│                        +                     │
│  Error State → SteerMLP → Δsteer            │
│                        ↓                     │
│         steer_final = steer_pp + Δsteer     │
└─────────────────────────────────────────┘

[Throttle]
┌─────────────────────────────────────────┐
│  Error State → ThrottleMLP → throttle ∈ [0,1] │
└─────────────────────────────────────────┘
```

**이유**: 
- Pure Pursuit가 90% 기본 조향 담당 → 학습 안정성 ↑
- MLP는 작은 보정만 담당 (Residual Learning)
- Steering/Throttle 분리 → 독립적 최적화 가능


### 1.4 MLP 입력 변수

**SteerCorrectionMLP (4D)**:
| Index | 변수 | 설명 |
|:---:|:---|:---|
| 0 | `steer_pp` | Pure Pursuit 기본 조향값 |
| 1 | `e_lat` | 횡방향 오차 |
| 2 | `e_head` | 헤딩 오차 |
| 3 | `speed` | 현재 속도 |

**ThrottleMLP (4D)**:
| Index | 변수 | 설명 |
|:---:|:---|:---|
| 0 | `e_lat` | 횡방향 오차 |
| 1 | `e_head` | 헤딩 오차 |
| 2 | `speed` | 현재 속도 |
| 3 | `κ` | 곡률 (미구현, 0.0) |

---

## 2. 목표 지점 선택 (Lookahead Point)

### 2.1 설계 원칙

> **"조금 앞을 보고 미리 핸들을 꺽는다"**

| 방식 | 설명 | 사용 위치 |
|:---|:---|:---|
| **Lookahead Point** | 현재 위치에서 일정 거리(`L_d`) 앞의 경로 지점 | **Pure Pursuit 조향** |
| **Closest Point** | 현재 위치에서 경로까지 최단거리 지점 | **Error State 계산** |

### 2.2 Lookahead Distance (`L_d`)

```python
LOOKAHEAD_DIST = 1.0  # 미터 (속도에 따라 조정 가능)
```

- **높은 속도**: `L_d` 증가 → 더 먼 앞을 보고 조향
- **낮은 속도/급커브**: `L_d` 감소 → 가까운 목표 추종

### 2.3 구현

```python
def find_lookahead_point(pos_x, pos_y, lookahead_dist, reference):
    """
    현재 위치에서 lookahead_dist 만큼 떨어진 경로 지점을 찾는다.
    Pure Pursuit 조향 계산에 사용.
    """
    closest_idx = find_closest_reference(pos_x, pos_y, reference)
    
    for i in range(closest_idx, len(reference["pos_x"])):
        dx = reference["pos_x"][i] - pos_x
        dy = reference["pos_y"][i] - pos_y
        dist = math.sqrt(dx**2 + dy**2)
        if dist >= lookahead_dist:
            return i
    
    return len(reference["pos_x"]) - 1
```

### 2.4 Error State는 Closest Point 기준

```python
def find_closest_reference(pos_x, pos_y, reference):
    """
    현재 위치에서 경로까지 최단거리 지점을 찾는다.
    e_lat, e_head, e_speed 계산에 사용.
    """
    distances = np.sqrt(
        (reference["pos_x"] - pos_x)**2 +
        (reference["pos_y"] - pos_y)**2
    )
    return np.argmin(distances)
```

## 3. Loss 함수 정의

### 3.1 steering loss

**"Genesis 차량이 Reference Path를 얼마나 잘 따라가는가?"**

#### ~~기존 (Position Distance 기반)~~ → 수정됨
```python
# 기존 방식 (사용 안 함)
L_path = Σ || pos_genesis - pos_ref ||²
```

#### 수정된 Steering Loss (방향 분리)
```python
L_path = w_lat * e_lat² + w_head * e_head²
```

| 변수 | 설명 | 계산식 |
|:---|:---|:---|
| `e_lat` | 횡방향 오차(위치) | `dy*cos(ref_yaw) - dx*sin(ref_yaw)` |
| `e_head` | 헤딩 오차(방향) | `wrap_to_pi(heading - ref_yaw)` |
| `w_lat` | 횡방향 weight(위치) | 1.0 |
| `w_head` | 헤딩 weight(방향) | 0.5 (라디안 스케일 조정) |


> `pos distance²`의 문제:
> ```
> 1. 차량이 경로보다 오른쪽으로 1m 벗어남
>    → pos_error = 1m
>    → MLP: "왼쪽으로 가야 함" (맞음 ✓)
> 
> 2. 차량이 경로 위에 있지만 heading이 45도 틀어짐
>    → pos_error = 0.1m (아직 작음)
>    → MLP: "거의 괜찮네" (틀림! 곧 벗어날 것 ✗)
> ```
> 
> `e_lat² + e_head²`는 이 문제를 해결:
> - `e_lat`이 작아도 `e_head`가 크면 → 큰 penalty
> - **Heading error가 미래의 lateral error를 예측하는 효과**
> - Gradient 방향이 물리적으로 명확함

| 구분 | `pos distance²` | `e_lat² + e_head²` |
|:---|:---:|:---:|
| Gradient 명확성 | heading(방향) 섞임 | 독립적 |
| 미래 오차 예측 | x | e_head로 예측 |
| 튜닝 용이성 | 낮음 | w_lat, w_head 분리 |

**권장 Weight**:
```python
w_lat = 1.0   # 횡방향 오차
w_head = 0.5  # 방향 오차 (라디안 단위이므로 스케일 조정)
```

### 3.2 throttle loss (비대칭 Penalty)

**"느린 건 실패, 빠른 건 허용 가능"**

```python
e_speed = speed - ref_speed

if e_speed < 0:  # 목표보다 느림 → 강한 penalty
    L_speed = w_slow * e_speed²
else:            # 목표보다 빠름 → 약한 penalty
    L_speed = w_fast * e_speed²

L_throttle = L_speed + w_smooth * Δthrottle²
```

| 변수 | 설명 | 권장값 |
|:---|:---|:---:|
| `w_slow` | 느린 경우 weight | **0.5** |
| `w_fast` | 빠른 경우 weight | 0.05 |
| `w_smooth` | 쓰로틀 변화량 weight | 0.1 |

**비대칭 설계 이유**:
| 상황 | e_speed | 실제 위험도 | Penalty |
|:---|:---:|:---:|:---:|
| 목표보다 1m/s **느림** | -1 | ⚠️ 높음 (정지/실패) | **0.5** |
| 목표보다 1m/s **빠름** | +1 | ✅ 낮음 (허용 가능) | 0.05 |

### 3.3 Speed Gate (Inference 시 Hard Rule) 

> **"방향이 크게 틀어졌을 때는 학습된 모델보다 안전 규칙이 우선한다."**

Loss로 속도를 제어하는 대신, **추론(Inference) 단계에서 강력한 Hard Rule**을 적용합니다.

```python
# Speed Gate Logic
HEAD_THRESHOLD = 0.5  # 약 30도

if abs(e_head) > HEAD_THRESHOLD:
    # 방향이 많이 틀어졌으면, 가속을 제한하거나 감속한다.
    throttle = min(throttle, 0.3)
```
*   **이유**: 고속 주행 중 큰 조향은 차량을 불안정하게 만듭니다(전복/스핀). 방향을 먼저 잡고 가속해야 합니다.

### 3.4 Total Loss 
**학습 방식에 따라 Loss 또는 Reward로 사용**
```python
L_steer   = w_lat * e_lat² + w_head * e_head²
```
```python
L_throttle = w_slow/fast * e_speed² + w_smooth * Δthrottle²
```

---

## 4. Loss 최적화 방법론

### 4.1 시도한 방법 비교

| 방법 | 설명 | 결과 | 비고 | 사용 여부 |
|:---|:---|:---:|:---|:---|
| **Genesis Differentiable Physics** | `sim_options.requires_grad=True`로 물리 시뮬레이션을 통한 gradient flow | ❌ 실패 | `get_qpos().requires_grad=False` 반환 | `X` |
| **PyTorch 변환** | PyTorch → Genesis tensor 변환으로 gradient 연결 시도 | `X` | Scene context 단절 | `X` |

### 4.2 Genesis Differentiable Physics 시도 결과

[](../26_0107_tryRigidSolverDiff.md)

### 4.3 PyTorch 변환 시도 결과

[](../26_0107_tryPyTorch.md)

### 5.4 Pure Persuit Algorithm + 2MLP
* 1MLP: steer_pp + error_state(mlp) -> steer
* 2MLP: error_state(mlp) -> throttle

` 링크 `



## 5. 학습 데이터 분할 (Validation Strategy)

### Lap-Aware Time-Block Split 

시계열 데이터의 특성상 랜덤 셔플(Random Split)은 **미래 데이터 유출(Leakage)** 문제를 일으킵니다. 이를 방지하기 위해 **Lap(바퀴)** 단위로 데이터를 분할합니다.

| 구분 | 설명 | 비율 (예시) |
|:---|:---|:---|
| **Train** | 초기 80% 주행 구간 (앞쪽 Lap) | Lap 1 ~ 8 |
| **Validation** | 후반 20% 주행 구간 (뒤쪽 Lap) | Lap 9 ~ 10 |

*   **기준**: 1 Lap ≈ 250 Frames
*   **목표**: "학습하지 않은 미래의 경로"를 얼마나 잘 따라가는지 평가계산

## 6. Error State 계산

### 6.1 MLP 입력 (5D)

| Index | 변수 | 설명 |
|:---:|:---|:---|
| 0 | `steer_t` | 현재 조향값 |
| 1 | `throttle_t` | 현재 가속값 |
| 2 | `e_lat` | 횡방향 오차 (경로에서 벗어난 거리) |
| 3 | `e_head` | 헤딩 오차 (방향 차이) |
| 4 | `e_speed` | 속도 오차 |

### 6.2 계산 코드

```python
def compute_error_state(pos, heading, speed, reference):
    # 1. 최단거리 기준 Reference 점 찾기
    closest_idx = find_closest_reference(pos, reference)
    
    ref_x = reference["pos_x"][closest_idx]
    ref_y = reference["pos_y"][closest_idx]
    ref_yaw = reference["heading"][closest_idx]
    ref_v = reference["speed"][closest_idx]
    
    # 2. Lateral Error (횡방향 오차)
    dx = pos[0] - ref_x
    dy = pos[1] - ref_y
    e_lat = dx * sin(ref_yaw) - dy * cos(ref_yaw)
    
    # 3. Heading Error (방향 오차)
    e_head = wrap_to_pi(heading - ref_yaw)
    
    # 4. Speed Error (속도 오차)
    e_speed = speed - ref_v
    
    return e_lat, e_head, e_speed
```

---

## 7. Fallback 학습 전략 (필수)

### 6.1 미분 기반 학습이 불안정할 경우

> **"경로 오차 기반 미분 학습이 안정적이지 않을 경우,
> 동일한 경로 오차를 보상(Reward)으로 사용하는 강화학습 방식으로 전환할 수 있다."**

### 7.1 대안 전략

| 상황 | 전략 |
|:---|:---|
| Gradient 폭발/소실 | **RL (PPO)** 전환: Reward = -L_path |
| 초기 수렴 불안정 | **BC + RL 혼합**: BC로 초기화 후 RL fine-tuning |
| 접촉/마찰 미분 불가 | Genesis의 미분 가능 GJK/EPA 활용 또는 RL |

### 7.2 RL 전환 시 Reward 설계

```python
# 미분 학습의 L_path를 그대로 Reward로 변환
reward = -L_path - 0.01 * L_smooth
```

---

## 8. 학습 과정 요약

```
1. CSV 로드 (Reference Path)
   └─→ drive_8_test.csv → pos_x, pos_y, heading, speed 추출

2. Genesis 시뮬레이션 초기화
   └─→ 차량을 경로 시작점에 배치

3. 매 Step마다:
   a) Genesis에서 현재 상태 측정
   b) 최단거리 기준 Reference 점 찾기
   c) Error State 계산 (e_lat, e_head, e_speed)
   d) MLP에 [steer, throttle, e_lat, e_head, e_speed] 입력
   e) MLP 출력: [Δsteer, Δthrottle]
   f) 제어값 업데이트 및 Genesis에 적용

4. K-Step (Horizon) 후 Loss 계산
   └─→ L_total = L_path + L_smooth

5. Backward & Update (MLP 파라미터)
```

---

## 9. 요약

| 항목 | 내용 |
|:---|:---|
| **모델 목적** | 경로 추종 제어기 (보정기) |
| **MLP 역할** | 기존 제어값을 **보정**하여 경로 오차 최소화 |
| **Reference 기준** | 최단거리 기준 (로봇틱스 표준) |
| **Horizon** | 3~5 프레임 (Short-Horizon) |
| **Loss** | L_path (경로) + L_smooth (부드러움) |
| **MLP 구조** | 통합 MLP (steering/throttle 동시 출력) |
| **Fallback** | 미분 불안정 시 RL(PPO)로 전환 가능 |
