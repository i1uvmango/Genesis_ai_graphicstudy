# Blender2Genesis: Sim2Sim Calibration via Inverse Dynamics Mapping

> **한 줄 요약**: 운동학(Kinematics) 기반 시뮬레이터 Blender의 주행 궤적을, 동역학(Dynamics) 기반 시뮬레이터 Genesis가 물리적으로 동일하게 재현하도록 하는 **실시간 Inverse Dynamics Mapper**를 설계·구현한다.

---

## 목차

1. [연구 배경 및 동기](#1-연구-배경-및-동기)
2. [문제 정의: 왜 T,S 전달만으로는 부족한가](#2-문제-정의-왜-ts-전달만으로는-부족한가)
3. [시뮬레이션 환경](#3-시뮬레이션-환경)
4. [파이프라인 개요](#4-파이프라인-개요)
5. [Stage 1: MPPI — 정답 데이터 생성](#5-stage-1-mppi--정답-데이터-생성)
6. [Stage 2: MLP — Inverse Dynamics Mapper 학습](#6-stage-2-mlp--inverse-dynamics-mapper-학습)
7. [Stage 3: 추론 및 일반화 검증](#7-stage-3-추론-및-일반화-검증)
8. [결론 및 향후 연구](#8-결론-및-향후-연구)

---

## 1. 연구 배경 및 동기

### 최종 목표: Sim2Real Transfer

자율주행 에이전트를 현실에 바로 배포하는 것은 높은 비용과 위험을 수반한다. 이를 해결하는 핵심 전략은 **시뮬레이터에서 학습한 정책(Policy)을 현실에 Zero-shot으로 전이**하는 것이다.

```
Real World ──Real2Sim──▶ Simulation ──Sim2Real──▶ Real World
                              ▲
                        (본 연구 대상)
```

> **Sim2Real이 가능하려면, Real2Sim이 완벽해야 한다.**  
> 현실의 물리 현상을 시뮬레이터가 정확히 재현할수록, 시뮬레이터에서 학습된 정책은 현실에서도 그대로 작동한다.

### 본 연구 단계: Sim2Sim

현실 데이터 수집에 제약이 있는 상황에서, 본 연구는 **Sim2Sim Calibration**을 먼저 수행한다.

| 단계 | 소스 | 타겟 | 설명 |
|---|---|---|---|
| **Sim2Sim** (본 연구) | Blender | Genesis | 시뮬레이터 간 물리 모델 차이 극복 |
| Real2Sim (다음 단계) | Real World | Genesis | 동일한 파이프라인으로 확장 가능 |

Blender를 "이상적 현실의 대리자(Surrogate)"로 설정하고, Genesis가 Blender의 주행을 완벽히 재현하도록 Calibration을 수행한다. **이 파이프라인이 성립하면, Blender 자리에 Real World를 그대로 치환하여 Real2Sim을 구현할 수 있다.**

---

## 2. 문제 정의: 왜 T,S 전달만으로는 부족한가

### 핵심 관찰: 동일 입력 → 서로 다른 궤적

가장 단순한 접근은 Blender의 제어 입력 `(Throttle, Steer)`를 Genesis에 그대로 전달하는 것이다. 그러나 이 방식은 **근본적으로 실패**한다.

| | Blender (Kinematics) | Genesis (Dynamics) |
|---|---|---|
| **동일 (T,S) 입력 시** | ![stage1](../res/0216/stage1.png) | ![stage2](../res/0216/stage2_2.png) |
| **물리 모델** | 이상적·수학적 경로 추종 | 관성, 마찰, 슬립 등 물리 현상 반영 |
| **결과** | 경로와 완벽하게 일치 | 경로 이탈, 오차 누적 |

**두 시뮬레이터는 물리 엔진 자체가 다르기 때문에, 동일한 제어 입력은 동일한 움직임을 보장하지 않는다.**

### 무엇이 다른가: Kinematics vs Dynamics

| 구분 | Blender (Kinematics) | Genesis (Dynamics) |
|---|---|---|
| 기반 모델 | Unicycle Kinematics (UKMAC) | Rigid Body Dynamics |
| 제어 변수 | acceleration, curvature | velocity, angular velocity, yaw rate |
| 현실 제약 | `X` 슬립, 관성, 마찰 무시 | `O` 슬립, 원심력, 마찰 반영 |
| 경로 추종 | 수학적으로 완벽 | 물리적 오차 존재 |
| 역할 | 이상적 레퍼런스 | 현실과 유사한 동역학 |

### 따라서, 본 연구의 핵심 질문은:

> **"Blender의 동역학적 주행 궤적(Dynamical Trajectory)이 주어졌을 때,  
> Genesis 물리 엔진 안에서 동일한 움직임을 재현하는 최적 제어 입력 `(T*, S*)`는 무엇인가?"**

이는 단순한 인터페이스 매핑이 아니라, **두 물리 세계 사이의 Inverse Dynamics 문제**다.

---

## 3. 시뮬레이션 환경

### Blender — 이상적 주행의 레퍼런스

https://github.com/user-attachments/assets/e0609422-8a9c-4695-98d5-4110debb4fde


Blender는 Bullet Physics 기반의 운동학 시뮬레이터로, 본 연구에서 **Real World의 대리자**로 사용된다.

- 차량의 움직임이 수학적으로 정의되며, 슬립·관성 등 동역학적 노이즈가 없음
- 임의의 경로를 정확히 추종하는 이상적 주행 데이터 생성 가능
- **역할**: 목표 궤적(Reference Trajectory) 제공

### Genesis — 물리 기반 시뮬레이터

Genesis는 GenesisAI의 자체 물리 엔진으로, **현실의 동역학을 시뮬레이션**하는 공간이다.

- 질량, 관성, 원심력, 마찰 등 실제 물리 현상을 반영
- Bullet Engine 대비 **43만 배 빠른 연산 속도** → 대규모 RL/Imitation Learning 학습에 적합
- Non-differentiable Solver → Gradient 기반 최적화 불가 (MPPI 도입 배경)
- **역할**: 최종 정책이 동작할 물리 세계

> **Genesis의 동역학 상태 공간**
> ```
> State = { velocity, angular_velocity, yaw_rate, slip_angle, friction, ... }
> ```

### 환경 동기화: 좌표계 & URDF

두 시뮬레이터를 연결하기 전, 물리적 환경을 일치시키는 작업이 선행된다.

**좌표계 변환 (Basis Transformation)**

$$R_{\text{genesis}} = M \cdot R_{\text{blender}} \cdot M^{-1}$$

| | Genesis | Blender |
|---|---|---|
| 좌표계 | RHS, Forward: +X | RHS, Forward: +Y |
| 주의 | — | RBC Car Addon이 −Y 방향 → 추가 변환 필요 |

데이터 추출 및 URDF 로딩 시 좌표계 변환을 정확히 처리하지 않으면, 이후 모든 단계에서 오차가 누적된다.

**차량 URDF 설계**

Blender 차체를 Genesis에서 구동하기 위해 URDF(Unified Robot Description Format)로 정의한다.

| 컴포넌트 | 설정 내용 |
|---|---|
| Chassis | 실제 질량·관성 텐서 입력 |
| Steering Joint | 조향각 한계 `[-0.35, 0.35] rad`, 강성 `(Kp, Kv)` |
| Wheel Joints | 각속도 제어 가능한 continuous 타입, 마찰 계수 설정 |

| Blender 원본 | 초기 URDF | 최종 URDF |
|---|---|---|
| ![blender_car](../res/0316/blender_car.gif) | ![car_img](../res/car_img.png) | ![car_final](../res/0316/car.png) |

---

## 4. 파이프라인 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    Blender (Reference)                      │
│  임의 경로 설계 → 운동학 기반 주행 → 동역학 State 추출       │
└──────────────────────┬──────────────────────────────────────┘
                       │  Blender CSV
                       │  (pos, vel, kappa, accel, steer_rad, ...)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 1: MPPI (정답 데이터 생성)                │
│  Genesis Engine에서 Blender 궤적을 추종하는                  │
│  최적 (T*, S*)를 병렬 샘플링으로 탐색                        │
└──────────────────────┬──────────────────────────────────────┘
                       │  Golden CSV  (T_golden, S_golden + dynamics)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Stage 2: MLP Supervised Learning                  │
│  [Blender CSV + Golden CSV] → MLP → (T*, S*)                │
│  MPPI의 물리적 정답을 실시간 추론으로 근사                    │
└──────────────────────┬──────────────────────────────────────┘
                       │  Trained MLP (Inverse Dynamics Mapper)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 3: Real-Time Inference                   │
│  Blender 경로 입력 → MLP → Genesis에서 동일 궤적 재현        │
└─────────────────────────────────────────────────────────────┘
```

**공통 Interface: `(Throttle, Steer)`**

두 시뮬레이터를 연결하는 공통 제어 인터페이스로 `(T, S)`를 사용한다.

| 제어량 | 범위 | 물리적 의미 |
|---|---|---|
| Throttle (T) | [-1, 1] | 가속(+) / 속도유지(0) / 감속(-) |
| Steer (S) | [-1, 1] | 좌회전(+) / 직진(0) / 우회전(-) |

> **(T,S)는 두 시뮬레이터 사이의 통로(Interface)일 뿐이다.**  
> 핵심은 Blender의 동역학 궤적 정보를 (T,S)에 올바르게 인코딩하는 것이다.

---

## 5. Stage 1: MPPI — 정답 데이터 생성

### 왜 MPPI인가

Genesis의 물리 Solver(Rigid Solver)는 **Non-differentiable**하다.  
따라서 전통적인 Gradient 기반 최적화(e.g., Backprop through simulation)가 불가능하다.

이를 해결하기 위해 **샘플링 기반 MPC인 MPPI**를 도입한다.  
MPPI는 미분 없이 수백 개의 병렬 시나리오를 동시에 실행하고, 확률적 가중합으로 최적 제어를 도출한다.

> MPPI의 목적: **"Genesis 물리 엔진을 통과한, 수학·물리적으로 feasible한 정답 데이터"** 생성

### 동작 원리 (Receding Horizon, H=10)

```
매 Frame t 마다:

  ① 600개 병렬 환경 복제 (현재 state 복사)
  ② 각 환경에 서로 다른 노이즈 (perturbation) 주입
     → 600가지 서로 다른 (T,S) 시퀀스 생성
  ③ 각 환경을 10 horizon (약 0.4초) 동안 롤아웃
  ④ 각 롤아웃에 대해 cost 계산
  ⑤ exp(-cost/λ) 가중합으로 최적 제어 T_golden, S_golden 도출
  ⑥ t+1로 이동, 이전 golden 근처에서 warm-start 탐색
```

### MPPI Cost Function

$$J = \sum_{h=1}^{H} \left[ w_v |v_h - v_{\text{ref},h}| + w_\kappa |\kappa_h - \kappa_{\text{ref},h}| + w_{\text{cte}} |\text{CTE}_h| + w_{\text{he}} |\Delta\psi_h| + w_a |a_h - a_{\text{ref},h}| + w_{\Delta u} \|u_h - u_{h-1}\| + w_{\text{ff}} \|u_h - u_{\text{ref},h}\| \right]$$

| 항 | 의미 |
|---|---|
| $w_v, w_\kappa$ | 속도·곡률 오차 → Blender 궤적과 물리량을 맞춤 |
| $w_{\text{cte}}, w_{\text{he}}$ | 위치·방향 이탈 억제 → 경로 추종 강성 확보 |
| $w_a$ | 가속도 오차 → 물리적으로 자연스러운 움직임 |
| $w_{\Delta u}, w_{\text{ff}}$ | 제어 smooth화 + 원본 참조 → 급격한 입력 변화 방지 |

**가중치 튜닝 인사이트**

> - `w_dist`, `w_heading`을 높게 설정 → 고속 코너링 시 원심력에 의한 경로 이탈 억제에 결정적
> - `w_vel`이 너무 낮으면 프레임당 이동 거리 차이가 벌어져, **시간 인덱스 기반 Behavior Cloning이 무너짐**
> - MPPI는 공간 인덱스가 아닌 **시간 인덱스 기반**으로 설계해야 한다: 같은 시각에 같은 위치에 있어야 동작 모사가 의미 있다

### 추출된 Golden Data

MPPI를 통해 각 프레임마다 다음이 기록된다:

| 분류 | 주요 컬럼 | 설명 |
|---|---|---|
| Pose | `g_pos_x/y/z`, `g_qw/x/y/z` | Genesis 내 차량 위치·자세 |
| Dynamics (Genesis) | `v_long`, `v_lat`, `yaw_rate`, `accel`, `kappa` | Genesis 물리 엔진이 계산한 동역학 상태 |
| Error | `cte`, `he` | Genesis vs Blender 간 위치·방향 오차 |
| **Label** | **`T_golden`, `S_golden`** | **MPPI가 도출한 최적 제어값 (학습 정답)** |
| Reference | `throttle_raw`, `steer_rad`, `a_target`, `k_target` | Blender 원본 데이터 (MLP의 Feed-Forward 입력) |

> **왜 Blender CSV도 함께 필요한가?**  
> MLP는 _"이 Blender 주행 데이터가 들어왔을 때, Genesis에서는 이런 제어값이 필요하다"_ 는 **전이 관계(Transfer Mapping)** 를 학습해야 한다.  
> Golden Data만으론 Genesis 내부 상태만 알 뿐, Blender→Genesis 간의 차이를 학습할 수 없다.

---

## 6. Stage 2: MLP — Inverse Dynamics Mapper 학습

### 설계 철학

MPPI는 매 프레임마다 600개 환경을 병렬 롤아웃하므로 **실시간 추론에 적합하지 않다**.  
MLP는 MPPI가 생성한 정답 데이터를 학습하여 **동일한 물리적 판단을 실시간으로 근사**한다.

> MPPI = 고비용 정답 생성기  
> MLP = MPPI를 실시간으로 근사하는 Differentiable Surrogate

### 입력 특징 벡터 (25차원)

$$\mathbf{X} = [\underbrace{v_{\text{current}},\ \kappa_{\text{current}}}_{\text{현재 상태 (2D)}},\ \underbrace{\Delta v,\ \text{CTE},\ \text{HE}}_{\text{오차 피드백 (3D)}},\ \underbrace{v_{\text{bl},t+1},\ \kappa_{\text{bl},t+1},\ \ldots,\ v_{\text{bl},t+10},\ \kappa_{\text{bl},t+10}}_{\text{미래 경로 Lookahead (20D)}}]$$

| 그룹 | 피처 | 설계 의도 |
|---|---|---|
| **현재 상태** | `v_current`, `kappa_current` | Genesis 차량의 현재 물리 상태 파악 |
| **오차 피드백 (FB)** | `CTE`, `HE`, `Δv` | 누적 오차를 명시적으로 인지 → 자기 교정(Self-Correction) 가능 |
| **미래 경로 (FF)** | `(v_bl, κ_bl)` × 10스텝 | 앞으로 어떻게 움직여야 하는지 사전 인지 → MPPI의 Receding Horizon 설계와 동일한 철학 |

### MLP Input State Sheet (25D 상세)

| # | 그룹 | 피처명 | 출처 | 설명 |
|---|---|---|---|---|
| 1 | **현재 상태** | `v_current` | Genesis CSV | 현재 Genesis 차량의 종방향 속도 ($v_{long\_gen}$) |
| 2 | | `kappa_current` | Genesis CSV | 현재 Genesis 차량의 주행 곡률 ($\kappa_{gen}$) |
| 3 | **오차 피드백** | `cte` | 계산값 | 횡방향 위치 오차, 부호 포함 (Genesis vs Blender) |
| 4 | | `he` | 계산값 | 헤딩 오차, 부호 포함 (Genesis vs Blender) |
| 5 | | `delta_v` | 계산값 | 속도 오차 ($v_{long\_bl} - v_{long\_gen}$) |
| 6~25 | **미래 경로 (Lookahead)** | `v_long_bl[t+1..t+10]`, `k_bl[t+1..t+10]` | Blender CSV | Blender 경로의 t+1 ~ t+10 스텝 속도·곡률 (20D) |

> **출처가 두 CSV에 걸쳐 있다는 점이 핵심이다.**  
> `v_current`, `kappa_current`, `cte`, `he`, `delta_v`는 **Genesis의 현재 물리 상태**에서,  
> Lookahead 20D는 **Blender의 미래 경로**에서 온다.  
> MLP는 이 두 세계의 정보를 동시에 받아 Genesis 안에서 Blender를 재현하는 제어값을 출력한다.

**인사이트**

> - **절댓값 대신 Delta 사용**: 입력의 스케일 분산을 줄여 학습 안정성 향상
> - **FB 항의 명시적 부여**: 오차를 입력으로 주지 않으면 모델이 궤적 이탈을 스스로 교정할 수 없다. 이는 고전 PID 제어의 오차 항을 신경망으로 구현한 것과 같다
> - **Lookahead의 필요성**: 곡선 구간에서 미리 조향을 준비하지 않으면 고속 주행 시 경로 이탈이 불가피하다

### 네트워크 구조

```
Input (25D)
    │
    ▼
Linear(25→128) → ELU
    │
    ▼
Linear(128→128) → ELU
    │
    ▼
Linear(128→64) → ELU
    │
    ▼
Linear(64→2) → Tanh
    │
    ▼
Output: (T*, S*) ∈ [-1, 1]²
```

> **왜 ELU인가?**  
> CTE, HE, Δv는 **부호가 있는 오차값**이다. ReLU는 음수 입력을 0으로 죽이므로, 음의 오차 정보가 소실된다. ELU는 음수 구간에서도 부드러운 비선형 응답을 유지하여 방향 정보를 보존한다.

### 학습 데이터 구성: Blender CSV + Golden CSV의 결합

```
┌──────────────────────┐     ┌──────────────────────┐
│    Blender CSV       │     │    Golden CSV (MPPI)  │
│  (이상적 주행 궤적)   │     │  (Genesis 최적 제어)  │
│                      │     │                      │
│ · 미래 경로 (v, κ)   │     │ · Genesis 동역학 state│
│ · 목표 속도/곡률     │  +  │ · CTE, HE, Δv        │
│ · 원본 steer/accel   │     │ · T_golden, S_golden │
└──────────────────────┘     └──────────────────────┘
            │                           │
            └──────────┬────────────────┘
                       ▼
              MLP Supervised Learning
                       │
                       ▼
         "Blender 경로로 주행할 때,
          Genesis에서는 이런 제어가 필요하다"
               (Transfer Mapping 학습)
```

> **왜 두 CSV를 반드시 함께 써야 하는가?**
>
> - **Blender CSV만 쓰면**: Blender 세계의 이상적 경로만 알 뿐, Genesis에서 실제로 어떤 오차가 발생하는지 알 수 없다. 경로를 암기할 수는 있지만 물리 차이를 보정할 수 없다.
> - **Golden CSV만 쓰면**: Genesis 안에서의 최적 제어 관계는 알지만, "Blender의 어떤 경로가 들어왔을 때" 라는 맥락이 없다. 새로운 경로에 일반화되지 않는다.
> - **두 CSV를 함께 쓰면**: _"Blender에서 이런 경로가 주어지고, Genesis 차량이 현재 이 상태일 때 → 이 제어값이 최적이다"_ 라는 **두 시뮬레이터 간의 전이 관계(Transfer Mapping)** 를 학습한다.

이것이 본 모델이 특정 경로를 암기하는 것이 아니라, **Blender→Genesis 간의 물리적 인과관계를 일반화**할 수 있는 이유다.

- 8개 서로 다른 주행 경로의 데이터 수집 (~3,000 프레임)
- Data Augmentation: **좌우 반전** (스티어링 대칭) + **가우시안 노이즈 주입** (train set only)
- 노이즈는 학습 시에만 주입 → 다양한 물리 상황 경험, 과적합 억제

### 출력

$$\mathbf{Y} = \begin{bmatrix} T^* \\ S^* \end{bmatrix} = \begin{bmatrix} T_{\text{golden}} \\ S_{\text{golden}} \end{bmatrix}$$

> 이 MLP가 학습하는 것은 특정 경로의 암기가 아니다.  
> **"현재 물리 상태 + 오차 + 앞으로의 경로"가 주어졌을 때, Genesis 물리 법칙 안에서 그것을 최적으로 추종하는 제어값"** 이라는 인과 관계, 즉 **Physics Intuition**이다.

---

## 7. Stage 3: 추론 및 일반화 검증

### 추론 흐름

```
Blender CSV (경로 + 동역학 state)
    │
    ├── 현재 Genesis 상태와 비교 → CTE, HE, Δv 계산
    │
    ▼
MLP Inference  (실시간)
    │
    ▼
(T*, S*)  →  Genesis World  →  Blender와 동일한 궤적 재현
```

### 학습 경로 결과

| Blender (Reference) | MPPI (정답값) | MLP Inference (실시간 추론) |
|---|---|---|
| [![blender](../res/0316/blender.png)](https://github.com/user-attachments/assets/94549c51-5cd4-41d1-a187-f1262d5e1e53) | [![mppi](../res/0222/path_new2.png)](https://github.com/user-attachments/assets/14f37b64-8207-4769-9267-a65f0ed32e82) | [![mlp](../res/0222/curve1.png)](https://github.com/user-attachments/assets/e031fadc-0774-46c6-a9dc-267cfccdd9be) |
* 이미지 클릭 시 영상 재생

MLP는 MPPI의 주행 품질을 실시간으로 재현하였다.


### 일반화 검증: 미학습 경로

학습에 사용하지 않은 새로운 경로를 입력하여 일반화 성능을 검증한다.

* 곡률이 클수록 MPPI 최적화가 어려웠던 걸 고려하여 일반화 성능 테스트 시 어려운 경로 테스트


| 미학습 경로 1 | 미학습 경로 2 |
|---|---|
| [![new1](../res/0222/new1.png)](https://github.com/user-attachments/assets/1897e0ea-6dc8-4ebf-bfb7-7b46bf2e321d) | [![new2](../res/0222/new2.png)](https://github.com/user-attachments/assets/63a83b3d-1214-4fb0-9b9a-8dc2beb2fbee) |
* 이미지 클릭 시 영상 재생



**핵심 관찰**

> 모델이 학습하지 않은 임의의 경로에서도 안정적인 궤적 추종을 보였다.  
> 이는 MLP가 특정 경로를 암기한 것이 아니라, **[상태 오차 → 최적 제어]로 이어지는 물리적 인과관계(Physics Intuition)를 학습했음**을 시사한다.

- 곡률이 높은 경로일수록 MPPI 정답 생성에 컴퓨팅 비용이 급증 → MLP 근사의 실용적 가치가 더욱 부각됨
- 복잡한 미학습 경로에서도 안정적 성능 → 본 Mapper가 특정 트랙이 아닌 **일반적 Inverse Dynamics**를 근사했음을 확인

---

## 8. 결론 및 향후 연구

### 요약

본 연구는 서로 다른 물리 모델을 가진 두 시뮬레이터, Blender(Kinematics)와 Genesis(Dynamics) 사이에서 **동역학적 주행 궤적을 전이**하는 Inverse Dynamics Mapper를 설계·구현하였다.

| 구성 요소 | 역할 | 결과 |
|---|---|---|
| **MPPI** | Genesis 물리 엔진 내 최적 제어 탐색 | 물리적으로 feasible한 정답 데이터 생성 |
| **MLP (25D → 2D)** | MPPI를 실시간으로 근사 | 미학습 경로에서도 안정적 궤적 재현 |
| **Feedback 항** | 오차 자기 교정 | 동역학적 누적 오차 억제 |
| **Lookahead 항** | 미래 경로 사전 인지 | 고속 코너링에서의 조기 제어 가능 |

### (T,S) Control Transfer vs 본 연구 방식

| | **(T,S) Control Transfer** | **본 연구 (Inverse Dynamics Mapping)** |
|---|---|---|
| **입력** | Blender의 `(T,S)` 제어값 | 동역학 상태 + 오차 피드백 + 미래 경로 (25D) |
| **연결 인터페이스** | `(T,S)` | `(T,S)` |
| **출력** | Genesis의 `(T*,S*)` | Genesis의 `(T*, S*)` |
| **학습하는 것** | `제어값` → `제어값` 스케일·변환 관계 | `물리 상태` → `최적 제어`의 인과관계 (Inverse Dynamics) |
| **물리 모델 차이 처리** | `X` 무시 | `O` 동역학 state로 명시적 보정 |
| **일반화** | blender의 정보 손실 | 동역학 정보를 모두 보존 |

> **(T,S)는 두 방식 모두 동일한 출력 인터페이스다.**  
> 차이는 그 (T,S)를 결정하는 **입력 정보**에 있다.  
> (T,S) Control Transfer는 Blender의 제어값 자체가 정보의 전부지만, 본 연구는 **"Genesis가 지금 얼마나 이탈했는지, 앞으로 어떻게 가야 하는지"** 라는 물리적 맥락을 입력으로 삼는다.  
> Blender의 T,S만으로는 이 정보를 담을 수 없기 때문에, 동역학 state가 필수적이다.

---
### 향후 연구
> Real World 의 데이터를 받아서 Real2Sim Calibration 진행 → Sim2Real 의 Policy Transfer 완성
* World Model 의 base model 로 적용

## 참고 자료

- [Genesis Embodied AI](https://genesis-embodied-ai.github.io/)
- [Genesis Performance Benchmarking](https://placid-walkover-0cc.notion.site/genesis-performance-benchmarking)
- [MPPI Troubleshooting](https://github.com/i1uvmango/Genesis_ai_graphicstudy/blob/main/car_test/docs/tech/%5B26-03-15%5D_MPPI_troubleshooting.md)
- [BC Inverse Mapper 결과](https://github.com/i1uvmango/Genesis_ai_graphicstudy/blob/main/car_test/docs/%5B26-03-05%5D_BC_inverse_mappper.md)
- [발표 자료 (PDF)](https://github.com/i1uvmango/Genesis_ai_graphicstudy/blob/main/car_test/docs/Blender2Genesis_Sim2Sim_Calibration.pdf)