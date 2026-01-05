# Walkthrough - Dual MLP Architecture

2개의 독립적인 MLP 모델을 사용하여 차량의 Steering과 Throttle을 각각 제어하는 시스템을 구현했습니다.

![](../res/1231_2mlp구조도.png)

# 기술 보고서: Dual MLP 아키텍처

## 1. 아키텍처 개요
기존 단일 모델 방식과 달리, 제어의 핵심 요소인 **조향(Steering)**과 **가속(Throttle)**을 분리하여 학습합니다.
*   **SteerNet**: 차량의 방향 제어 (좌/우)
*   **ThrottleNet**: 차량의 속도 제어 (가속/감속)
*   **목적**: 서로 다른 물리적 특성을 가진 제어 요소를 분리하여 간섭(Interference)을 방지하고 학습 안정성을 높입니다.

## 2. 데이터 흐름 (Data Flow)

### 입력 데이터 (Input): 14차원
두 모델은 **동일한 14차원 상태 벡터**를 입력으로 받습니다.

| 인덱스 | 피처 | 설명 |
| :--- | :--- | :--- |
| 0~3 | `g_qw`, `g_qx`, `g_qy`, `g_qz` | 차량의 방향 (Quaternion) |
| 4~6 | `g_lin_v{x,y,z}` | 차량의 선형 속도 |
| 7~9 | `g_ang_v{x,y,z}` | 차량의 각속도 |
| 10 | `v_long` | 전진 속도 (Longitudinal Velocity) |
| 11 | `spin_R` | 뒷바퀴 회전 속도 |
| **12** | **`dx_local`** | **목표점까지의 상대 X 거리 (Body Frame)** |
| **13** | **`dy_local`** | **목표점까지의 상대 Y 거리 (Body Frame)** |

> `dx_local`, `dy_local`은 `drive_combined_wp.csv` 생성 시 `add_waypoint.py`를 통해 계산된 값으로, 차량이 트랙을 따라가기 위한 "내비게이션" 역할을 합니다.

### 출력 데이터 (Output)
*   **SteerNet** → `steer` (1차원, -1.0 ~ 1.0)
*   **ThrottleNet** → `throttle_norm` (1차원, -1.0 ~ 1.0)

## 3. 통합 메커니즘 (Integration)

1.  **학습 (Train)**:
    *   두 모델은 하나의 스크립트에서 로드되지만, **Optimizer와 Loss 계산은 완전히 분리**되어 있습니다.
    *   `SteerNet`은 오직 조향 오차에 대해서만, `ThrottleNet`은 속도 오차에 대해서만 역전파(Backprop)를 수행합니다.
    *   **독립적 최적화 (Independent Optimization)**: `Steer`와 `Throttle` 각각의 검증 손실(Validation Loss)이 최소일 때의 가중치를 독립적으로 추적하여, 최종 체크포인트에는 **각각 최고 성능을 낸 시점의 모델들을 조합**하여 저장합니다.

2.  **추론 (Inference)**:
    *   시뮬레이터에서 실시간으로 12차원 센서 데이터를 수집하고, Waypoint CSV에서 목표점을 찾아 14차원 벡터를 실시간 구성합니다.
    *   이 벡터를 두 모델에 **병렬(Parallel)**로 입력하여 각각 조향값과 가속값을 얻습니다.
    *   최종적으로 Genesis 시뮬레이터의 `control_dofs_position`(조향)과 `control_dofs_force`(가속) 함수에 각 값을 전달하여 차량을 제어합니다.

    
## 실행 가이드

### 1. 전용 폴더 및 데이터 준비
- **폴더**: `2mlp/` (모든 스크립트와 데이터는 이 폴더를 기준으로 동작합니다)
- **데이터 전처리**:
    - `drive_combined.csv`에는 waypoint 정보가 없으므로, `add_waypoint.py`를 사용하여 `dx_local`, `dy_local`이 추가된 `drive_combined_wp.csv`를 생성했습니다.
    - `source_file`별로 그룹화하여 불연속 구간의 오차를 방지했습니다.
    - **파일 위치**: `2mlp/drive_combined_wp.csv`에 위치시켜야 합니다.

### 2. 학습 (Train)
```bash
python 2mlp/train_2mlp.py --csv drive_combined_wp.csv --epochs 200
```
- **스크립트**: `2mlp/train_2mlp.py`
- **기능**: `SteerNet`과 `ThrottleNet`을 독립적으로 학습시킵니다.
- **출력**: `2mlp/checkpoint/dual_mlp_best.pth` (두 모델의 가중치가 통합 저장됨)

### 3. 추론 (Inference)
```bash
python 2mlp/inference_2mlp.py --model 2mlp/checkpoint/dual_mlp_best.pth --csv drive_combined_wp.csv
```
- **스크립트**: `2mlp/inference_2mlp.py`
- **기능**: 저장된 모델을 로드하여 시뮬레이션 상에서 차량을 제어합니다.

---
