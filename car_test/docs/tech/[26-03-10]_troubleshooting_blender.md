# Genesis–Blender 차량 제어 최적화 및 MLP 학습 보고서

---

# 1. 현황 및 문제점 분석 (Problem Identification)

## 1.1 초기 데이터의 물리적 부정합성

### Blender 데이터의 급가속 문제

원본 Blender 애니메이션 데이터에서  
**0번 프레임 → 1번 프레임 (1 dt ≈ 0.04s)** 사이에

```
속도: 0 → 12 m/s (약 43 km/h)
```

로 급격히 증가하는 현상이 발생함.

---

### 물리적 불가능성

Genesis Engine 내부의 **MPPI(Model Predictive Path Integral) 제어기**가  
해당 속도 변화를 추종하려면 다음 조건이 필요함.

- 매우 큰 토크
- 극단적인 가속도
- 물리 엔진 제한 초과

즉,

```
현실적으로 거의 무한대에 가까운 토크 필요
```

결과적으로 Genesis 물리 엔진의 한계를 초과하여  
**제어 불능 상태(Control Failure)**가 발생함.

---

## 1.2 기존 접근 방식의 한계

### 가중치 강제 조정

기존에는 MPPI의 **cost weight**를 강제로 조정하여  
Blender의 비물리적인 궤적을 최대한 추종하도록 설정.

하지만 이 방식은 다음 문제를 초래함.

- 물리적으로 불가능한 trajectory 학습
- 제어 정책의 일반화 실패
- MLP 모델의 물리 정합성 붕괴
- **MLP 추론 시 blender2genesis mapping 실패**

---

### Inverse Mapping(Blender2Genesis) 추론 실패

#### 단순 경로

곡률이 낮은 경우

```
속도만 충분히 높이면 경로 추종 가능
```

따라서 모델이 어느 정도 추론 가능.

---

#### 고곡률 / 난해한 경로

다음 조건이 동시에 발생할 경우

- 초기 급가속
- 높은 곡률
- 물리 제약

MPPI가 **물리적으로 도달 가능한 해(Solution)**를 찾지 못함.

결과적으로

```
MLP 제어 출력 발산 (Divergence)
```

현상이 발생.

---

# 2. 해결 전략 (Solution Implementation)

## 2.1 Blender 가속 프로파일 최적화

### Blender Auto Drive 활용

Blender 내부 설정을 수정하여 차량이

```
정지 → 점진적 가속 (Ramping)
```

하도록 변경.

이를 통해

- 물리적 연속성 확보
- 비현실적 속도 점프 제거
- Genesis 물리 엔진과의 정합성 확보

---

### 데이터 전처리

기존의

```
0 → 12 m/s jump
```

를 다음과 같은 **부드러운 가속 곡선**으로 대체.

```
0 → 2 → 4 → 6 → 8 → 10 → 12 m/s
```

이를 통해

- MPPI가 안정적으로 동작
- Golden Data 마이닝 가능
- 제어 정책 학습 안정화

---

## 2.2 물리 기반 Inverse Mapping 고도화

단순한 **수치 회귀 문제**로 접근하지 않고,

Genesis 엔진의 **물리 제약 조건**을 고려하여

```
도달 가능한 제어 입력
```

을 MLP가 학습하도록 설계.

고려된 제약 조건 예

- Max Torque
- Tire Friction
- Vehicle Dynamics
- Acceleration Limits

즉,

```
MLP가 물리적으로 가능한 control space만 학습
```

하도록 데이터와 학습 구조를 정렬.

---

# 3. 향후 로드맵 (Future Roadmap)

현재 최적화된 데이터를 기반으로  
MLP 제어 모델의 성능을 **단계적으로 검증**할 예정.

---

## Step 1. Overfitting Test

학습에 사용한 **특정 경로**에 대해

```
완벽한 경로 추종 가능 여부
```

를 검증.

목적

- 모델 학습 용량 확인
- 네트워크 구조 타당성 검증

---

## Step 2. 통합 데이터 학습 및 추론  
(Multi-Trajectory Integration)

다양한 환경을 포함한 데이터셋 구축

- 다양한 곡률
- 다양한 속도 프로파일
- 다양한 trajectory

목표

```
범용적인 Inverse Mapping 능력 확보
```

---

## Step 3. 미학습 경로 추론 (Generalization)

학습 데이터에 포함되지 않은

```
Unseen Path
```

에 대해

- 제어 안정성
- 경로 추종 성능

을 평가.

---

## Step 4. Residual RL 도입 (Fine-tuning)

MLP가 예측한 **기본 제어값(Base Control)** 위에  
**Residual Reinforcement Learning** 정책을 추가.

구조

```
Final Control
= MLP Controller
+ Residual RL Policy
```

목표

- 시뮬레이션 오차 보정
- 모델링 불완전성 보완
- 시스템 강건성(Robustness) 향상

---

# 핵심 인사이트 (Key Insight)

> **"결국 양질의 데이터(물리적으로 가능한 궤적)가 성능의 90%를 결정한다."**

모델 구조보다 중요한 것은

```
물리적으로 가능한 데이터
```

이며,

**데이터의 물리 정합성 확보가  
MLP 기반 차량 제어 모델의 안정성과 일반화 성능을 결정짓는 핵심 요소이다.**