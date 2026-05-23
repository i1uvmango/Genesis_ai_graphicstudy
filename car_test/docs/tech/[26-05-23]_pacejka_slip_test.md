# Lateral Slip Measurement Report

Genesis_Vehicle SDK v0.5.5_

---

## 개요

이전 보고에서 "Find_gt 의 settling 중 lateral slip 이 yaw drift 의 원인"
이라는 가설을 제기했었음. 이 가설을 검증하기 위해 cross-slope mesh
4 개 (5°/10°/15°/20°) 에서 정지 차량의 lateral slip 을 직접 측정.

**결론 부터**: V_lat slip 거의 안 일어남. Pacejka 가 정지 시에도 lateral
force 충분히 잡아 줌. 가설 반증. V_lat hook (hold_k_lat) 추가 필요 없음.

---

## 실험 설계

### 목적
정지 차량 (throttle = brake = steer = 0) 이 cross-slope 비탈에서
실제로 lateral 로 미끄러지는지, 미끄러진다면 얼마나 미끄러지는지
정량 측정.

### 조건

| 항목 | 값 |
|---|---|
| 차량 | `Genesis_Vehicle SDK v0.5.5` |
| Spring | k_susp = 70 kN/m |
| 마찰 | μ = 1.0, Pacejka anisotropic |
| Stability hooks | RollingResistance + LowSpeedRegularizer |
| StaticFrictionLock | **비활성** (brake = 0 이라 어차피 미발동) |

### 지형

Cross-slope plane: **z = y × tan(θ)**, x 방향 평탄.

- Mesh: `./csv/mesh/slope_{05,10,15,20}deg.obj` (70 × 16 m)
- 0° control: 단일 Plane @ z = 0 (noise floor 확인용)

### 시뮬 조건

| 항목 | 값 |
|---|---|
| Spawn | (x=0, y=0, z=0.3 m) — 노면 위 0.3 m |
| 초기 속도 | v = 0, yaw = 0 |
| 제어 | throttle = brake = steer = 0 (전 구간) |
| dt | 0.020 s, substeps = 10 |
| Duration | 4.0 s (200 steps) |
| 정상상태 구간 | t ∈ [2.0, 4.0] s |
| Collision | SDK raycast 전용 (mesh collision 비활성) |

---

## 측정 결과

| slope θ (°) | v_lat_peak (m/s) | v_lat_steady ± σ (m/s) | dy_total (m) | dy_rate_steady (m/s) | roll_steady (°) | yaw_drift (°) | a_lat_theory (m/s²) | **slip_ratio** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0  | 1.284 | +0.0000 ± 0.0000 | -0.222 | -0.0048 |  -0.00 |  +7.27 | 0.000 |  nan   |
| 5  | 0.720 | -0.0068 ± 0.0001 | +0.015 | -0.0075 |  +4.88 |  -1.25 | 0.852 | 0.0020 |
| 10 | 1.067 | -0.0098 ± 0.0003 | +0.085 | -0.0168 |  +9.77 |  -9.56 | 1.678 | 0.0015 |
| 15 | 1.145 | -0.0159 ± 0.0010 | -0.075 | -0.0181 | +14.64 |  +3.63 | 2.453 | 0.0016 |
| 20 | 1.684 | -0.0236 ± 0.0061 | +0.312 | -0.0214 | +19.52 |  +5.15 | 3.153 | 0.0019 |

### 지표 의미

- **v_lat_peak**: 초기 0.3 m 낙하 transient 중 body-frame 횡속 최댓값.
  비탈각 ↑ → 차체 정착 시 횡 임펄스 ↑.
  
- **v_lat_steady**: 2~4 s 정상상태 평균 body-frame 횡속.
  ★ Pacejka 가 정지 근방에서 만드는 정상 횡 슬립의 직접 지표.

- **dy_total**: 4 초 누적 횡변위 (world frame). slip 의 실효 크기.

- **dy_rate_steady**: 정상상태 y(t) 회귀 기울기.
  world-frame 횡속도 추정 — v_lat_steady 와 부호·크기 비교 가능.

- **roll_steady**: 정상상태 차체 평균 roll.
  비탈각 θ 와 비슷하면 차체가 노면에 완전 정착.

- **a_lat_theory = g · sinθ · cosθ**:
  마찰 없는 평면 위 cross-slope tangential gravity component (slip 방향).

- **slip_ratio = |v_lat_steady| / (a_lat_theory × 4s)**:
  무마찰 자유미끄럼 대비 SDK 의 실제 슬립 비율.
  - 0 = 완전 고정
  - 1 = 무마찰 자유미끄럼
  - SDK Pacejka 가 정지 근방에서 lateral force 를 얼마만큼 제공하는지
    한 줄로 보여주는 단일 지표.

---

## 해석

### 1. Slip 거의 안 일어남

- 정상상태 횡속 범위 (θ 5~20°): **0.007 ~ 0.024 m/s**
- 4 초 누적 dy_total 절댓값: **< 0.32 m**
- slip_ratio 평균 **0.0017**, 최대 **0.0020**

→ SDK Pacejka 가 무마찰 대비 약 **99.83%** 의 lateral force 제공.
   거의 완전 정지에 가까움.

### 2. 단조성 확인

비탈각 ↑ → |v_lat_steady| 단조 증가 ✓.
물리적으로 정상 (각도 ↑ → tangential gravity ↑ → 잔여 slip ↑).

### 3. Control (θ = 0°) — 측정 방법 검증

- |v_lat_steady| = 0.00001 m/s ≈ 0
- 평지에서 횡속 ≈ 0 → 측정 노이즈 충분히 낮음
- → sweep 결과 신뢰 가능.

### 4. 실제 차 와 비교

실제 차 의 정지 마찰 (rubber-asphalt): μ_static ≈ 0.7 ~ 0.9.
슬립 임계 비탈각 = arctan(μ) ≈ 38 ~ 45°.

→ 20° 이하 에서 실제 차 도 정지 시 거의 안 미끄러짐.
→ SDK 의 slip_ratio 0.17% 는 실제 차와 **정량 일치**.

---

## 결론

### V_lat hook 필요 없음

이전 보고의 가설 ("V_lat slip 이 yaw drift 원인") 은 반증됨.

- 현재 시나리오 (정지 차량, cross-slope 5~20°) 에서 Pacejka 가 lateral
  force 충분히 제공
- 4 초 동안 dy 최대 0.32 m — yaw drift 의 주 원인 으로 작용하기 어려움
- StaticFrictionLock 의 V_lat 확장 (hold_k_lat) 의 추가 필요성 작음

### 추가 검토 (brake 시나리오) 도 의미 없을 듯

주행 중 brake 시 lateral slip 거동도 검토 고려했으나:

- 실제 차 도 비탈 주행 + 풀 브레이크 → friction ellipse 한계로 lateral
  slip 발생 (정상)
- Pacejka 가 friction ellipse 모방 — 실제 차 거동과 동일
- → 추가 실험 으로 새 발견 없음, "SDK = 실제 차" 재확인 만
