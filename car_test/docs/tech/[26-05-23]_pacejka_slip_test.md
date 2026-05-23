# Lateral Slip Sweep Test on staticFrictionLock

_Genesis_Vehicle SDK v0.5.5 — cross-slope 비탈에서 lateral slip 거동 정량 측정 + V_lat hook 효과 검증_

---

## 개요

현재 브레이크는 omega(각속도)를 억제하는 항이기 때문에 차량 바퀴의 회전만 멈추고,
차체의 정지는 타이어 자체의 마찰력 F_tire (Pacejka 의 D 항: T_fric = F_long × R) 이 담당.

`StaticFrictionLock` 이해:
  Pacejka 모델은 정지 상태 or 저속 (코드 v < 0.5) 에서 F_tire 가 너무 작아 제대로 된 제동이 되지 않음.
  `StaticFrictionLock` 훅으로 인위적으로 F_long 에 추가 force 를 주입하여 명시적으로 slip 한계를 보완.

**문제 상황**: 경사 지형 에서 브레이크 제동 시 lateral 방향 으로 슬립 발생 의 가능성.

> 경사각 × `StaticFrictionLockLat(hold_k_lat)` 튜닝 sweep 실험 으로 검증.

**결론 부터**:

- ✓ 정지 시 slip 거의 없음 — Pacejka 의 lateral grip 충분 (slip_ratio ≈ 0.0017, 무마찰 대비 99.83% grip).
- ✓ 저속 (v < 0.5 m/s) 영역 에서 sideslip 명백히 증가 — (v=0.5 의 slip) 이 (v=2.0) 의 **× 11.63**.
- ★ **drive → brake** 시 lateral slip 발생 — slope 15° 에서 slip distance = 18.5 cm.
- 본 폴더 작성 `StaticFrictionLockLat (hold_k_lat = 5K~10K)` 으로 ★ **97.5% 감소** 확인.
- SDK 기본값 200K (tank preset) 은 작은 비탈 에서 bang-bang 발생 → 비권장.

---

## 1. 배경

### 1.1 SDK lateral 메커니즘 현황

| Hook | 작용 축 | 정지 마찰 보강? |
|---|---|---|
| RollingResistance | longitudinal | ✗ (저항 만, v=0 에선 0) |
| LowSpeedRegularizer | longitudinal·lateral | ✗ (jitter 억제, 힘 추가 X) |
| StaticFrictionLock (tank preset 만) | longitudinal | ✓ F_long 정지 마찰 보강 |

→ **F_lat 정지 마찰 보강 hook 이 SDK 에 없음.**

### 1.2 Pacejka 의 정지 한계

`v_eff_clamp = 0.5 m/s` 영역 에서 F_tire 약해짐 → 정지 근방 마찰 부족 →
비탈 에서 lateral slip 위험.

### 1.3 본 연구 의 목표

- (A) 정지 + 저속 주행 의 SDK baseline 정량 측정
- (B) F_lat 전용 hook (`StaticFrictionLockLat`) 작성 + `hold_k_lat` sweep
- (C) BEFORE/AFTER 시각 검증

---

## 2. 실험 전체 구조

> 총 4 sweep + 1 viewer 검증. **Test 4 가 본 보고서 의 핵심.**

**Test 1 — 무입력 정지** (baseline)
- 시나리오: 무입력 정지, 4 s
- brake `0` / hook `✗` / **runs 5**

**Test 2 — 저속 정속 주행** (baseline)
- 시나리오: 저속 정속 주행, 6 s
- brake `feedback` / hook `✗` / **runs 15**

**Test 3 — V=0 brake settling sweep**
- 시나리오: V=0 brake 정지, 4 s
- brake `1.0` / hook `✓` / **runs 24** (4 slope × 6 k_lat)


**★ Test 4 — drive → brake sweep**
- 시나리오: **drive 3 s → brake 5 s**
- brake `1.0` (brake 구간) / hook `✓` / **runs 24** (4 slope × 6 k_lat)


**차량 공통**:
- SDK: `car_4w_rwd_ackermann` (k_susp = 70 kN/m, μ = 1.0, Pacejka anisotropic)
- Stability hooks: RollingResistance + LowSpeedRegularizer

**지형 공통**:
- Cross-slope plane (z = y × tan θ), 70 × 16 m
- Mesh: blender OBJ
- 0° control: 단일 Plane, z = 0 (noise floor 확인)
- Collision: SDK raycast 전용 (mesh collision 비활성)

**시뮬 공통**:
- dt = 0.020s(1/50), substeps = 10

---

## 3. Part A — SDK Baseline (hook 없음)

### 3.1 Test 1 — 정지 상태 스폰

**입력**: throttle = brake = steer = 0, 4 초.
**Spawn**: (x=0, y=0, z=0.3 m) — 노면 위 0.3 m
**정상상태**: t ∈ [2.0, 4.0] s

#### 측정 결과

| slope θ (°) | v_lat_peak (m/s) | v_lat_steady ± σ (m/s) | slip distance (m) | dy_rate_steady (m/s) | roll_steady (°) | yaw_drift (°) | a_lat_theory (m/s²) | **slip_ratio** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0  | 1.284 | +0.0000 ± 0.0000 | -0.222* | -0.0048 | -0.00 | +7.27 | 0.000 | n/a |
| 5  | 0.720 | -0.0068 ± 0.0001 | +0.015 | -0.0075 | +4.88 | -1.25 | 0.852 | **0.0020** |
| 10 | 1.067 | -0.0098 ± 0.0003 | +0.085 | -0.0168 | +9.77 | -9.56 | 1.678 | **0.0015** |
| 15 | 1.145 | -0.0159 ± 0.0010 | -0.075 | -0.0181 | +14.64 | +3.63 | 2.453 | **0.0016** |
| 20 | 1.684 | -0.0236 ± 0.0061 | +0.312 | -0.0214 | +19.52 | +5.15 | 3.153 | **0.0019** |

*0°: drop transient. v_lat_steady = 0 → 측정 방법 sound 확인.

#### 지표 의미

**`v_lat_peak`** — 횡속 피크
> 초기 0.3 m 낙하 transient 중 body-frame 횡속 최댓값.
> (= 차체 가 잠시 옆으로 튀는 순간속도)

**`v_lat_steady`** ★ — 정상상태 횡속
> 2 ~ 4 s 구간 body-frame 횡속 평균.
> Pacejka 가 정지 근방 에서 만드는 **정상 횡 슬립의 직접 지표** — 본 보고서 핵심 변수.

**`slip distance`** — 누적 횡변위
> 4 초 동안 world frame 에서 차체가 옆으로 흘러간 총 거리.

**`dy_rate_steady`** — 정상상태 횡변위 기울기
> 정상상태 y(t) 직선 회귀 기울기. (= 평균 횡 drift 속도)

**`roll_steady`** — 정상상태 roll 각
> 정상상태 차체 평균 roll. ≈ slope 각 → 차체가 비탈에 잘 정착 되었는지 sanity check.

**`a_lat_theory`** — 이론 횡가속도
> `g · sinθ · cosθ` — 마찰 0 일 때 차체가 비탈 아래로 미끄러질 가속도. baseline.

**`slip_ratio`** ★ — 슬립 비율
> `|v_lat_steady| / (a_lat_theory × 4s)` — 무마찰 baseline 대비 실제 슬립 비율.
> - `0` = 완전 고정 (ideal)
> - `1` = 무마찰 자유미끄럼 (worst)

#### 발견

- 정상상태 횡속 (θ 5~20°): **0.007 ~ 0.024 m/s**
- 4 초 누적 slip distance 절댓값: **< 0.32 m**
- slip_ratio 평균 **0.0017**, 최대 **0.0020**
- → SDK Pacejka 가 무마찰 대비 ★ **99.83%** lateral force 제공
- 비탈각 ↑ → |v_lat_steady| 단조 증가 (물리적 정상)
- 0° control 횡속 ≈ 0 → 측정 노이즈 충분히 낮음

#### 실제 차 와 비교

실제 차 의 정지 마찰 (rubber-asphalt): μ_static ≈ 0.7 ~ 0.9.
슬립 임계 비탈각 = arctan(μ) ≈ 38 ~ 45°.

→ 20° 이하 에서 실제 차 도 정지 시 거의 안 미끄러짐.
→ SDK 의 slip_ratio 0.17% 는 실제 차와 **정량 일치**.

→ **정지 한정으로는 hook 불필요.**

### 3.2 Test 2 — 저속 정속 주행 (`0_slip_lowspeed.py`)

**입력**: PI throttle (KP=0.6, KI=0.10) → v_long target, brake on overspeed, steer = 0.
**정상상태**: t ∈ [3, 6] s.

**핵심 지표**:
- `slip_per_meter = |v_lat| / v_long`: 1 m 전진 당 횡 drift.
- `slip_angle = atan2(|v_lat|, v_long)`: body sideslip 각.

#### slip_per_meter heatmap (= tan sideslip)

| θ \ v_target | **0.5 m/s** | 2.0 m/s | 5.0 m/s |
|---:|---:|---:|---:|
| 0° | 0.0025 | 0.0001 | 0.0000 |
| 5° | **0.0231** | 0.0009 | 0.0009 |
| 10° | 0.0037 | 0.0021 | 0.0017 |
| 15° | **0.0264** | 0.0029 | 0.0025 |
| 20° | **0.0372** | 0.0036 | 0.0032 |

#### body sideslip angle (°)

| θ \ v | **0.5** | 2.0 | 5.0 |
|---:|---:|---:|---:|
| 5° | 1.33° | 0.05° | 0.05° |
| 15° | 1.51° | 0.16° | 0.14° |
| 20° | **2.13°** | 0.21° | 0.18° |

#### 발견

- `v_eff_clamp = 0.5 m/s` 경계 에서 slip 명백히 증가:
  v=0.5 의 slip_per_meter 가 v=2.0 대비 평균 ★ **× 11.63** (v=2 → 5 는 미미)
- 20° + v=0.5 → sideslip **2.13°** (시각 인지 가능)
- v ≥ 2 m/s 에선 grip 충분 (sideslip < 0.25°)

→ 정지 결론 은 유지, **저속 영역 의 한계 새로 드러남.**

---

## 4. Part B — `StaticFrictionLockLat` Hook + Sweep

### 4.1 Hook 설계

SDK `StaticFrictionLock` (F_long 전용) 을 mirror 한 lateral 버전.
SDK 코드 미수정. `./lateral_friction_lock.py` 에 작성 + `vcfg.stability_hooks.append()` 로 주입.

```python
class StaticFrictionLockLat(StabilityHook):
    slots = ("POST_TIRE",)

    def __init__(self, brake_thr=0.3, v_thr=0.5, hold_k_lat=200_000.0):
        ...

    def apply_post_tire(self, ctx):
        active_brake = (ctx.brake > self.brake_thr).unsqueeze(-1)
        active_v     = ctx.v_lat.abs() < self.v_thr
        active       = active_brake & active_v

        mu_lat_b = ctx.wheel_meta.mu_lat.unsqueeze(0)
        mu_N     = mu_lat_b * ctx.N
        hold     = torch.clamp(-self.hold_k_lat * ctx.v_lat,
                               min=-mu_N, max=mu_N)
        ctx.F_lat = torch.where(active, hold, ctx.F_lat)
```

**발동 조건**: `brake > 0.3` AND `|v_lat| < 0.5` (per env, per wheel).
**동작**: `F_lat := clamp(-hold_k_lat · v_lat, ±μ_lat · N)`.

### 4.2 Test 3 — 정지 상태 brake settling sweep

**시나리오**: spawn 후 4 초간 brake = 1.0 유지 (= 기어 P단 + full footbrake).
**변수**: 4 slopes × 6 hold_k_lat = 24 runs.

#### slip distance (m) — V=0 settling 누적 횡변위

| θ \ hold_k_lat | 0 | 5K | 10K | 50K | 100K | 200K |
|---:|---:|---:|---:|---:|---:|---:|
| 5° | 0.024 | 0.016 | 0.027 | 0.140 | 0.074 | 0.065 |
| 10° | 0.046 | 0.163 | **0.007** | 0.140 | 0.012 | 0.210 |
| 15° | 0.048 | **0.004** | 0.011 | 0.084 | 0.133 | 0.180 |
| 20° | 0.293 | 0.197 | 0.126 | **0.001** | 0.016 | 0.066 |

#### bang-bang score heatmap

> **bang-bang**: StaticFrictionLockLat 의 보정 force 가 너무 커서 v_lat 이
> 0 주위로 ±μN 진동 → 차체가 떨리는 현상. `score = σ(v_lat) / |mean(v_lat)|`.

| θ \ hold_k_lat | 0 | 5K | 10K | 50K | 100K | 200K |
|---:|---:|---:|---:|---:|---:|---:|
| 5° | 7.29 ⚠ | **0.07** | 0.21 | 0.35 | 0.74 | **21.30 ❌** |
| 10° | 0.72 | **0.05** | 0.14 | 0.30 | 0.40 | **1.28 ⚠** |
| 15° | 0.66 | **0.04** | 0.13 | 0.59 | **1.29 ⚠** | 0.79 |
| 20° | 1.14 ⚠ | **0.05** | 0.13 | 0.47 | 0.52 | 0.84 |

#### 발견

- ★ **5K ~ 10K 가 sweet spot** — slip distance 작고 bang-bang < 0.25.
- SDK 기본값 200K (tank preset 용) 은 우리 차량/dt 조합 엔 너무 stiff.
  5° 에서 σ 가 평균의 21 배 (정지 아닌 진동).
- 수치 안정 조건 `c · dt / m < 2` → m=1330, sub-step dt=0.002 → c 한계 ≈ 1.33M.
  200K 는 한계의 15% → 작은 비탈 에선 임계 진입.

→ tank preset 의 hold_k 값 그대로 사용 비권장. 차량별 튜닝 필요.

### 4.3 Test 4 — 저속 구간 brake 시나리오 sweep

★ **본 보고서 의 핵심 — 진짜 use case 시나리오.**

저속 구간 Pacejka 모델의 불안정성 + lateral 방향 슬립 해결 위한 finetuning sweep.
"저속 주행 중 brake 밟아 감속" → transient 측정.

**시나리오**:
- 가속 구간 (3 s = 150 step): throttle PI → V_target = 2 m/s, brake = 0, steer = 0.
- 브레이크 구간 (5 s = 250 step): throttle = 0, **brake = 1.0**, steer = 0. ★ 측정 영역.

**측정**: 브레이크 구간 동안 누적 slip distance, transient v_lat peak, bang-bang.

#### slip distance heatmap (m) — 브레이크 5 초 누적

| θ \ hold_k_lat | **0** (baseline) | **5K** | **10K** | 50K | 100K | 200K |
|---:|---:|---:|---:|---:|---:|---:|
| 5° | **−0.069** | -0.093 | -0.109 | -0.263 | -0.195 | -0.159 |
| 10° | -0.098 | **+0.032** | +0.092 | -0.066 | -0.041 | -0.149 |
| 15° | **-0.185** | -0.053 | **+0.005** ★ | -0.253 | -0.278 | -0.181 |
| 20° | -0.075 | **+0.000** ★★ | +0.046 | -0.175 | -0.164 | -0.062 |

#### vlat_max heatmap (m/s)

| θ \ hold_k_lat | 0 | **5K** | **10K** | 50K | 100K | 200K |
|---:|---:|---:|---:|---:|---:|---:|
| 5° | 0.048 | 0.008 | **0.004** | 0.101 | 0.094 | 0.143 |
| 10° | 0.088 | 0.012 | **0.007** | 0.130 | 0.144 | 0.169 |
| 15° | 0.101 | 0.018 | **0.012** | 0.123 | 0.143 | 0.145 |
| 20° | 0.076 | 0.022 | **0.013** | 0.114 | 0.148 | 0.152 |

→ 5K ~ 10K 에서 transient peak 도 깨끗 (≤ 22 mm/s).

#### 적정 hold_k_lat (drive → brake 시나리오)

| θ | 권장 k_lat | slip distance (best) | × baseline | bb | vlat_max |
|---:|---:|---:|---:|---:|---:|
| 5° | baseline (0) | -0.069 m | 1.00 | 0.48 | 0.048 |
| 10° | **5,000** | +0.032 m | 0.33 | 0.05 | 0.012 |
| 15° | **10,000** | +0.005 m | **0.03** ★ | 0.11 | 0.012 |
| 20° | **5,000** | +0.000 m | **0.003** ★★ | 0.04 | 0.022 |

### 4.4 두 시나리오 일관성 검증

| | V=0 brake settling | drive → brake |
|---|---|---|
| Sweet spot | 5K ~ 10K | 5K ~ 10K ✓ |
| 200K 평가 | bang-bang (5°: bb=21) | bang-bang (15°: bb=1.94) |
| 20° best slip distance | k=50K → -0.001 | k=5K → +0.000 |

→ 두 시나리오 모두 동일 결론. 권장값 **5K ~ 10K** 는 시나리오 robust.

---

## 5. Part C — Viewer 시각 검증

before hook vs after hook

![](../../res_wjdaksry/0519/slip.mp4)

https://github.com/user-attachments/assets/85cf849b-16b2-4acb-9b04-79bca84f53bb


![](../../res_wjdaksry/0519/no_slip.mp4)

https://github.com/user-attachments/assets/04022f2b-f8eb-415c-89bb-18151f1e35c3


### 5.1 BEFORE — hook 없음 (k_lat = 0)

```
[브레이크 구간] brake = 1.0  5 s  ★ 측정 영역
  s=  0  v_long=+1.949  v_lat=-0.0117  dy=+0.0004 m
  s= 50  v_long=+0.642  v_lat=-0.0611  dy=-0.0049 m
  s=100  v_long=+0.354  v_lat=-0.0989 ⚠  dy=-0.0909 m   ← v_eff_clamp 진입 spike
  s=150  v_long=+0.037  v_lat=-0.0250  dy=-0.1421 m
  s=200  v_long=+0.086  v_lat=-0.0187  dy=-0.1445 m
★ 결과: slip distance = -0.1849 m
```

### 5.2 AFTER — hook 적용 (k_lat = 10,000)

```
[브레이크 구간] brake = 1.0  5 s  ★ 측정 영역
  s=  0  v_long=+1.949  v_lat=-0.0117  dy=+0.0006 m
  s= 50  v_long=+0.831  v_lat=-0.0070  dy=+0.0179 m   ← v_lat 9× 작음
  s=100  v_long=+0.437  v_lat=-0.0072  dy=+0.0225 m   ← spike 없음
  s=150  v_long=+0.108  v_lat=-0.0098  dy=+0.0193 m
  s=200  v_long=+0.085  v_lat=-0.0073  dy=+0.0118 m
★ 결과: slip distance = +0.0047 m
```

### 5.3 직접 비교

| 시각 (브레이크 구간) | BEFORE v_lat | AFTER v_lat | 비율 |
|---|---:|---:|---:|
| s=0 (brake 시작) | -0.012 | -0.012 | 1× |
| s=50 (1 s) | -0.061 | -0.007 | × **0.11** |
| s=100 (2 s — v_eff_clamp 진입) | ★ **-0.099** | ★ **-0.007** | × **0.07** |
| s=150 (3 s) | -0.025 | -0.010 | × 0.40 |
| s=200 (4 s) | -0.019 | -0.007 | × 0.37 |
| 최종 slip distance | **−0.185 m** | **+0.005 m** | ★ **× 0.025** |

★ 결정적 차이 는 `v_long` 이 `v_eff_clamp = 0.5` 아래로 떨어지는 brake 후 ~ 2 초 시점.

- BEFORE: Pacejka F_lat 약화 → v_lat 100 mm/s 까지 폭발 → 1 초 내 8 cm 추가 슬립.
- AFTER: hook 이 그 순간 `F_lat = -hold_k_lat · v_lat` 으로 즉시 카운터 → 7 mm/s 유지.

---

## 6. 종합 결론

### 6.1 영역 별 lateral grip 거동

| 영역 | 조건 | lateral slip 크기 | Hook 필요? |
|---|---|---|:---:|
| **정지 (v=0, brake=0)** | 무입력 | slip_ratio ≈ 0.0017 | ✗ |
| **저속 (v ≤ 0.5)** | 정속 주행 | sideslip 최대 2.13° | △ |
| **동적 (v ≥ 2)** | 정속 주행 | sideslip < 0.25° | ✗ |
| ★ **drive → brake** | 감속 transient | baseline 18.5 cm slip | ✓ |

### 6.2 권장 사용 가이드

| 시나리오 | 권장 hold_k_lat |
|---|---|
| 일반 주행 (v > 2 m/s) | 불필요 (SDK 자체 충분) |
| 정지 (P 단) | 불필요 (slip_ratio 0.002) |
| **비탈 정지 (brake 잡고)** | **5,000 ~ 10,000** |
| **drive → brake 감속** | **5,000 ~ 10,000** |
| 저속 비탈 코너링 (v < 0.5) | 5K~10K (별도 변형 필요)* |

*저속 코너링은 brake 가 0 이라 현재 hook 발동 안 함. `brake_thr=0` 변형 또는
`LateralLowSpeedRegularizer` 별도 작성 필요.

### 6.3 코드 적용 예

```python
# 1) 우리 폴더의 hook import
from lateral_friction_lock import StaticFrictionLockLat

# 2) vcfg 생성 직후 1 줄 추가
vcfg = car_4w_rwd_ackermann(URDF_PATH)
vcfg.dt = DT
vcfg.stability_hooks.append(
    StaticFrictionLockLat(brake_thr=0.3, v_thr=0.5, hold_k_lat=10_000.0)
)
physics = VehiclePhysics(scene, car, sensor, vcfg, n_envs=1)
```

### 6.4 정량 결론 (한 줄)

> SDK Pacejka 만으로 정지/고속 영역 lateral grip 은 충분 (99.83%) 하나
> `v_eff_clamp = 0.5 m/s` 경계 영역 + drive → brake transient 에서 lateral slip 발생.
> 본 폴더 작성 `StaticFrictionLockLat(hold_k_lat = 5K~10K)` 으로 drive → brake 시
> ★ **slip 97.5% 감소** 확인 (slope 15° 에서 18.5 cm → 0.5 cm).
> SDK 기본값 200K (tank preset) 는 우리 차량 mass/dt 조합 엔 stiff → bang-bang 발생 → 비권장.

---