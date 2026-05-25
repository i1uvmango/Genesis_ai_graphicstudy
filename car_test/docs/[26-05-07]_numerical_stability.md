
### Raycast Wheel

### 비대칭 suspension/damper 설계

![](../res_wjdaksry/0512/structure.png)

#### 시도 방법1: chassis 중심 속도 반영 서스펜션/댐핑

* 차체의 updown(z축 속도)을 damping에 : vel[2] 방식 (chassis 중심 속도)

문제점

* 4바퀴가 동일한 vel[2] 공유
* pitch 운동 시 앞뒤 구분 불가
* 가속 → pitch → vel[2] 양수 누적 → 발산 (양성 피드백 루프 현상 발생)


#### 시도 방법2: 바퀴별 독립 서스펜션/댐핑 (현재 사용)

![](../res_wjdaksry/0512/per_wheel.png)

> 차체가 서스펜션에 의해 기울여졌을때의 그림

* 타이어 접지점 속도 v_R = w(각속도) * r(거리)
* 의미: 같은 바퀴여도 CoM(중심)까지의 거리에 따라 damper에 가해지는 힘이 달라짐

**핵심 아이디어**: 4바퀴 각각의 raycaster 측정값으로 독립적인 vertical force를 계산 → pitch 운동 시 앞뒤 댐퍼가 반대 방향으로 작용해 평균 알짜력이 0이 되므로 pitch 감쇠 가능.

##### 알고리즘 4단계

**거리 측정 → compression**

```
REST_D         = TIRE_RADIUS + L_SUSP = 0.335 + 0.07 = 0.405 m
d[i]           = raycaster 측정 (각 바퀴, shape [1,4])
compression[i] = clamp(REST_D - d[i], 0, L_SUSP)
```

- 바퀴가 지면에 가까울수록 compression 커짐
- `d > REST_D` → `compression = 0` (접지 없음)
- 4바퀴 모두 독립된 값

**스프링 힘**

```
F_spring[i] = K × compression[i]      (K = 80,000)
```

**비대칭 댐핑**

```
comp_rate[i] = (compression[현재] - compression[이전]) / DT
C_damp[i]    = C_COMP (4,000)  if comp_rate > 0   ← 압축 중 (스프링 수축 빠르게)
               C_EXT  (14,000) otherwise          ← 신장 중 (스프링 팽창 느리게)
```

> 실제 차량도 rebound(신장) 댐핑이 bump(압축)보다 강함 — **차체가 튀어오르는 걸 억제**하기 위해.



##### 통합 코드 (벡터화, shape [1, 4])

```python
compression = (REST_D - dists_t).clamp(0, L_SUSP)
comp_rate   = (compression - prev_comp_t) / DT
C_damp      = where(comp_rate > 0, C_COMP, C_EXT)
F_vert      = K * compression + C_damp * comp_rate
```

각 바퀴마다 측정 / 변화율 / 댐퍼 부호 / force가 모두 독립.

##### Pitch 운동 시 동작

| 위치 | v_hit_z | 적용 댐퍼 |
|------|---------|----------|
| 앞바퀴 | > 0 (올라옴) | C_EXT |
| 뒷바퀴 | < 0 (내려감) | C_COMP |

> 목표: 사이클 평균 알짜력 ≈ 0 이 목표지만 비대칭 이기 때문에 편향이 존재하므로 K(스프링)의 진동수 와의 관계가 중요


## Troubleshooting : K 진동수, Zero Order Hold instability
### K 값과 dt의 진동 주기 

K(스프링 상수)와 dt(타임스텝)와의 관계성

https://github.com/user-attachments/assets/3ce2396e-03f8-4a94-802d-c866eba51c28

![[../res_wjdaksry/0512/aliasing.mp4]]

### Outer-loop ZOH (Zero-Order Hold) Instability

#### 1. 외부 힘(Force) 인가 주기가 너무 느림 (24Hz의 한계)

*  DT = 0.04167 (약 24Hz)
   * 물리엔진의 **물리량 계산 주기**: (24hz * 10substep) -> 0.004167 (**240hz**)
   * suspension/damper **제어 주기**: 그런데 힘을 계산하는 python loop는 0.04167(**24hz**)

* 문제: 서스펜션 힘($F_{vert}$)이 한 번 계산되면 다음 0.0416초 동안 값이 고정
* 현상: 물리 엔진 내부에서는 10번의 substep이 도는 동안 지면 거리가 계속 변하는데, 외부에서 가해주는 힘은 "과거의 거리"를 기준으로 고정 &rarr; 에너지가 감쇄되지 않고 증폭되어 차체가 튀어 오름


#### 2. comp_rate 계산의 lag

 댐핑(Damping)은 속도에 비례하는 저항력입니다. 
 
 $$comp\_rate = \frac{compression_{now} - compression_{prev}}{DT}$$

  * 오류: 여기서 DT는 0.04167입니다. 
  * 매우 긴 시간 동안의 "평균 속도"를 댐핑에 사용하고 있습니다.
  * 결과: 바퀴가 미세하게 떨릴 때 즉각적인 저항(댐핑)이 생겨야 하는데, 1 dt 전의 상태를 참조하여 오차 발생


#### 3. 비대칭 댐퍼의 불연속성 (Discontinuity)

* 문제: comp_rate가 $0$을 지나는 순간 댐핑 계수가 4,000에서 14,000으로 급격히(Step) 점프
* 현상: 불연속적인 값의 변화는 차량에 충격을 줌 + 낮은 dt &rarr; 진동 증폭



#### 4. Spring-dt Aliasing으로 인한 오차 누적 현상

suspension/damper 의 진동 현상 해결 중 substep을 낮춤에 따라 오차가 줄어드는 현상을 발견

* dt와 K의 aliasing으로 인한 오차가 substep을 잘게 나누면서 더 증폭 된 현상

https://github.com/user-attachments/assets/8b2e6729-9586-45b9-9d59-0cb6c664964a

substep 10

![[../res_wjdaksry/0512/substep10.mp4]]


https://github.com/user-attachments/assets/1f219cf2-632f-4531-86dc-afd919cafacd

substep 50 

![[../res_wjdaksry/0512/substep50.mp4]]

$$F_{\text{spring}} = K \cdot x$$


**K**: 얼마나 강한 스프링인지를 나타내는 비율 상수 (스프링 강성, N/m)
**dt**: 시간간격

* 눌린 길이 만큼 힘의 압축

> dt 랑 무슨 상관인가?

* K가 클수록 스프링이 빨리 진동하는데, **dt**는 그대로니까 한 주기당 샘플 수가 줄어들어 suspension/damper 특성이 이상해짐


### K 스프링 상수 값 평가
![](../res_wjdaksry/0512/K.png)

* 초록: 40k
* 노랑: 80k
* 빨강: 100k


Nyquist 샘플링 — 시뮬레이션 한계수치

시뮬레이션이 안정하려면:

| 조건식 | 의미 |
|---|---|
| `DT < T / 2` | 최소 조건 (Nyquist) |
| `DT < T / 10` | 정확도 확보 |
| `DT < T / 50` | 안전한 안정성 |

**정량 지표 (DT=24Hz 기준)**

| K 값 | ω·DT/2 | 위상 지연 | cos(지연) | 댐핑 손실 | 정적 sag |
|------|--------|----------|-----------|----------|---------|
| **40,000** | 0.365 rad | 21° | 0.934 | **6.6%** | 82 mm |
| **80,000** | 0.516 rad | 30° | 0.866 | **13.4%** | 41 mm |
| **100,000** | 0.577 rad | 33° | 0.838 | **16.2%** | 33 mm |

* 이 정도면 셋 다 모두 안전 지대이지만, 1 step 늦은 댐핑제어 + 낮은 dt(24hz)로 인해 오차가 증폭되어 진동 한 것으로 예상됨

**정성 평가**

| K 값 | 성격 | 장점 | 단점 |
|------|------|------|------|
| **40,000** | 부드러움 / 안전 | 24Hz에서 안정, 감쇠비 0.55 → 부드러운 댐핑 | 진동수 낮아 bump 회복 느림, nose-dive/squat(앞뒤 차체 기울기) 큼, 헤드룸 부족(chassis, tire 사이 공간) |
| **80,000** | 균형형 / 경계 | 일반 sedan 수준의 stiffness(reaction) | 24Hz 경계선 — ZOH 발진 위험 (비선형 결합 시 발산), 기존 코드 발진 원인 |
| **100,000** | 단단 / 위험 | 헤드룸 충분, 빠른 응답  | (샘플/주기) — 임계점(90°) 근접, bump 충격, MPPI 학습 어려움 |




**해결법**
* dt 를 더 작게 하여 sampling 주기를 늘림
* K값을 낮춰 진동수 자체를 낮춤

현재 계산의 안정성을 위해 dt = 0.04167 &rarr; 0.02 사용하고, K값은 80000을 사용(곡률 큰 주행을 위해 k 증가)


## 주행 안정성 검증 결과

### 일반 평지 주행

https://github.com/user-attachments/assets/87e31e28-3406-4264-a0f2-0517f9b58014

![[../res_wjdaksry/0512/plane.mp4]]

* 진동 없이 주행 되는 것을 검증


### mesh 위 주행

* mesh 위 주행까지 안정적으로 진행 되는 것을 확인

https://github.com/user-attachments/assets/8f082d18-9b44-40ac-9f41-67ab1bbe2e99

[](../res_wjdaksry/0512/terrain_hud.mp4)  
[](../res_wjdaksry/0512/terrain.mp4)  

### Next Step
* path to ST mapper 구조 변경 및 재학습 


## 재검토 — 진동 source는 제거되지 않았다

> 위 "검증 결과"는 평지와 완만한 mesh에 한정된 것이다. 이후 비탈 시나리오에서 진동이 재발했고, 재검토 결과 — 앞의 dt/K 조정은 제거가 아니라 **완화** 였다. (상세 분석은 ray3)

Troubleshooting에서 진동 원인을 4가지로 진단했지만, 실제 적용한 fix는 `dt` 0.04167 &rarr; 0.02 와 `K` = 80000 두 가지뿐이다. 이 조정으로 4가지 원인 중 실제 해결된 것은 일부에 불과하다.

| 원인 | dt/K 조정으로 해결? | 이유 |
|------|---------------------|------|
| #1 Outer-loop ZOH (24Hz force 인가 vs 240Hz 물리) | 증상만 완화 | hold 오차 크기는 줄었지만, force를 한 스텝 고정하는 hold 구조 자체는 그대로 |
| #2 compression_rate (DT 기반 댐핑 속도) | 미해결 | dt/K 튜닝과 무관한 문제 — 손대지 않음 |
| #3 비대칭 댐퍼 불연속성 (comp_rate=0에서 4,000 ↔ 14,000 step 점프) | 미해결 | dt/K 튜닝과 무관한 문제 — 손대지 않음 |
| #4 Spring-dt aliasing | 해결 | 샘플링 주기를 늘려 직접 해소 |

&rarr; 직접 해결된 것은 #4 하나뿐이며, #1은 증상만 완화됐고 #2·#3은 손대지 않았다. 그러나 이 4가지는 모두 numerical instability 계열이고, 비탈 시나리오에서 드러난 진동의 진짜 원인은 따로 있다.

진동의 실제 원인은 다음으로 추정된다:

* **평지** — 4바퀴의 compression force가 앞뒤·좌우 대칭으로 들어와 안정적으로 주행된다.
* **비탈** — 중력에 의해 앞바퀴와 뒷바퀴의 compression force가 비대칭으로 적용되는 순간 진동이 시작된다.
* **저속 + Pacejka** — 저속 구간에서 Pacejka 모델의 불안정성·예민함이 겹치면서 오차가 증폭된다.

&rarr; 비탈 진입 &rarr; 앞·뒤 compression force 비대칭 &rarr; 진동 발생 &rarr; 저속 Pacejka 예민성과 겹쳐 오차 증폭 &rarr; 진동 확대 &rarr; (앞 단계로 되먹임) **진동 악순환**

> 정리: 위 검증은 "fix가 동작함"을 보인 것이 아니라 "평지의 대칭 조건에서는 이 악순환이 시작되지 않음"을 보인 것이다. 진동 원인의 상세 분석은 ray3의 과제로 이어진다.


## 부록: 구현 파라미터

### 비대칭 댐퍼(현재구현)
    # --- 비대칭 댐퍼 ---
    C_COMP    = 14_000.0    # bump 댐핑 (N·s/m) — 압축 시 강하게 감쇠
    C_EXT     = 4_000.0   # rebound 댐핑 (N·s/m) — 신장 시 부드럽게 감쇠
    L_SUSP    = 0.07 # 최대 stroke (m) — compression 상한 클램프

* compression을 강하게 감쇠, recovery를 빠르게 하여 진동을 최소화한다

### 휠 동역학
    # --- 휠 동역학 ---
    I_WHEEL   = 2.0         # 바퀴 회전 관성 모멘트 (kg·m²)
    T_MAX     = 1500.0      # 최대 구동/제동 토크 (N·m) — throttle/brake 스케일 기준
    OMEGA_MAX = 80.0        # 바퀴 최대 각속도 클램프 (rad/s) ≈ 250 km/h @ r=0.358m
    OMEGA_EPS = 1.0         # T_brake tanh 연속화 — |ω| < OMEGA_EPS 구간에서 선형 근사



    # --- 내부 텐서 (shape 고정) ---
    WHEEL_OFF_T  = torch.tensor(WHEEL_OFFSETS, device='cuda')           # (4,3) 바퀴 오프셋
    T_DRIVE_MASK = torch.tensor([[0., 0., 1., 1.]], device='cuda')      # 구동륜 마스크 (후륜만)
