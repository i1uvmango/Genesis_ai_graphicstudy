# Genesis AI Research History

## Documentation Index

### 최종 목표 정리

#### [_motivation.md](docs/_motivation.md)
Blender와 Genesis 간 파이프라인과 Generative Physics 목표 (현실-가상 일치) 및 데이터·역할 정리


### 2024년 10월
---
#### [[24-10-14] genesis_car_urdf.md](docs/1014_genesis_car_urdf.md)
Genesis 환경에서 간단한 URDF 자동차 구성 및 DOF 추출, Pygame 입력 기반 바퀴 속도 제어 실습

#### [[24-10-15] vehicle_blending.md](docs/1015_vehicle_blending.md)
Blender 모델을 차체+4휠로 분리하여 Genesis로 임포트하는 과정, 좌표 불일치 원인과 서스펜션 적용 스크립트 정리

#### [[24-10-16] blender1.md](docs/1016_blender1.md)
Blender → Genesis(URDF) 분리/정렬 이슈 기록, 오브젝트 원점·변환·좌표계·스케일 문제 진단과 단계별 해결 체크리스트

#### [[24-10-21] blender2.md](docs/1021_blender2.md)
Blender 리그 본 구조(CONTROL/MCH/DEF) 분석, 어떤 본을 조작해야 하는지 및 물리 제약 적용 원칙 설명

#### [[24-10-23] blender3.md](docs/1023_blender3.md)
Blender 차량에 질량·서스펜션·휠 회전·브레이크·충돌 제약 추가 구현 및 주행 테스트 결과

---

### 2024년 11월
---
#### [[24-11-03] blendertoGenesis.md](docs/1103_blendertoGenesis.md)
Blender 에셋을 Genesis에 적용할 때의 좌표·조인트·서스펜션 오류 사례와 URDF/dae 사용·보정 전략 및 MLP 보정 계획

#### [[24-11-13] mlp_report.md](docs/1113_mlp_report.md)
CSV 타깃 데이터로 MLP를 이용해 마찰·질량·드라이브/스티어 게인 등 물리 파라미터를 학습하는 시스템 아키텍처와 유한차분 학습 절차

#### [[24-11-25] addingMLP.md](docs/1125_addingMLP.md)
MLP를 도입해 Genesis 시뮬레이터의 물리 파라미터를 학습·보정하는 방안 제안, 쿼터니언 기반 6DoF 상태 추출 및 Behavior Cloning 학습

#### [[24-11-26] drift.md](docs/1126_drift.md)
Blender에서 가이드 경로 주행 데이터 수집 중 핸들러 잔존·글로벌 상태 오염 문제와 Kick-start·정규화 등 해결책 정리

---

### 2024년 12월
---
#### [[24-12-17] steer_throttle_unify.md](docs/1217_steer_throttle_unify.md)
앞바퀴 조향·뒷바퀴 스로틀 통일, throttle 노이즈 deadzone 적용 및 Blender↔Genesis 좌표계 재정렬으로 데이터 안정화
#### [[24-12-28] MLP 개선 및 토크/스루틀 정의](docs/1228_mlp_change.md) – 데이터 설명, 전처리 개선 내용 & MLP 구조 개선

#### [[25-12-30] 1230_waypoint_and_dual_mlp.md](car_test/docs/25_1230_waypoint_and_dual_mlp.md)
- Waypoint 기반 학습 및 Dual MLP 구조 소개, 데이터 통합 및 성능 분석.

#### [[26-01-06] 0106_waypoint.md](car_test/docs/26_0106_waypoint.md)
- Waypoint 방법론 적용, GenesisAI의 Rigid Solver differential 미지원, pure-pursuit 알고리즘 실험 결과, 경로 추적 개선 내용.

#### [[26-01-12] ppo_residualRL.md](car_test/docs/[26-01-12]_ppo_residualRL.md)
PPO(Proximal Policy Optimization) 기반 Residual RL 설계. Actor-Critic 구조, 4D SteerCorrectionMLP·ThrottleMLP 입력 변수 정리, 커리큘럼 학습(CurriculumScheduler), 보상 함수(R_align·R_recover·R_proj·R_arc·R_forward) 및 페널티 항목(P_steer·P_rate·P_speed·P_stuck·P_lat) 상세 정의. 에피소드 종료 조건 설계.

#### [[26-01-19] supervised_learning.md](car_test/docs/[26-01-19]_supervised_learning.md)
UKMAC(Unicycle Kinematic Model for Acceleration and Curvature) 기반 지도 학습으로 Blender↔Genesis Sim2Sim Calibration 시도. 4단계 파이프라인: Stage1(Env Sync MLP) → Stage2(Dynamics Sync MLP) → Stage3(Ground Truth 최적화) → Stage4(Inverse Dynamics MLP). Open/Closed Loop 비교, 수정된 GT Objective Function(Pure Pursuit 기반 위치 페널티) 설계.

#### [[26-01-19] supervised_learning2.md](car_test/docs/[26-01-19]_supervised_learning2.md)
Stage1·2의 성능이 pure Genesis 환경보다 낮음을 확인하여 사용 중단 결정. 데이터 부족 및 과적합을 원인으로 예측. `scipy.optimize.minimize` 활용 GT 탐색으로 전환.

#### [[26-01-26] troubleshooting.md](car_test/docs/[26-01-26]_troubleshooting.md)
지도 학습 실패 원인 분석: ① 학습 데이터의 과도한 이상성(CTE≈0 → 복원력 미학습) → 데이터 증강·노이즈 주입으로 해결, ② 제어 지연·진동(Latency·Oscillation) → Look-ahead 예견 제어·Damping 강화, ③ 목적 함수 가중치 불균형, ④ Stage 1·2 도메인 분리 전략, ⑤ 국소 최적해 문제(Warm Start·Bound 제약으로 대응).

---

### 2026년 2월
---

#### [[26-02-08] pid.md](car_test/docs/[26-02-08]_pid.md)
PID 피드백 제어를 활용해 Blender (a,k)를 Genesis에서 재현하는 Ground Truth 추출. PID→FeedForward(10프레임 look-ahead)로 예측 제어 전환. MLP 학습 시 CTE·HE 노이즈 증강(robustness)으로 복구 능력 확보. 오버피팅 경로 및 처음 보는 경로 주행 테스트 결과 포함.

#### [[26-02-16] mpc2mppi.md](car_test/docs/[26-02-16]_mpc2mppi.md)
MPC 슬라이딩 윈도우 최적화 시도 및 한계 분석: 로컬 속도→글로벌 속도 추출로 슬립 반영 개선, URDF steering effort·max torque 조정, fine-tuning 파라미터 표 정리. 병렬 구간 최적화의 오차 누적 문제 발견 후 MPPI로 전환 결정.

#### [[26-02-22] MPPI.md](car_test/docs/[26-02-22]_MPPI.md)
GPU 기반 MPPI(Model Predictive Path Integral) 도입. 600개 병렬 환경·10 horizon으로 최적 제어 입력 샘플링. Stage1(bicycle model, 10cm 이내 오차) → Stage2(dynamics model, grid-search 96조합 파라미터 튜닝). Best 파라미터 도출 및 golden T,S 주행 결과 검증.

---

### 2026년 3월
---

#### [[26-03-05] BC_inverse_mapper.md](car_test/docs/[26-03-05]_BC_inverse_mappper.md)
MPPI golden T,S를 정답으로 삼아 Behavior Cloning MLP 학습(Inverse Dynamics Mapping). 25D 입력(현재 state 2D + Feedback 3D + Lookahead 20D), 4-layer MLP (ELU + Tanh 출력). 단일 경로 과적합 → 6개 경로 통합 데이터 + 좌우 반전 증강 → 일반화 성능 달성. 학습하지 않은 경로에서도 안정적 추론 확인.

#### [[26-03-15] blender2genesis.md](car_test/docs/[26-03-15]_blender2genesis.md)
Blender2Genesis Sim2Sim Calibration 전체 파이프라인 정리 (최종 보고). Sim2Real 최종 목표 및 Real2Sim 중간 단계 개념 정의. Blender(Bullet·Kinematics) vs Genesis(자체 엔진·Dynamics) 비교, 좌표계 변환·URDF 설계, Stage1(MPPI 정답 데이터 생성) → Stage2(MLP 지도 학습, 25D Input) → Stage3(실시간 추론) 전 과정 문서화. 미학습 경로 일반화 성능 검증.

---



## Project Progress

### Monthly Progress
- **10월**: Genesis URDF 차량 모델 구축 및 Blender → Genesis 임포트 파이프라인 구축
- **11월**: MLP 기반 물리 파라미터 학습 시스템 개발 및 Behavior Cloning 적용
- **12월**: 데이터 전처리 최적화 및 좌표계 정렬 이슈 해결
- **1월**: PPO Residual RL 설계, UKMAC 기반 지도 학습 파이프라인(Stage 1~4) 시도 및 실패 원인 분석
- **2월**: PID→FeedForward GT 추출, MPC→MPPI 전환, GPU 기반 MPPI 파라미터 튜닝 및 golden T,S 확보
- **3월**: Behavior Cloning Inverse Dynamics MLP 구현, 6개 경로 통합 학습으로 일반화 달성, Sim2Sim Calibration 파이프라인 완성

---

*Last Updated: 2026-03-27*
