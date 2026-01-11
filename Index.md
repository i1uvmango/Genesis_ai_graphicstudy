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

---




## Project Progress

### Monthly Progress
- **10월**: Genesis URDF 차량 모델 구축 및 Blender → Genesis 임포트 파이프라인 구축
- **11월**: MLP 기반 물리 파라미터 학습 시스템 개발 및 Behavior Cloning 적용
- **12월**: 데이터 전처리 최적화 및 좌표계 정렬 이슈 해결 
- **1월**: Waypoint 기반 학습 및 Dual MLP 구조 적용, 통합 데이터 활용 및 강화학습 적용

---

*Last Updated: 2026-01-12*
