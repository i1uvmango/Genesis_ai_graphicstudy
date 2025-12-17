# Index

**Code Index**

아래는 주요 코드 파일과 간단한 목적 설명입니다.

| 파일 | 코드 목적 |
|---|---|

| [src/train_bc.py](src/train_bc.py) | Behavior Cloning 정책 학습 스크립트(데이터 전처리·MLP 훈련·체크포인트 저장). |
| [src/test_bc.py](src/test_bc.py) | 학습된 BC 모델을 Genesis에서 실행·시각화하는 추론 런너(URDF 로드, 제어 루프). |
| [src/shared_data_extracter.py](src/shared_data_extracter.py) | 연구실 공용 데이터 추출 유틸리티(Blender/Genesis 데이터 공통 처리). |
| [src/on_off_data_blender_data.py](src/on_off_data_blender_data.py) | Blender 프레임별 데이터 로거(핸들러 등록/제거, CSV 추출, spin/steer 처리). |
| [src/car_test_urdf.py](urdf/car.urdf) | URDF  |


**Docs Index**

| 문서 | 한 줄 요약 |
|---|---|
| [docs/_motivation.md](docs/_motivation.md) | Blender와 Genesis 간 파이프라인과 Generative Physics 목표(현실-가상 일치) 및 데이터·역할 정리. |
| [docs/genesis_car_urdf.md](docs/genesis_car_urdf.md) | 간단한 URDF 자동차 구성과 `car_test.py`를 통한 DOF 추출·Pygame 입력 기반 바퀴 속도 제어 실습 설명. |
| [docs/blender.md](docs/blender.md) | Blender 모델을 Genesis로 옮길 때 발생하는 오브젝트 원점·변환·좌표계·스케일 문제 진단과 단계별 해결 체크리스트. |
| [docs/vehicle_blending.md](docs/vehicle_blending1.md) | Blender 모델을 차체+4휠로 분리해 Genesis로 임포트하는 과정, 좌표 불일치 원인과 서스펜션 적용 스크립트 요약. |
| [docs/blender2.md](docs/blender2.md) | 리그 본 구조(CONTROL/MCH/DEF) 분석과 어떤 본을 조작해야 하는지, 물리 제약 적용 원칙 설명. |
| [docs/blender3.md](docs/blender3.md) | Blender 차량에 질량·서스펜션·휠 회전·브레이크·충돌 제약을 추가한 구현 및 주행 테스트 결과와 후속 계획. |
| [docs/mlp_report.md](docs/mlp_report.md) | CSV 타깃 데이터로 MLP를 이용해 마찰·질량·드라이브/스티어 게인 등 물리 파라미터를 학습하는 시스템 아키텍처와 유한차분 학습 절차. |
| [docs/blendertoGenesis.md](docs/blendertoGenesis.md) | Blender 에셋을 Genesis에 적용할 때의 좌표·조인트·서스펜션 오류 사례와 URDF/dae 사용·보정 전략 및 MLP 보정 계획. |
| [docs/1125_addingMLP.md](docs/1125_addingMLP.md) | MLP를 도입해 Genesis 시뮬레이터의 물리 파라미터를 학습·보정하는 방안 제안. |
| [docs/1126_drift.md](docs/1126_drift.md) | Blender에서 가이드 경로 주행 데이터 수집 중 핸들러 잔존·글로벌 상태 오염 문제와 Kick-start·정규화 등 해결책 정리. |
| [docs/1217_steer_throttle_unify.md](docs/1217_steer_throttle_unify.md) | 앞바퀴 조향·뒷바퀴 스로틀 통일, throttle 노이즈 deadzone 적용 및 Blender↔Genesis 좌표계 재정렬으로 데이터 안정화. |


