✅Todo-list
```
1. Genesis 소개.페이지에 나온 material 관련 예제 확인
2. drone 학습에 따른 데이터 확인 -> 0913md 확인
3. 자동차 구동   
```



* 메인 페이지 구성: Rigid(강체)/Elastic(탄성체)/Cloth(옷) + Sand/Water 은 examples 에 있음
  
-----  
  
구성: FEM, MPM, PBD, SF, SPH, Tools, Rigid,  모델s   
* 솔버: 물리 법칙을 계산하는 정교한 계산 툴
* 모델: sand, muscle 등 특성을 가지고 있는 개체의 물리 파라미터를 정의




# Genesis 주요 솔버 정리

## 1. MPM (Material Point Method, 재료점법)
- **개념**: 입자(Particle)와 격자(Grid)를 혼합해 고체·유체·분말을 시뮬레이션하는 방법.
- **원리**
  - 입자: 질량, 속도, 물성 저장
  - 격자: 힘과 속도 계산
  - Particle $\leftrightarrow$ Grid $\leftrightarrow$ Particle 변환 반복
- **특징**
  - 장점: 모래, 눈, 진흙 같은 대규모 변형/파괴 표현 가능, **미분 가능성 지원**
  - 단점: 계산량 많음, 수치적 확산(오차) 발생 가능
- **적용 사례**: 모래흐름, 눈사태, 진흙 충돌

---

## 2. FEM (Finite Element Method, 유한요소법)
- **개념**: 연속체를 작은 요소(Element)로 분할하고 응력–변형률 관계를 풀어 변형 계산.
- **수학적 모델**
  - 응력(모으는 힘)–변형률 관계:  
    $$
    \sigma = \mathbf{C} : \varepsilon
    $$
  - Explicit FEM: 작은 $dt$(시간 변화율) 필요, 빠르지만 불안정  
  - Implicit FEM: 큰 $dt$ 가능, 안정적이지만 계산량 많음
- **특징**
  - 장점: 공학적으로 높은 정확성
  - 단점: 계산량 크고 복잡
- **적용 사례**: 젤리, 탄성체 구조물, 로봇 조작 대상




---

## 3. SPH (Smoothed Particle Hydrodynamics, 입자 유체법)
- **개념**: 유체를 완전히 입자 기반으로 표현하고, 커널 함수로 입자 간 상호작용 계산.
- **수학적 모델**
  - Navier–Stokes 방정식 근사:  
    $$
    \rho \left( \frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} \right) 
    = -\nabla p + \mu \nabla^2 \mathbf{v} + \mathbf{f}
    $$
- **특징**
  - 장점: 자유 표면 표현 우수, 물/기체 거동 직관적
  - 단점: 불안정, 인접 입자 탐색에 비용 큼
- **적용 사례**: 물, 액체 흐름, 기체 시뮬레이션

---

## 4. PBD (Position-Based Dynamics, 위치 기반 동역학)
- **개념**: 힘 적분 대신 제약 조건을 만족하도록 위치를 직접 보정하는 방식.
- **원리**
  1. 질량점의 예측 위치 계산
  2. 거리/체적 보존 같은 제약조건을 통해 위치 수정
  3. 보정된 위치로 업데이트
- **특징**
  - 장점: 매우 안정적, 실시간 시뮬레이션에 적합
  - 단점: 물리적 정확성 낮음 (실제 힘 기반 아님)
- **적용 사례**: 천(Cloth), 캐릭터 피부, 게임용 소프트바디

---

## 5. SF (Shape Matching, 형상 매칭 솔버)
- **개념**: 초기 모양과 현재 모양을 비교하여 가장 가까운 강체 변환(회전, 스케일)을 적용해 보정.
- **원리**
  - 현재 입자 분포 $\rightarrow$ 초기 모양으로 되돌리는 변환 행렬 계산
- **특징**
  - 장점: 매우 빠르고 안정적
  - 단점: 물리적 정확성 낮음 (실제 힘 계산이 아님, visualize 특화, 애니메이션)
- **적용 사례**: 게임 속 젤리, 단순 말랑한 물체

---

## 6. Rigid Body Solver (강체 솔버)
- **개념**: 변형되지 않는 강체의 운동(이동, 회전)을 계산하는 솔버.
- **수학적 모델**
  - 병진 운동: $F = m a$  
  - 회전 운동: $\tau = I \alpha$
- **특징**
  - 장점: 계산이 빠르고 안정적
  - 단점: 내부 변형 불가능
- **적용 사례**: 로봇 링크, 바퀴, 기계 부품, 바닥

---

## 7. Tool Solver (도구/상호작용 솔버)
- **개념**: Genesis에서 제공하는 **도구형 상호작용 솔버**. FEM/MPM/PBD 객체와 로봇 또는 외부 도구 간 상호작용을 지원.
- **특징**
  - 충돌, 접촉, 제약을 통합적으로 처리
  - 학습/제어 환경에서 유용 (예: 로봇 그립, 집기 동작)
  - **미분가능성** 지원 (어떻게 사용해야 효율이 좋은가? - 로봇에게 학습)
- **적용 사례**: 로봇 팔로 물체 잡기, 집기 실험, 조작(manipulation) 태스크

---




## sand_wheel.py
[![sand_wheel 실습](./res/sand_wheel.mp4)](https://github.com/user-attachments/assets/2e9c4321-23b7-43b3-ba9a-fb7aee5f8c02)

### 코드 분석
```
import argparse
import numpy as np
import genesis as gs


def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(seed=0, precision="32", logging_level="debug")




    # ===============================
    # Scene 생성
    # - MPM(Material Point Method) solver 사용
    # - dt=0.003, substeps=10 → 세밀한 시간 스텝
    # - grid_density=64 : 입자-격자 해상도
    # - Viewer/Vis 옵션으로 카메라 및 시각화 설정
    # ===============================
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=3e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -1.0, -0.1),
            upper_bound=(0.57, 1.0, 2.4),
            grid_density=64,
        ),
        show_viewer=args.vis,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.5, 0.0, 1.42),
            camera_lookat=(1.0, 0.0, 1.0),
            camera_fov=30,
            max_FPS=120,
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
            rendered_envs_idx=[0],
        ),
    )



    # ===============================
    # Plane (바닥 강체) 추가
    # - URDF 파일 로드
    # - needs_coup=True : 접촉 계산 활성화
    # - coup_friction=0.2 : 마찰 계수
    # - fixed=True : 고정됨
    # ===============================
    plane = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.2,
        ),
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )

    # ===============================
    # Wheel에 사용할 강체 재질 정의
    # - coup_softness=0.0 : 딱딱한 접촉
    # ===============================
    mat_wheel = gs.materials.Rigid(
        needs_coup=True,
        coup_softness=0.0,
    )

    # ===============================
    # Wheel 강체 4개 추가 (위치/크기 다름)
    # ===============================
    wheel_0 = scene.add_entity(
        material=mat_wheel,
        morph=gs.morphs.URDF(
            file="urdf/wheel/wheel.urdf",
            pos=(0.5, -0.2, 1.6),
            euler=(0, 0, 90),
            scale=0.6,
            convexify=False,
            fixed=True,
        ),
    )

    wheel_0 = scene.add_entity(
        material=mat_wheel,
        morph=gs.morphs.URDF(
            file="urdf/wheel/wheel.urdf",
            pos=(0.5, 0.3, 1.2),
            euler=(0, 0, 90),
            scale=0.6,
            convexify=False,
            fixed=True,
        ),
    )

    wheel_0 = scene.add_entity(
        material=mat_wheel,
        morph=gs.morphs.URDF(
            file="urdf/wheel/wheel.urdf",
            pos=(0.5, -0.3, 0.8),
            euler=(0, 0, 90),
            scale=0.6,
            convexify=False,
            fixed=True,
        ),
    )

    wheel_0 = scene.add_entity(
        material=mat_wheel,
        morph=gs.morphs.URDF(
            file="urdf/wheel/wheel.urdf",
            pos=(0.5, 0.4, 0.4),
            euler=(0, 0, 90),
            scale=0.6,
            convexify=False,
            fixed=True,
        ),
    )

    # ===============================
    # 입자 방출기(Emitter) 추가
    # - MPM.Sand() : 모래 재질
    # - max_particles=200000 : 최대 입자 수
    # - Rough surface : 색상/질감
    # ===============================
    emitter = scene.add_emitter(
        material=gs.materials.MPM.Sand(),
        max_particles=200000,
        surface=gs.surfaces.Rough(
            color=(1.0, 0.9, 0.6, 1.0),
        ),
    )

    # ===============================
    # Scene 빌드
    # - n_envs=5 : 병렬 환경 5개 생성
    # ===============================
    scene.build(n_envs=5)

    # ===============================
    # 메인 시뮬레이션 루프
    # - horizon=1000 스텝 실행
    # - 각 step에서 모래 입자 방출 + scene.step()으로 물리 계산
    # ===============================
    horizon = 1000
    for i in range(horizon):
        print(i)
        emitter.emit(
            pos=np.array([0.5, 0.0, 2.3]),                  # 방출 위치
            direction=np.array([0.0, np.sin(i / 10) * 0.35, -1.0]),  # 방향 (좌우 흔들림 포함)
            speed=8.0,                                      # 발사 속도
            droplet_shape="rectangle",                      # 방출 모양
            droplet_size=[0.03, 0.05],                      # 방울 크기
        )
        scene.step()  # 한 타임스텝 시뮬레이션 계산

# 프로그램 실행 엔트리포인트
if __name__ == "__main__":
    main()

```

요약: 강체 바닥 + 4개의 wheel 날개 + 떨어지는 모래입자
MPM 솔버로 떨어지는 모래를 계산, Rigid 솔버로 강체 계산
- 물리법칙을 실제 계산하는 시뮬레이션

----
## 자동차 구동체 예제 조사

## 자동차 외부 엔진 비교 

1. PyBullet 👉 “빠르고 가볍게 실험해보는 용도”

* 자동차/드론/로봇 예제가 내장되어 있어 바로 실행해볼 수 있음
* 강화학습 환경도 많음
* **Genesis에서 돌릴 수 있는가?**: Bullet은 자체 물리 엔진이라 Genesis 내부와 별개. 다만 Bullet 스타일 URDF 모델은 Genesis에서도 불러올 수 있음 (Genesis 엔진은 아님).
---

2. MuJoCo 👉 “정교한 수학적 모델 + 연구 표준”
* 수치적으로 매우 안정적이고 논문 벤치마크 표준 엔진
* 정확한 역학/접촉 시뮬이 필요하면 MuJoCo가 우세
단, 자동차 예제 같은 건 직접 모델링해야 함
* **Genesis에서 돌릴 수 있는가?**: MuJoCo는 독자적인 시뮬레이터. 다만 MuJoCo 환경(MJCF/XML 모델)은 Genesis로 변환하거나 비슷하게 구성 가능.
-------

## Genesis와 외부 엔진(PyBullet, MuJoCo) 호환 x

## 1. Genesis 엔진 구조
- Genesis는 **GPU 기반 자체 물리 솔버**를 내장한 독립 시뮬레이터.  
- 지원되는 주요 솔버:
  - Rigid, FEM, PBD, MPM, SPH, Tools  
- `gs.init(backend=...)`에서 선택하는 것은 **계산 자원(CPU/GPU)**이지 외부 엔진이 아님.  
  - 예:  
    ```python
    gs.init(backend=gs.gpu)
    gs.init(backend=gs.cpu)
    ```

---

## 2. 외부 엔진(PyBullet, MuJoCo) 직접 사용 여부
- `gs.init(backend=gs.mujoco)` 같은 방식은 **지원되지 않음 ❌**  
- 이유:
  - PyBullet/MuJoCo는 Genesis와 **완전히 별도의 물리 엔진 코드베이스**  
  - 각 엔진은 자체 API(바디, 조인트, 충돌, 파라미터)를 가지고 있으며, Genesis의 `Scene` / `Entity` / `Morph`와 호환되지 않음.  
  - Genesis는 GPU 최적화된 내부 자료구조만 받아들임.  

---
## Genesis 용 직접설계 (솔버 사용)
### 필요 요소
1. 차체 (Rigid body)
  * 자동차 본체 → 강체(rigid body)로 모델링.
  * 질량, 관성 모멘트, 중력 적용.
  * 필요 솔버: Rigid, PBD(스프링,서스펜션 계산)

2. 엔진 / 토크 모델
- 실제 엔진은 회전력을 내고 → 기어비/차동기어를 통해 바퀴에 토크 전달.
- 물리 엔진에서는 **바퀴 joint에 구동 토크(force)**를 주는 방식으로 구현.
* 필요 솔버: Rigid,

3. 바퀴 (Wheel collider or rigid body)
- 원형 강체로 바닥과 접촉.
- 마찰계수(traction), 슬립 계산 필요.
- 현실적인 구현은 “휠 콜라이더(wheel collider)” 또는 joint + 마찰 모델로 처리.
* 필요 솔버: Rigid,

4. 서스펜션 (Suspension)
- 바퀴와 차체 사이를 스프링-댐퍼로 연결.
- 노면 충격 흡수 및 접지력 유지.
* 필요 솔버: PBD 솔버

5. 스티어링 (Steering)
- 앞바퀴 joint 회전각 제어 → 조향 가능.
* 필요 솔버: 

6. 주행환경
- 도로, 빗길, 눈 등 주행 환경
* 필요 솔버 : Rigid Solver(아스팔트) + Joint(차량 다이나믹스) + MPM Solver(노면)

