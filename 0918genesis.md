# 0918 면담 요약
### 피드백
```
GPT 결과를 그대로 붙여넣지 말 것: 본인이 이해하지 못한 상태에서 복붙하면 무의미함.

이미지·시각자료 필수: 모델 구조, 시뮬레이션 결과, 샘플 이미지가 반드시 들어가야 함.

LaTeX/Markdown 수식: 깨지는 현상 없이 제대로 렌더링하도록 방법을 찾아야 함.

GIF 활용: 긴 동영상보다 GIF로 짧게 보여주는 것이 효과적.

MLP 구조나 모델 그림: 논문처럼 구조 다이어그램을 직접 넣는 습관 필요.
```

----
## ✅Todo-list
```
1. 솔버가 CPU / GPU로 돌아 가는지 확인
    * GPU 모드 실행 확인 (gs.init(backend=gs.gpu))
    * Viewer 없이 MP4 출력 테스트
    * 로그 찍어서 GPU 사용량 확인
2. 자동차 Genesis 솔버 기반으로 설계 확인
    * 자동차 모델링: 단순(바퀴 4개+질량체) vs 복잡(서스펜션/조향 포함) 구조 설계

3. 0916레포트 그림/gif 활용 +  수식 깨지지 않게 보완
```
----




----

## GPU 확인
1. gs.init() 로그

  ![git_init](/res/git_init.png)

2. nvidia-smi(터미널에서)

  ![nvidia_smi](/res/nvidia_smi.png)

3. Genesis 옵션 확인 : genesis에서 어떤 디바이스 쓰는 지
```
gs.init(backend=gs.gpu, logging_level="debug")
```

  ![git_init](/res/git_init.png)
  
4. 속도&파티클 스케일 감지 : GPU라면 1억개 입자까지 커버 가능 / 1000개 미만 했는데 빨라졌다면 CPU 사용하는 것
 * 결과 : 개수에 따라 느려지는거 확인 ( 1000개, 50000개, 200000개 )
    * 하지만 유의미한 차이는 아님

5. 코드 내부 GPU 체크
```
import gstaichi as ti
print("Current backend:", ti.cfg.arch)
```
  ![inner_gpu_check](/res/inner_gpu.png)  


## 솔버가 느린 이유 가능성
1. dt(step:particle to grid &rarr;grid to particle 계산 주기) : 엄청 많은 step 을 사용함
```
dt=3e-3,
substeps=10,
```

2. grid_density(해상도) 높음
```
grid_density=64,
```

3. 입자수 과다 : 20만개 -> 5만 ~ 10만 줄여보기
```
max_particles=200000,

```  

4. substeps 고정값 10
  * dt(시간간격)를 10번의 내부 미세 step으로 적분 한다는 뜻
  * dt=0.003, substeps=10 → 실제 내부 적분 간격은 0.0003
    * 목적: 수치적 안정성 확보 (특히 MPM, FEM 같은 solver에서 급격한 힘·충돌이 있을 때 안정화).

5. `print(i)` 를 통한 루프 count 출력 
    * 파이썬 인터프리터가 실행하는 순수 CPU 연산
    * GPU에서 MPM 계산을 하더라도, 매 step마다 print가 실행되면 CPU ↔ 콘솔 출력 병목이 생겨서 전체 속도를 떨어뜨릴 수 있습니다.

6. VRAM 메모리 부족 여부


---

### 결과 :
1. dt를 높이기 
    * 3e-3 에서 &rarr; e-3, 2e-3로 change 후 변화보기
    * 더 느려졌음
    * 3번 particle의 개수에 따른 속도 변화 말고는 정상이었음  

2. grid_density(주어진 공간을 나누는 격자: 64 &rarr; 32)
    * 실행이 안됨(너무 프레임이 적음)
    * grid_density(128) 돌리니깐 확실히 느려졌음 -> 성능과 관계 있음    
     ![grid_density=128](/res/griddensity128.gif)  


3. max_particles 낮추기  
    * 성능 차이 없음

 
     ![max_particle1000](/res/sand_wheel1000.gif) 

    * 1000개 입자(조금 빠름)  
  
     ![max_particle50000](/res/sand_wheel50000.gif)

    * 50000ro 입자(느림)  


4. substep=10 낮춰보기(step을 쪼개서 수치 안정성을 가져가는 세부 step)
    * 낮췄더니 해상도에 따른 substep이 너무 낮다고 불안정 메세지가 뜨긴 함
    * 하지만 시뮬레이션 속도에는 딱히 차이 없음

5. 단순 로그 출력이라 의미 없음






---
### 해결 point
1. dt를 올려볼 것(시간 간격 크게) ** 가장 유력
2. 해상도 낮춰보기 (계산량 낮춤)
3. 입자 개수 줄여보기 (줄였을 시 해상도 개선된다면 cpu 사용하고 있는 것 : gpu는 1억개 까지 커버 가능)
4. 미세 step 적분 낮춰보기 (계산량 낮춤)
5. 루프 를 주석처리해서 안찍어보기
6. 3060(실행환경)에서는 문제 없음

## 결론 : GPU 환경에서 실행되고 있는 것

  
## 자동차 Genesis 에서 설계하기
간단한 차체 움직임 설계

#### 설계 포인트
* 차체 : rigid body
* 바퀴 : cylinder * 4 + revolute joint(회전 관절)
* 조향 : 앞바퀴 회전 각도를 `control input`으로 제어
* 구동 : 바퀴에 토크, 속도 제어 입력
* 지면 : plane (rigid)
---
* 사용 솔버 : Rigid Body Solver
    * Genesis의 `rigid` + `joint` 활용
* 장점
    * 빠른 시뮬레이션 (RL/제어 알고리즘 테스트)
* 한계
    * 충돌이나 서스펜션, 차체 변형 테스트 불가

---
충돌 및 역학 계산 포함 심화 모델 설계

#### 설계 포인트
* 차체 : MPM(물질점법, deformable body) 또는 FEM(유한요소법)으로 프레임 구현  
* 혼합 구조 : 강체 프레임 + 연성체(범퍼, 차체 패널)  
* 바퀴 : `Rigid body cylinder` + 탄성 있는 타이어(선택적으로 SPH/Soft body)  
* 충돌 : 고해상도 `grid_density` + `contact solver` 적용  
* 역학 계산 : 관성, 탄성, 충돌 후 반발/변형 모두 `MPM / FEM` solver 처리  
* 환경 : Plane, Ramp, Wall 등 다양한 지형 추가 → 충돌 테스트  

---

#### 사용 솔버
* Rigid Solver + MPM(FEM 가능) 혼합  
  * 바퀴 → Rigid solver
  * 차체 프레임(steel beam 등) → FEM Solver  
  * 외장 패널(변형 가능) → MPM Solver  
* Collision Detection → Genesis의 contact solver 활용  

---

#### 장점
* 실제 충돌 테스트, 변형 분석 가능  

---

#### 한계
* 시뮬레이션 속도 느림 (입자 수/격자 해상도 ↑, dt ↓)  
* 고사양 GPU 필요 (RTX 3090 이상 권장)  
* 제어/RL보다는 물리적 분석 목적에 적합  

-----
## 자동차 만들어보기 -> 실패

  * ![rigid_body](/res/rigid_body.png)
* rigid 솔버로 body를 만든 다음 바퀴를 joint 솔버로 이으려고 했음  
  * joint 를 미지원해서 더 나갈 수가 없었음 
  * 붙인다고 해도 애니메이션일 뿐 


---
## Genesis에서 자동차(조인트 기반) 시뮬레이션 불가 원인 정리
### 1. PyBullet/MuJoCo API 미지원
- PyBullet, MuJoCo에서는 `applyExternalForce()`, `setJointMotorControl2()` 같은 API를 통해
  물체에 **외부 힘**이나 **토크**를 직접 가할 수 있음.
- 그러나 **Genesis (0.3.3 버전)** 에는 `apply_external_force`, `apply_external_torque`
  와 같은 외부 힘 적용 API가 구현되어 있지 않음.
- 따라서 바퀴를 엔진처럼 구동하는 것이 불가능함.

---

### 2. 조인트(Joint) 시스템 미구현
- 자동차를 물리적으로 자연스럽게 구현하려면 차체(Box)와 바퀴(Cylinder)를
  **Hinge Joint(회전축 연결)** 으로 묶어야 함.
- 하지만 Genesis에는 `Scene.add_joint()` 와 같은 **조인트 생성 기능이 제공되지 않음.**
- 결국 차체와 바퀴는 서로 독립적인 Rigid Body일 뿐, 기계적으로 연결할 수 없음.
- Mujoco/Pybullet 등 외부 API도 지원 x

---

### 3. 대체 가능 방법
- 현 시점에서 Genesis에서 자동차를 흉내 내려면:
  - `set_vel()` 같은 속도 강제 세팅 (물리적으로 올바르지 않음)
  - 단순히 Box(차체)만 이동시키는 데모
- 하지만 이는 **굴러가는 바퀴 자동차**와는 거리가 멂.

---

### 결론
- Genesis README에는 나와있지만 실제 구동해보면 미지원/미개발 상태  
  - 조인트 기반 자동차 시뮬레이션을 구현할 수 없음

