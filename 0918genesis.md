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

### best : mpm, rigid 솔버를 좀 더 경량화하는 솔버 학습?

## 솔버 자체가 gpu로 안돌아가고 cpu로 돌아가서 그런걸수도 있다 check

particle engine -> 이런 조합으로 material 결정 -> cpu(1000) / gpu(1억개)


자동차: ipg trucker : youtube demotruckmaker 레퍼런스



----

## GPU 확인
1. gs.init() 로그
```
gs.init(backend=gs.gpu)
```

2. nvidia-smi(터미널에서)
```
watch -n 1 nvidia-smi

```
3. Genesis 옵션 확인 : genesis에서 어떤 디바이스 쓰는 지
```
gs.init(backend=gs.gpu, logging_level="debug")
```

4. 속도&파티클 스케일 감지 : GPU라면 1억개 입자까지 커버 가능 / 1000개 미만 했는데 빨라졌다면 CPU 사용하는 것

5. 코드 내부 GPU 체크
```
import gstaichi as ti
print("Current backend:", ti.cfg.arch)
```

### 결과 :

##### 확인하기

---

## 솔버가 느린 이유 가능성
1. dt(시간 간격이 너무 짧음) : 엄청 많은 step 을 사용함
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

### 해결 point
1. dt를 올려볼 것(시간 간격 크게) ** 가장 유력
2. 해상도 낮춰보기 (계산량 낮춤)
3. 입자 개수 줄여보기 (줄였을 시 해상도 개선된다면 cpu 사용하고 있는 것 : gpu는 1억개 까지 커버 가능)
4. 미세 step 적분 낮춰보기 (계산량 낮춤)
5. 루프 를 주석처리해서 안찍어보기
6. 3060(실행환경)에서는 문제 없음



## 자동차 Genesis 에서 설계하기
---
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


