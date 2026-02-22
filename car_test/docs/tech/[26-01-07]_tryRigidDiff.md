# 5.2 Genesis Differentiable Physics 시도 결과

## 1. 목표

Genesis 시뮬레이터의 **미분 가능한 물리(Differentiable Physics)** 기능을 활용하여, 경로 추종 Loss를 직접 역전파하여 MLP 제어기를 학습시키고자 함.

**기대 효과:**
- 물리 엔진을 통해 gradient가 직접 흐름
- End-to-End 학습으로 최적의 제어 정책 학습
- Behavior Cloning보다 효율적인 학습

## 2. 구현 내용

### 2.1 Genesis 미분 모드 활성화

```python
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        substeps=10,
        requires_grad=True,  # 미분 가능한 물리 활성화
    ),
)
```

### 2.2 PyTorch → Genesis Tensor 변환

```python
# MLP 출력을 Genesis tensor로 변환
steer_cmds_torch = torch.stack([new_steer * 0.8, new_steer * 0.8])
steer_cmds_gs = gs.from_torch(steer_cmds_torch)

car.control_dofs_position(
    position=steer_cmds_gs,
    dofs_idx_local=np.array([fl_idx, fr_idx])
)
```

### 2.3 시뮬레이션 결과로 Loss 계산

```python
scene.step()

# 시뮬레이션 결과 가져오기
qpos = scene.sim.rigid_solver.get_qpos()
new_pos_x = qpos[0]
new_pos_y = qpos[1]

# Loss 계산
path_error = (new_pos_x - ref_x) ** 2 + (new_pos_y - ref_y) ** 2
L_path = L_path + path_error
```

## 3. 디버깅 결과

### 3.1 Gradient Tracking 확인

```
[DEBUG] qpos check:
  qpos.requires_grad = False    ← 문제 발생!
  qpos.shape = torch.Size([13])
  qpos.grad_fn = None

[DEBUG] L_path check:
  type(L_path) = <class 'torch.Tensor'>
  L_path.requires_grad = False  ← gradient 끊어짐
  L_path.grad_fn = None
```

### 3.2 추가 시도

| 시도 | 방법 | 결과 |
|:---|:---|:---:|
| `sim_options.requires_grad=True` | Scene 생성 시 미분 모드 활성화 | ❌ 실패 |
| `gs.from_torch()` | PyTorch tensor를 Genesis tensor로 변환 | ❌ 실패 |
| `rigid_solver.get_state()` | RigidSolver 상태 직접 접근 | ❌ 에러 발생 |
| 변환 최소화 | PyTorch에서 모든 연산 후 한 번만 변환 | ❌ 실패 |
| `compute_error_state_torch` | Error state를 torch 연산으로 계산 | ❌ 실패 |

### 3.3 Franka 로봇 예제 테스트

Code Wiki에서 제공한 Franka Panda 로봇 예제도 테스트:

```
[DEBUG] Gradient Check:
  current_qpos.requires_grad: False
  get_qvel() failed: 'RigidSolver' object has no attribute 'get_qvel'
  current_dofs_vel.requires_grad: False
```

**동일한 설정인데도 `requires_grad=False` 반환!**

## 4. 원인 분석


#### rigid solver 의 gradient tracking 미지원
Genesis의 `RigidSolver.get_qpos()`가 **gradient tracking을 지원하지 않음**

```
MLP 출력 → Genesis 제어 → 물리 시뮬레이션 → 차량 위치 → Loss
                                          ❌
                       (qpos.requires_grad = False)
```



#### **RigidEntity vs MPMEntity 차이**
   - Genesis의 `differentiable_push.py` 예제는 **MPMEntity** 사용
   - 우리는 URDF/MJCF 기반 **RigidEntity** 사용
   - RigidEntity는 미분 가능한 상태 조회를 완전히 지원하지 않을 수 있음


## 5. 결론

Genesis의 현재 구현 제약으로 인해 **End-to-End Differentiable Training 방식은 적용 불가**

**대안:**
- 강화학습 (PPO): Reward 기반 policy gradient (미분 불필요)
- Behavior Cloning: CSV 데이터로 supervised learning
- Genesis 버전 업그레이드 또는 MPMEntity 사용

## 6. 관련 코드 파일

- `train_e2e.py`: End-to-End 미분 학습 시도 코드
- `test_diff_physics.py`: Franka 로봇 gradient 테스트 코드
