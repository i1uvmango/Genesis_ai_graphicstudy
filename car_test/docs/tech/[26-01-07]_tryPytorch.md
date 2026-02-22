# 5.3 PyTorch 변환 시도 결과

## 1. 목표

PyTorch MLP 출력을 Genesis tensor로 변환할 때 **gradient chain을 유지**하여, Loss → Physics → Control → MLP로 역전파가 가능하도록 함.

## 2. 시도한 방법들

### 2.1 `gs.from_torch()` 사용

Genesis 공식 API `gs.from_torch()`를 사용하여 PyTorch tensor를 Genesis tensor로 변환.

```python
# MLP 출력 (PyTorch tensor)
new_steer = model(state)[0]  # requires_grad=True

# Genesis tensor로 변환
steer_gs = gs.from_torch(new_steer.unsqueeze(0))

# Genesis 차량 제어
car.control_dofs_position(position=steer_gs, ...)
```

**결과:** ❌ 실패
- `steer_gs`는 Genesis tensor가 되었지만, 물리 시뮬레이션 후 `get_qpos()`에서 gradient가 끊어짐

### 2.2 중첩 변환 패턴 문제

초기 코드에서 불필요한 변환이 발생:

```python
# 문제가 있는 패턴 (Genesis → PyTorch → Genesis)
steer_gs = gs.from_torch(steer_cmd_val.unsqueeze(0))
car.control_dofs_position(
    position=gs.from_torch(torch.stack([steer_gs[0], steer_gs[0]]))  # ❌ 재변환!
)
```

**문제점:**
- `steer_gs[0]` 인덱싱 시 PyTorch tensor로 변환
- 다시 `gs.from_torch()`로 Genesis tensor로 변환
- 이 과정에서 **Scene context 단절** 발생 가능

### 2.3 변환 최소화 패턴

개선된 패턴: PyTorch에서 모든 연산 완료 후 한 번만 변환

```python
# 개선된 패턴
steer_cmd = new_steer * 0.8

# PyTorch에서 stack 먼저, Genesis 변환은 한 번만!
steer_cmds_torch = torch.stack([steer_cmd, steer_cmd])
steer_cmds_gs = gs.from_torch(steer_cmds_torch)

car.control_dofs_position(position=steer_cmds_gs, ...)
```

**결과:** ❌ 여전히 실패
- 변환 횟수를 줄여도 `get_qpos().requires_grad = False`

### 2.4 Error State를 Torch 연산으로 계산

MLP 입력인 `e_lat`, `e_head`, `e_speed`도 gradient가 흐르도록 torch 연산으로 변경:

```python
def compute_error_state_torch(g_pos_x, g_pos_y, g_heading, g_speed, reference, ref_idx):
    # Reference 값들 (상수로 취급)
    ref_x = torch.tensor(reference["pos_x"][ref_idx], device=device)
    ref_y = torch.tensor(reference["pos_y"][ref_idx], device=device)
    
    # dx, dy (gradient flows through g_pos)
    dx = g_pos_x - ref_x
    dy = g_pos_y - ref_y
    
    # Lateral Error
    e_lat = dy * torch.cos(ref_yaw) - dx * torch.sin(ref_yaw)
    
    return e_lat, e_head, e_speed  # 모두 torch tensor
```

**결과:** ❌ 여전히 실패
- 입력 `g_pos_x`, `g_pos_y` 자체가 `requires_grad=False`이므로 무의미

## 3. 디버깅 결과

### 3.1 변환 후 requires_grad 확인

```python
steer_gs = gs.from_torch(new_steer.unsqueeze(0))
print(f"steer_gs.requires_grad: {steer_gs.requires_grad}")
# Output: True (입력까지는 gradient 유지됨)

# 하지만 시뮬레이션 후...
qpos = scene.sim.rigid_solver.get_qpos()
print(f"qpos.requires_grad: {qpos.requires_grad}")
# Output: False (시뮬레이션 거치면서 gradient 끊어짐!)
```

### 3.2 `__torch_function__` 프로토콜

Genesis는 `__torch_function__` 프로토콜을 구현하여 PyTorch 연산이 Genesis tensor에서도 gradient를 유지하도록 설계됨.

- `gs.from_torch()`로 생성된 tensor는 `requires_grad=True` 유지
- `torch.stack()`, 인덱싱 등의 연산도 지원
- **하지만** 물리 시뮬레이션 내부에서 gradient chain이 끊어짐

## 4. 원인 분석

### 4.1 Gradient 단절 지점

```
MLP 출력 (requires_grad=True)
    ↓
gs.from_torch() (requires_grad=True)
    ↓
control_dofs_position() (내부 처리)
    ↓
scene.step() (물리 시뮬레이션)
    ↓                        ← 여기서 단절!
get_qpos() (requires_grad=False)
    ↓
Loss 계산 (requires_grad=False)
```

### 4.2 근본 원인

1. **RigidSolver 상태 조회 문제**
   - `get_qpos()`가 내부적으로 Taichi 필드 → PyTorch tensor 변환
   - 이 변환 과정에서 `requires_grad=False`로 생성됨

2. **Scene Context 문제**
   - Genesis tensor의 scene context가 시뮬레이션 과정에서 유지되지 않음
   - 제어 입력과 상태 출력 사이의 연결이 끊어짐

## 5. 결론

PyTorch ↔ Genesis 간 tensor 변환 방법을 다양하게 시도했지만, **물리 시뮬레이션을 통한 gradient flow는 불가능**

**핵심 문제:**
- `gs.from_torch()`는 입력 side에서는 gradient 유지
- 하지만 `get_qpos()` 등 출력 side에서 gradient가 끊어짐
- Genesis RigidSolver 구현 자체의 제한으로 추정

## 6. 관련 코드

- `train_e2e.py`: 변환 패턴 적용된 학습 코드
- `compute_error_state_torch()`: Torch 기반 Error State 계산 함수
