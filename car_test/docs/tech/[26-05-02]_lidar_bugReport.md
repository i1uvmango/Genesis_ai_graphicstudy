# Genesis Lidar / Ray-Wheel 구현 중 발견된 버그

Ray-Wheel 방식([26-05-02]_ray_bvh.md)을 구현하면서 Genesis 내부에서 발견한 버그와 우회 방법 정리.

---

## 버그 1: Genesis Lidar 다중 센서 캐시 오프셋 버그 (`raycaster.py` `sensor_cache_offsets`)

### 개요

Genesis Lidar 센서를 3개 이상 사용하면 인덱스 2번부터 항상 `distance = 0.0`을 반환하는 버그.

- **영향 버전**: genesis-world 0.2.1 (확인)
- **패치 대상 파일**: `genesis/engine/sensors/raycaster.py`

### 한 줄 요약

```
증상: Lidar 4개 동시 사용 시 3, 4번째 센서가 항상 0.0 반환

원인: genesis/engine/sensors/raycaster.py
      sensor_cache_offsets 누적합 계산 오류
      올바른 값: [0, 4, 8, 12, 16]
      실제 값:   [0, 4, 4,  4,  4]  ← 누적 아님

해결: raycaster.py 직접 패치
```

### 수정한 파일 (오버라이딩 여부)

Yes — 설치된 패키지 파일을 직접 패치했다.

| 파일 | 상태 |
|------|------|
| `~/.../genesis_env/lib/python3.10/site-packages/genesis/engine/sensors/raycaster.py` | **실제로 실행되는 코드. 여기를 패치함.** |
| `~/Genesis/genesis/engine/sensors/raycaster.py` | 소스 레포 (실행에 무관). 동일하게 수정했으나 효과 없음. |

> 주의: `genesis_env/` 가상환경이 소스 레포와 별개로 설치되어 있어, 소스를 수정해도 실행 코드에 반영되지 않는다.

### 버그 위치

```
raycaster.py, 약 line 393–396 (add_sensor() 메서드 내)
```

### 문제 코드 (BUGGY)

```python
# output_hits를 [N_total_rays] 크기로 확장한 뒤
self._shared_metadata.output_hits = ...

# sensor_cache_offsets에 오프셋을 추가
self._shared_metadata.sensor_cache_offsets = concat_with_tensor(
    self._shared_metadata.sensor_cache_offsets,
    self._cache_size          # ← BUG: 상수값 4를 그대로 append
)
```

### 실제 결과

센서를 4개 추가하면 `sensor_cache_offsets`가 다음처럼 쌓인다:

```
초기: [0]
센서 0 추가 후: [0, 4]    ← 올바름 (0번 센서는 offset 0부터)
센서 1 추가 후: [0, 4, 4]  ← 틀림! 4가 아니라 8이어야 함
센서 2 추가 후: [0, 4, 4, 4]
센서 3 추가 후: [0, 4, 4, 4, 4]
```

기대값:

```
[0, 4, 8, 12, 16]
```

### 영향

- CUDA 커널: `sensor_cache_offsets[i]` 위치에 데이터를 씀 (누적 오프셋 사용 → 정상)
- read() 경로: `_cache_idx` = `sensor_cache_offsets[i]` 사용 → 센서 1 이후 `_cache_idx`가 8, 12, 16이 되어야 하나, 저장된 값이 4, 4, 4라 커널이 쓴 위치와 읽는 위치가 불일치
- 결과: **센서 인덱스 2, 3 (3번째, 4번째 센서)는 항상 0.0 반환**

### 수정 코드 (FIX)

```python
# output_hits를 확장한 직후, 확장된 크기(=누적 전체 ray 수)를 append
self._shared_metadata.sensor_cache_offsets = concat_with_tensor(
    self._shared_metadata.sensor_cache_offsets,
    self._shared_metadata.output_hits.shape[-1]   # ← FIX: 확장 후 총 크기
)
```

확장 후 `output_hits.shape[-1]`은 이미 이번 센서 rays를 포함한 누적 총합이므로,
다음 센서의 시작 오프셋으로 정확히 사용된다.

수정 후 `sensor_cache_offsets`:

```
[0, 4, 8, 12, 16]  ← 모든 센서가 올바른 cache 슬롯에서 읽음
```

### 재현 조건

- 동일 타입의 Lidar 센서를 **3개 이상** 추가할 때
- 2개까지는 `[0, 4]`로 정상 동작 (인덱스 1 센서는 offset=4 → 올바름)
- 센서 개수가 많아질수록 뒤쪽 센서들이 offset=4(센서 1의 슬롯)에 덮어쓰기

### 확인 방법

```python
import genesis as gs
import torch

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
car = scene.add_entity(gs.morphs.URDF(file="..."))

lidars = []
for wx, wy, wz in [(1.35, 0.80, 0.34), (1.35, -0.80, 0.34),
                   (-1.35, 0.80, 0.34), (-1.35, -0.80, 0.34)]:
    lidar = scene.add_sensor(gs.sensors.Lidar(
        pattern=gs.sensors.GridPattern(resolution=1.0, size=(0.0, 0.0), direction=(0.0, 0.0, -1.0)),
        entity_idx=car.idx, pos_offset=(wx, wy, wz), return_world_frame=False))
    lidars.append(lidar)

scene.build(n_envs=1)
car.set_pos(torch.tensor([[0.0, 0.0, 1.0]]))
scene.step()

for i, lidar in enumerate(lidars):
    d = float(lidar.read().distances.flatten()[0])
    print(f"Lidar {i}: {d:.4f}")
    # 패치 전: Lidar 2, 3 → 0.0000
    # 패치 후: Lidar 2, 3 → ~1.34 (정상)
```

### 패치 적용 시 주의사항

1. `genesis_env` 재설치(`pip install genesis-world`)하면 패치가 사라진다.
2. 재설치 후 동일 수정 필요:
   ```bash
   vi ~/.../genesis_env/lib/python3.10/site-packages/genesis/engine/sensors/raycaster.py
   # line ~395: self._cache_size → self._shared_metadata.output_hits.shape[-1]
   ```
3. 또는 genesis-world upstream에 PR 제출 권장.

---

## 버그 2: link.idx global/local 불일치

```
증상: force가 엉뚱한 link에 적용됨

원인: link.idx = global index (1~11)
      force_tensor = local index (0~10) 기준
      → front_right_wheel.idx=11이 tensor 범위 초과

해결:
  links_idx = [l.idx for l in car.links]
  local_pos = links_idx.index(global_idx)
  force_tensor[local_pos, 2] = force_mag
```

---

## 버그 3: 첫 스텝 Lidar 미초기화

```
증상: step 0에서 distances = 0.0 반환

해결:
  prev_d = [TIRE_RADIUS * 2] * 4
  if hit_dist <= 1e-6:
      hit_dist = prev_d[i]
  prev_d[i] = hit_dist
```
