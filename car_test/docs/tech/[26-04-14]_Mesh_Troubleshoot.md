# HillRoad OBJ Mesh Dense Export 문제 분석 및 수정 보고서

## 1. 문제 현상

### 1-1. Genesis 실행 시 NaN 경고
```
/genesis_env/lib/python3.10/site-packages/genesis/utils/terrain.py:476:
RuntimeWarning: All-NaN slice encountered
  h_coarse = np.nanmax(h_fine.reshape(...))
```

### 1-2. 주행 이탈 현상 (P03_SCurve_HillPeak)
- BC-Policy 추론 시 S커브 중반부터 경로 이탈
- MPPI 최적화(`1_find_gt.py`) 시에도 기준 경로 대비 커브 바깥쪽으로 밀림
- Genesis 뷰어에서 terrain이 계단(staircase)처럼 보이는 현상 확인

---

## 2. 근본 원인 분석

### 2-1. `export_hillroad_mesh()` 함수의 구조적 문제

기존 함수는 **HillRoad MESH 오브젝트**를 export 대상으로 사용:

```python
# 기존 코드 (문제)
mesh_obj = bpy.data.objects.get(f"HillRoad_{name}")   # MESH 타입
me_eval  = mesh_obj.evaluated_get(dep).to_mesh()       # 여전히 22~24개
```

**HillRoad MESH는 `_make_hill_mesh()`가 경로 제어점으로만 생성한 단순 폴리곤**이어서, 커브의 `resolution_u=64` 설정이 전혀 반영되지 않았음.

### 2-2. 버텍스 수 비교

| 경로 | HillRoad MESH (기존) | 커브 evaluated (수정 후) |
|---|---|---|
| P01_GentleCurve_Hill | 22개 | 1,280개 |
| P02_SharpCurve_Uphill | 20개 | 1,152개 |
| P03_SCurve_HillPeak | 24개 | 1,408개 |
| P04~P15 (평균) | 22~24개 | 1,280~1,408개 |

### 2-3. NaN 발생 메커니즘

```
OBJ 버텍스: 24개  (경로점 12개 × 양쪽 2개)
height field spacing=0.1m → 약 450×100 = 45,000 격자 생성

45,000개 격자 중 24개 버텍스만 존재
→ 대부분의 격자에 지형 데이터 없음 → NaN
→ mesh_to_heightfield의 nanmax() 연산에서 All-NaN slice 경고
→ terrain이 계단 형태로 렌더링됨
→ 차량이 폴리곤 경계마다 충격 → heading 불안정 → 경로 이탈
```

### 2-4. `mesh_all` 실행해도 동일 문제가 반복된 이유

```
MODE = "mesh_all"
    → export_all_meshes()
        → export_hillroad_mesh(num)         ← HillRoad MESH 오브젝트 사용
            → mesh_obj.data.vertices        ← 22~24개 (커브 무관)
```

**커브의 resolution_u=64는 Blender 뷰포트 렌더링에만 반영되고,
OBJ export 시에는 적용되지 않았음.**

---

## 3. 수정 내용

### 3-1. `export_hillroad_mesh()` 함수 교체

**HillRoad MESH 오브젝트 → 커브 evaluated 데이터** 기반으로 변경:

```python
# 수정 후 코드
def export_hillroad_mesh(path_num):
    # 커브 오브젝트 탐색 (P10은 .001 suffix)
    curve_obj = bpy.data.objects.get(name)
    if not curve_obj or curve_obj.type != 'CURVE':
        curve_obj = bpy.data.objects.get(f"{name}.001")

    # 커브 evaluated 포인트 추출 (resolution_u=64 반영)
    dep      = bpy.context.evaluated_depsgraph_get()
    eval_obj = curve_obj.evaluated_get(dep)
    mesh_tmp = eval_obj.to_mesh()
    pts      = [(mat_w @ v.co).copy() for v in mesh_tmp.vertices]
    # → 640~704개 포인트 (12개 제어점 × resolution_u=64)

    # 양쪽 도로 버텍스 + 삼각형 면 생성 → OBJ 저장
    ...
```

### 3-2. `export_all_meshes()` 실행 결과 (수정 후)

```
[메시 일괄 Export] 대상: 15개
✅ [01] P01_GentleCurve_Hill           |  1,280 verts | Z:[0.000, 0.965]
✅ [02] P02_SharpCurve_Uphill          |  1,152 verts | Z:[0.000, 1.334]
✅ [03] P03_SCurve_HillPeak            |  1,408 verts | Z:[0.000, 1.170]
✅ [04] P04_RightCurve_Downhill        |  1,408 verts | Z:[0.158, 1.377]
✅ [05] P05_Gentle_0p4m                |  1,280 verts | Z:[0.000, 0.390]
✅ [06] P06_Medium_1p0m                |  1,280 verts | Z:[0.000, 0.975]
✅ [07] P07_Steep_2p0m                 |  1,280 verts | Z:[0.000, 1.949]
✅ [08] P08_Up_then_Down               |  1,280 verts | Z:[0.000, 1.462]
✅ [09] P09_Down_then_Up               |  1,280 verts | Z:[0.000, 1.200]
✅ [10] P10_Rolling_3cycle             |  1,280 verts | Z:[0.166, 0.515]
✅ [11] P11_Bumpy_Road                 |  1,280 verts | Z:[0.000, 0.125]
✅ [12] P12_CrossSlope_Left            |  1,280 verts | Z:[0.000, 0.270]
✅ [13] P13_LeftCurve_DoubleHill       |  1,408 verts | Z:[0.000, 0.725]
✅ [14] P14_RightCurve_Rolling         |  1,280 verts | Z:[0.000, 0.357]
✅ [15] P15_LongStraight_GradualHill   |  1,280 verts | Z:[0.012, 1.174]
완료: 15/15개
```

---

## 4. 수정 후 예상 효과

| 항목 | 기존 | 수정 후 |
|---|---|---|
| OBJ 버텍스 수 | 22~24개 | 1,152~1,408개 |
| height field NaN 비율 | 극히 높음 (대부분 NaN) | 대폭 감소 |
| terrain 계단 현상 | 심각 | 해소 |
| 폴리곤 경계 충격 | 매 ~3m마다 발생 | 매 ~0.05m로 완화 |
| MPPI heading 진동 | 계단 충격 → 불안정 | 안정화 |
| `All-NaN slice` 경고 | 발생 | 발생 안 함 |

---

## 5. 이후 필요한 작업

```bash
# 1. P03 golden_inputs 삭제 (기존 unstable 데이터 제거)
rm ./csv/3d/golden_inputs_P03_SCurve_HillPeak.csv

# 2. 전체 golden_inputs 재생성 (새 dense OBJ 기반)
#    기존 파일은 SKIP, P03만 새로 생성됨
python autofind.py

# 3. 재학습
python train_bc_3d.py
```

> **주의:** 기존에 생성된 다른 경로의 golden_inputs도 24개짜리 OBJ 기반이므로,
> 학습 품질 향상을 위해서는 전체 경로 golden_inputs 재생성을 권장함.

---

## 6. 재발 방지

`rbc_auto.py`의 `export_hillroad_mesh()` 함수가 **커브 evaluated 기반**으로 수정되었으므로, 이후 `MODE = "mesh_all"` 실행 시 자동으로 dense OBJ(1280~1408 verts)가 생성됨.

추가적으로 `mesh_to_heightfield` 파라미터도 확인 필요:

```python
# 현재 설정
height, xs, ys = mesh_to_heightfield(
    cfg.hill_mesh,
    spacing=0.1,      # 0.1m 격자 → 촘촘한 height field
    oversample=3,
)
```

dense OBJ(1408 verts)와 `spacing=0.1` 조합이면 NaN 격자가 대폭 줄어들어 terrain이 연속적으로 생성됨.