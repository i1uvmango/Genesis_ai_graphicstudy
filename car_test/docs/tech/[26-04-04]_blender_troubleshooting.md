# RBC Auto Drive Troubleshooting

> Blender RBC (Rigid Body Car) Addon에서 커스텀 GuidePath 사용 시  
> Auto Drive가 작동하지 않는 원인 분석 및 해결책 기록

---

## 환경

- Blender 4.5.3 LTS
- RBC Addon Pro (`rbc_addon_pro` 모듈)
- 커스텀 GuidePath (단거리 직선/커브 경로, ~45m)

---

## 문제 1. `guide_menu = 'Guide Object'` 설정 오류

### 증상
Auto Drive를 켜도 차량이 전혀 움직이지 않음.

### 원인
RBC의 `frame_change_post_handler_299E4`는 `guide_menu` 값에 따라 다른 로직을 실행한다.

```python
if guide_menu == "Guide Object":
    # 오브젝트 방향으로 조향 (경로 무시)
elif guide_menu == "Guide Path":
    # 경로를 따라 Auto Drive 실행  ← 이게 실행되어야 함
```

GUI에서 경로를 설정했어도 내부 속성이 `'Guide Object'`로 남아 있으면  
Auto Drive가 GuidePath를 완전히 무시한다.

### 해결

```python
gc = rig.rig_armature.sna_rbc_rig_armature_props.rig_guide_control
gc.guide_menu = 'Guide Path'
```

---

## 문제 2. `evaluate_curve_length(vertices, 0)` 버그 — RBC 소스코드

### 증상
`guide_menu`를 올바르게 설정해도 차량이 출발 즉시 멈춤.

### 원인
`sna_guide_path_func_21185` 내부의 `evaluate_curve_length` 함수에 버그가 있다.

```python
# 원본 코드 (버그)
def evaluate_curve_length(vertices, until_idx=None):
    until_idx = until_idx or len(vertices) - 1  # ← 버그!
    ...
```

`until_idx = 0`(경로 시작점)을 전달하면 Python에서 `0`은 falsy이므로  
`0 or len(vertices) - 1` → 전체 길이를 반환한다.

```
nearest_point_index = 0 (경로 시작)
evaluate_curve_length(verts, 0) → 전체 길이 반환 (버그)
curve_position = 전체길이 / 전체길이 = 1.0
→ "경로 완주" 로 인식
→ Auto Drive 즉시 정지
```

### 해결
`None` 명시적 체크로 monkey-patch 적용:

```python
def evaluate_curve_length(vertices, until_idx=None):
    if until_idx is None:          # ← 수정: None 명시 체크
        until_idx = len(vertices) - 1
    ...
```

---

## 문제 3. Speed Curve 시작/끝 속도 = 0 → 데드락

### 증상
위 두 문제를 해결해도 차량이 수십 프레임 동안 움직이지 않거나  
Speed Curve를 잘못 수정한 후 아예 출발 불가.

### 원인 A — 원본 Speed Curve 구조 문제

RBC의 원본 Speed Curve는 경로 위치(0~1)를 속도 배율로 변환한다.

| 경로 위치 | 속도 배율 |
|-----------|-----------|
| 0% (시작) | 0%        |
| 50% (중간)| 100%      |
| 100% (끝) | 0%        |

커스텀 단거리 경로(45m)에서는 차량이 경로 0% 위치에서 출발하므로:

```
cpos = 0.0 → Speed Curve(0.0) = 0.0
→ target_speed = top_speed × 0 = 0
→ 차량 안 움직임
→ cpos 계속 0.0
→ 무한 루프 (데드락)
```

### 원인 B — 잘못된 Speed Curve 수정

버그 수정 과정에서 Speed Curve를 `(0→1, 1→1)` 평탄화하면  
원래의 점진 가속 동작이 사라지고 출발 즉시 최대속도가 된다.

### 해결
Speed Curve를 아래 사다리꼴 형태로 설정:

```python
curve.points[0].location = (0.0,  0.1)   # 시작: 10% 속도 (데드락 방지)
curve.points.new(0.15, 1.0)              # 15%에서 최대속도 도달
curve.points.new(0.85, 1.0)              # 85%까지 최대속도 유지
curve.points[-1].location  = (1.0,  0.0) # 끝: 정지 (자연스러운 감속)
for p in curve.points: p.handle_type = 'VECTOR'  # 오버슛 방지
```

```
0%  → 10%  ← 최소 속도 보장 (데드락 방지)
15% → 100% ← 점진 가속
85% → 100% ← 최대속도 유지
100%→  0%  ← 자연스러운 감속
```

---

## 최종 해결 플로우 (switch_path 실행 시 자동 적용)

```
switch_path(num) 호출
    │
    ├─ 1. GuidePath 커브 교체
    ├─ 2. HillRoad 메시 전환 (비활성 메시 z=-500 격리)
    ├─ 3. guide_menu = 'Guide Path'        ← 문제 1 해결
    ├─ 4. GuidePath cached_verts 갱신
    ├─ 5. monkey-patch 적용               ← 문제 2 해결
    │       - evaluate_curve_length 버그 수정
    │       - curve_position 정확히 0~1 반환
    ├─ 6. Speed Curve 사다리꼴 설정       ← 문제 3 해결
    │       - 시작 10%, 중간 100%, 끝 0%
    └─ 7. nearest_point_index = 0 초기화
```

---

## 관련 파일

| 파일 | 설명 |
|------|------|
| `rbc_auto.py` | 경로 전환 + 시뮬 + CSV 추출 통합 자동화 스크립트 |

---

## 참고: 수동 Speed 방식과의 차이

| 항목 | Auto Drive | 수동 Speed |
|------|------------|-----------|
| 속도 제어 | Speed Curve(경로 위치 기반) | 고정 속도 |
| 가속 방식 | 경로 시작 구간에서 점진 가속 | 물리 엔진이 자연스럽게 가속 |
| 적합한 경우 | 경로 끝에서 자동 감속 필요 시 | 단순 주행 데이터 수집 |