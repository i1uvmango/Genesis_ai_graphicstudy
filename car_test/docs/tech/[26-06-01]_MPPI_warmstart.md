# 3d ray MPPI warm-start configuration

warmstart 용 최적 파라미터: 이 기준을 근처로 grid search 추천.


## Sim / 정규화

| 항목 | 값 |
| :--- | :--- |
| `dt`        | 1/48 (≈ 0.0208 s) |
| `substeps`  | 10 |
| `max_omega` | 60.0 (throttle 정규화 분모) |
| `max_steer` | 0.7 rad (steer 정규화 분모, ≈ 40°) |


## MPPI 코어

| 항목 | 값 |
| :--- | :--- |
| `n_samples`      | 2048 (병렬 env) |
| `horizon`        | 20 |
| `lookahead_step` | 2 |
| `corr_alpha`     | 0.5 (노이즈 시간 스무딩) |
| `mppi_lambda`    | 0.05 |


## Cost 가중치

| 가중치 | 값 | 가중치 | 값 |
| :--- | ---: | :--- | ---: |
| `w_dist`    | 6000 | `w_heading` | 4000 |
| `w_vel`     | 2500 | `w_pitch`   |  800 |
| `w_roll`    |  600 | `w_vz`      |  250 |
| `w_rate`    |  150 | `w_kappa`   |   50 |
| `w_accel`   |    1 | `w_ff`      |    1 |


## 노이즈 / blend / 기타

| 항목 | 값 |
| :--- | :--- |
| `noise_throttle` / `noise_steer` | 0.6 / 0.30 |
| `alpha_throttle` / `alpha_steer` | 0.2 / 0.3 (MPPI 영향력; CSV 0.8 / 0.7) |
| `overspeed_mult`     | 4.0 (과속 4배 벌점) |
| `kappa_v_threshold`  | 0.2 |
| `spawn_before_mesh`  | 0 |


## 실패 감지 / Rollback

| 항목 | 값 |
| :--- | :--- |
| `rollback_enabled`     | **False** (현재 rollback OFF — 끝까지 진행) |
| `cte_thresh` / `he_thresh`                          | 2.0 / 1.0 |
| `fail_window` / `rollback_frames` / `max_retries`   | 10 / 40 / 3 |


---

## ⚠️ 짚어둘 점 — 현재 cfg ≠ grid search 우승값 (run_126)

| 파라미터 | 현재 cfg | run_126 (grid 우승) |
| :--- | :--- | :--- |
| `mppi_lambda`                  | 0.05               | **0.20** |
| `alpha_throttle`               | 0.2                | **0.4** |
| `w_rate`                       | 150                | **40** |
| `w_vel` / `w_heading` / `w_dist` | 2500 / 4000 / 6000 | 2500 / 4000 / 6000 ✓ |

> `w_vel` / `w_heading` / `w_dist` 는 **우연히 일치**.
> 하지만 `lambda` · `alpha_throttle` · `w_rate` 는 grid 최적값과 다름.
> → 지금 마이닝은 **grid best 가 아니라 기본값(default) 기준**으로 돌고 있음.
