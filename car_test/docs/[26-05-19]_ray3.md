https://github.com/user-attachments/assets/5108072b-4fd8-4ff4-ac3f-117cc89c6a96

현재 속도를 따라가지 못하는 문제 발생

dt 동기화 되어있음:  blender도 50hz 로 추출

genesis

  1. Drive torque 키우기

  - SDK 기본 t_drive_max=1000 N·m → 2000 → 3000 (current, find_gt/play 동일)
  - 가설: hill 등판 시 토크 부족으로 가속 못 따라감
  - 결과: 일부 개선되었으나 여전히 V_sim < V_ref

  2. Index-based mining 도입 (1_find_gt_index.py)

  - _smooth_targets 가 df.iloc[i+h+lookahead] 로 직접 미래 frame 참조
  - 결과: V_sim 10.89 → 11.33, HE +1.01° → +0.63° (개선되었으나 ref 11.40 대비 여전히 -0.07)
  - Time index 에 맞지 않아서 사용하지 않음

  
  3. CSV 재생성 (Blender)

  - 스무딩
  - 속도 낮춰서 시도했지만 결과 동일

  4. - MPPI cost 재조정 중
  - 속도 극단적으로 주어도 별 차이 없음

  ---
  아직 시도 안 한 것들 (논의용):
  - 타이어 Pacejka 계수 (B, C, D, E) 튜닝
  - horizon 길이 증가
  - Drivetrain 의 기어비 / 효율 파라미터 확인
  - 공기저항 / rolling resistance 계수