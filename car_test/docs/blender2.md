![car2](../res/blender_car2.png)

https://www.youtube.com/watch?v=yPGreAE1NWw&t=8s 여기 유튜브 채널









# Blender 자동차 주행 세팅 작업 로그 (MD)

## 1) 목표
- Blender에서 **자동차가 움직이는 장면**을 구현하고  
  **단순 애니메이션(키프레임)** 인지 **물리 기반(중력/마찰/충돌)** 인지 구분/검증.
- Rigacar 리그를 활용해 **경로(Path) 따라 자동 주행**까지 구현.

---

## 2) 작업 환경
- **Blender**: 4.5.3 LTS (한국어 UI)
- **애드온**: `rigacar-Jeanyan3D-Fix-v3` (유튜버 수정판)
- (참고) 초기엔 Rigacar 7.1(라이트/혼용)로 **Path 메뉴 미표시** 이슈 발생.

---

## 3) 개념 정리 (교수님 요구사항 대비)
- **단순 애니메이션**: 위치/회전을 **키프레임**으로 보간 → 물리 계산 없음.
- **물리 시뮬레이션**: **Rigid Body(충돌/중력/마찰)** + **Constraints(힌지/모터)** 에 의해 계산.
- **구분법**
  - 타임라인 키프레임만 있고 Physics 설정 없음 → **단순 애니메이션**
  - Physics 탭에 **Rigid Body** 적용, 재생 시 중력/충돌 반응 → **물리 시뮬레이션**

---

## 4) 진행 타임라인 요약
1. BlenderKit 사용
2. Rigid Body/Constraint/마찰·중력 등 물리 개념 확인.
3. Rigacar 리그 확인: **노란 Car_CTRL** 및 Custom Properties 확인.
4. Path Animation 메뉴 없음 → 자동 주행 불가.
5. **유튜버 수정판 설치(`rigacar-Jeanyan3D-Fix-v3`)** 후 Path 기능 기대.
6. Path Animation 존재 하지 않음
7. 유튜버 설명이 숙련자를 위한 설명임 


## 자동 주행(Path)
* **아직 미실행**
1. **Car_CTRL 선택** → Rigacar 탭 하단 **Path Animation** 섹션.
2. **Create Path**: `RC_Path`(Bezier Curve) 생성 → Edit Mode에서 XY 평면 위로 경로 편집.
3. **Attach to Path**: Car_CTRL을 경로에 연결(Follow Path 제약 자동).
4. **Drive Along Path**: 이동/조향/바퀴회전 드라이버 자동 설정.  
   - Start: `1`, End: `250`, Speed: `3.0~6.0`, Auto Steering: 체크, Wheel Rotation Factor: `1.0`
5. 재생 ▶ → 경로 주행 + 커브 자동 조향 + 바퀴 회전.
6. (선택) **Bake Path Animation** → 키프레임으로 고정.



## 10) 단축키 메모
- **N**: 오른쪽 사이드바(Rigacar 탭)
- **Tab**: 오브젝트/에딧 모드 전환
- **G / R / S**: 이동/회전/스케일
- **Ctrl + A**: 변환 적용
- **Space / ▶**: 재생
