# Transfering Blender Physics Engine to GenesisAI

## How
Blender는 물리 엔진을 외부에 제공하지 않음
* Bullet Physics Library를 기반함(C++)
* Pybullet을 사용하면 됨.  
단, "거의" 동일이지 100% 비트 단위 일치는 아님.
아래에서 이유와 구현 범위를 구체적으로 설명할게.

1. Physics Engine Calibration (물리파라미터 보정법)
    * `bpy` 로 물리 parameter 추출 : JSON 으로 저장
    * Genesis 에 JSON 파라미터 적용 &rarr; solver가 달라서 다른 output이 나옴. 
2. Custom Physics Module Embedding(동일 물리 엔진 삽입)
    * Pybullet 엔진 Genesis에 적용
3. Hybrid Integration(부분적 엔진 동기화)
    * Blender에서 계산된 `주요 물리효과(충돌, 서스펜션 등..)`만 Blender 사용, 나머지는 Genesis 물리로 계산