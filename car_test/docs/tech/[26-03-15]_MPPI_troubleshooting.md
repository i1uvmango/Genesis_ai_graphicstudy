# MPPI trouble shooting & Insight

## MPPI 에서 base index 결정 : Time VS Spacial

![](../../res/0106/0106_2.png)

| 상황        | **Time-based** | Spatial-based |
| --------- | ---------- | ------------- |
| 차가 느려짐    | 목표는 계속 도망  | 목표 일정 간격 유지    |
| 차가 멈춤     | 목표는 저 멀리   | 목표도 멈춤        |
| 코너에서 미끄러짐 | 오류 폭발      | 오류 제한         |
| 학습 안정성    | 매우 나쁨      | **매우 좋아짐**    |

* 초기에 차량이 blender 만큼 빠르게 가속하지 못하여 Spatial Based 인덱스를 사용
* Behavior Cloning 은 시간에 따라 같은 움직임을 해야하므로, 근본적인 해결이 되지 않음
* Time-based Index 사용해야 하는 이유



## Blender 의 급가속 TroubleShooting
