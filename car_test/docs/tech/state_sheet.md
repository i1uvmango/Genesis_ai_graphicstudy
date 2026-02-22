# State Sheet
 ![](../../res/0216/mlp.png)


#### Input Features (7 Dim)

$$\mathbf{X} = [v_{long}, v_{lat}, \omega, CTE, HE, T_{raw}, S_{rad}]$$

| 그룹 | 피처 | 설명 |
| :--- | :--- | :--- |
| **Vehicle Dynamics** | `v_long` | 종속도 |
| | `v_lat` | 횡속도 |
| | `yaw_rate` | 각속도 |
| **Genesis Feedback** | `cte` | 횡방향 오차 |
| | `he` | 헤딩 오차 |
| **Blender FeedForward** | `throttle_ref` | golden throttle 참조 값 |
| | `steer_ref` | golden steer 참조 값 |
* Genesis 의 오차도 반영한 closed-loop 학습
    * (golden t,s 추출 시에도 genesis engine 활용하여 오차 반영했으니 학습에도 똑같이 반영)
    * inference 시에도 feedback 변수들 들어가야함
* Standard Scaler 