
## 강화학습 추가

```
pip uninstall -y rsl_rl
pip install rsl-rl-lib==2.2.4

pip install tensorboard tensorboardX
pip install numpy matplotlib

cd ~/Genesis/examples/tutorials
python control_your_robot.py -v

```
- rsl-rl-lib 라는 reinforcement learning library(강화 학습) 설치

> 실행한 AI model들
1. drone.py
    * hover_train.py 를 통해 훈련
    * 구체가 나타나면 드론이 구체를 향해 날아감
    * 강화학습: drone이 날아가는 목표, 추락여부, 움직임 최적화 대한 보상 학습
2. backflip.py
    * [Backflip AI 모델](https://drive.google.com/drive/folders/1ZxBaDP4_Br0ZhriQx_8A3JIwZZLnxix4?usp=drive_link) 여기서 체크포인트 다운로드
    * go2_train 학습
3. grasp.py
    * 로봇 팔이 물체를 잡는 행위에 대한 시뮬레이션
    * 강화학습이라는게 시뮬레이션을 돌렸을때 모델의 행위를 판단해서 +-로 스스로 평가를 해서 그에 대한 평가 + 피드백
    * 가장 좋은 예시라고 생각했지만 다음과 같은 에러 메세지로 실행 못했음

    ```
    WSL2에서 Vulkan을 사용할 때의 주요 문제는 GPU 드라이버와 Vulkan 드라이버가 완전히 호환되지 않는 경우가 있다는 점입니다. NVIDIA는 WSL2에서 GPU를 사용할 수 있도록 드라이버를 제공하지만, 그 자체로 Vulkan을 지원하는 데 제한이 있을 수 있습니다.
    ```




### tensorboard Parameter 기록용 코드
* update 된 parameter들이 궁금해서 찾아봄
```
import torch
from torch.utils.tensorboard import SummaryWriter
import os

# TensorBoard 기록을 위한 writer 생성
log_dir = "logs/tensorboard"
writer = SummaryWriter(log_dir=log_dir)

# 모델, 옵티마이저, 데이터셋 설정
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    
    # 훈련 데이터 로드 (예시)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        
        # 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()

        # loss 기록
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

        # 모델 가중치 기록
        for name, param in model.named_parameters():
            writer.add_histogram(f'Weights/{name}', param, epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 훈련 종료 후 기록 종료
writer.close()

# 훈련 후 TensorBoard를 실행하여 로그를 시각화
# 터미널에서 다음 명령어 실행:
# tensorboard --logdir=logs/tensorboard
```