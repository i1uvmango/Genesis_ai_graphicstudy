# train_blender_fixed.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from numpy.linalg import norm

# =========================
# 설정: 경로 / 디바이스
# =========================
# CSV_PATH가 첫 번째 스크립트의 OUTPUT_CSV_PATH와 일치하는지 확인!
CSV_PATH = "drive_8_test.csv"
MODEL_PATH = "checkpoint/drive_8_test.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Dataset 정의
# =========================
class CarDataset(Dataset):
    def __init__(self, states, actions, state_mean, state_std):
        # 데이터는 이미 정규화되어 들어옴 (states는 스케일링, actions는 클리핑)
        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.float32) 
        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        # 상태 정규화: (상태 - 평균) / 표준편차
        s = (self.states[idx] - self.state_mean) / self.state_std
        a = self.actions[idx]
        return torch.from_numpy(s), torch.from_numpy(a)


# =========================
# MLP 정의 (test_bc.py와 동일한 구조)
# =========================
class CarControllerMLP(nn.Module):
    def __init__(self, state_dim=6, hidden_dim=64, action_dim=2):
        super().__init__()
        self.network = nn.Sequential(  # Changed from 'backbone' to 'network'
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Moved Tanh inside Sequential
        )

    def forward(self, x):
        return self.network(x)  # Simplified - Tanh is already inside


# =========================
# CSV 로드 + Global -> Local 변환 (핵심 수정 부분)
# =========================
def load_csv_and_build_targets(csv_path):
    df = pd.read_csv(csv_path)

    # 1. State 가져오기 (Global)
    req_cols = [
        "g_lin_vx", "g_lin_vy", "g_lin_vz",
        "g_ang_vx", "g_ang_vy", "g_ang_vz",
        "g_qw", "g_qx", "g_qy", "g_qz", 
        "steer", "throttle_norm" # Action Target
    ]
    for col in req_cols:
        if col not in df.columns:
            # 첫 번째 스크립트에서 CSV 컬럼을 확인하세요.
            raise ValueError(f"CSV에 '{col}' 컬럼이 없습니다. (Blender 스크립트 확인 필요)")

    # numpy 변환
    g_lin_v = df[["g_lin_vx", "g_lin_vy", "g_lin_vz"]].to_numpy(dtype=np.float32)
    g_ang_v = df[["g_ang_vx", "g_ang_vy", "g_ang_vz"]].to_numpy(dtype=np.float32)
    g_q = df[["g_qw", "g_qx", "g_qy", "g_qz"]].to_numpy(dtype=np.float32) # (N, 4) [w, x, y, z]


    # 2. Global -> Local 변환 (현재 사용 안 함 - 이중 변환 버그 제거)
    
    l_lin_v = g_lin_v  # Already local
    l_ang_v = g_ang_v  # Already local

    # 4. State 구성
    states = np.concatenate([l_lin_v, l_ang_v], axis=1) # (N, 6)

    # 5. Action 가져오기 (CSV에서 정규화된 최종값 사용)
    actions = df[["steer", "throttle_norm"]].to_numpy(dtype=np.float32)

    # 6. 정규화 스케일 (Action Target은 이미 정규화되어 있으므로, 1.0으로 설정)
    steer_scale = 1.0
    omega_max = 1.0

    return states, actions, steer_scale, omega_max


# =========================
# DataLoader 구성 (state_mean[1:] = 0.0 로직 유지)
# =========================
def make_dataloaders(states, actions, batch_size=128, val_ratio=0.2, seed=42):
    N = states.shape[0]
    idx = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    split = int(N * (1.0 - val_ratio))
    train_idx = idx[:split]
    val_idx = idx[split:]

    states_tr = states[train_idx]
    actions_tr = actions[train_idx]
    states_val = states[val_idx]
    actions_val = actions[val_idx]

    state_mean = states_tr.mean(axis=0)
    state_std = states_tr.std(axis=0) + 1e-6

    # [Bias Fix] 횡방향/회전 속도는 물리적으로 0이 중심이어야 함.
    # 전진 속도(idx 0)를 제외한 나머지는 평균을 0으로 강제함.
    state_mean[1:] = 0.0

    train_ds = CarDataset(states_tr, actions_tr, state_mean, state_std)
    val_ds = CarDataset(states_val, actions_val, state_mean, state_std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, state_mean, state_std


# =========================
# 학습 루프 (Train Loop)
# =========================
# (나머지 train 함수는 동일하게 유지됩니다.)

def train(
    csv_path=CSV_PATH,
    model_path=MODEL_PATH,
    num_epochs=100,
    batch_size=128,
    lr=1e-3,
):
    states, actions, steer_scale, omega_max = load_csv_and_build_targets(csv_path)

    print(f"[INFO] Loaded CSV: {csv_path}")
    print(f"       states:      {states.shape}")
    print(f"       actions:     {actions.shape}")
    
    # states의 첫 번째 컬럼(l_lin_vx)의 평균을 확인하여 축 방향 체크
    l_lin_vx_mean = states[:, 0].mean()
    if abs(l_lin_vx_mean) < 0.1:
        print("[WARNING] Local Lin Vel X is near zero. Check coordinate system.")


    train_loader, val_loader, state_mean, state_std = make_dataloaders(
        states, actions, batch_size=batch_size
    )

    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    model = CarControllerMLP(state_dim=state_dim, action_dim=action_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    best_val = float("inf")

    # ... (Train/Validation Loop은 이전 코드와 동일) ...
    for epoch in range(1, num_epochs + 1):
        # ----- Train -----
        model.train()
        tr_loss_sum = 0.0
        tr_n = 0

        for s_batch, a_batch in train_loader:
            s_batch = s_batch.to(DEVICE)
            a_batch = a_batch.to(DEVICE)

            pred = model(s_batch)
            loss = criterion(pred, a_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = s_batch.size(0)
            tr_loss_sum += loss.item() * bs
            tr_n += bs

        train_loss = tr_loss_sum / max(1, tr_n)

        # ----- Validation -----
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for s_batch, a_batch in val_loader:
                s_batch = s_batch.to(DEVICE)
                a_batch = a_batch.to(DEVICE)

                pred = model(s_batch)
                loss = criterion(pred, a_batch)

                bs = s_batch.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs

        val_loss = val_loss_sum / max(1, val_n)

        print(f"[Epoch {epoch:03d}] train={train_loss:.6f}  val={val_loss:.6f}")

        # 베스트 모델 저장 (test_bc.py와 호환되는 형식)
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state_dict": model.state_dict(),  # test_bc.py expects this key
                "state_mean": state_mean.tolist(),
                "state_std": state_std.tolist(),
                "config": {  # test_bc.py expects this structure
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "hidden_dim": 64,  # from CarControllerMLP default
                },
                "epoch": epoch,
                "val_loss": val_loss,
            }
            torch.save(ckpt, model_path)
            print(f"  ↳ Saved best model to '{model_path}' (val={val_loss:.6f})")

    print("Done.")
    print("Best val_loss =", best_val)
    print("state_mean:", state_mean)
    print("state_std :", state_std)

if __name__ == "__main__":
    train()