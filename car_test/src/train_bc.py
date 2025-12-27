import os
import math
import argparse
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Config
# -----------------------------
SEED = 42
def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class Cols:
    # 입력 피처(기본 권장: 자세+속도만)
    X = [
        "g_pos_x", "g_pos_y", "g_pos_z", # [NEW] Add Position for Map Context
        "g_qw", "g_qx", "g_qy", "g_qz",
        "g_lin_vx", "g_lin_vy", "g_lin_vz",
        "g_ang_vx", "g_ang_vy", "g_ang_vz",
        # 선택(있으면 도움이 됨): 전진방향 성분, rear spin
        "v_long", "spin_R",
    ]
    # 라벨(Genesis action)
    Y = ["steer", "throttle_norm"]


# -----------------------------
# Dataset
# -----------------------------
class CarDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# -----------------------------
# Model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2, hidden=(128, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Utils
# -----------------------------
def standardize(train_X: np.ndarray, val_X: np.ndarray):
    mu = train_X.mean(axis=0, keepdims=True)
    sigma = train_X.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (train_X - mu) / sigma, (val_X - mu) / sigma, mu, sigma

def mse_metrics(pred: torch.Tensor, y: torch.Tensor):
    # pred,y: (B,2)
    mse = torch.mean((pred - y) ** 2, dim=0)  # (2,)
    rmse = torch.sqrt(mse)
    return mse.detach().cpu().numpy(), rmse.detach().cpu().numpy()

def clip_actions(a: torch.Tensor):
    # Genesis action range 가정 [-1,1]
    return torch.clamp(a, -1.0, 1.0)


# -----------------------------
# Main train
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="drive_8_test.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--drop_first_row", action="store_true", default=True)  # ✅ 기본 True
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_all()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv} (같은 폴더에 두었는지 확인)")

    df = pd.read_csv(args.csv)

    # ✅ 첫 행은 dt=0 영향(속도/스핀 초기화) 때문에 기본적으로 버림
    if args.drop_first_row and len(df) > 1:
        df = df.iloc[1:].reset_index(drop=True)

    # 필요한 컬럼 체크
    missing = [c for c in (Cols.X + Cols.Y) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없음: {missing}\n현재 컬럼: {list(df.columns)}")

    # NaN/inf 정리
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=Cols.X + Cols.Y).reset_index(drop=True)

    # X, Y
    X = df[Cols.X].to_numpy(dtype=np.float32)
    Y = df[Cols.Y].to_numpy(dtype=np.float32)

    # Y는 이미 [-1,1] 형태(steer, throttle_norm)라고 가정하지만, 안전하게 clip
    Y = np.clip(Y, -1.0, 1.0)

    # Train/Val split (시계열이면 shuffle split이 leakage일 수 있음)
    # 여기선 간단히 마지막 val_ratio를 validation으로 둠 (시간 순서 유지)
    n = len(df)
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    if n_train < 10:
        raise ValueError(f"데이터가 너무 적음: n_train={n_train}, n_val={n_val}")

    train_X, val_X = X[:n_train], X[n_train:]
    train_Y, val_Y = Y[:n_train], Y[n_train:]

    # Standardize X only
    train_Xs, val_Xs, mu, sigma = standardize(train_X, val_X)

    train_ds = CarDataset(train_Xs, train_Y)
    val_ds = CarDataset(val_Xs, val_Y)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    model = MLP(in_dim=train_Xs.shape[1], out_dim=2, hidden=(128, 128), dropout=0.1).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_path = "checkpoint/drive_8_test.pth"
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Train: {len(train_ds)} rows | Val: {len(val_ds)} rows")
    print(f"X cols ({len(Cols.X)}): {Cols.X}")
    print(f"Y cols ({len(Cols.Y)}): {Cols.Y}")

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)

            pred = model(xb)
            pred = clip_actions(pred)  # ✅ 출력 안정화
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_losses.append(loss.item())

        # --- val ---
        model.eval()
        va_losses = []
        all_mse = []
        all_rmse = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(args.device)
                yb = yb.to(args.device)

                pred = clip_actions(model(xb))
                loss = loss_fn(pred, yb)
                va_losses.append(loss.item())

                mse, rmse = mse_metrics(pred, yb)
                all_mse.append(mse)
                all_rmse.append(rmse)

        tr = float(np.mean(tr_losses))
        va = float(np.mean(va_losses))
        mse_mean = np.mean(np.stack(all_mse, axis=0), axis=0)
        rmse_mean = np.mean(np.stack(all_rmse, axis=0), axis=0)

        if va < best_val:
            best_val = va
            torch.save({
                "model_state": model.state_dict(),
                "mu": mu,
                "sigma": sigma,
                "x_cols": Cols.X,
                "y_cols": Cols.Y,
                "config": {
                    "state_dim": train_Xs.shape[1],
                    "action_dim": 2,
                    "hidden_dim": 128
                }
            }, best_path)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[{epoch:04d}] train_loss={tr:.6f} val_loss={va:.6f} | "
                f"RMSE steer={rmse_mean[0]:.4f} throttle={rmse_mean[1]:.4f} | best_val={best_val:.6f}"
            )

    print(f"\nDone. Best saved to: {best_path}")
    print("Inference할 때는 저장된 mu/sigma로 X를 표준화하고 model 출력(clip된)을 steer/throttle로 쓰면 됨.")


if __name__ == "__main__":
    main()