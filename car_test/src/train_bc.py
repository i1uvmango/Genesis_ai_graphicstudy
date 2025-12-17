"""
Professional Behavior Cloning Training Script for Genesis AI Navigation

This script trains a neural network policy to clone expert driving behavior from 
demonstration data collected in Blender and prepared for deployment in Genesis AI simulation.

Data Pipeline:
1. Load global-frame velocities and quaternions from CSV
2. Transform to local vehicle frame using quaternion rotation
3. Normalize states for stable training
4. Train MLP policy with supervised learning

Author: Genesis AI Team
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration"""
    # Paths
    csv_path: str = "drive_8_test.csv"
    checkpoint_dir: str = "checkpoint"
    model_name: str = "bc_policy.pth"
    
    # Training hyperparameters
    num_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    val_ratio: float = 0.2
    random_seed: int = 42
    
    # Model architecture
    state_dim: int = 6  # [lin_vx, lin_vy, lin_vz, ang_vx, ang_vy, ang_vz] in local frame
    action_dim: int = 2  # [steering, throttle]
    hidden_dim: int = 64
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Data Processing
# ============================================================================

def quaternion_rotate_inverse(vectors: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    """
    Rotate vectors from global frame to local frame using quaternion inverse.
    
    Args:
        vectors: (N, 3) array of vectors in global frame
        quaternions: (N, 4) array of quaternions [w, x, y, z]
    
    Returns:
        (N, 3) array of vectors in local frame
    """
    # Extract quaternion components
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]
    
    # Extract vector components
    vx = vectors[:, 0]
    vy = vectors[:, 1]
    vz = vectors[:, 2]
    
    # Apply inverse quaternion rotation
    # Formula: v_local = q_conj * v_global * q
    # Using optimized vector rotation formula
    
    # Conjugate quaternion (inverse for unit quaternions)
    qx, qy, qz = -x, -y, -z
    qw = w
    
    # First rotation: t = q_conj * v
    t_w = -qx * vx - qy * vy - qz * vz
    t_x = qw * vx + qy * vz - qz * vy
    t_y = qw * vy + qz * vx - qx * vz
    t_z = qw * vz + qx * vy - qy * vx
    
    # Second rotation: v_local = t * q
    out_x = t_w * x + t_x * w + t_y * z - t_z * y
    out_y = t_w * y - t_x * z + t_y * w + t_z * x
    out_z = t_w * z + t_x * y - t_y * x + t_z * w
    
    return np.stack([out_x, out_y, out_z], axis=1)


def load_and_process_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CSV data and process into state-action pairs.
    
    Args:
        csv_path: Path to CSV file with columns:
            - g_lin_vx, g_lin_vy, g_lin_vz (global linear velocity)
            - g_ang_vx, g_ang_vy, g_ang_vz (global angular velocity)
            - g_qw, g_qx, g_qy, g_qz (orientation quaternion)
            - steer (steering angle)
            - throttle_norm (normalized throttle)
    
    Returns:
        states: (N, 6) array of local-frame velocities
        actions: (N, 2) array of [steering, throttle]
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = [
        "g_lin_vx", "g_lin_vy", "g_lin_vz",
        "g_ang_vx", "g_ang_vy", "g_ang_vz",
        "g_qw", "g_qx", "g_qy", "g_qz",
        "steer", "throttle_norm"
    ]
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Extract global velocities
    g_lin_vel = df[["g_lin_vx", "g_lin_vy", "g_lin_vz"]].values.astype(np.float32)
    g_ang_vel = df[["g_ang_vx", "g_ang_vy", "g_ang_vz"]].values.astype(np.float32)
    
    # Extract orientation
    g_quat = df[["g_qw", "g_qx", "g_qy", "g_qz"]].values.astype(np.float32)
    
    # Transform to local frame
    print("Transforming velocities to local frame...")
    l_lin_vel = quaternion_rotate_inverse(g_lin_vel, g_quat)
    l_ang_vel = quaternion_rotate_inverse(g_ang_vel, g_quat)
    
    # CRITICAL: Blender data has 90° Z-rotation applied (ROT_B2G in blender_data_extract.py)
    # We need to UNDO this rotation to get proper Genesis local frame (X=forward, POSITIVE)
    # 90° Z-rotation: X→Y, Y→-X. Reverse: X=-Y, Y=X (then adjust for left convention)
    print("Reversing Blender's Z-axis 90° rotation...")
    
    l_lin_vel_corrected = np.zeros_like(l_lin_vel)
    l_lin_vel_corrected[:, 0] = -l_lin_vel[:, 1]  # X_genesis = -Y_rotated (forward = POSITIVE)
    l_lin_vel_corrected[:, 1] = -l_lin_vel[:, 0]  # Y_genesis = -X_rotated (left)
    l_lin_vel_corrected[:, 2] = l_lin_vel[:, 2]   # Z unchanged
    
    l_ang_vel_corrected = np.zeros_like(l_ang_vel)
    l_ang_vel_corrected[:, 0] = -l_ang_vel[:, 1]  # Roll
    l_ang_vel_corrected[:, 1] = -l_ang_vel[:, 0]  # Pitch
    l_ang_vel_corrected[:, 2] = l_ang_vel[:, 2]   # Yaw unchanged
    
    l_lin_vel = l_lin_vel_corrected
    l_ang_vel = l_ang_vel_corrected
    
    
    # Combine into state vector
    states = np.concatenate([l_lin_vel, l_ang_vel], axis=1)  # (N, 6)
    
    # Extract actions
    actions = df[["steer", "throttle_norm"]].values.astype(np.float32)  # (N, 2)
    
    # Data validation
    print(f"\nData Summary:")
    print(f"  Total samples: {len(states)}")
    print(f"  State shape: {states.shape}")
    print(f"  Action shape: {actions.shape}")
    print(f"\nState statistics (local frame):")
    print(f"  Lin vel X: mean={states[:, 0].mean():.3f}, std={states[:, 0].std():.3f}")
    print(f"  Lin vel Y: mean={states[:, 1].mean():.3f}, std={states[:, 1].std():.3f}")
    print(f"  Lin vel Z: mean={states[:, 2].mean():.3f}, std={states[:, 2].std():.3f}")
    print(f"  Ang vel X: mean={states[:, 3].mean():.3f}, std={states[:, 3].std():.3f}")
    print(f"  Ang vel Y: mean={states[:, 4].mean():.3f}, std={states[:, 4].std():.3f}")
    print(f"  Ang vel Z: mean={states[:, 5].mean():.3f}, std={states[:, 5].std():.3f}")
    print(f"\nAction statistics:")
    print(f"  Steering:  mean={actions[:, 0].mean():.3f}, std={actions[:, 0].std():.3f}")
    print(f"  Throttle:  mean={actions[:, 1].mean():.3f}, std={actions[:, 1].std():.3f}")
    
    return states, actions


# ============================================================================
# Dataset
# ============================================================================

class BehaviorCloningDataset(Dataset):
    """PyTorch dataset for behavior cloning"""
    
    def __init__(self, states: np.ndarray, actions: np.ndarray, 
                 state_mean: np.ndarray, state_std: np.ndarray):
        self.states = torch.from_numpy(states.astype(np.float32))
        self.actions = torch.from_numpy(actions.astype(np.float32))
        self.state_mean = torch.from_numpy(state_mean.astype(np.float32))
        self.state_std = torch.from_numpy(state_std.astype(np.float32))
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalize state
        state = (self.states[idx] - self.state_mean) / self.state_std
        action = self.actions[idx]
        return state, action


def create_dataloaders(states: np.ndarray, actions: np.ndarray, 
                       config: Config) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """
    Create training and validation dataloaders with proper normalization.
    
    Args:
        states: (N, state_dim) state array
        actions: (N, action_dim) action array
        config: Training configuration
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        state_mean: State normalization mean
        state_std: State normalization std
    """
    # Split train/val
    np.random.seed(config.random_seed)
    n_samples = len(states)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * (1 - config.val_ratio))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    states_train = states[train_idx]
    actions_train = actions[train_idx]
    states_val = states[val_idx]
    actions_val = actions[val_idx]
    
    # Compute normalization statistics from training set only
    state_mean = states_train.mean(axis=0)
    state_std = states_train.std(axis=0) + 1e-6
    
    # Apply zero-centering to lateral and rotational velocities
    # Only forward velocity should have non-zero mean
    state_mean[1:] = 0.0
    
    print(f"\nNormalization parameters:")
    print(f"  State mean: {state_mean}")
    print(f"  State std:  {state_std}")
    
    # Create datasets
    train_dataset = BehaviorCloningDataset(states_train, actions_train, state_mean, state_std)
    val_dataset = BehaviorCloningDataset(states_val, actions_val, state_mean, state_std)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=0)
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader, state_mean, state_std


# ============================================================================
# Model
# ============================================================================

class BehaviorCloningPolicy(nn.Module):
    """MLP policy for behavior cloning"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


# ============================================================================
# Training
# ============================================================================

def train_epoch(model: nn.Module, dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                device: str) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for states, actions in dataloader:
        states = states.to(device)
        actions = actions.to(device)
        
        # Forward pass
        predictions = model(states)
        loss = criterion(predictions, actions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(states)
    
    return total_loss / len(dataloader.dataset)


def validate(model: nn.Module, dataloader: DataLoader, 
            criterion: nn.Module, device: str) -> float:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for states, actions in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            
            predictions = model(states)
            loss = criterion(predictions, actions)
            
            total_loss += loss.item() * len(states)
    
    return total_loss / len(dataloader.dataset)


def train(config: Config):
    """Main training loop"""
    
    # Load and process data
    states, actions = load_and_process_data(config.csv_path)
    
    # Create dataloaders
    train_loader, val_loader, state_mean, state_std = create_dataloaders(
        states, actions, config
    )
    
    # Initialize model
    model = BehaviorCloningPolicy(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim
    ).to(config.device)
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training on {config.device}")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_dir, config.model_name)
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, config.device)
        
        # Print progress
        print(f"Epoch {epoch:3d}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f}", end="")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save checkpoint with all necessary information
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'state_mean': state_mean.tolist(),
                'state_std': state_std.tolist(),
                'config': {
                    'state_dim': config.state_dim,
                    'action_dim': config.action_dim,
                    'hidden_dim': config.hidden_dim,
                },
                'epoch': epoch,
                'val_loss': val_loss,
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f" ← Best model saved!")
        else:
            print()
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"{'='*60}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Behavior Cloning Policy")
    parser.add_argument('--csv', type=str, default=Config.csv_path,
                       help='Path to CSV data file')
    parser.add_argument('--output', type=str, default=Config.model_name,
                       help='Output model filename')
    parser.add_argument('--epochs', type=int, default=Config.num_epochs,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=Config.batch_size,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=Config.learning_rate,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Update config from args
    config = Config()
    config.csv_path = args.csv
    config.model_name = args.output
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
