"""
PPO-based Direct Control System (3D) - Multi-Environment
=========================================================

핵심: Pure Pursuit 조향 공식을 사용하지 않고,
공간 기반 waypoint 데이터 구조만을 활용하여 PPO 정책이 
steering과 throttle을 직접 학습.

Multi-Env (Genesis Native Parallelization):
    - scene.build(n_envs=N) 으로 병렬 환경 생성
    - 모든 상태/제어가 배치 처리: (num_envs, dim)
    - Done된 환경만 선택적 reset

Observation (18D):
    - target_rel (3D): 타겟 상대 위치 (body-frame)
    - v_body (3D): 차량 좌표계 속도
    - omega_body (3D): 차량 좌표계 각속도
    - g_body (3D): 중력 벡터 (body-frame)
    - tangent_rel (3D): 경로 접선 방향 (body-frame)
    - slip_proxy (1D): 슬립 프록시
    - prev_action (2D): 이전 스텝 [throttle, steer]

Action (2D):
    - throttle: [-1, 1]
    - steer: [-1, 1]

사용법:
    python train_ppo.py                    # 기본 학습 (8 envs)
    python train_ppo.py --num-envs 16      # 환경 수 지정
    python train_ppo.py --no-headless      # 뷰어 활성화 (1 env)
"""

import os
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

import genesis as gs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import pandas as pd
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict
from scipy.spatial.transform import Rotation


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """
    Configuration for PPO Direct Control.
    
    Design Philosophy:
    - Braking is not a rewarded action but a means to avoid penalties.
    - Reverse motion is treated as an invalid state rather than a target behavior.
    - The agent is encouraged to regulate speed through acceleration and braking
      to maintain stability and tracking performance.
    """
    # Environment
    num_envs: int = 8
    dt: float = 0.01
    substeps: int = 10
    max_steps: int = 2000
    
    # Control Limits
    max_steer: float = 0.5      # rad
    max_engine_torque: float = 300.0   # Nm (Increased for 1200kg car)
    max_brake_torque: float = 200.0    # Nm (Matched to engine)
    max_speed: float = 10.0     # m/s (for normalization)
    
    # Arc-length Lookahead (Steering Guidance)
    lookahead_distance: float = 3.0   # Look L meters ahead along the path
    
    # PPO Hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Training
    num_iterations: int = 120
    steps_per_iteration: int = 256
    update_epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 3e-4
    hidden_sizes: Tuple[int, int] = (128, 128)
    
    # Reward Weights
    # NOTE: No reward for braking itself. Brake is learned indirectly
    # to avoid overspeed penalty and irrecoverable conditions.
    w_track: float = 1.5        # Lateral error penalty (starts lower, curriculum applied)
    w_progress: float = 1.0     # Only for v_long > 0 (Projection)
    w_arc: float = 1.0          # Absolute arc-length progress
    w_forward: float = 0.3      # Only for v_long > 0
    w_steer: float = 0.2
    w_rate: float = 0.1
    w_speed: float = 0.3        # Penalty for exceeding target speed
    w_stuck: float = 1.0        # Penalty when v_long < 0.1
    w_heading: float = 0.5      # Heading error penalty
    # NOTE: w_backward removed - reverse is handled by done condition
    
    # Curriculum (iterations)
    curriculum_warmup_iters: int = 30
    curriculum_steps: int = 50000
    irrecoverable_curriculum_steps: int = 200000
    
    # Start Position Noise
    start_pos_noise: float = 0.3
    start_yaw_noise: float = 0.1
    
    # Done Thresholds
    off_track_distance: float = 2.0
    max_episode_length: int = 500
    reverse_velocity_threshold: float = -0.3  # v_long < this = done
    
    # Speed settings
    forward_speed_cap: float = 6.0
    target_speed: float = 5.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    reference_csv: str = "../drive_8_test.csv"
    checkpoint_dir: str = "./checkpoints"


# =============================================================================
# Utility Functions (Batched 3D Transformations)
# =============================================================================

def to_torch(x, device):
    """Convert numpy array to torch tensor on device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def to_numpy(tensor):
    """Convert tensor to numpy, handling both CPU and CUDA tensors."""
    if hasattr(tensor, 'cpu'):
        return tensor.cpu().numpy()
    elif hasattr(tensor, 'numpy'):
        return tensor.numpy()
    return np.array(tensor)


def quat_to_rotation_matrix_batch(quat: torch.Tensor) -> torch.Tensor:
    """
    Batch quaternion (N, 4) [w, x, y, z] to rotation matrices (N, 3, 3).
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Rotation matrix elements
    r00 = 1 - 2*(y*y + z*z)
    r01 = 2*(x*y - z*w)
    r02 = 2*(x*z + y*w)
    r10 = 2*(x*y + z*w)
    r11 = 1 - 2*(x*x + z*z)
    r12 = 2*(y*z - x*w)
    r20 = 2*(x*z - y*w)
    r21 = 2*(y*z + x*w)
    r22 = 1 - 2*(x*x + y*y)
    
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)
    
    return R


def transform_to_body_frame_batch(target_world: torch.Tensor, car_pos: torch.Tensor, 
                                   R_car: torch.Tensor) -> torch.Tensor:
    """
    Batch transform world coords to body frame.
    target_world: (N, 3), car_pos: (N, 3), R_car: (N, 3, 3)
    Returns: (N, 3)
    """
    delta = target_world - car_pos
    # R_car.T @ delta = (R_car^T @ delta.unsqueeze(-1)).squeeze(-1)
    target_rel = torch.bmm(R_car.transpose(1, 2), delta.unsqueeze(-1)).squeeze(-1)
    return target_rel


def compute_gravity_body_batch(R_car: torch.Tensor) -> torch.Tensor:
    """
    Batch compute gravity vector in body frame.
    R_car: (N, 3, 3)
    Returns: (N, 3)
    """
    device = R_car.device
    g_world = torch.tensor([0.0, 0.0, -1.0], device=device).unsqueeze(0)
    g_world = g_world.expand(R_car.shape[0], -1)
    g_body = torch.bmm(R_car.transpose(1, 2), g_world.unsqueeze(-1)).squeeze(-1)
    return g_body


def compute_velocity_body_batch(vel_world: torch.Tensor, R_car: torch.Tensor) -> torch.Tensor:
    """Batch transform world velocity to body frame."""
    return torch.bmm(R_car.transpose(1, 2), vel_world.unsqueeze(-1)).squeeze(-1)


def compute_angular_velocity_body_batch(ang_vel_world: torch.Tensor, R_car: torch.Tensor) -> torch.Tensor:
    """Batch transform world angular velocity to body frame."""
    return torch.bmm(R_car.transpose(1, 2), ang_vel_world.unsqueeze(-1)).squeeze(-1)


# =============================================================================
# Reference Trajectory (3D)
# =============================================================================

def load_reference_3d(csv_path: str, device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """Load 3D reference trajectory from CSV."""
    df = pd.read_csv(csv_path)
    
    required_cols = ["g_pos_x", "g_pos_y"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    pos_x = df["g_pos_x"].values.astype(np.float32)
    pos_y = df["g_pos_y"].values.astype(np.float32)
    
    if "g_pos_z" in df.columns:
        pos_z = df["g_pos_z"].values.astype(np.float32)
    else:
        pos_z = np.zeros_like(pos_x)
    
    path_points = np.stack([pos_x, pos_y, pos_z], axis=1)
    
    if "v_long" in df.columns:
        speed = df["v_long"].values.astype(np.float32)
    else:
        speed = np.ones(len(pos_x)) * 3.0
    
    print(f"Loaded 3D reference: {len(pos_x)} points")
    
    # Precompute arc-length for continuous progress reward
    path_tensor = torch.tensor(path_points, dtype=torch.float32, device=device)
    diffs = path_tensor[1:] - path_tensor[:-1]  # (M-1, 3)
    segment_lengths = torch.norm(diffs, dim=1)  # (M-1,)
    arc_length = torch.zeros(len(pos_x), dtype=torch.float32, device=device)
    arc_length[1:] = torch.cumsum(segment_lengths, dim=0)
    
    return {
        "path_points": path_tensor,
        "speed": torch.tensor(speed, dtype=torch.float32, device=device),
        "arc_length": arc_length,  # s[i] = cumulative distance to waypoint i
    }


# =============================================================================
# State-dependent Lookahead (Batched)
# =============================================================================

def compute_lookahead_idx_batch(v_long: torch.Tensor, base: int = 3, k_v: float = 0.5, 
                                 max_idx: int = 15) -> torch.Tensor:
    """Batch compute lookahead indices."""
    lookahead = base + k_v * torch.abs(v_long)
    return torch.clamp(lookahead.long(), max=max_idx)


def compute_target_waypoint_batch(pos_car: torch.Tensor, path_points: torch.Tensor,
                                   v_long: torch.Tensor, config: Config,
                                   arc_length: torch.Tensor,
                                   progress_idx: torch.Tensor = None,
                                   back_window: int = 10, forward_window: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch compute target waypoints using ARC-LENGTH based logic.
    
    Key difference from Fixed Frame Offset:
    - "항상 L미터 앞을 보라" (Always look L meters ahead)
    - 경로 밀도·속도·sampling과 무관 (Independent of path density/sampling)
    - target 이동 = progress에 연속적으로 종속 (Target movement is continuous)
    
    1. nearest_idx: Closest point in search window (for Progress/Done)
    2. target_idx: arc_length[nearest_idx] + lookahead_distance (Arc-length based)
    """
    N = pos_car.shape[0]
    M = path_points.shape[0]
    device = pos_car.device
    
    if progress_idx is None:
        # Fallback: global search
        distances = torch.norm(path_points.unsqueeze(0) - pos_car.unsqueeze(1), dim=2)
        nearest_idx = torch.argmin(distances, dim=1)
    else:
        # Window search logic (same as before)
        idx_min = torch.clamp(progress_idx - back_window, min=0)
        idx_max = torch.clamp(progress_idx + forward_window, max=M - 1)
        
        nearest_idx = torch.zeros(N, dtype=torch.long, device=device)
        
        # Determine nearest within window
        for i in range(N):
            start = idx_min[i].item()
            end = idx_max[i].item() + 1
            segment = path_points[start:end]  # (K, 3)
            dists = torch.norm(segment - pos_car[i], dim=1)
            local_min = torch.argmin(dists)
            nearest_idx[i] = start + local_min
    
    # ===== ARC-LENGTH BASED LOOKAHEAD =====
    # s_nearest: current arc-length position (N,)
    s_nearest = arc_length[nearest_idx]
    
    # s_target: "Look L meters ahead along the path" (N,)
    s_target = s_nearest + config.lookahead_distance
    
    # Find index where arc_length >= s_target using searchsorted
    target_idx = torch.searchsorted(arc_length, s_target, side='left')
    target_idx = torch.clamp(target_idx, 0, M - 1)
    
    target_pos = path_points[target_idx]
    
    return target_idx, target_pos, nearest_idx


def compute_tangent_direction_batch(path_points: torch.Tensor, nearest_idx: torch.Tensor,
                                     R_car: torch.Tensor) -> torch.Tensor:
    """
    Batch compute path tangent direction in body frame.
    """
    N = nearest_idx.shape[0]
    M = path_points.shape[0]
    device = path_points.device
    
    next_idx = torch.clamp(nearest_idx + 1, max=M - 1)
    
    tangent_world = path_points[next_idx] - path_points[nearest_idx]
    norm = torch.norm(tangent_world, dim=1, keepdim=True)
    tangent_world = torch.where(norm > 1e-6, tangent_world / norm, 
                                 torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(N, -1))
    
    tangent_rel = torch.bmm(R_car.transpose(1, 2), tangent_world.unsqueeze(-1)).squeeze(-1)
    return tangent_rel


# =============================================================================
# Observation Building (Batched, 18D)
# =============================================================================

def compute_slip_proxy_batch(throttle_prev: torch.Tensor, v_long: torch.Tensor,
                              max_speed: float = 10.0) -> torch.Tensor:
    """Batch compute slip proxy."""
    v_norm = v_long / max_speed
    slip = throttle_prev - v_norm
    return torch.clamp(slip, -1.0, 1.0)


def build_observation_batch(target_rel: torch.Tensor, v_body: torch.Tensor,
                             omega_body: torch.Tensor, g_body: torch.Tensor,
                             tangent_rel: torch.Tensor, slip_proxy: torch.Tensor,
                             prev_action: torch.Tensor) -> torch.Tensor:
    """
    Build observation batch. (N, 18)
    """
    obs = torch.cat([
        target_rel,                     # (N, 3)
        v_body,                         # (N, 3)
        omega_body,                     # (N, 3)
        g_body,                         # (N, 3)
        tangent_rel,                    # (N, 3)
        slip_proxy.unsqueeze(-1),       # (N, 1)
        prev_action,                    # (N, 2)
    ], dim=-1)
    return obs  # (N, 18)


# =============================================================================
# Reward Functions (Batched)
# =============================================================================

class CurriculumScheduler:
    """Curriculum Learning scheduler with warmup for progress/rate penalties."""
    
    def __init__(self, total_steps: int = 50000, warmup_iters: int = 30):
        self.total_steps = total_steps
        self.warmup_iters = warmup_iters
    
    def get_w_dir(self, iteration: int) -> float:
        """Direction reward weight: 0.1 → 0.3"""
        progress = min(iteration / 100, 1.0)  # ~100 iterations to full
        return 0.1 + 0.2 * progress
    
    def process_R_dir(self, R_dir: torch.Tensor, iteration: int) -> torch.Tensor:
        """Clamp R_dir early in training."""
        if iteration < 50:
            return torch.clamp(R_dir, 0, 1)
        return R_dir
    
    def get_progress_weight(self, iteration: int) -> float:
        """Progress reward weight: 0 for first warmup_iters, then ramp to 1.0"""
        if iteration < self.warmup_iters:
            return 0.0
        # Ramp from 0 to 1 over next warmup_iters
        ramp = min((iteration - self.warmup_iters) / self.warmup_iters, 1.0)
        return ramp
    
    def get_rate_weight(self, iteration: int) -> float:
        """Rate penalty weight: 0 for first warmup_iters, then ramp to 1.0"""
        if iteration < self.warmup_iters:
            return 0.0
        ramp = min((iteration - self.warmup_iters) / self.warmup_iters, 1.0)
        return ramp
    
    def get_brake_multiplier(self, iteration: int) -> float:
        """Brake torque curriculum: 0.5 at start, ramps to 1.0
        
        Design: Start with limited brake to encourage forward motion.
        As training progresses, allow full braking for speed control.
        """
        # Ramp from 0.5 to 1.0 over first 60 iterations
        ramp = min(iteration / 60.0, 1.0)
        return 0.5 + 0.5 * ramp


def compute_tracking_reward_batch(target_rel: torch.Tensor, v_body: torch.Tensor,
                                   step: int, scheduler: CurriculumScheduler) -> torch.Tensor:
    """Batch tracking reward."""
    w_dist = 1.0
    w_dir = scheduler.get_w_dir(step)
    
    # Distance reward
    d = torch.norm(target_rel, dim=1)
    R_dist = torch.exp(-d)
    
    # Direction reward (XY plane only for 3D stability)
    v_xy = v_body[:, :2]
    t_xy = target_rel[:, :2]
    
    speed_xy = torch.norm(v_xy, dim=1)
    t_norm_xy = torch.norm(t_xy, dim=1)
    
    v_hat = v_xy / (speed_xy.unsqueeze(-1) + 1e-6)
    t_hat = t_xy / (t_norm_xy.unsqueeze(-1) + 1e-6)
    
    R_dir = torch.sum(v_hat * t_hat, dim=1)
    R_dir = torch.where(speed_xy > 0.1, R_dir, torch.zeros_like(R_dir))
    R_dir = scheduler.process_R_dir(R_dir, step)
    
    return w_dist * R_dist + w_dir * R_dir


def compute_progress_reward_batch(d_prev: torch.Tensor, d_curr: torch.Tensor,
                                   delta_max: float = 0.5) -> torch.Tensor:
    """Batch progress reward."""
    progress = d_prev - d_curr
    return torch.clamp(progress, -delta_max, delta_max)


def compute_total_reward_batch(target_rel: torch.Tensor, v_body: torch.Tensor,
                                steer: torch.Tensor, steer_prev: torch.Tensor,
                                iteration: int, scheduler: CurriculumScheduler,
                                config: Config,
                                tangent_rel: torch.Tensor,
                                prev_lat_error: torch.Tensor,
                                arc_length: torch.Tensor,
                                progress_idx: torch.Tensor,
                                prev_progress_idx: torch.Tensor,
                                nearest_rel: torch.Tensor) -> torch.Tensor:
    """
    Batch total reward computation with Explicit Steering Strategy.
    
    1. R_align (1st Order): "Steer towards the target" (Lookahead)
       - target_rel.y * steer
       
    2. R_recover (2nd Order): "Did you reduce the error?" (Current)
       - |e_prev| - |e_curr| (using nearest_rel)
       
    3. R_proj & R_arc (Dual Progress): "Move forward"
    """
    # 1. Alignment Reward (1st Order Steering Command)
    # Target is at target_rel (Body Frame).
    # If target is Left (y>0), Steer Left (>0) -> Reward > 0
    # Increased clamp range for stronger steering signal
    R_align = torch.clamp(target_rel[:, 1] * steer, -1.0, 1.0)
    
    # 2. Recovery Reward (2nd Order Steering Command)
    # validation: did current error decrease?
    # Use NEAREST_REL for current error (true cross track error)
    curr_lat_error_abs = torch.abs(nearest_rel[:, 1])
    error_diff = prev_lat_error - curr_lat_error_abs
    R_recover = torch.clamp(error_diff, -0.2, 0.2)
    
    # 3. Progress Rewards
    # 3.1 Projection (Speed/Dir) relative to Path
    proj_speed = v_body[:, 0] * tangent_rel[:, 0] + v_body[:, 1] * tangent_rel[:, 1]
    R_proj = torch.clamp(proj_speed / config.target_speed, -0.2, 0.5)
    
    # 3.2 Arc-length (Absolute Mileage) - PREVENTS STOPPING
    s_curr = arc_length[progress_idx]
    s_prev = arc_length[prev_progress_idx]
    R_arc = torch.clamp(s_curr - s_prev, min=0.0, max=0.5)
    
    # 4. Forward reward (Auxiliary)
    v_long = v_body[:, 0]
    R_forward = torch.tanh(v_long)
    
    # Penalties
    P_steer = steer ** 2
    P_rate = torch.abs(steer - steer_prev)
    P_speed = torch.clamp(v_long - config.target_speed, min=0.0)
    P_stuck = torch.clamp(0.1 - v_long, min=0.0)
    
    # CRITICAL: Absolute Lateral Error Penalty (SQUARED for smoother learning)
    # "붙어서 가는 게 무조건 이득" 구조
    # Squared form: small errors penalized less, large errors capped
    lat_err_clamped = torch.clamp(torch.abs(nearest_rel[:, 1]), 0.0, 2.0)
    P_lat = lat_err_clamped ** 2  # Max = 4.0
    
    # Curriculum weights
    w_progress = config.w_progress * scheduler.get_progress_weight(iteration)
    w_arc = config.w_arc * scheduler.get_progress_weight(iteration) 
    w_rate = config.w_rate * scheduler.get_rate_weight(iteration)
    
    # w_track curriculum: ramp from 0.5x to 1.0x over warmup period
    w_track_mult = 0.5 + 0.5 * scheduler.get_progress_weight(iteration)
    w_track_curr = config.w_track * w_track_mult
    
    # R_recover is critical
    w_recover = 2.0
    
    total_reward = (
        R_align
        + w_recover * R_recover
        + w_progress * R_proj
        + w_arc * R_arc
        + config.w_forward * R_forward
        - w_track_curr * P_lat       # Absolute lateral error penalty (curriculum)
        - config.w_steer * P_steer
        - w_rate * P_rate
        - config.w_speed * P_speed
        - config.w_stuck * P_stuck
    )
    
    return total_reward


# =============================================================================
# Done Conditions (Batched)
# =============================================================================

def get_irrecoverable_thresholds(step: int, curriculum_steps: int = 200000) -> dict:
    """Get curriculum-adjusted thresholds."""
    progress = min(step / curriculum_steps, 1.0)
    y_max = 5.0 - 2.0 * progress
    omega_max = 3.0 - 1.0 * progress
    return {'y_max': y_max, 'omega_max': omega_max}


def check_done_batch(pos_car: torch.Tensor, path_points: torch.Tensor,
                      target_idx: torch.Tensor, nearest_idx: torch.Tensor,
                      nearest_rel: torch.Tensor,
                      v_body: torch.Tensor, omega_body: torch.Tensor,
                      episode_lengths: torch.Tensor, global_step: int,
                      config: Config) -> torch.Tensor:
    """
    Batch done condition check.
    Returns: done_mask (N,) boolean tensor
    
    Uses nearest_rel for irrecoverable check (consistent with reward).
    Uses nearest_idx for off-track check (optimized, no global search).
    """
    N = pos_car.shape[0]
    M = path_points.shape[0]
    device = pos_car.device
    
    done = torch.zeros(N, dtype=torch.bool, device=device)
    
    # 1. Goal reached
    done = done | (target_idx >= M - 1)
    
    # 2. Off-track (OPTIMIZED: use nearest_idx instead of global search)
    # min_dist ≈ norm(pos - path_points[nearest_idx])
    nearest_pos = path_points[nearest_idx]
    min_dist = torch.norm(pos_car - nearest_pos, dim=1)
    done = done | (min_dist > config.off_track_distance)
    
    # 3. Reverse Motion = Invalid State (not penalized, just terminated)
    done = done | (v_body[:, 0] < config.reverse_velocity_threshold)
    
    # 4. Irrecoverable (curriculum) - USE nearest_rel for true cross-track error
    thresholds = get_irrecoverable_thresholds(global_step, config.irrecoverable_curriculum_steps)
    lateral_error = torch.abs(nearest_rel[:, 1])  # Fixed: use nearest_rel
    yaw_rate = torch.abs(omega_body[:, 2])
    irrecoverable = (lateral_error > thresholds['y_max']) & (yaw_rate > thresholds['omega_max'])
    done = done | irrecoverable
    
    # 5. Timeout
    done = done | (episode_lengths >= config.max_episode_length)
    
    return done


# =============================================================================
# Neural Networks
# =============================================================================

class ActorNetwork(nn.Module):
    """Actor (Policy Network) for Direct Control."""
    
    def __init__(self, obs_dim: int = 18, action_dim: int = 2,
                 hidden_sizes: Tuple[int, int] = (128, 128)):
        super().__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.mean = nn.Linear(hidden_sizes[1], action_dim)
        
        # Separate log_std: [throttle, steer] - lower initial variation
        self.log_std = nn.Parameter(torch.tensor([-1.0, -2.0]))
        
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    def get_dist(self, obs: torch.Tensor) -> Normal:
        mean, std = self(obs)
        return Normal(mean, std)


class CriticNetwork(nn.Module):
    """Critic (Value Network)."""
    
    def __init__(self, obs_dim: int = 18,
                 hidden_sizes: Tuple[int, int] = (128, 128)):
        super().__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.value = nn.Linear(hidden_sizes[1], 1)
        
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.value(x)


class ActorCritic(nn.Module):
    """PPO Actor-Critic for Direct Control."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = config.device
        
        obs_dim = 18
        action_dim = 2
        
        self.actor = ActorNetwork(obs_dim, action_dim, config.hidden_sizes).to(self.device)
        self.critic = CriticNetwork(obs_dim, config.hidden_sizes).to(self.device)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple:
        """
        Get action for batch of observations.
        obs: (N, 18)
        Returns: action (N, 2), log_prob (N,), value (N,), raw_action (N, 2)
        """
        dist = self.actor.get_dist(obs)
        value = self.critic(obs).squeeze(-1)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(-1)
        action_squashed = torch.tanh(action)
        log_prob = log_prob - torch.log(1 - action_squashed.pow(2) + 1e-6).sum(-1)
        
        return action_squashed, log_prob, value, action
    
    def evaluate_actions(self, obs_batch: torch.Tensor,
                          raw_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        dist = self.actor.get_dist(obs_batch)
        values = self.critic(obs_batch).squeeze(-1)
        
        log_prob = dist.log_prob(raw_actions).sum(-1)
        action_squashed = torch.tanh(raw_actions)
        log_prob = log_prob - torch.log(1 - action_squashed.pow(2) + 1e-6).sum(-1)
        
        entropy = dist.entropy().sum(-1)
        
        return log_prob, entropy, values


# =============================================================================
# Genesis Multi-Env Simulation
# =============================================================================

def initialize_simulation_multienv(config: Config, start_pos: np.ndarray,
                                    start_quat: np.ndarray, headless: bool = True) -> Tuple:
    """Initialize Genesis simulation with multiple environments."""
    print(f"\nInitializing Genesis simulation with {config.num_envs} environments...")
    
    backend = gs.gpu if torch.cuda.is_available() else gs.cpu
    gs.init(backend=backend, debug=False, logging_level='warning')
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=config.dt,
            substeps=config.substeps,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[0] if not headless else None,
        ),
        show_viewer=not headless,
    )
    
    scene.add_entity(gs.morphs.Plane())
    
    car = scene.add_entity(
        gs.morphs.URDF(file="../URDF/test_v1_simple.urdf"),
    )
    
    # Build with multiple environments
    scene.build(n_envs=config.num_envs)
    
    # Get DOF indices
    fl_idx = int(car.get_joint("front_left_steer_joint").dofs_idx_local[0])
    fr_idx = int(car.get_joint("front_right_steer_joint").dofs_idx_local[0])
    rl_idx = int(car.get_joint("rear_left_wheel_joint").dofs_idx_local[0])
    rr_idx = int(car.get_joint("rear_right_wheel_joint").dofs_idx_local[0])
    dof_indices = (fl_idx, fr_idx, rl_idx, rr_idx)
    
    # Set PD gains (same for all environments, use 1D array)
    car.set_dofs_kp(
        kp=np.array([4000.0, 4000.0], dtype=np.float32),
        dofs_idx_local=np.array([fl_idx, fr_idx])
    )
    car.set_dofs_kv(
        kv=np.array([100.0, 100.0], dtype=np.float32),
        dofs_idx_local=np.array([fl_idx, fr_idx])
    )
    car.set_dofs_force_range(
        lower=np.array([-400.0, -400.0], dtype=np.float32),
        upper=np.array([400.0, 400.0], dtype=np.float32),
        dofs_idx_local=np.array([fl_idx, fr_idx])
    )
    
    # Rear wheel DOFs: VELOCITY control needs high kv for 1200 kg car
    # kp=0 (no position hold), kv=200 (velocity servo gain)
    car.set_dofs_kp(
        kp=np.array([0.0, 0.0], dtype=np.float32),
        dofs_idx_local=np.array([rl_idx, rr_idx])
    )
    car.set_dofs_kv(
        kv=np.array([200.0, 200.0], dtype=np.float32),  # High kv for heavy car
        dofs_idx_local=np.array([rl_idx, rr_idx])
    )
    car.set_dofs_force_range(
        lower=np.array([-2000.0, -2000.0], dtype=np.float32),  # Increased limit
        upper=np.array([2000.0, 2000.0], dtype=np.float32),
        dofs_idx_local=np.array([rl_idx, rr_idx])
    )
    
    # Set initial positions for all environments
    N = config.num_envs
    start_pos_batch = np.tile(start_pos, (N, 1)).astype(np.float32)
    start_quat_batch = np.tile(start_quat, (N, 1)).astype(np.float32)
    car.set_pos(start_pos_batch)
    car.set_quat(start_quat_batch)
    
    print(f"Simulation initialized with {config.num_envs} parallel environments!")
    return scene, car, dof_indices


def get_car_state_batch(car, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get batched car state from Genesis.
    genesis.Tensor is a subclass of torch.Tensor, so we can use it directly.
    Returns: pos (N, 3), quat (N, 4), vel (N, 3), ang_vel (N, 3)
    """
    # genesis.Tensor ⊂ torch.Tensor, already on GPU
    pos = car.get_pos()
    quat = car.get_quat()
    vel = car.get_vel()
    ang_vel = car.get_ang()
    
    return pos, quat, vel, ang_vel


def apply_control_batch(car, accel_brake: torch.Tensor, steer: torch.Tensor,
                         dof_indices: tuple, config: Config,
                         iteration: int = 0, scheduler: CurriculumScheduler = None):
    """
    Apply batched control to all environments.
    
    Action Space:
        accel_brake: (N,) in [-1, 1]
            > 0: Engine torque (acceleration)
            < 0: Brake (reduces forward torque, NEVER creates reverse)
        steer: (N,) in [-1, 1]
    
    Design Philosophy:
        - Braking is not a rewarded action but a means to avoid penalties.
        - drive_torque = clamp(engine_torque - brake_torque, min=0)
        - Negative torque (reverse force) is structurally blocked.
    """
    fl_idx, fr_idx, rl_idx, rr_idx = dof_indices
    N = accel_brake.shape[0]
    
    # Convert to numpy for Genesis API
    actual_steer = (steer * config.max_steer).cpu().numpy()
    
    # Brake curriculum: start with limited braking, increase over iterations
    brake_mult = scheduler.get_brake_multiplier(iteration) if scheduler else 1.0
    
    # Accel/Brake logic:
    # action > 0: engine torque = action * max_engine_torque
    # action < 0: brake torque = |action| * max_brake_torque * curriculum_mult
    # Final drive_torque = max(engine - brake, 0)  -- no reverse!
    accel_brake_np = accel_brake.cpu().numpy()
    
    engine_torque = np.maximum(accel_brake_np, 0.0) * config.max_engine_torque
    brake_torque = np.maximum(-accel_brake_np, 0.0) * config.max_brake_torque * brake_mult
    
    # Drive torque: engine - brake, clamped to non-negative (no reverse)
    # Drive torque: engine - brake, clamped to non-negative (no reverse)
    drive_torque = np.maximum(engine_torque - brake_torque, 0.0)
    
    # Fix #3: Minimum target velocity (Idle Creep)
    # If drive_torque is tiny, we still want to target at least 0.3 m/s
    # to avoid "stuck due to friction" and help exploration.
    target_v_raw = (drive_torque / config.max_engine_torque) * config.target_speed
    target_v = np.maximum(target_v_raw, 0.3)
    
    # Steering (position control): (N, 2)
    steer_cmd = np.stack([actual_steer, actual_steer], axis=1).astype(np.float32)
    car.control_dofs_position(
        position=steer_cmd,
        dofs_idx_local=np.array([fl_idx, fr_idx])
    )
    
    # Throttle (VELOCITY control): (N, 2)
    # Convert drive_torque to target angular velocity
    # target_omega = (drive_torque / max_torque) * max_speed / wheel_radius
    WHEEL_RADIUS = 0.358  # From URDF
    target_v_raw = (drive_torque / config.max_engine_torque) * config.target_speed
    target_v = np.maximum(target_v_raw, 0.3)  # Idle Creep (Fix #3)
    target_omega = target_v / WHEEL_RADIUS
    
    velocity_cmd = np.stack([target_omega, target_omega], axis=1).astype(np.float32)
    car.control_dofs_velocity(
        velocity=velocity_cmd,
        dofs_idx_local=np.array([rl_idx, rr_idx])
    )


def reset_envs(car, envs_idx: torch.Tensor, start_pos: np.ndarray,
               start_quat: np.ndarray, config: 'Config'):
    """Reset specific environments to initial state with noise for diversity."""
    if len(envs_idx) == 0:
        return
    
    N = len(envs_idx)
    
    # Position noise (XY only, Z fixed)
    pos_reset = np.tile(start_pos, (N, 1)).astype(np.float32)
    pos_reset[:, 0:2] += np.random.normal(
        0, config.start_pos_noise, size=(N, 2)
    ).astype(np.float32)
    
    # Yaw noise quaternion * start_quat (with normalization)
    quat_reset = np.zeros((N, 4), dtype=np.float32)
    yaw_noise = np.random.normal(0, config.start_yaw_noise, size=N).astype(np.float32)
    
    for i in range(N):
        yaw = float(yaw_noise[i])
        q_yaw = np.array([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)], dtype=np.float32)
        q = quat_multiply(q_yaw, start_quat)
        quat_reset[i] = quat_normalize(q).astype(np.float32)
    
    car.set_pos(pos_reset, envs_idx=envs_idx.cpu().numpy(), zero_velocity=True)
    car.set_quat(quat_reset, envs_idx=envs_idx.cpu().numpy(), zero_velocity=True)
    
    # Zero all DOF velocities for clean reset (API best practice)
    car.zero_all_dofs_velocity(envs_idx=envs_idx.cpu().numpy())


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    return q / (np.linalg.norm(q) + 1e-8)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (w, x, y, z) convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


# =============================================================================
# PPO Training (Multi-Env)
# =============================================================================

def compute_gae_batch(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                       gamma: float, lam: float, last_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE for all environments.
    rewards, values, dones: (T, N)
    last_values: (N,)
    Returns: advantages (T, N), returns (T, N)
    """
    T, N = rewards.shape
    device = rewards.device
    
    advantages = torch.zeros(T, N, device=device)
    gae = torch.zeros(N, device=device)
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_values
            next_non_terminal = 1.0 - dones[t].float()
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t].float()
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
    
    returns = advantages + values
    return advantages, returns


def train(config: Config, headless: bool = True):
    """Main multi-env training loop."""
    device = config.device
    
    # Load reference
    reference = load_reference_3d(config.reference_csv, device)
    path_points = reference["path_points"]
    arc_length = reference["arc_length"]
    
    # Get start position
    start_pos = to_numpy(path_points[0].cpu()).copy()
    start_pos[2] = 0.1
    
    # Calculate start heading
    if len(path_points) > 5:
        tangent = to_numpy((path_points[5] - path_points[0]).cpu())
        start_heading = math.atan2(tangent[1], tangent[0])
    else:
        start_heading = 0.0
    
    start_quat = np.array([
        math.cos(start_heading / 2), 0, 0, math.sin(start_heading / 2)
    ], dtype=np.float32)
    
    print(f"Start position: {start_pos}")
    print(f"Start heading: {start_heading:.4f} rad")
    
    # Initialize simulation
    scene, car, dof_indices = initialize_simulation_multienv(
        config, start_pos, start_quat, headless
    )
    
    # Create model and optimizer
    model = ActorCritic(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Curriculum scheduler
    scheduler = CurriculumScheduler(config.curriculum_steps, config.curriculum_warmup_iters)
    
    N = config.num_envs
    T = config.steps_per_iteration
    
    print("\n" + "=" * 60)
    print("Starting Multi-Env PPO Training (Direct Control)")
    print(f"  Environments: {N}")
    print(f"  Steps per iteration: {T}")
    print(f"  Samples per iteration: {N * T}")
    print(f"  Iterations: {config.num_iterations}")
    print("=" * 60)
    
    best_avg_reward = -float('inf')
    global_step = 0
    
    # Initialize environment states
    prev_actions = torch.zeros(N, 2, device=device)
    episode_lengths = torch.zeros(N, dtype=torch.long, device=device)
    prev_d_targets = torch.zeros(N, device=device)
    
    # Auto-creep: minimum throttle for N steps after reset to get car moving
    WARMUP_STEPS = 10
    MIN_THROTTLE = 0.3
    warmup_steps = torch.zeros(N, dtype=torch.long, device=device)
    
    # Progress-based target: per-env monotonically increasing index
    progress_idx = torch.zeros(N, dtype=torch.long, device=device)
    prev_progress_idx = torch.zeros(N, dtype=torch.long, device=device)
    
    # Stuck detection: consecutive frames with v_long < threshold
    # Stuck detection: consecutive frames with v_long < threshold
    STUCK_THRESHOLD = 0.1   # m/s (increased from 0.05)
    STUCK_MAX_STEPS = 40    # steps (decreased from 80) -> 0.4s
    stuck_counter = torch.zeros(N, dtype=torch.long, device=device)
    
    # State for Recovery Reward (prev lateral error)
    prev_lat_error = torch.zeros(N, device=device)
    
    # Initial position with noise for diversity from the start
    all_idx = torch.arange(N, device=device)
    reset_envs(car, all_idx, start_pos, start_quat, config)
    warmup_steps[:] = WARMUP_STEPS  # Start all envs with warmup
    progress_idx[:] = 0  # Reset progress
    prev_lat_error[:] = 0.0 # Reset prev error
    
    for iteration in range(config.num_iterations):
        # Rollout storage
        obs_buffer = torch.zeros(T, N, 18, device=device)
        action_buffer = torch.zeros(T, N, 2, device=device)
        raw_action_buffer = torch.zeros(T, N, 2, device=device)
        log_prob_buffer = torch.zeros(T, N, device=device)
        value_buffer = torch.zeros(T, N, device=device)
        reward_buffer = torch.zeros(T, N, device=device)
        done_buffer = torch.zeros(T, N, dtype=torch.bool, device=device)
        
        # Logging stats
        total_rewards = torch.zeros(N, device=device)
        steer_sum = torch.zeros(N, device=device)
        throttle_sum = torch.zeros(N, device=device)
        lateral_error_sum = torch.zeros(N, device=device)
        done_count = 0
        off_track_count = 0
        timeout_count = 0
        
        for t in range(T):
            # Get batched state
            pos, quat, vel_world, ang_vel_world = get_car_state_batch(car, device)
            R_car = quat_to_rotation_matrix_batch(quat)
            
            # Transform to body frame
            v_body = compute_velocity_body_batch(vel_world, R_car)
            omega_body = compute_angular_velocity_body_batch(ang_vel_world, R_car)
            g_body = compute_gravity_body_batch(R_car)
            
            v_long = v_body[:, 0]
            
            # Get target waypoints using progress-based search window
            target_idx, target_pos, nearest_idx = compute_target_waypoint_batch(
                pos, path_points, v_long, config, arc_length, progress_idx=progress_idx
            )
            
            # Update progress monotonically (never decreases)
            prev_progress_idx = progress_idx.clone()
            progress_idx = torch.maximum(progress_idx, nearest_idx)
            
            target_rel = transform_to_body_frame_batch(target_pos, pos, R_car)
            
            # Nearest Rel for Reward (Recovery & Align Check)
            nearest_pos = path_points[nearest_idx]
            nearest_rel = transform_to_body_frame_batch(nearest_pos, pos, R_car)
            
            tangent_rel = compute_tangent_direction_batch(path_points, nearest_idx, R_car)
            
            # Slip proxy
            slip_proxy = compute_slip_proxy_batch(prev_actions[:, 0], v_long, config.max_speed)
            
            # Build observation
            obs = build_observation_batch(
                target_rel, v_body, omega_body, g_body, tangent_rel,
                slip_proxy, prev_actions
            )
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value, raw_action = model.get_action(obs)
            
            # Action interpretation:
            # action[0] in [-1, 1]: accel (>0) / brake (<0)
            # action[1] in [-1, 1]: steer
            accel_brake = action[:, 0]
            steer = action[:, 1]
            
            # Auto-creep: enforce minimum throttle during warmup
            warmup_mask = warmup_steps > 0
            accel_brake = torch.where(
                warmup_mask,
                torch.maximum(accel_brake, torch.tensor(MIN_THROTTLE, device=device)),
                accel_brake
            )
            warmup_steps = torch.clamp(warmup_steps - 1, min=0)
            
            # Apply control (accel/brake handled internally with brake curriculum)
            apply_control_batch(car, accel_brake, steer, dof_indices, config, iteration, scheduler)
            scene.step()
            global_step += N
            episode_lengths += 1
            
            # Compute reward
            d_curr = torch.norm(target_rel, dim=1)
            d_prev = torch.where(prev_d_targets > 0, prev_d_targets, d_curr)
            
            reward = compute_total_reward_batch(
                target_rel, v_body, steer, prev_actions[:, 1],
                iteration, scheduler, config,
                tangent_rel, prev_lat_error,
                arc_length, progress_idx, prev_progress_idx,
                nearest_rel # Passed for R_recover
            )
            
            # Update prev_lat_error for next step (Recovery Reward)
            # Use detached magnitude of CURRENT lateral error (nearest_rel)
            prev_lat_error = torch.abs(nearest_rel[:, 1]).detach()
            
            # Update stuck counter (for stuck termination)
            is_stuck = v_long < STUCK_THRESHOLD
            stuck_counter = torch.where(is_stuck, stuck_counter + 1, torch.zeros_like(stuck_counter))
            
            # Check done (including stuck and sink termination)
            done = check_done_batch(
                pos, path_points, target_idx, nearest_idx, nearest_rel,
                v_body, omega_body, episode_lengths, global_step, config
            )
            
            # Strict Heading Termination: If car faces backward relative to path tangent
            # tangent_rel is in Body Frame. x < 0 means > 90 deg error.
            bad_heading = tangent_rel[:, 0] < 0.0
            done = done | bad_heading
            
            # Penalize bad heading termination
            reward = torch.where(bad_heading, reward - 2.0, reward)
            
            # Add stuck termination (40+ consecutive steps stuck)
            stuck_done = stuck_counter >= STUCK_MAX_STEPS
            
            # Update done flag
            done = done | stuck_done
            
            # Fix #2: Termination Penalty for Stuck
            reward = torch.where(stuck_done, reward - 2.0, reward)
            
            # Add sink/fly termination (z out of bounds)
            z = pos[:, 2]
            sink_fly_done = (z < -0.2) | (z > 2.0)
            
            # Combine all done conditions
            done = done | stuck_done | sink_fly_done
            
            # Store
            obs_buffer[t] = obs
            action_buffer[t] = action
            raw_action_buffer[t] = raw_action
            log_prob_buffer[t] = log_prob
            value_buffer[t] = value
            reward_buffer[t] = reward
            done_buffer[t] = done
            
            total_rewards += reward
            
            # Accumulate logging stats
            steer_sum += torch.abs(steer)
            throttle_sum += torch.abs(accel_brake)  # Fixed: was 'throttle' which doesn't exist
            lateral_error_sum += torch.abs(nearest_rel[:, 1])  # Use nearest_rel (consistent with reward)
            
            # Update prev states
            prev_actions = action.clone()
            prev_d_targets = d_curr.clone()
            
            # Reset done environments and count done reasons
            done_idx = torch.where(done)[0]
            if len(done_idx) > 0:
                done_count += len(done_idx)
                
                # Count done reasons (check off-track)
                for idx in done_idx:
                    distances = torch.norm(path_points - pos[idx:idx+1], dim=1)
                    min_dist = torch.min(distances).item()
                    if min_dist > config.off_track_distance:
                        off_track_count += 1
                    elif episode_lengths[idx] >= config.max_episode_length:
                        timeout_count += 1
                
            # Reset environments that are done
            if done_idx.numel() > 0:
                reset_envs(car, done_idx, start_pos, start_quat, config)
                episode_lengths[done_idx] = 0
                prev_actions[done_idx] = 0
                warmup_steps[done_idx] = WARMUP_STEPS
                prev_d_targets[done_idx] = 0
                progress_idx[done_idx] = 0
                prev_progress_idx[done_idx] = 0
                stuck_counter[done_idx] = 0
                prev_lat_error[done_idx] = 0.0  # Reset prev error state
        
        # Compute last values for bootstrapping
        with torch.no_grad():
            pos, quat, vel_world, ang_vel_world = get_car_state_batch(car, device)
            R_car = quat_to_rotation_matrix_batch(quat)
            v_body = compute_velocity_body_batch(vel_world, R_car)
            omega_body = compute_angular_velocity_body_batch(ang_vel_world, R_car)
            g_body = compute_gravity_body_batch(R_car)
            v_long = v_body[:, 0]
            
            target_idx, target_pos, nearest_idx = compute_target_waypoint_batch(
                pos, path_points, v_long, config, arc_length, progress_idx=progress_idx
            )
            target_rel = transform_to_body_frame_batch(target_pos, pos, R_car)
            tangent_rel = compute_tangent_direction_batch(path_points, nearest_idx, R_car)
            slip_proxy = compute_slip_proxy_batch(prev_actions[:, 0], v_long, config.max_speed)
            
            obs_last = build_observation_batch(
                target_rel, v_body, omega_body, g_body, tangent_rel,
                slip_proxy, prev_actions
            )
            last_values = model.critic(obs_last).squeeze(-1)
        
        # Compute GAE
        advantages, returns = compute_gae_batch(
            reward_buffer, value_buffer, done_buffer,
            config.gamma, config.gae_lambda, last_values
        )
        
        # Flatten for PPO update
        obs_flat = obs_buffer.reshape(-1, 18)
        raw_actions_flat = raw_action_buffer.reshape(-1, 2)
        old_log_probs_flat = log_prob_buffer.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        
        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        # PPO update with mini-batches
        total_samples = obs_flat.shape[0]
        batch_size = config.batch_size
        
        for epoch in range(config.update_epochs):
            # Shuffle indices for mini-batch sampling
            indices = torch.randperm(total_samples, device=config.device)
            
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                batch_idx = indices[start:end]
                
                obs_batch = obs_flat[batch_idx]
                raw_actions_batch = raw_actions_flat[batch_idx]
                old_log_probs_batch = old_log_probs_flat[batch_idx]
                advantages_batch = advantages_flat[batch_idx]
                returns_batch = returns_flat[batch_idx]
                
                new_log_probs, entropies, new_values = model.evaluate_actions(obs_batch, raw_actions_batch)
                
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                clipped_ratio = torch.clamp(ratio, 1 - config.epsilon_clip, 1 + config.epsilon_clip)
                policy_loss = -torch.min(ratio * advantages_batch, clipped_ratio * advantages_batch).mean()
                
                value_loss = F.mse_loss(new_values, returns_batch)
                entropy_mean = entropies.mean()
                
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        
        avg_reward = total_rewards.mean().item() / T
        
        # Compute logging metrics
        mean_steer = (steer_sum.mean().item() / T)
        mean_throttle = (throttle_sum.mean().item() / T)
        mean_lat_error = (lateral_error_sum.mean().item() / T)
        done_rate = done_count / (N * T) * 100  # percentage
        off_track_rate = off_track_count / max(done_count, 1) * 100
        timeout_rate = timeout_count / max(done_count, 1) * 100
        
        # Logging (every 5 iterations: summary, every iteration: debug)
        # DEBUG LOG (every iteration)
        max_lat_err = lateral_error_sum.max().item() / T
        print(f"[{iteration+1:3d}] R={avg_reward:.3f} LatErr={mean_lat_error:.3f}(max={max_lat_err:.2f}) "
              f"Steer={mean_steer:.3f} Done={done_rate:.1f}%")
        
        # SUMMARY LOG (every 5 iterations)
        if (iteration + 1) % 5 == 0:
            print(f"  ▶ Iter {iteration + 1}/{config.num_iterations}: "
                  f"R={avg_reward:.3f} | "
                  f"Steer={mean_steer:.3f} Thr={mean_throttle:.3f} | "
                  f"LatErr={mean_lat_error:.3f} | "
                  f"Done={done_rate:.1f}% (off={off_track_rate:.0f}% to={timeout_rate:.0f}%) | "
                  f"Loss={loss.item():.3f}")
        
        # Save best checkpoint (no early stopping)
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save({
                'model_actor': model.actor.state_dict(),
                'model_critic': model.critic.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration,
                'avg_reward': avg_reward,
                'global_step': global_step,
            }, os.path.join(config.checkpoint_dir, 'best_ppo_multienv.pth'))
            print(f"  ★ New best checkpoint saved! R={avg_reward:.4f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Average Reward: {best_avg_reward:.4f}")
    print(f"Total Samples: {global_step}")
    print("=" * 60)
    
    gs.destroy()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Env PPO Direct Control Training')
    parser.add_argument('--checkpoint', type=str, default='best_ppo_multienv.pth')
    parser.add_argument('--no-headless', dest='headless', action='store_false',
                        help='Show viewer (will use 1 env)')
    parser.set_defaults(headless=True)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()
    
    config = Config()
    config.num_envs = args.num_envs
    config.num_iterations = args.iterations
    config.steps_per_iteration = args.steps
    
    # If not headless, use single env for visualization
    if not args.headless:
        config.num_envs = 1
        print("Viewer enabled: using single environment")
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    train(config, headless=args.headless)


if __name__ == "__main__":
    main() 