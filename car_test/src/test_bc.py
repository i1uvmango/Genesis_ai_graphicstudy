"""
Professional Behavior Cloning Inference Script for Genesis AI
Updated to match the robust 'friend's code' training script.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import genesis as gs

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Inference configuration"""
    # Paths
    model_path: str = "checkpoint/drive_8_test.pth"
    urdf_path: str = "URDF/test_v1_simple.urdf"
    
    # Simulation parameters
    dt: float = 0.02
    substeps: int = 2
    n_steps: int = 2000
    
    # Control parameters
    # Control parameters
    max_steer_limit: float = 0.8  # rad (Reset to safe limit ~45 deg)
    max_wheel_torque: float = 500.0  # Nm

    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Model (Must match new_train.py MLP)
# ============================================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2, hidden=(128, 128), dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            # Inference doesn't strictly need dropout, but we keep structure
            # to avoid state_dict mismatch if keys contain dropout.
            # However, usually state_dict keys are just index based for Sequential.
            # Let's match exact structure of new_train.py: Linear -> ReLU -> Dropout
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def load_policy(model_path: str, device: str):
    """Load trained policy from checkpoint"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    print(f"Loading policy from {model_path}...")
    # weights_only=False is required because we are loading numpy arrays (mu, sigma)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract normalization parameters - New keys: mu, sigma
    mu = np.array(checkpoint['mu'], dtype=np.float32).squeeze()
    sigma = np.array(checkpoint['sigma'], dtype=np.float32).squeeze()

    # Extract config if available, else assume defaults
    train_config = checkpoint.get('config', {})
    state_dim = train_config.get('state_dim', 12) # Default to 12 (new feature set)
    action_dim = train_config.get('action_dim', 2)
    hidden_dim = train_config.get('hidden_dim', 128)
    
    # Initialize model
    # Note: Training used dropout=0.1, but for inference we effectively disable it via eval()
    # Constructing with same structure ensures key matching.
    model = MLP(in_dim=state_dim, out_dim=action_dim, hidden=(hidden_dim, hidden_dim)).to(device)
    model.load_state_dict(checkpoint['model_state']) # Fixed key name
    model.eval()
    
    print(f"Policy loaded successfully!")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Mu shape: {mu.shape}")
    print(f"  Sigma shape: {sigma.shape}")
    
    return model, mu, sigma


# ============================================================================
# Coordinate Transformation
# ============================================================================

def quaternion_rotate_inverse(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Rotate a single vector from global to local frame using quaternion inverse."""
    w, x, y, z = quaternion
    vx, vy, vz = vector
    
    # Conjugate quaternion
    qx, qy, qz = -x, -y, -z
    qw = w
    
    # t = q_conj * v
    t_w = -qx * vx - qy * vy - qz * vz
    t_x = qw * vx + qy * vz - qz * vy
    t_y = qw * vy + qz * vx - qx * vz
    t_z = qw * vz + qx * vy - qy * vx
    
    # v_local = t * q
    out_x = t_w * x + t_x * w + t_y * z - t_z * y
    out_y = t_w * y - t_x * z + t_y * w + t_z * x
    out_z = t_w * z + t_x * y - t_y * x + t_z * w
    
    return np.array([out_x, out_y, out_z], dtype=np.float32)


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().astype(np.float32)
    return np.array(tensor, dtype=np.float32)


# ============================================================================
# Genesis Simulation
# ============================================================================

def initialize_simulation(config: Config):
    print("\nInitializing Genesis simulation...")
    gs.init(backend=gs.gpu)
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=config.dt,
            substeps=config.substeps,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -3.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=45,
        ),
        show_viewer=True,
    )
    
    scene.add_entity(gs.morphs.Plane())
    
    if not os.path.exists(config.urdf_path):
        raise FileNotFoundError(f"URDF file not found: {config.urdf_path}")
    
    car = scene.add_entity(
        gs.morphs.URDF(
            file=config.urdf_path,
            pos=(0.0, 0.0, 0.3),
            euler=(0.0, 0.0, 0.0),
            fixed=False,
        )
    )
    
    scene.build()
    print("Simulation initialized!")
    return scene, car

def get_joint_indices(car):
    try:
        fl_steer_idx = int(car.get_joint("front_left_steer_joint").dofs_idx_local[0])
        fr_steer_idx = int(car.get_joint("front_right_steer_joint").dofs_idx_local[0])
        rl_wheel_idx = int(car.get_joint("rear_left_wheel_joint").dofs_idx_local[0])
        rr_wheel_idx = int(car.get_joint("rear_right_wheel_joint").dofs_idx_local[0])
        return fl_steer_idx, fr_steer_idx, rl_wheel_idx, rr_wheel_idx
    except AttributeError as e:
        raise RuntimeError(f"Failed to find joints. Check URDF file. Error: {e}")


# ============================================================================
# Main Inference Loop
# ============================================================================

def run_inference(config: Config):
    # Load policy and norm stats
    model, mu, sigma = load_policy(config.model_path, config.device)
    mu_torch = torch.from_numpy(mu).to(config.device)
    sigma_torch = torch.from_numpy(sigma).to(config.device)
    
    # Initialize simulation
    scene, car = initialize_simulation(config)
    fl_idx, fr_idx, rl_idx, rr_idx = get_joint_indices(car)
    
    print(f"\n{'='*60}")
    print("Starting inference")
    print(f"{'='*60}\n")
    
    for step in range(config.n_steps):
        
        g_pos = to_numpy(car.get_pos())          # [NEW] Global Pos (3,)
        g_lin_vel = to_numpy(car.get_vel())      # Global Lin Vel (3,)
        g_ang_vel = to_numpy(car.get_ang())      # Global Ang Vel (3,)
        g_quat = to_numpy(car.get_quat())        # [w, x, y, z] (4,)
        
        # Calculate Local Forward Velocity (v_long)
        # Genesis body frame: X is forward.
        # So v_long is simply the X component of local velocity.
        l_lin_vel = quaternion_rotate_inverse(g_lin_vel, g_quat)
        v_long = l_lin_vel[0]
        
        # Calculate Spin R (Rear wheels avg)
        dofs_vel = to_numpy(car.get_dofs_velocity())
        spin_rl = dofs_vel[rl_idx]
        spin_rr = dofs_vel[rr_idx]
        # In Blender script we used: spin_R = 0.5 * (spin_RL + spin_RR)
        # Assuming Genesis wheel spin aligns with this sign convention (usually + for forward)
        spin_R = 0.5 * (spin_rl + spin_rr)
        
        # --- 2. Construct Input Feature Vector (12-dim) ---
        # Order must match Cols.X from new_train.py:
        # [g_pos_x, g_pos_y, g_pos_z, (NEW)
        #  g_qw, g_qx, g_qy, g_qz, 
        #  g_lin_vx, g_lin_vy, g_lin_vz, 
        #  g_ang_vx, g_ang_vy, g_ang_vz, 
        #  v_long, spin_R]
        
        state = np.concatenate([
            g_pos,        # 3 (NEW)
            g_quat,       # 4
            g_lin_vel,    # 3
            g_ang_vel,    # 3
            [v_long],     # 1
            [spin_R]      # 1
        ]).astype(np.float32) # Total 12
        
        # --- 3. Normalize & Inference ---
        state_vals = (state - mu) / sigma
        state_tensor = torch.from_numpy(state_vals).float().unsqueeze(0).to(config.device)
        
        with torch.no_grad():
            raw_out = model(state_tensor)
            action = raw_out.cpu().numpy()[0]
            
            # DEBUG: Check shape if error occurs
            if len(action) < 2:
                print(f"!!! CRITICAL ERROR !!!")
                print(f"Model output shape: {raw_out.shape}")
                print(f"Action array: {action}")
                print(f"Config action_dim: {config.action_dim if hasattr(config, 'action_dim') else 'Unknown'}")
        
        # --- 4. Post-process Action ---
        # Clip to [-1, 1] as done in training script utils
        action = np.clip(action, -1.0, 1.0)
        
        steering = action[0]
        throttle_norm = action[1]
        
        # --- 5. Debug & Kickstart ---
        current_speed = np.linalg.norm(g_lin_vel[:2])
        
        # [REMOVED] Kickstart logic removed as requested. Model controls from step 0.
        
        if step % 50 == 0:
            print(f"\n[Step {step}] Speed: {current_speed:.2f} m/s")
            print(f"  Input: v_long={v_long:.2f}, spin_R={spin_R:.2f}")
            print(f"  Pred : steer={steering:.3f}, throttle={throttle_norm:.3f}")
            
            # DEBUG: Check State Normalization
            print(f"  [DEBUG] State Normalization Check:")
            print(f"    Raw State: {state}")
            print(f"    Mu       : {mu}")
            print(f"    Sigma    : {sigma}")
            print(f"    Norm State: {state_vals}") # Should be roughly within [-3, 3]
            print(f"    Quat (w,x,y,z): {g_quat}") # Check if w is ~1.0 for straight

        # --- 6. Apply Control ---
        
        # Steering Smoothing (Low-Pass Filter)
        alpha = 0.2  # Smoothing factor (0.0=frozen, 1.0=no smoothing)
        
        # Initialize prev_steer if not exists (using function attribute for persistency)
        if not hasattr(run_inference, "prev_steer"):
            run_inference.prev_steer = 0.0
            
        # [FIX] Reverted steering sign (User request)
        target_steer = np.clip(steering, -config.max_steer_limit, config.max_steer_limit)
        
        steer_cmd = (1 - alpha) * run_inference.prev_steer + alpha * target_steer
        run_inference.prev_steer = steer_cmd
        
        # Throttle norm -> Target Wheel Vel
        # Reduced from 60.0 to 30.0 to prevent understeer (increase traction)
        target_wheel_vel = throttle_norm * 30.0 
        
        # Steering (Position Control)
        car.control_dofs_position(
            position=np.array([steer_cmd, steer_cmd]),
            dofs_idx_local=np.array([fl_idx, fr_idx])
        )
        
        # Drive (Velocity Control with Torque Limit)
        # Using simple P-control logic approximation or direct velocity command if supported
        # Genesis 'control_dofs_velocity' is usually preferred for wheels if available/stable
        # But here we use force control to mimic torque limits
        
        cur_vel_rl = dofs_vel[rl_idx]
        cur_vel_rr = dofs_vel[rr_idx]
        Kp_accel = 50.0  # 가속할 때 (반응성 확보)
        Kp_brake = 10.0  # 감속할 때 (진동 방지, 부드러운 감속)

        # Rear Left
        err_rl = target_wheel_vel - cur_vel_rl
        kp_rl = Kp_accel if err_rl > 0 else Kp_brake
        torque_rl = kp_rl * err_rl

        # Rear Right
        err_rr = target_wheel_vel - cur_vel_rr
        kp_rr = Kp_accel if err_rr > 0 else Kp_brake
        torque_rr = kp_rr * err_rr
        
        max_T = config.max_wheel_torque
        torque_rl = np.clip(torque_rl, -max_T, max_T)
        torque_rr = np.clip(torque_rr, -max_T, max_T)
        
        car.control_dofs_force(
            force=np.array([torque_rl, torque_rr]),
            dofs_idx_local=np.array([rl_idx, rr_idx])
        )
        
        scene.step()

    print("Inference completed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=Config.model_path)
    parser.add_argument('--csv', type=str, help="Optional CSV to check integrity")
    args = parser.parse_args()
    
    cfg = Config()
    if args.model:
        cfg.model_path = args.model
        
    run_inference(cfg)

if __name__ == "__main__":
    main()
