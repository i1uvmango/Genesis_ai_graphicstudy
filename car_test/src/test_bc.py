"""
Professional Behavior Cloning Inference Script for Genesis AI

This script loads a trained behavior cloning policy and deploys it in Genesis AI
simulation with real-time visualization.

Author: Genesis AI Team
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
    model_path: str = "checkpoint/drive_8_test.pth"  # Updated to use newly trained model
    urdf_path: str = "URDF/test_v1_simple.urdf"
    
    # Simulation parameters
    dt: float = 0.02
    substeps: int = 2
    n_steps: int = 2000
    
    # Control parameters
    max_steer_limit: float = 0.65  # rad
    max_wheel_torque: float = 2000.0  # Nm
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Model
# ============================================================================

class BehaviorCloningPolicy(nn.Module):
    """MLP policy for behavior cloning (must match training architecture)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def load_policy(model_path: str, device: str):
    """Load trained policy from checkpoint"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    print(f"Loading policy from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract configuration
    config = checkpoint['config']
    state_dim = config['state_dim']
    action_dim = config['action_dim']
    hidden_dim = config['hidden_dim']
    
    # Extract normalization parameters
    state_mean = np.array(checkpoint['state_mean'], dtype=np.float32)
    state_std = np.array(checkpoint['state_std'], dtype=np.float32)
    
    # Initialize model
    model = BehaviorCloningPolicy(state_dim, action_dim, hidden_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Policy loaded successfully!")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  State mean: {state_mean}")
    print(f"  State std: {state_std}")
    print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    
    return model, state_mean, state_std


# ============================================================================
# Coordinate Transformation
# ============================================================================

def quaternion_rotate_inverse(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    Rotate a single vector from global to local frame using quaternion inverse.
    
    Args:
        vector: (3,) vector in global frame
        quaternion: (4,) quaternion [w, x, y, z]
    
    Returns:
        (3,) vector in local frame
    """
    w, x, y, z = quaternion
    vx, vy, vz = vector
    
    # Conjugate quaternion
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
    
    return np.array([out_x, out_y, out_z], dtype=np.float32)


def transform_to_genesis_frame(velocity: np.ndarray) -> np.ndarray:
    """
    Transform velocity from Blender local frame to Genesis local frame.
    
    Blender local: X=right, Y=forward (-Y is forward direction), Z=up
    Genesis local: X=forward, Y=left, Z=up
    
    Args:
        velocity: (3,) velocity in Blender local frame
    
    Returns:
        (3,) velocity in Genesis local frame
    """
    genesis_vel = np.zeros(3, dtype=np.float32)
    genesis_vel[0] = -velocity[1]  # Forward: -Blender Y → Genesis X
    genesis_vel[1] = -velocity[0]  # Lateral: -Blender X → Genesis Y
    genesis_vel[2] = velocity[2]   # Vertical: Z → Z
    return genesis_vel


# ============================================================================
# Genesis Simulation
# ============================================================================

def initialize_simulation(config: Config):
    """Initialize Genesis simulation environment"""
    
    print("\nInitializing Genesis simulation...")
    gs.init(backend=gs.gpu)  # Note: FPS logs cannot be suppressed in this version
    
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
    
    # Add ground plane
    scene.add_entity(gs.morphs.Plane())
    
    # Add vehicle
    if not os.path.exists(config.urdf_path):
        raise FileNotFoundError(f"URDF file not found: {config.urdf_path}")
    
    car = scene.add_entity(
        gs.morphs.URDF(
            file=config.urdf_path,
            pos=(0.0, 0.0, 0.3),
            euler=(0.0, 0.0, 0.0),
            fixed=False,  # Must be floating for velocity readings
        )
    )
    
    scene.build()
    
    print("Simulation initialized!")
    return scene, car


def get_joint_indices(car):
    """Get joint indices for vehicle control"""
    try:
        fl_steer_idx = int(car.get_joint("front_left_steer_joint").dofs_idx_local[0])
        fr_steer_idx = int(car.get_joint("front_right_steer_joint").dofs_idx_local[0])
        rl_wheel_idx = int(car.get_joint("rear_left_wheel_joint").dofs_idx_local[0])
        rr_wheel_idx = int(car.get_joint("rear_right_wheel_joint").dofs_idx_local[0])
        return fl_steer_idx, fr_steer_idx, rl_wheel_idx, rr_wheel_idx
    except AttributeError as e:
        raise RuntimeError(f"Failed to find joints. Check URDF file. Error: {e}")


def to_numpy(tensor):
    """Convert Genesis tensor to numpy array"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().astype(np.float32)
    return np.array(tensor, dtype=np.float32)


# ============================================================================
# Main Inference Loop
# ============================================================================

def run_inference(config: Config):
    """Main inference loop"""
    
    # Load policy
    model, state_mean, state_std = load_policy(config.model_path, config.device)
    state_mean_torch = torch.from_numpy(state_mean).to(config.device)
    state_std_torch = torch.from_numpy(state_std).to(config.device)
    
    # Initialize simulation
    scene, car = initialize_simulation(config)
    fl_idx, fr_idx, rl_idx, rr_idx = get_joint_indices(car)
    
    print(f"\n{'='*60}")
    print("Starting inference")
    print(f"{'='*60}\n")
    
    # Inference loop
    for step in range(config.n_steps):
        
        # 1. Get vehicle state from simulation
        g_lin_vel = to_numpy(car.get_vel())      # Global linear velocity
        g_ang_vel = to_numpy(car.get_ang())      # Global angular velocity
        g_quat = to_numpy(car.get_quat())        # Orientation [w, x, y, z]
        
        # 2. Transform to local frame
        l_lin_vel = quaternion_rotate_inverse(g_lin_vel, g_quat)
        l_ang_vel = quaternion_rotate_inverse(g_ang_vel, g_quat)
        
        # Note: Genesis local frame is already correct (X=forward, Y=left, Z=up)
        # No additional coordinate transformation needed!
        
        # 4. Create state vector
        state = np.concatenate([l_lin_vel, l_ang_vel])  # (6,)
        
        # 5. Normalize state
        state_normalized = (state - state_mean) / state_std
        state_tensor = torch.from_numpy(state_normalized).float().unsqueeze(0).to(config.device)
        
        # 6. Get action from policy
        with torch.no_grad():
            action = model(state_tensor).cpu().numpy()[0]  # [steering, throttle]
        
        steering = action[0]
        throttle = action[1]
        
        # Kickstart: Force acceleration during initial steps only
        current_speed = np.linalg.norm(l_lin_vel[:2])
        kickstart_active = (step < 100)  # Only first 100 steps
        
        # DEBUG: Show prediction (every 50 steps for more detail)
        if step % 50 == 0:
            direction = "LEFT" if action[0] > 0 else ("RIGHT" if action[0] < 0 else "STRAIGHT")
            print(f"\n[DEBUG Step {step}]")
            print(f"  Current speed: {current_speed:.2f} m/s")
            print(f"  Model raw output: steer={action[0]:.3f} ({direction}), throttle={action[1]:.3f}")
            print(f"  Kickstart active: {kickstart_active}")
            if kickstart_active:
                print(f"  → Overriding to: throttle=1.0, steer=0.0")
            else:
                print(f"  → Using model output as-is")

        
        if kickstart_active:
            # Override model output with full throttle during initial acceleration
            throttle = 1.0
            steering = 0.0  # Keep straight during acceleration
        
        # 7. Convert to control commands
        # Steering (already in [-1, 1] range from training data)
        steer_cmd = np.clip(steering, -config.max_steer_limit, config.max_steer_limit)
        
        # Throttle to wheel velocity (simple P-control)
        target_wheel_vel = throttle * 50.0  # Scale to reach training speed faster
        
        # 8. Apply control
        # Steering (position control)
        car.control_dofs_position(
            position=np.array([steer_cmd, steer_cmd]),
            dofs_idx_local=np.array([fl_idx, fr_idx])
        )
        
        # Drive (velocity control with torque)
        dofs_vel = to_numpy(car.get_dofs_velocity())
        cur_vel_rl = dofs_vel[rl_idx]
        cur_vel_rr = dofs_vel[rr_idx]
        
        Kp = 200.0
        torque_rl = Kp * (target_wheel_vel - cur_vel_rl)
        torque_rr = Kp * (target_wheel_vel - cur_vel_rr)
        
        torque_rl = np.clip(torque_rl, -config.max_wheel_torque, config.max_wheel_torque)
        torque_rr = np.clip(torque_rr, -config.max_wheel_torque, config.max_wheel_torque)
        
        car.control_dofs_force(
            force=np.array([torque_rl, torque_rr]),
            dofs_idx_local=np.array([rl_idx, rr_idx])
        )
        
        # 9. Step simulation
        scene.step()
        
        # 10. Debug output (reduced frequency)
        if step % 100 == 0:
            # Get position
            pos = to_numpy(car.get_pos())
            
            # Calculate speed
            speed = np.linalg.norm(l_lin_vel[:2])
            
            print(f"\n[Step {step:04d}]")
            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            print(f"  Lin Vel (local): vx={l_lin_vel[0]:.2f}, vy={l_lin_vel[1]:.2f}, vz={l_lin_vel[2]:.2f} m/s")
            print(f"  Ang Vel (local): wx={l_ang_vel[0]:.2f}, wy={l_ang_vel[1]:.2f}, wz={l_ang_vel[2]:.2f} rad/s")
            print(f"  Speed: {speed:.2f} m/s ({speed*3.6:.1f} km/h)")
            print(f"  Control: Steer={steer_cmd:.3f} rad, Throttle={throttle:.3f}")
            print(f"  Wheel Vel: target={target_wheel_vel:.1f} rad/s")
    
    print(f"\n{'='*60}")
    print("Inference completed!")
    print(f"{'='*60}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Behavior Cloning Inference")
    parser.add_argument('--model', type=str, default=Config.model_path,
                       help='Path to model checkpoint')
    parser.add_argument('--urdf', type=str, default=Config.urdf_path,
                       help='Path to URDF file')
    parser.add_argument('--steps', type=int, default=Config.n_steps,
                       help='Number of simulation steps')
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.model_path = args.model
    config.urdf_path = args.urdf
    config.n_steps = args.steps
    
    # Run inference
    run_inference(config)


if __name__ == "__main__":
    main()
