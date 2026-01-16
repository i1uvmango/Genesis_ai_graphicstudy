"""
PPO Inference & Visualization Script (Direct Control)
=====================================================
Matches train_ppo.py logic with force control for rear wheels.

Key: kp/kv=0 for rear wheels to allow force control to work!
"""

import os
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

import genesis as gs
import torch
import numpy as np
import argparse
import time
import math

from train_ppo import (
    Config, ActorCritic,
    load_reference_3d,
    quat_to_rotation_matrix_batch,
    compute_velocity_body_batch,
    compute_angular_velocity_body_batch,
    compute_gravity_body_batch,
    compute_target_waypoint_batch,
    compute_tangent_direction_batch,
    compute_slip_proxy_batch,
    build_observation_batch,
)


def initialize_simulation_single(config: Config, start_pos: np.ndarray, start_quat: np.ndarray):
    """Initialize Genesis simulation with 1 environment."""
    print(f"\nInitializing Genesis simulation...")
    
    backend = gs.gpu if torch.cuda.is_available() else gs.cpu
    gs.init(backend=backend, debug=False, logging_level='warning')
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=config.dt, substeps=config.substeps),
        viewer_options=gs.options.ViewerOptions(camera_pos=(5.0, 0.0, 5.0), camera_lookat=(0.0, 0.0, 0.0)),
        show_viewer=True,
    )
    
    scene.add_entity(gs.morphs.Plane())
    car = scene.add_entity(gs.morphs.URDF(file="../URDF/test_v1_simple.urdf"))
    # scene.build() moved to after lookahead entity addition
    
    # Logic moved to after scene.build()
    
    # Visualization entity for lookahead point
    lookahead_vis = scene.add_entity(
        gs.morphs.Sphere(radius=0.2, pos=(0, 0, -5.0), fixed=True, collision=False),
        surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0, 0.8))
    )
    
    scene.build(n_envs=1)
    
    # DOF indices
    fl_idx = int(car.get_joint("front_left_steer_joint").dofs_idx_local[0])
    fr_idx = int(car.get_joint("front_right_steer_joint").dofs_idx_local[0])
    rl_idx = int(car.get_joint("rear_left_wheel_joint").dofs_idx_local[0])
    rr_idx = int(car.get_joint("rear_right_wheel_joint").dofs_idx_local[0])
    dof_indices = (fl_idx, fr_idx, rl_idx, rr_idx)
    print(f"DOF indices: FL={fl_idx}, FR={fr_idx}, RL={rl_idx}, RR={rr_idx}")
    
    # Steering: position control with PD gains (Increased for responsiveness)
    car.set_dofs_kp(kp=np.array([8000.0, 8000.0], dtype=np.float32), dofs_idx_local=np.array([fl_idx, fr_idx]))
    car.set_dofs_kv(kv=np.array([100.0, 100.0], dtype=np.float32), dofs_idx_local=np.array([fl_idx, fr_idx]))
    
    # Rear wheels: VELOCITY control - high kv for heavy car (1200 kg)
    car.set_dofs_kp(kp=np.array([0.0, 0.0], dtype=np.float32), dofs_idx_local=np.array([rl_idx, rr_idx]))
    car.set_dofs_kv(kv=np.array([200.0, 200.0], dtype=np.float32), dofs_idx_local=np.array([rl_idx, rr_idx]))
    car.set_dofs_force_range(
        lower=np.array([-2000.0, -2000.0], dtype=np.float32),  # Increased limit
        upper=np.array([2000.0, 2000.0], dtype=np.float32),
        dofs_idx_local=np.array([rl_idx, rr_idx])
    )
    
    # Set initial position
    car.set_pos(start_pos.reshape(1, 3))
    car.set_quat(start_quat.reshape(1, 4))
    
    return scene, car, dof_indices, lookahead_vis


def apply_control_single(car, accel_brake: float, steer: float, dof_indices: tuple, config: Config):
    """Apply control using VELOCITY control for rear wheels (same as train_ppo.py)."""
    fl_idx, fr_idx, rl_idx, rr_idx = dof_indices
    
    # Steering (position control)
    actual_steer = steer * config.max_steer
    car.control_dofs_position(
        position=np.array([[actual_steer, actual_steer]], dtype=np.float32),
        dofs_idx_local=np.array([fl_idx, fr_idx])
    )
    
    # Throttle: same logic as train_ppo.py
    engine_torque = max(accel_brake, 0.0) * config.max_engine_torque
    brake_torque = max(-accel_brake, 0.0) * config.max_brake_torque
    drive_torque = max(engine_torque - brake_torque, 0.0)
    
    # VELOCITY control (matching train_ppo.py)
    WHEEL_RADIUS = 0.358
    target_v_raw = (drive_torque / config.max_engine_torque) * config.target_speed
    target_v = np.maximum(target_v_raw, 0.3)  # Idle Creep (Fix #3)
    target_omega = target_v / WHEEL_RADIUS
    
    car.control_dofs_velocity(
        velocity=np.array([[target_omega, target_omega]], dtype=np.float32),
        dofs_idx_local=np.array([rl_idx, rr_idx])
    )


def run_inference(args):
    config = Config()
    config.num_envs = 1
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.max_engine_torque = 100.0  # Increased for stronger acceleration
    
    # Load reference path
    csv_path = args.csv if args.csv else config.reference_csv
    ref_data = load_reference_3d(csv_path, config.device)
    path_points = ref_data["path_points"]
    arc_length = ref_data["arc_length"]
    
    # Start pose
    start_pos_np = path_points[0].cpu().numpy().copy()
    start_pos_np[2] = 0.5
    p1 = path_points[1].cpu().numpy()
    yaw = math.atan2(p1[1] - start_pos_np[1], p1[0] - start_pos_np[0])
    start_quat_np = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)], dtype=np.float32)
    
    scene, car, dof_indices, lookahead_vis = initialize_simulation_single(config, start_pos_np, start_quat_np)
    
    # Load model
    model = ActorCritic(config).to(config.device)
    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(config.checkpoint_dir, "best_ppo_multienv.pth")
    
    if os.path.exists(ckpt_path):
        print(f"Loading model from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)
        model.actor.load_state_dict(checkpoint['model_actor'])
        model.eval()
        print("Model loaded!")
    else:
        print(f"No model found! Using random policy.")
    
    # Draw reference path
    path_np = path_points.cpu().numpy()
    for i in range(0, len(path_np) - 10, 10):
        try:
            scene.draw_debug_line(
                (float(path_np[i][0]), float(path_np[i][1]), 0.05),
                (float(path_np[i+10][0]), float(path_np[i+10][1]), 0.05),
                radius=0.03, color=(0, 1, 0, 0.7)
            )
        except:
            pass
    
    prev_actions = torch.zeros(1, 2, device=config.device)
    driven_points = []
    
    print("\n" + "=" * 60)
    print("Starting Inference (Velocity Control)...")
    print("=" * 60)
    
    # Warmup
    for _ in range(50):
        scene.step()
    
    # Kickstart
    print("Kickstart (100 steps, full throttle)...")
    for _ in range(100):
        apply_control_single(car, accel_brake=1.0, steer=0.0, dof_indices=dof_indices, config=config)
        scene.step()
        prev_actions = torch.tensor([[1.0, 0.0]], device=config.device)
    
    # Check post-kickstart
    vel_world = car.get_vel()
    R_car = quat_to_rotation_matrix_batch(car.get_quat())
    v_body = compute_velocity_body_batch(vel_world, R_car)
    # Kickstart done
    print(f"Kickstart done! Speed = {v_body[0, 0].item():.2f} m/s")
    
    # Main loop
    progress_idx = torch.zeros(1, dtype=torch.long, device=config.device)
    
    for step in range(args.max_steps):
        # Get state
        pos = car.get_pos()
        quat = car.get_quat()
        vel_world = car.get_vel()
        ang_vel_world = car.get_ang()
        
        R_car = quat_to_rotation_matrix_batch(quat)
        v_body = compute_velocity_body_batch(vel_world, R_car)
        omega_body = compute_angular_velocity_body_batch(ang_vel_world, R_car)
        g_body = compute_gravity_body_batch(R_car)
        v_long = v_body[:, 0]
        
        # Observations
        target_idx, target_pos, nearest_idx = compute_target_waypoint_batch(
            pos, path_points, v_long, config, arc_length, progress_idx=progress_idx
        )
        
        # Update progress monotonically
        progress_idx = torch.maximum(progress_idx, nearest_idx)
        
        target_rel = torch.bmm(R_car.transpose(1, 2), (target_pos - pos).unsqueeze(-1)).squeeze(-1)
        tangent_rel = compute_tangent_direction_batch(path_points, nearest_idx, R_car)
        slip_proxy = compute_slip_proxy_batch(prev_actions[:, 0], v_long, config.max_speed)
        
        obs = build_observation_batch(target_rel, v_body, omega_body, g_body, tangent_rel, slip_proxy, prev_actions)
        
        # Policy
        with torch.no_grad():
            action, _, _, _ = model.get_action(obs, deterministic=True)
        
        accel_brake = action[0, 0].item()
        steer = action[0, 1].item()
        
        # Inference Auto-Creep (Fix #4): mimic idle creep for first 10 steps (Reduced duration)
        if step < 10:
            accel_brake = max(accel_brake, 0.1)
        
        # Control + Step
        apply_control_single(car, accel_brake, steer, dof_indices, config)
        
        # Update visualization (Entity)
        lookahead_vis.set_pos(target_pos[0].cpu().numpy())
            
        scene.step()
        prev_actions = action.clone()
        
        # Post-step state for logging
        pos_post = car.get_pos()
        vel_post = car.get_vel()
        R_post = quat_to_rotation_matrix_batch(car.get_quat())
        v_body_post = compute_velocity_body_batch(vel_post, R_post)
        speed = v_body_post[0, 0].item()
        
        target_idx_post, target_pos_post, _ = compute_target_waypoint_batch(
            pos_post, path_points, v_body_post[:, 0], config, arc_length, progress_idx=progress_idx
        )
        target_rel_post = torch.bmm(R_post.transpose(1, 2), (target_pos_post - pos_post).unsqueeze(-1)).squeeze(-1)
        lat_err = target_rel_post[0, 1].item()
        
        current_pos = pos_post[0].cpu().numpy()
        
        # Draw driven path
        if step % 20 == 0 and len(driven_points) > 0:
            prev_pt = driven_points[-1]
            try:
                scene.draw_debug_line(
                    (float(prev_pt[0]), float(prev_pt[1]), 0.1),
                    (float(current_pos[0]), float(current_pos[1]), 0.1),
                    radius=0.05, color=(1, 1, 0, 1.0)
                )
            except:
                pass
        
        # Draw target waypoint (red sphere) - Update position
        if step % 5 == 0:
            t_pos = target_pos_post[0].cpu().numpy()
            lookahead_vis.set_pos((float(t_pos[0]), float(t_pos[1]), 0.3))
        
        driven_points.append(current_pos.copy())
        
        # Log
        if step % 50 == 0:
            # Calculate actual drive_torque for logging
            engine_torque = max(accel_brake, 0.0) * config.max_engine_torque
            brake_torque = max(-accel_brake, 0.0) * config.max_brake_torque
            drive_torque = max(engine_torque - brake_torque, 0.0)
            print(f"Step {step:4d} | Speed={speed:.2f} m/s | Z={current_pos[2]:.3f} | LatErr={lat_err:.3f} | Steer={steer:.3f} | Thr={drive_torque:.1f}Nm")
        
        if current_pos[2] < -0.5:
            print("Car fell through ground!")
            break
    
    print("\nDone!")
    gs.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_ppo_multienv.pth")
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()
    run_inference(args)
