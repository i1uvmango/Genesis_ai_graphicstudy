# Blender에서 Data 추출 script

import bpy
import csv
import math
from mathutils import Vector, Matrix, Quaternion

########################################
# 사용자 설정
########################################

BODY_NAME = "103104.Vehicle Body.RB"

# 바퀴 rigid body 이름
FL_WHEEL = "103104.Body.0.FL1_Wheel.RB"
FR_WHEEL = "103104.Body.0.FR0_Wheel.RB"
RL_WHEEL = "103104.Body.1.BL1_Wheel.RB"
RR_WHEEL = "103104.Body.1.BR0_Wheel.RB"

LOG_PATH = bpy.path.abspath("//mlp_data.csv")

FPS = bpy.context.scene.render.fps
DT = 1.0 / FPS


########################################
# Blender → Genesis 좌표계 변환 (Z축 90도 회전)
########################################

ROT_B2G = Matrix.Rotation(math.radians(90.0), 3, 'Z')

def vec_B_to_G(v):
    return ROT_B2G @ v


########################################
# 전역 상태 저장
########################################

_state = {
    "initialized": False,
    "prev_loc": None,
    "prev_rot": None,  # 차량 회전
    "prev_wheel_rot": {"rl": None, "rr": None},  # 바퀴 회전 저장
    "csv": None,
    "writer": None
}


########################################
# 각속도 계산 (Quaternion 차분)
########################################

def compute_ang_vel(prev_q, curr_q):
    delta = prev_q.conjugated() @ curr_q
    angle = delta.angle
    axis = delta.axis
    if abs(angle) < 1e-8:
        return Vector((0, 0, 0))
    ang_vel = axis * (angle / DT)
    return ang_vel


########################################
# ★ Steering = 앞바퀴 회전(Quaternion) → Z축 성분
########################################

def get_true_steering():
    fl = bpy.data.objects.get(FL_WHEEL)
    fr = bpy.data.objects.get(FR_WHEEL)
    if fl is None or fr is None:
        return 0.0

    steer_fl = fl.matrix_world.to_quaternion().to_euler("XYZ").z
    steer_fr = fr.matrix_world.to_quaternion().to_euler("XYZ").z

    return 0.5 * (steer_fl + steer_fr)


########################################
# ★ Throttle = 뒤바퀴 Quaternion 기반 각속도(Y축 spin)
########################################

def get_true_throttle():

    rl = bpy.data.objects.get(RL_WHEEL)
    rr = bpy.data.objects.get(RR_WHEEL)
    if rl is None or rr is None:
        return 0.0

    rl_q = rl.matrix_world.to_quaternion()
    rr_q = rr.matrix_world.to_quaternion()

    # 첫 프레임은 기록만
    if _state["prev_wheel_rot"]["rl"] is None:
        _state["prev_wheel_rot"]["rl"] = rl_q.copy()
        _state["prev_wheel_rot"]["rr"] = rr_q.copy()
        return 0.0

    def quat_spin(prev_q, curr_q):
        delta = prev_q.conjugated() @ curr_q
        ang = delta.angle
        axis = delta.axis
        if abs(ang) < 1e-8:
            return 0.0
        ang_vel_vec = axis * (ang / DT)
        return ang_vel_vec.y   # 바퀴 spin = local Y축

    rl_spin = quat_spin(_state["prev_wheel_rot"]["rl"], rl_q)
    rr_spin = quat_spin(_state["prev_wheel_rot"]["rr"], rr_q)

    throttle_raw = 0.5 * (rl_spin + rr_spin)
    throttle_norm = math.tanh(throttle_raw / 10.0)

    _state["prev_wheel_rot"]["rl"] = rl_q.copy()
    _state["prev_wheel_rot"]["rr"] = rr_q.copy()

    return throttle_norm


########################################
# 프레임 핸들러
########################################

def frame_handler(scene, depsgraph):

    body = bpy.data.objects.get(BODY_NAME)

    if body is None:
        print("[6DoF] Body not found.")
        return

    curr_loc = body.matrix_world.translation.copy()
    curr_rot = body.matrix_world.to_quaternion().copy()

    if not _state["initialized"]:
        print(f"[6DoF] Logging started → {LOG_PATH}")

        _state["prev_loc"] = curr_loc
        _state["prev_rot"] = curr_rot

        f = open(LOG_PATH, "w", newline="")
        w = csv.writer(f)

        w.writerow([
            "frame", "time",
            "g_lin_vx","g_lin_vy","g_lin_vz",
            "g_ang_vx","g_ang_vy","g_ang_vz",
            "steering","throttle"
        ])

        _state["csv"] = f
        _state["writer"] = w
        _state["initialized"] = True
        return

    # 선속도
    lin_vel = (curr_loc - _state["prev_loc"]) / DT
    lin_vel = vec_B_to_G(lin_vel)

    # 차량 각속도 (Quaternion 차분)
    ang_vel = compute_ang_vel(_state["prev_rot"], curr_rot)
    ang_vel = vec_B_to_G(ang_vel)

    # Steering / Throttle
    steering = get_true_steering()
    throttle = get_true_throttle()

    frame = scene.frame_current
    time_sec = frame * DT

    _state["writer"].writerow([
        frame, f"{time_sec:.6f}",
        f"{lin_vel.x:.6f}", f"{lin_vel.y:.6f}", f"{lin_vel.z:.6f}",
        f"{ang_vel.x:.6f}", f"{ang_vel.y:.6f}", f"{ang_vel.z:.6f}",
        f"{steering:.6f}", f"{throttle:.6f}"
    ])

    _state["prev_loc"] = curr_loc
    _state["prev_rot"] = curr_rot


########################################
# 핸들러 등록/해제
########################################

def register_6dof_logger():
    unregister_6dof_logger()
    bpy.app.handlers.frame_change_post.append(frame_handler)
    print("[6DoF] 핸들러 등록 완료")

def unregister_6dof_logger():
    bpy.app.handlers.frame_change_post[:] = []
    if _state["csv"]:
        _state["csv"].close()
        print("[6DoF] CSV 저장 완료")
    _state["initialized"] = False
    print("[6DoF] 핸들러 해제 완료")


########################################
# 실행
########################################

if __name__ == "__main__":
    register_6dof_logger()
