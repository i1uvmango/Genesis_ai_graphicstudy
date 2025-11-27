import bpy
import csv
import math
from mathutils import Vector, Quaternion, Matrix

# =====================
# CONFIG
# =====================

CAR_OBJECT_NAME   = "Corvette.Vehicle Body.RB"
FRONT_LEFT_WHEEL  = "Corvette.Vehicle Body.0.FL1_Wheel.RB"
FRONT_RIGHT_WHEEL = "Corvette.Vehicle Body.0.FR0_Wheel.RB"
REAR_LEFT_WHEEL   = "Corvette.Vehicle Body.1.BL1_Wheel.RB"
REAR_RIGHT_WHEEL  = "Corvette.Vehicle Body.1.BR0_Wheel.RB"

BLENDER_FORWARD_LOCAL = Vector((0.0, -1.0, 0.0))
WHEEL_SPIN_AXIS_LOCAL = Vector((1.0, 0.0, 0.0)) 

OUTPUT_CSV_PATH = "C:/Users/USER/Desktop/sensor_log.csv"

ROT_B2G = Matrix.Rotation(math.radians(90.0), 3, 'Z')

def vec_B_to_G(v_b: Vector) -> Vector:
    return ROT_B2G @ v_b

def mat3_B_to_G(R_b: Matrix) -> Matrix:
    return ROT_B2G @ R_b

def quat_B_to_G(q_b: Quaternion) -> Quaternion:
    R_b = q_b.to_matrix()
    R_g = mat3_B_to_G(R_b)
    return R_g.to_quaternion().normalized()

def get_obj(name):
    if name and name in bpy.data.objects:
        return bpy.data.objects[name]
    return None

def get_world_vel_B(obj, dt):
    if dt <= 0.0:
        return Vector((0.0, 0.0, 0.0))

    prev_loc = obj.get("_prev_world_loc_B", None)
    curr_loc = obj.matrix_world.translation.copy()

    if prev_loc is None:
        obj["_prev_world_loc_B"] = curr_loc
        return Vector((0.0, 0.0, 0.0))

    prev_loc = Vector(prev_loc)
    vel = (curr_loc - prev_loc) / dt
    obj["_prev_world_loc_B"] = curr_loc
    return vel

def get_world_ang_vel_B(obj, dt):
    if dt <= 0.0:
        return Vector((0.0, 0.0, 0.0))

    prev_rot = obj.get("_prev_world_rot_quat_B", None)
    curr_rot = obj.matrix_world.to_quaternion().normalized()

    if prev_rot is None:
        obj["_prev_world_rot_quat_B"] = curr_rot[:]
        return Vector((0.0, 0.0, 0.0))

    prev_rot = Quaternion(prev_rot)
    delta = prev_rot.conjugated() @ curr_rot
    angle = delta.angle
    if angle == 0.0:
        ang_vel = Vector((0.0, 0.0, 0.0))
    else:
        axis = delta.axis
        ang_vel = Vector(axis) * (angle / dt)

    obj["_prev_world_rot_quat_B"] = curr_rot[:]
    return ang_vel

def signed_yaw_between(forward_a_world: Vector, forward_b_world: Vector) -> float:
    a = Vector((forward_a_world.x, forward_a_world.y, 0.0))
    b = Vector((forward_b_world.x, forward_b_world.y, 0.0))

    if a.length == 0.0 or b.length == 0.0:
        return 0.0

    a.normalize()
    b.normalize()

    dot = max(-1.0, min(1.0, a.dot(b)))
    angle = math.acos(dot)

    cross_z = a.cross(b).z
    if cross_z < 0:
        angle = -angle

    return angle


# ✅ 수정된 함수 - 쿼터니언 버그 수정
def get_wheel_spin_rate_B_fixed(wheel_obj, dt: float) -> float:
    """
    휠 스핀 속도 계산 - 쿼터니언 shortest path 버그 수정
    """
    if wheel_obj is None or dt <= 0.0:
        return 0.0

    curr_rot = wheel_obj.matrix_world.to_quaternion().normalized()
    prev_rot = wheel_obj.get("_prev_wheel_rot_quat_B", None)

    if prev_rot is None:
        wheel_obj["_prev_wheel_rot_quat_B"] = curr_rot[:]
        return 0.0

    prev_rot = Quaternion(prev_rot)
   
    # ✅ 핵심 수정: 두 쿼터니언의 dot product로 방향 확인
    dot_product = prev_rot.w * curr_rot.w + \
                  prev_rot.x * curr_rot.x + \
                  prev_rot.y * curr_rot.y + \
                  prev_rot.z * curr_rot.z
   
    # dot < 0이면 반대편 쿼터니언을 사용 (shortest path)
    if dot_product < 0:
        curr_rot_corrected = Quaternion((-curr_rot.w, -curr_rot.x, -curr_rot.y, -curr_rot.z))
    else:
        curr_rot_corrected = curr_rot
   
    delta = prev_rot.conjugated() @ curr_rot_corrected
    angle = delta.angle
   
    if abs(angle) < 1e-6:
        ang_vel_world = Vector((0.0, 0.0, 0.0))
    else:
        axis = delta.axis
        ang_vel_world = Vector(axis) * (angle / dt)

    # 휠 로컬 회전축 투영
    R_w = wheel_obj.matrix_world.to_3x3()
    spin_axis_world = (R_w @ WHEEL_SPIN_AXIS_LOCAL).normalized()
    spin_rate = ang_vel_world.dot(spin_axis_world)
   
    # ✅ 안전장치: 비정상적으로 큰 값 클리핑
    if abs(spin_rate) > 100.0:
        print(f"[WARN] Frame {bpy.context.scene.frame_current}: "
              f"Abnormal spin_rate {spin_rate:.2f}, clamping to ±100")
        spin_rate = math.copysign(100.0, spin_rate)

    wheel_obj["_prev_wheel_rot_quat_B"] = curr_rot_corrected[:]
    return spin_rate


def init_csv(path, scene):
    full_path = bpy.path.abspath(path)
    with open(full_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "frame", "time",
            "g_pos_x", "g_pos_y", "g_pos_z",
            "g_qw", "g_qx", "g_qy", "g_qz",
            "g_lin_vx", "g_lin_vy", "g_lin_vz",
            "g_ang_vx", "g_ang_vy", "g_ang_vz",
            "steer_L", "steer_R",
            "spin_FL", "spin_FR", "spin_RL", "spin_RR",
        ]
        writer.writerow(header)
    print(f"[CarLogger] CSV 초기화: {full_path}")

def append_row(path, row):
    full_path = bpy.path.abspath(path)
    with open(full_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

last_frame = None

def car_logger_handler(scene):
    global last_frame

    car = get_obj(CAR_OBJECT_NAME)
    if car is None:
        return

    frame = scene.frame_current
    fps = scene.render.fps
    time_sec = frame / fps

    if last_frame is None:
        dt = 0.0
    else:
        dt = (frame - last_frame) / fps
    last_frame = frame

    mw_B = car.matrix_world
    loc_B = mw_B.translation
    rot_B = mw_B.to_quaternion().normalized()

    v_B = get_world_vel_B(car, dt)
    w_B = get_world_ang_vel_B(car, dt)

    pos_G = vec_B_to_G(loc_B)
    rot_G = quat_B_to_G(rot_B)
    lin_v_G = vec_B_to_G(v_B)
    ang_v_G = vec_B_to_G(w_B)

    wheel_FL = get_obj(FRONT_LEFT_WHEEL)
    wheel_FR = get_obj(FRONT_RIGHT_WHEEL)
    wheel_RL = get_obj(REAR_LEFT_WHEEL)
    wheel_RR = get_obj(REAR_RIGHT_WHEEL)

    R_body = car.matrix_world.to_3x3()
    body_fwd_world = R_body @ BLENDER_FORWARD_LOCAL

    if v_B.length > 1e-3:
        v_dir = v_B.normalized()
        if body_fwd_world.dot(v_dir) < 0:
            body_fwd_world = -body_fwd_world

    def wheel_steer_angle(wheel_obj):
        if wheel_obj is None:
            return 0.0
        R_wheel = wheel_obj.matrix_world.to_3x3()
        wheel_fwd_world = R_wheel @ BLENDER_FORWARD_LOCAL
        angle = signed_yaw_between(body_fwd_world, wheel_fwd_world)
        if angle > math.pi / 2:
            angle -= math.pi
        elif angle < -math.pi / 2:
            angle += math.pi
        return angle

    steer_L = wheel_steer_angle(wheel_FL)
    steer_R = wheel_steer_angle(wheel_FR)

    # ✅ 수정된 함수 사용
    spin_FL = get_wheel_spin_rate_B_fixed(wheel_FL, dt)
    spin_FR = get_wheel_spin_rate_B_fixed(wheel_FR, dt)
    spin_RL = get_wheel_spin_rate_B_fixed(wheel_RL, dt)
    spin_RR = get_wheel_spin_rate_B_fixed(wheel_RR, dt)

    row = [
        frame, time_sec,
        pos_G.x, pos_G.y, pos_G.z,
        rot_G.w, rot_G.x, rot_G.y, rot_G.z,
        lin_v_G.x, lin_v_G.y, lin_v_G.z,
        ang_v_G.x, ang_v_G.y, ang_v_G.z,
        steer_L, steer_R,
        spin_FL, spin_FR, spin_RL, spin_RR,
    ]

    append_row(OUTPUT_CSV_PATH, row)

def register_car_logger():
    global last_frame
    last_frame = None

    scene = bpy.context.scene
    init_csv(OUTPUT_CSV_PATH, scene)

    handlers = bpy.app.handlers.frame_change_post
    for h in list(handlers):
        if getattr(h, "__name__", "") == "car_logger_handler":
            handlers.remove(h)

    handlers.append(car_logger_handler)
    print("[CarLogger] 프레임 체인지 핸들러 등록 완료 (버그 수정 버전)")

def export_all_frames(start_frame=1, end_frame=250):
    global last_frame
    scene = bpy.context.scene
    register_car_logger()
    last_frame = None
    current_frame = scene.frame_current

    print(f"[CarLogger] {start_frame}~{end_frame} 프레임 내보내기 시작")
    for f in range(start_frame, end_frame + 1):
        scene.frame_set(f)
    scene.frame_set(current_frame)
    print(f"[CarLogger] 완료! sensor_log.csv 생성됨")

if __name__ == "__main__":
    register_car_logger()
    export_all_frames(1, 250)