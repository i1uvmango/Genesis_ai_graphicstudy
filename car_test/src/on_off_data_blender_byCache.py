## Data extracting from blender (Script)
### True : data recording  ### False: only simulation

import bpy
import csv
import math
from mathutils import Vector, Quaternion, Matrix

# =====================
# ★ 사용자 설정 (CONFIG)
# =====================

# ▶ 로깅 ON/OFF 스위치
LOGGING_SWITCH = False  # True: 데이터 추출 시작 / False: 기능 끄기 (핸들러 해제)

# ▶ 추출할 프레임 범위 (자동 실행 시)
START_FRAME = 1
END_FRAME = 250

# ▶ 객체 이름 설정
CAR_OBJECT_NAME   = "Corvette.Vehicle Body.RB"
FRONT_LEFT_WHEEL  = "Corvette.Vehicle Body.0.FL1_Wheel.RB"
FRONT_RIGHT_WHEEL = "Corvette.Vehicle Body.0.FR0_Wheel.RB"
REAR_LEFT_WHEEL   = "Corvette.Vehicle Body.1.BL1_Wheel.RB"
REAR_RIGHT_WHEEL  = "Corvette.Vehicle Body.1.BR0_Wheel.RB"

# ▶ 벡터 설정
BLENDER_FORWARD_LOCAL = Vector((0.0, -1.0, 0.0))
WHEEL_SPIN_AXIS_LOCAL = Vector((1.0, 0.0, 0.0)) 

# ▶ 저장 경로
OUTPUT_CSV_PATH = "E:/MK/data/drive_8.csv"

# =====================
# MATH HELPERS
# =====================

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

def get_wheel_spin_rate_B_fixed(wheel_obj, dt: float) -> float:
    """
    휠 스핀 속도 계산 - 쿼터니언 shortest path 버그 수정 버전
    """
    if wheel_obj is None or dt <= 0.0:
        return 0.0

    curr_rot = wheel_obj.matrix_world.to_quaternion().normalized()
    prev_rot = wheel_obj.get("_prev_wheel_rot_quat_B", None)

    if prev_rot is None:
        wheel_obj["_prev_wheel_rot_quat_B"] = curr_rot[:]
        return 0.0

    prev_rot = Quaternion(prev_rot)
    
    # 쿼터니언 방향 보정 (Shortest Path 문제 해결)
    dot_product = prev_rot.w * curr_rot.w + \
                  prev_rot.x * curr_rot.x + \
                  prev_rot.y * curr_rot.y + \
                  prev_rot.z * curr_rot.z
    
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

    R_w = wheel_obj.matrix_world.to_3x3()
    spin_axis_world = (R_w @ WHEEL_SPIN_AXIS_LOCAL).normalized()
    spin_rate = ang_vel_world.dot(spin_axis_world)
    
    # 비정상 값 클리핑
    if abs(spin_rate) > 200.0:
        spin_rate = math.copysign(200.0, spin_rate)

    wheel_obj["_prev_wheel_rot_quat_B"] = curr_rot_corrected[:]
    return spin_rate

# =====================
# CSV & HANDLER LOGIC
# =====================

def init_csv(path):
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
    print(f"[CarLogger] CSV 초기화 완료: {full_path}")

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
        # 프레임이 뒤로 점프하면(Loop) 리셋
        if dt < 0: dt = 0.0 
        
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

    # 후진 감지 보정
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
        
        # -pi ~ pi 보정
        if angle > math.pi / 2: angle -= math.pi
        elif angle < -math.pi / 2: angle += math.pi
        return angle

    steer_L = wheel_steer_angle(wheel_FL)
    steer_R = wheel_steer_angle(wheel_FR)

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

# =====================
# REGISTRATION (SWITCH)
# =====================

def unregister_car_logger():
    """핸들러를 안전하게 제거합니다."""
    handlers = bpy.app.handlers.frame_change_post
    to_remove = [h for h in handlers if getattr(h, "__name__", "") == "car_logger_handler"]
    for h in to_remove:
        handlers.remove(h)
    print("[CarLogger] 로깅 중지 (핸들러 제거됨)")

def register_car_logger():
    """핸들러를 등록하고 CSV를 초기화합니다."""
    unregister_car_logger() # 중복 방지용 선제거
    
    global last_frame
    last_frame = None
    
    init_csv(OUTPUT_CSV_PATH) # CSV 헤더 작성
    
    bpy.app.handlers.frame_change_post.append(car_logger_handler)
    print("[CarLogger] 로깅 시작 (핸들러 등록됨)")

def run_batch_export(start, end):
    """현재 씬을 처음부터 끝까지 돌며 강제로 추출합니다."""
    print(f"[CarLogger] 배치 추출 시작: Frame {start} ~ {end}")
    scene = bpy.context.scene
    original_frame = scene.frame_current
    
    # 처음으로 돌리기 전 핸들러 등록
    register_car_logger()
    
    for f in range(start, end + 1):
        scene.frame_set(f)
        
    scene.frame_set(original_frame)
    print(f"[CarLogger] 배치 추출 완료!")

# =====================
# MAIN EXECUTION
# =====================

if __name__ == "__main__":
    if LOGGING_SWITCH:
        # 1. 로깅 활성화 (핸들러 등록)
        # register_car_logger() 
        
        # 2. 혹은 바로 전체 프레임 추출을 원하면 아래 함수 사용
        run_batch_export(START_FRAME, END_FRAME)
    else:
        # 로깅 끄기
        unregister_car_logger()