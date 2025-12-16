import bpy
import csv
import math
from mathutils import Vector, Matrix, Quaternion

# =====================
# ★ 사용자 설정 (CONFIG)
# =====================

# ▶ 로깅 ON/OFF 스위치
LOGGING_SWITCH = True

# ▶ 추출할 프레임 범위
START_FRAME = 1
END_FRAME = 250

# ▶ 객체 이름 설정 (blender_script.py와 동일하게 맞춤)
CAR_OBJECT_NAME   = "Corvette.Vehicle Body.RB"
FRONT_LEFT_WHEEL  = "Corvette.Vehicle Body.0.FL1_Wheel.RB"
REAR_LEFT_WHEEL   = "Corvette.Vehicle Body.1.BL1_Wheel.RB"
REAR_RIGHT_WHEEL  = "Corvette.Vehicle Body.1.BR0_Wheel.RB"

# ▶ 저장 경로
OUTPUT_CSV_PATH = "E:/MK/data/drive_integrated.csv"

# ▶ 파라미터 설정
MAX_STEER_RAD = math.radians(30.0)
MAX_SPIN_RATE = 50.0 # 30km/h ≈ 8.3m/s. r=0.35m일 때 약 23.8rad/s. 여유있게 25.0 설정
FPS = 30.0

# ▶ 벡터 설정
BLENDER_FORWARD_LOCAL = Vector((0.0, -1.0, 0.0))
WHEEL_SPIN_AXIS_LOCAL = Vector((1.0, 0.0, 0.0))

# =====================
# MATH HELPERS (from blender_script.py)
# =====================

def get_obj(name):
    if name and name in bpy.data.objects:
        return bpy.data.objects[name]
    return None

def signed_yaw_between(forward_a_world: Vector, forward_b_world: Vector) -> float:
    """두 벡터 사이의 Signed Angle (Z축 기준) 계산"""
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
    휠 스핀 속도 계산 - 쿼터니언 shortest path 버그 수정 버전 (blender_script.py 원본 로직)
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
    
    # 비정상 값 클리핑 (Raw Value 기준)
    if abs(spin_rate) > MAX_SPIN_RATE:
        spin_rate = math.copysign(MAX_SPIN_RATE, spin_rate)

    wheel_obj["_prev_wheel_rot_quat_B"] = curr_rot_corrected[:]
    return spin_rate

# =====================
# HANDLER LOGIC
# =====================

last_frame_handler = None

def init_csv():
    full_path = bpy.path.abspath(OUTPUT_CSV_PATH)
    try:
        with open(full_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "steering", "throttle"])
        print(f"[IntegratedLogger] CSV 초기화: {full_path}")
    except Exception as e:
        print(f"[Error] CSV 초기화 실패: {e}")

def append_row(row):
    full_path = bpy.path.abspath(OUTPUT_CSV_PATH)
    try:
        with open(full_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print(f"[Error] Row 추가 실패: {e}")

def car_logger_handler(scene):
    global last_frame_handler, FPS
    
    frame = scene.frame_current
    FPS = float(scene.render.fps)
    
    car = get_obj(CAR_OBJECT_NAME)
    wheel_fl = get_obj(FRONT_LEFT_WHEEL)
    wheel_rl = get_obj(REAR_LEFT_WHEEL)
    wheel_rr = get_obj(REAR_RIGHT_WHEEL)
    
    if not car or not wheel_fl or not wheel_rl or not wheel_rr:
        return

    dt = 1.0 / FPS if FPS > 0 else 0.033

    # 1. Steering (Front Left)
    R_body = car.matrix_world.to_3x3()
    body_fwd_world = R_body @ BLENDER_FORWARD_LOCAL
    
    R_wheel = wheel_fl.matrix_world.to_3x3()
    wheel_fwd_world = R_wheel @ BLENDER_FORWARD_LOCAL
    
    raw_steer_angle = signed_yaw_between(body_fwd_world, wheel_fwd_world)
    
    # -pi ~ pi 보정
    if raw_steer_angle > math.pi / 2: raw_steer_angle -= math.pi
    elif raw_steer_angle < -math.pi / 2: raw_steer_angle += math.pi
    
    # 정규화 및 클리핑
    steer_norm = raw_steer_angle / MAX_STEER_RAD
    steer_norm = max(-1.0, min(1.0, steer_norm))
    
    # 2. Throttle (Average of Rear Left & Rear Right)
    spin_rl = get_wheel_spin_rate_B_fixed(wheel_rl, dt)
    spin_rr = get_wheel_spin_rate_B_fixed(wheel_rr, dt)
    
    # 정규화 (각각 정규화 후 평균)
    throttle_l = spin_rl / MAX_SPIN_RATE
    throttle_r = spin_rr / MAX_SPIN_RATE
    
    # 클리핑
    throttle_l = max(-1.0, min(1.0, throttle_l))
    throttle_r = max(-1.0, min(1.0, throttle_r))
    
    # 평균
    throttle_avg = (throttle_l + throttle_r) * 0.5
        
    row = [frame, steer_norm, throttle_avg]
    append_row(row)
    
    if frame % 10 == 0:
        print(f"[Log] F{frame}: Steer={steer_norm:.3f}, Thr={throttle_avg:.3f}")

def register_logger():
    unregister_logger()
    init_csv()
    bpy.app.handlers.frame_change_post.append(car_logger_handler)
    print("[IntegratedLogger] 로깅 핸들러 등록됨. 'Play'를 누르면 기록됩니다.")

def unregister_logger():
    handlers = bpy.app.handlers.frame_change_post
    to_remove = [h for h in handlers if getattr(h, "__name__", "") == "car_logger_handler"]
    for h in to_remove:
        handlers.remove(h)
    print("[IntegratedLogger] 로깅 핸들러 제거됨.")

# =====================
# EXECUTION
# =====================

if __name__ == "__main__":
    if LOGGING_SWITCH:
        register_logger()
    else:
        unregister_logger()
