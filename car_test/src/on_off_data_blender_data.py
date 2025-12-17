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

OUTPUT_CSV_PATH = "E:\MK\data\drive_8_test.csv"

LOGGING_SWITCH = True 

# 네가 쓰던 forward 정의(그대로)
BLENDER_FORWARD_LOCAL = Vector((0.0, -1.0, 0.0))

# spin 축(네 디버그 기준으로 -Z가 맞다고 봤던 설정 유지)
WHEEL_SPIN_AXIS_LOCAL = Vector((0.0, 0.0, -1.0))

# 전/후진 부호 판단 deadzone (m/s)
V_LONG_EPS = 0.10

# throttle 정규화 스케일( rad/s 기준 )
THROTTLE_OMEGA_MAX = 60.0

# Blender -> Genesis 좌표 변환(네 기존 유지)
ROT_B2G = Matrix.Rotation(math.radians(90.0), 3, 'Z')
def vec_B_to_G(v_b: Vector) -> Vector: return ROT_B2G @ v_b
def mat3_B_to_G(R_b: Matrix) -> Matrix: return ROT_B2G @ R_b
def quat_B_to_G(q_b: Quaternion) -> Quaternion:
    R_g = mat3_B_to_G(q_b.to_matrix())
    return R_g.to_quaternion().normalized()

# =====================
# evaluated + cache
# =====================
def get_obj(name: str):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise RuntimeError(f'Object not found: "{name}"')
    return obj

def get_eval(obj, depsgraph):
    return obj.evaluated_get(depsgraph)

PREV = {}        # name -> {"loc": Vector, "quat": Quaternion}
PREV_SPIN = {}   # wheel_name -> prev_twist_angle_unwrapped

def reset_caches():
    PREV.clear()
    PREV_SPIN.clear()

def vel_angvel_from_eval(name: str, obj_eval, dt: float):
    loc = obj_eval.matrix_world.to_translation().copy()
    quat = obj_eval.matrix_world.to_quaternion().normalized()

    if dt <= 0.0 or name not in PREV:
        PREV[name] = {"loc": loc.copy(), "quat": quat.copy()}
        return Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0))

    prev_loc = PREV[name]["loc"]
    prev_quat = PREV[name]["quat"]

    vel = (loc - prev_loc) / dt

    dq = prev_quat.conjugated() @ quat
    dq.normalize()
    angle = dq.angle
    if angle < 1e-12:
        angvel = Vector((0.0, 0.0, 0.0))
    else:
        angvel = Vector(dq.axis) * (angle / dt)

    PREV[name] = {"loc": loc.copy(), "quat": quat.copy()}
    return vel, angvel

# =====================
# steer
# =====================
def signed_yaw_between(forward_a_world: Vector, forward_b_world: Vector) -> float:
    a = Vector((forward_a_world.x, forward_a_world.y, 0.0))
    b = Vector((forward_b_world.x, forward_b_world.y, 0.0))
    if a.length < 1e-12 or b.length < 1e-12:
        return 0.0
    a.normalize(); b.normalize()
    dot = max(-1.0, min(1.0, a.dot(b)))
    ang = math.acos(dot)
    if a.cross(b).z < 0:
        ang = -ang
    return ang

# =====================
# spin clean (stable twist)
# =====================
def unwrap_angle(prev, curr):
    if prev is None:
        return curr
    d = curr - prev
    while d > math.pi:
        curr -= 2.0 * math.pi
        d = curr - prev
    while d < -math.pi:
        curr += 2.0 * math.pi
        d = curr - prev
    return curr

def twist_angle_about_axis_stable(q: Quaternion, axis_local: Vector) -> float:
    axis = axis_local.normalized()
    v = Vector((q.x, q.y, q.z))
    proj = axis * v.dot(axis)  # q 벡터부 축 투영
    w = float(q.w)
    p = float(proj.length)
    ang = 2.0 * math.atan2(p, w)
    if ang > math.pi:
        ang -= 2.0 * math.pi
    sign = 1.0 if proj.dot(axis) >= 0.0 else -1.0
    return sign * abs(ang)

def wheel_spin_rate_clean(wheel_name: str, wheel_eval, car_eval, dt: float) -> float:
    q_car = car_eval.matrix_world.to_quaternion().normalized()
    q_w   = wheel_eval.matrix_world.to_quaternion().normalized()
    q_rel = q_car.conjugated() @ q_w
    q_rel.normalize()

    axis_world = (wheel_eval.matrix_world.to_3x3() @ WHEEL_SPIN_AXIS_LOCAL).normalized()
    axis_car_local = (car_eval.matrix_world.to_3x3().inverted() @ axis_world).normalized()

    ang = twist_angle_about_axis_stable(q_rel, axis_car_local)

    prev = PREV_SPIN.get(wheel_name, None)
    ang_u = unwrap_angle(prev, ang)

    if prev is not None:
        # prev와 가장 가까운 ±2pi 후보로 선택(급격한 튐 완화)
        cands = [ang_u, ang_u + 2*math.pi, ang_u - 2*math.pi]
        ang_u = min(cands, key=lambda a: abs(a - prev))

    spin = 0.0 if (dt <= 0.0 or prev is None) else (ang_u - prev) / dt
    PREV_SPIN[wheel_name] = float(ang_u)
    return float(spin)

# =====================
# CSV
# =====================
def init_csv(path):
    full_path = bpy.path.abspath(path)
    with open(full_path, mode="w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "time",
            "g_pos_x", "g_pos_y", "g_pos_z",
            "g_qw", "g_qx", "g_qy", "g_qz",
            "g_lin_vx", "g_lin_vy", "g_lin_vz",
            "g_ang_vx", "g_ang_vy", "g_ang_vz",
            "steer",
            "v_long",
            "spin_R",
            "throttle_raw",
            "throttle_norm"
        ])
    print(f"[CarLogger] CSV 초기화: {full_path}")

def append_row(path, row):
    full_path = bpy.path.abspath(path)
    with open(full_path, mode="a", newline="") as f:
        csv.writer(f).writerow(row)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

# =====================
# per-frame sampling
# =====================
def sample_and_write(scene, depsgraph, frame, dt):
    car = get_eval(get_obj(CAR_OBJECT_NAME), depsgraph)
    wFL = get_eval(get_obj(FRONT_LEFT_WHEEL), depsgraph)
    wFR = get_eval(get_obj(FRONT_RIGHT_WHEEL), depsgraph)
    wRL = get_eval(get_obj(REAR_LEFT_WHEEL), depsgraph)
    wRR = get_eval(get_obj(REAR_RIGHT_WHEEL), depsgraph)

    fps = scene.render.fps / scene.render.fps_base
    time_sec = frame / fps

    # pose
    loc_B = car.matrix_world.to_translation()
    rot_B = car.matrix_world.to_quaternion().normalized()

    # vel/angvel (evaluated 기반)
    v_B, w_B = vel_angvel_from_eval(CAR_OBJECT_NAME, car, dt)

    # coord convert
    pos_G = vec_B_to_G(loc_B)
    rot_G = quat_B_to_G(rot_B)
    lin_v_G = vec_B_to_G(v_B)
    ang_v_G = vec_B_to_G(w_B)

    # body forward(world)
    R_body = car.matrix_world.to_3x3()
    body_fwd_world = R_body @ BLENDER_FORWARD_LOCAL

    # steer 통일 (앞바퀴 두개 평균)
    def wheel_steer_angle(wheel_eval):
        R_w = wheel_eval.matrix_world.to_3x3()
        wheel_fwd_world = R_w @ BLENDER_FORWARD_LOCAL
        ang = signed_yaw_between(body_fwd_world, wheel_fwd_world)
        if ang > math.pi / 2:
            ang -= math.pi
        elif ang < -math.pi / 2:
            ang += math.pi
        return ang

    steer_raw = 0.5 * (wheel_steer_angle(wFL) + wheel_steer_angle(wFR))
    
    # Store raw steering angle (no normalization)
    steer = steer_raw

    # rear spin 통일
    spin_RL = wheel_spin_rate_clean(REAR_LEFT_WHEEL,  wRL, car, dt)
    spin_RR = wheel_spin_rate_clean(REAR_RIGHT_WHEEL, wRR, car, dt)
    spin_R = 0.5 * (spin_RL + spin_RR)

    # longitudinal speed (차체 전방방향 성분)
    v_long = float(v_B.dot(body_fwd_world))

    # throttle_raw = |spin_R| * sign(v_long), 단 정지 near 0은 0
    if abs(v_long) < V_LONG_EPS:
        throttle_raw = 0.0
    else:
        throttle_raw = abs(spin_R) * (1.0 if v_long > 0.0 else -1.0)

    # [-1,1] 정규화
    throttle_norm = clamp(throttle_raw / THROTTLE_OMEGA_MAX, -1.0, 1.0)

    row = [
        frame, time_sec,
        pos_G.x, pos_G.y, pos_G.z,
        rot_G.w, rot_G.x, rot_G.y, rot_G.z,
        lin_v_G.x, lin_v_G.y, lin_v_G.z,
        ang_v_G.x, ang_v_G.y, ang_v_G.z,
        steer,  # Raw steering angle in radians
        v_long,
        spin_R,
        throttle_raw,
        throttle_norm
    ]
    append_row(OUTPUT_CSV_PATH, row)

def export_all_frames(start_frame=None, end_frame=None):
    scene = bpy.context.scene

    s = scene.frame_start if start_frame is None else int(start_frame)
    e = scene.frame_end   if end_frame   is None else int(end_frame)

    init_csv(OUTPUT_CSV_PATH)
    reset_caches()

    fps = scene.render.fps / scene.render.fps_base
    cur = scene.frame_current

    print(f"[CarLogger] Export 시작: {s} ~ {e} (fps={fps})")

    prev_f = None
    for f in range(s, e + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()

        dt = 0.0 if prev_f is None else (f - prev_f) / fps
        sample_and_write(scene, depsgraph, f, dt)
        prev_f = f

    scene.frame_set(cur)
    bpy.context.view_layer.update()
    print(f"[CarLogger] 완료! -> {OUTPUT_CSV_PATH}")

# =====================
# HANDLER LOGIC
# =====================
last_frame_handler = None

def car_logger_handler(scene):
    global last_frame_handler
    
    # 씬/의존성 그래프 가져오기
    depsgraph = bpy.context.evaluated_depsgraph_get()
    frame = scene.frame_current
    
    # FPS 계산
    fps = scene.render.fps / scene.render.fps_base
    
    # dt 계산
    if last_frame_handler is None:
        dt = 0.0
    else:
        dt = (frame - last_frame_handler) / fps
        if dt < 0: dt = 0.0 # 프레임 루프/점프 시 방어
        
    last_frame_handler = frame
    
    # 샘플링 및 저장
    sample_and_write(scene, depsgraph, frame, dt)

def register_logger():
    unregister_logger() # 기존 핸들러 제거
    init_csv(OUTPUT_CSV_PATH)
    reset_caches()
    
    global last_frame_handler
    last_frame_handler = None
    
    bpy.app.handlers.frame_change_post.append(car_logger_handler)
    print(f"[CarLogger] 핸들러 등록됨. Play를 누르면 '{OUTPUT_CSV_PATH}'에 기록됩니다.")

def unregister_logger():
    handlers = bpy.app.handlers.frame_change_post
    to_remove = [h for h in handlers if getattr(h, "__name__", "") == "car_logger_handler"]
    for h in to_remove:
        handlers.remove(h)
    print("[CarLogger] 핸들러 제거됨.")

if __name__ == "__main__":
    # 로깅 스위치 (Config 상단에 추가 권장, 여기서는 하드코딩 혹은 상단 변수 참조)
    # 사용자가 직접 상단에서 LOGGING_SWITCH = True/False 조절한다고 가정
    
    
    if LOGGING_SWITCH:
        register_logger()
        # 만약 배치 추출(강제 순회)을 원하면 아래 주석 해제
        # export_all_frames(1, 250)
    else:
        unregister_logger()
