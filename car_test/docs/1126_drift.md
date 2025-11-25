## Blender Drift to Genesis
* RBC Pro 사용

![](../res/path_point.png)
youtube reference: `https://www.youtube.com/watch?v=rrI6wzFquhU` 

```
import bpy
import csv
import math
import os
from mathutils import Vector, Matrix, Quaternion

########################################
# ★ 사용자 설정
########################################
LOGGING_SWITCH = True 
USE_ACTIVE_OBJECT = True 
BODY_NAME = "Corvette.Vehicle Body.RB" 

# ---------------------------------------------------------
# 저장 경로
# ---------------------------------------------------------
if bpy.data.is_saved:
    LOG_PATH = bpy.path.abspath("//drift.csv")
else:
    LOG_PATH = os.path.join(os.path.expanduser("~"), "drift.csv")

FPS = bpy.context.scene.render.fps
DT = 1.0 / FPS

########################################
# 좌표계 변환
########################################
ROT_B2G = Matrix.Rotation(math.radians(90.0), 3, 'Z')
def vec_B_to_G(v):
    return ROT_B2G @ v

########################################
# 전역 상태
########################################
_state = {
    "initialized": False,
    "target_obj_name": None, 
    "wheels": {}, 
    "prev_loc": None,
    "prev_rot": None,
    "prev_rot_matrix": None,
    "prev_wheel_rot": {"rl": None, "rr": None},
    "prev_frame": None,
    "csv": None,
    "writer": None
}

########################################
# [핵심] 바퀴 자동 찾기
########################################
def find_wheels_automatically():
    print("----------------------------------------")
    print("[6DoF] 바퀴 자동 검색 및 연결 중...")
    
    found_wheels = {"FL": None, "FR": None, "RL": None, "RR": None}
    
    for obj in bpy.data.objects:
        name = obj.name.lower()
        if "tire" in name or "wheel" in name:
            is_front = "front" in name or " f" in name
            is_back = "back" in name or "rear" in name or " b" in name
            is_left = "left" in name or "_l" in name or " l" in name
            is_right = "right" in name or "_r" in name or " r" in name
            
            if is_front and is_left: found_wheels["FL"] = obj
            elif is_front and is_right: found_wheels["FR"] = obj
            elif is_back and is_left: found_wheels["RL"] = obj
            elif is_back and is_right: found_wheels["RR"] = obj
            
    for key, obj in found_wheels.items():
        if obj: print(f"  ✅ {key}: {obj.name}")
        else: print(f"  ❌ {key}: 못 찾음 (이름 확인 필요)")
            
    print("----------------------------------------")
    return found_wheels

########################################
# ★ [수정됨] 쿼터니안 차분 계산 로직
########################################

def get_quaternion_difference(q_base, q_target):
    """
    두 쿼터니안 사이의 '차이' 회전을 구합니다.
    (Base 좌표계에서 Target으로 가기 위한 회전량)
    Diff = Q_base.conjugated() @ Q_target
    """
    return q_base.conjugated() @ q_target

def get_true_steering(wheels, body_obj):
    """
    Steering = (차체 쿼터니안)과 (앞바퀴 쿼터니안)의 차이(Difference)
    결과: 차체 로컬 좌표계 기준 바퀴의 회전각 (Z축 = 조향)
    """
    fl = wheels["FL"]
    fr = wheels["FR"]
    
    if fl is None or fr is None or body_obj is None:
        return 0.0
    
    q_body = body_obj.matrix_world.to_quaternion()
    
    # 1. 왼쪽 바퀴의 차체 상대 회전 (Quaternion Difference)
    q_fl_world = fl.matrix_world.to_quaternion()
    q_rel_fl = get_quaternion_difference(q_body, q_fl_world)
    steer_fl = q_rel_fl.to_euler("XYZ").z
    
    # 2. 오른쪽 바퀴의 차체 상대 회전
    q_fr_world = fr.matrix_world.to_quaternion()
    q_rel_fr = get_quaternion_difference(q_body, q_fr_world)
    steer_fr = q_rel_fr.to_euler("XYZ").z
    
    # 평균값 반환
    return 0.5 * (steer_fl + steer_fr)

def get_true_throttle(wheels):
    """
    Throttle = (이전 프레임 바퀴 쿼터니안)과 (현재 바퀴 쿼터니안)의 차이(Difference)
    결과: 바퀴의 회전 속도 (Angular Velocity) -> 스로틀 근사치
    """
    rl = wheels["RL"]
    rr = wheels["RR"]
    if rl is None or rr is None: return 0.0

    rl_q_curr = rl.matrix_world.to_quaternion()
    rr_q_curr = rr.matrix_world.to_quaternion()

    # 첫 프레임 처리
    if _state["prev_wheel_rot"]["rl"] is None:
        _state["prev_wheel_rot"]["rl"] = rl_q_curr.copy()
        _state["prev_wheel_rot"]["rr"] = rr_q_curr.copy()
        return 0.0

    def calculate_spin_speed(q_prev, q_curr):
        # 1. 쿼터니안 차분 (시간에 따른 변화량)
        # diff = q_prev^-1 * q_curr
        q_diff = get_quaternion_difference(q_prev, q_curr)
        
        # 2. 각도(Angle)와 축(Axis) 추출
        angle = q_diff.angle
        axis = q_diff.axis
        
        # 3. 아주 작은 회전은 무시 (노이즈 제거)
        if abs(angle) < 1e-6:
            return 0.0
            
        # 4. 각속도 크기 (rad/s)
        angular_speed = angle / DT
        
        # 5. 방향 판별 (바퀴가 앞으로 구르는지 뒤로 구르는지)
        # 바퀴의 로컬 Y축(일반적인 굴림 축)과 회전축의 내적을 구함
        # 로컬 Y축으로 투영
        # 주의: 쿼터니안 차분은 월드 기준 변화량이므로, 로컬 축으로 변환 필요할 수 있음
        # 하지만 단순 스칼라 속도를 원한다면 angular_speed만 써도 됨.
        # 여기서는 부호(방향)를 살리기 위해 로컬 변환 수행.
        
        # q_curr(현재 바퀴 자세)를 이용해 월드 회전축을 바퀴 로컬 회전축으로 변환
        axis_local = q_curr.conjugated() @ axis
        
        # 바퀴는 보통 로컬 X 또는 Y축을 중심으로 회전함 (블렌더 리깅에 따라 다름)
        # 대부분 Y축 아니면 X축. 둘 중 큰 값을 취함.
        spin_val = axis_local.y if abs(axis_local.y) > abs(axis_local.x) else axis_local.x
        
        # 최종 속도 (부호 포함)
        return angular_speed * (1.0 if spin_val > 0 else -1.0)

    rl_speed = calculate_spin_speed(_state["prev_wheel_rot"]["rl"], rl_q_curr)
    rr_speed = calculate_spin_speed(_state["prev_wheel_rot"]["rr"], rr_q_curr)
    
    # 상태 업데이트
    _state["prev_wheel_rot"]["rl"] = rl_q_curr.copy()
    _state["prev_wheel_rot"]["rr"] = rr_q_curr.copy()
    
    return 0.5 * (rl_speed + rr_speed)

########################################
# 일반 각속도 계산 (차체용)
########################################
def compute_ang_vel_body(prev_q, curr_q):
    q_diff = get_quaternion_difference(prev_q, curr_q)
    angle = q_diff.angle
    axis = q_diff.axis
    if abs(angle) < 1e-8: return Vector((0, 0, 0))
    return axis * (angle / DT)

########################################
# 프레임 핸들러
########################################
def frame_handler(scene, depsgraph):
    if not _state["initialized"]:
        # 1. 차체 찾기
        target_obj = None
        if USE_ACTIVE_OBJECT: target_obj = bpy.context.active_object
        if target_obj is None: target_obj = bpy.data.objects.get(BODY_NAME)
        
        if target_obj is None:
            print(f"[CRITICAL] 차체를 찾을 수 없습니다!")
            return

        _state["target_obj_name"] = target_obj.name
        
        # 2. 바퀴 자동 찾기
        _state["wheels"] = find_wheels_automatically()
        
        # 3. CSV 초기화
        curr_loc = target_obj.matrix_world.translation.copy()
        curr_rot = target_obj.matrix_world.to_quaternion().copy()
        
        _state["prev_loc"] = curr_loc
        _state["prev_rot"] = curr_rot
        _state["prev_frame"] = scene.frame_current
        
        try:
            f = open(LOG_PATH, "w", newline="")
            w = csv.writer(f)
            w.writerow([
                "frame", "time",
                "g_lin_vx","g_lin_vy","g_lin_vz",
                "g_ang_vx","g_ang_vy","g_ang_vz",
                "steering_rad","wheel_speed_rad_s"
            ])
            _state["csv"] = f
            _state["writer"] = w
            _state["initialized"] = True
            print(f"[6DoF] 로깅 시작: {LOG_PATH}")
        except:
            print(f"[Error] 파일 열기 실패")
            return
        return

    # 매 프레임 로직
    body = bpy.data.objects.get(_state["target_obj_name"])
    if body is None: return

    curr_loc = body.matrix_world.translation.copy()
    curr_rot = body.matrix_world.to_quaternion().copy()
    curr_rot_matrix = curr_rot.to_matrix()

    if _state.get("prev_frame") is not None:
        if scene.frame_current < _state["prev_frame"]:
            _state["prev_loc"] = curr_loc
            _state["prev_rot"] = curr_rot
            _state["prev_frame"] = scene.frame_current
            return
    _state["prev_frame"] = scene.frame_current

    # 1. 선속도
    world_velocity = (curr_loc - _state["prev_loc"]) / DT
    local_velocity_blender = curr_rot_matrix.transposed() @ world_velocity
    local_velocity_genesis = vec_B_to_G(local_velocity_blender)
    
    # 2. 각속도 (차체)
    ang_vel_world = compute_ang_vel_body(_state["prev_rot"], curr_rot)
    ang_vel_local_blender = curr_rot_matrix.transposed() @ ang_vel_world
    ang_vel_local_genesis = vec_B_to_G(ang_vel_local_blender)
    
    # 3. ★ Steering & Throttle (쿼터니안 차분 적용)
    steering = get_true_steering(_state["wheels"], body)
    throttle = get_true_throttle(_state["wheels"]) # 각속도(rad/s)

    frame = scene.frame_current
    time_sec = frame * DT

    if _state["writer"]:
        _state["writer"].writerow([
            frame, f"{time_sec:.6f}",
            f"{local_velocity_genesis.x:.6f}", f"{local_velocity_genesis.y:.6f}", f"{local_velocity_genesis.z:.6f}",
            f"{ang_vel_local_genesis.x:.6f}", f"{ang_vel_local_genesis.y:.6f}", f"{ang_vel_local_genesis.z:.6f}",
            f"{steering:.6f}", f"{throttle:.6f}"
        ])

    _state["prev_loc"] = curr_loc
    _state["prev_rot"] = curr_rot

########################################
# 등록/해제
########################################
def unregister_car_logger():
    if _state["csv"]: 
        try: _state["csv"].close()
        except: pass
    
    hs = bpy.app.handlers.frame_change_post
    rm = [h for h in hs if h.__name__ == "frame_handler"]
    for r in rm: hs.remove(r)
    
    _state["initialized"] = False

def register_car_logger():
    unregister_car_logger()
    bpy.app.handlers.frame_change_post.append(frame_handler)
    print(f"[6DoF] 준비 완료.")

if __name__ == "__main__":
    if LOGGING_SWITCH: register_car_logger()
    else: unregister_car_logger()
```