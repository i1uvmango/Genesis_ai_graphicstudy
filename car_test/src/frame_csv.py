import bpy, csv, os

# 🔹 설정
output_path = os.path.join(bpy.path.abspath("//"), "car_frame_data.csv")

# 🔹 추출 대상 오브젝트 이름
car_name = "Car_Body"   # 필요에 따라 변경 (예: Car_Body, Vehicle, jeep 등)
car_obj = bpy.data.objects.get(car_name)

if not car_obj or not car_obj.rigid_body:
    raise ValueError(f"Rigid Body가 설정된 '{car_name}' 오브젝트를 찾을 수 없습니다.")

# 🔹 시뮬레이션 구간 설정
start_frame = bpy.context.scene.frame_start
end_frame = bpy.context.scene.frame_end

# 🔹 CSV 파일 헤더
header = [
    "frame",
    "time_sec",
    "pos_x", "pos_y", "pos_z",
    "rot_x", "rot_y", "rot_z",
    "lin_vel_x", "lin_vel_y", "lin_vel_z",
    "ang_vel_x", "ang_vel_y", "ang_vel_z"
]

# 🔹 CSV 파일 생성
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    # 프레임별로 물리 데이터 샘플링
    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

        loc = car_obj.location
        rot = car_obj.rotation_euler
        rb = car_obj.rigid_body

        if rb:
            lin_vel = rb.linear_velocity
            ang_vel = rb.angular_velocity
        else:
            lin_vel = (0, 0, 0)
            ang_vel = (0, 0, 0)

        time_sec = frame / bpy.context.scene.render.fps

        writer.writerow([
            frame,
            round(time_sec, 4),
            round(loc.x, 4), round(loc.y, 4), round(loc.z, 4),
            round(rot.x, 4), round(rot.y, 4), round(rot.z, 4),
            round(lin_vel.x, 4), round(lin_vel.y, 4), round(lin_vel.z, 4),
            round(ang_vel.x, 4), round(ang_vel.y, 4), round(ang_vel.z, 4)
        ])

print(f"✅ 차량 프레임별 데이터 저장 완료: {output_path}")
