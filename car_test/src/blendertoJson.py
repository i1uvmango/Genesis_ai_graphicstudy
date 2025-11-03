import bpy, json, os
from mathutils import Vector

def world_axis_from_object(obj, local_axis='Z'):
    R = obj.matrix_world.to_3x3()
    axis = {'X': Vector((1,0,0)), 'Y': Vector((0,1,0)), 'Z': Vector((0,0,1))}[local_axis.upper()]
    w = R @ axis
    w.normalize()
    return [round(v, 4) for v in w]

data = {"links": [], "joints": []}

# 🔹 1️⃣ 링크 정보 추출 (Rigid Body가 있는 모든 오브젝트)
for obj in bpy.data.objects:
    if not obj.rigid_body:
        continue
    rb = obj.rigid_body
    data["links"].append({
        "name": obj.name,
        "mass": rb.mass,
        "friction": rb.friction,
        "restitution": rb.restitution,
        "linear_damping": rb.linear_damping,
        "angular_damping": rb.angular_damping,
        "collision_shape": rb.collision_shape,
        "origin": {
            "xyz": [round(x, 4) for x in obj.location],
            "rpy": [round(x, 4) for x in obj.rotation_euler]
        }
    })

# 🔹 2️⃣ 조인트 정보 추출 (Rigid Body Constraints)
for obj in bpy.data.objects:
    rbc = obj.rigid_body_constraint
    if not rbc:
        continue

    j = {
        "name": obj.name,
        "type": rbc.type,
        "object1": rbc.object1.name if rbc.object1 else None,
        "object2": rbc.object2.name if rbc.object2 else None,
        "origin": {
            "xyz": [round(x, 4) for x in obj.location],
            "rpy": [round(x, 4) for x in obj.rotation_euler]
        }
    }

    if rbc.type == 'HINGE' or rbc.type == 'MOTOR':
        j["axis_world"] = world_axis_from_object(obj, 'Z')

    if hasattr(rbc, "use_motor_ang") and rbc.use_motor_ang:
        j["motor"] = {
            "velocity": rbc.motor_ang_target_velocity,
            "max_impulse": rbc.motor_ang_max_impulse
        }

    data["joints"].append(j)

# 🔹 3️⃣ 저장
save_path = os.path.join(bpy.path.abspath("//"), "car_full_physics.json")
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"✅ Blender → URDF 변환용 데이터 저장 완료: {save_path}")
