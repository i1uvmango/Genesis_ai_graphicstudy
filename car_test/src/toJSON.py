import bpy
import json
import os

# === 1️⃣ 기본 설정 ===
output_path = os.path.join(bpy.path.abspath("//"), "car_structure.json")

def get_obj_info(obj):
    """각 오브젝트의 기본 정보 추출"""
    info = {
        "name": obj.name,
        "type": obj.type,
        "location": list(round(c, 6) for c in obj.location[:]),
        "rotation_euler": list(round(r, 6) for r in obj.rotation_euler[:]),
        "parent": obj.parent.name if obj.parent else None,
        "children": [child.name for child in obj.children],
    }

    # Rigid Body 정보 (있을 때만)
    if obj.rigid_body:
        info["mass"] = round(obj.rigid_body.mass, 6)
        info["friction"] = round(obj.rigid_body.friction, 6)
        info["restitution"] = round(obj.rigid_body.restitution, 6)
        info["collision_shape"] = obj.rigid_body.collision_shape
    else:
        info["mass"] = None

    return info

# === 2️⃣ 씬 전체 순회 ===
scene = bpy.context.scene
objects_data = []

for obj in scene.objects:
    if obj.type == 'MESH':  # mesh만 추출
        objects_data.append(get_obj_info(obj))

# === 3️⃣ JSON 구조로 저장 ===
data = {
    "scene_name": scene.name,
    "num_objects": len(objects_data),
    "objects": objects_data,
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"✅ JSON exported successfully → {output_path}")
