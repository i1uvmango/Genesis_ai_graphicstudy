import json, os

# 🔹 네 JSON 파일 이름 맞게 수정
json_file = r"E:\MK\car_full_physics.json"  


with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

save_path = os.path.join(os.getcwd(), "car.urdf")

def vec_to_str(v): return f"{v[0]} {v[1]} {v[2]}"

with open(save_path, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0"?>\n<robot name="car">\n\n')

    for link in data["links"]:
        name = link["name"]
        mass = link["mass"]
        friction = link["friction"]
        shape = link["collision_shape"].lower()
        xyz = vec_to_str(link["origin"]["xyz"])
        rpy = vec_to_str(link["origin"]["rpy"])

        f.write(f'  <link name="{name}">\n')
        f.write('    <inertial>\n')
        f.write(f'      <mass value="{mass}"/>\n')
        f.write('      <inertia ixx="1" iyy="1" izz="1"/>\n')
        f.write('    </inertial>\n')
        f.write('    <collision>\n      <geometry>\n')
        if "box" in shape:
            f.write('        <box size="2 1 0.5"/>\n')
        elif "cylinder" in shape:
            f.write('        <cylinder radius="0.3" length="0.1"/>\n')
        else:
            f.write('        <sphere radius="0.3"/>\n')
        f.write(f'      </geometry>\n      <origin xyz="{xyz}" rpy="{rpy}"/>\n    </collision>\n')
        f.write(f'  </link>\n\n')

    for j in data["joints"]:
        parent = j["object1"]
        child = j["object2"]
        jtype = j["type"].lower()
        xyz = vec_to_str(j["origin"]["xyz"])
        rpy = vec_to_str(j["origin"]["rpy"])
        axis = vec_to_str(j.get("axis_world", [0,1,0]))
        f.write(f'  <joint name="{j["name"]}" type="{jtype}">\n')
        f.write(f'    <parent link="{parent}"/>\n')
        f.write(f'    <child link="{child}"/>\n')
        f.write(f'    <origin xyz="{xyz}" rpy="{rpy}"/>\n')
        f.write(f'    <axis xyz="{axis}"/>\n')
        if "motor" in j:
            f.write(f'    <limit effort="{j["motor"]["max_impulse"]}" velocity="{j["motor"]["velocity"]}"/>\n')
        f.write('  </joint>\n\n')

    f.write('</robot>\n')

print(f"✅ URDF 생성 완료 → {save_path}")
