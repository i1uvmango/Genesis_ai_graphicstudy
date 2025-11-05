import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

# === 1️⃣ 입력 / 출력 경로 설정 ===
input_path = "car_structure.json"
output_path = "car_from_json.urdf"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

objects = data["objects"]

# === 2️⃣ 링크 이름으로 빠르게 접근하기 위한 dict ===
obj_dict = {obj["name"]: obj for obj in objects}

# === 3️⃣ URDF 루트 노드 생성 ===
robot = ET.Element("robot", {"name": "car_model"})

# === 4️⃣ 링크 노드 생성 ===
for obj in objects:
    link = ET.SubElement(robot, "link", {"name": obj["name"]})

    # inertial (질량 / 관성)
    inertial = ET.SubElement(link, "inertial")
    mass = ET.SubElement(inertial, "mass", {"value": str(obj.get("mass") or 1.0)})
    inertia = ET.SubElement(inertial, "inertia", {
        "ixx": "1.0", "ixy": "0.0", "ixz": "0.0",
        "iyy": "1.0", "iyz": "0.0", "izz": "1.0"
    })

    # visual (시각 모델)
    visual = ET.SubElement(link, "visual")
    geom_v = ET.SubElement(visual, "geometry")
    ET.SubElement(geom_v, "box", {"size": "0.1 0.1 0.1"})  # placeholder geometry

    # collision (충돌 모델)
    collision = ET.SubElement(link, "collision")
    geom_c = ET.SubElement(collision, "geometry")
    ET.SubElement(geom_c, "box", {"size": "0.1 0.1 0.1"})

# === 5️⃣ Joint 노드 생성 ===
joint_id = 0
for obj in objects:
    parent_name = obj.get("parent")
    if parent_name:
        parent = obj_dict[parent_name]
        # joint 생성
        j_name = f"joint_{joint_id}"
        joint_id += 1

        joint = ET.SubElement(robot, "joint", {
            "name": j_name,
            "type": "revolute"
        })

        ET.SubElement(joint, "parent", {"link": parent_name})
        ET.SubElement(joint, "child", {"link": obj["name"]})

        # origin (xyz, rpy)
        loc = obj.get("location", [0, 0, 0])
        rot = obj.get("rotation_euler", [0, 0, 0])
        origin = ET.SubElement(joint, "origin", {
            "xyz": f"{loc[0]} {loc[1]} {loc[2]}",
            "rpy": f"{rot[0]} {rot[1]} {rot[2]}"
        })

        # 회전축 기본값 (Z축 회전)
        ET.SubElement(joint, "axis", {"xyz": "0 0 1"})

        # 제한 (기본 범위)
        limit = ET.SubElement(joint, "limit", {
            "lower": "-3.14", "upper": "3.14",
            "effort": "100", "velocity": "10"
        })

# === 6️⃣ Pretty print & 저장 ===
xml_str = ET.tostring(robot, encoding="utf-8")
xml_pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(xml_pretty)

print(f"✅ URDF exported successfully → {output_path}")
