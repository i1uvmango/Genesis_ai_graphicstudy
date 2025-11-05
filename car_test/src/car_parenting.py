############blender 실행용##################

####객체 이름 확인 및 parenting 확인

import bpy

print("=" * 80)
print("🎯 Blender Object & Parenting Structure")
print("=" * 80)

scene = bpy.context.scene
print(f"📁 Scene: {scene.name}")
print(f"🧱 Object count: {len(scene.objects)}\n")

# 전체 오브젝트 출력
for obj in scene.objects:
    print(f"🧩 Object: {obj.name}")
    print(f"   • Type: {obj.type}")

    # 부모 관계
    if obj.parent:
        print(f"   • Parent: {obj.parent.name}")
    else:
        print(f"   • Parent: None (Root object)")

    # 자식 관계
    if obj.children:
        print(f"   • Children: {[child.name for child in obj.children]}")
    else:
        print(f"   • Children: None")

    # 본(Armature) 구조
    if obj.type == 'ARMATURE':
        print(f"   • Armature bones:")
        for bone in obj.data.bones:
            parent_name = bone.parent.name if bone.parent else "None"
            child_names = [child.name for child in bone.children]
            print(f"      - Bone: {bone.name}")
            print(f"        Parent bone: {parent_name}")
            print(f"        Child bones: {child_names if child_names else 'None'}")

    # 메시 정보
    if obj.type == 'MESH':
        mesh = obj.data
        print(f"   • Mesh: {mesh.name}")
        print(f"      - Vertices: {len(mesh.vertices)}")
        print(f"      - Faces: {len(mesh.polygons)}")

    print("-" * 80)

print("✅ Done. Parenting and structure summary complete.")
