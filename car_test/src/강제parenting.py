
##################blender 강제 parenting##############

import bpy
from mathutils import Matrix

# 1) 반드시 Object Mode
if bpy.ops.object.mode_set.poll():
    bpy.ops.object.mode_set(mode='OBJECT')

# 2) 이름 매핑 (필요하면 여기서 바꿔도 됩니다)
PARENT_ROOT = "bone_body"
PAIRS = [
    ("bone_fl", "wheel_fl"),
    ("bone_fr", "wheel_fr"),
    ("bone_rl", "wheel_rl"),
    ("bone_rr", "wheel_rr"),
]
EXTRA_CHILD = ("bone_body", "car_body")  # car_body를 bone_body에 매달기

def get_obj(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        print(f"⚠️  객체를 찾을 수 없음: {name}")
    return obj

def safe_parent(child, parent):
    # 선택/잠금/숨김 해제
    for o in (child, parent):
        if o is None:
            return False
        o.hide_set(False)
        o.hide_viewport = False
        o.hide_select = False

    # 기존 부모 해제 후 'Keep Transform' 방식으로 부모 지정
    child_parent_before = child.parent.name if child.parent else None
    child.matrix_world = child.matrix_world  # 보정용 (명시적)
    child.parent = parent
    # Keep Transform: parent inverse 설정
    child.matrix_parent_inverse = parent.matrix_world.inverted()

    print(f"✅ {child.name} → {parent.name} (이전 부모: {child_parent_before})")
    return True

def print_tree():
    print("============================================================")
    print("🎯 Blender Object Hierarchy (Parent → Child)")
    print("============================================================")
    def rec(o, indent=0):
        print("   " * indent + f"🧩 {o.name} ({o.type})")
        for c in o.children:
            rec(c, indent+1)
    roots = [o for o in bpy.context.scene.objects if o.parent is None]
    # 보기 쉽게 root를 이름순으로
    for r in sorted(roots, key=lambda x: x.name):
        rec(r)
    print("============================================================")
    print("✅ Done. Tree view complete.")

# 3) 실제 작업
root = get_obj(PARENT_ROOT)
if root:
    # car_body → bone_body
    c, p = get_obj(EXTRA_CHILD[1]), get_obj(EXTRA_CHILD[0])
    if c and p: safe_parent(c, p)

    # 각 바퀴 본과 메시 연결은 이미 되어 있다면 건너뛰고, 아니면 시도
    for bone_name, wheel_name in PAIRS:
        bone = get_obj(bone_name)
        wheel = get_obj(wheel_name)
        if wheel and wheel.parent != bone:
            safe_parent(wheel, bone)

    # 바퀴 본들을 body에 매달기
    for bone_name, _ in PAIRS:
        bone = get_obj(bone_name)
        if bone and bone.parent != root:
            safe_parent(bone, root)

# 4) 결과 트리 출력
print_tree()
