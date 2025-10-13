import genesis as gs
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    # ✅ Genesis 초기화
    gs.init(backend=gs.gpu, logging_level="info")

    # ✅ Scene 생성
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3),
        show_viewer=args.vis
    )

    # ✅ Ground (평면) 추가
    ground = gs.morphs.Plane()
    scene.add_entity(ground)

    # ✅ URDF 불러오기 (차체 + 바퀴)
    car = gs.morphs.URDF(
        file="./car.urdf",
        fixed=False,
        pos=(0, 0, -0.6)  # ✅ 초기 위치 보정 (공중 날림 방지)
    )
    scene.add_entity(car)

    # ✅ Scene 빌드
    scene.build()

    # ✅ 시뮬레이션 루프
    for i in range(2000):
        scene.step()
        if args.vis:
            scene.viewer.update()

    print("✅ Simulation finished successfully.")

if __name__ == "__main__":
    main()
