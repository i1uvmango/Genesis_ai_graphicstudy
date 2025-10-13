## Project Structure

```
car_test/
├─ src/
│  ├─ __init__.py
│  └─ car_test.py
├─ urdf/
│  └─ car.urdf
├─ docs/
│  ├─ blender.md
│  ├─ genesis_car_urdf.md
│  └─ vehicle_blending.md
├─ res/  # assets (images, videos, blend/obj, gif)
├─ requirements.txt
└─ .gitignore
```

## Run

- Install deps (Python 3.10+ recommended):
```
pip install -r requirements.txt
# Install Genesis following vendor docs
```

- Run simulation (from repo root):
```
python -m src.car_test --vis
```

URDF path used in `src/car_test.py` is `../urdf/car.urdf`.

