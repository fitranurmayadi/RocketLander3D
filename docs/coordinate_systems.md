# Coordinate System & Export Notes

## FreeCAD vs PyBullet (URDF)

### 1. Global Coordinate System
- **+Z**: UP (Gravity is -Z)
- **+X**: Forward/Front
- **+Y**: Right/Side

### 2. The Relationship between Placement and Joint Origin
In our `generate_urdf.py` script:
1.  We read the `Rotation (Quaternion)` and `Position` from FreeCAD.
2.  We apply this directly to the `<joint><origin .../></joint>` tag.
3.  **Crucial Insight:** This rotates the **Joint Frame**.
    - If a leg is rotated 90 degrees around Z in FreeCAD to face "Right", the **Joint Frame's X axis points Right**.
    - The **Joint Frame's Y axis** points Back (relative to the global frame).

### 3. Axis Definition (The "Propeller" Bug)
The `<axis xyz="...">` tag in URDF is defined in the **Local Joint Frame**, not the Global Frame.

- **Mistake**: We calculated the tangent vector in Global Coordinates (e.g., `-1 0 0`) and put it in the axis tag.
- **Result**: Because the Joint Frame was *also* rotated, applying a "Global X" vector into a "Rotated Frame" meant we were actually selecting the "Local Radial" axis (pointing into the rocket), causing the leg to spin like a propeller.
- **Fix**: Since the mesh is standardized (hinge along its own Y axis), we should **ALWAYS set axis="0 1 0"** (Local Y). The `origin rpy` handles the global direction.

## Actuator Mapping
| Component | FreeCAD | URDF Joint Structure | Control Axis |
| :--- | :--- | :--- | :--- |
| **Legs** | Rotated Mesh | Revolute Joint | **Local Y (`0 1 0`)** |
| **Gimbal** | Fixed Part | 2x Revolute (Pitch -> Yaw) | **Pitch (`0 1 0`), Yaw (`1 0 0`)** |

## Convention Check
- **Leg +**: Deploy Down? (May vary by quadrant, verified with simple logic test).
- **Gimbal +**: Tilt Nose Up/Right? (Standard aviation: Pitch Up = Nose Up).
