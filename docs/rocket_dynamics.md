# Rocket Dynamics & Control Axes

## Body Frame (The Rocket)
Assuming the rocket stands vertically (**Z-Up**):
-   **Roll (Google: Z-Axis)**: Spinning around the long vertical axis (like a drill).
-   **Pitch (Global Y-Axis)**: Nose tips Forward/Backward.
-   **Yaw (Global X-Axis)**: Nose tips Left/Right.

*(Note: In airplanes, Yaw is usually Z, but rockets are often Z-up)*

## Engine Gimbal Frame (The Thruster)
Why can't a single engine control Roll?

1.  **Engine Pitch (Tilt Nozzle Front/Back)**
    -   Thrust Vector points slightly backwards.
    -   Creates a torque that rotates the **Body Pitch**.
    -   Effect: Rocket flips over.

2.  **Engine Yaw (Tilt Nozzle Left/Right)**
    -   Thrust Vector points slightly sideways.
    -   Creates a torque that rotates the **Body Yaw**.
    -   Effect: Rocket leans sideways.

3.  **Engine Roll (Twist Nozzle)**
    -   Imagine twisting a round pipe or a flashlight.
    -   The direction of the flame (vector) **does not change**.
    -   **Result**: No force, no torque. Zero effect on the rocket.

## How to Control Roll?
Since the main engine determines Pitch and Yaw but cannot stop the rocket from spinning (Roll), you need:
1.  **RCS Thrusters**: Small jets on the side (like on your model) firing in opposite directions.
2.  **Fins/Grid Fins**: Archodynamic control (only works in atmosphere).
3.  **Multiple Engines**: Two engines side-by-side can gimbal differentially (one pushes up-left, one pushes up-right) to create torque. **Starship has 3 or 6 engines**, so it CAN use engines for roll control, but a single engine simulation cannot.

## Physics Specifications (Updated v2)

The simulation now uses **Hardcore/Realistic Physics** settings:

*   **Mass**:
    *   **Wet Mass (Full)**: 3,000 kg
    *   **Dry Mass (Empty)**: 300 kg
    *   **Variable Mass**: The physics engine updates the mass in real-time as fuel is consumed.
*   **Propulsion**:
    *   **Main Engine**: 60,000 N (Max Thrust)
    *   **RCS Thrusters**:
        *   **Pitch/Roll**: 1,000 N (Scaled by Mass)
        *   **Yaw**: 200 N (Scaled by Mass) - Reduced to prevent excessive rotation.
        *   **Variable Gain**: RCS force scales linearly with mass (Full Tank=100%, Empty=10%) to maintain consistent control authority.
*   **Thrust-to-Weight Ratio (TWR)**:
    *   **Full Tank**: ~2.0 (60kN / 30kN weight) -> Heavy, slow to stop.
    *   **Full Tank**: ~2.0 (60kN / 30kN weight) -> Heavy, slow to stop.
    *   **Empty Tank**: ~20.0 (60kN / 3kN weight) -> Extremely sensitive, requires precise throttle control.

## System Specifications

### 1. Physical Configuration
- **Shape**: Slender "Pencil" Cylinder.
- **Height**: ~5.0 meters.
- **Mass Properties**: 
    - **Wet Mass**: 3,000 kg.
    - **Dry Mass**: 300 kg.
    - **Variable Mass**: Fuel consumption reduces mass in real-time.

### 2. Actuators (Action Space)
Actions are a `Box(16,)` normalized to `[-1, 1]`.

| Index | Component | Math Mapping | Physical Range / Behavior |
| :--- | :--- | :--- | :--- |
| **0** | **Main Engine** | `(x+1)/2` | **0%** to **100%** (60,000 N Max). |
| **1** | **Gimbal Pitch**| `x * 0.35` | **-0.35** s/d **+0.35** rad. <br> Torsi Joint Max: **20,000 N·m**. |
| **2** | **Gimbal Roll** | `x * 0.35` | **-0.35** s/d **+0.35** rad. <br> Torsi Joint Max: **20,000 N·m**. |
| **3-14**| **RCS** | `(x+1)/2` | **0%** s/d **100%** (1,000N/200N base). |
| **15** | **Landing Legs**| Linear Lerp | **1.507 rad** (Up) ke **-1.2 rad** (Down). |

### Penerapan Gaya Utama (Thrust Application)
Untuk meningkatkan realisme mekanis, gaya dorong Main Engine diterapkan dengan **offset 0,5 meter** di bawah poros (*pivot*) gimbal. Hal ini memastikan bahwa kemiringan gimbal menghasilkan beban torsi nyata pada joint, yang harus dilawan oleh motor gimbal sebesar maksimal **40.000 N·m**.

> [!IMPORTANT]
> **Prinsip Gimbal-Thrust Coupling**:
> Efektivitas gimbal (orientasi roket) bergantung penuh pada adanya gaya dorong (*Thrust*). Jika mesin mati (*Throttle* = 0), gimbal tidak memiliki gaya untuk dibelokkan, sehingga roket kehilangan kendali orientasi melalui mesin. Selalu pertahankan nilai *Throttle* minimum (~10-20%) selama manuver orientasi kritis.

> [!TIP]
> **Stabilitas Kontrol & Inersia**:
> 1. **Integral Anti-Windup**: Semua kontroler PID wajib memiliki limitasi pada akumulasi error (I-term) untuk mencegah *overshoot* masif akibat inersia roket 3 ton.
> 2. **Dynamic Min Throttle**: Sistem kontrol harus menjamin daya mesin minimal ~10% saat bermanuver agar gimbal tetap aktif, namun tetap memperhatikan rasio dorong-terhadap-berat (*TWR*) agar roket tidak melompat tak terkendali saat sudah ringan (dry mass 300kg).

---

## Skenario Misi (Mission Scenarios)

Environment ini mendukung berbagai kondisi awal melalui parameter `options` pada `reset()`:

### Sky-to-Ground Landing (Recovery)
Skenario utama untuk melatih fase pendaratan (Booster Recovery).
*   **Initial Altitude**: 80 - 300 meter.
*   **Initial Velocity**: Mendukung inersia horizontal (up to 20 m/s) dan vertikal.
*   **Initial Orientation**: Mendukung posisi miring (*tilted*) hingga 30 derajat.

Dapat diakses melalui modul: `examples/scenario_landing.py`.

> [!NOTE]
> For Thrusters (Main & RCS), `-1.0` means **OFF**, `0.0` means **50%**, and `1.0` means **100%`.

### 3. Sensors & Observations
The observation space is a `Box(19,)`.

---

## Visualisasi & Monitoring

Environment ini dioptimalkan untuk performa tinggi dan tampilan visual yang halus (*smooth*):

### Smooth Camera Tracking
Sistem kamera menggunakan algoritma **LERP (Linear Interpolation)** untuk mengikuti posisi roket. Hal ini menghilangkan "stuttering" atau gerakan patah-patah yang sering terjadi pada tracking kamera standar.
*   **Smoothing Factor**: `0.1` (Memberikan efek lag kamera sinematik).
*   **Dynamic Distance**: Kamera otomatis menjauh saat roket berada di ketinggian tinggi dan mendekat saat *final approach* untuk detail pendaratan yang lebih baik.

### Headless Mode (--no-render)
Untuk proses training RL atau simulasi batch yang cepat, Anda dapat mematikan GUI PyBullet guna menghemat resource CPU/GPU:
*   **Penggunaan**: Tambahkan argumen `--no-render` pada script controller.
*   **Keuntungan**: Kecepatan simulasi meningkat hingga 5-10x lipat dibandingkan mode visual.

---

| Index | Observation | Type | Detail |
| :--- | :--- | :--- | :--- |
| **0-2** | **Position** | Vector3 | World Coordinates (X, Y, Z). |
| **3-6** | **Orientation** | Quat | World Orientation Quaternion (x, y, z, w). |
| **7-9** | **Linear Vel** | Vector3 | World velocity (m/s). |
| **10-12**| **Angular Vel** | Vector3 | **Local body frame** angular velocity (rad/s). |
| **13-16**| **Leg Contacts**| 4x Float | `1.0` if in contact with terrain, `0.0` otherwise. |
| **17** | **Altitude** | Float | Current **Global Z** position (meters). |
| **18** | **Fuel** | Scalar | Normalized fuel level (1.0 to 0.0). |

---

---

## Kendali Translasi Kaskade (Cascaded Translation Control)

Mulai Versi 0.7.0, roket menggunakan arsitektur kendali kaskade untuk menjaga posisi (*Station Keeping*). Loop luar mendeteksi kecepatan horizontal dan memerintahkan loop dalam (Attitude) untuk miring.

### Mekanisme Pengereman (Braking Physics)
| Kecepatan | Arah Gerak | Target Orientasi | Fungsi |
| :--- | :--- | :--- | :--- |
| **+Vx** | Forward | **-Pitch** (Nose Down) | Mengarahkan semburan mesin ke depan untuk mengerem. |
| **-Vx** | Backward | **+Pitch** (Nose Up) | Mengarahkan semburan mesin ke belakang untuk mengerem. |
| **+Vy** | Rightward | **+Roll** (Lean Left) | Mengarahkan semburan mesin ke kanan untuk mengerem. |
| **-Vy** | Leftward | **-Roll** (Lean Right) | Mengarahkan semburan mesin ke kiri untuk mengerem. |

> [!CAUTION]
> **Aerodynamic Drag Flip**: Pada kecepatan horizontal tinggi (>20 m/s), hambatan udara pada bodi roket silindris dapat menghasilkan torsi yang lebih besar daripada kemampuan gimbal mesin (**40.000 Nm**). Sistem translasi harus membatasi kemiringan maksimal (**15 derajat**) untuk menjaga stabilitas aerodinamis.

---

## Strategi Tuning PID (Incremental Tuning)

Untuk menjamin kestabilan roket seberat 3.000 kg, proses tuning dilakukan secara bertahap menggunakan script `examples/pid_calibration.py`:

### Tahap 1: Hover Stability (High Altitude)
Mengunci roket pada ketinggian tetap (misal: 300m) untuk mencari *gain* dasar:
*   **Vertical PID**: Mencari nilai yang pas agar roket tidak naik-turun (*oscillating*) terhadap gravitasi.
*   **Attitude PID**: Memastikan roket tegak lurus (RPY = 0) saat mesin menyala.

### Tahap 2: Attitude Perturbation
Memberikan gangguan (*disturbance*) pada orientasi saat sedang hover:
*   **Roll Recovery**: Memiringkan badan ke arah Roll, lalu melihat seberapa cepat dan stabil roket kembali tegak.
*   **Pitch Recovery**: Memiringkan hidung ke atas/bawah, melihat respon gimbal dalam melakukan koreksi.
*   **Yaw Recovery**: Memutar roket, melihat kekuatan RCS dalam menahan torsi rotasi.

### Tahap 3: Integrated Landing
Setelah semua sumbu stabil, kontroler digabungkan untuk melakukan pendaratan (Sky-to-Ground) dengan target kecepatan pendaratan < 2.5 m/s.

---

## Lessons Learned & Advanced Tuning

### 1. The Polarity Trap (Negative Gain)
One of the most common mistakes in this environment is assuming positive feedback (Error = Target - Actual). 
*   **Discovery**: Both Pitch and Roll axes in this simulation have a **Negative torque relationship**. A positive control action (+0.5) results in a negative angular change (-9.0 deg).
*   **Fix**: To achieve corrective feedback, the PID $K_p$ gains must be **Negative** (e.g., -1.5). Positive gains will cause the rocket to flip exponentially.

### 2. Mechanical Authority Imbalance (Roll vs Pitch)
Even with correct polarity, Roll may feel "mushy" compared to Pitch.
*   **Discovery**: Manual testing revealed that the Roll axis achieves only ~40% of the angular acceleration of the Pitch axis for the same control input. This is likely due to the nested joint friction or URDF damping properties.
*   **Fix**: Use **Asymmetric Gain Magnitudes**. In Version 0.6.0, Roll gains are doubled ($K_p = -3.0$) relative to Pitch to achieve similar recovery performance.

### 3. The "Pendulum Snap" (Throttle-Attitude Coupling)
When the rocket is tilted, increasing throttle doesn't just push it up; it pushes it **sideways**. 
*   **Tip**: Avoid aggressive attitude correction at high throttle. The rapid acceleration "snaps" the rocket's longitudinal axis, often inducing oscillations that the gimbal cannot damp in time. 
*   **Solution**: Set a `min_throttle` (0.15) for authority but use a moderate `max_vz` limit in the vertical controller to keep the TWR manageable during attitude recovery.

---
## Verification History
Final verification for v0.6.0 completed on 2026-02-12. See [v0.6.0 report](file:///home/aiot/Projects/Reinforcement-Learning/RocketLander/docs/reports/final_verification_v0.6.0.md).
