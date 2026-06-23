# Experiment: LunarLander3D-like 3D Rocket Trajectory Controller (PID)

Dokumen ini menjelaskan tahapan algoritma controller yang mirip pola **LunarLander3D** (phase/trajectory tracking → attitude outer loop → rate inner loop → mix ke aktuator).

> Catatan: Implementasi aktual akan menggunakan env `RocketLander-v0` yang sudah ada di repo ini (Gymnasium + PyBullet).

## 0) Input & State yang Dipakai
- `obs` dari env berisi (sesuai `rocket_lander_env.py`):
  - `pos_n`: obs[0:3]
  - `quat`: obs[3:7]
  - `vel_n`: obs[7:10]
  - `ang_vel_local`: obs[10:13] = (wx, wy, wz)
  - `foot_contacts`: obs[13:17]
  - `alt`: obs[17]
  - `fuel`: obs[18]

## 1) Estimasi Euler & Wrap-aware Attitude Error
1. Konversi quaternion → Euler (roll, pitch, yaw)
2. Hitung error terhadap target (umumnya 0):
   - `err = wrap_to_pi(target - angle)` agar tidak rusak karena diskontinuitas ±π.

## 2) Planner / Reference Generator (mirip phase-based guidance)
Controller membagi misi menjadi beberapa fase berdasarkan altitude atau milestone lain (mis. `RECOVERY`, `APPROACH`, `FINAL`).

Output tiap fase:
- referensi posisi/kecepatan horizontal: `ref_vx, ref_vy`
- referensi kecepatan vertikal: `ref_vz`
- limit tilt maksimum yang makin ketat mendekati ground

## 3) Outer Loop: Attitude (Angle) → Desired Body Rates
Gunakan PID attitude (kp/ki/kd) untuk mengubah error roll/pitch/yaw menjadi **rate references**:
- `roll_rate_ref  = PID_roll(err_roll)`
- `pitch_rate_ref = PID_pitch(err_pitch)`
- `yaw_rate_ref   = PID_yaw(err_yaw)`

Gains outer PID diambil dari kalibrasi attitude PID yang stabil di mode stabilisasi (tanpa gimbal) dan kemudian disesuaikan sign/axis mixing untuk mapping aktual aktuator di env.

## 4) Inner Loop: Rate PID (Body Rates Tracking)
PID rate men-`track` rate aktual ke rate reference:
- `u_roll = PID_wx(wx, target=roll_rate_ref)`
- `u_pitch = PID_wy(wy, target=pitch_rate_ref)`
- `u_yaw = PID_wz(wz, target=yaw_rate_ref)`

Inner rate PID dibuat lebih “damped” supaya tidak osilasi saat error kecil.

## 5) Actuator Mixing ke 16D Action Vector
- Main engine: biasanya di **OFF** untuk kalibrasi attitude-only (bisa diatur untuk misi penuh).
- Gimbal: jika mode RCS-only, gimbal tetap 0.
- RCS thrusters:
  - Mapping dilakukan ke pasangan thruster yang menghasilkan torque sesuai axis.
  - Gunakan threshold/ hysteresis kecil untuk mencegah chattering.
- Landing legs:
  - deploy saat `alt < LEG_DEPLOY_ALT` (set action[15]).

## 6) Saturation, Authority Scaling, dan Anti-Chattering
- clamp output torque command / rate references agar tetap dalam batas fisik
- scale authority terhadap massa (fuel) bila dibutuhkan
- gunakan `eps` pada mixing supaya RCS tidak terus menyalakan/mematikan

## 7) Termination & Success Criteria
- terminate on touchdown/landing success (kontak kaki, kecepatan & tilt dalam toleransi)
- terminate on crash (kecepatan terlalu besar saat kontak atau tilt berlebihan)

---
## TODO untuk implementasi kode
1. Pastikan mapping RCS → tanda torque roll/pitch/yaw benar (gunakan skrip `verify_*` yang sudah ada).
2. Port gains dari `examples/pid_stabilization_calibration.py` ke parameter kelas PID di experiment controller.
3. Buat logging CSV/PNG serupa agar bisa dibandingkan antar fase.

