
# RCS Calculation
# Rocket Dimensions (Cylinder)
# Radius (r) = 0.45 m
# Length (h) = 5.0 m
# Mass (m) = 3000 kg (Full) / 300 kg (Empty)

# Moment of Inertia (Cylinder)
# I_z (Roll axis) = 0.5 * m * r^2
# I_x, I_y (Pitch/Yaw axes) = (1/12) * m * (3*r^2 + h^2)

# Full Tank (3000 kg)
# I_z_full = 0.5 * 3000 * 0.45^2 = 1500 * 0.2025 = 303.75 kg*m^2
# I_x_full = (1/12) * 3000 * (3*0.45^2 + 5.0^2) 
#          = 250 * (0.6075 + 25.0) = 250 * 25.6075 = 6401.875 kg*m^2

# Empty Tank (300 kg)
# I_z_empty = 30.375 kg*m^2
# I_x_empty = 640.1875 kg*m^2

# RCS Force
# Force per thruster = 1000 N
# Torque Calculation (Tau = r x F)

# 1. Yaw (Rotation around Z)
# Thrusters at mid-body (y-plus/neg links?), firing tangentially?
# Looking at URDF/Env: 
#   Mid Thrusters (Yaw): pos=[0, +/-0.5, 0]. Dir=[+/-1, 0, 0].
#   Moment Arm = 0.5 m (Distance from center axis)
#   Torque_Yaw = Force * Arm = 1000 * 0.5 = 500 N*m
#   Number of thrusters typically firing: 2 (e.g. 11 and 14 for CCW) -> Total Torque = 1000 N*m

# Angular Accel Yaw (Full):
#   alpha_z = Torque / I_z = 1000 / 303.75 ~= 3.29 rad/s^2 (~188 deg/s^2)
#   Verdict: VERY HIGH for Yaw. 

# 2. Pitch/Roll (Rotation around X/Y)
# Top/Bottom Thrusters.
#   Top: pos=[0, 0, 2.0]. (Relative to COM? URDF origin is center of cylinder?)
#   URDF visual origin is 0,0,0. Cylinder length 5.0. Center is at 0,0,0? 
#   Collision cylinder length 5.0. 
#   If COM is at geometric center (0,0,0).
#   Top thrusters at Z=2.0. Moment Arm (Lever) = 2.0 m.
#   Bottom thrusters at Z=-2.0. Moment Arm = 2.0 m.
#    firing pair (Top+Bottom opposite) -> Torque = (1000 * 2.0) + (1000 * 2.0) = 4000 N*m.

# Angular Accel Pitch (Full):
#   alpha_x = Torque / I_x = 4000 / 6401.8 ~= 0.62 rad/s^2 (~35 deg/s^2)
#   Verdict: Good authority. 35 deg/s^2 is agile but realistic for a lander.

# Angular Accel Pitch (Empty):
#   alpha_x = 4000 / 640.18 ~= 6.2 rad/s^2 (~355 deg/s^2)
#   Verdict: EXTREMELY AGGRESSIVE. Might be uncontrollable for human without damping.

# Comparison with Starship
# Starship (approx): Mass ~100t (landing). Length 50m.
# RCS on Starship is cold gas (weak) or hot gas (strong).
# Our lander is 1/10th scale (5m vs 50m).
# 3000kg is heavy for 5m.
# 35 deg/s^2 is quite high performance (fighter jet like).
# 0.5 - 2.0 deg/s^2 is typical for large vehicles.
# 10 - 50 deg/s^2 is typical for drones/missiles.

# Conclusion:
# Yaw is overpowered.
# Pitch/Roll at Full Tank is nice (35 deg/s^2).
# Pitch/Roll at Empty Tank is crazy fast (350 deg/s^2).

# Recommendation to User:
# 1. Scale down RCS Force? Or implementation logic?
# 2. Current 1000N is maybe too high for Empty mass.
# 3. But necessary for Full mass?
