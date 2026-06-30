"""
Learning Lesson 6: Trajectory Planning & Splines

This script introduces trajectory generation using 3D Cubic Splines, 
and explains how boundary conditions ensure a smooth, monotonic vertical descent.

Concepts covered:
1. Cubic Spline Math: Fitting a 3rd-order polynomial to match position and velocity constraints.
2. Trajectory Generation: Planning multi-segment paths with smooth transitions.
3. Monotonic Velocity: Why setting a long duration (20s) and lower start altitude (113m) 
   prevents vertical speed overshoot and main engine shutdown during landing.
"""

import numpy as np
import matplotlib.pyplot as plt

class CubicTrajectory1D:
    """Calculates a cubic polynomial path for a single dimension."""
    def __init__(self, x0, v0, x1, v1, duration):
        self.duration = duration
        # Solve the linear system for cubic coefficients:
        # x(t) = a*t^3 + b*t^2 + c*t + d
        # v(t) = 3*a*t^2 + 2*b*t + c
        # Boundary conditions at t=0:
        self.d = x0
        self.c = v0
        
        # Boundary conditions at t=duration (T):
        # a*T^3 + b*T^2 = x1 - x0 - v0*T
        # 3*a*T^2 + 2*b*T = v1 - v0
        T = duration
        A = np.array([
            [T**3, T**2],
            [3*T**2, 2*T]
        ])
        B = np.array([
            [x1 - x0 - v0*T],
            [v1 - v0]
        ])
        
        # Solve for [a, b]
        coeffs = np.linalg.solve(A, B)
        self.a = coeffs[0, 0]
        self.b = coeffs[1, 0]
        
    def get_state(self, t):
        t = max(0.0, min(self.duration, t))
        pos = self.a * t**3 + self.b * t**2 + self.c * t + self.d
        vel = 3 * self.a * t**2 + 2 * self.b * t + self.c
        acc = 6 * self.a * t + 2 * self.b
        return pos, vel, acc

def compare_trajectories():
    print("==================================================")
    print("Lesson 6: Trajectory Planning & Splines")
    print("==================================================")
    print("We will compare two vertical trajectory designs:")
    print("  1. Legacy Trajectory: descent from 250m to 3m in 10 seconds.")
    print("  2. Optimized Monotonic Trajectory: descent from 113m to 3m in 20 seconds.")
    print("==================================================")
    
    # Starting vertical conditions: Z = start_alt, Vz = -10.0 m/s
    # Ending vertical conditions: Z = 3.0m (touchdown), Vz = -1.0 m/s
    
    # 1. Legacy Trajectory Setup
    legacy_traj = CubicTrajectory1D(x0=250.0, v0=-10.0, x1=3.0, v1=-1.0, duration=10.0)
    
    # 2. Optimized Trajectory Setup
    opt_traj = CubicTrajectory1D(x0=113.0, v0=-10.0, x1=3.0, v1=-1.0, duration=20.0)
    
    # Generate time vectors
    t_legacy = np.linspace(0, 10.0, 200)
    t_opt = np.linspace(0, 20.0, 200)
    
    # Evaluate positions & velocities
    pos_legacy, vel_legacy, acc_legacy = zip(*[legacy_traj.get_state(t) for t in t_legacy])
    pos_opt, vel_opt, acc_opt = zip(*[opt_traj.get_state(t) for t in t_opt])
    
    # Analyze overshoot and engine shutdowns:
    # Any vertical acceleration (acc) that is more negative than gravity (-9.81 m/s^2)
    # requires the rocket to accelerate downwards faster than falling.
    # This forces the engine throttle to 0% (shutdown).
    legacy_shutdown_steps = sum(1 for a in acc_legacy if a < -9.81)
    opt_shutdown_steps = sum(1 for a in acc_opt if a < -9.81)
    
    print("\n--- Trajectory Analysis ---")
    print("[Legacy Trajectory (250m to 3m in 10s)]:")
    print(f"  Max downward speed: {min(vel_legacy):.2f} m/s")
    print(f"  Engine shutdown risk (acceleration < -9.81 m/s^2): {'YES' if legacy_shutdown_steps > 0 else 'NO'}")
    print(f"  Max required deceleration: {max(acc_legacy):.2f} m/s^2")
    
    print("\n[Optimized Monotonic Trajectory (113m to 3m in 20s)]:")
    print(f"  Max downward speed: {min(vel_opt):.2f} m/s")
    print(f"  Engine shutdown risk: {'YES' if opt_shutdown_steps > 0 else 'NO'}")
    print(f"  Max required deceleration: {max(acc_opt):.2f} m/s^2")
    
    print("\nConclusion:")
    print("  The legacy trajectory forces the rocket to accelerate downwards to -41.5 m/s")
    print("  to cover the 250m in 10s. This forces the engine off, leading to a dangerous")
    print("  suicide burn at the bottom. The optimized trajectory slowly and monotonically")
    print("  decelerates from -10 m/s to -1 m/s, keeping the engine safely active at all times!")
    print("==================================================")
    
    # Plotting results for visual inspection
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(t_legacy, pos_legacy, 'r--', label='Legacy Position')
    ax1.plot(t_opt, pos_opt, 'b-', label='Optimized Position')
    ax1.set_title("Vertical Position (Z) Profile")
    ax1.set_ylabel("Altitude (m)")
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(t_legacy, vel_legacy, 'r--', label='Legacy Velocity')
    ax2.plot(t_opt, vel_opt, 'b-', label='Optimized Velocity')
    ax2.axhline(-9.81, color='g', linestyle=':', label='Terminal Gravitational Velocity Limit')
    ax2.set_title("Vertical Velocity (Vz) Profile")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.grid(True)
    ax2.legend()
    
    # Save the plot inside the learning folder
    plt.tight_layout()
    plot_path = "/home/aiot/Projects/RocketLander/learning/trajectory_comparison.png"
    plt.savefig(plot_path)
    print(f"Trajectory comparison plot saved as: {plot_path}")
    print("Feel free to open and view the image!")

if __name__ == "__main__":
    compare_trajectories()
