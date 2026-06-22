import numpy as np
import scipy.optimize
import time
import gymnasium as gym
import rocket_lander
import pybullet as p
import argparse
from scenario_landing import setup_landing_scenario

class MPCController:
    def __init__(self):
        self.g = 9.81
        self.mass = 2000.0 # Estimated Mass (Between 3000kg wet and 300kg dry?? No, URDF is 200kg? Wait, verify_dimensions said 3000kg?)
        # Let's use the value from pid_controller_basic.py or verify_physics.
        # PID controller uses mass=3000.0? verify_physics said 3000kg.
        self.mass = 3000.0 
        
        self.DT = 1.0/30.0 # Prediction Step (Same as Sim)
        self.N = 30 # Increased Horizon (1.0s) to see overshoot
        
        # Physics Constants from verify_physics.py / rocket_lander_env.py
        self.lever_arm = 2.8
        # Ixx, Iyy, Izz
        # URDF Analysis + Mass=3000kg -> Ixx~6437, Izz~375.
        # Let's use slightly conservative values.
        self.inertia = np.array([6500.0, 6500.0, 400.0])
        
        # Constraints
        self.MAX_GIMBAL = np.radians(15.0) 
        self.MAX_THROTTLE = 1.0
        self.MIN_THROTTLE = 0.4 
        self.MAX_TILT = np.radians(45.0) # Relaxed to 45 deg
        self.MAX_THRUST = 60000.0 # N
        
        # Weights
        # State: [x, y, z, phi, theta, psi, vx, vy, vz, wx, wy, wz]
        self.Q = np.array([
            10.0, 10.0, 30.0,  # Pos (High Z priority)
            200.0, 200.0, 10.0,  # Angle (CRITICAL: Keep stable!)
            5.0,  5.0,  5.0,   # Vel (Track reference velocity)
            1.0,  1.0,  0.5    # Rates (Damping)
        ])
        
        # Control: [gimbal_roll, gimbal_pitch, throttle]
        self.R = np.array([0.1, 0.1, 0.1]) # Cheaper control
        
    def get_linear_model(self, current_mass, yaw=0.0):
        # Linearized Dynamics around Hover
        # State: [x, y, z, phi, theta, psi, vx, vy, vz, wx, wy, wz]
        # Control: [gimbal_roll, gimbal_pitch, throttle]
        
        A = np.zeros((12, 12))
        
        # Identity (State Persistence)
        A += np.eye(12)
        
        # Position kineamtics: p_dot = v
        A[0, 6] = self.DT # dx = vx * dt (Wait, A[0,6] was 1.0 before? No, x_new = x_old + v_old * dt)
        A[1, 7] = self.DT
        A[2, 8] = self.DT
        # Angle -> Rate
        A[3, 9] = self.DT
        A[4, 10] = self.DT
        A[5, 11] = self.DT
        
        # B Matrix (Control Influence)
        B = np.zeros((12, 3))
        
        T_hover = current_mass * self.g
        
        # Torque = F * lever * sin(delta) ~ F * lever * delta
        torque_per_rad = self.MAX_THRUST * self.lever_arm 
        
        # Angular Accel from Gimbal
        # Positive Gimbal -> Negative Torque -> Negative Rate
        B[9, 0] = -(torque_per_rad / self.inertia[0]) * self.DT
        B[10, 1] = -(torque_per_rad / self.inertia[1]) * self.DT
        
        # Vertical Accel from Throttle
        B[8, 2] = (self.MAX_THRUST / current_mass) * self.DT
        
        # Horizontal coupling (Gravity / Tilt)
        # Rotated by Yaw
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        
        # Base coupling in Body Frame:
        # Pitch (Theta) -> X Accel. (Empirical Flight Log: +Theta -> +Accel)
        # Roll (Phi) -> Y Accel. (Empirical Authority Check: +Phi -> +Accel) -> Let's keep +g
        
        # Wait. If +Theta -> +Accel (Forward). And we want Brake (Neg Accel).
        # We need Neg Theta (Nose Up??).
        # Let's trust the logic: Theta term creates Accel.
        
        # World X Accel = g * (Theta * cy - Phi * sy) ?
        # Rotation of Tilt Vector.
        # Tilt X (Pitch) contributes to World X via Cos(Yaw).
        # Tilt Y (Roll) contributes to World X via -Sin(Yaw).
        
        # A[6, 4] (Pitch -> X)
        # If Yaw=0: A[6,4] = +g * cy = +g.
        A[6, 4] = self.g * self.DT * cy
        A[6, 3] = self.g * self.DT * sy # Roll -> X (via sin yaw)
        
        # A[7, 3] (Roll -> Y)
        # If Yaw=0: A[7,3] = +g * cy = +g.
        A[7, 3] = self.g * self.DT * cy
        A[7, 4] = -self.g * self.DT * sy # Pitch -> Y
        
        return A, B

    def predict(self, x0, U, A, B):
        """ Integrate dynamics forward given control sequence U """
        # U is flattened [u0_0, u1_0, u2_0, u0_1, ...]
        predictions = []
        x = x0.copy()
        
        u_dim = 3
        
        for i in range(self.N):
            u = U[i*u_dim : (i+1)*u_dim]
            x = A @ x + B @ u
            # Gravity compensation for Z (Hover throttle roughly cancels g, but linear model needs explicit g)
            # If u[2] is total thrust fraction:
            # z_accel = (T_max * u[2]) / m - g
            # In linear model B[8,2] * u[2] gives (T_max/m)*u[2].
            # So we must subtract g.
            x[8] -= self.g * self.DT 
            
            predictions.append(x)
            
        return np.array(predictions)

    def cost_function(self, U, x0, ref_states, A, B):
        """ Calculate total cost of trajectory """
        cost = 0.0
        x = x0.copy()
        u_dim = 3
        
        for i in range(self.N):
            u = U[i*u_dim : (i+1)*u_dim]
            
            # State Update
            x = A @ x + B @ u
            x[8] -= self.g * self.DT 
            
            # State Cost using Reference Trajectory for this step
            target = ref_states[i]
            err = x - target
            
            # Weighted Squared Error
            cost += np.sum((err**2) * self.Q)
            
            # Control Cost (deviation from hover?)
            # u[2] cost should be deviation from hover throttle ~0.6
            u_cost = u.copy()
            u_cost[2] -= 0.6 
            cost += np.sum((u_cost**2) * self.R)
            
            # Soft Constraint on Tilt (Barrier)
            tilt_sq = x[3]**2 + x[4]**2
            if tilt_sq > self.MAX_TILT**2:
                cost += 10000.0 * (tilt_sq - self.MAX_TILT**2)
                
        return cost

    def generate_reference_trajectory(self, current_state, N, dt):
        """
        Generate a reference trajectory from current state to origin.
        Simple logic: Move towards (0,0,0.5) with max velocity cap.
        """
        ref_states = []
        
        # Current Goal: (0, 0, 0.5)
        goal_pos = np.array([0.0, 0.0, 0.5])
        
        # Max Velocity for Approach
        max_vel = 20.0 # m/s (Horizontal) - Needs to be high to handle initial 15m/s
        max_desc_rate = 5.0 # m/s (Vertical)
        
        curr_pos = current_state[0:3]
        
        for i in range(N):
            # Vector to goal
            diff_pos = goal_pos - curr_pos
            dist = np.linalg.norm(diff_pos)
            
            # Desired Velocity Vector
            if dist > 0.1:
                vel_dir = diff_pos / dist
                # Scale by proximity?
                # V_ref = min(max_vel, dist / 2.0) # Slow down as we get closer (Time constant 2s)
                # Let's be explicit:
                v_mag = min(max_vel, dist * 1.0) # Gain 1.0 (Smoother)
                v_ref = vel_dir * v_mag
                
                # Cap descent rate
                v_ref[2] = max(v_ref[2], -max_desc_rate) 
                
            else:
                v_ref = np.zeros(3)
                
            # Update Ref Pos
            curr_pos = curr_pos + v_ref * dt
            
            # Construct State Vector
            ref_state = np.zeros(12)
            # IMPORTANT: Reference Angle should be zero (level)
            # Reference Velocity should be v_ref
            ref_state[0:3] = curr_pos
            ref_state[7:10] = v_ref
            
            ref_states.append(ref_state)
            
        return np.array(ref_states)

    def optimize(self, obs):
        """
        Main MPC Step
        obs: PID controller observation style (State vector)
        """
        # Parse Observation to State Vector (12x1)
        # Obs: [x, y, z, r, p, y, vx, vy, vz, wr, wp, wy, ...]
        
        px, py, pz = obs[0], obs[1], obs[2]
        vx, vy, vz = obs[7], obs[8], obs[9]
        import pybullet as p
        quat = obs[3:7]
        rpy = p.getEulerFromQuaternion(quat)
        current_yaw = rpy[2]
        # Corrected: obs[10-12] are wr, wp, wy (which align with wx, wy, wz in body frame)
        wx, wy, wz = obs[10], obs[11], obs[12]
        
        state = np.array([px, py, pz, rpy[0], rpy[1], rpy[2], vx, vy, vz, wx, wy, wz])
        
        # Generate Reference Trajectory
        # Start reference from current state? No, reference should be the "Ideal Path".
        # If we start typically from current state, it ensures continuity.
        # But we want to guide it back to path? 
        # Simple Approach: Reference starts at current pos and moves to goal.
        ref_states = self.generate_reference_trajectory(state, self.N, self.DT)
        
        # Get Model
        current_mass = self.mass # Should extrapolate
        A, B = self.get_linear_model(current_mass, yaw=current_yaw)
        
        # Initial Guess (Heuristic)
        # If Pitch is Large Positive -> We need Positive Gimbal to correct (reduce angle)
        # u0_pitch_guess = k * pitch
        k_guess = 5.0
        u_pitch = np.clip(rpy[1] * k_guess, -self.MAX_GIMBAL, self.MAX_GIMBAL)
        u_roll = np.clip(rpy[0] * k_guess, -self.MAX_GIMBAL, self.MAX_GIMBAL)
        u_guess = [u_roll, u_pitch, 0.8] # Hover throttle guess
        u0 = np.tile(u_guess, self.N) 
        
        # Bounds
        bounds = []
        for _ in range(self.N):
            bounds.append((-self.MAX_GIMBAL, self.MAX_GIMBAL)) # Roll Gimbal
            bounds.append((-self.MAX_GIMBAL, self.MAX_GIMBAL)) # Pitch Gimbal
            bounds.append((self.MIN_THROTTLE, self.MAX_THROTTLE)) # Throttle
            
        # Optimization
        start_time = time.time()
        res = scipy.optimize.minimize(
            self.cost_function,
            u0,
            args=(state, ref_states, A, B),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-3, 'disp': False} 
        )
        solve_time = time.time() - start_time
        
        if res.success:
            updates = res.x[0:3]
            # Debug: Evaluate solution
        # DEBUG: Print Plan for the first step only (or periodically)
        # Using a simple attribute hack or just always print if it's the first call?
        # Let's use a static attribute on the class or just check time?
        if not hasattr(self, "has_printed_debug"):
            self.has_printed_debug = True
            print("\n--- DEBUG MPC MODEL & PLAN (T=0) ---")
            print(f"A[6,4](Pitch->Ax)={A[6,4]:.4f}, A[7,3](Roll->Ay)={A[7,3]:.4f}")
            print(f"A[6,3](Roll->Ax)={A[6,3]:.4f}, A[7,4](Pitch->Ay)={A[7,4]:.4f}")
            print(f"B[9,0](RollG->Wx)={B[9,0]:.4f}, B[10,1](PitchG->Wy)={B[10,1]:.4f}")
            
            if res.success:
                opt_u = res.x
                pred_x = state.copy()
                print(f"Ref Start: {ref_states[0][0:3]}")
                print(f"Ref End:   {ref_states[-1][0:3]}")
                print("Predicted Trajectory:")
                for k in range(5): # First 5 steps
                    uk = opt_u[k*3 : (k+1)*3]
                    pred_x = A @ pred_x + B @ uk
                    pred_x[8] -= self.g * self.DT
                    print(f"  Step {k}: U=[{uk[0]:.2f}, {uk[1]:.2f}, {uk[2]:.2f}] -> Pos={pred_x[0:3]} Vel={pred_x[6:9]}")
            else:
                 print("MPC Optimization Failed at T=0")
            print("------------------------------------\n")
            import sys
            sys.stdout.flush()
            
        else:
            # Fallback: Use simple PD logic (Locked-Down style) to stay safe
            # Target Rate = -Angle * Gain
            roll = rpy[0]
            pitch = rpy[1]
            kp_r = -0.5
            gp = np.clip((0 - wy)*kp_r - pitch*1.0, -self.MAX_GIMBAL, self.MAX_GIMBAL)
            gr = np.clip((0 - wx)*kp_r - roll*1.0, -self.MAX_GIMBAL, self.MAX_GIMBAL)
            hover_thr = 0.6
            updates = [gr, gp, hover_thr]
            # print(f"MPC Fail: {res.message}") # Debug only
            
        return updates, solve_time

def run_mpc_mission(no_render=False, difficulty="medium"):
    print(f"\nStarting MPC Landing Mission Scenario (Render: {not no_render}, Diff: {difficulty})...")
    
    render_mode = "human" if not no_render else None
    env = gym.make("RocketLander-v0", render_mode=render_mode)
    
    # Use the reusable scenario setup
    # Re-import inside function to avoid circular deps if any
    from scenario_landing import setup_landing_scenario
    obs = setup_landing_scenario(env, difficulty=difficulty)
    
    controller = MPCController()
    dt = 1.0 / 30.0 # Env steps
    
    try:
        # Super Hard Mode Heuristic:
        # If velocity is high, force a pitch-back maneuver for a few steps to help MPC find the solution
        # (Kick-start the braking)
        if difficulty == "super_hard":
            print("Executing Initial Hard Braking Maneuver...")
            for k in range(5): # 0.17 seconds (shorter to avoid over-rotation)
                # Hard Pitch Back
                action = np.zeros(16)
                action[0] = 1.0 # Throttle
                # Fix: Positive Gimbal -> Negative Rate -> Negative Pitch (-Theta)
                # Model says +Theta -> +Accel. So to brake (+Vx -> -Ax), we need -Theta.
                # So we need Negative Rate.
                # B[10,1] is negative. So we need POSITIVE Gimbal.
                action[1] = 1.0 
                
                action[3:16] = -1.0
                obs, reward, terminated, truncated, info = env.step(action)
                if not no_render: env.render()
        
        for i in range(1000):
            # MPC Step
            u_opt, t_solve = controller.optimize(obs)
            
            # Action Mapping
            # u_opt = [gimbal_roll, gimbal_pitch, throttle]
            # Action Space: [Throttle, GimbalP, GimbalR, RCS...]
            # Throttle: Map [0,1] -> [-1, 1]
            thr_cmd = np.clip(u_opt[2], 0.0, 1.0)
            thr_action = 2.0 * thr_cmd - 1.0
            
            # Gimbal: MPC outputs radians. Environment expects [-1, 1] mapped to range.
            # But wait, env expects normalized action?
            # rocket_lander_env.py:
            # gimbal_p = np.clip(action[1], -1.0, 1.0) * self.GIMBAL_RANGE
            # Wait, GIMBAL_RANGE is 0.35 rad (20 deg).
            # MPC outputs radians directly in u_opt[0,1].
            # So action = u_opt / 0.35 (normalized)
            
            gimbal_range = 0.35
            gp_action = u_opt[1] / gimbal_range
            gr_action = u_opt[0] / gimbal_range
            
            action = np.zeros(16)
            action[0] = thr_action
            action[1] = gp_action
            action[2] = gr_action
            
            # Independent Yaw Damping (Simple P-Controller)
            # wz is obs[12]. Target 0.
            wz = obs[12]
            yaw_cmd = np.clip(0.0 - wz * 2.0, -1.0, 1.0)
            # Wrapper for RCS Mapping based on Env Config
            # Pos Torque (Turn Left): Indices 11, 12 (Config 8, 9) -> Net Force 0, Net Torque +
            # Neg Torque (Turn Right): Indices 13, 14 (Config 10, 11) -> Net Force 0, Net Torque -
            
            action[3:16] = -1.0 # Reset
            
            # Yaw Damping
            if yaw_cmd > 0.05: # Need Pos Torque
                 action[3+8] = yaw_cmd # Index 11
                 action[3+9] = yaw_cmd # Index 12
                 
            if yaw_cmd < -0.05: # Need Neg Torque
                 action[3+10] = -yaw_cmd # Index 13
                 action[3+11] = -yaw_cmd # Index 14
            
            # Legs
            if obs[17] < 20.0: action[15] = 1.0
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            
            if not no_render:
                env.render()
            
            if i % 10 == 0:
                print(f"T={i/30:.2f}s | Alt: {obs[17]:.1f} | Thr: {thr_cmd:.2f} | Solve: {t_solve*1000:.1f}ms")
                # User Request: Full Obs & Action Log
                # Obs: Pos(0-2), Quat(3-6) -> RPY, Vel(7-9), AngVel(10-12)
                rpy_deg = np.degrees(p.getEulerFromQuaternion(obs[3:7]))
                print(f"  State: Pos={obs[0:3]}, Vel={obs[7:10]}, RPY={rpy_deg}, Rates={obs[10:13]}")
                print(f"  Action: Thr={thr_action:.2f}, G_Roll={gr_action:.2f}, G_Pitch={gp_action:.2f}, Yaw_Cmd={yaw_cmd:.2f}")
            
            if terminated or truncated:
                print("\nMission Ended.")
                if reward > 0: print("SUCCESSFUL LANDING! 🚀")
                else: 
                    print("CRASH OR OUT OF BOUNDS. 💥")
                    print(f"Final State: Alt={obs[17]:.1f}, Tilt={np.degrees(np.linalg.norm(obs[3:5])):.1f}deg")
                    print(f"Pos: {obs[0:3]}")
                    print(f"Vel: {obs[7:10]}")
                    print(f"AngVel: {obs[10:13]}")
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", help="Run without PyBullet GUI")
    parser.add_argument("--difficulty", type=str, default="medium", help="Scenario difficulty (easy, medium, hard)")
    args = parser.parse_args()
    
    run_mpc_mission(no_render=args.no_render, difficulty=args.difficulty)
