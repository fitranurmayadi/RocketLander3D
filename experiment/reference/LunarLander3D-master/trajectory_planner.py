import numpy as np

class QuinticPolynomial:
    def __init__(self, xi, vi, ai, xf, vf, af, T):
        # Calculate coefficients a0..a5
        # x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5 * ai

        A = np.array([
            [T**3, T**4, T**5],
            [3*T**2, 4*T**3, 5*T**4],
            [6*T, 12*T**2, 20*T**3]
        ])
        
        b = np.array([
            xf - self.a0 - self.a1*T - self.a2*T**2,
            vf - self.a1 - 2*self.a2*T,
            af - 2*self.a2
        ])

        try:
            x = np.linalg.solve(A, b)
            self.a3 = x[0]
            self.a4 = x[1]
            self.a5 = x[2]
        except np.linalg.LinAlgError:
            print("Error: Matrix Turn Singular. Check Duration T.")
            self.a3 = 0
            self.a4 = 0
            self.a5 = 0

    def calc_point(self, t):
        xt = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5
        vt = self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4
        at = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3
        return xt, vt, at

class Trajectory3D:
    def __init__(self, start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, duration):
        self.Tx = QuinticPolynomial(start_pos[0], start_vel[0], start_acc[0], end_pos[0], end_vel[0], end_acc[0], duration)
        self.Ty = QuinticPolynomial(start_pos[1], start_vel[1], start_acc[1], end_pos[1], end_vel[1], end_acc[1], duration)
        self.Tz = QuinticPolynomial(start_pos[2], start_vel[2], start_acc[2], end_pos[2], end_vel[2], end_acc[2], duration)
        self.duration = duration

    def get_state(self, t):
        if t > self.duration:
            t = self.duration # Clamp to end state
        
        rx, vx, ax = self.Tx.calc_point(t)
        ry, vy, ay = self.Ty.calc_point(t)
        rz, vz, az = self.Tz.calc_point(t)
        
        return np.array([rx, ry, rz]), np.array([vx, vy, vz]), np.array([ax, ay, az])

if __name__ == "__main__":
    # Test
    traj = Trajectory3D([0,0,0], [0,0,0], [0,0,0], [10,10,10], [0,0,0], [0,0,0], 2.0)
    p, v, a = traj.get_state(1.0)
    print(f"t=1.0: Pos={p}, Vel={v}, Acc={a}")
