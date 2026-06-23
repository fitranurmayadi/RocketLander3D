import numpy as np

class QuinticPolynomial:
    """
    1D Quintic Polynomial for Trajectory Generation.
    Solves for coefficients a0..a5 to satisfy position, velocity, and acceleration
    at t=0 and t=T.
    """
    def __init__(self, p0, v0, a0, pT, vT, aT, T):
        self.a0 = p0
        self.a1 = v0
        self.a2 = a0 / 2.0

        if T <= 0:
            self.a3 = 0.0
            self.a4 = 0.0
            self.a5 = 0.0
            self.T = 0.0
            return
            
        self.T = T

        # Formulate linear equations for a3, a4, a5
        # p(T) = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4 + a5*T^5 = pT
        # v(T) = a1 + 2*a2*T + 3*a3*T^2 + 4*a4*T^3 + 5*a5*T^4 = vT
        # a(T) = 2*a2 + 6*a3*T + 12*a4*T^2 + 20*a5*T^3 = aT
        
        A = np.array([
            [T**3, T**4, T**5],
            [3 * T**2, 4 * T**3, 5 * T**4],
            [6 * T, 12 * T**2, 20 * T**3]
        ])
        
        b = np.array([
            pT - self.a0 - self.a1 * T - self.a2 * T**2,
            vT - self.a1 - 2 * self.a2 * T,
            aT - 2 * self.a2
        ])
        
        x = np.linalg.solve(A, b)
        
        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        if t < 0: t = 0
        if t > self.T: t = self.T
        return (self.a0 + self.a1 * t + self.a2 * t**2 + 
                self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5)

    def calc_first_derivative(self, t):
        if t < 0: t = 0
        if t > self.T: t = self.T
        return (self.a1 + 2 * self.a2 * t + 
                3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4)

    def calc_second_derivative(self, t):
        if t < 0: t = 0
        if t > self.T: t = self.T
        return (2 * self.a2 + 6 * self.a3 * t + 
                12 * self.a4 * t**2 + 20 * self.a5 * t**3)


class Trajectory3D:
    """
    3D Trajectory composed of 3 independent quintic polynomials (x, y, z).
    """
    def __init__(self, pos0, vel0, acc0, posT, velT, accT, T):
        self.T = T
        self.x_poly = QuinticPolynomial(pos0[0], vel0[0], acc0[0], posT[0], velT[0], accT[0], T)
        self.y_poly = QuinticPolynomial(pos0[1], vel0[1], acc0[1], posT[1], velT[1], accT[1], T)
        self.z_poly = QuinticPolynomial(pos0[2], vel0[2], acc0[2], posT[2], velT[2], accT[2], T)

    def get_state(self, t):
        pos = np.array([
            self.x_poly.calc_point(t),
            self.y_poly.calc_point(t),
            self.z_poly.calc_point(t)
        ])
        vel = np.array([
            self.x_poly.calc_first_derivative(t),
            self.y_poly.calc_first_derivative(t),
            self.z_poly.calc_first_derivative(t)
        ])
        acc = np.array([
            self.x_poly.calc_second_derivative(t),
            self.y_poly.calc_second_derivative(t),
            self.z_poly.calc_second_derivative(t)
        ])
        return pos, vel, acc
