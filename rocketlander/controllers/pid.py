import math

class PID:
    def __init__(self, kp, ki, kd, dt, limit, name=""):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.limit = limit
        self.name = name

        self.integral = 0.0
        self.prev_error = 0.0
        self.value = 0.0

    def update(self, error, is_angular=False, derivative=None):
        if is_angular:
            # Normalize angular error to [-pi, pi]
            error = (error + math.pi) % (2 * math.pi) - math.pi

        self.integral += error * self.dt
        
        # Anti-windup
        if self.limit is not None and self.ki > 0:
            max_int = self.limit / self.ki
            self.integral = max(-max_int, min(max_int, self.integral))

        if derivative is None:
            derivative = (error - self.prev_error) / self.dt

        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative

        self.value = p_term + i_term + d_term

        if self.limit is not None:
            self.value = max(-self.limit, min(self.limit, self.value))

        self.prev_error = error
        return self.value

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.value = 0.0
