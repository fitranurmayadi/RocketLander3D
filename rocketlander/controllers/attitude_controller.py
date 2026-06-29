from rocketlander.controllers.pid import PID

class AttitudeController:
    """Controls RCS/gimbal from attitude error (roll, pitch, yaw)."""
    def __init__(self, dt):
        self.roll_pid = PID(kp=2.0, ki=0.0, kd=1.0, dt=dt, limit=1.0)
        self.pitch_pid = PID(kp=2.0, ki=0.0, kd=1.0, dt=dt, limit=1.0)
        self.yaw_pid = PID(kp=2.0, ki=0.0, kd=1.0, dt=dt, limit=1.0)

    def compute(self, roll_err, pitch_err, yaw_err) -> (float, float, float):
        """
        Returns u_roll, u_pitch, u_yaw in [-1, 1].
        Assumes errors are in radians and target - current.
        """
        r_cmd = self.roll_pid.update(roll_err, is_angular=True)
        p_cmd = self.pitch_pid.update(pitch_err, is_angular=True)
        y_cmd = self.yaw_pid.update(yaw_err, is_angular=True)

        return r_cmd, p_cmd, y_cmd
