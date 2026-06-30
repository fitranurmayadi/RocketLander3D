from rocketlander.controllers.pid import PID

class AttitudeController:
    """Controls RCS/gimbal from attitude error (roll, pitch, yaw)."""
    def __init__(self, dt):
        self.roll_pid = PID(kp=4.0, ki=0.2, kd=2.5, dt=dt, limit=1.0)
        self.pitch_pid = PID(kp=4.0, ki=0.2, kd=2.5, dt=dt, limit=1.0)
        self.yaw_pid = PID(kp=4.0, ki=0.2, kd=2.5, dt=dt, limit=1.0)

    def compute(self, roll_err, pitch_err, yaw_err, ang_vel=None) -> (float, float, float):
        """
        Returns r_cmd, p_cmd, y_cmd in [-1, 1].
        Assumes errors are in radians and target - current.
        If ang_vel [roll_rate, pitch_rate, yaw_rate] is provided, uses direct rate feedback.
        """
        if ang_vel is not None:
            r_cmd = self.roll_pid.update(roll_err, is_angular=True, derivative=-ang_vel[0])
            p_cmd = self.pitch_pid.update(pitch_err, is_angular=True, derivative=-ang_vel[1])
            y_cmd = self.yaw_pid.update(yaw_err, is_angular=True, derivative=-ang_vel[2])
        else:
            r_cmd = self.roll_pid.update(roll_err, is_angular=True)
            p_cmd = self.pitch_pid.update(pitch_err, is_angular=True)
            y_cmd = self.yaw_pid.update(yaw_err, is_angular=True)

        return r_cmd, p_cmd, y_cmd
