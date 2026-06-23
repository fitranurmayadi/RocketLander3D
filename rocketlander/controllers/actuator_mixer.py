import numpy as np

class ActuatorMixer:
    """
    Maps logical high-level commands to the 16D action vector for RocketLander-v1.
    Action Space [-1, 1]:
    0: Throttle (mapped via (action[0]+1)/2 to 0..1 in env)
    1: Gimbal Pitch
    2: Gimbal Roll
    3-14: RCS Thrusters
    15: Legs Deploy (>0 means deploy)
    """
    def __init__(self):
        pass

    def mix(self, throttle: float, gimbal_pitch: float, gimbal_roll: float,
            rcs_roll: float, rcs_pitch: float, rcs_yaw: float, legs_deploy: bool) -> np.ndarray:
        
        action = np.zeros(16, dtype=np.float32)

        # Map 0..1 throttle to -1..1 action
        throttle = max(0.0, min(1.0, throttle))
        action[0] = throttle * 2.0 - 1.0
        
        # Gimbal
        # Positive gimbal_pitch pushes bottom +X -> negative pitch -> so we negate
        # Positive gimbal_roll pushes bottom -Y -> positive roll -> wait! Let's re-verify:
        # If gimbal_roll > 0, f_y = -sin(gimbal_roll). Bottom pushed -Y.
        # Nose goes +Y. Positive roll is nose -Y. So nose +Y is negative roll!
        # So we negate both.
        # However, `gimbal_pitch` argument here represents the direct control input.
        # Let's map it:
        action[1] = max(-1.0, min(1.0, -gimbal_pitch))
        action[2] = max(-1.0, min(1.0, -gimbal_roll))

        # RCS mapping
        # 3-14: (rcs * 2.0 - 1.0)
        # We need to map abstract roll/pitch/yaw to the 12 RCS thrusters
        rcs = np.zeros(12, dtype=float)
        # Simple RCS mapping based on previous controller tuning
        u_roll = max(-1.0, min(1.0, rcs_roll))
        u_pitch = max(-1.0, min(1.0, rcs_pitch))
        u_yaw = max(-1.0, min(1.0, rcs_yaw))

        # PITCH (Positive pitch torque needed = rcs[1] + rcs[4])
        # Negative pitch torque needed = rcs[0] + rcs[5]
        if u_pitch > 0:
            rcs[1] = u_pitch
            rcs[4] = u_pitch
            rcs[0] = 0.0
            rcs[5] = 0.0
        else:
            rcs[0] = -u_pitch
            rcs[5] = -u_pitch
            rcs[1] = 0.0
            rcs[4] = 0.0

        # ROLL (Positive roll torque needed = rcs[2] + rcs[7])
        # Negative roll torque needed = rcs[3] + rcs[6]
        if u_roll > 0:
            rcs[2] = u_roll
            rcs[7] = u_roll
            rcs[3] = 0.0
            rcs[6] = 0.0
        else:
            rcs[3] = -u_roll
            rcs[6] = -u_roll
            rcs[2] = 0.0
            rcs[7] = 0.0

        # YAW (Positive yaw torque needed = rcs[8] + rcs[9])
        # Negative yaw torque needed = rcs[10] + rcs[11]
        if u_yaw > 0:
            rcs[8] = u_yaw
            rcs[9] = u_yaw
            rcs[10] = 0.0
            rcs[11] = 0.0
        else:
            rcs[10] = -u_yaw
            rcs[11] = -u_yaw
            rcs[8] = 0.0
            rcs[9] = 0.0

        # Map to action array (rcs mapping: action[3..14] = rcs * 2 - 1)
        action[3:15] = (rcs * 2.0 - 1.0).astype(np.float32)

        # Legs
        action[15] = 1.0 if legs_deploy else -1.0

        return action
