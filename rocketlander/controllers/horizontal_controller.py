import numpy as np
import math
from rocketlander.controllers.pid import PID

class HorizontalController:
    """Controls desired pitch/roll based on horizontal position/velocity error."""
    def __init__(self, dt):
        pass
        
    def compute(self, pos_error, vel_error, desired_acc, current_yaw) -> (float, float):

        # kp=0.8, kd=1.8 is CRITICALLY DAMPED (kd = 2*sqrt(kp)). 
        # This tightly locks onto the trajectory preventing any lazy drift or floating,
        # ensuring the landing is exactly at [0.0, 0.0].
        # Extremely overdamped to account for attitude response delays
        kp = 0.4
        kd = 2.0
        
        # Add feedforward acceleration to eliminate tracking lag
        ax_world = desired_acc[0] + kp * pos_error[0] + kd * vel_error[0]
        ay_world = desired_acc[1] + kp * pos_error[1] + kd * vel_error[1]
        
        # Rotate acceleration commands to body frame using yaw
        cy = math.cos(current_yaw)
        sy = math.sin(current_yaw)
        
        bx = ax_world * cy + ay_world * sy
        by = -ax_world * sy + ay_world * cy  
        
        # Physics mapping: a = g * tan(theta) => theta = atan(a/g)
        # Positive pitch tilts nose forward -> thrust points forward (+X)
        # In PyBullet, pitch is around Y (forward/back) and roll is around X (left/right).
        desired_pitch = math.atan2(bx, 9.81)
        desired_roll = math.atan2(-by, 9.81)
        
        # Clamp tilt to prevent tumbling (limit to ~34 degrees to prevent extreme swinging)
        desired_pitch = max(-0.60, min(0.60, desired_pitch))
        desired_roll = max(-0.60, min(0.60, desired_roll))
        
        # Required horizontal acceleration magnitude corresponding to the clamped desired tilt
        # Since a = g * tan(theta), we calculate the actual commanded horizontal acceleration
        ax_clamped = 9.81 * math.tan(desired_pitch)
        ay_clamped = 9.81 * math.tan(desired_roll)
        ah_mag = math.sqrt(ax_clamped**2 + ay_clamped**2)
        
        return desired_pitch, desired_roll, ah_mag
