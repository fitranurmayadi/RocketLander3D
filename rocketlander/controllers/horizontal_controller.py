import numpy as np
import math
from rocketlander.controllers.pid import PID

class HorizontalController:
    """Controls desired pitch/roll based on horizontal position/velocity error."""
    def __init__(self, dt):
        pass
        
    def compute(self, pos_error, vel_error, desired_acc, current_yaw, phase=None) -> (float, float):
        # Dynamically adjust gains for landing burn precision
        from rocketlander.mission.state_machine import MissionPhase
        if phase == MissionPhase.LANDING_BURN:
            kp = 0.8
            kd = 2.5
        else:
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
        
        # Use inverse kinematics of XYZ Euler sequence to calculate physically correct desired pitch/roll
        # (This resolves the acceleration asymmetry between X and Y at high tilts where X acceleration
        # is scaled by cos(roll) but Y is not scaled by cos(pitch))
        tot_acc = math.sqrt(bx**2 + by**2 + 9.81**2)
        ux = bx / tot_acc
        uy = by / tot_acc
        uz = 9.81 / tot_acc
        
        # Clamp to avoid asin domain errors (limit to max ~57 degrees to stay safe from 80 deg crash limit)
        uy_clamped = max(-0.84, min(0.84, uy))
        desired_roll = math.asin(-uy_clamped)
        desired_pitch = math.atan2(ux, uz)
        
        desired_pitch = max(-1.00, min(1.00, desired_pitch))
        desired_roll = max(-1.00, min(1.00, desired_roll))
        
        # Calculate actual clamped acceleration magnitude
        ax_clamped = 9.81 * math.sin(desired_pitch) * math.cos(desired_roll)
        ay_clamped = -9.81 * math.sin(desired_roll)
        ah_mag = math.sqrt(ax_clamped**2 + ay_clamped**2)
        
        return desired_pitch, desired_roll, ah_mag
