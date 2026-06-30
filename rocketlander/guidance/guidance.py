import numpy as np
import math
from rocketlander.mission.state_machine import MissionPhase

class GuidanceCommand:
    def __init__(self):
        self.desired_pos = np.zeros(3)
        self.desired_vel = np.zeros(3)
        self.desired_acc = np.zeros(3)
        self.desired_heading = 0.0
        self.pos_error = np.zeros(3)
        self.vel_error = np.zeros(3)

class GuidanceModule:
    def __init__(self, trajectory_planner):
        self.planner = trajectory_planner

    def compute(self, current_pos, current_vel, current_att, mission_time, phase=None) -> GuidanceCommand:
        cmd = GuidanceCommand()
        
        # Get reference from planner
        ref_pos, ref_vel, ref_acc = self.planner.get_reference(mission_time)
        
        cmd.desired_pos = ref_pos
        cmd.desired_vel = ref_vel
        cmd.desired_acc = ref_acc
        
        cmd.pos_error = ref_pos - np.array(current_pos)
        cmd.vel_error = ref_vel - np.array(current_vel)
        
        # Determine desired heading (yaw).
        # We calculate the constant heading vector per phase to avoid dynamic yaw wobble,
        # which causes gyroscopic cross-talk between pitch and roll.
        # To prevent Euler angle cross-coupling (which causes lateral deviation in top-down view),
        # we align the rocket's yaw EXACTLY with the straight-line path of the trajectory.
        # Since it's a constant angle per phase, it will NOT oscillate!
        # Keep the heading (Yaw) fixed at 0.0 to prevent any yaw rotation.
        # Targeting will be handled purely by roll and pitch.
        cmd.desired_heading = 0.0
        return cmd
