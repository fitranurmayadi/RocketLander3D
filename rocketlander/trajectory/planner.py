import numpy as np
from rocketlander.trajectory.quintic import Trajectory3D

class TrajectorySegment:
    def __init__(self, traj_3d: Trajectory3D, duration: float, name: str):
        self.traj_3d = traj_3d
        self.duration = duration
        self.name = name

class TrajectoryPlanner:
    def __init__(self, mission_config):
        self.config = mission_config
        self.segments = []
        self.total_duration = 0.0

    def plan_full_mission(self, start_pos, start_vel):
        self.segments = []
        
        pos0 = np.array(start_pos)
        vel0 = np.array(start_vel)
        acc0 = np.zeros(3)
        
        # Segment 1: Tower Clear (Vertical Launch)
        pos1 = np.array([start_pos[0], start_pos[1], 100.0])
        vel1 = np.array([0.0, 0.0, 40.0])
        acc1 = np.zeros(3)
        t1 = 5.0
        
        seg1 = Trajectory3D(pos0, vel0, acc0, pos1, vel1, acc1, t1)
        self.segments.append(TrajectorySegment(seg1, t1, "Launch"))
        
        # Segment 2: Gravity Turn / Ascent Arc - Stop at waypoint!
        pos2 = np.array(self.config.waypoint)
        vel2 = np.zeros(3)
        acc2 = np.zeros(3)
        t2 = 25.0
        
        seg2 = Trajectory3D(pos1, vel1, acc1, pos2, vel2, acc2, t2)
        self.segments.append(TrajectorySegment(seg2, t2, "GravityTurn"))
        
        # Segment 3: Waypoint Coast (Hover at waypoint)
        pos3 = pos2
        vel3 = np.zeros(3)
        acc3 = np.zeros(3)
        t3 = 5.0
        
        seg3 = Trajectory3D(pos2, vel2, acc2, pos3, vel3, acc3, t3)
        self.segments.append(TrajectorySegment(seg3, t3, "WaypointNav"))
        
        # Segment 4: Boostback / Entry
        pos4 = np.array([self.config.landing_pad[0], self.config.landing_pad[1], 250.0])
        vel4 = np.array([0.0, 0.0, -10.0])
        acc4 = np.zeros(3)
        t4 = 25.0
        
        seg4 = Trajectory3D(pos3, vel3, acc3, pos4, vel4, acc4, t4)
        self.segments.append(TrajectorySegment(seg4, t4, "Boostback"))
        
        # Segment 5: Landing Burn (Terminal descent)
        pos5 = np.array([self.config.landing_pad[0], self.config.landing_pad[1], 3.0])
        vel5 = np.array([0.0, 0.0, -1.0])
        acc5 = np.zeros(3)
        t5 = 10.0
        
        seg5 = Trajectory3D(pos4, vel4, acc4, pos5, vel5, acc5, t5)
        self.segments.append(TrajectorySegment(seg5, t5, "LandingBurn"))
        
        self.total_duration = t1 + t2 + t3 + t4 + t5

    def get_reference(self, t):
        current_t = t
        for seg in self.segments:
            if current_t <= seg.duration:
                return seg.traj_3d.get_state(current_t)
            current_t -= seg.duration
            
        # If past end, return last state of last segment
        if self.segments:
            return self.segments[-1].traj_3d.get_state(self.segments[-1].duration)
            
        return np.zeros(3), np.zeros(3), np.zeros(3)
