from enum import Enum
import math

class MissionPhase(Enum):
    PRELAUNCH = 0
    ASCENT = 1
    WAYPOINT_NAV = 2
    BOOSTBACK = 3
    ENTRY_BURN = 4
    LANDING_BURN = 5
    LANDED = 6
    CRASHED = 7

class MissionConfig:
    def __init__(self):
        self.ignition_delay = 2.0
        self.ascent_target_alt = 1000.0
        self.waypoint = [500.0, 500.0, 1000.0]
        self.landing_pad = [0.0, 0.0, 0.0]

class RocketState:
    """Dataclass to hold parsed env state for mission logic"""
    def __init__(self):
        self.pos = [0,0,0]
        self.vel = [0,0,0]
        self.orn_quat = [0,0,0,1]
        self.orn_euler = [0,0,0]
        self.ang_vel = [0,0,0]
        self.fuel = 1.0
        self.t = 0.0
        self.contacts = [0,0,0,0]

class MissionStateMachine:
    def __init__(self, config: MissionConfig):
        self.config = config
        self._phase = MissionPhase.PRELAUNCH

    @property
    def phase(self) -> MissionPhase:
        return self._phase

    @property
    def is_terminal(self) -> bool:
        return self._phase in [MissionPhase.LANDED, MissionPhase.CRASHED]

    def update(self, state: RocketState) -> MissionPhase:
        # State machine transition logic based on time
        t = state.t - self.config.ignition_delay
        
        if self._phase == MissionPhase.PRELAUNCH:
            if state.t > self.config.ignition_delay:
                self._phase = MissionPhase.ASCENT
                
        elif self._phase not in [MissionPhase.LANDED, MissionPhase.CRASHED]:
            if t < 30.0:
                self._phase = MissionPhase.ASCENT
            elif t < 35.0:
                self._phase = MissionPhase.WAYPOINT_NAV
            elif t < 60.0:
                self._phase = MissionPhase.BOOSTBACK
            else:
                hor_dist = math.sqrt(state.pos[0]**2 + state.pos[1]**2)
                if state.pos[2] < 150.0 and hor_dist < 20.0:
                    self._phase = MissionPhase.LANDING_BURN
                elif self._phase != MissionPhase.LANDING_BURN:
                    self._phase = MissionPhase.ENTRY_BURN
                
            # Landing checks during LANDING_BURN
            if self._phase == MissionPhase.LANDING_BURN:
                speed = (state.vel[0]**2 + state.vel[1]**2 + state.vel[2]**2)**0.5
                tilt = max(abs(state.orn_euler[0]), abs(state.orn_euler[1]))
                contact = sum(state.contacts) > 0
                
                if contact:
                    if speed < 2.5 and tilt < math.radians(10):
                        self._phase = MissionPhase.LANDED
                    else:
                        print(f"DEBUG: Landing crash. Speed: {speed:.2f}, Tilt: {math.degrees(tilt):.2f}")
                        self._phase = MissionPhase.CRASHED

        # Global crash conditions (structural failure or extreme tilt during flight)
        if self._phase not in [MissionPhase.PRELAUNCH, MissionPhase.LANDED, MissionPhase.CRASHED]:
            tilt = max(abs(state.orn_euler[0]), abs(state.orn_euler[1]))
            max_tilt = math.radians(80.0)
            if tilt > max_tilt:
                print(f"DEBUG: Crash due to extreme tilt: {math.degrees(tilt):.1f} deg (limit: {math.degrees(max_tilt):.1f})")
                self._phase = MissionPhase.CRASHED
                
            # Floor collision outside of landing phase
            contact = sum(state.contacts) > 0
            if contact and self._phase not in [MissionPhase.PRELAUNCH, MissionPhase.LANDING_BURN]:
                speed = (state.vel[0]**2 + state.vel[1]**2 + state.vel[2]**2)**0.5
                if speed > 2.0 or tilt > math.radians(10):
                    print(f"DEBUG: Crash due to premature ground contact. Speed: {speed:.1f}, Tilt: {math.degrees(tilt):.1f}")
                    self._phase = MissionPhase.CRASHED

        return self._phase
