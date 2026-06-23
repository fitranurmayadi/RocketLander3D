from rocketlander.controllers.pid import PID

class AltitudeController:
    """Controls throttle based on altitude and vz error."""
    def __init__(self, dt, gravity_ff=0.10):
        self.gravity_ff = gravity_ff
        # PID for vertical velocity
        # During ascent/hover we control vz
        self.vz_pid = PID(kp=0.8, ki=0.15, kd=0.0, dt=dt, limit=1.0)
        # Outer loop for altitude -> desired vz
        self.alt_pid = PID(kp=1.5, ki=0.0, kd=0.0, dt=dt, limit=25.0)

    def compute(self, alt_error, vz_error, az_ff=0.0, current_alt=None) -> float:
        # Cascade control:
        # 1. Altitude error -> desired extra vz
        desired_vz_delta = self.alt_pid.update(alt_error)
        
        # Soft touchdown limit: don't command a massive descent rate if we are just a few meters off
        # If alt_error is negative (we are higher than target), desired_vz_delta is negative (downward)
        if desired_vz_delta < 0:
            if current_alt is not None:
                if current_alt > 300.0:
                    max_desc = 80.0
                elif current_alt > 100.0:
                    max_desc = 40.0
                elif current_alt > 30.0:
                    max_desc = 15.0
                else:
                    max_desc = 5.0
                desired_vz_delta = max(-max_desc, desired_vz_delta)
            else:
                # Fallback
                desired_vz_delta = max(-5.0, desired_vz_delta)
        
        # 2. Add to existing vz error
        total_vz_err = vz_error + desired_vz_delta
        
        # 3. Vz error -> throttle adjustment
        throttle_adj = self.vz_pid.update(total_vz_err)
        
        # Base throttle to hover + feedforward accel + PID correction
        throttle = self.gravity_ff + (az_ff * 0.05) + throttle_adj
        return max(0.0, min(1.0, throttle))
    
    def compute_direct_vz(self, vz_error, az_ff=0.0) -> float:
        """When we only want to track velocity, not position (e.g. landing phase)"""
        throttle_adj = self.vz_pid.update(vz_error)
        throttle = self.gravity_ff + (az_ff * 0.05) + throttle_adj
        return max(0.0, min(1.0, throttle))
