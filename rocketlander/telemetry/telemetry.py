import numpy as np

class TelemetryLogger:
    def __init__(self):
        self.history = {
            't': [],
            'phase': [],
            'x': [], 'y': [], 'z': [],
            'vx': [], 'vy': [], 'vz': [],
            'roll': [], 'pitch': [], 'yaw': [],
            'throttle': [],
            'rcs_roll': [], 'rcs_pitch': [], 'rcs_yaw': [],
            'ref_x': [], 'ref_y': [], 'ref_z': [],
            'des_roll': [], 'des_pitch': [], 'des_yaw': []
        }

    def log(self, t, phase, pos, vel, orn_euler, throttle, u_roll, u_pitch, u_yaw, ref_pos, des_r, des_p, des_y):
        self.history['t'].append(t)
        self.history['phase'].append(phase.value)
        self.history['x'].append(pos[0])
        self.history['y'].append(pos[1])
        self.history['z'].append(pos[2])
        self.history['vx'].append(vel[0])
        self.history['vy'].append(vel[1])
        self.history['vz'].append(vel[2])
        self.history['roll'].append(orn_euler[0])
        self.history['pitch'].append(orn_euler[1])
        self.history['yaw'].append(orn_euler[2])
        self.history['throttle'].append(throttle)
        self.history['rcs_roll'].append(u_roll)
        self.history['rcs_pitch'].append(u_pitch)
        self.history['rcs_yaw'].append(u_yaw)
        self.history['ref_x'].append(ref_pos[0])
        self.history['ref_y'].append(ref_pos[1])
        self.history['ref_z'].append(ref_pos[2])
        self.history['des_roll'].append(des_r)
        self.history['des_pitch'].append(des_p)
        self.history['des_yaw'].append(des_y)

    def get_data(self):
        # Convert lists to numpy arrays
        return {k: np.array(v) for k, v in self.history.items()}

    def print_hud(self, t, phase, pos, vel, orn_euler, throttle):
        r_deg = orn_euler[0] * 57.295779513
        p_deg = orn_euler[1] * 57.295779513
        y_deg = orn_euler[2] * 57.295779513
        print(f"[{t:5.1f}s] {phase.name:<12} | "
              f"Pos: ({pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:6.1f}) | "
              f"Ori: ({r_deg:5.1f}, {p_deg:5.1f}, {y_deg:5.1f}) | "
              f"Thrust: {throttle*100:5.1f}%")
