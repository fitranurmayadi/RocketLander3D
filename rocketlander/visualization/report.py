import matplotlib.pyplot as plt
import numpy as np
import os

def generate_report(telemetry_data, filename="images/rocketlander3d_report.png"):
    d = telemetry_data
    if len(d['t']) == 0:
        print("No telemetry data to plot.")
        return

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    
    # 1. Top-Down XY
    ax = axs[0, 0]
    ax.plot(d['ref_x'], d['ref_y'], 'g--', label='Planned')
    ax.plot(d['x'], d['y'], 'b-', label='Actual')
    ax.plot(0, 0, 'rx', markersize=10, label='Pad')
    ax.set_title("Top-Down XY Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # 2. Vertical Profile
    ax = axs[0, 1]
    ax.plot(d['t'], d['ref_z'], 'g--', label='Planned Alt')
    ax.plot(d['t'], d['z'], 'b-', label='Actual Alt')
    ax2 = ax.twinx()
    ax2.plot(d['t'], d['vz'], 'r-', alpha=0.5, label='Vz')
    ax.set_title("Vertical Profile")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax2.set_ylabel("Vz (m/s)")
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True)

    # 3. Attitude
    ax = axs[1, 0]
    ax.plot(d['t'], np.degrees(d['des_roll']), 'c--', label='Planned Roll')
    ax.plot(d['t'], np.degrees(d['des_pitch']), 'm--', label='Planned Pitch')
    ax.plot(d['t'], np.degrees(d['des_yaw']), 'y--', label='Planned Yaw')
    ax.plot(d['t'], np.degrees(d['roll']), 'b-', label='Actual Roll')
    ax.plot(d['t'], np.degrees(d['pitch']), 'r-', label='Actual Pitch')
    ax.plot(d['t'], np.degrees(d['yaw']), 'g-', label='Actual Yaw')
    ax.set_title("Attitude (degrees)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.legend(fontsize=8)
    ax.grid(True)

    # 4. Phase Timeline
    ax = axs[1, 1]
    ax.plot(d['t'], d['phase'], 'k-')
    ax.set_title("Mission Phase")
    ax.set_yticks(range(8))
    ax.set_yticklabels(["PRE", "ASC", "WAY", "BST", "ENT", "LND", "DONE", "CRSH"])
    ax.set_xlabel("Time (s)")
    ax.grid(True)

    # 5. Control Actions
    ax = axs[2, 0]
    ax.plot(d['t'], d['throttle'], label='Throttle')
    ax.plot(d['t'], d['rcs_pitch'], label='RCS Pitch', alpha=0.6)
    ax.set_title("Control Actions")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Command [0..1] / [-1..1]")
    ax.legend()
    ax.grid(True)

    # 6. Tracking Error
    ax = axs[2, 1]
    err = np.sqrt((d['x']-d['ref_x'])**2 + (d['y']-d['ref_y'])**2 + (d['z']-d['ref_z'])**2)
    ax.plot(d['t'], err, 'r-')
    ax.set_title("3D Tracking Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.grid(True)

    plt.tight_layout()
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"Report saved to {filename}")
