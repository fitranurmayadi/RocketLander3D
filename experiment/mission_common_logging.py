import os
from typing import Any, Dict, List, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_step_log(log: Dict[str, List[Any]], t: float, obs, action, reward: float, debug: Dict[str, Any]):
    # Expect obs normalized; map to physical units used in other scripts
    log['t'].append(float(t))
    log['reward'].append(float(reward))

    # obs indices from rocket_lander_env.py (normalize_obs=True):
    # pos_n(0:3), quat(3:7), vel_n(7:10), ang_vel_local_n(10:13), contacts(13:17), alt_n(17), fuel(18)
    pos_n = obs[0:3]
    vel_n = obs[7:10]
    quat = obs[3:7]
    ang_vel_local_n = obs[10:13]

    # Store raw normalized and scaled versions (alt/vel)
    log['pos_x'].append(float(pos_n[0] * 500.0))
    log['pos_y'].append(float(pos_n[1] * 500.0))
    log['pos_z'].append(float(pos_n[2] * 500.0))

    log['vel_x'].append(float(vel_n[0] * 50.0))
    log['vel_y'].append(float(vel_n[1] * 50.0))
    log['vel_z'].append(float(vel_n[2] * 50.0))

    log['fuel'].append(float(obs[18]))

    # Use debug dict for attitude/phase already computed by controllers
    if 'r_deg' in debug:
        log['r_deg'].append(float(debug['r_deg']))
    else:
        log['r_deg'].append(float('nan'))

    if 'p_deg' in debug:
        log['p_deg'].append(float(debug['p_deg']))
    else:
        log['p_deg'].append(float('nan'))

    if 'yaw_deg' in debug:
        log['yaw_deg'].append(float(debug['yaw_deg']))
    else:
        log['yaw_deg'].append(float('nan'))

    log['state'].append(float(debug.get('state', 0.0)))
    log['dist_h'].append(float(debug.get('dist_h', 0.0)))
    log['alt'].append(float(debug.get('alt', 0.0)))
    log['vz'].append(float(debug.get('vz', 0.0)))

    # action can be large; keep minimal
    log['throttle'].append(float(action[0]))
    log['legs_cmd'].append(float(action[15]))

