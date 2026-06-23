"""
osc_sender.py
=============
Lightweight OSC sender helper used by all mission scripts.
Sends telemetry data to the live_dashboard.py process.

Protocol:
    /episode  (int ep_num, str label)
    /state    (int step, float H, float Vz, float dist, float reward, int state_id, float cum_reward)
    /attitude (float roll_deg, float pitch_deg, float yaw_deg)
    /velocity (float vx, float vy, float vz)
    /action   (float main_thrust, float mean_rcs, float g_force)
"""

from pythonosc import udp_client

_OSC_IP   = "127.0.0.1"
_OSC_PORT = 9001
_client   = None

def init():
    """Call once at mission start to create the UDP client."""
    global _client
    try:
        _client = udp_client.SimpleUDPClient(_OSC_IP, _OSC_PORT)
        print(f"[OSC] Telemetry sender ready -> {_OSC_IP}:{_OSC_PORT}")
    except Exception as e:
        print(f"[OSC] Warning: Could not init sender: {e}")
        _client = None

def send_episode(ep_num: int, label: str = "RANDOM"):
    if _client:
        try:
            _client.send_message("/episode", [ep_num, label])
        except Exception:
            pass

def send_state(step: int, h: float, vz: float, dist: float,
               reward: float, state_id: int, cum_reward: float):
    if _client:
        try:
            _client.send_message("/state",
                [int(step), float(h), float(vz), float(dist),
                 float(reward), int(state_id), float(cum_reward)])
        except Exception:
            pass

def send_attitude(roll_deg: float, pitch_deg: float, yaw_deg: float):
    if _client:
        try:
            _client.send_message("/attitude",
                [float(roll_deg), float(pitch_deg), float(yaw_deg)])
        except Exception:
            pass

def send_velocity(vx: float, vy: float, vz: float):
    if _client:
        try:
            _client.send_message("/velocity", [float(vx), float(vy), float(vz)])
        except Exception:
            pass

def send_action(main_thrust: float, mean_rcs: float, g_force: float):
    if _client:
        try:
            _client.send_message("/action",
                [float(main_thrust), float(mean_rcs), float(g_force)])
        except Exception:
            pass
