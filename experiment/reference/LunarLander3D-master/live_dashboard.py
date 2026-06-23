#!/usr/bin/env python3
"""
LunarLander3D Live Dashboard
============================
Real-time telemetry viewer using OSC protocol.
Run this BEFORE starting a mission script.

Usage:
    python live_dashboard.py [--no-osc]
"""

import sys
import threading
import collections
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server
import argparse

# --- Configuration ---
OSC_IP   = "127.0.0.1"
OSC_PORT = 9001
MAX_LEN  = 2000       # How many steps to keep in rolling buffer
WIN_W    = 800        # pixels wide
WIN_H    = 800        # pixels tall
WIN_X    = 20         # position: left side of screen
WIN_Y    = 60         # position: top margin

# --- Thread-safe rolling buffers ---
buf = {
    "step":    collections.deque(maxlen=MAX_LEN),
    "h":       collections.deque(maxlen=MAX_LEN),
    "vz":      collections.deque(maxlen=MAX_LEN),
    "dist":    collections.deque(maxlen=MAX_LEN),
    "reward":  collections.deque(maxlen=MAX_LEN),
    "cum_rew": collections.deque(maxlen=MAX_LEN),
    "roll":    collections.deque(maxlen=MAX_LEN),
    "pitch":   collections.deque(maxlen=MAX_LEN),
    "yaw":     collections.deque(maxlen=MAX_LEN),
    "vx":      collections.deque(maxlen=MAX_LEN),
    "vy":      collections.deque(maxlen=MAX_LEN),
    "thrust":  collections.deque(maxlen=MAX_LEN),
    "rcs":     collections.deque(maxlen=MAX_LEN),
    "g_force": collections.deque(maxlen=MAX_LEN),
    "state":   collections.deque(maxlen=MAX_LEN),
}
_lock  = threading.Lock()
_ep    = [1]
_label = ["—"]

def _reset_buffers():
    with _lock:
        for v in buf.values():
            v.clear()

# --- OSC Handlers ---
def _on_state(addr, step, h, vz, dist, reward, state_id, cum_reward):
    with _lock:
        buf["step"].append(step)
        buf["h"].append(h)
        buf["vz"].append(vz)
        buf["dist"].append(dist)
        buf["reward"].append(reward)
        buf["cum_rew"].append(cum_reward)
        buf["state"].append(state_id)

def _on_attitude(addr, roll, pitch, yaw):
    with _lock:
        buf["roll"].append(roll)
        buf["pitch"].append(pitch)
        buf["yaw"].append(yaw)

def _on_velocity(addr, vx, vy, vz):
    with _lock:
        buf["vx"].append(vx)
        buf["vy"].append(vy)

def _on_action(addr, main_thrust, mean_rcs, g_force):
    with _lock:
        buf["thrust"].append(main_thrust)
        buf["rcs"].append(mean_rcs)
        buf["g_force"].append(g_force)

def _on_episode(addr, ep_num, label):
    _ep[0] = ep_num
    _label[0] = label
    _reset_buffers()
    print(f"[Dashboard] New Episode {ep_num} ({label})")

def _start_osc_server():
    d = osc_dispatcher.Dispatcher()
    d.map("/state",    _on_state)
    d.map("/attitude", _on_attitude)
    d.map("/velocity", _on_velocity)
    d.map("/action",   _on_action)
    d.map("/episode",  _on_episode)
    server = osc_server.ThreadingOSCUDPServer((OSC_IP, OSC_PORT), d)
    print(f"[Dashboard] Listening on {OSC_IP}:{OSC_PORT}")
    server.serve_forever()

# --- Matplotlib Setup ---
plt.style.use('dark_background')
fig, axs = plt.subplots(3, 2, figsize=(WIN_W / 100, WIN_H / 100), dpi=100)
fig.canvas.manager.set_window_title("LunarLander3D Dashboard")  # Fixed window title for xdotool positioning
fig.patch.set_facecolor('#0d1117')
fig.suptitle("LunarLander3D — Live Telemetry", fontsize=14, color='white', fontweight='bold')

ACCENT = '#58a6ff'
RED    = '#f85149'
GREEN  = '#3fb950'
YELLOW = '#e3b341'
PURPLE = '#bc8cff'
GRAY   = '#6e7681'

for ax in axs.flat:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors=GRAY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.grid(True, color='#21262d', linewidth=0.6, linestyle='--')

# Sub-plots
ax_h,   ax_att  = axs[0]
ax_vel, ax_safe = axs[1]
ax_act, ax_perf = axs[2]

# Twin axes
ax_vz   = ax_h.twinx()
ax_vy   = ax_vel.twinx()
ax_rcs  = ax_act.twinx()
ax_cum  = ax_perf.twinx()

# Titles
for ax, t in [
    (ax_h,    "Altitude & Vertical Speed"),
    (ax_att,  "Attitude (Roll / Pitch / Yaw)"),
    (ax_vel,  "Horizontal Velocity"),
    (ax_safe, "Safety: G-Force & Distance"),
    (ax_act,  "Control Actions"),
    (ax_perf, "Mission Performance"),
]:
    ax.set_title(t, color='white', fontsize=9, pad=4)

plt.tight_layout(rect=[0, 0, 1, 0.96])

def _update(frame):
    with _lock:
        step    = list(buf["step"])
        h_data  = list(buf["h"])
        vz_data = list(buf["vz"])
        dist    = list(buf["dist"])
        rew     = list(buf["reward"])
        cumr    = list(buf["cum_rew"])
        roll    = list(buf["roll"])
        pitch   = list(buf["pitch"])
        yaw     = list(buf["yaw"])
        vx_data = list(buf["vx"])
        vy_data = list(buf["vy"])
        thr     = list(buf["thrust"])
        rcs     = list(buf["rcs"])
        g_f     = list(buf["g_force"])
        st_id   = list(buf["state"])
    n = min(len(step), len(h_data))
    if n < 2:
        return
    t = list(range(n))
    # Altitude & Vz
    ax_h.cla(); ax_vz.cla()
    ax_h.set_facecolor('#161b22')
    ax_h.grid(True, color='#21262d', linewidth=0.6, linestyle='--')
    ax_h.set_title("Altitude & Vertical Speed", color='white', fontsize=9, pad=4)
    ax_h.set_ylim(0, 2000)
    ax_vz.set_ylim(-20, 5)
    ax_h.plot(t, h_data[:n], color=ACCENT, lw=1.2, label='H (m)')
    ax_vz.plot(t, vz_data[:n], color=RED, lw=1, alpha=0.7, label='Vz (m/s)')
    ax_vz.axhline(-15, color=RED, linestyle=':', lw=0.8, alpha=0.5)
    ax_h.set_ylabel('H (m)', color=ACCENT, fontsize=8)
    ax_vz.set_ylabel('Vz (m/s)', color=RED, fontsize=8)
    ax_h.tick_params(colors=GRAY, labelsize=7)
    ax_vz.tick_params(colors=RED, labelsize=7)
    h1, l1 = ax_h.get_legend_handles_labels()
    h2, l2 = ax_vz.get_legend_handles_labels()
    ax_h.legend(h1+h2, l1+l2, loc='upper right', fontsize=7, framealpha=0.3)
    # Attitude
    rn = min(n, len(roll), len(pitch), len(yaw))
    ax_att.cla()
    ax_att.set_facecolor('#161b22')
    ax_att.grid(True, color='#21262d', linewidth=0.6, linestyle='--')
    ax_att.set_title("Attitude (Roll / Pitch / Yaw)", color='white', fontsize=9, pad=4)
    ax_att.set_ylim(-180, 180)
    if rn > 1:
        ax_att.plot(t[:rn], roll[:rn], color=RED, lw=1, label='Roll')
        ax_att.plot(t[:rn], pitch[:rn], color=GREEN, lw=1, label='Pitch')
        ax_att.plot(t[:rn], yaw[:rn], color=PURPLE, lw=0.8, alpha=0.7, label='Yaw')
    ax_att.set_ylabel('deg', color=GRAY, fontsize=8)
    ax_att.tick_params(colors=GRAY, labelsize=7)
    ax_att.legend(loc='upper right', fontsize=7, framealpha=0.3)
    # Horizontal Velocity
    vn = min(n, len(vx_data), len(vy_data))
    ax_vel.cla(); ax_vy.cla()
    ax_vel.set_facecolor('#161b22')
    ax_vel.grid(True, color='#21262d', linewidth=0.6, linestyle='--')
    ax_vel.set_title("Horizontal Velocity", color='white', fontsize=9, pad=4)
    ax_vel.set_ylim(-15, 15)
    ax_vy.set_ylim(-15, 15)
    if vn > 1:
        ax_vel.plot(t[:vn], vx_data[:vn], color=ACCENT, lw=1, label='Vx')
        ax_vy.plot(t[:vn], vy_data[:vn], color=YELLOW, lw=1, alpha=0.8, label='Vy')
    ax_vel.set_ylabel('Vx (m/s)', color=ACCENT, fontsize=8)
    ax_vy.set_ylabel('Vy (m/s)', color=YELLOW, fontsize=8)
    ax_vel.tick_params(colors=GRAY, labelsize=7)
    ax_vy.tick_params(colors=YELLOW, labelsize=7)
    h1, l1 = ax_vel.get_legend_handles_labels()
    h2, l2 = ax_vy.get_legend_handles_labels()
    ax_vel.legend(h1+h2, l1+l2, loc='upper right', fontsize=7, framealpha=0.3)
    # Safety Metrics
    gn = min(n, len(g_f), len(dist))
    ax_safe.cla()
    ax_safe.set_facecolor('#161b22')
    ax_safe.grid(True, color='#21262d', linewidth=0.6, linestyle='--')
    ax_safe.set_title("Safety: G-Force & Distance", color='white', fontsize=9, pad=4)
    ax_safe2 = ax_safe.twinx()
    ax_safe.set_ylim(0, 3)
    ax_safe2.set_ylim(0, 2000)
    if gn > 1:
        ax_safe.plot(t[:gn], g_f[:gn], color=PURPLE, lw=1, label='G-Force')
        ax_safe.axhline(2.0, color=RED, linestyle=':', lw=0.8, alpha=0.5)
        ax_safe2.plot(t[:gn], dist[:gn], color=YELLOW, lw=1, alpha=0.7, label='Dist (m)')
    ax_safe.set_ylabel('G', color=PURPLE, fontsize=8)
    ax_safe2.set_ylabel('Dist (m)', color=YELLOW, fontsize=8)
    ax_safe.tick_params(colors=GRAY, labelsize=7)
    ax_safe2.tick_params(colors=YELLOW, labelsize=7)
    h1, l1 = ax_safe.get_legend_handles_labels()
    h2, l2 = ax_safe2.get_legend_handles_labels()
    ax_safe.legend(h1+h2, l1+l2, loc='upper right', fontsize=7, framealpha=0.3)
    # Control Actions
    tn = min(n, len(thr), len(rcs))
    ax_act.cla(); ax_rcs.cla()
    ax_act.set_facecolor('#161b22')
    ax_act.grid(True, color='#21262d', linewidth=0.6, linestyle='--')
    ax_act.set_title("Control Actions", color='white', fontsize=9, pad=4)
    ax_act.set_ylim(0, 1)
    ax_rcs.set_ylim(0, 1)
    if tn > 1:
        ax_act.plot(t[:tn], thr[:tn], color='white', lw=1, label='Main Thrust')
        ax_rcs.plot(t[:tn], rcs[:tn], color=YELLOW, lw=0.8, alpha=0.6, label='Mean RCS')
    ax_act.set_ylabel('Thrust', color=GRAY, fontsize=8)
    ax_rcs.set_ylabel('RCS', color=YELLOW, fontsize=8)
    ax_act.tick_params(colors=GRAY, labelsize=7)
    ax_rcs.tick_params(colors=YELLOW, labelsize=7)
    h1, l1 = ax_act.get_legend_handles_labels()
    h2, l2 = ax_rcs.get_legend_handles_labels()
    ax_act.legend(h1+h2, l1+l2, loc='upper right', fontsize=7, framealpha=0.3)
    # Performance
    pn = min(n, len(cumr), len(st_id))
    ax_perf.cla(); ax_cum.cla()
    ax_perf.set_facecolor('#161b22')
    ax_perf.grid(True, color='#21262d', linewidth=0.6, linestyle='--')
    ax_perf.set_title("Mission Performance", color='white', fontsize=9, pad=4)
    ax_perf.set_ylim(-300, 300)
    if pn > 1:
        ax_perf.plot(t[:pn], cumr[:pn], color=GREEN, lw=1.2, label='Cum. Reward')
        ax_cum.plot(t[:pn], st_id[:pn], color=GRAY, lw=0.6, alpha=0.5, label='State ID')
    ax_perf.set_ylabel('Reward', color=GREEN, fontsize=8)
    ax_cum.set_ylabel('State', color=GRAY, fontsize=8)
    ax_perf.tick_params(colors=GRAY, labelsize=7)
    ax_cum.tick_params(colors=GRAY, labelsize=7)
    h1, l1 = ax_perf.get_legend_handles_labels()
    h2, l2 = ax_cum.get_legend_handles_labels()
    ax_perf.legend(h1+h2, l1+l2, loc='upper left', fontsize=7, framealpha=0.3)
    # Update title
    fig.suptitle(
        f"LunarLander3D — Live Telemetry  |  Episode {_ep[0]} ({_label[0]})  |  Step {n}",
        fontsize=11, color='white', fontweight='bold'
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LunarLander3D Live Dashboard')
    parser.add_argument('--no-osc', action='store_true', help='Run without OSC server')
    args = parser.parse_args()
    if not args.no_osc:
        t = threading.Thread(target=_start_osc_server, daemon=True)
        t.start()
    manager = plt.get_current_fig_manager()
    try:
        manager.window.wm_geometry(f"{WIN_W}x{WIN_H}+{WIN_X}+{WIN_Y}")
    except Exception:
        pass
    ani = animation.FuncAnimation(fig, _update, interval=200, blit=False, cache_frame_data=False)
    plt.show()
