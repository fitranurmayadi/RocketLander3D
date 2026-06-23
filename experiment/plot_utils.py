import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_timeseries_png(
    out_dir: str,
    prefix: str,
    series: Dict[str, List[float]],
    t: List[float],
) -> None:
    """Save one PNG per series key."""
    ensure_dir(out_dir)

    # Single figure but with multiple subplots (LunarLander3D-like report style)
    # Expected `series` keys (some may be missing depending on controller):
    # - alt_z_m, dist_h_m, vz_mps, roll_deg, pitch_deg, yaw_deg, reward_step

    fig = plt.figure(figsize=(16, 12))

    # 2x3 grid like the reference (we only use what we have)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    t = np.asarray(t, dtype=float)

    def _plot_if(ax, k, y, label=None):
        if y is None or len(y) == 0:
            return
        ax.plot(t, y, label=label or k)
        ax.grid(True, alpha=0.3)

    alt = series.get("alt_z_m")
    dist_h = series.get("dist_h_m")
    vz = series.get("vz_mps")
    roll = series.get("roll_deg")
    pitch = series.get("pitch_deg")
    yaw = series.get("yaw_deg")
    reward_step = series.get("reward_step")

    # Top-Left: Altitude + vz
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_if(ax1, "alt_z_m", alt, label="Altitude (m)")
    ax1.set_title("Vertical Profile")
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("Altitude (m)")

    if vz is not None and len(vz) > 0:
        ax1b = ax1.twinx()
        _plot_if(ax1b, "vz_mps", vz, label="Vz (m/s)")
        ax1b.set_ylabel("Vz (m/s)")

    # Top-Right: Dist_h
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_if(ax2, "dist_h_m", dist_h, label="Horizontal Dist (m)")
    ax2.set_title("Horizontal Distance")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("Dist_h (m)")

    # Middle-Left: Attitude (roll/pitch/yaw)
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_if(ax3, "roll_deg", roll, label="Roll (deg)")
    _plot_if(ax3, "pitch_deg", pitch, label="Pitch (deg)")
    _plot_if(ax3, "yaw_deg", yaw, label="Yaw (deg)")
    ax3.set_title("Attitude")
    ax3.set_xlabel("t (s)")
    ax3.set_ylabel("Degrees")
    if ax3.lines:
        ax3.legend(loc="best")

    # Middle-Right: Reward (step + cumulative)
    ax4 = fig.add_subplot(gs[1, 1])
    if reward_step is not None and len(reward_step) > 0:
        ax4.plot(t, reward_step, label="Reward (step)", color="tab:green")
        ax4.grid(True, alpha=0.3)
        ax4.set_title("Mission Performance")
        ax4.set_xlabel("t (s)")
        ax4.set_ylabel("Reward")

        ax4b = ax4.twinx()
        cum = np.cumsum(np.asarray(reward_step, dtype=float))
        ax4b.plot(t, cum, label="Cumulative Reward", color="tab:blue", alpha=0.6)
        ax4b.set_ylabel("Cum Reward")

        # legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        if lines1 or lines2:
            ax4.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # Bottom row: keep it as a duplicate view if limited series exist
    ax5 = fig.add_subplot(gs[2, 0])
    if vz is not None and len(vz) > 0:
        _plot_if(ax5, "vz_mps", vz, label="Vz (m/s)")
        ax5.set_title("Vertical Velocity")
        ax5.set_xlabel("t (s)")
        ax5.set_ylabel("Vz (m/s)")
    else:
        ax5.axis('off')

    ax6 = fig.add_subplot(gs[2, 1])
    if roll is not None and len(roll) > 0:
        # Simple safety proxy: |roll|+|pitch|
        safety = None
        if roll is not None and pitch is not None and len(pitch) == len(roll):
            safety = np.abs(np.asarray(roll)) + np.abs(np.asarray(pitch))
        if safety is not None:
            _plot_if(ax6, "safety", safety, label="|roll|+|pitch|")
            ax6.set_title("Safety Metric")
            ax6.set_xlabel("t (s)")
            ax6.set_ylabel("deg")
        else:
            ax6.axis('off')
    else:
        ax6.axis('off')

    fig.suptitle(prefix, fontsize=16)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(os.path.join(out_dir, f"{prefix}_report.png"), dpi=150)
    plt.close(fig)



