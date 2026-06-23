#!/usr/bin/env bash
# =============================================================================
# launch_mission.sh
# LunarLander3D — Smart Launcher
# =============================================================================
# Usage:
#   ./launch_mission.sh v1 [extra args...]
#   ./launch_mission.sh v2 --fixed
#   ./launch_mission.sh v3 --spawn -1000 -1000 1000 --orient 45 45 45
#   ./launch_mission.sh v1 --no-dashboard   # run without live dashboard
# =============================================================================

# Parse optional flag
NO_DASHBOARD=0
# Collect all args; support --no-dashboard anywhere
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--no-dashboard" ]]; then
    NO_DASHBOARD=1
  else
    ARGS+=("$arg")
  fi
done

# Reassign positional parameters without the flag
set -- "${ARGS[@]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MISSION_ID="${1:-v1}"
shift || true   # remove first arg, pass the rest to the mission

case "$MISSION_ID" in
    v1) MISSION_SCRIPT="mission_v1_classic.py"   ;;
    v2) MISSION_SCRIPT="mission_v2_direct.py"    ;;
    v3) MISSION_SCRIPT="mission_v3_trajectory.py";;
    *)  echo "Usage: $0 [v1|v2|v3] [--no-dashboard] [args...]"; exit 1 ;;
esac

echo "======================================"
echo "  LunarLander3D Launcher"
echo "  Mission : $MISSION_SCRIPT"
echo "  Args    : $*"
echo "======================================"

# 1. Start live dashboard in the background unless disabled
if [[ $NO_DASHBOARD -eq 0 ]]; then
  echo "[Launcher] Starting live_dashboard.py ..."
  python live_dashboard.py &
  DASHBOARD_PID=$!
  echo "[Launcher] Dashboard PID = $DASHBOARD_PID"
  # 2. Wait for dashboard process to be ready
  sleep 1.0
fi

# 4. Launch the mission script in background
echo "[Launcher] Starting mission: $MISSION_SCRIPT ..."
python "$MISSION_SCRIPT" "$@" &
MISSION_PID=$!

# No window positioning (xdotool removed for better portability)



# Wait for mission to finish
wait $MISSION_PID
EXIT_CODE=$?

echo ""
echo "[Launcher] Mission complete (exit $EXIT_CODE)."
if [[ $NO_DASHBOARD -eq 0 ]]; then
  echo "[Launcher] Dashboard is still running (PID $DASHBOARD_PID). Press Ctrl+C to quit."
  wait $DASHBOARD_PID
fi
