# source env.sh — kích hoạt venv có Isaac Sim 6.0 + Isaac Lab 3.0b2, thêm src/ vào PYTHONPATH.
# Venv ở chỗ khác: HOT3D_VENV=/đường/dẫn/.venv source env.sh
source "${HOT3D_VENV:-$HOME/fall26_envtest/.venv}/bin/activate"
export HOT3D_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HOT3D_WS/src:${PYTHONPATH}"
