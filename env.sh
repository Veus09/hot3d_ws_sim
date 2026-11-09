# source env.sh  — dùng lại venv Isaac Sim 6.0 + Isaac Lab 3.0b2 của đồ án
source "$HOME/fall26_envtest/.venv/bin/activate"
export HOT3D_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HOT3D_WS/src:${PYTHONPATH}"
