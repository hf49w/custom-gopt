#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_FALLBACK="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${1:-${SCRIPT_DIR}/server_paths.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  echo "Copy ${SCRIPT_DIR}/server_paths.env.example to ${SCRIPT_DIR}/server_paths.env and edit it." >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

sudo apt-get update
sudo apt-get install -y git git-lfs rsync ffmpeg libsndfile1 python3 python3-venv python3-pip build-essential

mkdir -p "$(dirname "${REPO_DIR}")"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
else
  git -C "${REPO_DIR}" fetch --all --prune
  git -C "${REPO_DIR}" pull --ff-only
fi

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ -n "${TORCH_WHEEL_INDEX_URL:-}" ]]; then
  python -m pip install torch torchvision torchaudio --index-url "${TORCH_WHEEL_INDEX_URL}"
fi
python -m pip install -r "${REPO_DIR}/requirements.txt"

mkdir -p "${SERVER_DATA_ROOT}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}"

cat <<EOF
Server environment is ready.
Repo: ${REPO_DIR}
Venv: ${VENV_DIR}
Data root: ${SERVER_DATA_ROOT}
EOF
