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

ENV_MANAGER="${ENV_MANAGER:-conda}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TMPDIR="${TMPDIR:-${SERVER_DATA_ROOT}/tmp}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SERVER_DATA_ROOT}/pip_cache}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${SERVER_DATA_ROOT}/conda_pkgs}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${REPO_DIR}/.conda_env}"

sudo apt-get update
sudo apt-get install -y git git-lfs rsync ffmpeg libsndfile1 python3 python3-venv python3-pip build-essential

mkdir -p "$(dirname "${REPO_DIR}")"
if [[ -d "${REPO_DIR}/.git" ]]; then
  git -C "${REPO_DIR}" fetch --all --prune
  git -C "${REPO_DIR}" pull --ff-only
elif [[ -d "${REPO_DIR}" ]] && [[ -n "$(ls -A "${REPO_DIR}" 2>/dev/null)" ]]; then
  echo "Repo directory already exists and is non-empty without .git; skipping clone and using local files in ${REPO_DIR}."
else
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

mkdir -p "${SERVER_DATA_ROOT}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}" "${TMPDIR}" "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"
export TMPDIR="${TMPDIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}"

if [[ "${ENV_MANAGER}" == "conda" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available but ENV_MANAGER=conda" >&2
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if [[ ! -d "${CONDA_ENV_PREFIX}" ]]; then
    conda create -y -p "${CONDA_ENV_PREFIX}" "python=${PYTHON_VERSION}"
  fi
  conda activate "${CONDA_ENV_PREFIX}"
else
  python3 -m venv "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel

if [[ -n "${TORCH_WHEEL_INDEX_URL:-}" ]]; then
  python -m pip install torch torchvision torchaudio --index-url "${TORCH_WHEEL_INDEX_URL}"
fi
python -m pip install -r "${REPO_DIR}/requirements.txt"

cat <<EOF
Server environment is ready.
Repo: ${REPO_DIR}
Env manager: ${ENV_MANAGER}
Conda env: ${CONDA_ENV_PREFIX}
Venv: ${VENV_DIR}
Data root: ${SERVER_DATA_ROOT}
TMPDIR: ${TMPDIR}
PIP_CACHE_DIR: ${PIP_CACHE_DIR}
CONDA_PKGS_DIRS: ${CONDA_PKGS_DIRS}
EOF
