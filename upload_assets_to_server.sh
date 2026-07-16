#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-${SCRIPT_DIR}/server_paths.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  echo "Copy ${SCRIPT_DIR}/server_paths.env.example to ${SCRIPT_DIR}/server_paths.env and edit it." >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

SERVER_USER="${SERVER_USER:?SERVER_USER is required in server_paths.env}"
SERVER_HOST="${SERVER_HOST:?SERVER_HOST is required in server_paths.env}"
SERVER_PORT="${SERVER_PORT:-22}"
SSH_TARGET="${SERVER_USER}@${SERVER_HOST}"
RSYNC_SSH="ssh -p ${SERVER_PORT}"

LOCAL_DATASET_ROOT="${LOCAL_DATASET_ROOT:?LOCAL_DATASET_ROOT is required in server_paths.env}"
LOCAL_WHISPER_BASE_MODEL_DIR="${LOCAL_WHISPER_BASE_MODEL_DIR:?LOCAL_WHISPER_BASE_MODEL_DIR is required in server_paths.env}"
LOCAL_CHARSIU_SRC_DIR="${LOCAL_CHARSIU_SRC_DIR:-}"
LOCAL_ALIGNER_MODEL_DIR="${LOCAL_ALIGNER_MODEL_DIR:-}"

ssh -p "${SERVER_PORT}" "${SSH_TARGET}" "mkdir -p '${SERVER_DATA_ROOT}/speechocean762' '${SERVER_DATA_ROOT}/models' '${SERVER_DATA_ROOT}/src'"

rsync -avh --progress -e "${RSYNC_SSH}" \
  "${LOCAL_DATASET_ROOT}/" \
  "${SSH_TARGET}:${SERVER_DATA_ROOT}/speechocean762/speechocean762/"

rsync -avh --progress -e "${RSYNC_SSH}" \
  "${LOCAL_WHISPER_BASE_MODEL_DIR}/" \
  "${SSH_TARGET}:${WHISPER_BASE_MODEL_DIR}/"

if [[ -n "${LOCAL_ALIGNER_MODEL_DIR}" ]]; then
  ALIGNER_BASENAME="$(basename "${LOCAL_ALIGNER_MODEL_DIR}")"
  ssh -p "${SERVER_PORT}" "${SSH_TARGET}" "mkdir -p '${SERVER_DATA_ROOT}/models/${ALIGNER_BASENAME}'"
  rsync -avh --progress -e "${RSYNC_SSH}" \
    "${LOCAL_ALIGNER_MODEL_DIR}/" \
    "${SSH_TARGET}:${SERVER_DATA_ROOT}/models/${ALIGNER_BASENAME}/"
fi

if [[ -n "${LOCAL_CHARSIU_SRC_DIR}" ]]; then
  CHARSIU_SRC_BASENAME="$(basename "${LOCAL_CHARSIU_SRC_DIR}")"
  ssh -p "${SERVER_PORT}" "${SSH_TARGET}" "mkdir -p '${SERVER_DATA_ROOT}/src/${CHARSIU_SRC_BASENAME}'"
  rsync -avh --progress -e "${RSYNC_SSH}" \
    "${LOCAL_CHARSIU_SRC_DIR}/" \
    "${SSH_TARGET}:${SERVER_DATA_ROOT}/src/${CHARSIU_SRC_BASENAME}/"
fi

echo "Upload completed."
