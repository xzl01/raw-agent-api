#!/usr/bin/env bash
# =============================================================================
# start.sh — One-click local launcher for RAW Agent API
#
# Usage:
#   chmod +x start.sh
#   ./start.sh
#
# Options (environment variables):
#   HOST        Bind host          (default: 127.0.0.1)
#   PORT        Bind port          (default: 8000)
#   OPENAI_API_KEY  Your OpenAI key (required for LLM agent to call GPT)
# =============================================================================
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# ---- colors -----------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }

# ---- banner -----------------------------------------------------------------
echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║        📷  RAW Agent API             ║"
echo "  ║   LLM-guided Sony ARW processor      ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${RESET}"

# ---- Python check -----------------------------------------------------------
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Please install Python 3.9 or later."
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
    error "Python 3.9+ required (found ${PY_VER})."
    exit 1
fi
info "Python ${PY_VER} detected."

# ---- Virtual environment ----------------------------------------------------
if [ ! -d "${VENV_DIR}" ]; then
    info "Creating virtual environment at .venv …"
    python3 -m venv "${VENV_DIR}"
    success "Virtual environment created."
fi

# Activate venv
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---- Dependencies -----------------------------------------------------------
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
STAMP_FILE="${VENV_DIR}/.install_stamp"

# Re-install only if requirements.txt is newer than the stamp
if [ ! -f "${STAMP_FILE}" ] || [ "${REQ_FILE}" -nt "${STAMP_FILE}" ]; then
    info "Installing / updating dependencies …"
    pip install --quiet --upgrade pip
    pip install --quiet -r "${REQ_FILE}"
    touch "${STAMP_FILE}"
    success "Dependencies ready."
else
    info "Dependencies already up to date."
fi

# ---- OpenAI key warning -----------------------------------------------------
if [ -z "${OPENAI_API_KEY:-}" ]; then
    warn "OPENAI_API_KEY is not set."
    warn "The LLM agent will fail when processing real RAW files."
    warn "Set it with:  export OPENAI_API_KEY=sk-..."
    echo
fi

# ---- Start server -----------------------------------------------------------
info "Starting API server on http://${HOST}:${PORT} …"
echo
echo -e "  ${BOLD}Frontend (UI):${RESET}  ${GREEN}http://${HOST}:${PORT}/${RESET}"
echo -e "  ${BOLD}API docs:${RESET}       ${GREEN}http://${HOST}:${PORT}/docs${RESET}"
echo
echo -e "  Press ${BOLD}Ctrl+C${RESET} to stop."
echo

# Open browser after a short delay (best-effort, ignores failure)
if command -v xdg-open &>/dev/null; then
    (sleep 2 && xdg-open "http://${HOST}:${PORT}/") &>/dev/null &
elif command -v open &>/dev/null; then
    (sleep 2 && open "http://${HOST}:${PORT}/") &>/dev/null &
fi

cd "${SCRIPT_DIR}"
exec uvicorn main:app --host "${HOST}" --port "${PORT}" --reload
