#!/usr/bin/env bash
# Process control for VDU using only kill(1) and integer signal semantics.
# No containers or virtualization are involved.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${ROOT_DIR}/snapshots/vdu.pid"
SNAP_DIR="${ROOT_DIR}/snapshots"

start() {
  local tier="${1:-coarse}"
  mkdir -p "${SNAP_DIR}"
  cd "${ROOT_DIR}"
  nohup python3 -m vdu run \
      --samples samples \
      --config configs/calibration.json \
      --snapshots snapshots \
      --tier "${tier}" > snapshots/run.log 2>&1 &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  cd - >/dev/null
  echo "started pid=${pid} tier=${tier}"
}

signal_pid() {
  local sig="$1"
  if [[ ! -f "${PID_FILE}" ]]; then
    echo "pid file ${PID_FILE} not found" >&2
    exit 2
  fi
  local pid
  pid="$(cat "${PID_FILE}")"
  kill "-${sig}" "${pid}"
  echo "sent SIG${sig} to pid=${pid}"
}

await_exit() {
  local pid
  pid="$(cat "${PID_FILE}")"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 1
  done
  echo "pid=${pid} exited"
}

await_running() {
  local pid
  pid="$(cat "${PID_FILE}")"
  local deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    if kill -0 "${pid}" 2>/dev/null; then
      sleep 0.3
      return 0
    fi
    sleep 0.1
  done
  echo "process never became runnable" >&2
  return 1
}

await_partial() {
  local state="${SNAP_DIR}/current/state.json"
  local deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    if [[ -f "${state}" ]] && python3 -c "import json,sys; sys.exit(0 if not json.load(open('${state}')).get('complete') else 1)" 2>/dev/null; then
      return 0
    fi
    sleep 0.1
  done
  echo "partial snapshot did not appear in time" >&2
  return 1
}

resume() {
  start "${1:-coarse}"
  await_exit
  echo "resumed; state at snapshots/current/state.json"
}

case "${1:-}" in
  start)   start "${2:-coarse}" ;;
  term)    await_partial && signal_pid TERM ;;
  kill)    await_partial && signal_pid KILL ;;
  wait)    await_exit ;;
  resume)  resume "${2:-coarse}" ;;
  *)
    echo "usage: $0 {start|term|kill|wait|resume} [tier]" >&2
    exit 1
    ;;
esac
