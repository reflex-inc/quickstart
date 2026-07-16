#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Reflex Labs — Bimanual YAM setup helper
#
# Gets a fresh Linux machine from nothing → ready to run closed-loop inference
# on two YAM arms. The inference model runs in the Reflex cloud (B200); this
# machine just needs the client + the robot driver + cameras.
#
# USAGE:
#   ./setup_yam.sh install     # install SDK + robot driver + RealSense
#   ./setup_yam.sh can         # bring up the CAN interfaces (needs sudo)
#   ./setup_yam.sh cameras     # list connected RealSense serials (for the yaml)
#   ./setup_yam.sh check       # verify everything is ready
#   ./setup_yam.sh all         # install + can + cameras + check
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Pin the audited-stable SDK. Bump deliberately after verifying on hardware.
REFLEX_SDK_VERSION="${REFLEX_SDK_VERSION:-0.6.6}"
CAN_BITRATE="${CAN_BITRATE:-1000000}"
CAN_INTERFACES="${CAN_INTERFACES:-can0 can1}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# Pick a SUPPORTED interpreter. i2rt's pinned dm-env==1.6 pulls dm-tree, which
# only has usable wheels for CPython 3.10–3.12 — on 3.13/3.14 it tries to
# cmake-build from source and fails. HAL runs 3.10. Honor $PYTHON if set,
# else probe 3.12 → 3.11 → 3.10, else fall back to python3 with a version gate.
pick_python() {
  if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
  for cand in python3.12 python3.11 python3.10; do
    command -v "$cand" >/dev/null 2>&1 && { echo "$cand"; return; }
  done
  echo "python3"
}
PY="$(pick_python)"

require_supported_python() {
  "$PY" - <<'PYEOF' || die "Unsupported Python. Install 3.10–3.12 (e.g. 'pyenv install 3.12') and re-run with PYTHON=python3.12 ./setup_yam.sh ..."
import sys
maj, minor = sys.version_info[:2]
ok = maj == 3 and 10 <= minor <= 12
print(f"  using {sys.executable} (Python {maj}.{minor}) — {'supported' if ok else 'UNSUPPORTED'}")
sys.exit(0 if ok else 1)
PYEOF
}

install() {
  bold "Python interpreter"
  require_supported_python

  bold "Installing Reflex SDK (stable v${REFLEX_SDK_VERSION}) + WebRTC extras"
  # [webrtc] pulls aiortc/av/msgpack/numpy/Pillow — required for target.kind=webrtc
  "$PY" -m pip install --upgrade "reflex-sdk[webrtc]==${REFLEX_SDK_VERSION}"
  ok "reflex-sdk[webrtc]==${REFLEX_SDK_VERSION}"

  bold "Installing the i2rt YAM robot driver"
  if ! "$PY" -c "import i2rt" 2>/dev/null; then
    "$PY" -m pip install "i2rt @ git+https://github.com/i2rt-robotics/i2rt.git" \
      || warn "i2rt pip install failed — clone https://github.com/i2rt-robotics/i2rt and 'pip install -e .' manually"
  fi
  "$PY" -c "import i2rt" 2>/dev/null && ok "i2rt importable" || warn "i2rt not importable yet"

  bold "Installing RealSense Python bindings (skip if you use USB webcams)"
  "$PY" -m pip install pyrealsense2 || warn "pyrealsense2 failed — only needed for kind: realsense cameras"
  ok "install step done"
}

can_up() {
  bold "Bringing up CAN interfaces: ${CAN_INTERFACES} @ ${CAN_BITRATE} bps"
  for iface in $CAN_INTERFACES; do
    if ip link show "$iface" >/dev/null 2>&1; then
      sudo ip link set "$iface" down 2>/dev/null || true
      sudo ip link set "$iface" type can bitrate "$CAN_BITRATE"
      sudo ip link set "$iface" up
      ok "$iface up @ ${CAN_BITRATE}"
    else
      warn "$iface not present — check your CAN adapter / USB connection"
    fi
  done
  ip -br link | grep -E "can[0-9]" || warn "no can interfaces visible"
}

cameras() {
  bold "Connected RealSense devices (copy serials into yam_bimanual.yaml)"
  if command -v rs-enumerate-devices >/dev/null 2>&1; then
    rs-enumerate-devices -s 2>/dev/null || rs-enumerate-devices 2>/dev/null | grep -iE "serial|name" | head -20
  else
    "$PY" - <<'PYEOF' || warn "pyrealsense2 not installed — run ./setup_yam.sh install"
import pyrealsense2 as rs
ctx = rs.context()
devs = list(ctx.query_devices())
if not devs:
    print("  (no RealSense devices found — check USB)")
for d in devs:
    print(f"  {d.get_info(rs.camera_info.name):32s} serial={d.get_info(rs.camera_info.serial_number)}")
PYEOF
  fi
}

check() {
  bold "Readiness check"
  "$PY" -c "import sys; v=sys.version_info; ok=v[0]==3 and 10<=v[1]<=12; print(f'  Python {v[0]}.{v[1]}', '(supported)' if ok else '(UNSUPPORTED — use 3.10–3.12)')" 2>/dev/null || warn "no usable python"
  "$PY" -c "import reflex; print('  reflex-sdk', reflex.__version__)" 2>/dev/null || warn "reflex-sdk not installed — run ./setup_yam.sh install"
  "$PY" -c "import aiortc, av, msgpack, numpy, PIL; print('  webrtc extras OK')" 2>/dev/null || warn "webrtc extras missing — pip install 'reflex-sdk[webrtc]'"
  "$PY" -c "import i2rt; print('  i2rt OK')" 2>/dev/null || warn "i2rt missing"
  ip -br link | grep -qE "can[0-9].*UP" && ok "a CAN interface is UP" || warn "no CAN interface UP — run ./setup_yam.sh can"
  if [ -n "${REFLEX_API_KEY:-}" ]; then ok "REFLEX_API_KEY is set"; else warn "REFLEX_API_KEY not set — export it or run 'reflex login'"; fi
  echo
  bold "Next:"
  echo "  1. ./setup_yam.sh cameras      # get your 3 RealSense serials"
  echo "  2. edit yam_bimanual.yaml      # paste serials + set CAN channels + prompt"
  echo "  3. reflex connect --config yam_bimanual.yaml   # (start with mode: dry_run)"
}

case "${1:-all}" in
  install) install ;;
  can)     can_up ;;
  cameras) cameras ;;
  check)   check ;;
  all)     install; can_up; cameras; check ;;
  *) echo "usage: $0 {install|can|cameras|check|all}"; exit 1 ;;
esac
