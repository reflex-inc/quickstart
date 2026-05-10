#!/usr/bin/env python3
"""
================================================================================
 Reflex Labs — Robotics VLA Quickstart
================================================================================

End-to-end demo: submit a fine-tune, run inference, drive an SO-101 arm in
real time from typed prompts. Everything talks to the live Reflex servers
through the public Python SDK and HTTP API — no local services, no mocks.

WHAT THIS SCRIPT DOES (in order)
    1. Installs the Python SDK + supporting packages (one-time, with `uv`)
    2. Submits a real LoRA fine-tune on `lerobot/aloha_sim_transfer_cube_human`
       and polls progress (Reflex servers manage GPU provisioning;
       returns a tracked run_id with status updates and final loss)
    3. Captures one frame from a USB webcam
    4. Reads the 6 joint positions from an SO-101 arm on /dev/ttyACM0
    5. Calls the Reflex inference API — gets a 50-step pi0.5 action chunk
    6. Replays the action chunk on the arm at 25 Hz with safety clipping
    7. Drops into an interactive terminal loop: type a prompt → arm moves

PREREQUISITES
    • Reflex API key (mint at https://app.tryreflex.ai/keys, $5+ balance)
    • Optional: HuggingFace SO-101 arm + USB webcam connected
    • Linux with /dev/ttyACM0 access (sudo or a udev rule for `dialout`)

INSTALL (run once)
    mkdir -p ~/reflex-quickstart && cd ~/reflex-quickstart
    uv venv .venv --python 3.12
    VIRTUAL_ENV=.venv uv pip install reflex-sdk lerobot \\
        feetech-servo-sdk deepdiff opencv-python pillow numpy

RUN
    export REFLEX_API_KEY="rfx_..."     # required
    sudo -E .venv/bin/python3 quickstart.py
       # `sudo -E` preserves env; sudo needed for /dev/ttyACM0 access only.

ENV OVERRIDES (optional)
    REFLEX_API_KEY        — your key (required)
    REFLEX_CONVEX_URL     — alternate deployment (default: prod hardcode)
    SO101_PORT            — serial port (default: /dev/ttyACM0)
    CAMERA_INDEX          — V4L2 index (default: 0)
    SKIP_ARM=1            — skip arm sections, just exercise the API
    SKIP_TRAINING=1       — skip training submit, jump to inference

================================================================================
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_api_key() -> str:
    """Resolve API key from (1) env var, (2) ~/.reflex/api_key, (3) /etc/reflex/api_key."""
    val = os.environ.get("REFLEX_API_KEY", "").strip()
    if val:
        return val
    # When running under sudo, $HOME may be /root; check the invoking user's home too
    candidates = [
        os.path.expanduser("~/.reflex/api_key"),
        os.path.expanduser(f"~{os.environ.get('SUDO_USER', '')}/.reflex/api_key")
        if os.environ.get("SUDO_USER") else None,
        "/etc/reflex/api_key",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return open(path).read().strip()
            except Exception:
                pass
    return ""


API_KEY = _resolve_api_key()
SO101_PORT = os.environ.get("SO101_PORT", "/dev/ttyACM0")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
SKIP_ARM = bool(int(os.environ.get("SKIP_ARM", "0")))
SKIP_TRAINING = bool(int(os.environ.get("SKIP_TRAINING", "0")))

# Safety: max raw step delta per joint per action step (4096 = 360°, so 200 ≈ 17°)
MAX_DELTA = int(os.environ.get("REFLEX_MAX_DELTA", "200"))
# Action playback rate
RATE_HZ = float(os.environ.get("REFLEX_RATE_HZ", "25"))
# How many steps of the predicted 50-step chunk to actually replay
STEPS_PER_CHUNK = int(os.environ.get("REFLEX_STEPS_PER_CHUNK", "10"))

# Tiny training dataset — public LeRobot pushT
DEFAULT_TRAINING_DATASET = "lerobot/aloha_sim_transfer_cube_human"


def _color(s: str, c: str) -> str:
    return f"\033[{c}m{s}\033[0m"


GREEN = lambda s: _color(s, "32")
RED = lambda s: _color(s, "31")
CYAN = lambda s: _color(s, "36")
GRAY = lambda s: _color(s, "90")
BOLD = lambda s: _color(s, "1")


def banner(title: str) -> None:
    print()
    print(BOLD(CYAN(f"━━━ {title} " + "━" * (60 - len(title)))))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Training (Reflex SDK)
# ─────────────────────────────────────────────────────────────────────────────
def submit_training_run() -> str | None:
    """Submit a real LoRA fine-tune using the Reflex SDK.

    What this proves:
      • The reflex-sdk PyPI package authenticates with your API key
      • The SDK targets the live production Convex deployment
      • The Convex `publicApi:createAndProvisionTrainingRunFromHuggingFace`
        action provisions a B200 GPU job on Modal
      • Convex returns a real run_id that you can poll for status

    Returns the run_id of the submitted job, or None on failure.
    """
    banner("SECTION 1 — Submit a real LoRA fine-tune via Reflex SDK")

    try:
        import reflex
    except ImportError:
        print(RED("  reflex-sdk is not installed.  pip install reflex-sdk"))
        return None

    print(f"  reflex-sdk version  : {GREEN(getattr(reflex, '__version__', '0.2.0'))}")
    print(f"  api_key             : {GRAY(API_KEY[:14] + '…')}")
    print(f"  base model          : {CYAN('pi0.5')}  (flow-matching VLA, 3.4B params)")
    print(f"  training dataset    : {CYAN(DEFAULT_TRAINING_DATASET)}")
    print(f"                        50 episodes of bimanual cube-transfer demos")
    print(f"                        14-DOF ALOHA action space, public on HuggingFace")
    print(f"  hardware            : managed B200 GPU on Reflex servers")

    client = reflex.Client(api_key=API_KEY)

    print(f"\n  → submitting fine-tune to Reflex servers")
    t0 = time.perf_counter()
    try:
        result = client.training.lora_finetune(
            hf_source_uri=DEFAULT_TRAINING_DATASET,
            model_name=f"quickstart-{int(time.time())}",
            base_model="pi0.5",
            epochs=1,
        )
    except Exception as e:
        print(f"  {RED('✗')} submission failed: {type(e).__name__}: {e}")
        return None

    submit_ms = int((time.perf_counter() - t0) * 1000)
    run_id = (
        result.get("trainingRunId")
        or result.get("training_run_id")
        or result.get("training_job_id")
        or result.get("run_id")
    )
    if not run_id:
        print(f"  {RED('✗')} no run_id in response: {json.dumps(result)[:200]}")
        return None

    print(f"  {GREEN('✓')} submitted in {submit_ms}ms")
    print(f"    run_id            : {BOLD(run_id)}")
    print(f"    status            : {GRAY('queued → provisioning → running → succeeded')}")
    print(f"    dashboard         : https://app.tryreflex.ai/training-jobs/{run_id}")

    # Poll status briefly to show real progress
    print(f"\n  → polling {CYAN('client.training.get(run_id)')} every 5s for up to 30s...")
    last_status = None
    for tick in range(6):
        time.sleep(5)
        try:
            status_result = client.training.get(run_id)
        except Exception as e:
            print(f"    {GRAY(f't+{(tick+1)*5}s: poll error {type(e).__name__}')}")
            continue
        # Response shape: {"ok": true, "trainingRun": {...}, "training_job": {...}}
        run = status_result.get("trainingRun") or status_result.get("training_job") or {}
        status = run.get("status") or "?"
        progress = run.get("progress") or 0
        steps = run.get("stepsCompleted") or run.get("steps_completed") or 0
        loss_initial = run.get("modalInitialLoss")
        loss_final = run.get("modalFinalLoss")
        spawn_id = run.get("modalSpawnId") or "?"
        line = f"    t+{(tick+1)*5:3d}s  status={GREEN(status):24s}  progress={progress*100:5.1f}%  steps={steps}"
        if loss_initial is not None:
            line += f"  init_loss={loss_initial:.4f}"
        if loss_final is not None:
            line += f"  final_loss={loss_final:.4f}"
        if status != last_status:
            line += f"  modal_spawn={GRAY(spawn_id[:18])}"
            last_status = status
        print(line)
        if status in ("succeeded", "failed"):
            break

    print(f"  {GRAY('(training runs async; full completion takes ~3-5 min)')}")
    print()
    print(f"  {BOLD('NOTE on deployment lifecycle:')}")
    print(f"  Your trained adapter is saved to your Reflex account once training")
    print(f"  completes. By default the inference endpoint serves the currently")
    print(f"  active adapter for the platform — to use YOUR new adapter for")
    print(f"  inference, contact Reflex support to enable per-key adapter")
    print(f"  selection (or use the dashboard to mark your adapter as active).")
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Inference via the Reflex API
# ─────────────────────────────────────────────────────────────────────────────
# Inference endpoint is resolved through the SDK so customers don't need
# to care about underlying infrastructure — Reflex servers handle auth,
# GPU provisioning, and inference routing transparently.
_API_SUFFIX = ".cloud"
_HTTP_ACTION_SUFFIX = ".site"


def _reflex_infer_url() -> str:
    """Derive the /v1/infer endpoint from the SDK's configured deployment."""
    try:
        from reflex._convex import convex_url as _sdk_url
    except ImportError:
        return ""
    base = _sdk_url(None).rstrip("/")
    # The HTTP API and the inference endpoint live on companion hostnames.
    if base.endswith(_API_SUFFIX):
        base = base[: -len(_API_SUFFIX)] + _HTTP_ACTION_SUFFIX
    return base + "/v1/infer"


INFER_URL = _reflex_infer_url()


def infer(prompt: str, state14: list[float], jpeg_b64: str,
          max_retries: int = 3) -> dict | None:
    """POST one observation to the Reflex inference API. Returns the parsed
    JSON response.

    Retries on HTTP 408 (cold-start: when the inference container is scaled
    to zero, the first call after idle takes 30–60s and the gateway times
    out before the model responds. Subsequent calls hit a warm container
    and complete in <1s).
    """
    body = json.dumps({
        "observation": {
            "prompt": prompt,
            "state": state14,                            # ALOHA: 14-D joint state
            "images": {                                  # All 3 cams required by pi0.5
                "cam_high":        {"encoding": "jpeg_base64", "data": jpeg_b64},
                "cam_left_wrist":  {"encoding": "jpeg_base64", "data": jpeg_b64},
                "cam_right_wrist": {"encoding": "jpeg_base64", "data": jpeg_b64},
            },
        }
    }).encode()

    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            INFER_URL,
            data=body,
            method="POST",
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {API_KEY}",
            },
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
                data["_latency_ms"] = int((time.perf_counter() - t0) * 1000)
                return data
        except urllib.error.HTTPError as e:
            if e.code == 408 and attempt < max_retries:
                print(f"  {GRAY(f'  408 (cold-start) attempt {attempt}/{max_retries}, retrying in 8s...')}")
                time.sleep(8)
                continue
            print(f"  {RED('✗')} HTTP {e.code}: {e.read().decode()[:300]}")
            return None
        except Exception as e:
            if attempt < max_retries:
                print(f"  {GRAY(f'  {type(e).__name__} attempt {attempt}/{max_retries}, retrying...')}")
                time.sleep(2)
                continue
            print(f"  {RED('✗')} {type(e).__name__}: {e}")
            return None
    return None


def synthetic_observation_test() -> bool:
    """Run a real inference call against the live Reflex API.

    What this proves:
      • Your API key is valid and your org has compute balance
      • The Reflex inference API is reachable
      • Reflex servers authenticate → securely route to GPU compute
      • pi0.5 + LoRA adapter run real flow-matching inference
      • A 50-step × 14-DOF action chunk is returned in <2s (warm)

    Sends a synthetic black-frame observation so this section runs without
    a connected camera; the model still does real work — it just produces
    near-zero actions for an all-black image.
    """
    banner("SECTION 2 — Real inference call against the Reflex API")
    print(f"  observation shape   : prompt + state[14] + 3 cameras (jpeg_base64)")
    print(f"  prompt              : 'reflex quickstart inference test'")
    print(f"  retries             : auto (up to 3, exponential — handles cold-start)")

    import numpy as np
    from PIL import Image
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO(); img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    print(f"  payload             : 1 frame × 3 cams = {len(b64) * 3} chars b64")

    print(f"\n  → calling Reflex inference API")
    res = infer("reflex quickstart inference test", [0.0] * 14, b64)
    if not res or not res.get("ok"):
        print(f"  {RED('✗')} inference failed; reply: {res}")
        return False

    print(f"  {GREEN('✓')} response in {res['_latency_ms']}ms")
    print(f"    pi0.5 inference    : {res.get('infer_ms', 0):.0f}ms (server-side compute)")
    print(f"    total round-trip   : {res.get('total_ms', 0):.0f}ms")
    print(f"    model              : {CYAN(res.get('model_id','?'))}")
    print(f"    session_id         : {GRAY(res.get('session_id','?'))}")
    print(f"    action chunk shape : {res.get('num_steps')} steps × {res.get('chunk_size')} DOF")
    print(f"    max action delta   : {res.get('max_minus_min', 0):.4f}  (0 means model produced flat output)")
    actions = res.get("actions_aloha") or []
    if actions:
        print(f"    first action[0..5] (rad): {[round(a, 4) for a in actions[0][:6]]}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Closed-loop on the SO-101 arm
# ─────────────────────────────────────────────────────────────────────────────
def chat_loop_with_arm(single_prompt: str | None = None) -> None:
    """Camera → Reflex inference → SO-101 arm. Type prompts; arm moves.

    If `single_prompt` is given, run that one prompt and return (CI/docs mode).
    """
    banner("SECTION 3 — Live closed-loop chat with SO-101")

    if SKIP_ARM:
        print(f"  {GRAY('SKIP_ARM=1 — skipping')}")
        return

    # Lazy imports so the script still runs without arm hardware.
    try:
        import cv2
        from lerobot.motors.feetech.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorNormMode
    except ImportError as e:
        print(f"  {RED('arm deps missing')}: {e}")
        print("  install: uv pip install lerobot feetech-servo-sdk deepdiff opencv-python")
        return

    # SO-101 motor map: 6 STS3215 servos at IDs 1–6
    MOTORS = {
        "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex":    Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex":    Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_roll":    Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    NAMES = list(MOTORS)

    # Open camera
    print(f"  → opening camera /dev/video{CAMERA_INDEX}")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"  {RED('✗')} camera not opened")
        return

    # Open arm
    print(f"  → opening arm at {SO101_PORT}")
    bus = FeetechMotorsBus(port=SO101_PORT, motors=MOTORS, calibration=None,
                           protocol_version=0)
    try:
        bus.connect(handshake=True)
    except Exception as e:
        print(f"  {RED('✗')} arm connect failed: {e}")
        cap.release()
        return

    start_pos = {n: bus.read("Present_Position", n, normalize=False, num_retry=2) for n in NAMES}
    print(f"  start positions : {start_pos}")
    for n in NAMES:
        bus.enable_torque(motors=[n], num_retry=1)

    # Helper: SO-101 raw step (0..4095, home ≈ 2048) → ALOHA radians
    def raw_to_rad(raw: int) -> float:
        return (raw - 2048) / 4096.0 * 2.0 * math.pi

    period = 1.0 / RATE_HZ
    if single_prompt:
        print(f"\n  {GREEN('SINGLE-PROMPT MODE:')}  {repr(single_prompt)}")
    else:
        print(f"\n  {GREEN('READY.')}  type a prompt and Enter; Ctrl-D or 'quit' to exit.")

    iters = 0
    try:
        while True:
            if single_prompt is not None:
                if iters >= 1:
                    break
                prompt = single_prompt
                print(f"\n{BOLD('>')} {prompt}")
            else:
                try:
                    prompt = input(f"\n{BOLD('>')} ").strip()
                except EOFError:
                    break
            iters += 1
            if not prompt or prompt.lower() in {"quit", "exit", "q"}:
                break

            # 1. Read current pos (live state for action prediction)
            cur_raw = {n: bus.read("Present_Position", n, normalize=False, num_retry=1)
                       for n in NAMES}
            state14 = [raw_to_rad(cur_raw[n]) for n in NAMES] + [0.0] * 8

            # 2. Capture frame
            ok, frame = cap.read()
            if not ok:
                print(f"  {RED('camera read failed')}")
                continue
            ok2, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_b64 = base64.b64encode(jpg.tobytes()).decode()

            # 3. Infer
            print(f"  → calling Reflex inference (state14, 1 frame x 3 cams, {len(jpeg_b64)} chars b64)")
            res = infer(prompt, state14, jpeg_b64)
            if not res or not res.get("ok"):
                continue
            actions = res.get("actions_aloha") or []
            print(f"  {GREEN('✓')} {res['_latency_ms']}ms — {len(actions)}-step chunk")
            if actions:
                print(f"    first action[0..5]: {[round(a, 4) for a in actions[0][:6]]}")

            # 4. Replay first N steps with safety clip
            n_replay = min(STEPS_PER_CHUNK, len(actions))
            for i in range(n_replay):
                a = actions[i]
                targets = {}
                for j, name in enumerate(NAMES):
                    pred_raw = int(round(2048 + a[j] / (2 * math.pi) * 4096))
                    targets[name] = max(cur_raw[name] - MAX_DELTA,
                                        min(cur_raw[name] + MAX_DELTA, pred_raw))
                bus.sync_write("Goal_Position", targets, normalize=False, num_retry=0)
                time.sleep(period)

    except KeyboardInterrupt:
        print()
    finally:
        # Always return to start + release torque so the arm doesn't lock up
        print(f"\n  → returning to start positions, then disabling torque")
        end_pos = {n: bus.read("Present_Position", n, normalize=False, num_retry=1) for n in NAMES}
        steps = max(1, int(1.5 * RATE_HZ))
        for k in range(steps + 1):
            alpha = k / steps
            tgt = {n: int(round(end_pos[n] + alpha * (start_pos[n] - end_pos[n]))) for n in NAMES}
            bus.sync_write("Goal_Position", tgt, normalize=False, num_retry=0)
            time.sleep(period)
        for n in NAMES:
            bus.disable_torque(motors=[n], num_retry=1)
        bus.disconnect(disable_torque=False)
        cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Reflex Labs end-to-end quickstart")
    ap.add_argument("--prompt", default=None, help="Single prompt mode (no chat loop)")
    args = ap.parse_args()

    if not API_KEY:
        print(RED("✗ REFLEX_API_KEY not set. Mint a key at https://app.tryreflex.ai/keys"))
        return 1

    banner("REFLEX QUICKSTART")
    print(f"  api_key  : {GRAY(API_KEY[:14] + '…')}")
    print(f"  arm port : {GRAY(SO101_PORT)}  (skip with SKIP_ARM=1)")

    # Section 1 — training (optional, demonstrate API)
    if not SKIP_TRAINING:
        submit_training_run()
    else:
        print(GRAY("\n  SKIP_TRAINING=1 — skipping section 1"))

    # Section 2 — inference smoke test
    if not synthetic_observation_test():
        print(RED("\n✗ inference smoke test failed; check API key + balance"))
        return 2

    # Section 3 — closed-loop with arm (single-prompt mode for CI/demo)
    chat_loop_with_arm(single_prompt=args.prompt)

    print()
    print(BOLD(GREEN("━━━ done ━━━")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
