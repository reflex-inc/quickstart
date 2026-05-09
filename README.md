# Reflex Labs Quickstart

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Reflex API](https://img.shields.io/badge/API-tryreflex.ai-orange.svg)](https://app.tryreflex.ai)

End-to-end demo of the Reflex Labs robotics fine-tune + inference API.
**Submit a real LoRA fine-tune, run real inference, drive an SO-101 arm
from typed prompts** — all in one Python file, against the live API,
with no mocks.

```bash
pip install reflex-sdk
export REFLEX_API_KEY="rfx_..."
python quickstart.py
```

That's it. Read on for what each section does and why.

---

## What this proves

| Section | What it does | What it proves |
|---|---|---|
| **§1 Training** | Submits a LoRA fine-tune of pi0.5 on `lerobot/aloha_sim_transfer_cube_human` (50 episodes, public HF dataset) and polls real progress | The Reflex SDK authenticates, provisions a B200 GPU, runs real training, returns a tracked `run_id` with status updates and final loss |
| **§2 Inference** | Calls the Reflex inference API with a synthetic observation and prints the action chunk | Your API key works, the inference endpoint returns real 50-step × 14-DOF pi0.5 action chunks in <2s (warm) |
| **§3 Closed loop** | Camera + SO-101 arm: every typed prompt becomes an observation → inference → arm motion | The full deployment pipeline (camera → state → API → action → servos) works end-to-end |

The script is **resilient to cold-starts** (auto-retries on HTTP 408) and
ships safety clipping on every joint motion (`±200` raw step delta from
current — bigger arm moves require explicit opt-in via `REFLEX_MAX_DELTA`).

---

## Prerequisites

| Required | How to get it |
|---|---|
| Python 3.12+ | `pyenv install 3.12` or your distro |
| Reflex API key | Sign up at [app.tryreflex.ai](https://app.tryreflex.ai), then **Settings → API Keys → Mint** |
| $5+ org balance | Go to **Billing**, redeem `REFLEX_100X` for $100 free credit, or top up via Stripe |

| Optional (for §3 — arm control) | Notes |
|---|---|
| HuggingFace SO-101 arm | $100 hobby kit ([huggingface.co/lerobot/so101](https://huggingface.co/lerobot/so101)) |
| USB webcam | Any V4L2-compatible cam |
| Linux + `/dev/ttyACM0` access | `sudo` or a `dialout` udev rule |

---

## Install

We recommend [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python venvs:

```bash
git clone https://github.com/reflex-inc/quickstart.git reflex-quickstart
cd reflex-quickstart
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Or with plain pip:

```bash
git clone https://github.com/reflex-inc/quickstart.git reflex-quickstart
cd reflex-quickstart
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs `reflex-sdk` (the official Python SDK) plus arm/camera dependencies (`lerobot`, `feetech-servo-sdk`, `opencv-python`).

---

## Run

### Set your API key (one-time)

Either env var:

```bash
export REFLEX_API_KEY="rfx_your_key_here"
```

Or save to a file the script reads:

```bash
mkdir -p ~/.reflex
echo -n "rfx_your_key_here" > ~/.reflex/api_key
chmod 600 ~/.reflex/api_key
```

### Full quickstart (training + inference + arm)

```bash
sudo -E python quickstart.py
```

(`sudo` only required for `/dev/ttyACM0` access — set up a udev rule to skip it.)

### Skip flags for partial runs

```bash
SKIP_ARM=1 python quickstart.py                 # no arm hardware needed
SKIP_ARM=1 SKIP_TRAINING=1 python quickstart.py # API smoke test only
```

### Single-prompt mode (CI / scripted)

```bash
sudo python quickstart.py --prompt "pick up the red cube"
```

---

## Expected output

```
━━━ REFLEX QUICKSTART ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ SECTION 1 — Submit a real LoRA fine-tune via Reflex SDK ━━━━━
  reflex-sdk version  : 0.1.4
  api_key             : rfx_abcdef0123…
  base model          : pi0.5  (flow-matching VLA, 3.4B params)
  training dataset    : lerobot/aloha_sim_transfer_cube_human
                        50 episodes of bimanual cube-transfer demos
                        14-DOF ALOHA action space, public on HuggingFace
  hardware            : managed B200 GPU on Reflex servers

  → submitting fine-tune to Reflex servers
  ✓ submitted in 943ms
    run_id            : m579ye2zbk3f2fr6bpfbsakes186djqf
    status            : queued → provisioning → running → succeeded
    dashboard         : https://app.tryreflex.ai/training-jobs/m579ye2…

  → polling client.training.get(run_id) every 5s for up to 30s...
    t+  5s  status=running   progress=  0.0%  steps=0  modal_spawn=fc-01KR…
    t+ 10s  status=running   progress=  0.0%  steps=0
    ...

━━━ SECTION 2 — Real inference call against the Reflex API ━━━━━━
  → calling Reflex inference API
  ✓ response in 1065ms
    pi0.5 inference    : 480ms (server-side compute)
    total round-trip   : 1062ms
    model              : lerobot/pi05_base
    action chunk shape : 10 steps × 50 DOF
    max action delta   : 0.9741

━━━ SECTION 3 — Live closed-loop chat with SO-101 ━━━━━━━━━━━━━━━
  → opening camera /dev/video0
  → opening arm at /dev/ttyACM0
  start positions : {'shoulder_pan': 2081, ...}

  READY.  type a prompt and Enter; Ctrl-D or 'quit' to exit.

> pick up the red cube
  → calling Reflex inference (state14, 1 frame x 3 cams, 41712 chars b64)
  ✓ 1506ms — 50-step chunk
    first action[0..5]: [-0.0043, -0.0028, 0.0138, ...]
```

---

## Architecture (what each section actually does)

### §1 Training

```
your machine
    │  client.training.lora_finetune(hf_source_uri="...", epochs=1)
    ▼
Reflex SDK (PyPI: reflex-sdk)
    │  authenticates with your API key
    │  calls publicApi:createAndProvisionTrainingRunFromHuggingFace
    ▼
Reflex servers
    │  validate key, create trainingRun row, provision GPU
    ▼
Managed B200 GPU
    │  download HF dataset
    │  load pi0.5 base weights
    │  run LoRA gradient steps
    │  save adapter
    ▼
Returned to you: run_id, status, modal_spawn_id
```

Background polling (`client.training.get(run_id)` every 5s) shows real
status transitions: `queued → provisioning → running → succeeded`. The
final response includes `modalInitialLoss` and `modalFinalLoss` — that's
your proof the fine-tune actually learned something.

### §2 Inference

```
POST  /v1/infer
Auth: Bearer rfx_...
{
  "observation": {
    "prompt": "...",
    "state": [14 floats],         # ALOHA joint positions in radians
    "images": {
      "cam_high":        {"encoding":"jpeg_base64","data":"..."},
      "cam_left_wrist":  {"encoding":"jpeg_base64","data":"..."},
      "cam_right_wrist": {"encoding":"jpeg_base64","data":"..."}
    }
  }
}
```

Returns:

```json
{
  "ok": true,
  "actions_aloha": [[14 floats], [14 floats], ...],   // 50 steps
  "actions_pi":    [[32 floats], ...],                // raw pi0.5 output
  "infer_ms": 426.5,
  "total_ms": 429.2,
  "model_id": "lerobot/pi05_base",
  "session_id": "..."
}
```

The script auto-retries on HTTP 408 (cold-start: first inference after
idle takes 30–60s while a container spins up; subsequent calls are <1s).

### §3 Closed loop on SO-101

```
loop:
    cur_raw  = read 6 joint positions via Feetech RS485
    state14  = pad raw → radians → 14-DOF ALOHA shape
    jpeg     = capture frame, JPEG-encode
    actions  = POST /v1/infer { prompt, state14, jpeg×3 }
    for i in first N steps:
        target = clip(predicted_radians_to_raw, ±MAX_DELTA from current)
        bus.sync_write("Goal_Position", target)  # one packet, all 6 motors
        sleep 1/RATE_HZ
```

`sync_write` sends all 6 motor goal positions in a single Feetech
protocol packet — critical for staying within RS485 bandwidth at 25 Hz.

---

## Configuration

Environment variables, all optional except `REFLEX_API_KEY`:

| Variable | Default | Purpose |
|---|---|---|
| `REFLEX_API_KEY` | (none — required) | Mint at app.tryreflex.ai |
| `REFLEX_CONVEX_URL` | (SDK default) | Override deployment endpoint |
| `SO101_PORT` | `/dev/ttyACM0` | Serial port for the arm |
| `CAMERA_INDEX` | `0` | V4L2 camera index |
| `REFLEX_MAX_DELTA` | `200` | Per-joint raw step cap (~17°) |
| `REFLEX_RATE_HZ` | `25` | Action playback rate |
| `REFLEX_STEPS_PER_CHUNK` | `10` | Steps of 50 to actually replay |
| `SKIP_ARM` | `0` | Skip §3 (no hardware needed) |
| `SKIP_TRAINING` | `0` | Skip §1 |

---

## What's intentionally *not* in this quickstart

- **Custom dataset training.** Use `--dataset` or set `DEFAULT_TRAINING_DATASET` to point at any public HF dataset.
- **Per-key adapter selection.** The inference endpoint currently serves the platform's active adapter. Per-key adapter switching is an upcoming feature — DM us if you want early access.
- **Bimanual control.** This quickstart drives a single 6-DOF SO-101. The model output is 14-DOF ALOHA (bimanual); we use the first 6 for the SO-101 and ignore the rest.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `HTTP 402 below_minimum_balance` | Your org has < $5 | Redeem `REFLEX_100X` at [app.tryreflex.ai/billing](https://app.tryreflex.ai/billing) |
| `HTTP 401 invalid_key` | Wrong/revoked key | Mint a new one in **Settings → API Keys** |
| `HTTP 408` (cold start) | First call after 10+ min idle | Script auto-retries — wait, it'll work |
| Arm doesn't move | `MAX_DELTA` too low or wrong control mode | `REFLEX_MAX_DELTA=400 REFLEX_STEPS_PER_CHUNK=50 ...` |
| `Permission denied: /dev/ttyACM0` | No serial access | Either `sudo` it or add yourself to the `dialout` group + udev rule |
| Camera black / wrong device | Wrong V4L2 index | Try `CAMERA_INDEX=1` (or 2, 3...); list with `v4l2-ctl --list-devices` |

---

## Where to go from here

- **API reference** — [docs.tryreflex.ai](https://docs.tryreflex.ai)
- **Bring your own dataset** — push a LeRobot-compatible dataset to HF Hub, then pass `hf_source_uri="<your-org>/<dataset>"`
- **Production deployment** — same SDK; just bump `epochs`, swap the dataset, adjust `learning_rate` via the `parameters` kwarg
- **Issues / questions** — [open a GitHub issue](https://github.com/reflex-inc/quickstart/issues) or [Discord](https://discord.gg/reflex)

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
