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

---

## MolmoAct2-BimanualYAM Inference (NEW)

The same Reflex SaaS auth + billing flow now powers **MolmoAct2-BimanualYAM**,
an 8B vision-language-action model from AI2 fine-tuned for bimanual YAM arms.
Same `rfx_*` API key, same per-second billing, different `baseModel`:

```bash
pip install requests numpy pillow
export REFLEX_API_KEY="rfx_..."
python3 quickstart_molmoact.py
```

This script verifies in under 1 second that:

| Section | What it proves |
|---|---|
| **§1 Authorize** | API key authenticates → Convex picks a live Modal worker (us-west B200) → signs a 30-min HMAC session token |
| **§2 Worker health** | Authenticated GET to the Modal worker — proves the worker is up + reachable |

If you want the **closed-loop bimanual YAM demo** (arms move from camera observations):

```bash
git clone https://github.com/reflex-inc/reflex
cd reflex/sdk/python && pip install -e .

# Connect to your YAM arms + 3 cameras via the cloud BASELINE worker
reflex connect --config ../../examples/yam_bimanual_molmoact2_BASELINE.yaml
```

**Pricing:** $10/hr × actual GPU-seconds (≈ $0.001 per 200ms inference call).
**Quality:** WebRTC + adaptive JPEG q=95 — visually lossless (PSNR 38.8 dB vs raw).
**Latency:** ~220 ms p50 RTT from residential WAN to us-west.

See [§MolmoAct2 architecture](#molmoact2-bimanualyam-architecture) below for the
worker setup, primeNode pool, and HMAC session-token verification details.


## Bring your own robot (custom arm)

The SDK ships connectors for a handful of arms, but any robot works — you don't
need a PR to the SDK. Implement a small Python class and point your config at it
by `module:Class` path; Reflex imports it at connect time.

```yaml
# in your connect config
hardware:
  kind: "custom_robot:XArm5Connector"   # module:Class — Reflex imports this
  config:                               # passed to __init__ as kwargs
    ip: 192.168.1.220
    hz: 25
```

The contract (`reflex.connectors.base.RobotConnector`) is five methods:

| Method | Does |
|---|---|
| `start()` | open drivers + cameras |
| `get_observation() -> Observation` | one observation per control step |
| `apply_action(ActionChunk)` | apply an action chunk to the hardware |
| `safe_stop(reason="")` | halt motion safely — implement this for real arms |
| `stop()` | release hardware |

`Observation(state: list[float], cameras: {name: image}, task: str)` and
`ActionChunk(actions)` where `actions` is `[T, action_dim]`. You map the model's
action dimension onto your robot's DOF inside `apply_action`.

A working **xArm5 skeleton** is in [`custom_robot.py`](custom_robot.py) — copy it,
fill in the commented driver lines, and run `python custom_robot.py` to confirm
the class resolves through the registry the way `reflex connect` will load it.

> Heads-up: a base model (e.g. pi0.5) will run inference on any arm, but it
> won't do good manipulation on an embodiment it hasn't seen. For real task
> performance on a new arm, plan on a fine-tune with your own data.


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

| Optional (for `quickstart_byom.py` — bring your own pi0.5) | Notes |
|---|---|
| A HuggingFace repo with your fine-tuned pi0.5 adapter (private or public) | LoRA `adapter_model.safetensors` + `adapter_config.json`, or a full / merged-LoRA checkpoint. Private repos require a `BYOM_HF_TOKEN` read-scope token. |

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

## Advanced training parameters (SDK 0.2.0+)

`client.training.lora_finetune()` accepts 8 advanced knobs beyond the
basics shown above. All are optional; omit to use server defaults. All
are server-side validated — out-of-bounds values are rejected before any
GPU is provisioned.

```python
result = client.training.lora_finetune(
    hf_source_uri="lerobot/aloha_sim_transfer_cube_human",
    epochs=1,

    # LoRA shape
    lora_rank=16,                                       # {4, 8, 16, 32, 64}
    lora_alpha=32,                                      # [1, 256]
    lora_dropout=0.05,                                  # [0.0, 0.5]
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],

    # Optimizer schedule
    warmup_steps=200,                                   # [0, max_steps/2]
    learning_rate=1e-4,
    batch_size=2,
    max_steps=500,

    # Compute / memory
    gradient_checkpointing=True,                        # cuts ~40% GPU mem
    dtype="bfloat16",                                   # {"bfloat16", "float32"}

    # VLA-specific
    freeze_vision_encoder=True,                         # default for transfer

    # Checkpointing cadence
    save_freq=500,                                      # [50, max_steps]
)
```

`target_modules` whitelist for pi0.5: `q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate_proj`, `up_proj`, `down_proj`, `action_in_proj`,
`action_out_proj`.

LoRA-only kwargs (`lora_rank`, `lora_alpha`, `lora_dropout`,
`target_modules`) are rejected on `full_finetune`.

---

## What training actually does

Customer training runs real LoRA fine-tuning of `pi0.5` on your
HuggingFace LeRobot dataset, executed via the `lerobot-train` CLI on a
managed B200 GPU. The adapter is saved to your Reflex account on
completion. Typical wall time for a small dataset:

- 5 steps:  ~65–110s
- 30 steps: ~90–120s
- 200 steps: ~3–5 min

Your `lora_rank`, `target_modules`, `learning_rate`, `batch_size`,
`max_steps`, `warmup_steps`, and the rest of the params listed above are
honored by the underlying training loop. You can verify by polling
`client.training.get(run_id)` — the `modalAdapterPath` field will
contain `real_runs/<run_id>_<timestamp>/checkpoints/<step>/pretrained_model/`.

---

## Bring Your Own pi0.5 Model

New in reflex-sdk **0.3.0**. Already fine-tuned a pi0.5 adapter
somewhere else — on your own GPUs, on a third-party platform, or in a
previous Reflex training run that's living in your HuggingFace
account? `quickstart_byom.py` demonstrates the BYO-model surface
end-to-end: import → poll until ready → bind to an API key → prove the
bound adapter actually serves `/v1/infer`.

```bash
export REFLEX_API_KEY="rfx_..."
export BYOM_HF_REPO="<your-org>/<your-pi05-lora-repo>"
export BYOM_HF_TOKEN="hf_..."          # optional — private HF repos only
python3 quickstart_byom.py
```

### Why use it

- **Skip Reflex training entirely** for adapters you've already
  produced. Customers who train internally (on-prem clusters, in
  another cloud, or on local workstations) get the same managed
  inference path as Reflex-trained adapters.
- **Keep your weights in HuggingFace.** The import is a one-time
  Reflex-side pull triggered by the SDK; you don't have to surrender
  ownership of the artifact or duplicate it into a Reflex-specific
  bucket.
- **Per-key adapter routing.** `client.keys.bind_model(key_id,
  model_id)` makes `/v1/infer` for `key_id` serve YOUR adapter from
  `/vol/customer_models/<org>/<modelId>/` instead of the platform
  default — verified via the `adapter_path` field on the inference
  response.

### Supported artifact types

| Kind | Detected via | Notes |
|---|---|---|
| **BF16 LoRA adapter** | `adapter_model.safetensors` + `adapter_config.json` | Recommended path. Smallest, fastest hot-swap (~60-90s the first time, near-zero after). |
| **Full fine-tune** | `model.safetensors*` shards + `config.json` with pi0.5 architecture | Works, but every swap pays the full reload cost. Use when you've actually unfrozen the base. |
| **Merged LoRA** | `model.safetensors*` + `config.json` with `_merged_from_lora: true` or `_name_or_path` containing `merged_lora` / `lora_merged` | Treated as full fine-tune at load time. |

Quantized (INT4 / AWQ) BYOM is **deferred to v1.5**; submit the
unquantized weights for now.

### The full BYOM script

```python
#!/usr/bin/env python3
"""quickstart_byom.py — import → bind → infer → cleanup."""
import base64, io, json, os, time, urllib.request
import reflex
from PIL import Image
import numpy as np

API_KEY  = os.environ["REFLEX_API_KEY"]
HF_REPO  = os.environ["BYOM_HF_REPO"]                       # "yourorg/my-pi05-lora"
HF_TOKEN = os.environ.get("BYOM_HF_TOKEN") or None          # for private repos
NAME     = os.environ.get("BYOM_MODEL_NAME", f"quickstart-byom-{int(time.time())}")

client = reflex.Client(api_key=API_KEY)

# Resolve our own key_id (needed for bind_model below) via publicApi:whoami.
from reflex._convex import convex_call
who = convex_call("query", "publicApi:whoami", {"apiKey": API_KEY})
key_id = who["keyId"]

# 1) Import the adapter and poll until it's ready on Reflex storage.
res = client.models.import_from_hf(HF_REPO, NAME, hf_token=HF_TOKEN)
model_id = res["modelId"]
t0 = time.perf_counter()
while True:
    artifact = client.models.get(model_id).get("artifact", {})
    if artifact.get("status") == "ready":
        print(f"model ready in {time.perf_counter() - t0:.1f}s")
        break
    if artifact.get("status") == "failed":
        raise SystemExit(f"prepare failed: {artifact.get('failureReason')}")
    time.sleep(5)

# 2) Bind this API key to the imported model.
client.keys.bind_model(key_id, model_id)

# 3) Call /v1/infer — adapter_path should now point at /vol/customer_models/...
img = Image.fromarray(np.zeros((224, 224, 3), dtype="uint8"))
buf = io.BytesIO(); img.save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()
body = json.dumps({"observation": {
    "prompt": "test",
    "state": [0.0] * 14,
    "images": {n: {"encoding": "jpeg_base64", "data": b64}
               for n in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
}}).encode()
req = urllib.request.Request(
    "https://kindly-bullfrog-494.convex.site/v1/infer",
    data=body, method="POST",
    headers={"content-type": "application/json",
             "authorization": f"Bearer {API_KEY}"},
)
with urllib.request.urlopen(req, timeout=300) as r:
    out = json.loads(r.read())
print("adapter_path:", out.get("adapter_path"))   # → /vol/customer_models/<org>/<modelId>/...

# 4) Cleanup: unbind and (optionally) delete.
client.keys.unbind_model(key_id)
client.models.delete(model_id)
```

The bundled `quickstart_byom.py` adds banner output, error
diagnostics, configurable polling, and `BYOM_KEEP=1` to skip the
unbind/delete steps if you want to leave the binding in place for
follow-up testing.

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
