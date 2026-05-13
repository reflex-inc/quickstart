#!/usr/bin/env python3
"""
================================================================================
 Reflex Labs — Bring Your Own pi0.5 Model Quickstart
================================================================================

End-to-end demo of the **BYO-model** flow shipped in reflex-sdk 0.3.0: import
your own fine-tuned pi0.5 LoRA adapter from a HuggingFace repo, bind it to an
API key, and prove that subsequent /v1/infer calls serve the customer
adapter (not the platform default). Everything talks to the live Reflex
servers through the public Python SDK and HTTP API — no local services,
no mocks.

WHAT THIS SCRIPT DOES (in order)
    1. `client.models.import_from_hf(...)` — kicks off a background pull of
       your adapter from a public-or-private HuggingFace repo into Reflex
       storage; poll `client.models.get(model_id)` until status="ready"
    2. `client.keys.bind_model(key_id, model_id)` — routes future inference
       on this API key through your adapter
    3. POST /v1/infer with a synthetic observation — the response's
       `adapter_path` field is verified to contain "customer_models/",
       proving the bound adapter actually served the request
    4. `client.keys.unbind_model(key_id)` — revert to the platform default
       (and optionally `client.models.delete(model_id)` to clean up)

PREREQUISITES
    • Reflex API key (mint at https://app.tryreflex.ai/keys, $5+ balance)
    • A HuggingFace repo with your fine-tuned pi0.5 adapter (private or
      public). For private repos, set BYOM_HF_TOKEN to a read-scope HF
      access token.

INSTALL (run once)
    mkdir -p ~/reflex-quickstart && cd ~/reflex-quickstart
    uv venv .venv --python 3.12
    VIRTUAL_ENV=.venv uv pip install reflex-sdk>=0.3.0

RUN
    export REFLEX_API_KEY="rfx_..."
    export BYOM_HF_REPO="<your-org>/<your-pi05-lora-repo>"
    export BYOM_HF_TOKEN="hf_..."           # optional, private repos only
    python3 quickstart_byom.py

ENV OVERRIDES (optional)
    REFLEX_API_KEY      — your key (required)
    REFLEX_CONVEX_URL   — alternate deployment (default: prod hardcode)
    BYOM_HF_REPO        — HF repo id, e.g. "yourorg/my-pi05-lora" (required)
    BYOM_HF_TOKEN       — HF access token for private repos (optional)
    BYOM_HF_REVISION    — pin to a specific revision/branch (optional)
    BYOM_MODEL_NAME     — display name (default: "quickstart-byom-<ts>")
    BYOM_KEEP           — set to 1 to skip unbind+delete cleanup at the end

================================================================================
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_api_key() -> str:
    val = os.environ.get("REFLEX_API_KEY", "").strip()
    if val:
        return val
    for path in (
        os.path.expanduser("~/.reflex/api_key"),
        "/etc/reflex/api_key",
    ):
        if os.path.exists(path):
            try:
                return open(path).read().strip()
            except Exception:
                pass
    return ""


API_KEY = _resolve_api_key()
HF_REPO = os.environ.get("BYOM_HF_REPO", "").strip()
HF_TOKEN = os.environ.get("BYOM_HF_TOKEN", "").strip() or None
HF_REVISION = os.environ.get("BYOM_HF_REVISION", "").strip() or None
MODEL_NAME = os.environ.get("BYOM_MODEL_NAME", "").strip() or f"quickstart-byom-{int(time.time())}"
KEEP = bool(int(os.environ.get("BYOM_KEEP", "0")))
POLL_INTERVAL_S = float(os.environ.get("BYOM_POLL_INTERVAL_S", "5"))
POLL_TIMEOUT_S = float(os.environ.get("BYOM_POLL_TIMEOUT_S", "600"))


def _color(s: str, c: str) -> str:
    return f"\033[{c}m{s}\033[0m"


GREEN = lambda s: _color(s, "32")
RED = lambda s: _color(s, "31")
CYAN = lambda s: _color(s, "36")
GRAY = lambda s: _color(s, "90")
YELLOW = lambda s: _color(s, "33")
BOLD = lambda s: _color(s, "1")


def banner(title: str) -> None:
    print()
    print(BOLD(CYAN(f"━━━ {title} " + "━" * max(0, 60 - len(title)))))


# ─────────────────────────────────────────────────────────────────────────────
# Inference endpoint resolution (mirrors quickstart.py — keeps this script
# free of any private SDK surface beyond the documented `client.*` methods)
# ─────────────────────────────────────────────────────────────────────────────
_API_SUFFIX = ".cloud"
_HTTP_ACTION_SUFFIX = ".site"


def _reflex_infer_url() -> str:
    try:
        from reflex._convex import convex_url as _sdk_url
    except ImportError:
        return ""
    base = _sdk_url(None).rstrip("/")
    if base.endswith(_API_SUFFIX):
        base = base[: -len(_API_SUFFIX)] + _HTTP_ACTION_SUFFIX
    return base + "/v1/infer"


INFER_URL = _reflex_infer_url()


def _infer_via_http(prompt: str, state14: list[float], jpeg_b64: str,
                    max_retries: int = 3) -> dict | None:
    """POST one observation to /v1/infer. Retries on HTTP 408 cold-starts."""
    body = json.dumps({
        "observation": {
            "prompt": prompt,
            "state": state14,
            "images": {
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
            with urllib.request.urlopen(req, timeout=180) as r:
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


def _synthetic_jpeg_b64() -> str:
    """Build a 224×224 black JPEG, base64-encoded — used as a stand-in for
    real camera frames so the script runs without hardware."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print(f"  {RED('numpy + pillow are required.')}  pip install numpy pillow")
        sys.exit(2)
    img = Image.fromarray(np.zeros((224, 224, 3), dtype="uint8"))
    buf = io.BytesIO(); img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Whoami — resolve key_id from the API key string
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_key_id() -> str | None:
    """Use the SDK's underlying Convex transport to call publicApi:whoami
    and return the canonical apiKeys document id for the bound key.

    No special SDK surface for this is needed because keys.bind_model is the
    only operation that takes a key_id, and customers typically obtain it
    once at app startup."""
    try:
        from reflex._convex import convex_call
    except ImportError:
        return None
    try:
        result = convex_call("query", "publicApi:whoami", {"apiKey": API_KEY})
    except Exception as e:
        print(f"  {RED('✗')} whoami failed: {type(e).__name__}: {e}")
        return None
    if not isinstance(result, dict) or not result.get("ok"):
        print(f"  {RED('✗')} whoami returned: {json.dumps(result)[:200]}")
        return None
    return result.get("keyId")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Import a pi0.5 LoRA adapter from HuggingFace
# ─────────────────────────────────────────────────────────────────────────────
def import_and_wait_ready(client) -> tuple[str | None, float]:
    """Trigger import_from_hf, poll until status='ready'. Returns (model_id, elapsed_s)."""
    banner("SECTION 1 — Import your pi0.5 adapter from HuggingFace")
    print(f"  hf repo             : {CYAN(HF_REPO)}")
    if HF_REVISION:
        print(f"  hf revision         : {CYAN(HF_REVISION)}")
    print(f"  hf token            : {GRAY('(set)' if HF_TOKEN else '(none — public repo)')}")
    print(f"  model name          : {CYAN(MODEL_NAME)}")
    print(f"  destination         : {GRAY('/vol/customer_models/<org>/<modelId>')}")

    print(f"\n  → {CYAN('client.models.import_from_hf(...)')}")
    t0 = time.perf_counter()
    try:
        result = client.models.import_from_hf(
            HF_REPO,
            MODEL_NAME,
            hf_revision=HF_REVISION,
            hf_token=HF_TOKEN,
        )
    except Exception as e:
        print(f"  {RED('✗')} import failed: {type(e).__name__}: {e}")
        return None, 0.0

    if not result.get("ok"):
        print(f"  {RED('✗')} server rejected import: {json.dumps(result)[:300]}")
        return None, 0.0

    model_id = result.get("modelId")
    kind = result.get("kind", "?")
    size_mb = (result.get("sizeBytes") or 0) / 1024 / 1024
    print(f"  {GREEN('✓')} import queued in {int((time.perf_counter() - t0)*1000)}ms")
    print(f"    model_id          : {BOLD(model_id)}")
    print(f"    kind              : {CYAN(kind)}  ({size_mb:.1f} MB)")
    print(f"    initial status    : {YELLOW(result.get('status', '?'))}")

    print(f"\n  → polling {CYAN('client.models.get(model_id)')} every {POLL_INTERVAL_S:.0f}s "
          f"(timeout {POLL_TIMEOUT_S:.0f}s)")
    last_status = None
    deadline = time.perf_counter() + POLL_TIMEOUT_S
    while time.perf_counter() < deadline:
        time.sleep(POLL_INTERVAL_S)
        try:
            row = client.models.get(model_id)
        except Exception as e:
            print(f"    {GRAY(f'poll error: {type(e).__name__}: {e}')}")
            continue
        artifact = row.get("artifact") if isinstance(row, dict) else None
        status = (artifact or {}).get("status") or row.get("status") or "?"
        elapsed = time.perf_counter() - t0
        if status != last_status:
            tag = GREEN if status == "ready" else (RED if status == "failed" else YELLOW)
            print(f"    t+{elapsed:5.1f}s  status={tag(status)}")
            last_status = status
        if status == "ready":
            return model_id, elapsed
        if status == "failed":
            reason = (artifact or {}).get("failureReason") or "(no reason given)"
            print(f"  {RED('✗')} model failed to prepare: {reason}")
            return None, elapsed

    print(f"  {RED('✗')} model still {last_status!r} after {POLL_TIMEOUT_S:.0f}s; check the dashboard")
    return None, time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Bind the API key to the model
# ─────────────────────────────────────────────────────────────────────────────
def bind_key_to_model(client, key_id: str, model_id: str) -> bool:
    banner("SECTION 2 — Bind your API key to the imported model")
    print(f"  → {CYAN('client.keys.bind_model(key_id, model_id)')}")
    print(f"    key_id            : {GRAY(key_id)}")
    print(f"    model_id          : {GRAY(model_id)}")
    try:
        result = client.keys.bind_model(key_id, model_id)
    except Exception as e:
        print(f"  {RED('✗')} bind failed: {type(e).__name__}: {e}")
        return False
    if not result.get("ok"):
        print(f"  {RED('✗')} server rejected bind: {json.dumps(result)[:300]}")
        return False
    print(f"  {GREEN('✓')} bound — /v1/infer for this key will now serve the customer adapter")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Prove the bound adapter is actually serving inference
# ─────────────────────────────────────────────────────────────────────────────
def verify_inference_uses_bound_model() -> bool:
    banner("SECTION 3 — Verify /v1/infer serves the bound adapter")
    print(f"  endpoint            : {GRAY(INFER_URL)}")
    print(f"  payload             : prompt + state[14] + 3× black-frame JPEG")
    jpeg_b64 = _synthetic_jpeg_b64()

    print(f"\n  → POST /v1/infer (auto-retries on 408 cold-start)")
    res = _infer_via_http("quickstart byom proof", [0.0] * 14, jpeg_b64)
    if not res or not res.get("ok"):
        print(f"  {RED('✗')} inference failed; reply: {res}")
        return False

    adapter_path = res.get("adapter_path") or ""
    print(f"  {GREEN('✓')} response in {res['_latency_ms']}ms")
    print(f"    server infer_ms   : {res.get('infer_ms', 0):.0f}ms")
    print(f"    total_ms          : {res.get('total_ms', 0):.0f}ms")
    print(f"    model_id          : {CYAN(res.get('model_id', '?'))}")
    print(f"    action chunk      : {res.get('num_steps')} steps × {res.get('chunk_size')} DOF")
    print(f"    adapter_path      : {BOLD(adapter_path) or GRAY('(none)')}")

    if "customer_models" in adapter_path:
        print(f"  {GREEN('✓')} adapter_path contains 'customer_models/' — "
              f"the BOUND model is serving this key {GREEN('confirmed')}")
        return True
    print(f"  {RED('✗')} adapter_path does NOT contain 'customer_models/'; "
          f"the platform default is still being served")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Cleanup
# ─────────────────────────────────────────────────────────────────────────────
def cleanup(client, key_id: str, model_id: str) -> None:
    banner("SECTION 4 — Cleanup (unbind + delete)")
    if KEEP:
        print(f"  {GRAY('BYOM_KEEP=1 — skipping unbind + delete')}")
        return

    print(f"  → {CYAN('client.keys.unbind_model(key_id)')}")
    try:
        result = client.keys.unbind_model(key_id)
        if result.get("ok"):
            print(f"  {GREEN('✓')} key unbound — future /v1/infer falls back to the platform default")
        else:
            print(f"  {YELLOW('!')} unbind returned: {json.dumps(result)[:200]}")
    except Exception as e:
        print(f"  {RED('✗')} unbind failed: {type(e).__name__}: {e}")

    print(f"\n  → {CYAN('client.models.delete(model_id)')}")
    try:
        result = client.models.delete(model_id)
        if result.get("ok"):
            print(f"  {GREEN('✓')} delete queued — Convex GC will reclaim the Modal Volume dir")
        else:
            print(f"  {YELLOW('!')} delete returned: {json.dumps(result)[:200]}")
    except Exception as e:
        print(f"  {RED('✗')} delete failed: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    if not API_KEY:
        print(RED("✗ REFLEX_API_KEY not set. Mint a key at https://app.tryreflex.ai/keys"))
        return 1
    if not HF_REPO:
        print(RED("✗ BYOM_HF_REPO not set."))
        print(GRAY("  set BYOM_HF_REPO to a HuggingFace repo id, e.g. 'yourorg/my-pi05-lora'"))
        print(GRAY("  (BYOM_HF_TOKEN is required for private repos)"))
        return 1

    try:
        import reflex
    except ImportError:
        print(RED("✗ reflex-sdk is not installed.  pip install 'reflex-sdk>=0.3.0'"))
        return 1

    banner("REFLEX BYO-MODEL QUICKSTART")
    print(f"  reflex-sdk version  : {GREEN(getattr(reflex, '__version__', '?'))}")
    print(f"  api_key             : {GRAY(API_KEY[:14] + '…')}")
    print(f"  hf repo             : {CYAN(HF_REPO)}")

    client = reflex.Client(api_key=API_KEY)

    print(f"\n  → resolving key_id via {CYAN('publicApi:whoami')}")
    key_id = _resolve_key_id()
    if not key_id:
        print(RED("✗ could not resolve key_id; aborting"))
        return 2
    print(f"  {GREEN('✓')} key_id = {BOLD(key_id)}")

    model_id, elapsed = import_and_wait_ready(client)
    if not model_id:
        return 3
    print(f"\n  {GREEN('✓')} model ready in {BOLD(f'{elapsed:.1f}s')}")

    if not bind_key_to_model(client, key_id, model_id):
        cleanup(client, key_id, model_id)
        return 4

    ok = verify_inference_uses_bound_model()

    cleanup(client, key_id, model_id)

    if not ok:
        return 5

    print()
    print(BOLD(GREEN("━━━ done — BYO-pi0.5 flow verified end-to-end ━━━")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
