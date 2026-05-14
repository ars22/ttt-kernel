"""Sandboxed KernelBench eval subprocess.

The trainer worker spawns one of these per pair. It owns its own CUDA context
on the trainer GPU (separate from the parent's PEFT model context), runs
`eval_kernel_against_ref` for each request, and replies with a JSON line. If
an evaluation poisons the CUDA context (illegal memory access, etc.), the
subprocess exits non-zero and the parent respawns it — the parent's trainer
context is unaffected because it lives in a different process.

Protocol (newline-delimited JSON over stdin/stdout):
  First line in: init message with the KernelBench eval config.
  First line out: `{"status": "ready"}` once setup is complete.
  Each subsequent line in: `{"cmd": "evaluate", "ref_src": ..., "custom_src": ...}`
                         or `{"cmd": "exit"}`.
  Each evaluate reply: `{"status": "ok"|"harness_none"|"exception", ...}`.

stdout fd 1 is reserved for JSON. sys.stdout and fd 1 are redirected to stderr
at startup so any library prints don't corrupt the channel.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

# Isolate JSON channel: save fd 1, redirect fd 1 + sys.stdout to stderr.
_JSON_FD = os.dup(1)
os.dup2(2, 1)
_JSON_OUT = os.fdopen(_JSON_FD, "w", buffering=1)
sys.stdout = sys.stderr


def _emit(obj: dict) -> None:
    _JSON_OUT.write(json.dumps(obj, default=str) + "\n")
    _JSON_OUT.flush()


def _looks_like_cuda_context_kill(err_str: str) -> bool:
    """Heuristic: errors that typically mean the CUDA context is now poisoned."""
    s = err_str.lower()
    return (
        "illegal memory access" in s
        or "cuda error" in s
        or "device-side assert" in s
        or "an unspecified launch failure" in s
    )


def main() -> None:
    init_line = sys.stdin.readline()
    if not init_line:
        _emit({"status": "fatal", "error": "no init message"})
        sys.exit(2)
    try:
        init = json.loads(init_line)
    except Exception as e:  # noqa: BLE001
        _emit({"status": "fatal", "error": f"bad init json: {e}"})
        sys.exit(2)

    repo_path = init["repo_path"]
    gpu_arch = init["gpu_arch"]
    backend = init["backend"]
    precision = init["precision"]
    timing_method = init["timing_method"]
    n_correct = int(init["num_correct_trials"])
    n_perf = int(init["num_perf_trials"])

    # Make kernelbench importable.
    src = os.path.join(repo_path, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        from kernelbench.utils import set_gpu_arch
        set_gpu_arch([gpu_arch])
        from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
        dtype = get_torch_dtype_from_string(precision)
    except Exception as e:  # noqa: BLE001
        _emit({"status": "fatal", "error": str(e),
               "traceback": traceback.format_exc()})
        sys.exit(2)

    _emit({"status": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            _emit({"status": "exception", "error": f"bad-json: {line[:200]}"})
            continue

        cmd = msg.get("cmd")
        if cmd == "exit":
            return
        if cmd != "evaluate":
            _emit({"status": "exception", "error": f"unknown cmd: {cmd}"})
            continue

        ref_src = msg["ref_src"]
        custom_src = msg["custom_src"]

        try:
            res = eval_kernel_against_ref(
                original_model_src=ref_src,
                custom_model_src=custom_src,
                verbose=False,
                measure_performance=True,
                timing_method=timing_method,
                num_correct_trials=n_correct,
                num_perf_trials=n_perf,
                backend=backend,
                precision=dtype,
            )
        except Exception as e:  # noqa: BLE001
            err = str(e)
            tb = traceback.format_exc(limit=20)
            _emit({"status": "exception", "error": err, "traceback": tb})
            if _looks_like_cuda_context_kill(err):
                # Context is now poisoned; bail so the parent respawns a clean one.
                sys.exit(3)
            continue

        if res is None:
            _emit({"status": "harness_none"})
            continue

        try:
            payload = {
                "status": "ok",
                "compiled": bool(res.compiled),
                "correctness": bool(res.correctness),
                "runtime": float(res.runtime),
                "ref_runtime": float(res.ref_runtime),
                "feedback": res.summarize_for_feedback(),
            }
        except Exception as e:  # noqa: BLE001
            payload = {
                "status": "exception",
                "error": f"could not serialize kb_res: {e}",
                "traceback": traceback.format_exc(),
            }
        _emit(payload)


if __name__ == "__main__":
    main()
