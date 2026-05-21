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

# Give each sandbox subprocess its own torch_extensions build dir on node-local
# /tmp. Without this, all sandboxes across all envs share the same NFS dir at
# /project/flame/.cache/torch_extensions/<name>. PyTorch's load_inline takes a
# FileLock on that dir, so when N sandboxes all build the same kernel name
# (e.g. the model emits name="matmul_custom_ext" for every L1 matmul problem),
# they serialize on a cross-node NFS lock and hang for minutes.
_TORCH_EXT_DIR = f"/tmp/torch_ext_{os.getpid()}"
os.makedirs(_TORCH_EXT_DIR, exist_ok=True)
os.environ["TORCH_EXTENSIONS_DIR"] = _TORCH_EXT_DIR


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

    # Models in pre-training saw a lot of old CUDA examples and routinely emit
    # `extra_cflags=["-std=c++14", ...]` in their load_inline call. Since the
    # later -std= flag wins on the c++ command line, this downgrades PyTorch's
    # default -std=c++17 to c++14 and modern torch headers (which use C++17-only
    # `std::is_enum_v` etc.) fail to compile. Strip any `-std=c++14`/`-std=c++11`
    # from extra_cflags/extra_cuda_cflags so PyTorch's -std=c++17 wins.
    try:
        from torch.utils import cpp_extension as _cpp_ext  # noqa: WPS433
        _orig_load_inline = _cpp_ext.load_inline
        _STD_BANLIST = ("-std=c++14", "-std=c++11", "-std=gnu++14", "-std=gnu++11")
        def _filter_std(flags):
            if not flags:
                return flags
            return [f for f in flags if f.strip() not in _STD_BANLIST]
        def _patched_load_inline(*args, **kwargs):
            kwargs["extra_cflags"] = _filter_std(kwargs.get("extra_cflags"))
            kwargs["extra_cuda_cflags"] = _filter_std(kwargs.get("extra_cuda_cflags"))
            return _orig_load_inline(*args, **kwargs)
        _cpp_ext.load_inline = _patched_load_inline
        # Also patch load() in case the kernel uses the non-inline variant.
        _orig_load = _cpp_ext.load
        def _patched_load(*args, **kwargs):
            kwargs["extra_cflags"] = _filter_std(kwargs.get("extra_cflags"))
            kwargs["extra_cuda_cflags"] = _filter_std(kwargs.get("extra_cuda_cflags"))
            return _orig_load(*args, **kwargs)
        _cpp_ext.load = _patched_load
    except Exception:  # noqa: BLE001
        pass

    try:
        from kernelbench.utils import set_gpu_arch, NVIDIA_ARCHS
        if gpu_arch in NVIDIA_ARCHS:
            set_gpu_arch([gpu_arch])
        else:
            # Direct nvcc arch like "10.0a" or "sm_100a" — bypass KB lookup.
            tcl = gpu_arch.replace("sm_", "").replace("_", ".")
            if tcl and tcl[0].isdigit() and "." not in tcl:
                tcl = tcl[:-1] + "." + tcl[-1] if not tcl[-1].isalpha() else tcl[:-2] + "." + tcl[-2:]
            os.environ["TORCH_CUDA_ARCH_LIST"] = tcl
        from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
        dtype = get_torch_dtype_from_string(precision)
        # Cap each sandbox's GPU memory: with K=8 sandboxes per pair sharing
        # the trainer GPU (~180 GB), if each grabs 20+ GB the pair OOMs.
        # Force a tight ceiling so the trainer still has headroom for the
        # GRPO step. Fraction is read from env; default 0.06 (≈11 GB on B200).
        import torch
        frac = float(os.environ.get("TTT_SANDBOX_MEM_FRACTION", "0.06"))
        try:
            torch.cuda.set_per_process_memory_fraction(frac)
        except Exception:
            pass
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
            # KernelBench's KernelExecResult historically had .summarize_for_feedback(),
            # but the current version of the package only exposes the underlying fields
            # (compiled, correctness, metadata, runtime, ref_runtime). Build the
            # feedback string inline so this code works against both APIs.
            if hasattr(res, "summarize_for_feedback"):
                feedback_str = res.summarize_for_feedback()
            else:
                md = getattr(res, "metadata", {}) or {}
                parts = [f"compiled={bool(res.compiled)}",
                         f"correct={bool(res.correctness)}"]
                if not res.compiled:
                    err = md.get("compilation_error") or md.get("compile_error") or md.get("error")
                    if err:
                        parts.append(f"compile_error: {str(err)[:2000]}")
                elif not res.correctness:
                    err = md.get("correctness_issue") or md.get("max_difference") or md.get("error")
                    if err:
                        parts.append(f"incorrect: {str(err)[:2000]}")
                else:
                    parts.append(f"runtime_us={float(res.runtime):.2f} ref_runtime_us={float(res.ref_runtime):.2f}")
                if md:
                    extra = {k: v for k, v in md.items() if k not in {"compilation_error","compile_error","correctness_issue","max_difference","error"}}
                    if extra:
                        parts.append(f"metadata={extra}")
                feedback_str = " | ".join(parts)
            payload = {
                "status": "ok",
                "compiled": bool(res.compiled),
                "correctness": bool(res.correctness),
                "runtime": float(res.runtime),
                "ref_runtime": float(res.ref_runtime),
                "feedback": feedback_str,
            }
        except Exception as e:  # noqa: BLE001
            payload = {
                "status": "exception",
                "error": f"could not serialize kb_res: {e}",
                "traceback": traceback.format_exc(),
            }
        _emit(payload)

        # Drop refs to the eval result so the loaded .so module's GPU memory
        # (CUDA fatbin, kernel symbol tables, tensors used in trials) can be
        # GC'd. Without this, each load_inline() call leaks ~100-500 MB into
        # the sandbox CUDA context — after a few dozen turns we OOM.
        del res
        try:
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
