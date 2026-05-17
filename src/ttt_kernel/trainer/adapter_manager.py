"""LRU of resident LoRA adapters + per-adapter locks + save.

Concurrent /train calls on *different* adapters are allowed (the base forward
is shared; only A/B + AdamW are per-adapter). /train calls on the *same*
adapter serialize via this manager's per-adapter lock.

Cold-load semantics: when the orchestrator asks to train on adapter X with
`adapter_in_path=/.../v003`, we either reuse an in-memory snapshot if cached
or load the safetensors from disk into a fresh PEFT slot. After training,
the new weights are saved to `adapter_out_path=/.../v004` and the in-memory
copy stays under the new name so likely-next-turn reuse is fast.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from typing import Optional

import torch
from peft import PeftModel
from safetensors.torch import load_file as load_safetensors

log = logging.getLogger("ttt_kernel.trainer.adapter_manager")


def _adapter_param_names(peft_model: PeftModel, adapter_name: str) -> list[str]:
    """PEFT names per-adapter LoRA params with a `.<adapter>.` segment."""
    tag = f".{adapter_name}."
    return [n for n, _ in peft_model.named_parameters() if tag in n]


def _adapter_params(peft_model: PeftModel, adapter_name: str) -> list[torch.nn.Parameter]:
    tag = f".{adapter_name}."
    return [
        p for n, p in peft_model.named_parameters()
        if p.requires_grad and tag in n
    ]


def _zero_init_adapter(peft_model: PeftModel, adapter_name: str) -> None:
    """Kaiming-init lora_A, zero-init lora_B → effective zero-LoRA starting point.
    Required for v000 (seed)."""
    for n, p in peft_model.named_parameters():
        if f".{adapter_name}." not in n:
            continue
        if "lora_A" in n:
            torch.nn.init.kaiming_uniform_(p, a=5 ** 0.5)
        elif "lora_B" in n:
            torch.nn.init.zeros_(p)


def _load_adapter_weights_from_dir(
    peft_model: PeftModel,
    adapter_name: str,
    adapter_dir: str,
) -> None:
    """Pull adapter_model.safetensors from `adapter_dir` into the named adapter's
    parameters in place. Missing files → leave the adapter as-is (the caller
    just freshly initialized it)."""
    sd_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(sd_path):
        log.warning("no adapter_model.safetensors in %s; using fresh-init weights", adapter_dir)
        return
    state = load_safetensors(sd_path)
    # Map adapter_state keys (which use the 'default' name from the on-disk
    # adapter_config.json) onto our chosen `adapter_name`. PEFT's save format is
    # `base_model.model.<...>.lora_{A,B}.<adapter>.weight` so we just swap the
    # `<adapter>` token in the middle. We accept several common source names.
    own = {n: p for n, p in peft_model.named_parameters() if f".{adapter_name}." in n}
    if not own:
        raise RuntimeError(f"adapter '{adapter_name}' has no params after add_adapter")
    loaded = 0
    skipped = 0
    for src_key, src_val in state.items():
        # Walk possible source adapter names: try to swap any `.<x>.weight`
        # before `lora_A`/`lora_B` tail with `.<adapter_name>.weight`.
        candidate = None
        for marker in (".lora_A.", ".lora_B."):
            if marker in src_key:
                head, tail = src_key.split(marker, 1)
                # tail is like '<src_adapter>.weight'
                if "." in tail:
                    candidate = head + marker + adapter_name + "." + tail.split(".", 1)[1]
                break
        if candidate is None:
            skipped += 1
            continue
        param = own.get(candidate)
        if param is None:
            skipped += 1
            continue
        with torch.no_grad():
            param.data.copy_(src_val.to(param.data.dtype).to(param.data.device))
        loaded += 1
    log.info("loaded %d adapter tensors from %s (%d skipped)", loaded, adapter_dir, skipped)


class AdapterManager:
    """Owns the PEFT model and a dict of (name -> AdamW). Thread-of-control:
    single event loop calls these methods; the per-adapter locks gate
    concurrent /train calls."""

    def __init__(
        self,
        peft_model: PeftModel,
        peft_config,            # LoraConfig used to instantiate new adapters
        learning_rate: float,
        weight_decay: float,
        max_resident: int,
        device: torch.device,
        rank: int = 0,
        world: int = 1,
    ):
        if max_resident < 1:
            raise ValueError(f"max_resident must be >=1, got {max_resident}")
        self.peft_model = peft_model
        self.peft_config = peft_config
        self.lr = learning_rate
        self.wd = weight_decay
        self.max_resident = max_resident
        self.device = device
        self.rank = rank
        self.world = world
        # name -> AdamW; OrderedDict order is LRU.
        self.optimizers: OrderedDict[str, torch.optim.AdamW] = OrderedDict()
        # name -> lock; one /train on the same name serializes.
        self._locks: dict[str, asyncio.Lock] = {}
        # Global lock for adapter add/delete/save (PEFT's adapter dict isn't
        # internally thread-safe and we mutate set_adapter() in /train too).
        self._mgmt_lock = asyncio.Lock()

    def lock_for(self, name: str) -> asyncio.Lock:
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    async def ensure_resident(self, name: str, source_dir: Optional[str]) -> None:
        """Ensure the named adapter is loaded into the PEFT model. If
        `source_dir` is provided and the adapter is fresh, also copy weights
        from disk into the adapter's parameters.

        If the adapter is already resident (warm-cache hit), this is a no-op
        and `source_dir` is ignored — the caller has already decided this is
        the same logical adapter version.
        """
        async with self._mgmt_lock:
            if name in self.optimizers:
                # LRU touch
                self.optimizers.move_to_end(name)
                return
            while len(self.optimizers) >= self.max_resident:
                evict_name, opt = self.optimizers.popitem(last=False)
                opt.zero_grad(set_to_none=True)
                try:
                    self.peft_model.delete_adapter(evict_name)
                except Exception as e:  # noqa: BLE001
                    log.warning("delete_adapter(%s) failed during eviction: %s", evict_name, e)
                self._locks.pop(evict_name, None)
                log.info("evicted resident adapter %s", evict_name)
            self.peft_model.add_adapter(name, self.peft_config)
            self.peft_model.set_adapter(name)
            # Force requires_grad=True on this adapter's params — set_adapter
            # is unreliable across PEFT versions for adapters added after init.
            for n, p in self.peft_model.named_parameters():
                if f".{name}." in n and ("lora_A" in n or "lora_B" in n):
                    p.requires_grad = True
            _zero_init_adapter(self.peft_model, name)
            if source_dir is not None:
                _load_adapter_weights_from_dir(self.peft_model, name, source_dir)
            params = _adapter_params(self.peft_model, name)
            if not params:
                raise RuntimeError(f"no trainable params for adapter '{name}'")
            self.optimizers[name] = torch.optim.AdamW(
                params, lr=self.lr, weight_decay=self.wd,
            )

    async def activate(self, name: str) -> None:
        """Make `name` the active adapter for forward passes."""
        async with self._mgmt_lock:
            self.peft_model.set_adapter(name)

    def adapter_params(self, name: str) -> list[torch.nn.Parameter]:
        return _adapter_params(self.peft_model, name)

    async def save(self, name: str, out_dir: str) -> str:
        """Save the named adapter's weights to `out_dir` (atomic via tmp+rename).
        Only rank 0 writes; other ranks wait on a barrier. Tokenizer files are
        NOT saved (SGLang's LoRA manager rejects adapter dirs containing them).
        """
        import torch.distributed as dist
        os.makedirs(out_dir, exist_ok=True)
        if self.world > 1:
            # FSDP2 path: gather full tensors on rank 0 and write safetensors
            # directly (PEFT's save_pretrained doesn't know about DTensors).
            from .fsdp_save import save_adapter_fsdp
            save_adapter_fsdp(self.peft_model, name, out_dir, self.peft_config, rank=self.rank)
            if dist.is_initialized():
                dist.barrier()
            return out_dir
        if self.rank == 0:
            tmp = out_dir + ".tmp"
            if os.path.isdir(tmp):
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)
            self.peft_model.save_pretrained(tmp, selected_adapters=[name])
            # Atomic swap so a partially-written dir is never visible.
            if os.path.isdir(out_dir):
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)
            os.rename(tmp, out_dir)
        if self.world > 1 and dist.is_initialized():
            dist.barrier()
        return out_dir
