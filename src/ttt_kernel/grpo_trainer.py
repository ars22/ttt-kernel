"""GRPO LoRA trainer for ttt-kernel.

Holds a PEFT-wrapped HF model in-process. After each turn the orchestrator
hands us K rollouts (token strings + rewards). We:

  1. Tokenize prompt + completion, build per-token mask over the completion.
  2. Forward through the policy to get current-policy logprobs.
  3. Compute group-relative advantage:  A_i = (r_i - mean(r)) / (std(r) + eps)
  4. Compute KL to a frozen reference snapshot (base model w/ no adapter).
  5. PPO-clipped surrogate:  -E[ min(ratio * A, clip(ratio, 1±eps) * A) ] + beta * KL
  6. AdamW step, grad clip.
  7. Save adapter to disk so SGLang can hot-reload it.

Distributed: when launched under torchrun (WORLD_SIZE > 1), we wrap the policy
in DDP so multiple ranks share the gradient computation. Rollouts are sliced
across ranks; DDP's backward-hook all-reduce keeps weights in sync.

Notes:
- The reference logprobs are computed from the SAME HF model with the adapter
  disabled (PEFT `with adapter.disable()`), so we don't need a second model in VRAM.
- We do NOT recompute logprobs at sampling time; SGLang doesn't return them by
  default. Instead we run one extra forward pass per rollout to fill them in,
  which is also what's needed for the policy logprobs. This is a single-process
  re-derivation — fine for K=8.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Rollout:
    prompt: str
    completion: str
    reward: float


def _dtype_from_str(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


def _init_distributed() -> tuple[int, int, int]:
    """Initialize torch.distributed if launched under torchrun; return (rank, world, local_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


class GRPOLoRATrainer:
    def __init__(self, model_cfg, lora_cfg, grpo_cfg):
        self.model_cfg = model_cfg
        self.lora_cfg = lora_cfg
        self.grpo_cfg = grpo_cfg

        self.rank, self.world_size, self.local_rank = _init_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.name, trust_remote_code=model_cfg.trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        base = AutoModelForCausalLM.from_pretrained(
            model_cfg.name,
            torch_dtype=_dtype_from_str(model_cfg.dtype),
            trust_remote_code=model_cfg.trust_remote_code,
        )
        self._peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            bias=lora_cfg.bias,
            task_type="CAUSAL_LM",
        )
        peft_model: PeftModel = get_peft_model(base, self._peft_config)
        peft_model.to(self.device)
        peft_model.train()
        self._peft_model = peft_model  # unwrapped; used for save/disable_adapter

        if self.world_size > 1:
            # find_unused_parameters: PEFT freezes most base params, so DDP needs
            # to be told that's expected. gradient_as_bucket_view saves memory.
            self.model = DDP(
                peft_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
        else:
            self.model = peft_model

        # Per-adapter AdamW: keyed by adapter name. The "default" adapter exists
        # immediately from get_peft_model and stays around as a placeholder
        # (PEFT requires at least one adapter to be loaded at any time).
        self.optimizers: dict[str, torch.optim.AdamW] = {}

    # ---- multi-adapter API -------------------------------------------------

    def _adapter_params(self, name: str) -> list[torch.nn.Parameter]:
        """Trainable parameters belonging to the named adapter only."""
        # PEFT names LoRA params like
        #   base_model.model.model.layers.0.self_attn.q_proj.lora_A.<name>.weight
        # So we filter by `.<name>.` to keep the active adapter's A/B matrices.
        tag = f".{name}."
        return [
            p for n, p in self._peft_model.named_parameters()
            if p.requires_grad and tag in n
        ]

    def add_problem_adapter(self, name: str) -> None:
        """Create a fresh adapter for a problem and register an optimizer for it.

        Kaiming-inits the lora_A matrices, zero-inits lora_B (zero-effect
        starting point). Explicitly activates the new adapter and forces
        requires_grad=True on its params — PEFT's add_adapter alone doesn't
        reliably mark them trainable across versions.
        """
        if name in self.optimizers:
            return
        self._peft_model.add_adapter(name, self._peft_config)
        # set_adapter() flips requires_grad: the named adapter's lora_A/B get
        # True, every other adapter's get False. Required for the optimizer
        # construction below to see any trainable params.
        self._peft_model.set_adapter(name)
        # Belt-and-suspenders: some PEFT versions don't toggle requires_grad
        # on set_adapter for adapters loaded after the initial get_peft_model.
        for n, p in self._peft_model.named_parameters():
            if f".{name}." in n and ("lora_A" in n or "lora_B" in n):
                p.requires_grad = True
            if "lora_A" in n and f".{name}." in n:
                torch.nn.init.kaiming_uniform_(p, a=5 ** 0.5)
            elif "lora_B" in n and f".{name}." in n:
                torch.nn.init.zeros_(p)
        params = self._adapter_params(name)
        if not params:
            # Debug fallback: enumerate matching names so the error message is
            # actually useful next time the naming convention shifts.
            matches = [n for n, _ in self._peft_model.named_parameters() if f".{name}." in n]
            raise RuntimeError(
                f"no trainable params found for adapter '{name}'. "
                f"matching names ({len(matches)}): {matches[:5]}"
            )
        self.optimizers[name] = torch.optim.AdamW(
            params,
            lr=self.grpo_cfg.learning_rate,
            weight_decay=self.grpo_cfg.weight_decay,
        )

    def delete_problem_adapter(self, name: str) -> None:
        """Release the adapter's weights + optimizer state."""
        # Drop optimizer first so its param refs don't keep weights alive.
        self.optimizers.pop(name, None)
        try:
            self._peft_model.delete_adapter(name)
        except Exception:
            # delete_adapter raises if `name` isn't loaded — safe to ignore.
            pass

    def save_adapter(self, name: str, out_dir: str) -> str:
        """Save just the named adapter's weights to disk.

        NOTE: we deliberately do NOT save the tokenizer here. SGLang's LoRA
        manager rejects adapters whose directory contains added_tokens.json,
        even when those tokens are part of the BASE tokenizer.
        """
        os.makedirs(out_dir, exist_ok=True)
        if self.rank == 0:
            # `selected_adapters` ensures only this one adapter's files land
            # in `out_dir` — important because SGLang reads adapter_config.json
            # from that directory and would be confused by a multi-adapter dump.
            self._peft_model.save_pretrained(out_dir, selected_adapters=[name])
        if self.world_size > 1:
            dist.barrier()
        return out_dir

    # ---- one GRPO step over K rollouts -------------------------------------

    def step(
        self,
        rollouts: List[Rollout],
        *,
        adapter_name: str,
        group_ids: Optional[List[int]] = None,
    ) -> dict:
        """One GRPO update on the named adapter (batched over all rollouts).

        All K rollouts are tokenized, right-padded into one [B, T] batch, and
        run through the model in a SINGLE forward+backward. This replaces the
        previous Python for-loop over rollouts, which serialized 8 small
        forwards back-to-back and barely used the GPU. Batched pass is ~5–10×
        faster on a B200 because cuBLAS / flash-attention can process all K
        sequences in one go.

        Caller is responsible for serializing concurrent `step` calls against
        the trainer GPU (they all share base-model state).
        """
        if adapter_name not in self.optimizers:
            raise KeyError(f"adapter '{adapter_name}' not loaded; call add_problem_adapter first")
        self._peft_model.set_adapter(adapter_name)
        optimizer = self.optimizers[adapter_name]
        device = self.device
        rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32, device=device)

        if group_ids is None:
            group_ids = [0] * len(rollouts)
        assert len(group_ids) == len(rollouts)

        # Per-group advantage normalization (GRPO baseline is per-group).
        adv = torch.zeros_like(rewards)
        for g in sorted(set(group_ids)):
            idx = [i for i, gi in enumerate(group_ids) if gi == g]
            idx_t = torch.tensor(idx, device=device, dtype=torch.long)
            r_g = rewards[idx_t]
            if self.grpo_cfg.group_advantage_norm and r_g.numel() > 1:
                a = (r_g - r_g.mean()) / (r_g.std(unbiased=False) + 1e-6)
            else:
                a = r_g - r_g.mean()
            adv[idx_t] = a

        # ---- tokenize + pad ALL rollouts into one batched tensor ----------
        max_seq = self.grpo_cfg.max_seq_len
        pad_id = self.tokenizer.pad_token_id
        ids_list: list[torch.Tensor] = []
        comp_mask_list: list[torch.Tensor] = []
        for ro in rollouts:
            try:
                prompt_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": ro.prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                )[0]
            except Exception:
                prompt_ids = self.tokenizer(
                    ro.prompt, return_tensors="pt", add_special_tokens=False,
                ).input_ids[0]
            comp_ids = self.tokenizer(
                ro.completion, return_tensors="pt", add_special_tokens=False,
            ).input_ids[0]
            if prompt_ids.numel() + comp_ids.numel() > max_seq:
                # Thinking-model completions are "[CoT] </think> [kernel]" so
                # the IMPORTANT tokens are at the END. Keep the LAST chunk of
                # the completion (including </think> + final kernel + EOS) and
                # drop the head (mostly redundant reasoning). Previously we
                # kept comp_ids[:keep_comp] which trained on raw thinking and
                # missed the actual kernel — a real bug for RL-on-thinking.
                keep_comp = min(comp_ids.numel(), max_seq - prompt_ids.numel())
                # Cap prompt to at least 512 tokens of the FENCE around the
                # user request so the model has the problem in context.
                if keep_comp < 64:
                    keep_comp = min(comp_ids.numel(), max_seq // 2)
                keep_prompt = max_seq - keep_comp
                prompt_ids = prompt_ids[-keep_prompt:]
                comp_ids = comp_ids[-keep_comp:]  # keep LAST keep_comp tokens
            ids = torch.cat([prompt_ids, comp_ids])
            m = torch.zeros(ids.numel(), dtype=torch.float32)
            m[prompt_ids.numel():] = 1.0  # completion positions only
            ids_list.append(ids)
            comp_mask_list.append(m)

        B = len(ids_list)
        T = max(ids.numel() for ids in ids_list)
        input_ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((B, T), dtype=torch.long, device=device)
        comp_mask = torch.zeros((B, T), dtype=torch.float32, device=device)
        for i, (ids, m) in enumerate(zip(ids_list, comp_mask_list)):
            L = ids.numel()
            input_ids[i, :L] = ids.to(device)
            attn_mask[i, :L] = 1
            comp_mask[i, :L] = m.to(device)

        # Targets and shifted completion mask: logits[t] predicts input_ids[t+1].
        targets = input_ids[:, 1:]                       # [B, T-1]
        tgt_mask = comp_mask[:, 1:]                       # [B, T-1] (token i predicts comp at i+1)

        # Micro-batch along B to keep the logits tensor (B' × T × V) in memory.
        # Full-batch forward would materialize ~40 GB of bf16 logits at B=8,
        # T=16k, V=150k — plus the same for the reference forward — and OOM
        # the trainer GPU. With MB=2 the per-step peak drops to ~10 GB.
        mb = max(int(self.grpo_cfg.micro_batch_size), 1)
        chunks = [list(range(i, min(i + mb, B))) for i in range(0, B, mb)]

        # Memory-efficient per-token logprob: gather target logits and subtract
        # logsumexp instead of materializing log_softmax over the vocab dim.
        def _tok_logp(logits_bt_v: torch.Tensor, tgt_bt: torch.Tensor) -> torch.Tensor:
            gathered = logits_bt_v.gather(-1, tgt_bt.unsqueeze(-1)).squeeze(-1).float()
            lse = torch.logsumexp(logits_bt_v.float(), dim=-1)
            return gathered - lse

        total_loss_val = 0.0
        total_kl_val = 0.0
        total_pg_val = 0.0
        total_gn_val = 0.0

        for _ in range(self.grpo_cfg.update_epochs):
            pg_acc = torch.zeros((), device=device, dtype=torch.float32)
            kl_acc = torch.zeros((), device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)

            for idxs in chunks:
                idx_t = torch.tensor(idxs, device=device, dtype=torch.long)
                in_ids = input_ids.index_select(0, idx_t)
                am = attn_mask.index_select(0, idx_t)
                tgt = targets.index_select(0, idx_t)
                tm = tgt_mask.index_select(0, idx_t)
                adv_mb = adv.index_select(0, idx_t).unsqueeze(-1)

                # ---- current-policy forward (gradient flows) --------------
                out = self.model(input_ids=in_ids, attention_mask=am, use_cache=False)
                tok_logp = _tok_logp(out.logits[:, :-1], tgt)

                # ---- reference forward (adapter disabled, no grad) ---------
                with torch.no_grad():
                    with self._peft_model.disable_adapter():
                        ref_out = self.model(input_ids=in_ids, attention_mask=am, use_cache=False)
                    ref_tok_logp = _tok_logp(ref_out.logits[:, :-1], tgt)

                # ---- PPO-clipped objective + KL ----------------------------
                ratio = torch.exp(tok_logp - ref_tok_logp.detach())
                eps = self.grpo_cfg.epsilon_clip
                unclipped = ratio * adv_mb
                clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv_mb
                pg_per_tok = -torch.min(unclipped, clipped)
                kl_per_tok = tok_logp - ref_tok_logp

                denom = tm.sum(dim=-1).clamp(min=1.0)
                pg_per_seq = (pg_per_tok * tm).sum(dim=-1) / denom
                kl_per_seq = (kl_per_tok * tm).sum(dim=-1) / denom
                # Scale by chunk size so we get a true mean over the full batch
                # after summing all chunks' contributions.
                w = pg_per_seq.numel() / float(B)
                pg_chunk = pg_per_seq.mean() * w
                kl_chunk = kl_per_seq.mean() * w
                loss_chunk = pg_chunk + self.grpo_cfg.beta_kl * kl_chunk
                loss_chunk.backward()
                pg_acc = pg_acc + pg_chunk.detach()
                kl_acc = kl_acc + kl_chunk.detach()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._adapter_params(adapter_name),
                self.grpo_cfg.grad_clip,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss_val += float(pg_acc + self.grpo_cfg.beta_kl * kl_acc)
            total_pg_val += float(pg_acc)
            total_kl_val += float(kl_acc)
            total_gn_val += float(grad_norm)

        # Multi-adapter training fragments the cache across problems. Release
        # cached blocks back to CUDA so the next problem's add_adapter /
        # save_pretrained can allocate without hitting an OOM on a near-full
        # but mostly-cached GPU.
        torch.cuda.empty_cache()

        ne = max(self.grpo_cfg.update_epochs, 1)
        return {
            "loss": total_loss_val / ne,
            "kl": total_kl_val / ne,
            "pg": total_pg_val / ne,
            "grad_norm": total_gn_val / ne,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)) if rewards.numel() > 1 else 0.0,
            "advantage_mean": float(adv.mean()),
        }
