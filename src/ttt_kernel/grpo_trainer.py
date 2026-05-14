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
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            bias=lora_cfg.bias,
            task_type="CAUSAL_LM",
        )
        peft_model: PeftModel = get_peft_model(base, peft_config)
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

        # Optimizer over LoRA params only.
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=grpo_cfg.learning_rate,
            weight_decay=grpo_cfg.weight_decay,
        )

    # ---- IO helpers --------------------------------------------------------

    def save_adapter(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        # NOTE: we deliberately do NOT save the tokenizer here. SGLang's LoRA
        # manager rejects adapters whose directory contains added_tokens.json
        # ("LoRA serving currently doesn't support adapters that add tokens"),
        # even when those tokens are part of the BASE tokenizer (e.g. Qwen3-
        # Thinking's <think>/</think>). Tokenizer files live with the base model.
        # Only rank 0 writes; other ranks wait at a barrier so the next
        # reload_adapter call sees a complete adapter on disk.
        if self.rank == 0:
            self._peft_model.save_pretrained(out_dir)
        if self.world_size > 1:
            dist.barrier()
        return out_dir

    def reset_adapter(self) -> None:
        """Re-init LoRA weights to zero-effect (for fresh-per-problem mode).

        Under DDP we only init on rank 0 (kaiming_uniform_ draws from each
        rank's RNG independently — they would diverge otherwise), then
        broadcast the LoRA params to keep replicas bit-identical.
        """
        if self.rank == 0:
            for name, p in self._peft_model.named_parameters():
                if "lora_A" in name:
                    torch.nn.init.kaiming_uniform_(p, a=5 ** 0.5)
                elif "lora_B" in name:
                    torch.nn.init.zeros_(p)
        if self.world_size > 1:
            for name, p in self._peft_model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    dist.broadcast(p.data, src=0)
            dist.barrier()

    # ---- one GRPO step over K rollouts -------------------------------------

    def step(self, rollouts: List[Rollout], group_ids: Optional[List[int]] = None) -> dict:
        """One GRPO update from a flat list of rollouts, optionally split into groups.

        When `group_ids` is provided, advantages are computed PER group (so the
        baseline is per-problem). Under multi-problem batching this is essential
        — a single global baseline across problems would couple their rewards.

        Returns a dict of scalar metrics for logging.
        """
        device = self.device
        rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32, device=device)

        if group_ids is None:
            group_ids = [0] * len(rollouts)
        assert len(group_ids) == len(rollouts)

        # Per-group advantage normalization (GRPO baseline is per-group).
        adv = torch.zeros_like(rewards)
        unique_groups = sorted(set(group_ids))
        for g in unique_groups:
            idx = [i for i, gi in enumerate(group_ids) if gi == g]
            idx_t = torch.tensor(idx, device=device, dtype=torch.long)
            r_g = rewards[idx_t]
            if self.grpo_cfg.group_advantage_norm and r_g.numel() > 1:
                a = (r_g - r_g.mean()) / (r_g.std(unbiased=False) + 1e-6)
            else:
                a = r_g - r_g.mean()
            adv[idx_t] = a

        # Slice rollouts across ranks for data-parallel training.
        my_indices = list(range(self.rank, len(rollouts), self.world_size))

        total_loss = 0.0
        total_kl = 0.0
        total_pg = 0.0
        total_gn = 0.0
        n_updates = 0
        n = 0

        for epoch in range(self.grpo_cfg.update_epochs):
            for i in my_indices:
                ro = rollouts[i]
                metrics = self._loss_one_rollout(
                    prompt=ro.prompt,
                    completion=ro.completion,
                    advantage=adv[i].detach(),
                )
                loss = metrics["loss"]
                loss.backward()
                total_loss += float(loss.detach())
                total_kl += metrics["kl"]
                total_pg += metrics["pg"]
                n += 1

            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad),
                self.grpo_cfg.grad_clip,
            )
            total_gn += float(grad_norm)
            n_updates += 1
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        # Reduce per-rank scalars to a global mean for logging.
        if self.world_size > 1:
            t = torch.tensor(
                [total_loss, total_kl, total_pg, total_gn, float(n), float(n_updates)],
                device=device,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss, total_kl, total_pg, total_gn, n, n_updates = t.tolist()

        return {
            "loss": total_loss / max(n, 1),
            "kl": total_kl / max(n, 1),
            "pg": total_pg / max(n, 1),
            "grad_norm": total_gn / max(n_updates, 1),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)) if rewards.numel() > 1 else 0.0,
            "advantage_mean": float(adv.mean()),
        }

    # ---- per-rollout loss --------------------------------------------------

    def _loss_one_rollout(self, *, prompt: str, completion: str, advantage: torch.Tensor) -> dict:
        """Compute PPO-clipped policy loss + KL for a single (prompt, completion)."""
        device = self.device

        # Match SGLang's chat-completions sampling: prompt is wrapped as a
        # single user turn with `add_generation_prompt=True` (adds the
        # assistant-marker that the rollout was conditioned on). Completion is
        # the raw assistant text (SGLang strips the <|im_start|>/<|im_end|>
        # markers around it).
        try:
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )[0]
        except Exception:
            # Fallback for tokenizers without a chat template.
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        comp_ids = self.tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
        # Truncate (prompt + completion) to max_seq_len, keep the completion intact if possible.
        max_seq = self.grpo_cfg.max_seq_len
        if prompt_ids.numel() + comp_ids.numel() > max_seq:
            keep_comp = min(comp_ids.numel(), max_seq // 2)
            keep_prompt = max_seq - keep_comp
            prompt_ids = prompt_ids[-keep_prompt:]
            comp_ids = comp_ids[:keep_comp]

        input_ids = torch.cat([prompt_ids, comp_ids]).unsqueeze(0).to(device)
        # Loss is computed over completion tokens only.
        comp_start = prompt_ids.numel()

        # ---- current-policy logprobs (gradient flows here) -----------------
        out = self.model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, :-1]  # predict t+1 from t
        targets = input_ids[0, 1:]
        logprobs = F.log_softmax(logits.float(), dim=-1)
        tok_logp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # Mask: only completion positions count.
        # logits[t] predicts targets[t] = input_ids[t+1]; the first completion target
        # corresponds to position comp_start in input_ids → index (comp_start - 1) in tok_logp.
        mask = torch.zeros_like(tok_logp, dtype=torch.float32)
        mask[comp_start - 1 :] = 1.0

        # ---- reference logprobs (adapter disabled, no grad) ----------------
        # disable_adapter() lives on the PEFT model; under DDP that's _peft_model.
        with torch.no_grad():
            with self._peft_model.disable_adapter():
                ref_out = self.model(input_ids=input_ids, use_cache=False)
            ref_logits = ref_out.logits[0, :-1]
            ref_logprobs = F.log_softmax(ref_logits.float(), dim=-1)
            ref_tok_logp = ref_logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # ---- PPO-clipped objective ----------------------------------------
        # Reference policy *is* the importance-sampling proposal here (no separate
        # "old" snapshot per step; for single-update-per-batch this collapses to
        # importance-sampled policy gradient with clipping).
        ratio = torch.exp(tok_logp - ref_tok_logp.detach())
        eps = self.grpo_cfg.epsilon_clip
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantage
        pg_per_tok = -torch.min(unclipped, clipped)

        # Token-level KL ≈ (logπ - logπ_ref).
        kl_per_tok = (tok_logp - ref_tok_logp).detach() * 0 + (tok_logp - ref_tok_logp)
        # Use the masked mean so loss magnitude is independent of sequence length.
        denom = mask.sum().clamp(min=1.0)
        pg = (pg_per_tok * mask).sum() / denom
        kl = (kl_per_tok * mask).sum() / denom

        loss = pg + self.grpo_cfg.beta_kl * kl
        return {"loss": loss, "kl": float(kl.detach()), "pg": float(pg.detach())}
