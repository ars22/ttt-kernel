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
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Rollout:
    prompt: str
    completion: str
    reward: float


def _dtype_from_str(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


class GRPOLoRATrainer:
    def __init__(self, model_cfg, lora_cfg, grpo_cfg):
        self.model_cfg = model_cfg
        self.lora_cfg = lora_cfg
        self.grpo_cfg = grpo_cfg

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
        self.model: PeftModel = get_peft_model(base, peft_config)
        self.model.cuda()
        self.model.train()

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
        self.model.save_pretrained(out_dir)
        return out_dir

    def reset_adapter(self) -> None:
        """Re-init LoRA weights to zero-effect (for fresh-per-problem mode)."""
        for name, p in self.model.named_parameters():
            if "lora_A" in name:
                torch.nn.init.kaiming_uniform_(p, a=5 ** 0.5)
            elif "lora_B" in name:
                torch.nn.init.zeros_(p)

    # ---- one GRPO step over K rollouts -------------------------------------

    def step(self, rollouts: List[Rollout]) -> dict:
        """One GRPO update from a group of K rollouts on the SAME prompt.

        Returns a dict of scalar metrics for logging.
        """
        device = next(self.model.parameters()).device
        rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32, device=device)

        # Group-relative advantage.
        if self.grpo_cfg.group_advantage_norm and rewards.numel() > 1:
            adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
        else:
            adv = rewards - rewards.mean()

        total_loss = 0.0
        total_kl = 0.0
        total_pg = 0.0
        n = 0

        for epoch in range(self.grpo_cfg.update_epochs):
            for i, ro in enumerate(rollouts):
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

            torch.nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad),
                self.grpo_cfg.grad_clip,
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": total_loss / max(n, 1),
            "kl": total_kl / max(n, 1),
            "pg": total_pg / max(n, 1),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)) if rewards.numel() > 1 else 0.0,
            "advantage_mean": float(adv.mean()),
        }

    # ---- per-rollout loss --------------------------------------------------

    def _loss_one_rollout(self, *, prompt: str, completion: str, advantage: torch.Tensor) -> dict:
        """Compute PPO-clipped policy loss + KL for a single (prompt, completion)."""
        device = next(self.model.parameters()).device

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
        with torch.no_grad():
            with self.model.disable_adapter():
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
