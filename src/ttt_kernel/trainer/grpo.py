"""GRPO step — ported verbatim (math) from grpo_trainer.py::step.

Pure function over (peft_model, optimizer, tokenizer, rollouts, cfg) so we
can reuse it under both the single-GPU model wrap (task #5) and the FSDP2
wrap (task #6) without copy-paste.

The maintained invariant vs main is: per-token logprob is `gather - logsumexp`,
PPO ratio uses ref forward with adapter disabled, KL is the per-token
difference, advantages are per-group mean/std-normalized. An algorithmic
parity test against the `main` branch lives in tests/test_grpo_parity.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class GRPOStepCfg:
    beta_kl: float
    epsilon_clip: float
    group_advantage_norm: bool
    update_epochs: int
    grad_clip: float
    micro_batch_size: int
    max_seq_len: int


@dataclass(frozen=True)
class RolloutT:
    prompt: str
    completion: str
    reward: float


def _tok_logp(logits_bt_v: torch.Tensor, tgt_bt: torch.Tensor) -> torch.Tensor:
    """Memory-efficient per-token logprob: gather target logit and subtract
    logsumexp instead of materializing log_softmax across the vocab."""
    gathered = logits_bt_v.gather(-1, tgt_bt.unsqueeze(-1)).squeeze(-1).float()
    lse = torch.logsumexp(logits_bt_v.float(), dim=-1)
    return gathered - lse


def _tokenize_rollouts(
    tokenizer: PreTrainedTokenizerBase,
    rollouts: List[RolloutT],
    *,
    max_seq: int,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (input_ids[B,T], attn_mask[B,T], comp_mask[B,T])."""
    ids_list: list[torch.Tensor] = []
    comp_mask_list: list[torch.Tensor] = []
    for ro in rollouts:
        try:
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": ro.prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )[0]
        except Exception:
            prompt_ids = tokenizer(
                ro.prompt, return_tensors="pt", add_special_tokens=False,
            ).input_ids[0]
        comp_ids = tokenizer(
            ro.completion, return_tensors="pt", add_special_tokens=False,
        ).input_ids[0]
        if prompt_ids.numel() + comp_ids.numel() > max_seq:
            # Thinking-model completions have the important tokens (kernel)
            # at the END — keep the last keep_comp tokens of the completion.
            keep_comp = min(comp_ids.numel(), max_seq - prompt_ids.numel())
            if keep_comp < 64:
                keep_comp = min(comp_ids.numel(), max_seq // 2)
            keep_prompt = max_seq - keep_comp
            prompt_ids = prompt_ids[-keep_prompt:]
            comp_ids = comp_ids[-keep_comp:]
        ids = torch.cat([prompt_ids, comp_ids])
        m = torch.zeros(ids.numel(), dtype=torch.float32)
        m[prompt_ids.numel():] = 1.0
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
    return input_ids, attn_mask, comp_mask


def _group_advantages(
    rewards: torch.Tensor,
    group_ids: List[int],
    *,
    norm: bool,
    device: torch.device,
) -> torch.Tensor:
    adv = torch.zeros_like(rewards)
    for g in sorted(set(group_ids)):
        idx = [i for i, gi in enumerate(group_ids) if gi == g]
        idx_t = torch.tensor(idx, device=device, dtype=torch.long)
        r_g = rewards[idx_t]
        if norm and r_g.numel() > 1:
            a = (r_g - r_g.mean()) / (r_g.std(unbiased=False) + 1e-6)
        else:
            a = r_g - r_g.mean()
        adv[idx_t] = a
    return adv


def grpo_step(
    *,
    peft_model: PeftModel,
    model_call: torch.nn.Module,         # what to call .forward() on (peft_model or DDP/FSDP wrap)
    optimizer: torch.optim.Optimizer,
    tokenizer: PreTrainedTokenizerBase,
    adapter_name: str,
    adapter_params: list[torch.nn.Parameter],
    rollouts: List[RolloutT],
    cfg: GRPOStepCfg,
    device: torch.device,
    group_ids: Optional[List[int]] = None,
) -> dict:
    if group_ids is None:
        group_ids = [0] * len(rollouts)
    assert len(group_ids) == len(rollouts)
    peft_model.set_adapter(adapter_name)

    rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32, device=device)
    adv = _group_advantages(rewards, group_ids, norm=cfg.group_advantage_norm, device=device)

    input_ids, attn_mask, comp_mask = _tokenize_rollouts(
        tokenizer, rollouts,
        max_seq=cfg.max_seq_len,
        pad_id=tokenizer.pad_token_id,
        device=device,
    )
    targets = input_ids[:, 1:]
    tgt_mask = comp_mask[:, 1:]
    B = input_ids.size(0)
    mb = max(int(cfg.micro_batch_size), 1)
    chunks = [list(range(i, min(i + mb, B))) for i in range(0, B, mb)]

    total = {"loss": 0.0, "pg": 0.0, "kl": 0.0, "grad_norm": 0.0}
    for _ in range(cfg.update_epochs):
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

            out = model_call(input_ids=in_ids, attention_mask=am, use_cache=False)
            tok_logp = _tok_logp(out.logits[:, :-1], tgt)
            with torch.no_grad():
                with peft_model.disable_adapter():
                    ref_out = model_call(input_ids=in_ids, attention_mask=am, use_cache=False)
                ref_tok_logp = _tok_logp(ref_out.logits[:, :-1], tgt)

            ratio = torch.exp(tok_logp - ref_tok_logp.detach())
            eps = cfg.epsilon_clip
            unclipped = ratio * adv_mb
            clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv_mb
            pg_per_tok = -torch.min(unclipped, clipped)
            kl_per_tok = tok_logp - ref_tok_logp
            denom = tm.sum(dim=-1).clamp(min=1.0)
            pg_per_seq = (pg_per_tok * tm).sum(dim=-1) / denom
            kl_per_seq = (kl_per_tok * tm).sum(dim=-1) / denom
            w = pg_per_seq.numel() / float(B)
            pg_chunk = pg_per_seq.mean() * w
            kl_chunk = kl_per_seq.mean() * w
            loss_chunk = pg_chunk + cfg.beta_kl * kl_chunk
            loss_chunk.backward()
            pg_acc = pg_acc + pg_chunk.detach()
            kl_acc = kl_acc + kl_chunk.detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(adapter_params, cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total["loss"] += float(pg_acc + cfg.beta_kl * kl_acc)
        total["pg"] += float(pg_acc)
        total["kl"] += float(kl_acc)
        total["grad_norm"] += float(grad_norm)

    # Multi-adapter training fragments the cache; release back to CUDA so the
    # next problem's add_adapter / save_pretrained doesn't bump into OOM.
    torch.cuda.empty_cache()
    ne = max(cfg.update_epochs, 1)
    return {
        "loss": total["loss"] / ne,
        "pg": total["pg"] / ne,
        "kl": total["kl"] / ne,
        "grad_norm": total["grad_norm"] / ne,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std(unbiased=False)) if rewards.numel() > 1 else 0.0,
        "advantage_mean": float(adv.mean()),
    }
