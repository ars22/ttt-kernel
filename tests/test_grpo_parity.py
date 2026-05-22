"""Smoke + determinism checks for trainer/grpo.py::grpo_step.

The step is REINFORCE with group-relative advantages and an optional
KL-to-reference penalty. These tests only assert that metrics are finite
and bit-exact across identical runs; numerical correctness of the math
is checked by inspection in src/ttt_kernel/trainer/grpo.py.
"""
from __future__ import annotations

import math

import pytest


def _have_torch():
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM  # noqa: F401
        from peft import LoraConfig, get_peft_model  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _have_torch(), reason="torch / transformers / peft not available"
)


def _build_tiny_model():
    """Smallest GPT2-style model that still has q/k/v/o projections.
    Build on CPU; we don't need a GPU for parity checks."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2", torch_dtype=torch.float32)
    lora_cfg = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["c_attn"], bias="none", task_type="CAUSAL_LM",
    )
    peft = get_peft_model(base, lora_cfg)
    peft.train()
    return peft, tok


def _make_rollouts(n: int):
    from ttt_kernel.trainer.grpo import RolloutT
    return [
        RolloutT(
            prompt="add two numbers",
            completion=f"def add(a, b): return a + b  # {i}",
            reward=float(i) * 0.5,
        )
        for i in range(n)
    ]


def test_grpo_step_returns_finite_metrics():
    import torch
    from ttt_kernel.trainer.grpo import GRPOStepCfg, grpo_step

    peft, tok = _build_tiny_model()
    device = torch.device("cpu")
    cfg = GRPOStepCfg(
        beta_kl=0.04,
        group_advantage_norm=True,
        update_epochs=1,
        grad_clip=1.0,
        micro_batch_size=2,
        max_seq_len=64,
    )

    # The trainer's adapter is named 'default' for vanilla get_peft_model.
    adapter_name = "default"
    params = [p for n, p in peft.named_parameters() if p.requires_grad and ".default." in n]
    assert params, "tiny model has no LoRA params"
    opt = torch.optim.AdamW(params, lr=1e-4)

    metrics = grpo_step(
        peft_model=peft,
        model_call=peft,
        optimizer=opt,
        tokenizer=tok,
        adapter_name=adapter_name,
        adapter_params=params,
        rollouts=_make_rollouts(4),
        cfg=cfg,
        device=device,
        group_ids=None,
    )
    for k in ("loss", "pg", "kl", "grad_norm", "reward_mean", "reward_std", "advantage_mean"):
        assert k in metrics, f"missing {k}"
        assert math.isfinite(metrics[k]), f"{k}={metrics[k]} not finite"


def test_grpo_step_deterministic_at_seed():
    """Two runs with identical init must produce identical metrics."""
    import torch
    from ttt_kernel.trainer.grpo import GRPOStepCfg, grpo_step

    def _run():
        torch.manual_seed(0)
        peft, tok = _build_tiny_model()
        device = torch.device("cpu")
        cfg = GRPOStepCfg(
            beta_kl=0.04,
            group_advantage_norm=True, update_epochs=1,
            grad_clip=1.0, micro_batch_size=2, max_seq_len=64,
        )
        params = [p for n, p in peft.named_parameters() if p.requires_grad and ".default." in n]
        opt = torch.optim.AdamW(params, lr=1e-4)
        return grpo_step(
            peft_model=peft, model_call=peft, optimizer=opt, tokenizer=tok,
            adapter_name="default", adapter_params=params,
            rollouts=_make_rollouts(4), cfg=cfg, device=device, group_ids=None,
        )

    a = _run()
    b = _run()
    # Bit-exact match across runs.
    for k in a:
        assert math.isclose(a[k], b[k], rel_tol=0, abs_tol=1e-12), f"{k}: {a[k]} vs {b[k]}"


def test_advantage_normalization_per_group():
    """Advantages should normalize within each group separately."""
    import torch
    from ttt_kernel.trainer.grpo import _group_advantages

    device = torch.device("cpu")
    rewards = torch.tensor([1.0, 3.0, -1.0, 1.0], device=device)  # groups [0,0,1,1]
    adv = _group_advantages(rewards, [0, 0, 1, 1], norm=True, device=device)
    # Group 0: mean=2, std=1 → (-1, 1).  Group 1: mean=0, std=1 → (-1, 1).
    assert torch.allclose(adv, torch.tensor([-1.0, 1.0, -1.0, 1.0]), atol=1e-6)
