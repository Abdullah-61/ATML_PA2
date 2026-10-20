from __future__ import annotations

import torch

from common.data import prompt_messages
from common.generation import batch_generate
from common.models import token_values


def filter_prompts(tokenizer, rows, max_prompt_length):
    """Deterministically skip prompts longer than max_prompt_length (no truncation)."""
    kept, skipped = [], []
    for i, row in enumerate(rows):
        ids = tokenizer.apply_chat_template(prompt_messages(row), tokenize=True, add_generation_prompt=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        (kept if len(ids) <= max_prompt_length else skipped).append(i)
    return kept, skipped


def set_ppo_mode(model):
    """Train mode with every Dropout (incl. LoRA dropout) off, so pi_theta == pi_old at epoch 1."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()


def trainable_to_fp32(model):
    """Master weights for trainable params in fp32 (fp16 AdamW states underflow -> NaN)."""
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def generate_rollout(model, tok, prompts, cfg, max_new):
    g = cfg["generation"]
    prev = model.config.use_cache
    model.eval()
    model.config.use_cache = True
    try:
        gen = batch_generate(model, tok, prompts, int(cfg["max_prompt_length"]), max_new,
                             g["temperature"], g["top_p"], g["do_sample"])
        return {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in gen.items()}
    finally:
        model.config.use_cache = prev


def response_logprobs(model, sequences, attention_mask, prompt_width, response_ids,
                      grad=False, entropy=False, chunk_size=1):
    """Log-prob of each response token under `model` (and optionally the token entropy).

    sequences = [prompt (width prompt_width) | response (width W)].  Returns [B, W] tensors.
    """
    W = response_ids.shape[1]
    assert sequences.shape[1] == prompt_width + W, "sequences must be prompt_width + response width"
    lps, ents = [], []
    for s in range(0, sequences.shape[0], chunk_size):
        e = min(s + chunk_size, sequences.shape[0])
        with torch.set_grad_enabled(grad), _autocast(sequences.device):
            out = model(input_ids=sequences[s:e], attention_mask=attention_mask[s:e],
                        use_cache=False, return_dict=True, logits_to_keep=W + 1)
        logits = out.logits[:, :-1, :].float()  # positions that predict the W response tokens
        del out
        logp = torch.log_softmax(logits, dim=-1)
        del logits
        lps.append(logp.gather(-1, response_ids[s:e].unsqueeze(-1)).squeeze(-1))
        if entropy:
            with torch.no_grad():
                ents.append(-(logp.exp() * logp).sum(-1))
        del logp
    return torch.cat(lps, 0), (torch.cat(ents, 0) if entropy else None)


def response_values(value_model, sequences, attention_mask, prompt_width, n_resp):
    with _autocast(sequences.device):
        v = token_values(value_model, sequences, attention_mask)
    v = v.float()
    return v[:, prompt_width - 1: prompt_width - 1 + n_resp]


def optimizer_step(scaler, opt, params, max_grad_norm):
    """Unscale, clip, step (the scaler skips the step on inf/nan grads). Returns (grad_norm, stepped)."""
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    stepped = scaler.get_scale() >= before  # scale shrinks only when a step was skipped
    return float(gn), bool(stepped)
