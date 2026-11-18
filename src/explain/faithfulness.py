"""Faithfulness, counterfactual consistency, and concept-level attribution auditing."""

from typing import Dict, List, Tuple

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients

from src.data.concepts import CONCEPT_NAMES


def compute_concept_token_attribution(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    target_dim: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute signed gradient*concept token attributions and concept-axis contributions."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    sent_mask = batch["sent_mask"].to(device)
    asp_mask = batch["asp_mask"].to(device)
    rel_dist = batch["rel_dist"].to(device)

    with torch.no_grad():
        out_init = model(input_ids, attention_mask, token_type_ids, sent_mask, asp_mask, rel_dist)
        c_init = out_init["c_seq"].detach()

    c_var = c_init.clone().detach().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        out = model(
            input_ids,
            attention_mask,
            token_type_ids,
            sent_mask,
            asp_mask,
            rel_dist,
            override_c_seq=c_var,
        )
        target_val = out["pred"][:, target_dim].sum()
        model.zero_grad(set_to_none=True)
        target_val.backward()

    grad_c = c_var.grad.detach()
    # [B, T, K] signed contribution of each concept at each token
    signed_ck = (c_var.detach() * grad_c).cpu().numpy()
    # [B, T] token importance magnitude (zeroed outside sentence)
    tok_scores = np.abs(signed_ck).sum(axis=-1) * sent_mask.cpu().numpy()
    # [B, K] sentence-aggregated signed concept contribution
    concept_contrib = signed_ck.sum(axis=1)
    return tok_scores, concept_contrib


def compute_integrated_gradients_attribution(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    target_dim: int = 0,
    n_steps: int = 16,
) -> np.ndarray:
    """Compute post-hoc Layer Integrated Gradients over backbone word embeddings."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    sent_mask = batch["sent_mask"].to(device)
    asp_mask = batch["asp_mask"].to(device)
    rel_dist = batch["rel_dist"].to(device)

    # Locate word embedding layer across PEFT-wrapped or raw HuggingFace encoders
    base_enc = model.encoder.base_model.model if hasattr(model.encoder, "base_model") else model.encoder
    emb_layer = base_enc.embeddings.word_embeddings

    def forward_func(ids, attn_m, tt_ids, s_mask, a_mask, r_dist):
        out = model(ids, attn_m, tt_ids, s_mask, a_mask, r_dist)
        return out["pred"][:, target_dim]

    lig = LayerIntegratedGradients(forward_func, emb_layer)
    eff_t = int(attention_mask.sum(dim=1).max().item())
    eff_t = max(2, min(input_ids.shape[1], eff_t))
    in_trim = input_ids[:, :eff_t]
    base_trim = torch.zeros_like(in_trim)
    add_trim = (
        attention_mask[:, :eff_t],
        token_type_ids[:, :eff_t],
        sent_mask[:, :eff_t],
        asp_mask[:, :eff_t],
        rel_dist[:, :eff_t],
    )
    with torch.backends.cudnn.flags(enabled=False):
        attrs = lig.attribute(
            inputs=in_trim,
            baselines=base_trim,
            additional_forward_args=add_trim,
            n_steps=n_steps,
            internal_batch_size=max(1, in_trim.shape[0]),
        )
    scores_trim = attrs.abs().sum(dim=-1) * sent_mask[:, :eff_t]
    full_scores = torch.zeros_like(sent_mask)
    full_scores[:, :eff_t] = scores_trim
    return full_scores.detach().cpu().numpy()


@torch.no_grad()
def evaluate_faithfulness_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    token_scores: np.ndarray,
    pad_token_id: int = 0,
    mask_token_id: int = 103,
    target_dim: int = 0,
    top_fracs: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
) -> Dict[str, List[float]]:
    """Evaluate Sufficiency, Comprehensiveness, Deletion and Insertion shifts on continuous predictions."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    sent_mask = batch["sent_mask"].to(device)
    asp_mask = batch["asp_mask"].to(device)
    rel_dist = batch["rel_dist"].to(device)

    y_full = model(input_ids, attention_mask, token_type_ids, sent_mask, asp_mask, rel_dist)["pred"][:, target_dim]
    y_full_np = y_full.cpu().numpy()

    B, T = input_ids.shape
    sent_mask_np = sent_mask.cpu().numpy()

    suff_scores = {f: [] for f in top_fracs}
    comp_scores = {f: [] for f in top_fracs}

    for f in top_fracs:
        comp_ids = input_ids.clone()
        suff_ids = input_ids.clone()
        for b in range(B):
            valid_idx = np.where(sent_mask_np[b] > 0.5)[0]
            if len(valid_idx) < 2:
                continue
            k = max(1, int(round(len(valid_idx) * f)))
            ranked = valid_idx[np.argsort(-token_scores[b, valid_idx])]
            top_idx = ranked[:k]
            rest_idx = ranked[k:]
            comp_ids[b, top_idx] = mask_token_id
            suff_ids[b, rest_idx] = mask_token_id

        y_comp = model(comp_ids, attention_mask, token_type_ids, sent_mask, asp_mask, rel_dist)["pred"][:, target_dim].cpu().numpy()
        y_suff = model(suff_ids, attention_mask, token_type_ids, sent_mask, asp_mask, rel_dist)["pred"][:, target_dim].cpu().numpy()

        comp_scores[f].extend(np.abs(y_full_np - y_comp).tolist())
        suff_scores[f].extend(np.abs(y_full_np - y_suff).tolist())

    return {
        "comp": [float(np.mean(comp_scores[f])) for f in top_fracs],
        "suff": [float(np.mean(suff_scores[f])) for f in top_fracs],
    }


@torch.no_grad()
def evaluate_concept_counterfactual(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    concept_idx: int,
    new_val: float = 0.0,
    target_dim: int = 0,
) -> np.ndarray:
    """Intervene directly on a concept axis inside c_seq and return delta prediction."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    sent_mask = batch["sent_mask"].to(device)
    asp_mask = batch["asp_mask"].to(device)
    rel_dist = batch["rel_dist"].to(device)

    out_orig = model(input_ids, attention_mask, token_type_ids, sent_mask, asp_mask, rel_dist)
    c_mod = out_orig["c_seq"].clone()
    c_mod[:, :, concept_idx] = new_val * sent_mask
    out_cf = model(
        input_ids, attention_mask, token_type_ids, sent_mask, asp_mask, rel_dist, override_c_seq=c_mod
    )
    delta = (out_cf["pred"][:, target_dim] - out_orig["pred"][:, target_dim]).cpu().numpy()
    return delta
