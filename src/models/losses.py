"""Evidential, anti-collapse CCC, uncertainty-weighted, and concept bottleneck losses."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def nig_nll_loss(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Student-t marginal negative log-likelihood for Normal-Inverse-Gamma prior (float32)."""
    with torch.amp.autocast("cuda", enabled=False):
        y = y.float()
        gamma = gamma.float()
        nu = nu.float().clamp(min=1e-4)
        alpha = alpha.float().clamp(min=1.0 + 1e-4)
        beta = beta.float().clamp(min=1e-4)

        two_beta_lambda = (2.0 * beta * (1.0 + nu)).clamp(min=1e-6)
        nll = (
            0.5 * torch.log(torch.pi / nu)
            - alpha * torch.log(two_beta_lambda)
            + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + two_beta_lambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
    return nll


def evidential_reg_loss(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    lambda_u_consist: float = 0.05,
) -> torch.Tensor:
    """Evidence regularizer plus log-variance consistency to eliminate dead zones."""
    with torch.amp.autocast("cuda", enabled=False):
        y = y.float()
        gamma = gamma.float()
        nu = nu.float().clamp(min=1e-4)
        alpha = alpha.float().clamp(min=1.0 + 1e-4)
        beta = beta.float().clamp(min=1e-4)

        err = torch.abs(y - gamma)
        reg_evid = err * (2.0 * nu + alpha)

        # Smooth log-variance consistency prevents zero-evidence gradient collapse on outliers
        tot_var = (beta * (1.0 + nu)) / (nu * (alpha - 1.0))
        target_sq = (err.detach() ** 2).clamp(min=1e-4)
        reg_var = F.smooth_l1_loss(
            torch.log(tot_var.clamp(min=1e-6)),
            torch.log(target_sq),
            reduction="none",
        )
    return reg_evid + lambda_u_consist * reg_var


def batch_ccc_loss(y: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Differentiable Concordance Correlation Coefficient loss across batch and targets."""
    with torch.amp.autocast("cuda", enabled=False):
        y = y.float()
        pred = pred.float()
        if y.shape[0] < 3:
            return F.mse_loss(pred, y)
        mu_y = y.mean(dim=0)
        mu_p = pred.mean(dim=0)
        var_y = ((y - mu_y) ** 2).mean(dim=0)
        var_p = ((pred - mu_p) ** 2).mean(dim=0)
        cov = ((y - mu_y) * (pred - mu_p)).mean(dim=0)
        ccc = (2.0 * cov) / (var_y + var_p + (mu_y - mu_p) ** 2 + 1e-6)
        return (1.0 - ccc).mean()


def concept_losses(
    c_seq: torch.Tensor,
    c_priors: torch.Tensor,
    sent_mask: torch.Tensor,
    k_exp: int = 8,
    use_causal_constraints: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute weak alignment for explicit concepts and sparsity+orthogonality for latent concepts."""
    with torch.amp.autocast("cuda", enabled=False):
        c_seq = c_seq.float()
        c_priors = c_priors.float()
        mask = sent_mask.float().unsqueeze(-1)  # [B, T, 1]
        denom = mask.sum().clamp(min=1.0)

        c_exp = c_seq[:, :, :k_exp]
        # Per-channel class-balanced BCE ensures rare cues (Negation/Adversative/NegCue) are never starved
        align_err = F.binary_cross_entropy(c_exp.clamp(1e-5, 1.0 - 1e-5), c_priors, reduction="none")
        pos_m = (c_priors > 0.5).float() * mask
        neg_m = (c_priors <= 0.5).float() * mask
        pos_cnt_k = pos_m.sum(dim=(0, 1))
        neg_cnt_k = neg_m.sum(dim=(0, 1))
        pos_bce_k = (align_err * pos_m).sum(dim=(0, 1)) / pos_cnt_k.clamp(min=1.0)
        neg_bce_k = (align_err * neg_m).sum(dim=(0, 1)) / neg_cnt_k.clamp(min=1.0)
        has_pos_k = (pos_cnt_k > 0.5).float()
        has_neg_k = (neg_cnt_k > 0.5).float()
        pos_bce = (pos_bce_k * has_pos_k).sum() / has_pos_k.sum().clamp(min=1.0)
        neg_bce = (neg_bce_k * has_neg_k).sum() / has_neg_k.sum().clamp(min=1.0)
        loss_align = 0.5 * (pos_bce + neg_bce)

        if c_seq.shape[-1] > k_exp:
            c_lat = c_seq[:, :, k_exp:]
            loss_sparse = (torch.abs(c_lat) * mask).sum() / (denom * c_lat.shape[-1])
            flat_lat = (c_lat * mask).reshape(-1, c_lat.shape[-1])
            norms = flat_lat.norm(dim=0, keepdim=True).clamp(min=1e-6)
            normed = flat_lat / norms
            gram = normed.t() @ normed
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            loss_ortho = ((gram - eye) ** 2).mean()
            if use_causal_constraints:
                # Zero-safe cross-orthogonality (c_lat _|_ c_exp) + non-prior background suppression
                flat_exp = (c_exp * mask).reshape(-1, c_exp.shape[-1])
                exp_norms = flat_exp.norm(dim=0, keepdim=True)
                lat_norms = flat_lat.norm(dim=0, keepdim=True)
                norm_exp = torch.where(exp_norms > 1e-4, flat_exp / exp_norms.clamp(min=1e-4), torch.zeros_like(flat_exp))
                norm_lat = torch.where(lat_norms > 1e-4, flat_lat / lat_norms.clamp(min=1e-4), torch.zeros_like(flat_lat))
                cross_gram = norm_lat.t() @ norm_exp
                loss_ortho = loss_ortho + 2.0 * (cross_gram ** 2).mean()
        else:
            loss_sparse = torch.zeros((), device=c_seq.device)
            loss_ortho = torch.zeros((), device=c_seq.device)

        if use_causal_constraints:
            bg_mask = neg_m
            loss_bg = (c_exp * bg_mask).sum() / bg_mask.sum().clamp(min=1.0)
            loss_sparse = loss_sparse + 2.0 * loss_bg

    return {
        "concept_align": loss_align,
        "concept_sparse": loss_sparse,
        "concept_ortho": loss_ortho,
    }


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    head_mode: str = "evidential",
    use_concepts: bool = True,
    k_exp: int = 8,
    lambda_ccc: float = 0.5,
    lambda_reg: float = 0.05,
    lambda_concept_align: float = 0.15,
    lambda_concept_sparse: float = 0.02,
    lambda_concept_ortho: float = 0.05,
    lambda_ib: float = 0.01,
    eta_epi: float = 0.5,
    eta_ale: float = 0.5,
    use_dynamic_weight: bool = True,
    epoch: int = 0,
    warmup_epochs: int = 2,
    use_ccc: bool = True,
    use_causal_constraints: bool = False,
) -> Dict[str, torch.Tensor]:
    """Unified multi-component loss supporting proposed model and all ablations."""
    y = batch["targets"]
    lds_w = batch["lds_weight"]
    pred = outputs["pred"]

    if head_mode == "evidential":
        gamma = outputs["gamma"]
        nu = outputs["nu"]
        alpha = outputs["alpha"]
        beta = outputs["beta"]

        nll = nig_nll_loss(y, gamma, nu, alpha, beta).mean()
        ev_reg = evidential_reg_loss(y, gamma, nu, alpha, beta).mean()

        with torch.no_grad():
            ale_var = beta.float() / (alpha.float() - 1.0).clamp(min=1e-4)
            epi_var = ale_var / nu.float().clamp(min=1e-4)
            ale_s = ale_var.mean(dim=-1)
            epi_s = epi_var.mean(dim=-1)
            if use_dynamic_weight and epoch >= warmup_epochs:
                ale_norm = ale_s / ale_s.mean().clamp(min=1e-6)
                epi_norm = epi_s / epi_s.mean().clamp(min=1e-6)
                w = lds_w.float() * (1.0 + eta_epi * epi_norm) / (1.0 + eta_ale * ale_norm)
            else:
                w = lds_w.float()
            w = (w / w.mean().clamp(min=1e-6)).clamp(0.25, 4.0)

        huber = F.smooth_l1_loss(gamma.float(), y.float(), reduction="none").mean(dim=-1)
        reg_aux = (w * huber).mean()
        loss_main = nll + lambda_reg * ev_reg + reg_aux
    else:
        nll = torch.zeros((), device=y.device)
        ev_reg = torch.zeros((), device=y.device)
        w = lds_w.float()
        w = (w / w.mean().clamp(min=1e-6)).clamp(0.25, 4.0)
        mse_item = F.mse_loss(pred.float(), y.float(), reduction="none").mean(dim=-1)
        loss_main = (w * mse_item).mean()

    loss_ccc = batch_ccc_loss(y, pred) if use_ccc else torch.zeros((), device=y.device)
    total = loss_main + lambda_ccc * loss_ccc

    c_align = torch.zeros((), device=y.device)
    c_sparse = torch.zeros((), device=y.device)
    c_ortho = torch.zeros((), device=y.device)
    if use_concepts and "c_seq" in outputs:
        cl = concept_losses(
            outputs["c_seq"],
            batch["concept_priors"],
            batch["sent_mask"],
            k_exp=k_exp,
            use_causal_constraints=use_causal_constraints,
        )
        c_align = cl["concept_align"]
        c_sparse = cl["concept_sparse"]
        c_ortho = cl["concept_ortho"]
        total = (
            total
            + lambda_concept_align * c_align
            + lambda_concept_sparse * c_sparse
            + lambda_concept_ortho * c_ortho
        )

    if "ib_kl" in outputs:
        total = total + lambda_ib * outputs["ib_kl"]

    return {
        "total": total,
        "nll": nll.detach(),
        "ev_reg": ev_reg.detach(),
        "ccc": loss_ccc.detach(),
        "c_align": c_align.detach(),
        "c_sparse": c_sparse.detach(),
        "c_ortho": c_ortho.detach(),
    }
