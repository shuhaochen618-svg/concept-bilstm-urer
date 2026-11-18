"""Unified neural architectures for continuous sentiment regression and ablations."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel


class ResidualBiLSTMEncoder(nn.Module):
    """Multi-layer residual BiLSTM with LayerNorm and Dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=hidden_dim * 2,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim * 2) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, input_dim], mask: [B, T]
        h = self.in_proj(x) * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
        T = x.shape[1]
        last_fw_bw_diff = torch.zeros(x.shape[0], T, device=x.device, dtype=x.dtype)
        half = h.shape[-1] // 2
        for lstm, norm in zip(self.layers, self.norms):
            packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            out_packed, _ = lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
            out = self.drop(out)
            h = norm(h + out) * mask.unsqueeze(-1)
            # Track forward vs backward hidden discrepancy for rhetorical-turn diagnostics
            last_fw_bw_diff = (out[:, :, :half] - out[:, :, half:]).norm(dim=-1) * mask
        return h, last_fw_bw_diff


class EvidentialHead(nn.Module):
    """Normal-Inverse-Gamma (NIG) evidential regression head."""

    def __init__(self, in_dim: int, out_dim: int, scale_min: float = 1.0, scale_max: float = 9.0) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.proj = nn.Linear(in_dim, out_dim * 4)

    def forward(self, z: torch.Tensor, gamma_bias: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        raw = self.proj(z).view(z.shape[0], self.out_dim, 4)
        gamma_raw, nu_raw, alpha_raw, beta_raw = raw.unbind(dim=-1)
        if gamma_bias is not None:
            gamma_raw = gamma_raw + gamma_bias
        # Slightly padded smooth range avoids saturation near exact boundary labels
        pad = 0.25 * (self.scale_max - self.scale_min) / 8.0
        lo = self.scale_min - pad
        hi = self.scale_max + pad
        gamma = lo + (hi - lo) * torch.sigmoid(gamma_raw)
        nu = F.softplus(nu_raw) + 1e-4
        alpha = 1.0 + F.softplus(alpha_raw) + 1e-4
        beta = F.softplus(beta_raw) + 1e-4

        ale_var = beta / (alpha - 1.0).clamp(min=1e-4)
        epi_var = ale_var / nu.clamp(min=1e-4)
        tot_std = torch.sqrt((ale_var + epi_var).clamp(min=1e-6))
        return {
            "pred": gamma.clamp(self.scale_min, self.scale_max),
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
            "ale_var": ale_var,
            "epi_var": epi_var,
            "tot_std": tot_std,
        }


class DeterministicHead(nn.Module):
    """Standard deterministic regression head for baselines."""

    def __init__(self, in_dim: int, out_dim: int, scale_min: float = 1.0, scale_max: float = 9.0) -> None:
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.proj(z)
        pad = 0.25 * (self.scale_max - self.scale_min) / 8.0
        lo = self.scale_min - pad
        hi = self.scale_max + pad
        gamma = lo + (hi - lo) * torch.sigmoid(raw)
        return {"pred": gamma.clamp(self.scale_min, self.scale_max), "gamma": gamma}


class PLMAspectSentimentModel(nn.Module):
    """Unified framework supporting proposed Concept-BiLSTM-URER and all ablations."""

    def __init__(
        self,
        backbone_name: str,
        out_dim: int = 2,
        arch_mode: str = "concept_bilstm",
        head_mode: str = "evidential",
        use_lora: bool = True,
        lora_rank: int = 8,
        k_exp: int = 8,
        k_lat: int = 4,
        bilstm_hidden: int = 64,
        bilstm_layers: int = 2,
        dropout: float = 0.2,
        max_rel_dist: int = 64,
        pos_dim: int = 32,
        scale_min: float = 1.0,
        scale_max: float = 9.0,
        ib_dim: int = 16,
    ) -> None:
        super().__init__()
        self.arch_mode = arch_mode
        self.head_mode = head_mode
        self.out_dim = out_dim
        self.k_exp = k_exp
        self.k_lat = k_lat
        self.k_total = k_exp + k_lat
        self.scale_min = scale_min
        self.scale_max = scale_max

        self.encoder = AutoModel.from_pretrained(backbone_name)
        if use_lora:
            mod_names = [n for n, _ in self.encoder.named_modules()]
            if any("query_proj" in n for n in mod_names):
                targets = ["query_proj", "value_proj"]
            else:
                targets = ["query", "value"]
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                lora_dropout=0.05,
                target_modules=targets,
            )
            self.encoder = get_peft_model(self.encoder, lora_cfg)

        d_enc = self.encoder.config.hidden_size
        self.use_token_types = getattr(self.encoder.config, "type_vocab_size", 1) > 1
        self.rel_pos_emb = nn.Embedding(max_rel_dist * 2 + 2, pos_dim)
        self.enc_drop = nn.Dropout(dropout)

        if arch_mode in ("concept_bilstm", "concept_bilstm_causal", "ib_residual_bilstm", "concept_no_bilstm"):
            self.concept_proj = nn.Sequential(
                nn.Linear(d_enc * 2 + pos_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, self.k_total),
            )
            with torch.no_grad():
                self.concept_proj[-1].bias[: self.k_exp].fill_(-2.0)
        if arch_mode == "concept_bilstm_causal":
            self.causal_pos_raw = nn.Parameter(torch.tensor(0.5))
            self.causal_neg_raw = nn.Parameter(torch.tensor(0.5))
        if arch_mode in ("concept_bilstm", "concept_bilstm_causal", "ib_residual_bilstm"):
            self.seq_net = ResidualBiLSTMEncoder(self.k_total, bilstm_hidden, bilstm_layers, dropout)
            feat_dim = bilstm_hidden * 2
        elif arch_mode == "plm_bilstm_no_concept":
            self.seq_net = ResidualBiLSTMEncoder(d_enc + pos_dim, bilstm_hidden, bilstm_layers, dropout)
            feat_dim = bilstm_hidden * 2
        elif arch_mode == "concept_no_bilstm":
            self.concept_mlp = nn.Sequential(
                nn.Linear(self.k_total, bilstm_hidden * 2),
                nn.LayerNorm(bilstm_hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            feat_dim = bilstm_hidden * 2
        elif arch_mode == "plm_only":
            feat_dim = d_enc * 2
        else:
            raise ValueError(f"Unknown arch_mode: {arch_mode}")

        # Attention pooling strictly inside the active feature space (prevents concept bypass)
        self.attn_score = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.Tanh(),
            nn.Linear(feat_dim // 2, 1),
        )

        head_in_dim = feat_dim
        if arch_mode == "ib_residual_bilstm":
            self.ib_mu = nn.Linear(d_enc * 2, ib_dim)
            self.ib_logvar = nn.Linear(d_enc * 2, ib_dim)
            head_in_dim += ib_dim

        if head_mode == "evidential":
            self.head = EvidentialHead(head_in_dim, out_dim, scale_min, scale_max)
        elif head_mode == "deterministic":
            self.head = DeterministicHead(head_in_dim, out_dim, scale_min, scale_max)
        else:
            raise ValueError(f"Unknown head_mode: {head_mode}")

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        sent_mask: torch.Tensor,
        asp_mask: torch.Tensor,
        rel_dist: torch.Tensor,
        override_c_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.use_token_types:
            enc_kwargs["token_type_ids"] = token_type_ids
        H = self.encoder(**enc_kwargs).last_hidden_state  # [B, T, d_enc]
        H = self.enc_drop(H)

        # Aspect query representation (fallback to CLS if asp_mask is empty)
        asp_sum = asp_mask.sum(dim=1, keepdim=True)
        h_asp = torch.where(
            asp_sum > 0.5,
            self._masked_mean(H, asp_mask),
            H[:, 0, :],
        )

        extras: Dict[str, torch.Tensor] = {}
        if self.arch_mode == "plm_only":
            h_sent = self._masked_mean(H, sent_mask)
            z_final = torch.cat([h_sent, h_asp], dim=-1)
            attn_w = sent_mask / sent_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            pos_emb = self.rel_pos_emb(rel_dist)
            if self.arch_mode in ("concept_bilstm", "concept_bilstm_causal", "ib_residual_bilstm", "concept_no_bilstm"):
                if override_c_seq is not None:
                    c_seq = override_c_seq
                else:
                    h_asp_exp = h_asp.unsqueeze(1).expand(-1, H.shape[1], -1)
                    c_in = torch.cat([H, h_asp_exp, pos_emb], dim=-1)
                    c_raw = self.concept_proj(c_in)
                    c_exp = torch.sigmoid(c_raw[:, :, : self.k_exp])
                    if self.k_lat > 0:
                        c_lat = 0.25 * torch.tanh(c_raw[:, :, self.k_exp :])
                        c_seq = torch.cat([c_exp, c_lat], dim=-1)
                    else:
                        c_seq = c_exp
                    c_seq = c_seq * sent_mask.unsqueeze(-1)
                extras["c_seq"] = c_seq

                if self.arch_mode in ("concept_bilstm", "concept_bilstm_causal", "ib_residual_bilstm"):
                    seq_feats, fw_bw_diff = self.seq_net(c_seq, sent_mask)
                    extras["fw_bw_diff"] = fw_bw_diff
                else:
                    seq_feats = self.concept_mlp(c_seq) * sent_mask.unsqueeze(-1)
            else:
                # plm_bilstm_no_concept
                seq_in = torch.cat([H, pos_emb], dim=-1)
                seq_feats, fw_bw_diff = self.seq_net(seq_in, sent_mask)
                extras["fw_bw_diff"] = fw_bw_diff

            logits = self.attn_score(seq_feats).squeeze(-1)
            logits = logits.masked_fill(sent_mask < 0.5, -1e4)
            attn_w = torch.softmax(logits, dim=-1) * sent_mask
            attn_w = attn_w / attn_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            z_seq = torch.bmm(attn_w.unsqueeze(1), seq_feats).squeeze(1)

            if self.arch_mode == "ib_residual_bilstm":
                h_sent = self._masked_mean(H, sent_mask)
                ib_in = torch.cat([h_sent, h_asp], dim=-1)
                mu_ib = self.ib_mu(ib_in)
                logvar_ib = self.ib_logvar(ib_in).clamp(-10.0, 4.0)
                if self.training:
                    std_ib = torch.exp(0.5 * logvar_ib)
                    z_ib = mu_ib + std_ib * torch.randn_like(std_ib)
                else:
                    z_ib = mu_ib
                ib_kl = -0.5 * (1.0 + logvar_ib - mu_ib.pow(2) - logvar_ib.exp()).sum(dim=-1).mean()
                extras["ib_kl"] = ib_kl
                extras["z_seq_norm"] = z_seq.norm(dim=-1).mean().detach()
                extras["z_ib_norm"] = z_ib.norm(dim=-1).mean().detach()
                z_final = torch.cat([z_seq, z_ib], dim=-1)
            else:
                z_final = z_seq

        gamma_bias = None
        if self.arch_mode == "concept_bilstm_causal" and self.out_dim <= 2 and "c_seq" in extras:
            c_s = extras["c_seq"]
            sqrt_len = sent_mask.sum(dim=1).clamp(min=1.0).sqrt()
            pos_cue = (c_s[:, :, 1] * sent_mask).sum(dim=1) / sqrt_len
            neg_cue = (c_s[:, :, 2] * sent_mask).sum(dim=1) / sqrt_len
            w_pos = F.softplus(self.causal_pos_raw) + 0.5
            w_neg = F.softplus(self.causal_neg_raw) + 0.5
            delta_v = w_pos * pos_cue - w_neg * neg_cue
            gamma_bias = torch.zeros(z_final.shape[0], self.out_dim, device=z_final.device, dtype=z_final.dtype)
            gamma_bias[:, 0] = delta_v

        if self.head_mode == "evidential" and gamma_bias is not None:
            out = self.head(z_final, gamma_bias=gamma_bias)
        else:
            out = self.head(z_final)
        out["attn_weights"] = attn_w
        out.update(extras)
        return out


class ClassicRNNRegressor(nn.Module):
    """Original proposal baseline: Embedding + Residual BiLSTM/GRU + LayerNorm + Attention."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = "bilstm",
        out_dim: int = 2,
        scale_min: float = 1.0,
        scale_max: float = 9.0,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.rnn_type = rnn_type
        if rnn_type == "bilstm":
            self.seq = ResidualBiLSTMEncoder(embed_dim, hidden_dim, num_layers, dropout)
            feat_dim = hidden_dim * 2
        elif rnn_type == "bigru":
            self.gru = nn.GRU(
                embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden_dim * 2)
            feat_dim = hidden_dim * 2
        elif rnn_type == "cnn":
            self.convs = nn.ModuleList([nn.Conv1d(embed_dim, hidden_dim, k, padding=k // 2) for k in (2, 3, 4)])
            self.norm = nn.LayerNorm(hidden_dim * 3)
            feat_dim = hidden_dim * 3
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.attn = nn.Sequential(nn.Linear(feat_dim, feat_dim // 2), nn.Tanh(), nn.Linear(feat_dim // 2, 1))
        self.head = DeterministicHead(feat_dim, out_dim, scale_min, scale_max)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        mask = attention_mask.float()
        x = self.drop(self.emb(input_ids))
        if self.rnn_type == "bilstm":
            h, _ = self.seq(x, mask)
        elif self.rnn_type == "bigru":
            h, _ = self.gru(x)
            h = self.norm(h) * mask.unsqueeze(-1)
        else:
            xt = x.transpose(1, 2)
            cs = [F.relu(conv(xt))[:, :, : x.shape[1]] for conv in self.convs]
            h = self.norm(torch.cat(cs, dim=1).transpose(1, 2)) * mask.unsqueeze(-1)
        logits = self.attn(h).squeeze(-1).masked_fill(mask < 0.5, -1e4)
        w = torch.softmax(logits, dim=-1)
        z = torch.bmm(w.unsqueeze(1), h).squeeze(1)
        out = self.head(z)
        out["attn_weights"] = w
        return out


class EvidentialConformalCalibrator:
    """Split Conformal Prediction wrapper using heteroscedastic evidential std scaling."""

    def __init__(self, alpha: float = 0.10, scale_min: float = 1.0, scale_max: float = 9.0, delta: float = 1e-4) -> None:
        self.alpha = alpha
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.delta = delta
        self.q_hats: Optional[np.ndarray] = None

    def fit(self, y_cal: np.ndarray, y_pred_cal: np.ndarray, std_cal: np.ndarray) -> np.ndarray:
        n = len(y_cal)
        scores = np.abs(y_cal - y_pred_cal) / (std_cal + self.delta)
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / max(1, n))
        self.q_hats = np.quantile(scores, q_level, axis=0, method="higher")
        return self.q_hats

    def predict(self, y_pred: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.q_hats is not None, "Calibrator must be fit before calling predict."
        radius = self.q_hats[None, :] * (std + self.delta)
        lower = np.clip(y_pred - radius, self.scale_min, self.scale_max)
        upper = np.clip(y_pred + radius, self.scale_min, self.scale_max)
        return lower, upper
