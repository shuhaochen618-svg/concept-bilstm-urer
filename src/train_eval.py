"""Unified training, conformal calibration, robustness, and explainability runner."""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.concepts import CONCEPT_NAMES, to_simplified
from src.data.dataset import SentimentRegressionDataset, compute_lds_weights
from src.explain.faithfulness import (
    compute_concept_token_attribution,
    compute_integrated_gradients_attribution,
    evaluate_concept_counterfactual,
    evaluate_faithfulness_batch,
)
from src.models.architectures import (
    ClassicRNNRegressor,
    EvidentialConformalCalibrator,
    PLMAspectSentimentModel,
)
from src.models.losses import compute_total_loss
from src.utils.metrics import (
    concordance_correlation_coefficient,
    interval_calibration_metrics,
    regression_metrics_va,
    residual_ks_tests,
    safe_pearson,
    tail_rmse,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = PROJECT_ROOT / "data" / "processed"
OUT_ROOT = PROJECT_ROOT / "outputs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_task_splits(
    task: str,
    train_frac: float = 1.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Tuple[float, float], bool]:
    """Load train_sub, cal, dev, test DataFrames without leakage."""
    if task.startswith("dimabsa_"):
        df = pd.read_parquet(DATA_PROC / "dimabsa2026_zho_all.parquet")
        target_cols = ["valence", "arousal"]
        bounds = (1.0, 9.0)
        has_aspect = True
        if task == "dimabsa_cross_fin":
            src = df[df["domain"].isin(["zho_rest", "zho_lap"])].copy()
            tgt = df[df["domain"] == "zho_fin"].copy()
            train_df = src[src["split_cal"] == "train_sub"].copy()
            cal_df = src[src["split_cal"] == "cal"].copy()
            dev_df = tgt[tgt["split"] == "dev"].copy()
            test_df = tgt[tgt["split"] == "test"].copy()
        else:
            dom = task.replace("dimabsa_", "")
            sub = df[df["domain"] == dom].copy()
            train_df = sub[sub["split_cal"] == "train_sub"].copy()
            cal_df = sub[sub["split_cal"] == "cal"].copy()
            dev_df = sub[sub["split"] == "dev"].copy()
            test_df = sub[sub["split"] == "test"].copy()
    elif task == "brighter_chn":
        df = pd.read_parquet(DATA_PROC / "brighter_chn_intensities.parquet")
        target_cols = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        bounds = (0.0, 3.0)
        has_aspect = False
        train_df = df[df["split_cal"] == "train_sub"].copy()
        cal_df = df[df["split_cal"] == "cal"].copy()
        dev_df = df[df["split"] == "dev"].copy()
        test_df = df[df["split"] == "test"].copy()
    elif task == "logic_sentiment":
        df = pd.read_parquet(DATA_PROC / "logic_sentiment_zho.parquet")
        df["target_score"] = df["label"].astype(np.float32) * 8.0 + 1.0  # map 0/1 -> [1, 9]
        target_cols = ["target_score"]
        bounds = (1.0, 9.0)
        has_aspect = False
        rng = np.random.default_rng(seed)
        tr_all = df[df["split"] == "train"].copy()
        idx = np.arange(len(tr_all))
        rng.shuffle(idx)
        n_cal = max(1, int(round(0.15 * len(idx))))
        cal_df = tr_all.iloc[idx[:n_cal]].copy()
        train_df = tr_all.iloc[idx[n_cal:]].copy()
        dev_df = df[df["split"] == "dev"].copy()
        test_df = df[df["split"] == "test"].copy()
    else:
        raise ValueError(f"Unsupported task: {task}")

    if train_frac < 1.0:
        rng = np.random.default_rng(seed)
        if "text_id" in train_df.columns:
            tids = np.array(sorted(train_df["text_id"].unique()))
            rng.shuffle(tids)
            keep = set(tids[: max(1, int(round(len(tids) * train_frac)))])
            train_df = train_df[train_df["text_id"].isin(keep)].copy()
        else:
            n_keep = max(1, int(round(len(train_df) * train_frac)))
            train_df = train_df.sample(n=n_keep, random_state=seed).copy()

    return train_df, cal_df, dev_df, test_df, target_cols, bounds, has_aspect


def build_model_and_flags(
    variant: str,
    backbone: str,
    out_dim: int,
    bounds: Tuple[float, float],
    tokenizer_vocab_size: int,
    lora_rank: int = 8,
    bilstm_hidden: int = 64,
    bilstm_layers: int = 2,
    k_lat: int = 4,
    dropout: float = 0.2,
    use_lora: bool = True,
):
    scale_min, scale_max = bounds
    flags = {
        "head_mode": "evidential",
        "use_concepts": True,
        "use_ccc": True,
        "use_dynamic_weight": True,
        "use_lds": True,
        "use_rdrop": False,
        "use_causal_constraints": False,
        "is_classic": False,
    }

    if variant == "concept_bilstm_urer":
        arch_mode = "concept_bilstm"
    elif variant == "concept_bilstm_causal":
        arch_mode = "concept_bilstm_causal"
        flags["use_causal_constraints"] = True
    elif variant == "ib_residual_bilstm":
        arch_mode = "ib_residual_bilstm"
    elif variant == "plm_bilstm_no_concept":
        arch_mode = "plm_bilstm_no_concept"
        flags["use_concepts"] = False
    elif variant == "concept_no_bilstm":
        arch_mode = "concept_no_bilstm"
    elif variant == "concept_bilstm_no_ccc":
        arch_mode = "concept_bilstm"
        flags["use_ccc"] = False
    elif variant == "concept_bilstm_no_dyn":
        arch_mode = "concept_bilstm"
        flags["use_dynamic_weight"] = False
        flags["use_lds"] = False
    elif variant == "concept_bilstm_det":
        arch_mode = "concept_bilstm"
        flags["head_mode"] = "deterministic"
        flags["use_dynamic_weight"] = False
    elif variant == "plm_only_mse":
        arch_mode = "plm_only"
        flags["head_mode"] = "deterministic"
        flags["use_concepts"] = False
        flags["use_ccc"] = False
        flags["use_dynamic_weight"] = False
        flags["use_lds"] = False
    elif variant == "plm_only_ccc":
        arch_mode = "plm_only"
        flags["head_mode"] = "deterministic"
        flags["use_concepts"] = False
        flags["use_dynamic_weight"] = False
        flags["use_lds"] = False
    elif variant == "plm_only_rdrop":
        arch_mode = "plm_only"
        flags["head_mode"] = "deterministic"
        flags["use_concepts"] = False
        flags["use_ccc"] = False
        flags["use_dynamic_weight"] = False
        flags["use_lds"] = False
        flags["use_rdrop"] = True
    elif variant in ("classic_bilstm", "classic_bigru", "classic_cnn"):
        rnn_type = variant.replace("classic_", "")
        flags["head_mode"] = "deterministic"
        flags["use_concepts"] = False
        flags["use_ccc"] = False
        flags["use_dynamic_weight"] = False
        flags["use_lds"] = False
        flags["is_classic"] = True
        model = ClassicRNNRegressor(
            vocab_size=tokenizer_vocab_size,
            embed_dim=128,
            hidden_dim=bilstm_hidden,
            num_layers=bilstm_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            out_dim=out_dim,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        return model, flags
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    model = PLMAspectSentimentModel(
        backbone_name=backbone,
        out_dim=out_dim,
        arch_mode=arch_mode,
        head_mode=flags["head_mode"],
        use_lora=use_lora,
        lora_rank=lora_rank,
        k_exp=8,
        k_lat=k_lat,
        bilstm_hidden=bilstm_hidden,
        bilstm_layers=bilstm_layers,
        dropout=dropout,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    return model, flags


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    head_mode: str,
    mc_dropout_runs: int = 0,
) -> Dict[str, np.ndarray]:
    if mc_dropout_runs > 0:
        model.eval()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
    else:
        model.eval()

    preds_list, y_list, tot_std_list, ale_list, epi_list = [], [], [], [], []
    z_seq_norms, z_ib_norms = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        sent_mask = batch["sent_mask"].to(device)
        asp_mask = batch["asp_mask"].to(device)
        rel_dist = batch["rel_dist"].to(device)

        if mc_dropout_runs > 0:
            mc_preds = []
            for _ in range(mc_dropout_runs):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    sent_mask=sent_mask,
                    asp_mask=asp_mask,
                    rel_dist=rel_dist,
                )
                mc_preds.append(out["pred"].cpu().numpy())
            arr = np.stack(mc_preds, axis=0)
            preds_list.append(arr.mean(axis=0))
            std_mc = np.maximum(arr.std(axis=0), 1e-4)
            tot_std_list.append(std_mc)
            ale_list.append(np.zeros_like(std_mc))
            epi_list.append(std_mc)
        else:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                sent_mask=sent_mask,
                asp_mask=asp_mask,
                rel_dist=rel_dist,
            )
            preds_list.append(out["pred"].cpu().numpy())
            if head_mode == "evidential":
                tot_std_list.append(out["tot_std"].cpu().numpy())
                ale_list.append(torch.sqrt(out["ale_var"]).cpu().numpy())
                epi_list.append(torch.sqrt(out["epi_var"]).cpu().numpy())
            else:
                zeros = np.ones_like(preds_list[-1], dtype=np.float32)
                tot_std_list.append(zeros)
                ale_list.append(zeros)
                epi_list.append(zeros)

            if "z_ib_norm" in out:
                z_seq_norms.append(float(out["z_seq_norm"].item()))
                z_ib_norms.append(float(out["z_ib_norm"].item()))

        y_list.append(batch["targets"].numpy())

    res = {
        "y_true": np.concatenate(y_list, axis=0),
        "y_pred": np.concatenate(preds_list, axis=0),
        "tot_std": np.concatenate(tot_std_list, axis=0),
        "ale_std": np.concatenate(ale_list, axis=0),
        "epi_std": np.concatenate(epi_list, axis=0),
    }
    if z_ib_norms:
        zs = float(np.mean(z_seq_norms))
        zi = float(np.mean(z_ib_norms))
        res["ib_leakage_ratio"] = zi / max(zs + zi, 1e-8)
    return res


@torch.no_grad()
def measure_latency_ms(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    sent_mask = batch["sent_mask"].to(device)
    asp_mask = batch["asp_mask"].to(device)
    rel_dist = batch["rel_dist"].to(device)

    for _ in range(5):
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sent_mask=sent_mask,
            asp_mask=asp_mask,
            rel_dist=rel_dist,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_runs = 20
    for _ in range(n_runs):
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sent_mask=sent_mask,
            asp_mask=asp_mask,
            rel_dist=rel_dist,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    total_samples = n_runs * input_ids.shape[0]
    return float(elapsed_ms / max(1, total_samples))


def evaluate_all_metrics(
    task: str,
    res_test: Dict[str, np.ndarray],
    calibrator: Optional[EvidentialConformalCalibrator],
    tail_bounds: Optional[Dict[str, Tuple[float, float]]],
    alpha: float = 0.10,
    test_types: Optional[List[str]] = None,
) -> Dict[str, float]:
    yt = res_test["y_true"]
    yp = res_test["y_pred"]
    std = res_test["tot_std"]
    metrics: Dict[str, float] = {}

    if task.startswith("dimabsa_"):
        metrics.update(regression_metrics_va(yt[:, 0], yt[:, 1], yp[:, 0], yp[:, 1], tail_bounds=tail_bounds))
        ks_v = residual_ks_tests(yt[:, 0], yp[:, 0])
        ks_a = residual_ks_tests(yt[:, 1], yp[:, 1])
        metrics["KS_V_stat"] = ks_v["ks_norm_stat"]
        metrics["KS_V_p"] = ks_v["ks_norm_p"]
        metrics["KS_A_stat"] = ks_a["ks_norm_stat"]
        metrics["KS_A_p"] = ks_a["ks_norm_p"]
        if calibrator is not None:
            lo, hi = calibrator.predict(yp, std)
            c_v = interval_calibration_metrics(yt[:, 0], lo[:, 0], hi[:, 0], std[:, 0], yp[:, 0], alpha=alpha)
            c_a = interval_calibration_metrics(yt[:, 1], lo[:, 1], hi[:, 1], std[:, 1], yp[:, 1], alpha=alpha)
            metrics["PICP_V"] = c_v["PICP"]
            metrics["MPIW_V"] = c_v["MPIW"]
            metrics["Winkler_V"] = c_v["Winkler"]
            metrics["UncErr_Rho_V"] = c_v.get("UncErr_Spearman", 0.0)
            metrics["PICP_A"] = c_a["PICP"]
            metrics["MPIW_A"] = c_a["MPIW"]
            metrics["Winkler_A"] = c_a["Winkler"]
            metrics["UncErr_Rho_A"] = c_a.get("UncErr_Spearman", 0.0)
            metrics["Winkler_avg"] = 0.5 * (c_v["Winkler"] + c_a["Winkler"])
            # Uncalibrated DER raw Gaussian intervals for ablating Split Conformal calibration
            z = float(norm.ppf(1.0 - alpha / 2.0))
            lo_u = np.clip(yp - z * std, 1.0, 9.0)
            hi_u = np.clip(yp + z * std, 1.0, 9.0)
            u_v = interval_calibration_metrics(yt[:, 0], lo_u[:, 0], hi_u[:, 0], std[:, 0], yp[:, 0], alpha=alpha)
            u_a = interval_calibration_metrics(yt[:, 1], lo_u[:, 1], hi_u[:, 1], std[:, 1], yp[:, 1], alpha=alpha)
            metrics["uncal_PICP_V"] = u_v["PICP"]
            metrics["uncal_MPIW_V"] = u_v["MPIW"]
            metrics["uncal_Winkler_V"] = u_v["Winkler"]
            metrics["uncal_PICP_A"] = u_a["PICP"]
            metrics["uncal_MPIW_A"] = u_a["MPIW"]
            metrics["uncal_Winkler_A"] = u_a["Winkler"]
            metrics["uncal_Winkler_avg"] = 0.5 * (u_v["Winkler"] + u_a["Winkler"])
    elif task == "brighter_chn":
        rmses, pccs, cccs, maes = [], [], [], []
        for j in range(yt.shape[1]):
            rmses.append(float(np.sqrt(np.mean((yt[:, j] - yp[:, j]) ** 2))))
            maes.append(float(np.mean(np.abs(yt[:, j] - yp[:, j]))))
            pccs.append(safe_pearson(yt[:, j], yp[:, j]))
            cccs.append(concordance_correlation_coefficient(yt[:, j], yp[:, j]))
        metrics["RMSE_macro"] = float(np.mean(rmses))
        metrics["MAE_macro"] = float(np.mean(maes))
        metrics["PCC_macro"] = float(np.mean(pccs))
        metrics["CCC_macro"] = float(np.mean(cccs))
        if calibrator is not None:
            lo, hi = calibrator.predict(yp, std)
            picps, mpiws, winks, rhos = [], [], [], []
            for j in range(yt.shape[1]):
                cm = interval_calibration_metrics(yt[:, j], lo[:, j], hi[:, j], std[:, j], yp[:, j], alpha=alpha)
                picps.append(cm["PICP"])
                mpiws.append(cm["MPIW"])
                winks.append(cm["Winkler"])
                rhos.append(cm.get("UncErr_Spearman", 0.0))
            metrics["PICP_macro"] = float(np.mean(picps))
            metrics["MPIW_macro"] = float(np.mean(mpiws))
            metrics["Winkler_macro"] = float(np.mean(winks))
            metrics["UncErr_Rho_macro"] = float(np.mean(rhos))
    else:
        pred_bin = (yp[:, 0] >= 5.0).astype(np.int64)
        true_bin = (yt[:, 0] >= 5.0).astype(np.int64)
        metrics["Acc"] = float(np.mean(pred_bin == true_bin))
        metrics["RMSE"] = float(np.sqrt(np.mean((yt[:, 0] - yp[:, 0]) ** 2)))
        metrics["CCC"] = concordance_correlation_coefficient(yt[:, 0], yp[:, 0])
        if len(np.unique(true_bin)) > 1:
            metrics["AUC"] = float(roc_auc_score(true_bin, yp[:, 0]))
        if test_types is not None and len(test_types) == len(true_bin):
            rt_arr = np.asarray(test_types)
            for rt_name, rt_slug in [("简单句", "simple"), ("转折", "adversative"), ("双重否定", "double_neg"), ("反讽", "irony")]:
                m = rt_arr == rt_name
                if m.any():
                    metrics[f"Acc_{rt_slug}"] = float(np.mean(pred_bin[m] == true_bin[m]))
                    if len(np.unique(true_bin[m])) > 1:
                        metrics[f"AUC_{rt_slug}"] = float(roc_auc_score(true_bin[m], yp[m, 0]))

    if "ib_leakage_ratio" in res_test:
        metrics["ib_leakage_ratio"] = res_test["ib_leakage_ratio"]
    return metrics


def run_explainability_audit(
    model: torch.nn.Module,
    loader: DataLoader,
    tokenizer,
    device: torch.device,
    max_batches: int = 5,
) -> Dict[str, object]:
    """Run quantitative faithfulness comparison (Concept vs IG vs Attention) & concept interventions."""
    mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 103
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    agg = {
        "concept": {"comp": [], "suff": []},
        "ig": {"comp": [], "suff": []},
        "attention": {"comp": [], "suff": []},
        "cf_negation_abs_delta": [],
        "cf_adversative_abs_delta": [],
        "cf_pos_cue_deltas": [],
        "cf_neg_cue_deltas": [],
    }
    concept_contribs = []
    cue_active_vals, cue_bg_vals = [], []

    for b_idx, batch in enumerate(loader):
        if b_idx >= max_batches:
            break
        tok_c, c_contrib = compute_concept_token_attribution(model, batch, target_dim=0)
        concept_contribs.append(c_contrib)
        ig_scores = compute_integrated_gradients_attribution(model, batch, target_dim=0, n_steps=6)

        with torch.no_grad():
            out = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
                batch["sent_mask"].to(device),
                batch["asp_mask"].to(device),
                batch["rel_dist"].to(device),
            )
            attn_scores = out["attn_weights"].cpu().numpy()
            if "c_seq" in out:
                c_exp = out["c_seq"][:, :, :8].cpu()
                cp = batch["concept_priors"]
                sm = batch["sent_mask"].unsqueeze(-1) > 0.5
                act_m = (cp > 0.5) & sm
                bg_m = (cp < 0.1) & sm
                if act_m.any():
                    cue_active_vals.append(float(c_exp[act_m].mean().item()))
                if bg_m.any():
                    cue_bg_vals.append(float(c_exp[bg_m].mean().item()))

        f_c = evaluate_faithfulness_batch(model, batch, tok_c, pad_id, mask_id, target_dim=0)
        f_ig = evaluate_faithfulness_batch(model, batch, ig_scores, pad_id, mask_id, target_dim=0)
        f_at = evaluate_faithfulness_batch(model, batch, attn_scores, pad_id, mask_id, target_dim=0)

        for k, f_res in [("concept", f_c), ("ig", f_ig), ("attention", f_at)]:
            agg[k]["comp"].append(f_res["comp"])
            agg[k]["suff"].append(f_res["suff"])

        # Direct causal interventions on Positive/Negative Cues, Negation (4), and Adversative (5)
        has_pos = (batch["concept_priors"][:, :, 1].sum(dim=1) > 0.5).numpy()
        has_neg_cue = (batch["concept_priors"][:, :, 2].sum(dim=1) > 0.5).numpy()
        has_neg = (batch["concept_priors"][:, :, 4].sum(dim=1) > 0.5).numpy()
        has_adv = (batch["concept_priors"][:, :, 5].sum(dim=1) > 0.5).numpy()
        if has_pos.any():
            d_pos = evaluate_concept_counterfactual(model, batch, concept_idx=1, new_val=0.0, target_dim=0)
            agg["cf_pos_cue_deltas"].extend(d_pos[has_pos].tolist())
        if has_neg_cue.any():
            d_ncue = evaluate_concept_counterfactual(model, batch, concept_idx=2, new_val=0.0, target_dim=0)
            agg["cf_neg_cue_deltas"].extend(d_ncue[has_neg_cue].tolist())
        if has_neg.any():
            d_neg = evaluate_concept_counterfactual(model, batch, concept_idx=4, new_val=0.0, target_dim=0)
            agg["cf_negation_abs_delta"].extend(np.abs(d_neg[has_neg]).tolist())
        if has_adv.any():
            d_adv = evaluate_concept_counterfactual(model, batch, concept_idx=5, new_val=0.0, target_dim=0)
            agg["cf_adversative_abs_delta"].extend(np.abs(d_adv[has_adv]).tolist())

    summary = {}
    for k in ("concept", "ig", "attention"):
        summary[f"{k}_comp_mean"] = np.mean(agg[k]["comp"], axis=0).tolist() if agg[k]["comp"] else []
        summary[f"{k}_suff_mean"] = np.mean(agg[k]["suff"], axis=0).tolist() if agg[k]["suff"] else []
    summary["cf_negation_mean_shift"] = float(np.mean(agg["cf_negation_abs_delta"])) if agg["cf_negation_abs_delta"] else 0.0
    summary["cf_adversative_mean_shift"] = float(np.mean(agg["cf_adversative_abs_delta"])) if agg["cf_adversative_abs_delta"] else 0.0
    if agg["cf_pos_cue_deltas"]:
        arr_p = np.asarray(agg["cf_pos_cue_deltas"])
        summary["cf_pos_cue_dir_acc"] = float(np.mean(arr_p < 0.0))
        summary["cf_pos_cue_mean_delta"] = float(np.mean(arr_p))
    if agg["cf_neg_cue_deltas"]:
        arr_n = np.asarray(agg["cf_neg_cue_deltas"])
        summary["cf_neg_cue_dir_acc"] = float(np.mean(arr_n > 0.0))
        summary["cf_neg_cue_mean_delta"] = float(np.mean(arr_n))

    if concept_contribs:
        all_cc = np.concatenate(concept_contribs, axis=0)
        mean_abs_cc = np.mean(np.abs(all_cc), axis=0)
        summary["mean_abs_concept_contrib"] = {
            (CONCEPT_NAMES[i] if i < len(CONCEPT_NAMES) else f"Latent_{i - len(CONCEPT_NAMES)}"): float(mean_abs_cc[i])
            for i in range(len(mean_abs_cc))
        }
    if cue_active_vals:
        summary["concept_cue_active_mean"] = float(np.mean(cue_active_vals))
    if cue_bg_vals:
        summary["concept_bg_mean"] = float(np.mean(cue_bg_vals))
    return summary


def train_and_evaluate_single(args, seed: int) -> Dict[str, object]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, cal_df, dev_df, test_df, target_cols, bounds, has_aspect = load_task_splits(
        args.task, train_frac=args.train_frac, seed=seed
    )
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    model, flags = build_model_and_flags(
        variant=args.variant,
        backbone=args.backbone,
        out_dim=len(target_cols),
        bounds=bounds,
        tokenizer_vocab_size=len(tokenizer),
        lora_rank=args.lora_rank,
        bilstm_hidden=args.bilstm_hidden,
        bilstm_layers=args.bilstm_layers,
        k_lat=args.k_lat,
        dropout=args.dropout,
        use_lora=not args.full_finetune,
    )
    model.to(device)

    # Compute aggregated multi-target LDS extreme-tail weights strictly from train_df
    if flags["use_lds"]:
        lds_cols = [compute_lds_weights(train_df[c].to_numpy()) for c in target_cols]
        lds_arr = np.mean(lds_cols, axis=0).astype(np.float32)
        lds_arr = (lds_arr / max(float(np.mean(lds_arr)), 1e-6)).astype(np.float32)
    else:
        lds_arr = np.ones(len(train_df), dtype=np.float32)

    train_ds = SentimentRegressionDataset(
        train_df, tokenizer, target_cols, max_length=args.max_length, has_aspect=has_aspect, lds_weights=lds_arr
    )
    cal_ds = SentimentRegressionDataset(
        cal_df, tokenizer, target_cols, max_length=args.max_length, has_aspect=has_aspect
    )
    dev_ds = SentimentRegressionDataset(
        dev_df, tokenizer, target_cols, max_length=args.max_length, has_aspect=has_aspect
    )
    test_ds = SentimentRegressionDataset(
        test_df, tokenizer, target_cols, max_length=args.max_length, has_aspect=has_aspect
    )

    nw = args.num_workers
    use_pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=use_pin)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=use_pin)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=use_pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=use_pin)

    tail_bounds = None
    if args.task.startswith("dimabsa_"):
        v_tr = train_df["valence"].to_numpy()
        a_tr = train_df["arousal"].to_numpy()
        tail_bounds = {
            "V": (float(np.quantile(v_tr, 0.10)), float(np.quantile(v_tr, 0.90))),
            "A": (float(np.quantile(a_tr, 0.10)), float(np.quantile(a_tr, 0.90))),
        }

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.05)

    best_dev_score = float("inf")
    best_state = None
    best_epoch = -1
    patience_cnt = 0

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for epoch in range(args.epochs):
        model.train()
        ep_losses = []
        for batch in train_loader:
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                out = model(
                    input_ids=batch_dev["input_ids"],
                    attention_mask=batch_dev["attention_mask"],
                    token_type_ids=batch_dev["token_type_ids"],
                    sent_mask=batch_dev["sent_mask"],
                    asp_mask=batch_dev["asp_mask"],
                    rel_dist=batch_dev["rel_dist"],
                )
                loss_dict = compute_total_loss(
                    outputs=out,
                    batch=batch_dev,
                    head_mode=flags["head_mode"],
                    use_concepts=flags["use_concepts"],
                    k_exp=8,
                    lambda_ccc=args.lambda_ccc,
                    lambda_reg=args.lambda_reg,
                    lambda_concept_align=args.lambda_concept_align,
                    lambda_concept_sparse=args.lambda_concept_sparse,
                    lambda_concept_ortho=args.lambda_concept_ortho,
                    lambda_ib=args.lambda_ib,
                    eta_epi=args.eta_epi,
                    eta_ale=args.eta_ale,
                    use_dynamic_weight=flags["use_dynamic_weight"],
                    epoch=epoch,
                    warmup_epochs=args.warmup_epochs,
                    use_ccc=flags["use_ccc"],
                    use_causal_constraints=flags["use_causal_constraints"],
                )
                loss = loss_dict["total"]
                if flags["use_rdrop"]:
                    out2 = model(
                        input_ids=batch_dev["input_ids"],
                        attention_mask=batch_dev["attention_mask"],
                        token_type_ids=batch_dev["token_type_ids"],
                        sent_mask=batch_dev["sent_mask"],
                        asp_mask=batch_dev["asp_mask"],
                        rel_dist=batch_dev["rel_dist"],
                    )
                    loss2_dict = compute_total_loss(
                        outputs=out2,
                        batch=batch_dev,
                        head_mode=flags["head_mode"],
                        use_concepts=flags["use_concepts"],
                        k_exp=8,
                        lambda_ccc=args.lambda_ccc,
                        lambda_reg=args.lambda_reg,
                        lambda_concept_align=args.lambda_concept_align,
                        lambda_concept_sparse=args.lambda_concept_sparse,
                        lambda_concept_ortho=args.lambda_concept_ortho,
                        lambda_ib=args.lambda_ib,
                        eta_epi=args.eta_epi,
                        eta_ale=args.eta_ale,
                        use_dynamic_weight=flags["use_dynamic_weight"],
                        epoch=epoch,
                        warmup_epochs=args.warmup_epochs,
                        use_ccc=flags["use_ccc"],
                    )
                    loss = 0.5 * (loss + loss2_dict["total"]) + 0.5 * F.mse_loss(out["pred"].float(), out2["pred"].float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ep_losses.append(float(loss.item()))

        dev_res = predict_loader(model, dev_loader, device, flags["head_mode"])
        dev_metrics = evaluate_all_metrics(args.task, dev_res, None, tail_bounds, alpha=args.conformal_alpha)
        if args.task.startswith("dimabsa_"):
            dev_score = dev_metrics["RMSE_VA"]
        elif args.task == "brighter_chn":
            dev_score = dev_metrics["RMSE_macro"]
        else:
            dev_score = 1.0 - dev_metrics["Acc"]

        if dev_score < best_dev_score:
            best_dev_score = dev_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        print(
            f"[Epoch {epoch+1:02d}/{args.epochs}] train_loss={np.mean(ep_losses):.4f} "
            f"dev_score={dev_score:.4f} best={best_dev_score:.4f} (ep={best_epoch+1})"
        )
        if patience_cnt >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    mc_runs = 10 if args.mc_dropout else 0
    calibrator = None
    if flags["head_mode"] == "evidential" or args.mc_dropout:
        cal_res = predict_loader(model, cal_loader, device, flags["head_mode"], mc_dropout_runs=mc_runs)
        calibrator = EvidentialConformalCalibrator(
            alpha=args.conformal_alpha, scale_min=bounds[0], scale_max=bounds[1]
        )
        calibrator.fit(cal_res["y_true"], cal_res["y_pred"], cal_res["tot_std"])

    test_types = test_df["type"].astype(str).tolist() if "type" in test_df.columns else None
    test_res = predict_loader(model, test_loader, device, flags["head_mode"], mc_dropout_runs=mc_runs)
    test_metrics = evaluate_all_metrics(
        args.task, test_res, calibrator, tail_bounds, alpha=args.conformal_alpha, test_types=test_types
    )
    latency_ms = measure_latency_ms(model, test_loader, device)
    trainable_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)

    test_metrics["latency_ms"] = latency_ms
    test_metrics["trainable_params"] = int(trainable_cnt)
    test_metrics["best_epoch"] = int(best_epoch + 1)

    noise_results = {}
    if args.eval_noise:
        for nr in (0.10, 0.20):
            noisy_ds = SentimentRegressionDataset(
                test_df, tokenizer, target_cols, max_length=args.max_length,
                has_aspect=has_aspect, noise_rate=nr, noise_seed=seed,
            )
            noisy_loader = DataLoader(noisy_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw)
            nr_res = predict_loader(model, noisy_loader, device, flags["head_mode"], mc_dropout_runs=mc_runs)
            noise_results[f"noise_{int(nr*100)}"] = evaluate_all_metrics(
                args.task, nr_res, calibrator, tail_bounds, alpha=args.conformal_alpha
            )

    explain_results = {}
    if args.eval_explain and flags["use_concepts"] and not flags["is_classic"]:
        if flags["use_causal_constraints"]:
            # Deterministic seeded shuffle avoids block-ordered split bias (e.g. logic_sentiment test types)
            g_expl = torch.Generator().manual_seed(seed)
            expl_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, generator=g_expl, num_workers=nw)
        else:
            expl_loader = test_loader
        explain_results = run_explainability_audit(model, expl_loader, tokenizer, device, max_batches=10)

    out_dir = OUT_ROOT / args.exp_name / f"{args.task}__{args.variant}__seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if calibrator is not None:
        lo, hi = calibrator.predict(test_res["y_pred"], test_res["tot_std"])
        test_res["lower"] = lo
        test_res["upper"] = hi
    np.savez(out_dir / "test_predictions.npz", **test_res)

    if args.save_checkpoint:
        torch.save(model.state_dict(), out_dir / "best_model.pt")

    payload = {
        "task": args.task,
        "variant": args.variant,
        "backbone": args.backbone,
        "seed": seed,
        "test_metrics": test_metrics,
        "noise_results": noise_results,
        "explain_results": explain_results,
        "hparams": vars(args),
    }
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("TEST_METRICS:", json.dumps(test_metrics, indent=2))
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate continuous sentiment models.")
    p.add_argument("--exp-name", type=str, default="main_run")
    p.add_argument("--task", type=str, default="dimabsa_zho_rest")
    p.add_argument("--variant", type=str, default="concept_bilstm_urer")
    p.add_argument("--backbone", type=str, default="hfl/chinese-macbert-base")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--bilstm-hidden", type=int, default=64)
    p.add_argument("--bilstm-layers", type=int, default=2)
    p.add_argument("--k-lat", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lambda-ccc", type=float, default=0.5)
    p.add_argument("--lambda-reg", type=float, default=0.05)
    p.add_argument("--lambda-concept-align", type=float, default=0.35)
    p.add_argument("--lambda-concept-sparse", type=float, default=0.06)
    p.add_argument("--lambda-concept-ortho", type=float, default=0.05)
    p.add_argument("--lambda-ib", type=float, default=0.01)
    p.add_argument("--eta-epi", type=float, default=0.5)
    p.add_argument("--eta-ale", type=float, default=0.5)
    p.add_argument("--conformal-alpha", type=float, default=0.10)
    p.add_argument("--train-frac", type=float, default=1.0)
    p.add_argument("--full-finetune", action="store_true")
    p.add_argument("--mc-dropout", action="store_true")
    p.add_argument("--eval-noise", action="store_true")
    p.add_argument("--eval-explain", action="store_true")
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--num-workers", type=int, default=(0 if os.name == "nt" else 2))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    all_runs = []
    for s in args.seeds:
        print(f"\n=== Running {args.task} | {args.variant} | seed={s} ===")
        all_runs.append(train_and_evaluate_single(args, s))

    if len(all_runs) > 1:
        keys = [k for k, v in all_runs[0]["test_metrics"].items() if isinstance(v, (int, float))]
        agg = {}
        for k in keys:
            vals = [r["test_metrics"][k] for r in all_runs]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
        agg_dir = OUT_ROOT / args.exp_name
        agg_path = agg_dir / f"{args.task}__{args.variant}__summary_{len(args.seeds)}seeds.json"
        with agg_path.open("w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)
        print("\n=== MULTI-SEED SUMMARY ===")
        print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
