"""Cross-task rhetorical probes, OOD AUROC, statistical tests, LLM baselines, and publication figures."""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, ks_2samp, mannwhitneyu, wilcoxon
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from src.data.concepts import CONCEPT_NAMES, to_simplified
from src.utils.metrics import (
    concordance_correlation_coefficient,
    ood_detection_auroc,
    regression_metrics_va,
    residual_ks_tests,
    safe_pearson,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = PROJECT_ROOT / "data" / "processed"
OUT_ROOT = PROJECT_ROOT / "outputs"


def configure_matplotlib_fonts() -> None:
    """Configure clean sans-serif font with Chinese glyph fallback when available."""
    preferred = [
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "DejaVu Sans",
    ]
    avail = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [f for f in preferred if f in avail]
    if not chosen:
        chosen = ["DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(1, na + nb - 2))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def forward_detailed(
    model,
    loader,
    device,
    head_mode: str,
) -> Dict[str, np.ndarray]:
    import torch

    model.eval()
    preds, epi, ale, tot, c_mean, fw_bw = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            sent_mask = batch["sent_mask"].to(device)
            asp_mask = batch["asp_mask"].to(device)
            rel_dist = batch["rel_dist"].to(device)
            out = model(input_ids, attention_mask, token_type_ids, sent_mask, asp_mask, rel_dist)
            preds.append(out["pred"].cpu().numpy())
            if head_mode == "evidential":
                epi.append(torch.sqrt(out["epi_var"]).cpu().numpy())
                ale.append(torch.sqrt(out["ale_var"]).cpu().numpy())
                tot.append(out["tot_std"].cpu().numpy())
            else:
                z = np.ones_like(preds[-1])
                epi.append(z)
                ale.append(z)
                tot.append(z)
            if "c_seq" in out:
                sm = sent_mask.unsqueeze(-1)
                cm = (out["c_seq"] * sm).sum(dim=1) / sm.sum(dim=1).clamp(min=1.0)
                c_mean.append(cm.cpu().numpy())
            if "fw_bw_diff" in out:
                fb = (out["fw_bw_diff"] * sent_mask).sum(dim=1) / sent_mask.sum(dim=1).clamp(min=1.0)
                fw_bw.append(fb.cpu().numpy())
    res = {
        "pred": np.concatenate(preds, axis=0),
        "epi_std": np.concatenate(epi, axis=0),
        "ale_std": np.concatenate(ale, axis=0),
        "tot_std": np.concatenate(tot, axis=0),
    }
    if c_mean:
        res["c_mean"] = np.concatenate(c_mean, axis=0)
    if fw_bw:
        res["fw_bw_diff"] = np.concatenate(fw_bw, axis=0)
    return res


def run_rhetorical_probe(args) -> None:
    """Zero-shot cross-task probe of DimABSA checkpoint on Chinese rhetorical logic test set & OOD domain."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from src.data.dataset import SentimentRegressionDataset
    from src.explain.faithfulness import evaluate_concept_counterfactual
    from src.train_eval import build_model_and_flags, load_task_splits, set_seed

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    res_json = json.loads((ckpt_dir / "results.json").read_text(encoding="utf-8"))
    hp = res_json["hparams"]

    tokenizer = AutoTokenizer.from_pretrained(res_json["backbone"])
    model, flags = build_model_and_flags(
        variant=res_json["variant"],
        backbone=res_json["backbone"],
        out_dim=2,
        bounds=(1.0, 9.0),
        tokenizer_vocab_size=len(tokenizer),
        lora_rank=hp["lora_rank"],
        bilstm_hidden=hp["bilstm_hidden"],
        bilstm_layers=hp["bilstm_layers"],
        k_lat=hp["k_lat"],
        dropout=hp["dropout"],
        use_lora=not hp.get("full_finetune", False),
    )
    state = torch.load(ckpt_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 1. Logic Sentiment zero-shot rhetorical probe
    logic_df = pd.read_parquet(DATA_PROC / "logic_sentiment_zho.parquet")
    logic_test = logic_df[logic_df["split"] == "test"].reset_index(drop=True)
    logic_test["valence"] = logic_test["label"].astype(np.float32) * 8.0 + 1.0
    logic_test["arousal"] = 5.0
    logic_ds = SentimentRegressionDataset(
        logic_test, tokenizer, ["valence", "arousal"], max_length=hp["max_length"], has_aspect=False
    )
    nw = 0 if (os.name == "nt" or device.type != "cuda") else 2
    logic_loader = DataLoader(logic_ds, batch_size=32, shuffle=False, num_workers=nw)
    det_logic = forward_detailed(model, logic_loader, device, flags["head_mode"])

    v_pred = det_logic["pred"][:, 0]
    y_bin = logic_test["label"].to_numpy().astype(int)
    pos_v = v_pred[y_bin == 1]
    neg_v = v_pred[y_bin == 0]
    auc_all = float(roc_auc_score(y_bin, v_pred))
    acc_all = float(accuracy_score(y_bin, (v_pred >= 5.0).astype(int)))
    d_all = cohens_d(pos_v, neg_v)

    # Counterfactual direction check over logic test set
    cf_neg_deltas, cf_adv_deltas = [], []
    if flags["use_concepts"]:
        for batch in logic_loader:
            d_neg = evaluate_concept_counterfactual(model, batch, concept_idx=4, new_val=0.0, target_dim=0)
            d_adv = evaluate_concept_counterfactual(model, batch, concept_idx=5, new_val=0.0, target_dim=0)
            cf_neg_deltas.append(d_neg)
            cf_adv_deltas.append(d_adv)
    cf_neg_arr = np.concatenate(cf_neg_deltas) if cf_neg_deltas else np.zeros(len(logic_test))
    cf_adv_arr = np.concatenate(cf_adv_deltas) if cf_adv_deltas else np.zeros(len(logic_test))

    epi_v = det_logic["epi_std"][:, 0]
    ale_v = det_logic["ale_std"][:, 0]
    rtypes = logic_test["type"].astype(str).tolist()
    by_type = {}
    simple_epi = epi_v[np.array(rtypes) == "简单句"]
    for rt in ["简单句", "转折", "双重否定", "反讽"]:
        m = np.array(rtypes) == rt
        if not m.any():
            continue
        sub_y = y_bin[m]
        sub_vp = v_pred[m]
        sub_epi = epi_v[m]
        sub_ale = ale_v[m]
        row = {
            "n": int(m.sum()),
            "zero_shot_auc": float(roc_auc_score(sub_y, sub_vp)) if len(np.unique(sub_y)) > 1 else 0.5,
            "zero_shot_acc_at_5": float(accuracy_score(sub_y, (sub_vp >= 5.0).astype(int))),
            "pos_v_mean": float(np.mean(sub_vp[sub_y == 1])) if (sub_y == 1).any() else 0.0,
            "neg_v_mean": float(np.mean(sub_vp[sub_y == 0])) if (sub_y == 0).any() else 0.0,
            "cohens_d": cohens_d(sub_vp[sub_y == 1], sub_vp[sub_y == 0]),
            "epi_std_mean": float(np.mean(sub_epi)),
            "epi_std_median": float(np.median(sub_epi)),
            "ale_std_mean": float(np.mean(sub_ale)),
            "cf_neg_abs_mean": float(np.mean(np.abs(cf_neg_arr[m]))),
            "cf_adv_abs_mean": float(np.mean(np.abs(cf_adv_arr[m]))),
        }
        if rt != "简单句" and len(simple_epi) > 5:
            u_stat, p_val = mannwhitneyu(sub_epi, simple_epi, alternative="greater")
            row["mwu_epi_gt_simple_U"] = float(u_stat)
            row["mwu_epi_gt_simple_p"] = float(p_val)
        if "fw_bw_diff" in det_logic:
            row["fw_bw_diff_mean"] = float(np.mean(det_logic["fw_bw_diff"][m]))
        if "c_mean" in det_logic:
            for ci, cname in enumerate(CONCEPT_NAMES):
                row[f"concept_{cname}_mean"] = float(np.mean(det_logic["c_mean"][m, ci]))
        by_type[rt] = row

    groups_epi = [epi_v[np.array(rtypes) == rt] for rt in ["简单句", "转折", "双重否定", "反讽"] if (np.array(rtypes) == rt).any()]
    has_epi_var = flags.get("head_mode", "evidential") == "evidential" and len(groups_epi) >= 2 and np.std(np.concatenate(groups_epi)) > 1e-12
    if has_epi_var:
        _ks, _kp = kruskal(*groups_epi)
        kw_stat = float(_ks) if np.isfinite(_ks) else None
        kw_p = float(_kp) if np.isfinite(_kp) else None
    else:
        kw_stat, kw_p = None, None

    # 2. Cross-domain OOD detection (In-domain test vs other DimABSA domains)
    in_pred = np.load(ckpt_dir / "test_predictions.npz")
    in_epi = in_pred["epi_std"].mean(axis=-1)
    in_tot = in_pred["tot_std"].mean(axis=-1)

    ood_summary = {}
    for ood_task in ["dimabsa_zho_rest", "dimabsa_zho_lap", "dimabsa_zho_fin", "dimabsa_zho_env"]:
        if ood_task == res_json.get("task"):
            continue
        _, _, _, ood_test_df, tcols, _, has_asp = load_task_splits(ood_task, seed=args.seed)
        ood_ds = SentimentRegressionDataset(ood_test_df, tokenizer, tcols, max_length=hp["max_length"], has_aspect=has_asp)
        nw = 0 if (os.name == "nt" or device.type != "cuda") else 2
        ood_loader = DataLoader(ood_ds, batch_size=32, shuffle=False, num_workers=nw)
        det_ood = forward_detailed(model, ood_loader, device, flags["head_mode"])
        ood_epi = det_ood["epi_std"].mean(axis=-1)
        ood_tot = det_ood["tot_std"].mean(axis=-1)
        is_evid = flags["head_mode"] == "evidential"
        ood_summary[ood_task] = {
            "in_epi_mean": float(np.mean(in_epi)),
            "ood_epi_mean": float(np.mean(ood_epi)),
            "auroc_epi": ood_detection_auroc(in_epi, ood_epi) if is_evid else 0.5,
            "auroc_tot": ood_detection_auroc(in_tot, ood_tot) if is_evid else 0.5,
        }
    is_evid = flags["head_mode"] == "evidential"
    ood_summary["logic_rhetorical_shift"] = {
        "in_epi_mean": float(np.mean(in_epi)),
        "ood_epi_mean": float(np.mean(det_logic["epi_std"].mean(axis=-1))),
        "auroc_epi": ood_detection_auroc(in_epi, det_logic["epi_std"].mean(axis=-1)) if is_evid else 0.5,
        "auroc_tot": ood_detection_auroc(in_tot, det_logic["tot_std"].mean(axis=-1)) if is_evid else 0.5,
    }

    # 3. Extract 6 illustrative real case studies (token-level c_seq + prediction + counterfactual)
    cases = []
    for target_rt in ["转折", "双重否定", "反讽", "简单句"]:
        idxs = np.where((np.array(rtypes) == target_rt))[0]
        if len(idxs) == 0:
            continue
        # pick one positive and one negative example with clear prediction
        for lbl in [0, 1]:
            sub_idx = [i for i in idxs if y_bin[i] == lbl]
            if not sub_idx:
                continue
            pick = int(sub_idx[0])
            row_df = logic_test.iloc[[pick]].copy()
            single_ds = SentimentRegressionDataset(row_df, tokenizer, ["valence", "arousal"], max_length=hp["max_length"], has_aspect=False)
            item = single_ds[0]
            ids = item["input_ids"].unsqueeze(0).to(device)
            am = item["attention_mask"].unsqueeze(0).to(device)
            tt = item["token_type_ids"].unsqueeze(0).to(device)
            sm = item["sent_mask"].unsqueeze(0).to(device)
            apm = item["asp_mask"].unsqueeze(0).to(device)
            rd = item["rel_dist"].unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(ids, am, tt, sm, apm, rd)
            toks = tokenizer.convert_ids_to_tokens(item["input_ids"].tolist())
            valid_pos = [t for t, s in enumerate(item["sent_mask"].tolist()) if s > 0.5]
            c_mat = out["c_seq"][0, valid_pos, :8].cpu().numpy().tolist() if "c_seq" in out else []
            cases.append({
                "rhetorical_type": target_rt,
                "gold_label": int(lbl),
                "text": str(row_df["text"].iloc[0]),
                "pred_valence": float(out["pred"][0, 0].item()),
                "pred_arousal": float(out["pred"][0, 1].item()),
                "epi_std_v": float(torch.sqrt(out["epi_var"][0, 0]).item()) if "epi_var" in out else 0.0,
                "ale_std_v": float(torch.sqrt(out["ale_var"][0, 0]).item()) if "ale_var" in out else 0.0,
                "cf_zero_negation_delta_v": float(cf_neg_arr[pick]),
                "cf_zero_adversative_delta_v": float(cf_adv_arr[pick]),
                "tokens": [toks[t] for t in valid_pos],
                "explicit_concepts": c_mat,
            })

    probe_out = {
        "ckpt_dir": str(ckpt_dir),
        "logic_zero_shot": {
            "n": int(len(logic_test)),
            "auc": auc_all,
            "acc_at_5": acc_all,
            "cohens_d": d_all,
            "pos_v_mean": float(np.mean(pos_v)),
            "neg_v_mean": float(np.mean(neg_v)),
            "kruskal_epi_stat": float(kw_stat) if kw_stat is not None else None,
            "kruskal_epi_p": float(kw_p) if kw_p is not None else None,
            "by_type": by_type,
        },
        "ood_detection": ood_summary,
        "cases": cases,
    }
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / "rhetorical_ood_probe.json").open("w", encoding="utf-8") as f:
        json.dump(probe_out, f, ensure_ascii=False, indent=2)
    print(f"[PROBE] Saved probe analysis to {out_path / 'rhetorical_ood_probe.json'}")
    print(json.dumps({
        "auc": auc_all, "acc": acc_all, "kruskal_p": kw_p,
        "ood": ood_summary,
    }, indent=2))


def run_stats_aggregation(args) -> None:
    """Aggregate multi-seed experiments and compute paired Wilcoxon & K-S significance tests."""
    exp_dir = Path(args.exp_dir)
    runs = []
    for res_file in sorted(exp_dir.glob("*__seed*/results.json")):
        data = json.loads(res_file.read_text(encoding="utf-8"))
        data["_dir"] = str(res_file.parent)
        runs.append(data)

    if not runs:
        print(f"[STATS] No results.json files found under {exp_dir}")
        return

    # Group by (task, variant, backbone)
    grouped: Dict[Tuple[str, str, str], List[dict]] = {}
    for r in runs:
        key = (r["task"], r["variant"], r["backbone"])
        grouped.setdefault(key, []).append(r)

    summary_rows = []
    for (task, variant, backbone), items in sorted(grouped.items()):
        m_keys = [k for k, v in items[0]["test_metrics"].items() if isinstance(v, (int, float))]
        row = {"task": task, "variant": variant, "backbone": backbone, "n_seeds": len(items)}
        for mk in m_keys:
            vals = [it["test_metrics"][mk] for it in items if mk in it["test_metrics"]]
            row[f"{mk}_mean"] = float(np.mean(vals))
            row[f"{mk}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        # Include noise summary if evaluated
        if items[0].get("noise_results"):
            for n_lvl in ("noise_10", "noise_20"):
                if n_lvl in items[0]["noise_results"]:
                    for nk, nv in items[0]["noise_results"][n_lvl].items():
                        if isinstance(nv, (int, float)):
                            vals = [it["noise_results"][n_lvl][nk] for it in items if n_lvl in it.get("noise_results", {})]
                            row[f"{n_lvl}_{nk}_mean"] = float(np.mean(vals))
                            row[f"{n_lvl}_{nk}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        # Include explainability summary if evaluated
        if items[0].get("explain_results"):
            er0 = items[0]["explain_results"]
            for ek in ("cf_negation_mean_shift", "cf_adversative_mean_shift", "cf_pos_cue_dir_acc", "cf_neg_cue_dir_acc"):
                if ek in er0:
                    vals = [it["explain_results"][ek] for it in items if ek in it.get("explain_results", {})]
                    row[f"{ek}_mean"] = float(np.mean(vals))
            for attr_m in ("concept", "ig", "attention"):
                ck = f"{attr_m}_comp_mean"
                sk = f"{attr_m}_suff_mean"
                if ck in er0 and er0[ck]:
                    mats_c = [it["explain_results"][ck] for it in items if it.get("explain_results", {}).get(ck)]
                    mats_s = [it["explain_results"][sk] for it in items if it.get("explain_results", {}).get(sk)]
                    row[f"{attr_m}_comp_auc"] = float(np.mean(mats_c))
                    row[f"{attr_m}_suff_auc"] = float(np.mean(mats_s))
        summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows)
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(out_dir / "summary_table.csv", index=False)

    # Paired Wilcoxon & Two-Sample K-S vs proposed reference variant per task
    ref_pool = runs
    if args.ref_exp_dir and Path(args.ref_exp_dir).exists():
        ref_pool = []
        for res_file in sorted(Path(args.ref_exp_dir).glob("*__seed*/results.json")):
            d = json.loads(res_file.read_text(encoding="utf-8"))
            d["_dir"] = str(res_file.parent)
            ref_pool.append(d)

    ref_variant = args.ref_variant
    sig_rows = []
    tasks = sorted({r["task"] for r in runs})
    for task in tasks:
        ref_bb = args.ref_backbone
        ref_runs = {r["seed"]: r for r in ref_pool if r["task"] == task and r["variant"] == ref_variant and r["backbone"] == ref_bb}
        if not ref_runs:
            cand_ref = [r for r in ref_pool if r["task"] == task and r["variant"] == ref_variant]
            if cand_ref:
                ref_bb = cand_ref[0]["backbone"]
                ref_runs = {r["seed"]: r for r in cand_ref if r["backbone"] == ref_bb}
        if not ref_runs:
            continue
        other_keys = sorted({(r["variant"], r["backbone"]) for r in runs if r["task"] == task and not (r["variant"] == ref_variant and r["backbone"] == ref_bb)})
        for var, bb in other_keys:
            cand_runs = {r["seed"]: r for r in runs if r["task"] == task and r["variant"] == var and r["backbone"] == bb}
            common_seeds = sorted(set(ref_runs.keys()) & set(cand_runs.keys()))
            if not common_seeds:
                continue
            err_ref_all, err_cand_all = [], []
            res_ref_all, res_cand_all = [], []
            for s in common_seeds:
                p_ref = np.load(Path(ref_runs[s]["_dir"]) / "test_predictions.npz")
                p_cnd = np.load(Path(cand_runs[s]["_dir"]) / "test_predictions.npz")
                e_ref = np.sqrt(np.mean((p_ref["y_true"] - p_ref["y_pred"]) ** 2, axis=-1))
                e_cnd = np.sqrt(np.mean((p_cnd["y_true"] - p_cnd["y_pred"]) ** 2, axis=-1))
                err_ref_all.append(e_ref)
                err_cand_all.append(e_cnd)
                res_ref_all.append((p_ref["y_true"] - p_ref["y_pred"])[:, 0])
                res_cand_all.append((p_cnd["y_true"] - p_cnd["y_pred"])[:, 0])
            # Seed-averaged per-sample error & residual vectors avoid inflated repeated-measure p-values
            e_r = np.mean(np.stack(err_ref_all, axis=0), axis=0)
            e_c = np.mean(np.stack(err_cand_all, axis=0), axis=0)
            r_r = np.mean(np.stack(res_ref_all, axis=0), axis=0)
            r_c = np.mean(np.stack(res_cand_all, axis=0), axis=0)
            w_stat, w_p = wilcoxon(e_r, e_c) if np.any(np.abs(e_r - e_c) > 1e-12) else (0.0, 1.0)
            ks_stat, ks_p = ks_2samp(r_r, r_c)
            sig_rows.append({
                "task": task,
                "ref_variant": ref_variant,
                "ref_backbone": ref_bb,
                "cmp_variant": var,
                "cmp_backbone": bb,
                "n_common_seeds": len(common_seeds),
                "n_test_samples": int(len(e_r)),
                "ref_mean_sample_rmse": float(np.mean(e_r)),
                "cmp_mean_sample_rmse": float(np.mean(e_c)),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(w_p),
                "ks2samp_stat": float(ks_stat),
                "ks2samp_p": float(ks_p),
            })

    df_sig = pd.DataFrame(sig_rows)
    df_sig.to_csv(out_dir / "significance_tests.csv", index=False)
    print(f"[STATS] Saved {len(df_sum)} grouped rows and {len(df_sig)} significance tests to {out_dir}")


def extract_va_from_text(resp: str, scale_min: float = 1.0, scale_max: float = 9.0) -> Optional[Tuple[float, float]]:
    m = re.search(r"([1-9](?:\.\d+)?)\s*[#,，\s/]+\s*([1-9](?:\.\d+)?)", resp)
    if m:
        v = float(np.clip(float(m.group(1)), scale_min, scale_max))
        a = float(np.clip(float(m.group(2)), scale_min, scale_max))
        return v, a
    nums = re.findall(r"\d+(?:\.\d+)?", resp)
    if len(nums) >= 2:
        v = float(np.clip(float(nums[0]), scale_min, scale_max))
        a = float(np.clip(float(nums[1]), scale_min, scale_max))
        return v, a
    return None


def run_llm_baseline(args) -> None:
    """Zero-shot modern instruction LLM baseline (e.g. Qwen2.5-3B/7B-Instruct) on DimABSA2026."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.train_eval import load_task_splits, set_seed

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LLM] Loading {args.llm_model} on {device}...")
    tok = AutoTokenizer.from_pretrained(args.llm_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for task in args.tasks:
        train_df, _, _, test_df, target_cols, bounds, has_aspect = load_task_splits(task, seed=args.seed)
        v_tr = train_df["valence"].to_numpy()
        a_tr = train_df["arousal"].to_numpy()
        tail_bounds = {
            "V": (float(np.quantile(v_tr, 0.10)), float(np.quantile(v_tr, 0.90))),
            "A": (float(np.quantile(a_tr, 0.10)), float(np.quantile(a_tr, 0.90))),
        }

        preds_v, preds_a = [], []
        parsed_ok = 0
        t0 = time.perf_counter()

        texts = [to_simplified(t) for t in test_df["text"].astype(str).tolist()]
        aspects = [to_simplified(a) for a in test_df["aspect"].astype(str).tolist()] if has_aspect else ["整体"] * len(texts)

        prompts = []
        for txt, asp in zip(texts, aspects):
            prompt = (
                "请根据文本判断给定目标方面词的情感效价(Valence，1.00极负面~5.00中性~9.00极正面)"
                "和唤醒度(Arousal，1.00极平静~9.00极强烈)。"
                "只输出两个保留两位小数的数字，格式严格为 V#A（例如 6.50#5.80），不要输出其他任何解释。\n"
                f"文本：{txt}\n方面词：{asp}\n输出："
            )
            messages = [{"role": "user", "content": prompt}]
            if hasattr(tok, "apply_chat_template"):
                prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            else:
                prompts.append(prompt)

        bs = 32
        with torch.no_grad():
            for start in range(0, len(prompts), bs):
                batch_p = prompts[start : start + bs]
                enc = tok(batch_p, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
                gen = model.generate(
                    **enc,
                    max_new_tokens=16,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
                inp_len = enc["input_ids"].shape[1]
                for j in range(len(batch_p)):
                    resp = tok.decode(gen[j, inp_len:], skip_special_tokens=True).strip()
                    va = extract_va_from_text(resp, bounds[0], bounds[1])
                    if va is not None:
                        parsed_ok += 1
                        preds_v.append(va[0])
                        preds_a.append(va[1])
                    else:
                        preds_v.append(5.0)
                        preds_a.append(5.0)

        elapsed_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, len(test_df))
        vt = test_df["valence"].to_numpy(dtype=np.float64)
        at = test_df["arousal"].to_numpy(dtype=np.float64)
        vp = np.asarray(preds_v, dtype=np.float64)
        ap = np.asarray(preds_a, dtype=np.float64)

        m = regression_metrics_va(vt, at, vp, ap, tail_bounds=tail_bounds)
        m["parse_rate"] = float(parsed_ok / max(1, len(test_df)))
        m["latency_ms"] = float(elapsed_ms)
        all_metrics[task] = m
        np.savez(
            out_dir / f"{task}__llm_preds.npz",
            y_true=np.stack([vt, at], axis=-1),
            y_pred=np.stack([vp, ap], axis=-1),
        )
        print(f"[LLM] {task}: RMSE_VA={m['RMSE_VA']:.4f}, PCC_V={m['PCC_V']:.4f}, CCC_V={m['CCC_V']:.4f}, lat={elapsed_ms:.2f}ms/sample")

    with (out_dir / "llm_baseline_results.json").open("w", encoding="utf-8") as f:
        json.dump({"model": args.llm_model, "results": all_metrics}, f, ensure_ascii=False, indent=2)


def generate_figures(args) -> None:
    """Render publication-quality figures strictly from verified experiment outputs."""
    sns.set_style("whitegrid")
    configure_matplotlib_fonts()
    fig_dir = Path(args.out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # 1. MOBO Pareto + fANOVA Importance
    mobo_dir = Path(args.mobo_dir) if args.mobo_dir else None
    if mobo_dir and (mobo_dir / "mobo_trials.csv").exists() and (mobo_dir / "mobo_fanova.json").exists():
        df_t = pd.read_csv(mobo_dir / "mobo_trials.csv")
        fanova = json.loads((mobo_dir / "mobo_fanova.json").read_text(encoding="utf-8"))
        best_cfg = json.loads((mobo_dir / "mobo_best_config.json").read_text(encoding="utf-8"))

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        non_p = df_t[~df_t["is_pareto"]]
        pareto = df_t[df_t["is_pareto"]].sort_values("obj_err_ccc")
        sc = axes[0].scatter(
            non_p["obj_err_ccc"], non_p["obj_winkler"],
            c=non_p["obj_latency_ms"], cmap="viridis", alpha=0.65, s=48, label="Explored Trials"
        )
        axes[0].plot(pareto["obj_err_ccc"], pareto["obj_winkler"], "r--", lw=1.4, alpha=0.8)
        axes[0].scatter(
            pareto["obj_err_ccc"], pareto["obj_winkler"],
            c="crimson", edgecolors="black", s=95, marker="D", zorder=5, label="Pareto Frontier"
        )
        ko = best_cfg["knee_objectives"]
        axes[0].scatter(
            [ko["composite_err_ccc"]], [ko["winkler_90"]],
            c="gold", edgecolors="black", s=200, marker="*", zorder=6, label=f"Selected Knee (T{best_cfg['knee_trial']})"
        )
        cbar = plt.colorbar(sc, ax=axes[0])
        cbar.set_label("Latency (ms/sample)")
        axes[0].set_xlabel("Composite Error + (1 - Mean CCC)  ↓")
        axes[0].set_ylabel("90% Conformal Winkler Score  ↓")
        axes[0].set_title("(a) BoTorch qEHVI Multi-Objective Pareto Frontier")
        axes[0].legend(frameon=True, fontsize=8.5)

        # fANOVA Heatmap across 3 objectives
        objs = [k for k in ("composite_err_ccc", "winkler_90", "latency_ms") if isinstance(fanova.get(k), dict) and "error" not in fanova[k]]
        if objs:
            params = sorted({p for o in objs for p in fanova[o].keys()})
            mat = np.array([[fanova[o].get(p, 0.0) for o in objs] for p in params])
            sns.heatmap(
                mat, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=["Error+CCC", "Winkler-90", "Latency"],
                yticklabels=params, ax=axes[1]
            )
            axes[1].set_title("(b) fANOVA Hyperparameter Importance")
        plt.tight_layout()
        p_out = fig_dir / "fig1_mobo_pareto_fanova.png"
        fig.savefig(p_out, dpi=300)
        fig.savefig(fig_dir / "fig1_mobo_pareto_fanova.pdf")
        plt.close(fig)
        generated.append(str(p_out))

    # 2. Anti-Mean-Regression KDE & Tail-20% Comparison
    exp_dir = Path(args.exp_dir) if args.exp_dir else None
    if exp_dir and exp_dir.exists():
        # Compare representative seed=42 runs on primary domain dimabsa_zho_rest
        target_task = args.plot_task
        variants = [
            ("classic_bilstm", "Classic BiLSTM (Proposal)", "#7f7f7f"),
            ("plm_only_mse", "MacBERT-MSE", "#d62728"),
            ("concept_bilstm_no_ccc", "Ours w/o CCC", "#ff7f0e"),
            ("concept_bilstm_urer", "Ours (Concept-BiLSTM-URER)", "#1f77b4"),
        ]
        loaded_preds = []
        for var, lbl, col in variants:
            p42 = exp_dir / f"{target_task}__{var}__seed42" / "test_predictions.npz"
            cand = [p42] if p42.exists() else sorted(exp_dir.glob(f"{target_task}__{var}__seed*/test_predictions.npz"))
            if cand:
                npz = np.load(cand[0])
                loaded_preds.append((var, lbl, col, npz))

        if loaded_preds:
            fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
            y_gt = loaded_preds[-1][3]["y_true"][:, 0]
            sns.kdeplot(y_gt, ax=axes[0], color="black", lw=2.2, ls="--", label="Ground Truth (Valence)")
            for _, lbl, col, npz in loaded_preds:
                yp = npz["y_pred"][:, 0]
                ks_s, _ = ks_2samp(y_gt, yp)
                sns.kdeplot(yp, ax=axes[0], color=col, lw=1.8, label=f"{lbl} (K-S={ks_s:.3f})")
            axes[0].set_xlabel("Valence Score [1, 9]")
            axes[0].set_ylabel("Density")
            axes[0].set_title(f"(a) Prediction vs. Ground-Truth Distribution ({target_task})")
            axes[0].legend(fontsize=8)

            # Tail-20% RMSE bar chart across seeds if summary_table.csv exists
            sum_csv = exp_dir / "summary_table.csv"
            if sum_csv.exists():
                df_s = pd.read_csv(sum_csv)
                sub_s = df_s[df_s["task"] == target_task]
                plot_vars = [v[0] for v in variants if v[0] in set(sub_s["variant"])]
                means_v = [float(sub_s[sub_s["variant"] == v]["Tail20_RMSE_V_mean"].iloc[0]) for v in plot_vars]
                stds_v = [float(sub_s[sub_s["variant"] == v]["Tail20_RMSE_V_std"].iloc[0]) for v in plot_vars]
                lbls_v = [dict((v[0], v[1]) for v in variants)[v] for v in plot_vars]
                cols_v = [dict((v[0], v[2]) for v in variants)[v] for v in plot_vars]
                x = np.arange(len(plot_vars))
                axes[1].bar(x, means_v, yerr=stds_v, capsize=5, color=cols_v, alpha=0.88)
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(lbls_v, rotation=15, ha="right", fontsize=8.5)
                axes[1].set_ylabel("Tail-20% Extreme Valence RMSE  ↓")
                axes[1].set_title("(b) Mean-Regression Mitigation on Extreme Sentiment Tails")
            plt.tight_layout()
            p_out = fig_dir / "fig2_anti_mean_regression.png"
            fig.savefig(p_out, dpi=300)
            fig.savefig(fig_dir / "fig2_anti_mean_regression.pdf")
            plt.close(fig)
            generated.append(str(p_out))

    # 3. Rhetorical Uncertainty Stratification & OOD AUROC
    probe_json = Path(args.probe_json) if args.probe_json else None
    if probe_json and probe_json.exists():
        pdata = json.loads(probe_json.read_text(encoding="utf-8"))
        by_t = pdata["logic_zero_shot"]["by_type"]
        types_order = [t for t in ["简单句", "转折", "双重否定", "反讽"] if t in by_t]
        type_en = {"简单句": "Simple", "转折": "Adversative", "双重否定": "Double-Neg", "反讽": "Irony"}
        if types_order:
            fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
            epis = [by_t[t]["epi_std_mean"] for t in types_order]
            ales = [by_t[t]["ale_std_mean"] for t in types_order]
            x = np.arange(len(types_order))
            w = 0.36
            axes[0].bar(x - w / 2, epis, width=w, color="#1f77b4", label="Epistemic Uncertainty (σ_epi)")
            axes[0].bar(x + w / 2, ales, width=w, color="#ff7f0e", label="Aleatoric Uncertainty (σ_ale)")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([type_en[t] for t in types_order])
            axes[0].set_ylabel("Mean Uncertainty Std (Valence)")
            axes[0].set_ylim(0.0, max(max(epis), max(ales)) * 1.26)
            axes[0].set_title(f"(a) Uncertainty Across Rhetorical Complexity (Kruskal p={pdata['logic_zero_shot']['kruskal_epi_p']:.2e})")
            axes[0].legend(loc="upper left", fontsize=8.5)

            # Concept activation profile across rhetorical types
            c_keys = ["concept_Negation_Operator_mean", "concept_Adversative_Shift_mean", "concept_Modality_Hedging_mean"]
            c_lbls = ["Negation Gate", "Adversative Gate", "Modality Gate"]
            mat_c = np.array([[by_t[t].get(ck, 0.0) for t in types_order] for ck in c_keys])
            sns.heatmap(
                mat_c, annot=True, fmt=".3f", cmap="OrRd",
                xticklabels=[type_en[t] for t in types_order],
                yticklabels=c_lbls, ax=axes[1]
            )
            axes[1].set_title("(b) Mean Concept Gate Activation by Rhetorical Category")
            plt.tight_layout()
            p_out = fig_dir / "fig3_uncertainty_rhetorical_probe.png"
            fig.savefig(p_out, dpi=300)
            fig.savefig(fig_dir / "fig3_uncertainty_rhetorical_probe.pdf")
            plt.close(fig)
            generated.append(str(p_out))

    # 4. Explainability Faithfulness Curves & Counterfactual Impact
    if exp_dir and (exp_dir / "summary_table.csv").exists():
        # Average faithfulness curves across seeds for concept_bilstm_urer
        f_runs = sorted(exp_dir.glob(f"{args.plot_task}__concept_bilstm_urer__seed*/results.json"))
        f_runs = [json.loads(p.read_text(encoding="utf-8")) for p in f_runs]
        f_runs = [r["explain_results"] for r in f_runs if r.get("explain_results", {}).get("concept_comp_mean")]
        if f_runs:
            fracs = [10, 20, 30, 50]
            fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
            for method, lbl, col, mk in [
                ("concept", "Concept-Gate (Ours)", "#1f77b4", "o"),
                ("ig", "Integrated Gradients", "#2ca02c", "s"),
                ("attention", "Attention Weights", "#d62728", "^"),
            ]:
                comp_m = np.mean([r[f"{method}_comp_mean"] for r in f_runs], axis=0)
                suff_m = np.mean([r[f"{method}_suff_mean"] for r in f_runs], axis=0)
                axes[0].plot(fracs, comp_m, marker=mk, color=col, lw=2.0, label=lbl)
                axes[1].plot(fracs, suff_m, marker=mk, color=col, lw=2.0, label=lbl)
            axes[0].set_xlabel("Top-k% Masked Tokens")
            axes[0].set_ylabel("Comprehensiveness (Δ|y_pred| on Removal)  ↑")
            axes[0].set_title("(a) Faithfulness: Deletion Comprehensiveness")
            axes[0].legend(fontsize=8.5)
            axes[1].set_xlabel("Top-k% Retained Tokens")
            axes[1].set_ylabel("Sufficiency (|y_pred(full) - y_pred(top-k)|)  ↓")
            axes[1].set_title("(b) Faithfulness: Retention Sufficiency")
            axes[1].legend(fontsize=8.5)
            plt.tight_layout()
            p_out = fig_dir / "fig4_explainability_faithfulness.png"
            fig.savefig(p_out, dpi=300)
            fig.savefig(fig_dir / "fig4_explainability_faithfulness.pdf")
            plt.close(fig)
            generated.append(str(p_out))

    # 5. Noise Stress Test (0%, 10%, 20%)
    if exp_dir and (exp_dir / "summary_table.csv").exists():
        df_s = pd.read_csv(exp_dir / "summary_table.csv")
        sub_n = df_s[(df_s["task"] == args.plot_task) & df_s["noise_10_RMSE_VA_mean"].notna()] if "noise_10_RMSE_VA_mean" in df_s.columns else pd.DataFrame()
        if not sub_n.empty:
            plm_sub = sub_n[~sub_n["variant"].astype(str).str.startswith("classic_")]
            n_cols = 2 if not plm_sub.empty else 1
            fig, axes = plt.subplots(1, n_cols, figsize=(12.2 if n_cols == 2 else 6.8, 4.6))
            if n_cols == 1:
                axes = [axes]
            levels = [0, 10, 20]
            for _, row in sub_n.iterrows():
                ys = [row["RMSE_VA_mean"], row["noise_10_RMSE_VA_mean"], row["noise_20_RMSE_VA_mean"]]
                es = [row["RMSE_VA_std"], row["noise_10_RMSE_VA_std"], row["noise_20_RMSE_VA_std"]]
                axes[0].errorbar(levels, ys, yerr=es, marker="o", capsize=4, lw=2.0, label=row["variant"])
            axes[0].set_xticks(levels)
            axes[0].set_xticklabels(["Clean (0%)", "Noise 10%", "Noise 20%"])
            axes[0].set_ylabel("RMSE_VA  ↓")
            axes[0].set_title(f"(a) All Baselines Noise Stress ({args.plot_task})")
            axes[0].legend(fontsize=7.5)
            if n_cols == 2:
                for _, row in plm_sub.iterrows():
                    ys = [row["RMSE_VA_mean"], row["noise_10_RMSE_VA_mean"], row["noise_20_RMSE_VA_mean"]]
                    es = [row["RMSE_VA_std"], row["noise_10_RMSE_VA_std"], row["noise_20_RMSE_VA_std"]]
                    axes[1].errorbar(levels, ys, yerr=es, marker="s", capsize=4, lw=2.0, label=row["variant"])
                axes[1].set_xticks(levels)
                axes[1].set_xticklabels(["Clean (0%)", "Noise 10%", "Noise 20%"])
                axes[1].set_ylabel("RMSE_VA  ↓")
                axes[1].set_title(f"(b) PLM & Concept Zoom-In ({args.plot_task})")
                axes[1].legend(fontsize=8)
            plt.tight_layout()
            p_out = fig_dir / "fig5_noise_stress_test.png"
            fig.savefig(p_out, dpi=300)
            fig.savefig(fig_dir / "fig5_noise_stress_test.pdf")
            plt.close(fig)
            generated.append(str(p_out))

    # 6. Case Study Token-Concept Activation Heatmaps
    if probe_json and probe_json.exists():
        pdata = json.loads(probe_json.read_text(encoding="utf-8"))
        cases = pdata.get("cases", [])
        # Pick up to 3 illustrative non-empty cases with balanced Pos/Neg polarity
        picked = []
        for target_rt, pref_lbl in (("转折", 1), ("双重否定", 0), ("反讽", 1)):
            cands = [c for c in cases if c.get("rhetorical_type") == target_rt and c.get("explicit_concepts")]
            if not cands:
                continue
            exact = [c for c in cands if c.get("gold_label") == pref_lbl]
            picked.append(exact[0] if exact else cands[0])
        if picked:
            fig, axes = plt.subplots(len(picked), 1, figsize=(13.0, 3.2 * len(picked)))
            if len(picked) == 1:
                axes = [axes]
            type_en = {"简单句": "Simple", "转折": "Adversative", "双重否定": "Double-Neg", "反讽": "Irony"}
            for ax, c in zip(axes, picked):
                toks = c["tokens"][:28]
                mat = np.asarray(c["explicit_concepts"], dtype=np.float32)[:28, :].T
                sns.heatmap(
                    mat, cmap="YlOrRd", vmin=0.0, vmax=1.0,
                    xticklabels=toks, yticklabels=CONCEPT_NAMES, ax=ax, cbar=True
                )
                ax.set_xticklabels(toks, rotation=35, ha="right", fontsize=8)
                rt_en = type_en.get(c["rhetorical_type"], c["rhetorical_type"])
                lbl_s = "Pos" if c["gold_label"] == 1 else "Neg"
                ax.set_title(
                    f"[{rt_en} | Gold={lbl_s}] V_pred={c['pred_valence']:.2f}, "
                    f"σ_epi={c['epi_std_v']:.3f}, σ_ale={c['ale_std_v']:.3f}, "
                    f"CF(Neg=0) ΔV={c['cf_zero_negation_delta_v']:+.3f}"
                )
            plt.tight_layout()
            p_out = fig_dir / "fig6_concept_case_heatmaps.png"
            fig.savefig(p_out, dpi=300)
            fig.savefig(fig_dir / "fig6_concept_case_heatmaps.pdf")
            plt.close(fig)
            generated.append(str(p_out))

    # 7. Ablation Contributions & Uncalibrated vs Split Conformal Comparison
    out_root = exp_dir.parent if exp_dir else None
    if exp_dir and out_root and (exp_dir / "summary_table.csv").exists() and (out_root / "ablation_suite" / "summary_table.csv").exists():
        df_m = pd.read_csv(exp_dir / "summary_table.csv")
        df_a = pd.read_csv(out_root / "ablation_suite" / "summary_table.csv")
        df_comb = pd.concat([df_m[df_m["variant"] == "concept_bilstm_urer"], df_a], ignore_index=True)
        sub_ab = df_comb[df_comb["task"] == args.plot_task]
        ab_order = [
            ("concept_bilstm_urer", "Full Model (Ours)"),
            ("ib_residual_bilstm", "+IB Residual Bypass"),
            ("plm_bilstm_no_concept", "w/o Concept Bottleneck"),
            ("concept_no_bilstm", "w/o BiLSTM-Res"),
            ("concept_bilstm_no_ccc", "w/o CCC Loss"),
            ("concept_bilstm_no_dyn", "w/o URER+LDS Reweight"),
            ("concept_bilstm_det", "w/o DER (MC-Dropout)"),
        ]
        rows_ab = [sub_ab[sub_ab["variant"] == v].iloc[0] for v, _ in ab_order if v in set(sub_ab["variant"])]
        lbls_ab = [lbl for v, lbl in ab_order if v in set(sub_ab["variant"])]
        if rows_ab:
            fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
            x = np.arange(len(rows_ab))
            rmses = [float(r["RMSE_VA_mean"]) for r in rows_ab]
            r_stds = [float(r["RMSE_VA_std"]) for r in rows_ab]
            cols_ab = ["#1f77b4"] + ["#ff7f0e"] * (len(rows_ab) - 1)
            axes[0].bar(x, rmses, yerr=r_stds, capsize=4, color=cols_ab, alpha=0.88)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(lbls_ab, rotation=22, ha="right", fontsize=8.5)
            axes[0].set_ylabel("RMSE_VA  ↓")
            axes[0].set_title(f"(a) Architectural & Loss Ablation Study ({args.plot_task})")

            # Conformal vs Uncalibrated DER comparison
            full_r = rows_ab[0]
            comp_bars = []
            if "uncal_Winkler_avg_mean" in full_r and pd.notna(full_r["uncal_Winkler_avg_mean"]):
                comp_bars.append(("DER Uncalibrated", float(full_r["uncal_PICP_V_mean"]), float(full_r["uncal_Winkler_avg_mean"]), "#d62728"))
            if "concept_bilstm_det" in set(sub_ab["variant"]):
                det_r = sub_ab[sub_ab["variant"] == "concept_bilstm_det"].iloc[0]
                comp_bars.append(("MC-Dropout Conformal", float(det_r["PICP_V_mean"]), float(det_r["Winkler_avg_mean"]), "#ff7f0e"))
            comp_bars.append(("URER + Split Conformal", float(full_r["PICP_V_mean"]), float(full_r["Winkler_avg_mean"]), "#1f77b4"))
            xc = np.arange(len(comp_bars))
            w = 0.36
            ax2 = axes[1].twinx()
            axes[1].bar(xc - w / 2, [b[1] * 100.0 for b in comp_bars], width=w, color=[b[3] for b in comp_bars], alpha=0.85, label="PICP (%)")
            ax2.bar(xc + w / 2, [b[2] for b in comp_bars], width=w, color="#2ca02c", alpha=0.75, label="Winkler-90 ↓")
            axes[1].axhline(90.0, color="black", ls="--", lw=1.2, label="Nominal 90%")
            axes[1].set_xticks(xc)
            axes[1].set_xticklabels([b[0] for b in comp_bars], fontsize=8.5)
            axes[1].set_ylabel("Empirical Coverage PICP (%)")
            axes[1].set_ylim(0.0, 118.0)
            ax2.set_ylabel("Mean Winkler Score  ↓")
            axes[1].set_title("(b) Uncertainty Interval Calibration: Conformal vs. Raw DER")
            h1, l1 = axes[1].get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            axes[1].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
            plt.tight_layout()
            p_out = fig_dir / "fig7_ablation_and_conformal.png"
            fig.savefig(p_out, dpi=300)
            fig.savefig(fig_dir / "fig7_ablation_and_conformal.pdf")
            plt.close(fig)
            generated.append(str(p_out))

    # 8. Data Efficiency (25%/50%/100%), Cross-Domain Transfer, & OOD AUROC
    if exp_dir and out_root:
        has_eff = all((out_root / d / "summary_table.csv").exists() for d in ("data_eff_frac25", "data_eff_frac50")) and (exp_dir / "summary_table.csv").exists()
        has_cross = (out_root / "cross_ood_suite" / "summary_table.csv").exists()
        has_ood = probe_json is not None and probe_json.exists()
        if has_eff or has_cross or has_ood:
            n_pan = int(has_eff) + int(has_cross) + int(has_ood)
            fig, axes = plt.subplots(1, n_pan, figsize=(5.2 * n_pan, 4.4))
            if n_pan == 1:
                axes = [axes]
            idx_ax = 0
            if has_eff:
                ax = axes[idx_ax]; idx_ax += 1
                df25 = pd.read_csv(out_root / "data_eff_frac25" / "summary_table.csv")
                df50 = pd.read_csv(out_root / "data_eff_frac50" / "summary_table.csv")
                df100 = pd.read_csv(exp_dir / "summary_table.csv")
                df100 = df100[df100["task"] == args.plot_task]
                fracs_eff = [25, 50, 100]
                for var, lbl, col in [("classic_bilstm", "Classic BiLSTM", "#7f7f7f"), ("plm_only_mse", "MacBERT-MSE", "#d62728"), ("concept_bilstm_urer", "Ours (URER)", "#1f77b4")]:
                    ys, es = [], []
                    for dff in (df25, df50, df100):
                        sub = dff[dff["variant"] == var]
                        ys.append(float(sub["RMSE_VA_mean"].iloc[0]) if not sub.empty else np.nan)
                        es.append(float(sub["RMSE_VA_std"].iloc[0]) if not sub.empty else 0.0)
                    ax.errorbar(fracs_eff, ys, yerr=es, marker="o", capsize=4, lw=2.0, color=col, label=lbl)
                ax.set_xticks(fracs_eff)
                ax.set_xticklabels(["25%", "50%", "100%"])
                ax.set_xlabel("Labeled Training Data Ratio")
                ax.set_ylabel("RMSE_VA  ↓")
                ax.set_title("(a) Label Efficiency Curve")
                ax.legend(fontsize=8)
            if has_cross:
                ax = axes[idx_ax]; idx_ax += 1
                dfc = pd.read_csv(out_root / "cross_ood_suite" / "summary_table.csv")
                order_c = ["classic_bilstm", "plm_only_mse", "plm_only_ccc", "concept_bilstm_urer"]
                dfc_sub = [dfc[dfc["variant"] == v].iloc[0] for v in order_c if v in set(dfc["variant"])]
                xc = np.arange(len(dfc_sub))
                ax.bar(xc, [float(r["RMSE_VA_mean"]) for r in dfc_sub], yerr=[float(r["RMSE_VA_std"]) for r in dfc_sub], capsize=4, color=["#7f7f7f", "#d62728", "#ff7f0e", "#1f77b4"][:len(dfc_sub)], alpha=0.88)
                ax.set_xticks(xc)
                ax.set_xticklabels([r["variant"] for r in dfc_sub], rotation=18, ha="right", fontsize=8)
                ax.set_ylabel("Zero-Shot Cross-Domain RMSE_VA  ↓")
                ax.set_title("(b) Cross-Domain OOD (Rest+Lap → Fin)")
            if has_ood:
                ax = axes[idx_ax]
                pdata = json.loads(probe_json.read_text(encoding="utf-8"))
                ood_d = pdata.get("ood_detection", pdata.get("ood", {}))
                keys_o = list(ood_d.keys())
                xo = np.arange(len(keys_o))
                w = 0.36
                ax.bar(xo - w / 2, [ood_d[k]["auroc_epi"] for k in keys_o], width=w, color="#1f77b4", label="Epistemic (σ_epi)")
                ax.bar(xo + w / 2, [ood_d[k]["auroc_tot"] for k in keys_o], width=w, color="#ff7f0e", label="Total (σ_tot)")
                ax.axhline(0.5, color="black", ls="--", lw=1.0)
                ax.set_xticks(xo)
                ax.set_xticklabels([k.replace("dimabsa_zho_", "").replace("logic_rhetorical_shift", "rhetoric") for k in keys_o], rotation=15, ha="right", fontsize=8.5)
                ax.set_ylabel("OOD Detection AUROC  ↑")
                ax.set_title("(c) Cross-Domain / Rhetorical OOD AUROC")
                ax.legend(fontsize=8)
            plt.tight_layout()
            p_out = fig_dir / "fig8_data_efficiency_and_ood.png"
            fig.savefig(p_out, dpi=300)
            fig.savefig(fig_dir / "fig8_data_efficiency_and_ood.pdf")
            plt.close(fig)
            generated.append(str(p_out))

    # 9. Cross-Backbone Generalization & Zero-Shot LLM Comparison
    if exp_dir and out_root:
        bb_dfs = []
        if (exp_dir / "summary_table.csv").exists():
            df_mac = pd.read_csv(exp_dir / "summary_table.csv")
            df_mac = df_mac[(df_mac["task"].isin(["dimabsa_zho_rest", "dimabsa_zho_fin"])) & (df_mac["variant"].isin(["plm_only_mse", "concept_bilstm_urer"]))].copy()
            if not df_mac.empty:
                df_mac["bb_short"] = "MacBERT"
                bb_dfs.append(df_mac)
        for tag, lbl_bb in (("backbone_roberta", "RoBERTa-wwm"), ("backbone_deberta", "DeBERTa-v2")):
            p_bb = out_root / tag / "summary_table.csv"
            if p_bb.exists():
                d_bb = pd.read_csv(p_bb)
                d_bb["bb_short"] = lbl_bb
                bb_dfs.append(d_bb)
        p_llm = out_root / "llm_baseline" / "llm_baseline_results.json"
        has_bb = len(bb_dfs) >= 2
        has_llm = p_llm.exists() and (exp_dir / "summary_table.csv").exists()
        if has_bb or has_llm:
            n_p9 = int(has_bb) + int(has_llm)
            fig, axes = plt.subplots(1, n_p9, figsize=(6.2 * n_p9, 4.6))
            if n_p9 == 1:
                axes = [axes]
            ax_i = 0
            if has_bb:
                ax = axes[ax_i]; ax_i += 1
                df_all_bb = pd.concat(bb_dfs, ignore_index=True)
                groups = [
                    (t, b)
                    for t in ["dimabsa_zho_rest", "dimabsa_zho_fin"]
                    for b in ["MacBERT", "RoBERTa-wwm", "DeBERTa-v2"]
                    if not df_all_bb[(df_all_bb["task"] == t) & (df_all_bb["bb_short"] == b) & (df_all_bb["variant"] == "plm_only_mse")].empty
                    and not df_all_bb[(df_all_bb["task"] == t) & (df_all_bb["bb_short"] == b) & (df_all_bb["variant"] == "concept_bilstm_urer")].empty
                ]
                if groups:
                    x = np.arange(len(groups))
                    w = 0.36
                    y_mse = [float(df_all_bb[(df_all_bb["task"] == t) & (df_all_bb["bb_short"] == b) & (df_all_bb["variant"] == "plm_only_mse")]["RMSE_VA_mean"].iloc[0]) for t, b in groups]
                    y_our = [float(df_all_bb[(df_all_bb["task"] == t) & (df_all_bb["bb_short"] == b) & (df_all_bb["variant"] == "concept_bilstm_urer")]["RMSE_VA_mean"].iloc[0]) for t, b in groups]
                    ax.bar(x - w / 2, y_mse, width=w, color="#d62728", alpha=0.85, label="PLM-Only (MSE)")
                    ax.bar(x + w / 2, y_our, width=w, color="#1f77b4", alpha=0.85, label="Concept-BiLSTM-URER (Ours)")
                    ax.set_xticks(x)
                    ax.set_xticklabels([f"{t.replace('dimabsa_zho_', '')}\n{b}" for t, b in groups], fontsize=8)
                    ax.set_ylim(0.0, max(max(y_mse), max(y_our)) * 1.25)
                    ax.set_ylabel("RMSE_VA  ↓")
                    ax.set_title("(a) Cross-Backbone Generalization (3 Backbones)")
                    ax.legend(fontsize=8)
            if has_llm:
                ax = axes[ax_i]
                raw_llm = json.loads(p_llm.read_text(encoding="utf-8"))
                llm_d = raw_llm.get("results") or raw_llm.get("tasks") or {}
                df_m = pd.read_csv(exp_dir / "summary_table.csv")
                doms = [
                    t for t in ["dimabsa_zho_rest", "dimabsa_zho_lap", "dimabsa_zho_fin", "dimabsa_zho_env"]
                    if t in llm_d
                    and not df_m[(df_m["task"] == t) & (df_m["variant"] == "plm_only_mse")].empty
                    and not df_m[(df_m["task"] == t) & (df_m["variant"] == "concept_bilstm_urer")].empty
                ]
                if doms:
                    x = np.arange(len(doms))
                    w = 0.26
                    y_llm = [float(llm_d[t]["RMSE_VA"]) for t in doms]
                    y_mse = [float(df_m[(df_m["task"] == t) & (df_m["variant"] == "plm_only_mse")]["RMSE_VA_mean"].iloc[0]) for t in doms]
                    y_our = [float(df_m[(df_m["task"] == t) & (df_m["variant"] == "concept_bilstm_urer")]["RMSE_VA_mean"].iloc[0]) for t in doms]
                    ax.bar(x - w, y_llm, width=w, color="#9467bd", alpha=0.85, label="Qwen2.5-3B (Zero-Shot)")
                    ax.bar(x, y_mse, width=w, color="#d62728", alpha=0.85, label="MacBERT-MSE")
                    ax.bar(x + w, y_our, width=w, color="#1f77b4", alpha=0.85, label="Concept-BiLSTM-URER (1.01M)")
                    ax.set_xticks(x)
                    ax.set_xticklabels([t.replace("dimabsa_zho_", "") for t in doms], fontsize=8.5)
                    ax.set_ylim(0.0, max(max(y_llm), max(y_mse), max(y_our)) * 1.32)
                    ax.set_ylabel("RMSE_VA  ↓")
                    ax.set_title("(b) Fine-Tuned Bottleneck vs. Zero-Shot LLM Baseline")
                    ax.legend(fontsize=8)
            plt.tight_layout()
            p_out = fig_dir / "fig9_backbones_and_llm.png"
            fig.savefig(p_out, dpi=300)
            fig.savefig(fig_dir / "fig9_backbones_and_llm.pdf")
            plt.close(fig)
            generated.append(str(p_out))

    # 10. Option C Causal-Orthogonality Mechanism Patch Comparison
    if exp_dir and out_root and (out_root / "mechanism_patch").exists():
        base_runs = sorted(exp_dir.glob(f"{args.plot_task}__concept_bilstm_urer__seed*/results.json"))
        patch_runs = sorted((out_root / "mechanism_patch").glob(f"{args.plot_task}__concept_bilstm_causal__seed*/results.json"))
        if base_runs and patch_runs:
            b_js = [json.loads(p.read_text(encoding="utf-8")) for p in base_runs]
            p_js = [json.loads(p.read_text(encoding="utf-8")) for p in patch_runs]
            b_ex = [r["explain_results"] for r in b_js if r.get("explain_results")]
            p_ex = [r["explain_results"] for r in p_js if r.get("explain_results")]
            if b_ex and p_ex:
                fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
                m_keys = [("cf_pos_cue_dir_acc", "Pos-Cue CDA ↑"), ("cf_neg_cue_dir_acc", "Neg-Cue CDA ↑"), ("concept_bg_mean", "Bg Leakage ↓")]
                xm = np.arange(len(m_keys))
                w = 0.35
                vb = [float(np.mean([e.get(k, 0.0) for e in b_ex])) for k, _ in m_keys]
                vp = [float(np.mean([e.get(k, 0.0) for e in p_ex])) for k, _ in m_keys]
                axes[0].bar(xm - w / 2, vb, width=w, color="#ff7f0e", alpha=0.85, label="Concept-BiLSTM-URER (Unconstrained)")
                axes[0].bar(xm + w / 2, vp, width=w, color="#1f77b4", alpha=0.85, label="Concept-BiLSTM-Causal (Monotonic+Ortho)")
                axes[0].axhline(0.5, color="black", ls=":", lw=1.0)
                axes[0].set_xticks(xm)
                axes[0].set_xticklabels([lbl for _, lbl in m_keys], fontsize=8.5)
                axes[0].set_ylim(0.0, 1.12)
                axes[0].set_ylabel("Ratio / Mean Activation")
                axes[0].set_title(f"(a) Causal Direction & Background Suppression ({args.plot_task})")
                axes[0].legend(fontsize=8)

                fracs = [10, 20, 30, 50]
                for arr_ex, col, lbl_c in ((b_ex, "#ff7f0e", "Unconstrained Concept-Gate"), (p_ex, "#1f77b4", "Causal Concept-Gate (Patch)")):
                    c_curves = [e["concept_comp_mean"] for e in arr_ex if "concept_comp_mean" in e]
                    if c_curves:
                        axes[1].plot(fracs, np.mean(c_curves, axis=0), marker="o", lw=2.0, color=col, label=lbl_c)
                ig_curves = [e["ig_comp_mean"] for e in p_ex if "ig_comp_mean" in e]
                if ig_curves:
                    axes[1].plot(fracs, np.mean(ig_curves, axis=0), marker="s", lw=1.8, ls="--", color="#2ca02c", label="Integrated Gradients")
                axes[1].set_xlabel("Top-k% Masked Tokens")
                axes[1].set_ylabel("Comprehensiveness (Δ|y_pred| on Removal)  ↑")
                axes[1].set_title("(b) Deletion Faithfulness Before vs. After Causal Patch")
                axes[1].legend(fontsize=8)
                plt.tight_layout()
                p_out = fig_dir / "fig10_causal_patch_comparison.png"
                fig.savefig(p_out, dpi=300)
                fig.savefig(fig_dir / "fig10_causal_patch_comparison.pdf")
                plt.close(fig)
                generated.append(str(p_out))

    print(json.dumps({"generated_figures": generated}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluation, statistical tests, LLM baseline, and figure suite.")
    p.add_argument("--mode", choices=["probe", "stats", "llm_baseline", "plot"], required=True)
    p.add_argument("--ckpt-dir", type=str, default="")
    p.add_argument("--exp-dir", type=str, default="")
    p.add_argument("--ref-exp-dir", type=str, default="")
    p.add_argument("--mobo-dir", type=str, default="")
    p.add_argument("--probe-json", type=str, default="")
    p.add_argument("--out-dir", type=str, default="outputs/analysis")
    p.add_argument("--plot-task", type=str, default="dimabsa_zho_rest")
    p.add_argument("--ref-variant", type=str, default="concept_bilstm_urer")
    p.add_argument("--ref-backbone", type=str, default="hfl/chinese-macbert-base")
    p.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--tasks", nargs="+", default=["dimabsa_zho_rest", "dimabsa_zho_lap", "dimabsa_zho_fin", "dimabsa_zho_env"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.mode == "probe":
        run_rhetorical_probe(args)
    elif args.mode == "stats":
        run_stats_aggregation(args)
    elif args.mode == "llm_baseline":
        run_llm_baseline(args)
    elif args.mode == "plot":
        generate_figures(args)


if __name__ == "__main__":
    main()
