"""Verify dataset integrity, hyperparameter configs, visual assets, and repository hygiene."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
OUT_ROOT = ROOT / "outputs"


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[FAIL] {msg}")


def approx(a: float, b: float, tol: float, label: str) -> None:
    check(abs(a - b) <= tol, f"{label}: got {a}, expected {b} +/- {tol}")


def verify_datasets() -> None:
    dim = pd.read_parquet(DATA_PROC / "dimabsa2026_zho_all.parquet")
    br = pd.read_parquet(DATA_PROC / "brighter_chn_intensities.parquet")
    lg = pd.read_parquet(DATA_PROC / "logic_sentiment_zho.parquet")
    total = len(dim) + len(br) + len(lg)
    check(len(dim) == 27558, f"DimABSA rows={len(dim)}")
    check(len(br) == 5484, f"BRIGHTER-CHN rows={len(br)}")
    check(len(lg) == 3681, f"LogicSentiment rows={len(lg)}")
    check(total == 36723, f"Total rows={total}")

    for dom in ("zho_rest", "zho_lap", "zho_fin", "zho_env"):
        sub = dim[dim["domain"] == dom]
        tr = set(sub[sub["split_cal"] == "train_sub"]["text_id"])
        ca = set(sub[sub["split_cal"] == "cal"]["text_id"])
        dv = set(sub[sub["split"] == "dev"]["text_id"])
        ts = set(sub[sub["split"] == "test"]["text_id"])
        check(
            len(tr & ca) == 0 and len(tr & dv) == 0 and len(tr & ts) == 0 and len(dv & ts) == 0,
            f"Leakage detected in {dom}",
        )
    print(f"[OK] Datasets verified: {total:,} instances, 0 split overlaps.")


def verify_configs_and_assets() -> None:
    cfg_yaml = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    check("training" in cfg_yaml and cfg_yaml["training"]["seeds"] == [42, 102, 202, 302, 402], "Invalid seeds in default.yaml")
    knee = json.loads((ROOT / "configs" / "trial12_knee.json").read_text(encoding="utf-8"))
    check(knee["knee_trial"] == 12 and knee["n_trials"] == 24, "Unexpected trial12_knee.json metadata")
    fig1 = ROOT / "assets" / "fig1_architecture.png"
    check(fig1.exists() and fig1.stat().st_size > 10000, f"Missing architecture overview image: {fig1}")
    print("[OK] Configs and README architecture figure verified.")


def verify_runtime_outputs() -> None:
    ms = OUT_ROOT / "main_suite"
    df = pd.read_csv(ms / "summary_table.csv")

    def row(task: str, variant: str) -> pd.Series:
        sub = df[(df["task"] == task) & (df["variant"] == variant)]
        check(len(sub) == 1, f"Missing summary row {task}/{variant}")
        return sub.iloc[0]

    r_fin = row("dimabsa_zho_fin", "concept_bilstm_urer")
    approx(r_fin["RMSE_VA_mean"], 0.5128, 5e-4, "Fin URER RMSE_VA")
    approx(r_fin["CCC_V_mean"], 0.8464, 5e-4, "Fin URER CCC_V")
    approx(r_fin["Tail20_RMSE_V_mean"], 0.5003, 5e-4, "Fin URER Tail20_V")
    check(int(r_fin["trainable_params_mean"]) == 1006805, "Expected 1,006,805 URER params")

    r_rest = row("dimabsa_zho_rest", "concept_bilstm_urer")
    approx(r_rest["RMSE_VA_mean"], 0.9730, 5e-4, "Rest URER RMSE_VA")
    approx(r_rest["StdRatio_V_mean"], 0.8400, 5e-4, "Rest URER StdRatio_V")

    r_logic = row("logic_sentiment", "concept_bilstm_urer")
    approx(r_logic["Acc_mean"] * 100.0, 97.17, 0.05, "Logic URER Acc")
    approx(r_logic["AUC_mean"], 0.9744, 5e-4, "Logic URER AUC")

    mp = pd.read_csv(OUT_ROOT / "mechanism_patch" / "summary_table.csv").iloc[0]
    approx(mp["cf_pos_cue_dir_acc_mean"], 1.0, 1e-6, "Causal Pos CDA")
    approx(mp["cf_neg_cue_dir_acc_mean"], 1.0, 1e-6, "Causal Neg CDA")
    print("[OK] Generated experiment outputs verified.")


def verify_hygiene() -> None:
    pat = re.compile(r"(/root/|/home/|C:\\Users\\|D:\\|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.)", re.I)
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in {".git", "__pycache__", "outputs", ".venv"}]
        for f in fns:
            if not f.endswith((".py", ".yaml", ".yml", ".json", ".md", ".txt", ".csv")):
                continue
            if f == "verify_reproduction.py":
                continue
            fp = Path(dp) / f
            txt = fp.read_text(encoding="utf-8", errors="ignore")
            check(pat.search(txt) is None, f"Private path/IP pattern found in {fp.relative_to(ROOT)}")
    print("[OK] Repository hygiene verified (0 private paths/IPs).")


def main() -> None:
    p = argparse.ArgumentParser(description="Verify dataset, config, hygiene, and optional experiment outputs.")
    p.add_argument("--with-outputs", action="store_true", help="Also verify numerical tables under outputs/ after running experiments.")
    args = p.parse_args()

    verify_datasets()
    verify_configs_and_assets()
    verify_hygiene()
    if args.with_outputs:
        verify_runtime_outputs()
    elif (OUT_ROOT / "main_suite" / "summary_table.csv").exists():
        verify_runtime_outputs()
    print("\nALL REPOSITORY CHECKS PASSED CLEANLY.")


if __name__ == "__main__":
    main()
