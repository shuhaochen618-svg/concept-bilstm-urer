"""Audit and prepare unified schema datasets from 2024-2026 open-source corpora."""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"


def parse_dimabsa_file(path: Path, domain: str, split: str) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj["ID"]
            text = obj["Text"]
            items = obj.get("Aspect_VA") or obj.get("Quadruplet") or obj.get("Triplet") or []
            seen_aspect_counts: Dict[str, int] = defaultdict(int)
            for idx, it in enumerate(items):
                aspect = it["Aspect"]
                va_str = it["VA"]
                v_str, a_str = va_str.split("#")
                v, a = float(v_str), float(a_str)
                occ = seen_aspect_counts[aspect]
                seen_aspect_counts[aspect] += 1
                records.append(
                    {
                        "uid": f"{domain}__{split}__{sid}__{idx}",
                        "text_id": sid,
                        "domain": domain,
                        "split": split,
                        "text": text,
                        "aspect": aspect,
                        "aspect_occ": occ,
                        "opinion": it.get("Opinion", ""),
                        "category": it.get("Category", ""),
                        "valence": v,
                        "arousal": a,
                    }
                )
    return records


def summarize_va(df: pd.DataFrame, name: str) -> dict:
    v = df["valence"].to_numpy()
    a = df["arousal"].to_numpy()
    n_texts = df["text_id"].nunique()
    # Check duplicate aspects inside same text_id
    dup_asp = df.groupby(["text_id", "aspect"]).size()
    n_dup_aspect_pairs = int((dup_asp > 1).sum())
    # Extreme tail fractions in [1, 9]
    ext_v = float(((v <= 3.5) | (v >= 7.0)).mean())
    ext_a = float(((a <= 3.5) | (a >= 7.0)).mean())
    return {
        "subset": name,
        "texts": int(n_texts),
        "instances": int(len(df)),
        "dup_aspect_in_text": n_dup_aspect_pairs,
        "V_mean": round(float(v.mean()), 4),
        "V_std": round(float(v.std()), 4),
        "V_min": round(float(v.min()), 2),
        "V_max": round(float(v.max()), 2),
        "A_mean": round(float(a.mean()), 4),
        "A_std": round(float(a.std()), 4),
        "A_min": round(float(a.min()), 2),
        "A_max": round(float(a.max()), 2),
        "V_extreme_frac(<=3.5_or_>=7.0)": round(ext_v, 4),
        "A_extreme_frac(<=3.5_or_>=7.0)": round(ext_a, 4),
    }


def make_grouped_cal_split(
    df_train: pd.DataFrame, cal_ratio: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    """Split official train into train_sub and cal strictly grouped by text_id."""
    rng = np.random.default_rng(seed)
    text_ids = np.array(sorted(df_train["text_id"].unique()))
    rng.shuffle(text_ids)
    n_cal = max(1, int(round(len(text_ids) * cal_ratio)))
    cal_ids = set(text_ids[:n_cal])
    out = df_train.copy()
    out["split_cal"] = np.where(out["text_id"].isin(cal_ids), "cal", "train_sub")
    return out


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    dim_root = RAW_DIR / "DimABSA2026" / "task-dataset"

    dim_files = {
        "zho_rest": {
            "train": dim_root / "track_a" / "subtask_1" / "zho" / "zho_restaurant_train_alltasks.jsonl",
            "dev": dim_root / "track_a" / "subtask_1" / "zho" / "zho_restaurant_dev_task1.jsonl",
            "test": dim_root / "track_a" / "subtask_1" / "zho" / "zho_restaurant_test_task1.jsonl",
        },
        "zho_lap": {
            "train": dim_root / "track_a" / "subtask_1" / "zho" / "zho_laptop_train_alltasks.jsonl",
            "dev": dim_root / "track_a" / "subtask_1" / "zho" / "zho_laptop_dev_task1.jsonl",
            "test": dim_root / "track_a" / "subtask_1" / "zho" / "zho_laptop_test_task1.jsonl",
        },
        "zho_fin": {
            "train": dim_root / "track_a" / "subtask_1" / "zho" / "zho_finance_train_task1.jsonl",
            "dev": dim_root / "track_a" / "subtask_1" / "zho" / "zho_finance_dev_task1.jsonl",
            "test": dim_root / "track_a" / "subtask_1" / "zho" / "zho_finance_test_task1.jsonl",
        },
        "zho_env": {
            "train": dim_root / "track_b" / "subtask_1" / "zho" / "zho_environmental_protection_train_task1.jsonl",
            "dev": dim_root / "track_b" / "subtask_1" / "zho" / "zho_environmental_protection_dev_task1.jsonl",
            "test": dim_root / "track_b" / "subtask_1" / "zho" / "zho_environmental_protection_test_task1.jsonl",
        },
    }

    all_dim_rows = []
    summaries = []
    for domain, splits in dim_files.items():
        for sp, path in splits.items():
            rows = parse_dimabsa_file(path, domain, sp)
            df = pd.DataFrame(rows)
            if sp == "train":
                df = make_grouped_cal_split(df, cal_ratio=0.15, seed=42)
            else:
                df["split_cal"] = sp
            all_dim_rows.append(df)
            summaries.append(summarize_va(df, f"{domain}/{sp}"))

    dim_df = pd.concat(all_dim_rows, ignore_index=True)
    dim_out = PROC_DIR / "dimabsa2026_zho_all.parquet"
    dim_df.to_parquet(dim_out, index=False)
    dim_df.to_csv(PROC_DIR / "dimabsa2026_zho_all.csv", index=False)

    print("=== DimABSA2026 Chinese Subsets Audit ===")
    sum_df = pd.DataFrame(summaries)
    print(sum_df.to_string(index=False))

    # Check text_id overlap across splits per domain
    for domain in dim_files:
        sub = dim_df[dim_df["domain"] == domain]
        tr_ids = set(sub[sub["split"] == "train"]["text_id"])
        dv_ids = set(sub[sub["split"] == "dev"]["text_id"])
        ts_ids = set(sub[sub["split"] == "test"]["text_id"])
        tr_sub = set(sub[sub["split_cal"] == "train_sub"]["text_id"])
        cal_sub = set(sub[sub["split_cal"] == "cal"]["text_id"])
        print(
            f"[Overlap Check] {domain}: train∩dev={len(tr_ids & dv_ids)}, "
            f"train∩test={len(tr_ids & ts_ids)}, dev∩test={len(dv_ids & ts_ids)}, "
            f"train_sub∩cal={len(tr_sub & cal_sub)}"
        )

    # Audit and save chinese-logic-sentiment-dataset
    logic_dir = RAW_DIR / "chinese-logic-sentiment-dataset"
    if (logic_dir / "train.csv").exists():
        print("\n=== chinese-logic-sentiment-dataset Audit ===")
        logic_frames = []
        for sp_in, sp_out in [("train", "train"), ("val", "dev"), ("test", "test")]:
            ldf = pd.read_csv(logic_dir / f"{sp_in}.csv")
            ldf["split"] = sp_out
            logic_frames.append(ldf)
            print(f"{sp_in}: rows={len(ldf)}, cols={list(ldf.columns)}, pos_rate={ldf['label'].mean():.4f}")
            if "type" in ldf.columns:
                print("  types:", dict(Counter(ldf["type"])))
        logic_all = pd.concat(logic_frames, ignore_index=True)
        # Check exact text overlap across splits
        l_tr = set(logic_all[logic_all["split"] == "train"]["text"])
        l_dv = set(logic_all[logic_all["split"] == "dev"]["text"])
        l_ts = set(logic_all[logic_all["split"] == "test"]["text"])
        print(f"[Overlap Check] logic_sentiment: train∩dev={len(l_tr & l_dv)}, train∩test={len(l_tr & l_ts)}, dev∩test={len(l_dv & l_ts)}")
        logic_all.to_parquet(PROC_DIR / "logic_sentiment_zho.parquet", index=False)

    # Audit and save BRIGHTER-emotion-intensities (chn)
    brighter_dir = RAW_DIR / "BRIGHTER-emotion-intensities" / "chn"
    emo_cols = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    if (brighter_dir / "train-00000-of-00001.parquet").exists():
        print("\n=== BRIGHTER-emotion-intensities (chn) Audit ===")
        b_frames = []
        for sp in ["train", "dev", "test"]:
            bdf = pd.read_parquet(brighter_dir / f"{sp}-00000-of-00001.parquet")
            bdf["split"] = sp
            if sp == "train":
                rng = np.random.default_rng(42)
                ids = np.array(sorted(bdf["id"].unique()))
                rng.shuffle(ids)
                n_cal = max(1, int(round(len(ids) * 0.15)))
                cal_ids = set(ids[:n_cal])
                bdf["split_cal"] = np.where(bdf["id"].isin(cal_ids), "cal", "train_sub")
            else:
                bdf["split_cal"] = sp
            b_frames.append(bdf)
            means = {c: round(float(bdf[c].mean()), 3) for c in emo_cols}
            nonzero = {c: round(float((bdf[c] > 0).mean()), 3) for c in emo_cols}
            print(f"{sp}: rows={len(bdf)}, means={means}, nonzero_frac={nonzero}")
        b_all = pd.concat(b_frames, ignore_index=True)
        b_tr = set(b_all[b_all["split"] == "train"]["text"])
        b_dv = set(b_all[b_all["split"] == "dev"]["text"])
        b_ts = set(b_all[b_all["split"] == "test"]["text"])
        print(f"[Overlap Check] brighter_chn: train∩dev={len(b_tr & b_dv)}, train∩test={len(b_tr & b_ts)}, dev∩test={len(b_dv & b_ts)}")
        b_all.to_parquet(PROC_DIR / "brighter_chn_intensities.parquet", index=False)


if __name__ == "__main__":
    main()
