# Benchmark Data Preparation & Provenance

This directory holds the unified-schema benchmark splits used across all experiments in the paper.

## Included Processed Splits (`data/processed/`)

For zero-friction reproducibility, the exact group-split `.parquet` files (zero `text_id` overlap across `train_sub`, `cal`, `dev`, and `test`) are provided directly in `data/processed/`:

| File | Corpus | Task & Targets | Scale | Instances |
| :--- | :--- | :--- | :--- | ---: |
| `dimabsa2026_zho_all.parquet` | DimABSA 2026 (Chinese) | 2D aspect-level Valence–Arousal (`zho_rest`, `zho_lap`, `zho_fin`, `zho_env`) | `[1, 9]` | 27,558 |
| `brighter_chn_intensities.parquet` | BRIGHTER-CHN | 6D emotion intensities (`anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`) | `[0, 3]` | 5,484 |
| `logic_sentiment_zho.parquet` | LogicSentiment-Zho | 4-type Chinese rhetorical sentiment (`Simple`, `Adv`, `DNeg`, `Irony`) | `{0, 1}` / `{1, 9}` | 3,681 |

## Upstream Dataset Sources (Rebuilding from Raw Corpora)

Raw third-party review texts remain subject to their respective upstream licenses:
1. **DimABSA 2026 (Chinese)** — SemEval-2026 Task 3 (`DimABSA2026`)
   Place raw repo files under `data/raw/DimABSA2026/`.
2. **BRIGHTER-CHN** — BRIGHTER Chinese Emotion Intensity Corpus (`muhammad2025brighter`)
   Place raw `.parquet` splits under `data/raw/BRIGHTER-emotion-intensities/chn/`.
3. **LogicSentiment-Zho** — `YiMeng-SYSU/chinese-logic-sentiment-dataset` (`https://github.com/MengzhongRe/bert-logic-finetune`)
   Place `train.csv`, `val.csv`, `test.csv` under `data/raw/chinese-logic-sentiment-dataset/`.

To regenerate `data/processed/*.parquet` from raw files and verify zero text leakage:

```bash
python scripts/audit_and_prepare_data.py
```
