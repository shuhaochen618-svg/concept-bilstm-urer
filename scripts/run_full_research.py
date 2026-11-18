"""Resumable phase orchestrator for full paper reproduction."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "outputs"
CFG_ROOT = PROJECT_ROOT / "configs"
PY_BIN = sys.executable


def run_cmd(cmd: List[str], env: dict) -> None:
    print("\n>>> EXEC:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}")


def already_done(exp_name: str, task: str, variant: str, seed: int, backbone_tag: str = "") -> bool:
    p = OUT_ROOT / exp_name / f"{task}__{variant}__seed{seed}" / "results.json"
    return p.exists()


def load_mobo_hparams(mobo_dir: Path) -> Dict[str, object]:
    cfg_path = mobo_dir / "mobo_best_config.json"
    if not cfg_path.exists():
        fallback = CFG_ROOT / "trial12_knee.json"
        if fallback.exists():
            print(f"[INFO] {cfg_path} not found; loading reference Trial #12 config from {fallback}", flush=True)
            return json.loads(fallback.read_text(encoding="utf-8"))["knee_params"]
        raise FileNotFoundError(f"Missing MOBO config at {cfg_path} and fallback {fallback}. Run --phase mobo first.")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return data["knee_params"]


def train_variant(
    env: dict,
    exp_name: str,
    task: str,
    variant: str,
    seeds: List[int],
    backbone: str = "hfl/chinese-macbert-base",
    epochs: int = 12,
    patience: int = 4,
    warmup_epochs: int = 2,
    batch_size: int = 32,
    train_frac: float = 1.0,
    eval_noise: bool = False,
    eval_explain: bool = False,
    save_checkpoint: bool = False,
    mc_dropout: bool = False,
    hp: Dict[str, object] = None,
    skip_existing: bool = True,
) -> None:
    hp = hp or {}
    pending = [s for s in seeds if not (skip_existing and already_done(exp_name, task, variant, s))]
    if not pending:
        print(f"[SKIP] {exp_name} | {task} | {variant} | {backbone} (all {len(seeds)} seeds done)", flush=True)
        return

    if save_checkpoint and len(pending) > 1:
        # Save checkpoint only on first seed (seed=42) to save disk while supporting probe & case analysis
        first_seed = [pending[0]]
        rest_seeds = pending[1:]
        train_variant(
            env, exp_name=exp_name, task=task, variant=variant, seeds=first_seed, backbone=backbone,
            epochs=epochs, patience=patience, warmup_epochs=warmup_epochs, batch_size=batch_size,
            train_frac=train_frac, eval_noise=eval_noise, eval_explain=eval_explain,
            save_checkpoint=True, mc_dropout=mc_dropout, hp=hp, skip_existing=skip_existing,
        )
        train_variant(
            env, exp_name=exp_name, task=task, variant=variant, seeds=rest_seeds, backbone=backbone,
            epochs=epochs, patience=patience, warmup_epochs=warmup_epochs, batch_size=batch_size,
            train_frac=train_frac, eval_noise=eval_noise, eval_explain=eval_explain,
            save_checkpoint=False, mc_dropout=mc_dropout, hp=hp, skip_existing=skip_existing,
        )
        return

    cmd = [
        PY_BIN, "-u", "-m", "src.train_eval",
        "--exp-name", exp_name,
        "--task", task,
        "--variant", variant,
        "--backbone", backbone,
        "--seeds", *[str(s) for s in pending],
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--warmup-epochs", str(warmup_epochs),
        "--batch-size", str(batch_size),
        "--train-frac", str(train_frac),
        "--lr", str(hp.get("lr", 2e-4)),
        "--lora-rank", str(int(hp.get("lora_rank", 8))),
        "--bilstm-hidden", str(int(hp.get("bilstm_hidden", 64))),
        "--dropout", str(hp.get("dropout", 0.2)),
        "--lambda-ccc", str(hp.get("lambda_ccc", 0.5)),
        "--lambda-reg", str(hp.get("lambda_reg", 0.05)),
        "--lambda-concept-align", str(hp.get("lambda_concept_align", 0.35)),
        "--lambda-concept-sparse", str(hp.get("lambda_concept_sparse", 0.06)),
        "--eta-epi", str(hp.get("eta_epi", 0.5)),
        "--eta-ale", str(hp.get("eta_ale", 0.5)),
    ]
    if eval_noise:
        cmd.append("--eval-noise")
    if eval_explain:
        cmd.append("--eval-explain")
    if save_checkpoint:
        cmd.append("--save-checkpoint")
    if mc_dropout:
        cmd.append("--mc-dropout")
    run_cmd(cmd, env)


def run_phase(phase: str, args, env: dict, skip: bool, mobo_dir: Path) -> None:
    if phase == "smoke":
        train_variant(
            env, exp_name="smoke_check", task="dimabsa_zho_fin", variant="concept_bilstm_urer",
            seeds=[42], epochs=2, patience=2, warmup_epochs=1,
            eval_noise=True, eval_explain=True, save_checkpoint=True, skip_existing=False,
        )
        ckpt = OUT_ROOT / "smoke_check" / "dimabsa_zho_fin__concept_bilstm_urer__seed42"
        run_cmd([
            PY_BIN, "-u", "-m", "src.eval_suite", "--mode", "probe",
            "--ckpt-dir", str(ckpt), "--out-dir", str(OUT_ROOT / "smoke_check" / "probe"),
        ], env)
        print("[SMOKE] All smoke checks succeeded cleanly!")

    elif phase == "mobo":
        if not (mobo_dir / "mobo_best_config.json").exists() or not skip:
            run_cmd([
                PY_BIN, "-u", "-m", "src.mobo_search",
                "--exp-name", "mobo_zho_rest",
                "--task", "dimabsa_zho_rest",
                "--n-trials", str(args.mobo_trials),
                "--proxy-epochs", str(args.mobo_epochs),
            ], env)
        else:
            print("[SKIP] MOBO already completed at", mobo_dir)

    elif phase == "main":
        hp = load_mobo_hparams(mobo_dir)
        # Main comparison across 4 DimABSA2026 domains + BRIGHTER-chn + Logic-Sentiment
        all_tasks = [
            "dimabsa_zho_rest",
            "dimabsa_zho_lap",
            "dimabsa_zho_fin",
            "dimabsa_zho_env",
            "brighter_chn",
            "logic_sentiment",
        ]
        baselines = ["classic_bilstm", "classic_bigru", "classic_cnn", "plm_only_mse", "plm_only_ccc", "plm_only_rdrop"]
        for task in all_tasks:
            is_dim = task.startswith("dimabsa_")
            # Save proposed model checkpoints + run explainability & noise on primary datasets
            train_variant(
                env, exp_name="main_suite", task=task, variant="concept_bilstm_urer",
                seeds=args.seeds, epochs=args.epochs, hp=hp,
                eval_noise=is_dim,
                eval_explain=(task in ("dimabsa_zho_rest", "dimabsa_zho_lap", "dimabsa_zho_fin", "logic_sentiment")),
                save_checkpoint=(task == "dimabsa_zho_rest"),
                skip_existing=skip,
            )
            for base in baselines:
                base_epochs = 25 if base.startswith("classic_") else args.epochs
                base_lr = 1e-3 if base.startswith("classic_") else hp.get("lr", 2e-4)
                base_hp = dict(hp)
                base_hp["lr"] = base_lr
                train_variant(
                    env, exp_name="main_suite", task=task, variant=base,
                    seeds=args.seeds, epochs=base_epochs, hp=base_hp,
                    eval_noise=(task == "dimabsa_zho_rest"),
                    save_checkpoint=(task == "dimabsa_zho_rest" and base == "plm_only_mse"),
                    skip_existing=skip,
                )

    elif phase == "ablations":
        hp = load_mobo_hparams(mobo_dir)
        ablation_vars = [
            ("ib_residual_bilstm", False),
            ("plm_bilstm_no_concept", False),
            ("concept_no_bilstm", False),
            ("concept_bilstm_no_ccc", False),
            ("concept_bilstm_no_dyn", False),
            ("concept_bilstm_det", True),  # MC-Dropout calibration comparison
        ]
        for task in ("dimabsa_zho_rest", "dimabsa_zho_lap"):
            for var, use_mc in ablation_vars:
                train_variant(
                    env, exp_name="ablation_suite", task=task, variant=var,
                    seeds=args.seeds, epochs=args.epochs, hp=hp,
                    eval_noise=(task == "dimabsa_zho_rest"),
                    eval_explain=(var in ("ib_residual_bilstm", "concept_no_bilstm")),
                    save_checkpoint=(task == "dimabsa_zho_rest" and var in ("plm_bilstm_no_concept", "concept_no_bilstm")),
                    mc_dropout=use_mc,
                    skip_existing=skip,
                )

        # Cross-domain OOD transfer: train rest+lap -> evaluate zho_fin
        for var in ("classic_bilstm", "plm_only_mse", "plm_only_ccc", "concept_bilstm_urer"):
            h_cur = dict(hp)
            if var == "classic_bilstm":
                h_cur["lr"] = 1e-3
            train_variant(
                env, exp_name="cross_ood_suite", task="dimabsa_cross_fin", variant=var,
                seeds=args.seeds, epochs=(25 if var == "classic_bilstm" else args.epochs),
                hp=h_cur, skip_existing=skip,
            )

        # Data efficiency curve: 25% and 50% labeled data on dimabsa_zho_rest
        for frac, tag in [(0.25, "frac25"), (0.50, "frac50")]:
            for var in ("classic_bilstm", "plm_only_mse", "concept_bilstm_urer"):
                h_cur = dict(hp)
                if var == "classic_bilstm":
                    h_cur["lr"] = 1e-3
                train_variant(
                    env, exp_name=f"data_eff_{tag}", task="dimabsa_zho_rest", variant=var,
                    seeds=args.seeds, epochs=(25 if var == "classic_bilstm" else args.epochs),
                    train_frac=frac, hp=h_cur, skip_existing=skip,
                )

    elif phase == "secondary_backbones":
        hp = load_mobo_hparams(mobo_dir)
        for bb, tag in [
            ("hfl/chinese-roberta-wwm-ext", "roberta"),
            ("IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese", "deberta"),
        ]:
            for task in ("dimabsa_zho_rest", "dimabsa_zho_fin"):
                for var in ("plm_only_mse", "concept_bilstm_urer"):
                    train_variant(
                        env, exp_name=f"backbone_{tag}", task=task, variant=var,
                        backbone=bb, seeds=args.seeds[:3], epochs=args.epochs, hp=hp, skip_existing=skip,
                    )

    elif phase == "probe":
        ckpt = OUT_ROOT / "main_suite" / "dimabsa_zho_rest__concept_bilstm_urer__seed42"
        run_cmd([
            PY_BIN, "-u", "-m", "src.eval_suite", "--mode", "probe",
            "--ckpt-dir", str(ckpt), "--out-dir", str(OUT_ROOT / "analysis"),
        ], env)
        for sub_exp, var in [
            ("main_suite", "plm_only_mse"),
            ("ablation_suite", "plm_bilstm_no_concept"),
            ("ablation_suite", "concept_no_bilstm"),
        ]:
            cdir = OUT_ROOT / sub_exp / f"dimabsa_zho_rest__{var}__seed42"
            if (cdir / "best_model.pt").exists():
                run_cmd([
                    PY_BIN, "-u", "-m", "src.eval_suite", "--mode", "probe",
                    "--ckpt-dir", str(cdir), "--out-dir", str(OUT_ROOT / "analysis" / f"probe_{var}"),
                ], env)

    elif phase == "llm":
        run_cmd([
            PY_BIN, "-u", "-m", "src.eval_suite", "--mode", "llm_baseline",
            "--llm-model", "Qwen/Qwen2.5-3B-Instruct",
            "--out-dir", str(OUT_ROOT / "llm_baseline"),
            "--tasks", "dimabsa_zho_rest", "dimabsa_zho_lap", "dimabsa_zho_fin", "dimabsa_zho_env",
        ], env)

    elif phase == "mechanism_patch":
        hp = load_mobo_hparams(mobo_dir)
        train_variant(
            env, exp_name="mechanism_patch", task="dimabsa_zho_rest", variant="concept_bilstm_causal",
            seeds=args.seeds, epochs=args.epochs, eval_noise=True, eval_explain=True,
            save_checkpoint=True, hp=hp, skip_existing=skip,
        )
        cdir = OUT_ROOT / "mechanism_patch" / "dimabsa_zho_rest__concept_bilstm_causal__seed42"
        if (cdir / "best_model.pt").exists():
            run_cmd([
                PY_BIN, "-u", "-m", "src.eval_suite", "--mode", "probe",
                "--ckpt-dir", str(cdir), "--out-dir", str(OUT_ROOT / "analysis" / "probe_concept_bilstm_causal"),
            ], env)

    elif phase == "stats_plot":
        run_cmd([
            PY_BIN, "-u", "-m", "src.eval_suite", "--mode", "stats",
            "--exp-dir", str(OUT_ROOT / "main_suite"),
            "--out-dir", str(OUT_ROOT / "main_suite"),
        ], env)
        for ref_exp in ("ablation_suite", "mechanism_patch"):
            if (OUT_ROOT / ref_exp).exists():
                run_cmd([
                    PY_BIN, "-u", "-m", "src.eval_suite", "--mode", "stats",
                    "--exp-dir", str(OUT_ROOT / ref_exp),
                    "--ref-exp-dir", str(OUT_ROOT / "main_suite"),
                    "--out-dir", str(OUT_ROOT / ref_exp),
                ], env)
        for extra_exp in ("cross_ood_suite", "data_eff_frac25", "data_eff_frac50", "backbone_roberta", "backbone_deberta"):
            if (OUT_ROOT / extra_exp).exists():
                run_cmd([
                    PY_BIN, "-u", "-m", "src.eval_suite", "--mode", "stats",
                    "--exp-dir", str(OUT_ROOT / extra_exp),
                    "--out-dir", str(OUT_ROOT / extra_exp),
                ], env)
        run_cmd([
            PY_BIN, "-u", "-m", "src.pub_figures",
            "--out-root", str(OUT_ROOT),
            "--fig-dir", str(OUT_ROOT / "pub_figures"),
        ], env)


def main() -> None:
    p = argparse.ArgumentParser(description="Orchestrate full reproduction pipeline for Concept-BiLSTM-URER & Causal.")
    p.add_argument(
        "--phase",
        choices=["smoke", "mobo", "main", "ablations", "secondary_backbones", "probe", "llm", "mechanism_patch", "stats_plot", "all"],
        required=True,
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 102, 202, 302, 402])
    p.add_argument("--mobo-trials", type=int, default=24)
    p.add_argument("--mobo-epochs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--no-skip", action="store_true")
    args = p.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["TOKENIZERS_PARALLELISM"] = "false"
    skip = not args.no_skip
    mobo_dir = OUT_ROOT / "mobo_zho_rest"

    if args.phase == "all":
        for ph in ("mobo", "main", "ablations", "secondary_backbones", "mechanism_patch", "probe", "llm", "stats_plot"):
            run_phase(ph, args, env, skip, mobo_dir)
    else:
        run_phase(args.phase, args, env, skip, mobo_dir)


if __name__ == "__main__":
    main()
