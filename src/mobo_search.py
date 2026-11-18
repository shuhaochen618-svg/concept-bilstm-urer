"""Multi-Objective Bayesian Optimization (MOBO + qEHVI) and fANOVA sensitivity analysis."""

import argparse
import copy
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
from optuna.importance import FanovaImportanceEvaluator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    from optuna_integration import BoTorchSampler
    from optuna_integration.botorch import botorch as optuna_botorch
except ImportError:
    try:
        from optuna.integration import BoTorchSampler
        from optuna.integration.botorch import botorch as optuna_botorch
    except ImportError:
        BoTorchSampler = None
        optuna_botorch = None


def fast_qehvi_candidates_func(
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    train_con: Optional[torch.Tensor],
    bounds: torch.Tensor,
    pending_x: Optional[torch.Tensor],
) -> torch.Tensor:
    """Lightweight exact qLogEHVI/qEHVI candidate acquisition capped at 4 CPU threads.

    Prevents OpenMP 128-thread futex thrashing in container environments and cuts
    GP + hypervolume acquisition time from ~15-30 min to ~1.5-2.5 sec per trial.
    """
    if train_con is not None and optuna_botorch is not None:
        return optuna_botorch.qehvi_candidates_func(train_x, train_obj, train_con, bounds, pending_x)

    from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
    from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
    from botorch.utils.transforms import normalize, unnormalize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, prev_threads))
    try:
        train_x_n = normalize(train_x, bounds=bounds)
        pending_n = normalize(pending_x, bounds=bounds) if pending_x is not None else None
        model = SingleTaskGP(train_x_n, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

        alpha = 10 ** (-8 + train_obj.size(-1)) if train_obj.size(-1) > 4 else 0.0
        ref_point = train_obj.min(dim=0).values - 1e-5
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_obj, alpha=alpha)
        acqf_cls = (
            qLogExpectedHypervolumeImprovement
            if partitioning.get_hypercell_bounds().shape[-2] > 0
            else qExpectedHypervolumeImprovement
        )
        acqf = acqf_cls(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])),
            X_pending=pending_n,
        )
        std_bounds = torch.zeros_like(bounds)
        std_bounds[1] = 1.0
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=std_bounds,
            q=1,
            num_restarts=6,
            raw_samples=128,
            options={"batch_limit": 5, "maxiter": 80, "nonnegative": True},
            sequential=True,
        )
        return unnormalize(candidates.detach(), bounds=bounds)
    finally:
        torch.set_num_threads(prev_threads)

from src.data.dataset import SentimentRegressionDataset, compute_lds_weights
from src.models.architectures import EvidentialConformalCalibrator
from src.models.losses import compute_total_loss
from src.train_eval import (
    OUT_ROOT,
    build_model_and_flags,
    evaluate_all_metrics,
    load_task_splits,
    measure_latency_ms,
    predict_loader,
    set_seed,
)


def evaluate_proxy_trial(
    hparams: dict,
    backbone: str,
    task: str,
    target_cols: List[str],
    bounds: Tuple[float, float],
    tail_bounds: Dict[str, Tuple[float, float]],
    train_ds: SentimentRegressionDataset,
    cal_ds: SentimentRegressionDataset,
    dev_ds: SentimentRegressionDataset,
    vocab_size: int,
    proxy_epochs: int,
    batch_size: int,
    warmup_epochs: int,
    conformal_alpha: float,
    seed: int,
    device: torch.device,
) -> Tuple[Tuple[float, float, float], Dict[str, float]]:
    """Train a short proxy run without touching test set and return 3 MOBO objectives."""
    set_seed(seed)
    model, flags = build_model_and_flags(
        variant="concept_bilstm_urer",
        backbone=backbone,
        out_dim=len(target_cols),
        bounds=bounds,
        tokenizer_vocab_size=vocab_size,
        lora_rank=int(hparams["lora_rank"]),
        bilstm_hidden=int(hparams["bilstm_hidden"]),
        bilstm_layers=2,
        k_lat=4,
        dropout=float(hparams["dropout"]),
        use_lora=True,
    )
    model.to(device)

    nw = 0 if (os.name == "nt" or device.type != "cuda") else 2
    use_pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=use_pin)
    cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=use_pin)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=use_pin)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(hparams["lr"]), weight_decay=1e-2)
    total_steps = max(1, len(train_loader) * proxy_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=float(hparams["lr"]) * 0.05
    )

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    best_composite = float("inf")
    best_state = None
    for epoch in range(proxy_epochs):
        model.train()
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
                    lambda_ccc=float(hparams["lambda_ccc"]),
                    lambda_reg=float(hparams["lambda_reg"]),
                    lambda_concept_align=float(hparams["lambda_concept_align"]),
                    lambda_concept_sparse=0.06,
                    lambda_concept_ortho=0.05,
                    lambda_ib=0.01,
                    eta_epi=float(hparams["eta_epi"]),
                    eta_ale=float(hparams["eta_ale"]),
                    use_dynamic_weight=True,
                    epoch=epoch,
                    warmup_epochs=warmup_epochs,
                    use_ccc=True,
                )
                loss = loss_dict["total"]

            if not torch.isfinite(loss):
                return (99.0, 99.0, 99.0), {"diverged": 1.0}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

        dev_res = predict_loader(model, dev_loader, device, flags["head_mode"])
        dev_m = evaluate_all_metrics(task, dev_res, None, tail_bounds, alpha=conformal_alpha)
        if task.startswith("dimabsa_"):
            ccc_mean = 0.5 * (dev_m["CCC_V"] + dev_m["CCC_A"])
            comp = float(dev_m["RMSE_VA"] + (1.0 - ccc_mean))
        elif task == "brighter_chn":
            comp = float(dev_m["RMSE_macro"] + (1.0 - dev_m["CCC_macro"]))
        else:
            comp = float(dev_m["RMSE"] + (1.0 - dev_m["CCC"]))
        if comp < best_composite:
            best_composite = comp
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    cal_res = predict_loader(model, cal_loader, device, flags["head_mode"])
    calibrator = EvidentialConformalCalibrator(alpha=conformal_alpha, scale_min=bounds[0], scale_max=bounds[1])
    calibrator.fit(cal_res["y_true"], cal_res["y_pred"], cal_res["tot_std"])

    dev_res = predict_loader(model, dev_loader, device, flags["head_mode"])
    dev_m = evaluate_all_metrics(task, dev_res, calibrator, tail_bounds, alpha=conformal_alpha)
    latency_ms = measure_latency_ms(model, dev_loader, device)
    dev_m["latency_ms"] = latency_ms

    if task.startswith("dimabsa_"):
        ccc_mean = 0.5 * (dev_m["CCC_V"] + dev_m["CCC_A"])
        obj_err_ccc = float(dev_m["RMSE_VA"] + (1.0 - ccc_mean))
        obj_winkler = float(dev_m["Winkler_avg"])
    elif task == "brighter_chn":
        obj_err_ccc = float(dev_m["RMSE_macro"] + (1.0 - dev_m["CCC_macro"]))
        obj_winkler = float(dev_m["Winkler_macro"])
    else:
        obj_err_ccc = float(dev_m["RMSE"] + (1.0 - dev_m["CCC"]))
        obj_winkler = 0.0

    dev_m["composite_err_ccc"] = obj_err_ccc
    dev_m["winkler_obj"] = obj_winkler
    return (obj_err_ccc, obj_winkler, float(latency_ms)), dev_m


def select_pareto_knee(trials: List[optuna.trial.FrozenTrial]) -> optuna.trial.FrozenTrial:
    """Pick Pareto knee trial closest to utopia (0,0,0), guarding against sub-ms GPU latency jitter."""
    vals = np.array([t.values for t in trials], dtype=np.float64)
    vmin = vals.min(axis=0)
    vmax = vals.max(axis=0)
    raw_span = vmax - vmin
    span = np.where(raw_span > 1e-8, raw_span, 1.0)
    norm_vals = (vals - vmin) / span
    lat_w = 0.04 if (vals.shape[1] >= 3 and raw_span[2] < 0.25) else 0.15
    weights = np.array([(1.0 - lat_w) * 0.5, (1.0 - lat_w) * 0.5, lat_w], dtype=np.float64)
    dists = np.sqrt(((norm_vals ** 2) * weights[: vals.shape[1]]).sum(axis=1))
    return trials[int(np.argmin(dists))]


def main() -> None:
    p = argparse.ArgumentParser(description="MOBO hyperparameter optimization with BoTorch qEHVI and fANOVA.")
    p.add_argument("--exp-name", type=str, default="mobo_zho_rest")
    p.add_argument("--task", type=str, default="dimabsa_zho_rest")
    p.add_argument("--backbone", type=str, default="hfl/chinese-macbert-base")
    p.add_argument("--n-trials", type=int, default=24)
    p.add_argument("--n-startup", type=int, default=8)
    p.add_argument("--proxy-epochs", type=int, default=5)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--conformal-alpha", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, cal_df, dev_df, _, target_cols, bounds, has_aspect = load_task_splits(
        args.task, train_frac=1.0, seed=args.seed
    )
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    lds_cols = [compute_lds_weights(train_df[c].to_numpy()) for c in target_cols]
    lds_arr = np.mean(lds_cols, axis=0).astype(np.float32)
    lds_arr = (lds_arr / max(float(np.mean(lds_arr)), 1e-6)).astype(np.float32)

    print(f"[MOBO] Pre-tokenizing {args.task} (train_sub={len(train_df)}, cal={len(cal_df)}, dev={len(dev_df)})...")
    train_ds = SentimentRegressionDataset(
        train_df, tokenizer, target_cols, max_length=args.max_length, has_aspect=has_aspect, lds_weights=lds_arr
    )
    cal_ds = SentimentRegressionDataset(
        cal_df, tokenizer, target_cols, max_length=args.max_length, has_aspect=has_aspect
    )
    dev_ds = SentimentRegressionDataset(
        dev_df, tokenizer, target_cols, max_length=args.max_length, has_aspect=has_aspect
    )

    tail_bounds = {}
    if args.task.startswith("dimabsa_"):
        v_tr = train_df["valence"].to_numpy()
        a_tr = train_df["arousal"].to_numpy()
        tail_bounds = {
            "V": (float(np.quantile(v_tr, 0.10)), float(np.quantile(v_tr, 0.90))),
            "A": (float(np.quantile(a_tr, 0.10)), float(np.quantile(a_tr, 0.90))),
        }

    torch.set_num_threads(min(4, max(1, os.cpu_count() or 4)))
    if BoTorchSampler is not None:
        sampler = BoTorchSampler(
            candidates_func=fast_qehvi_candidates_func,
            n_startup_trials=args.n_startup,
            seed=args.seed,
        )
        sampler_name = "BoTorchSampler_fast_qLogEHVI"
    else:
        sampler = optuna.samplers.NSGAIISampler(seed=args.seed)
        sampler_name = "NSGAIISampler_fallback"

    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        sampler=sampler,
        study_name=args.exp_name,
    )

    # Seed Trial 0 with literature-motivated default prior so MOBO starts from a solid baseline
    study.enqueue_trial(
        {
            "lr": 2e-4,
            "lora_rank": 8,
            "bilstm_hidden": 64,
            "dropout": 0.20,
            "lambda_ccc": 0.50,
            "lambda_reg": 0.05,
            "lambda_concept_align": 0.35,
            "eta_epi": 0.50,
            "eta_ale": 0.50,
        }
    )

    def objective(trial: optuna.Trial) -> Tuple[float, float, float]:
        hp = {
            "lr": trial.suggest_float("lr", 5e-5, 5e-4, log=True),
            "lora_rank": trial.suggest_categorical("lora_rank", [4, 8, 16]),
            "bilstm_hidden": trial.suggest_categorical("bilstm_hidden", [32, 64, 96]),
            "dropout": trial.suggest_float("dropout", 0.10, 0.35),
            "lambda_ccc": trial.suggest_float("lambda_ccc", 0.10, 1.00),
            "lambda_reg": trial.suggest_float("lambda_reg", 0.01, 0.20, log=True),
            "lambda_concept_align": trial.suggest_float("lambda_concept_align", 0.15, 0.60),
            "eta_epi": trial.suggest_float("eta_epi", 0.10, 1.00),
            "eta_ale": trial.suggest_float("eta_ale", 0.10, 1.00),
        }
        t0 = time.perf_counter()
        objs, metrics = evaluate_proxy_trial(
            hparams=hp,
            backbone=args.backbone,
            task=args.task,
            target_cols=target_cols,
            bounds=bounds,
            tail_bounds=tail_bounds,
            train_ds=train_ds,
            cal_ds=cal_ds,
            dev_ds=dev_ds,
            vocab_size=len(tokenizer),
            proxy_epochs=args.proxy_epochs,
            batch_size=args.batch_size,
            warmup_epochs=args.warmup_epochs,
            conformal_alpha=args.conformal_alpha,
            seed=args.seed,
            device=device,
        )
        metrics["trial_wall_sec"] = float(time.perf_counter() - t0)
        for k, v in metrics.items():
            trial.set_user_attr(k, float(v) if isinstance(v, (int, float, np.floating)) else v)
        print(
            f"[Trial {trial.number:02d}] obj=(err_ccc={objs[0]:.4f}, winkler={objs[1]:.4f}, lat={objs[2]:.3f}ms) "
            f"RMSE_VA={metrics.get('RMSE_VA', 0.0):.4f} lr={hp['lr']:.2e} r={hp['lora_rank']} h={hp['bilstm_hidden']}"
        )
        return objs

    print(f"[MOBO] Starting {args.n_trials} trials using {sampler_name} on {device}...")
    study.optimize(objective, n_trials=args.n_trials)

    out_dir = OUT_ROOT / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number, "obj_err_ccc": t.values[0], "obj_winkler": t.values[1], "obj_latency_ms": t.values[2]}
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)

    import pandas as pd
    df_trials = pd.DataFrame(rows)
    df_trials.to_csv(out_dir / "mobo_trials.csv", index=False)

    pareto_trials = [t for t in study.best_trials if t.values[0] < 50.0]
    if not pareto_trials:
        pareto_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    pareto_nums = {t.number for t in pareto_trials}
    df_trials["is_pareto"] = df_trials["trial"].isin(pareto_nums)
    df_trials.to_csv(out_dir / "mobo_trials.csv", index=False)

    knee = select_pareto_knee(pareto_trials)
    best_cfg = {
        "sampler": sampler_name,
        "task": args.task,
        "backbone": args.backbone,
        "n_trials": args.n_trials,
        "knee_trial": int(knee.number),
        "knee_objectives": {
            "composite_err_ccc": float(knee.values[0]),
            "winkler_90": float(knee.values[1]),
            "latency_ms": float(knee.values[2]),
        },
        "knee_params": knee.params,
        "knee_dev_metrics": knee.user_attrs,
        "pareto_trials": [
            {"trial": int(t.number), "values": [float(x) for x in t.values], "params": t.params, "dev_metrics": t.user_attrs}
            for t in pareto_trials
        ],
    }
    with (out_dir / "mobo_best_config.json").open("w", encoding="utf-8") as f:
        json.dump(best_cfg, f, ensure_ascii=False, indent=2)

    # fANOVA sensitivity analysis across each of the 3 objectives
    fanova = {}
    obj_names = ["composite_err_ccc", "winkler_90", "latency_ms"]
    for idx, oname in enumerate(obj_names):
        try:
            imp = optuna.importance.get_param_importances(
                study,
                evaluator=FanovaImportanceEvaluator(seed=args.seed),
                target=lambda t, i=idx: float(t.values[i]),
            )
            fanova[oname] = {k: float(v) for k, v in imp.items()}
        except Exception as e:
            fanova[oname] = {"error": str(e)}

    with (out_dir / "mobo_fanova.json").open("w", encoding="utf-8") as f:
        json.dump(fanova, f, ensure_ascii=False, indent=2)

    print("\n=== MOBO COMPLETE ===")
    print("Knee Trial:", knee.number, "Objectives:", knee.values)
    print("Knee Params:", json.dumps(knee.params, indent=2))
    print("fANOVA:", json.dumps(fanova, indent=2))


if __name__ == "__main__":
    main()
