# Contributing to Concept-BiLSTM-URER

Thank you for your interest in contributing to **Concept-BiLSTM-URER & Concept-BiLSTM-Causal**.

## Guidelines

1. **Fork & Branch**: Create a feature branch from `main` for any bug fix, documentation improvement, or experiment extension.
2. **Code Quality & Determinism**:
   - Preserve fixed seeds (`[42, 102, 202, 302, 402]`) and zero-leakage split protocols when extending benchmark evaluations.
   - Keep Windows (`num_workers=0`) and Linux compatibility across all `DataLoader` calls.
   - Never commit experiment output directories (`outputs/`), model binaries (`*.pt`, `*.ckpt`), cache folders, private server IPs, or raw third-party corpora into `data/raw/`.
3. **Pre-PR Verification**:
   Run the offline repository checks before opening a pull request:
   ```bash
   python -m compileall -q src scripts
   python scripts/verify_reproduction.py
   ```
