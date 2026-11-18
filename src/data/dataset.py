"""PyTorch datasets with LDS extreme-tail weights and controlled noise injection."""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

from .concepts import (
    align_priors_to_tokens,
    compute_char_concept_priors,
    to_simplified,
)


def compute_lds_weights(
    targets: np.ndarray,
    n_bins: int = 32,
    ks: int = 5,
    sigma: float = 2.0,
    min_w: float = 0.5,
    max_w: float = 4.0,
) -> np.ndarray:
    """Compute smoothed inverse-density sample weights (LDS) on training targets only."""
    y = np.asarray(targets, dtype=np.float64)
    y_min, y_max = float(y.min()), float(y.max())
    if y_max - y_min < 1e-6:
        return np.ones(len(y), dtype=np.float32)

    bins = np.linspace(y_min, y_max + 1e-6, n_bins + 1)
    bin_idx = np.clip(np.digitize(y, bins) - 1, 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)

    # Smooth histogram with 1D Gaussian kernel
    eff_density = gaussian_filter1d(counts, sigma=sigma, mode="nearest")
    eff_density = np.maximum(eff_density, 1.0)

    inv_w = 1.0 / np.sqrt(eff_density[bin_idx])
    inv_w = inv_w / np.mean(inv_w)
    inv_w = np.clip(inv_w, min_w, max_w)
    return inv_w.astype(np.float32)


def inject_text_noise(text: str, noise_rate: float, rng: np.random.Generator, vocab_chars: Sequence[str]) -> str:
    """Inject random character deletion, swap, or substitution noise at a specified rate."""
    if noise_rate <= 0.0 or len(text) < 4:
        return text
    chars = list(text)
    n_ops = max(1, int(round(len(chars) * noise_rate)))
    for _ in range(n_ops):
        if len(chars) < 2:
            break
        op = rng.integers(0, 3)
        pos = int(rng.integers(0, len(chars)))
        if op == 0 and len(chars) > 3:
            chars.pop(pos)
        elif op == 1 and pos < len(chars) - 1:
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        else:
            chars[pos] = vocab_chars[int(rng.integers(0, len(vocab_chars)))]
    return "".join(chars)


class SentimentRegressionDataset(Dataset):
    """Unified dataset for DimABSA2026 (2D VA), BRIGHTER (6D intensity), and Logic probe."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        target_cols: List[str],
        max_length: int = 192,
        has_aspect: bool = True,
        lds_weights: Optional[np.ndarray] = None,
        noise_rate: float = 0.0,
        noise_seed: int = 42,
        max_rel_dist: int = 64,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.target_cols = target_cols
        self.max_length = max_length
        self.has_aspect = has_aspect

        if lds_weights is not None:
            self.lds_weights = np.asarray(lds_weights, dtype=np.float32)
        else:
            self.lds_weights = np.ones(len(self.df), dtype=np.float32)

        texts_raw = self.df["text"].astype(str).tolist()
        texts_simp = [to_simplified(t) for t in texts_raw]
        if noise_rate > 0.0:
            rng = np.random.default_rng(noise_seed)
            vocab_chars = list("的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严")
            texts_simp = [inject_text_noise(t, noise_rate, rng, vocab_chars) for t in texts_simp]

        aspects_raw = (
            self.df["aspect"].astype(str).tolist()
            if has_aspect and "aspect" in self.df.columns
            else ["整体评价"] * len(self.df)
        )
        aspects_simp = [to_simplified(a) for a in aspects_raw]
        occs = (
            self.df["aspect_occ"].astype(int).tolist()
            if "aspect_occ" in self.df.columns
            else [0] * len(self.df)
        )

        self.texts_raw = texts_raw
        self.texts_simp = texts_simp
        self.aspects_raw = aspects_raw
        self.aspects_simp = aspects_simp

        # Batch tokenize with offsets for concept prior alignment
        enc = tokenizer(
            texts_simp,
            aspects_simp,
            padding="max_length",
            truncation="only_first",
            max_length=max_length,
            return_offsets_mapping=True,
        )

        self.input_ids = np.asarray(enc["input_ids"], dtype=np.int64)
        self.attention_mask = np.asarray(enc["attention_mask"], dtype=np.int64)
        self.token_type_ids = (
            np.asarray(enc["token_type_ids"], dtype=np.int64)
            if "token_type_ids" in enc
            else np.zeros_like(self.input_ids)
        )

        T = max_length
        n = len(self.df)
        self.concept_priors = np.zeros((n, T, 8), dtype=np.float32)
        self.sent_mask = np.zeros((n, T), dtype=np.float32)
        self.asp_mask = np.zeros((n, T), dtype=np.float32)
        self.rel_dist = np.zeros((n, T), dtype=np.int64)

        for i in range(n):
            cp_char, asp_span = compute_char_concept_priors(texts_simp[i], aspects_simp[i], occ=occs[i])
            offsets = enc["offset_mapping"][i]
            seq_ids = enc.sequence_ids(i)
            tp, sm, rd = align_priors_to_tokens(cp_char, asp_span, offsets, seq_ids, max_rel_dist=max_rel_dist)
            self.concept_priors[i] = tp
            self.sent_mask[i] = sm
            self.rel_dist[i] = rd
            for t_idx, (sid, (cs, ce)) in enumerate(zip(seq_ids, offsets)):
                if sid == 1 and ce > cs:
                    self.asp_mask[i, t_idx] = 1.0

        self.targets = self.df[target_cols].to_numpy(dtype=np.float32)
        self.uids = self.df["uid"].astype(str).tolist() if "uid" in self.df.columns else [str(i) for i in range(n)]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "token_type_ids": torch.tensor(self.token_type_ids[idx], dtype=torch.long),
            "concept_priors": torch.tensor(self.concept_priors[idx], dtype=torch.float32),
            "sent_mask": torch.tensor(self.sent_mask[idx], dtype=torch.float32),
            "asp_mask": torch.tensor(self.asp_mask[idx], dtype=torch.float32),
            "rel_dist": torch.tensor(self.rel_dist[idx], dtype=torch.long),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float32),
            "lds_weight": torch.tensor(self.lds_weights[idx], dtype=torch.float32),
        }
