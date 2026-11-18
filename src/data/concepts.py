"""Chinese semantic-rhetorical concept priors and text normalization."""

from typing import List, Optional, Tuple

import jieba
import numpy as np

try:
    from opencc import OpenCC
    _CC = OpenCC("t2s")
except Exception:
    _CC = None

CONCEPT_NAMES = [
    "Aspect_Focus",
    "Positive_Cue",
    "Negative_Cue",
    "Intensity_Arousal",
    "Negation_Operator",
    "Adversative_Shift",
    "Modality_Hedging",
    "Domain_Attribute",
]

POS_WORDS = {
    "好", "很好", "不错", "棒", "很棒", "赞", "优秀", "出色", "完美", "满意", "喜欢", "爱",
    "推荐", "值得", "划算", "实惠", "便宜", "香", "鲜", "美味", "好吃", "好喝", "嫩", "脆",
    "爽", "顺", "流畅", "稳定", "快速", "快", "清晰", "细腻", "舒服", "舒适", "贴心", "热情",
    "友好", "友善", "亲切", "干净", "整洁", "惊喜", "惊艳", "超值", "耐用", "轻薄", "安静",
    "强劲", "强大", "提升", "增长", "成长", "获利", "盈利", "稳健", "领先", "突破", "创新",
    "优化", "高效", "环保", "绿色", "节能", "便利", "方便", "亮眼", "新高", "丰厚", "美观",
    "精致", "浓郁", "入味", "地道", "划算", "给力", "靠谱", "安心", "放心", "加分",
}

NEG_WORDS = {
    "差", "很差", "糟糕", "烂", "难吃", "难喝", "难用", "失望", "后悔", "垃圾", "坑", "贵",
    "太贵", "昂贵", "慢", "卡", "卡顿", "死机", "延迟", "吵", "噪音", "发热", "发烫", "掉电",
    "漏水", "故障", "损坏", "瑕疵", "敷衍", "冷漠", "恶劣", "脏", "乱", "挤", "少", "小",
    "偏小", "太小", "太少", "咸", "苦", "酸", "辣", "腥", "油腻", "干柴", "生硬", "寡淡",
    "普通", "乏味", "无聊", "麻烦", "复杂", "笨重", "粗糙", "下降", "下滑", "衰退", "亏损",
    "亏", "跌", "跌幅", "风险", "恶化", "拖延", "滞后", "超标", "污染", "浪费", "短缺", "不足",
    "缺陷", "缩水", "鸡肋", "智商税", "割韭菜", "抱怨", "投诉", "退货", "扣分", "堪忧",
}

INTENSITY_WORDS = {
    "很", "非常", "极其", "极为", "特别", "超级", "十分", "相当", "太", "最", "完全", "彻底",
    "简直", "绝对", "无比", "超", "巨", "极了", "实在", "真的", "确实", "格外", "尤其", "明显",
    "显著", "大幅", "剧烈", "严重", "深度", "极度", "异常", "极其", "疯狂", "爆", "!", "！", "?", "？",
}

NEGATION_WORDS = {
    "不", "没", "没有", "非", "并非", "绝非", "未", "无", "别", "莫", "勿", "毫不", "难",
    "不太", "不算", "不是", "不够", "不会", "不能", "无法", "毫无", "并未", "决不", "绝不",
}

ADVERSATIVE_WORDS = {
    "虽然", "尽管", "固然", "虽说", "虽", "但", "但是", "然而", "可是", "却", "不过",
    "可惜", "偏偏", "反而", "只是", "倒是", "反观", "相反", "另一方面", "只不过", "奈何",
}

HEDGING_WORDS = {
    "可能", "也许", "大概", "大约", "好像", "似乎", "感觉", "觉得", "应该", "一般", "勉强",
    "略", "稍微", "有点", "有些", "还算", "尚可", "还行", "中规中矩", "或许", "疑似", "大体",
    "基本", "相对", "还凑合", "凑合", "说得过去",
}

DOMAIN_WORDS = {
    "价格", "价钱", "性价比", "cp值", "服务", "态度", "环境", "位置", "交通", "分量", "份量",
    "口味", "味道", "口感", "食材", "菜单", "上菜", "屏幕", "显示", "电池", "续航", "散热",
    "性能", "配置", "内存", "硬盘", "显卡", "处理器", "键盘", "重量", "外观", "做工", "系统",
    "营收", "收入", "净利", "净利润", "毛利", "毛利率", "股利", "资本", "市场", "产能", "需求",
    "政策", "政府", "企业", "环保", "碳排放", "绿电", "太阳能", "能源", "材料", "成本",
}


def to_simplified(text: str) -> str:
    if not text:
        return ""
    if _CC is not None:
        return _CC.convert(text)
    return text


def find_aspect_char_span(text: str, aspect: str, occ: int = 0) -> Tuple[int, int]:
    """Locate character span [start, end) of aspect in text with fuzzy fallback."""
    if not aspect or aspect.lower() == "null":
        return 0, min(len(text), 1)
    matches = []
    start = 0
    while True:
        idx = text.find(aspect, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(aspect)))
        start = idx + 1
    if matches:
        chosen = matches[min(occ, len(matches) - 1)]
        return chosen

    # Case-insensitive match
    tl, al = text.lower(), aspect.lower()
    idx = tl.find(al)
    if idx != -1:
        return idx, idx + len(al)

    # Character overlap window fallback
    aset = set(al)
    best_score, best_span = -1, (0, min(len(text), max(1, len(aspect))))
    win = max(1, len(aspect))
    for i in range(max(1, len(text) - win + 1)):
        sub = tl[i : i + win]
        sc = len(aset & set(sub))
        if sc > best_score:
            best_score = sc
            best_span = (i, min(len(text), i + win))
    return best_span


def compute_char_concept_priors(text_simp: str, aspect_simp: str, occ: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Compute character-level 8D weak concept priors in [0, 1] for Chinese text."""
    n = len(text_simp)
    priors = np.zeros((n, len(CONCEPT_NAMES)), dtype=np.float32)
    if n == 0:
        return priors, (0, 0)

    if aspect_simp and aspect_simp not in ("整体评价", "null", "NULL"):
        asp_s, asp_e = find_aspect_char_span(text_simp, aspect_simp, occ=occ)
        priors[asp_s:asp_e, 0] = 1.0
        # Soft context window around aspect
        ctx_s, ctx_e = max(0, asp_s - 3), min(n, asp_e + 3)
        for i in range(ctx_s, ctx_e):
            if priors[i, 0] < 0.5:
                priors[i, 0] = 0.4
    else:
        asp_s, asp_e = 0, 0

    # Word segmentation alignment via jieba
    cur = 0
    for w in jieba.lcut(text_simp):
        wlen = len(w)
        if wlen == 0:
            continue
        wl = w.lower()
        end = min(n, cur + wlen)
        if wl in POS_WORDS or any(ch in POS_WORDS for ch in wl if len(wl) <= 2):
            priors[cur:end, 1] = 1.0
        if wl in NEG_WORDS or any(ch in NEG_WORDS for ch in wl if len(wl) <= 2):
            priors[cur:end, 2] = 1.0
        if wl in INTENSITY_WORDS or any(ch in {"!", "！", "?", "？"} for ch in wl):
            priors[cur:end, 3] = 1.0
        if wl in NEGATION_WORDS or (len(wl) <= 2 and any(ch in {"不", "没", "非", "未", "无"} for ch in wl)):
            priors[cur:end, 4] = 1.0
        if wl in ADVERSATIVE_WORDS:
            priors[cur:end, 5] = 1.0
        if wl in HEDGING_WORDS:
            priors[cur:end, 6] = 1.0
        if wl in DOMAIN_WORDS or any(ch.isdigit() for ch in wl) or "%" in wl:
            priors[cur:end, 7] = 1.0
        cur = end

    return priors, (asp_s, asp_e)


def align_priors_to_tokens(
    char_priors: np.ndarray,
    asp_span: Tuple[int, int],
    offsets: List[Tuple[int, int]],
    seq_ids: List[Optional[int]],
    max_rel_dist: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map character-level concept priors and relative aspect distance onto tokenizer tokens."""
    T = len(offsets)
    tok_priors = np.zeros((T, len(CONCEPT_NAMES)), dtype=np.float32)
    sent_mask = np.zeros(T, dtype=np.float32)
    rel_dist = np.full(T, max_rel_dist * 2 + 1, dtype=np.int64)  # special bucket for non-sentence tokens

    # Map aspect center to token space
    asp_c = 0.5 * (asp_span[0] + asp_span[1])
    asp_tok_indices = []
    for t, (sid, (cs, ce)) in enumerate(zip(seq_ids, offsets)):
        if sid == 0 and ce > cs:
            sent_mask[t] = 1.0
            slice_p = char_priors[cs:ce]
            if len(slice_p) > 0:
                tok_priors[t] = slice_p.max(axis=0)
            if not (ce <= asp_span[0] or cs >= asp_span[1]):
                asp_tok_indices.append(t)

    if asp_tok_indices:
        center_tok = int(round(float(np.mean(asp_tok_indices))))
    else:
        # fallback using char center
        best_t, best_d = 1, 1e9
        for t, (sid, (cs, ce)) in enumerate(zip(seq_ids, offsets)):
            if sid == 0 and ce > cs:
                d = abs(0.5 * (cs + ce) - asp_c)
                if d < best_d:
                    best_d = d
                    best_t = t
        center_tok = best_t

    for t in range(T):
        if sent_mask[t] > 0.5:
            diff = max(-max_rel_dist, min(max_rel_dist, t - center_tok))
            rel_dist[t] = diff + max_rel_dist  # in [0, 2*max_rel_dist]

    return tok_priors, sent_mask, rel_dist
