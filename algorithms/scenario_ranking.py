"""
시나리오 검색 후보 재정렬: RRF(이중 임베딩) + 전체 카탈로그 대비 오디오 백분위 + 메타 키워드 + FAISS 이웃.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

RRF_K = 58

_META_UP = (
    "댄스",
    "dance",
    "디스코",
    "팝",
    "pop",
    "힙합",
    "hip",
    "hop",
    "랩",
    "rap",
    "록",
    "rock",
    "edm",
    "일렉",
    "electro",
    "하우스",
    "house",
    "테크노",
    "스피드",
    "업템포",
    "빠른",
)
_META_DOWN = (
    "발라드",
    "ballad",
    "ost",
    "어쿠스틱",
    "acoustic",
    "피아노",
    "piano",
    "잔잔",
    "슬픈",
    "감성",
    "연주",
)


def rrf_from_ranked_lists(lists: list[list[str]]) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in lists:
        for r, sid in enumerate(ranked):
            scores[sid] += 1.0 / (RRF_K + r + 1)
    return dict(scores)


def normalize_scores(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi - lo < 1e-12:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def _row_text_lower(row: pd.Series) -> str:
    t = f"{row.get('title') or ''} {row.get('genre') or ''}"
    return t.lower()


def metadata_affinity(row: pd.Series, bias: float) -> float:
    t = _row_text_lower(row)
    has_up = any(k in t for k in _META_UP)
    has_down = any(k in t for k in _META_DOWN)
    if bias > 0.25:
        if has_down and not has_up:
            return 0.18
        if has_up:
            return 1.0
        return 0.52
    if bias < -0.25:
        if has_up and not has_down:
            return 0.22
        if has_down:
            return 1.0
        return 0.52
    return 0.55


def audio_affinity_for_row(
    e_ref: np.ndarray,
    b_ref: np.ndarray,
    energy: float | None,
    bpm: float | None,
    bias: float,
) -> float:
    """전체 DB energy/bpm 분포 대비 백분위 (후보 풀 내부 정규화 아님)."""
    if abs(bias) < 0.25:
        return 0.55

    se, sb = 0.5, 0.5
    if energy is not None and pd.notna(energy) and len(e_ref) > 0:
        e = float(energy)
        pct = float((e_ref < e).mean())
        se = pct if bias > 0 else 1.0 - pct
    if bpm is not None and pd.notna(bpm) and len(b_ref) > 0:
        b = float(bpm)
        pct = float((b_ref < b).mean())
        sb = pct if bias > 0 else 1.0 - pct

    if len(e_ref) == 0 and len(b_ref) == 0:
        return 0.55
    if len(e_ref) == 0:
        return float(sb)
    if len(b_ref) == 0:
        return float(se)
    return float(0.62 * se + 0.38 * sb)


def faiss_mood_boost(
    df: pd.DataFrame,
    faiss_index,
    bias: float,
    top_seed: int = 5,
    neighbor_k: int = 12,
) -> dict[str, float]:
    if faiss_index is None or not getattr(faiss_index, "is_built", False) or df is None or df.empty:
        return {}
    if abs(bias) < 0.25:
        return {}

    sub = df[df["song_id"].notna() & df["energy"].notna()].copy()
    if sub.empty:
        return {}
    sub["song_id"] = sub["song_id"].astype(str)
    if bias > 0:
        seeds_df = sub.nlargest(top_seed, "energy", keep="first")
    else:
        seeds_df = sub.nsmallest(top_seed, "energy", keep="first")
    seeds = seeds_df["song_id"].tolist()
    idmap = getattr(faiss_index, "id_to_idx", {})

    raw: dict[str, float] = defaultdict(float)
    for sid in seeds:
        if sid not in idmap:
            continue
        neigh = faiss_index.search_by_id(sid, neighbor_k)
        for nid, sc in neigh.items():
            raw[str(nid)] += float(sc)

    if not raw:
        return {}
    return normalize_scores(raw)


def combine_scenario_scores(
    text_norm: dict[str, float],
    audio_scores: dict[str, float],
    meta_scores: dict[str, float],
    faiss_norm: dict[str, float],
    bias: float,
) -> dict[str, float]:
    sids = set(text_norm) | set(audio_scores) | set(meta_scores) | set(faiss_norm)
    out: dict[str, float] = {}
    if abs(bias) >= 0.25:
        w_t, w_a, w_m, w_f = 0.26, 0.48, 0.18, 0.08
    else:
        w_t, w_a, w_m, w_f = 0.88, 0.04, 0.08, 0.0

    for sid in sids:
        t = text_norm.get(sid, 0.0)
        a = audio_scores.get(sid, 0.55)
        m = meta_scores.get(sid, 0.55)
        f = faiss_norm.get(sid, 0.0)
        out[sid] = w_t * t + w_a * a + w_m * m + w_f * f
    return out
