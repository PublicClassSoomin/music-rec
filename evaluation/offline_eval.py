from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data.database import (
    get_all_interactions,
    get_all_songs,
    get_distinct_interaction_user_ids,
    get_user_liked_song_ids,
)
from evaluation.metrics import evaluate

_METRIC_KEY_PREFIXES = ("Precision@", "Recall@", "NDCG@")


def hybrid_eval_weight_columns(base_options: dict[str, Any]) -> dict[str, Any]:
    """UI에서 넘긴 하이브리드 가중치·임계값을 표에 그대로 보여 줄 열."""
    if not base_options or "weights" not in base_options:
        return {}
    w = base_options.get("weights")
    if not isinstance(w, dict):
        return {}
    try:
        c = float(w.get("content", 0.55))
        co = float(w.get("cooc", 0.45))
    except (TypeError, ValueError):
        c, co = 0.55, 0.45
    th = base_options.get("threshold", 0)
    try:
        th = float(th)
    except (TypeError, ValueError):
        th = 0.0
    return {
        "weight_audio_pct": int(round(c * 100)),
        "weight_cooc_pct": int(round(co * 100)),
        "threshold": th,
    }


@dataclass
class EvalConfig:
    k_list: list[int]
    max_cases: int = 0 # 0이면 전체
    top_k_for_recommend: int = 20

def _build_leave_one_out_cases(max_cases: int=0) -> list[tuple[int, str, list[str]]]:
    """
    (user_id, query_song_id, relevant_song_ids) 케이스 생성.
    각 유저의 현재 좋아요 목록이 2개 이상일 때만 케이스를 만듦.
    """
    cases: list[tuple[int, str, list[str]]] = []

    for uid in get_distinct_interaction_user_ids():
        liked = [str(x) for x in get_user_liked_song_ids(uid) if x]
        liked = list(dict.fromkeys(liked)) # fromkeys()는 리스트를 딕셔너리로 변환하고, 중복 제거
        if len(liked) < 2:
            continue

        for q in liked:
            rel = [x for x in liked if x != q]
            if rel:
                cases.append((uid, q, rel))

    if max_cases > 0:
        cases = cases[:max_cases]
    return cases


def _build_play_based_cases(
    max_cases: int = 0,
    *,
    min_play_seconds: int = 20,
) -> list[tuple[int, str, list[str]]]:
    """
    좋아요가 없어도 평가할 수 있도록, 충분히 길게 재생한(play_seconds 기준) 곡들로 케이스를 만든다.
    같은 유저가 서로 다른 곡을 2곡 이상 '진득하게' 들었으면 leave-one-out 쌍을 만든다.
    """
    inter = get_all_interactions()
    if inter is None or inter.empty or "action" not in inter.columns:
        return []

    plays = inter[inter["action"] == "play"].copy()
    if plays.empty:
        return []

    ps = pd.to_numeric(plays["play_seconds"], errors="coerce").fillna(0)
    plays = plays.loc[ps >= min_play_seconds].copy()
    if plays.empty:
        return []

    by_user: dict[int, list[str]] = {}
    for _, row in plays.iterrows():
        try:
            uid = int(row["user_id"])
        except (TypeError, ValueError):
            continue
        sid = str(row.get("song_id") or "").strip()
        if not sid:
            continue
        by_user.setdefault(uid, []).append(sid)

    cases: list[tuple[int, str, list[str]]] = []
    for uid, sids in by_user.items():
        uniq = list(dict.fromkeys(sids))
        if len(uniq) < 2:
            continue
        for q in uniq:
            rel = [x for x in uniq if x != q]
            cases.append((uid, q, rel))

    if max_cases > 0:
        cases = cases[:max_cases]
    return cases


def _build_eval_cases(max_cases: int = 0) -> list[tuple[int, str, list[str]]]:
    """좋아요 기반 + 재생 기반 케이스를 합치되, (user, query_song) 중복은 앞선 소스(좋아요)를 유지."""
    like_cases = _build_leave_one_out_cases(0)
    play_cases = _build_play_based_cases(0)
    seen: set[tuple[int, str]] = set()
    out: list[tuple[int, str, list[str]]] = []
    for c in like_cases + play_cases:
        key = (c[0], c[1])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    if max_cases > 0:
        out = out[:max_cases]
    return out


def _song_title_artist_map() -> dict[str, tuple[str, str]]:
    """song_id → (title, artist) — 평가 표에 기준곡 메타를 붙일 때 사용."""
    df = get_all_songs()
    if df is None or df.empty:
        return {}
    out: dict[str, tuple[str, str]] = {}
    if "song_id" not in df.columns:
        return out
    for _, row in df.iterrows():
        sid = str(row.get("song_id") or "").strip()
        if not sid:
            continue
        out[sid] = (str(row.get("title") or ""), str(row.get("artist") or ""))
    return out


def _call_recommend(
    algo: Any,
    query_song_id: str,
    top_k: int,
    options: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    알고리즘이 recomment_with_options를 지원하면 옵션 전달,
    아니면 기본 recommend 호출.
    """
    if options and hasattr(algo, "recommend_with_options"):
        return algo.recommend_with_options(query_song_id, top_k, options)
    return algo.recommend(query_song_id, top_k)

def run_offline_eval_for_algorithm(
    algo: Any,
    *,
    base_options: dict[str, Any] | None = None,
    k_list: list[int] | None = None,
    max_cases: int = 0,
) -> dict[str, Any]:
    """
    단일 알고리즘 오프라인 평가.
    평가 케이스: (1) 좋아요 leave-one-out (2) 좋아요가 없으면 재생 로그(play, 최소 초) 기반.
    결과:
    {
        "rows": [...케이스별 지표...], (문제별 채점 결과표)
        "summary": {...평균 지표...}, (과목 평균 점수)
        "n_cases": int (실제로 체점한 문제 수)
    }
    """
    if k_list is None:
        k_list = [5, 10, 20]

    cfg = EvalConfig(
        k_list=k_list,
        max_cases=max_cases,
        top_k_for_recommend=max(k_list),
    )

    cases = _build_eval_cases(cfg.max_cases)
    rows: list[dict[str, Any]] = []
    song_meta = _song_title_artist_map()

    for uid, query_song_id, relevant in cases:
        rec_dict = _call_recommend(
            algo,
            query_song_id,
            top_k=cfg.top_k_for_recommend,
            options=base_options,
        )
        ranked = sorted(rec_dict, key=rec_dict.get, reverse=True)
        metrics = evaluate(ranked, relevant, cfg.k_list)

        q_title, q_artist = song_meta.get(query_song_id, ("", ""))

        rows.append(
            {
                "user_id": uid,
                "query_song_id": query_song_id,
                "query_title": q_title,
                "query_artist": q_artist,
                "n_relevant": len(relevant),
                **metrics,
            }
        )

    if not rows:
        return {"rows": [], "summary": {}, "n_cases": 0}

    # 지표 컬럼 추출 (Precision@K, Recall@K, NDCG@K)
    # startswith 에 튜플을 넘기면 접두사 중 하나라도 맞으면 True
    metric_cols = [k for k in rows[0].keys() if k.startswith(_METRIC_KEY_PREFIXES)]
    # 평균 지표 계산
    summary = {
        m: float(np.mean([float(r[m]) for r in rows]))
        for m in metric_cols
    }

    return {
        "rows": rows,
        "summary": summary,
        "n_cases": len(rows),
    }

def run_offline_eval_variants(
    algo: Any,
    *,
    base_options: dict[str, Any] | None = None,
    k_list: list[int] | None = None,
    max_cases: int = 0,
) -> dict[str, Any]:
    """
    UI의 '표보기' 용:
      - simple / advanced
      - use_llm_search on/off
    4가지 변형을 동일 데이터셋으로 평가.
    """
    if k_list is None:
        k_list = [5, 10, 20]
    
    base = dict(base_options or {})
    wcols = hybrid_eval_weight_columns(base)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    variants = [
        ("simple", False),
        ("simple", True),
        ("advanced", False),
        ("advanced", True),
    ]

    for mode, use_llm in variants:
        opt = dict(base)
        opt["mode"] = mode
        opt["use_llm_search"] = use_llm

        result = run_offline_eval_for_algorithm(
            algo,
            base_options=opt,
            k_list=k_list,
            max_cases=max_cases,
        )

        for r in result["rows"]:
            rows.append(
                {
                    **wcols,
                    "variant_mode": mode,
                    "variant_use_llm_search": use_llm,
                    **r,
                }
            )

        summary_rows.append(
            {
                **wcols,
                "variant_mode": mode,
                "variant_use_llm_search": use_llm,
                "n_cases": result["n_cases"],
                **result["summary"],
            }
        )

    # 예: {"simple|llm=False|Precision@5": 0.21, ...}
    summary_for_chart: dict[str, float] = {}
    for s in summary_rows:
        vmode = s["variant_mode"]
        vllm = s["variant_use_llm_search"]
        for k, v in s.items():
            if isinstance(k, str) and k.startswith(_METRIC_KEY_PREFIXES):
                summary_for_chart[f"{vmode}|llm={vllm}|{k}"] = float(v)

    return {
        "rows": rows,
        "summary_rows": summary_rows,
        "summary": summary_for_chart,
    }





