"""
LangGraph 시나리오 검색 (단일 파이프라인).

- RRF 임베딩 + 오디오·메타·FAISS 보조
- expand: (선택) Gemini 검색문 보강
- llm_select: SCENARIO_LLM_TOP3 + 키 있으면 후보만 보고 top-3 id (없으면 스킵 → 점수 순)
"""

from __future__ import annotations

import json
import re
import threading

import numpy as np
import pandas as pd

from algorithms.base import BaseRecommender
from algorithms.scenario_ranking import (
    audio_affinity_for_row,
    combine_scenario_scores,
    faiss_mood_boost,
    metadata_affinity,
    normalize_scores,
    rrf_from_ranked_lists,
)
from embeddings.song_text_index import SongTextIndex
from embeddings.text_embedder import encode_texts
from graphs.scenario_graph import ScenarioState, compile_scenario_graph, run_scenario_graph
from utils.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    LLM_MODEL,
    SCENARIO_LLM_CANDIDATES,
    SCENARIO_LLM_TOP3,
)
from utils.scenario_mood import mood_energy_bias_and_hint

_INIT_LOCK = threading.Lock()


def _parse_llm_json_id_list(raw: str) -> list[str] | None:
    t = (raw or "").strip()
    if not t:
        return None
    if "```" in t:
        for block in t.split("```"):
            block = block.strip()
            if block.lower().startswith("json"):
                block = block[4:].lstrip()
            if block.startswith("["):
                try:
                    arr = json.loads(block)
                    if isinstance(arr, list):
                        return [str(x).strip() for x in arr if x is not None]
                except json.JSONDecodeError:
                    continue
    try:
        arr = json.loads(t)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if x is not None]
    except json.JSONDecodeError:
        pass
    start, end = t.find("["), t.rfind("]")
    if start >= 0 and end > start:
        try:
            arr = json.loads(t[start : end + 1])
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if x is not None]
        except json.JSONDecodeError:
            return None
    return None


class LangGraphScenarioRecommender(BaseRecommender):
    name_slug = "langgraph_scenario"

    def __init__(self):
        super().__init__(self.name_slug)
        self._data: pd.DataFrame | None = None
        self._index = SongTextIndex(EMBEDDING_MODEL, batch_size=EMBEDDING_BATCH_SIZE)
        self._graph = None
        self._faiss = None

    def bind_audio_faiss(self, faiss_index) -> None:
        self._faiss = faiss_index

    def fit(self, data: pd.DataFrame) -> None:
        self._data = data
        self.is_fitted = True

    def _compile_graph(self):
        rec = self

        def expand(state: ScenarioState) -> ScenarioState:
            q = (state.get("user_query") or "").strip()
            if not q:
                return {
                    "retrieval_query": "",
                    "mood_energy_bias": 0.0,
                    "llm_note": "빈 질의",
                }
            bias, mood_hint = mood_energy_bias_and_hint(q)

            def _merge_hint(base: str) -> str:
                base = base.strip()
                if mood_hint and mood_hint.lower() not in base.lower():
                    return f"{base} {mood_hint}".strip()
                return base

            if not GEMINI_API_KEY:
                rq = _merge_hint(q)
                return {
                    "retrieval_query": rq,
                    "mood_energy_bias": bias,
                    "llm_note": "LLM 미사용(키 없음); 분위기 힌트 반영"
                    if mood_hint
                    else "LLM 미사용(키 없음)",
                }
            try:
                from langchain_core.messages import HumanMessage
                from langchain_google_genai import ChatGoogleGenerativeAI

                llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.2,
                )
                prompt = (
                    "사용자는 음악 추천을 위해 한국어로 상황·분위기를 설명했다. "
                    "아래 원문을 바탕으로, 곡 검색용 짧은 문장 하나만 출력하라. "
                    "업템포·댄스·발라드·잔잔함 등 음악적 성격이 드러나게 쓸 것. "
                    "출력은 한국어 또는 영어 단일 문장, 따옴표·번호·설명 없이 본문만.\n\n"
                    f"원문: {q}"
                )
                msg = llm.invoke([HumanMessage(content=prompt)])
                text = (getattr(msg, "content", None) or str(msg)).strip()
                text = re.sub(r"^[\"']|[\"']$", "", text).strip()
                if len(text) < 2:
                    text = q
                rq = _merge_hint(text)
                return {
                    "retrieval_query": rq,
                    "mood_energy_bias": bias,
                    "llm_note": "Gemini + 분위기 힌트",
                }
            except Exception as e:
                return {
                    "retrieval_query": _merge_hint(q),
                    "mood_energy_bias": bias,
                    "llm_note": f"LLM 실패, 원문+힌트: {e!s}",
                }

        def retrieve(state: ScenarioState) -> ScenarioState:
            q_raw = (state.get("user_query") or "").strip()
            rq = (state.get("retrieval_query") or "").strip()
            if not rq or not rec._index.is_ready():
                return {"candidate_scores": {}}
            bias = float(state.get("mood_energy_bias") or 0.0)

            texts = [q_raw, rq] if q_raw != rq else [rq]
            vecs = encode_texts(
                EMBEDDING_MODEL,
                texts,
                batch_size=min(EMBEDDING_BATCH_SIZE, 8),
            )
            lists: list[list[str]] = []
            pool = 45
            for i in range(len(vecs)):
                hits = rec._index.search(vecs[i], top_k=pool)
                lists.append([sid for sid, _ in hits])
            if len(lists) == 1:
                lists = [lists[0], lists[0]]

            rrf = rrf_from_ranked_lists(lists)
            text_norm = normalize_scores(rrf)

            df = rec._data
            if df is None or df.empty:
                return {"candidate_scores": text_norm}

            e_ref = df["energy"].dropna().to_numpy(dtype=np.float64)
            b_ref = df["bpm"].dropna().to_numpy(dtype=np.float64)

            df_u = df.drop_duplicates(subset=["song_id"], keep="last")
            df_u = df_u.set_index(df_u["song_id"].astype(str), drop=False)

            audio_s: dict[str, float] = {}
            meta_s: dict[str, float] = {}
            for sid in text_norm:
                if sid not in df_u.index:
                    audio_s[sid] = 0.55
                    meta_s[sid] = 0.55
                    continue
                row = df_u.loc[sid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                en = row.get("energy")
                bp = row.get("bpm")
                audio_s[sid] = audio_affinity_for_row(e_ref, b_ref, en, bp, bias)
                meta_s[sid] = metadata_affinity(row, bias)

            faiss_n = faiss_mood_boost(df, rec._faiss, bias)
            combined = combine_scenario_scores(text_norm, audio_s, meta_s, faiss_n, bias)
            return {"candidate_scores": combined}

        def llm_select(state: ScenarioState) -> ScenarioState:
            scores = state.get("candidate_scores") or {}
            q = (state.get("user_query") or "").strip()
            if len(scores) < 3 or not q:
                return {}
            if not GEMINI_API_KEY or not SCENARIO_LLM_TOP3:
                return {}

            n = max(3, min(SCENARIO_LLM_CANDIDATES, len(scores)))
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
            allowed_order = [sid for sid, _ in ranked]
            allowed_set = set(allowed_order)

            df = rec._data
            catalog_parts: list[str] = []
            for sid, sc in ranked:
                sc_r = round(float(sc), 4)
                if df is None or df.empty:
                    catalog_parts.append(f"- song_id: {sid}\n  rank_score: {sc_r}")
                    continue
                rows = df[df["song_id"].astype(str) == sid]
                if rows.empty:
                    catalog_parts.append(f"- song_id: {sid}\n  rank_score: {sc_r}")
                    continue
                r = rows.iloc[0]
                title = str(r.get("title") or "")
                artist = str(r.get("artist") or "")
                genre = str(r.get("genre") or "")
                bpm = r.get("bpm")
                energy = r.get("energy")
                ly = r.get("lyrics")
                ly_snip = ""
                if ly is not None and str(ly).strip():
                    s = " ".join(str(ly).split())
                    ly_snip = s[:200] + ("…" if len(s) > 200 else "")
                catalog_parts.append(
                    f"- song_id: {sid}\n"
                    f"  title: {title}\n"
                    f"  artist: {artist}\n"
                    f"  genre: {genre}\n"
                    f"  bpm: {bpm}\n"
                    f"  energy: {energy}\n"
                    f"  rank_score: {sc_r}\n"
                    f"  lyrics_excerpt: {ly_snip or '(없음)'}"
                )

            catalog = "\n".join(catalog_parts)
            prompt = (
                "아래 곡 목록은 우리 서비스 DB에 실제로 있는 곡만이다. "
                "사용자 요청에 가장 잘 맞는 곡을 정확히 3곡 고르라.\n"
                "규칙:\n"
                "1) 반드시 목록에 적힌 song_id 문자열만 사용.\n"
                "2) 목록에 없는 id를 만들지 말 것.\n"
                "3) 출력은 JSON 배열만.\n"
                "4) 예: [\"id1\",\"id2\",\"id3\"]\n\n"
                f"사용자 요청:\n{q}\n\n"
                f"곡 목록:\n{catalog}\n\n"
                "JSON 배열만 출력."
            )
            try:
                from langchain_core.messages import HumanMessage
                from langchain_google_genai import ChatGoogleGenerativeAI

                llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.15,
                )
                msg = llm.invoke([HumanMessage(content=prompt)])
                raw = (getattr(msg, "content", None) or str(msg)).strip()
                parsed = _parse_llm_json_id_list(raw)
                if not parsed:
                    return {"llm_note": (state.get("llm_note") or "") + " | LLM top3 파싱 실패"}

                picked: list[str] = []
                for x in parsed:
                    if x in allowed_set and x not in picked:
                        picked.append(x)
                    if len(picked) >= 3:
                        break
                for sid in allowed_order:
                    if len(picked) >= 3:
                        break
                    if sid not in picked:
                        picked.append(sid)

                return {
                    "llm_top_ids": picked[:3],
                    "llm_note": (state.get("llm_note") or "") + " | LLM 후보 내 top3",
                }
            except Exception as e:
                return {
                    "llm_note": (state.get("llm_note") or "") + f" | LLM top3 실패: {e!s}",
                }

        def finalize(state: ScenarioState) -> ScenarioState:
            scores = state.get("candidate_scores") or {}
            llm_ids = state.get("llm_top_ids") or []
            if (
                len(llm_ids) == 3
                and len(set(llm_ids)) == 3
                and all(sid in scores for sid in llm_ids)
            ):
                return {"top_song_ids": llm_ids}
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return {"top_song_ids": [sid for sid, _ in ranked[:3]]}

        return compile_scenario_graph(expand, retrieve, llm_select, finalize)

    def _ensure_graph(self) -> None:
        if self._graph is not None:
            return
        with _INIT_LOCK:
            if self._graph is not None:
                return
            if self._data is None or self._data.empty:
                self._graph = self._compile_graph()
                return
            self._index.build(self._data, use_cache=True)
            self._graph = self._compile_graph()

    def search_by_query(self, query: str, top_k: int = 10) -> dict[str, float]:
        self._check_fitted()
        self._ensure_graph()
        if self._graph is None:
            return {}
        k = max(1, top_k)
        out_state = run_scenario_graph(self._graph, query)
        scores = out_state.get("candidate_scores") or {}
        top_ids = out_state.get("top_song_ids") or []
        if not scores and not top_ids:
            return {}
        if top_ids:
            ordered = []
            s_map = dict(scores)
            for sid in top_ids:
                if sid in s_map:
                    ordered.append((sid, s_map[sid]))
            for sid, sc in sorted(s_map.items(), key=lambda x: x[1], reverse=True):
                if sid not in {x[0] for x in ordered}:
                    ordered.append((sid, sc))
                if len(ordered) >= k:
                    break
            return dict(ordered[:k])
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k])

    def recommend(self, song_id: str, top_k: int = 10) -> dict[str, float]:
        self._check_fitted()
        self._ensure_graph()
        if self._data is None or self._data.empty or not self._index.is_ready():
            return {}
        row = self._data[self._data["song_id"].astype(str) == str(song_id)]
        if row.empty:
            return {}
        title = str(row.iloc[0].get("title") or "")
        artist = str(row.iloc[0].get("artist") or "")
        genre = str(row.iloc[0].get("genre") or "")
        text = " ".join(p for p in (title, artist, genre) if p).strip() or song_id
        qv = encode_texts(
            EMBEDDING_MODEL,
            [text],
            batch_size=min(EMBEDDING_BATCH_SIZE, 8),
        )[0]
        hits = self._index.search(qv, top_k=top_k + 8)
        out: dict[str, float] = {}
        for sid, sc in hits:
            if sid == song_id:
                continue
            out[sid] = sc
            if len(out) >= top_k:
                break
        return out
