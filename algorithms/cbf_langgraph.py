"""
LangGraph pipeline for natural-language music recommendation.

5-node pipeline:
  1. intent_recognition   — Gemini extracts mood/activity/genre/energy from the query
  2. feature_mapping      — maps intent to audio feature ranges → query vector
  3. cbf_search           — runs CBFRecommender with the query vector via FAISS
  4. generate_explanation — Gemini writes a user-friendly explanation
  5. collect_feedback     — structures the feedback template (like / skip)

Usage:
    from algorithms.cbf_langgraph import CBFLangGraphPipeline
    pipeline = CBFLangGraphPipeline(cbf_recommender, song_df)
    result = pipeline.run("music for a rainy study session")
    print(result["explanation"])
    print(result["recommendations"])   # {song_id: score}
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from data.faiss_index import FEATURE_COLS
from utils.config import GEMINI_API_KEY, LLM_MODEL


# ── Pipeline state ────────────────────────────────────────────────────────────

class MusicQueryState(TypedDict, total=False):
    query: str                      # original user query
    intent: dict                    # Node 1 output
    feature_conditions: dict        # Node 2 output — BPM/energy ranges
    query_vector: list              # Node 2 output — 17-dim float list
    recommendations: dict           # Node 3 output — {song_id: score}
    rec_metadata: list              # Node 3 output — [{title, artist, genre}, ...]
    explanation: str                # Node 4 output
    feedback: dict                  # Node 5 output — {song_id: "pending"}
    error: str                      # set on unexpected failure


# ── Intent → feature condition mappings ──────────────────────────────────────
# Values are used to *filter* song_df and derive a representative query vector.
# BPM ranges are in beats-per-minute; energy/zcr are unitless ratios.

_BPM_RANGES: dict[str, tuple[float, float]] = {
    # mood
    "chill":     (60,  90),
    "sad":       (55,  85),
    "focused":   (70, 100),
    "happy":     (100, 140),
    "energetic": (120, 175),
    "romantic":  (65,  95),
    "neutral":   (70, 130),
    # activity
    "study":     (60,  95),
    "sleep":     (50,  75),
    "relax":     (60,  95),
    "commute":   (90, 130),
    "workout":   (120, 180),
    "party":     (120, 160),
    "general":   (70, 130),
}

_ENERGY_QUANTILE: dict[str, tuple[float, float]] = {
    "low":    (0.0,  0.33),
    "medium": (0.33, 0.67),
    "high":   (0.67, 1.0),
}


# ── Pipeline class ────────────────────────────────────────────────────────────

class CBFLangGraphPipeline:
    """Wraps a compiled LangGraph StateGraph for natural-language music search."""

    def __init__(self, cbf_recommender, song_df: pd.DataFrame):
        """
        Args:
            cbf_recommender: a fitted CBFRecommender instance (has ._index)
            song_df:         the full songs + audio_features DataFrame
        """
        self._rec = cbf_recommender
        self._song_df = song_df
        self._llm = self._init_llm()
        self._graph = self._build_graph()

    # ── LLM initialisation ───────────────────────────────────────────────

    def _init_llm(self):
        if not GEMINI_API_KEY:
            print("[CBFPipeline] GEMINI_API_KEY not set — LLM nodes will use fallbacks")
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=0.3,
            )
        except Exception as e:
            print(f"[CBFPipeline] LLM init failed: {e}")
            return None

    # ── Graph construction ───────────────────────────────────────────────

    def _build_graph(self):
        workflow = StateGraph(MusicQueryState)

        workflow.add_node("intent_recognition",   self._node_intent)
        workflow.add_node("feature_mapping",       self._node_feature_mapping)
        workflow.add_node("cbf_search",            self._node_cbf_search)
        workflow.add_node("generate_explanation",  self._node_explanation)
        workflow.add_node("collect_feedback",      self._node_feedback)

        workflow.set_entry_point("intent_recognition")
        workflow.add_edge("intent_recognition",  "feature_mapping")
        workflow.add_edge("feature_mapping",     "cbf_search")
        workflow.add_edge("cbf_search",          "generate_explanation")
        workflow.add_edge("generate_explanation","collect_feedback")
        workflow.add_edge("collect_feedback",    END)

        return workflow.compile()

    # ── Node 1: Intent Recognition ───────────────────────────────────────

    def _node_intent(self, state: MusicQueryState) -> dict:
        """
        Use Gemini to extract structured intent from the user's natural-language query.
        Falls back to neutral defaults if the LLM is unavailable or returns bad JSON.
        """
        query = state.get("query", "")

        default_intent = {
            "mood": "neutral",
            "activity": "general",
            "time_of_day": "any",
            "genre_preference": "any",
            "energy_level": "medium",
        }

        if not self._llm or not query:
            return {"intent": default_intent}

        prompt = f"""You are a music preference extractor.
Given the user's query, return ONLY a valid JSON object with these fields:
  "mood"             : one of [chill, energetic, sad, happy, focused, romantic, neutral]
  "activity"         : one of [study, workout, sleep, party, commute, relax, general]
  "time_of_day"      : one of [morning, afternoon, evening, night, any]
  "genre_preference" : one of [k-pop, pop, hiphop, jazz, classical, indie, electronic, any]
  "energy_level"     : one of [low, medium, high]

User query: "{query}"

JSON:"""

        try:
            from langchain_core.messages import HumanMessage
            response = self._llm.invoke([HumanMessage(content=prompt)])
            text = response.content.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            intent = json.loads(text)
            # Fill any missing keys with defaults
            for k, v in default_intent.items():
                intent.setdefault(k, v)
            print(f"[Node1] intent: {intent}")
            return {"intent": intent}
        except Exception as e:
            print(f"[Node1] LLM call failed ({e}), using defaults")
            return {"intent": default_intent}

    # ── Node 2: Feature Mapping ──────────────────────────────────────────

    def _node_feature_mapping(self, state: MusicQueryState) -> dict:
        """
        Convert the structured intent into:
          - feature_conditions: dict with BPM and energy quantile ranges
          - query_vector: 17-dim float array derived by averaging matching songs
        """
        intent = state.get("intent", {})
        mood = intent.get("mood", "neutral")
        activity = intent.get("activity", "general")
        energy_level = intent.get("energy_level", "medium")

        # Determine BPM range (activity takes priority over mood)
        bpm_range = _BPM_RANGES.get(activity) or _BPM_RANGES.get(mood) or (70, 130)
        energy_q  = _ENERGY_QUANTILE.get(energy_level, (0.33, 0.67))

        feature_conditions = {
            "bpm_min": bpm_range[0],
            "bpm_max": bpm_range[1],
            "energy_q_low":  energy_q[0],
            "energy_q_high": energy_q[1],
        }

        # Build a query vector from songs that match the conditions
        query_vector = self._build_query_vector(feature_conditions)

        print(f"[Node2] conditions: {feature_conditions}, "
              f"vector non-zero dims: {int(np.count_nonzero(query_vector))}")
        return {
            "feature_conditions": feature_conditions,
            "query_vector": query_vector.tolist(),
        }

    def _build_query_vector(self, conditions: dict) -> np.ndarray:
        """
        Filter song_df by BPM/energy conditions and return the mean feature vector.
        Falls back to the dataset mean if no songs match.
        """
        feat_cols_in_df = [c for c in FEATURE_COLS if c in self._song_df.columns]
        feat_df = self._song_df[["song_id"] + feat_cols_in_df].dropna()

        if feat_df.empty:
            return np.zeros(len(FEATURE_COLS), dtype=np.float32)

        # Energy quantile thresholds
        e_low  = feat_df["energy"].quantile(conditions["energy_q_low"])
        e_high = feat_df["energy"].quantile(conditions["energy_q_high"])

        mask = (
            (feat_df["bpm"]    >= conditions["bpm_min"]) &
            (feat_df["bpm"]    <= conditions["bpm_max"]) &
            (feat_df["energy"] >= e_low) &
            (feat_df["energy"] <= e_high)
        )
        matched = feat_df[mask]

        if matched.empty:
            # Widen: use only BPM range without energy constraint
            mask2 = (
                (feat_df["bpm"] >= conditions["bpm_min"]) &
                (feat_df["bpm"] <= conditions["bpm_max"])
            )
            matched = feat_df[mask2]

        if matched.empty:
            matched = feat_df   # full dataset mean as last resort

        # Pad or trim to exactly len(FEATURE_COLS) dimensions
        vec = matched[feat_cols_in_df].mean().to_numpy(dtype=np.float32)
        if len(vec) < len(FEATURE_COLS):
            vec = np.pad(vec, (0, len(FEATURE_COLS) - len(vec)))
        return vec

    # ── Node 3: CBF Search ───────────────────────────────────────────────

    def _node_cbf_search(self, state: MusicQueryState) -> dict:
        """
        Search FAISS using the query vector from Node 2.
        Enriches results with song metadata for Node 4.
        """
        query_vector = state.get("query_vector", [])
        if not query_vector:
            return {"recommendations": {}, "rec_metadata": []}

        if not getattr(self._rec._index, "is_built", False):
            print("[Node3] FAISS index not built — returning empty results")
            return {"recommendations": {}, "rec_metadata": []}

        vec = np.array(query_vector, dtype=np.float32)
        recommendations = self._rec._index.search_by_vector(vec, top_k=10)

        # Attach metadata for the explanation node
        rec_metadata = []
        for sid in list(recommendations.keys())[:5]:  # top 5 for explanation
            row = self._song_df[self._song_df["song_id"] == sid]
            if not row.empty:
                r = row.iloc[0]
                rec_metadata.append({
                    "song_id": sid,
                    "title":   r.get("title", "Unknown"),
                    "artist":  r.get("artist", "Unknown"),
                    "genre":   r.get("genre", ""),
                })

        print(f"[Node3] found {len(recommendations)} recommendations")
        return {"recommendations": recommendations, "rec_metadata": rec_metadata}

    # ── Node 4: Generate Explanation ─────────────────────────────────────

    def _node_explanation(self, state: MusicQueryState) -> dict:
        """
        Use Gemini to produce a friendly 2-3 sentence explanation of the results.
        Falls back to a template when the LLM is unavailable.
        """
        query    = state.get("query", "")
        intent   = state.get("intent", {})
        metadata = state.get("rec_metadata", [])

        if not metadata:
            return {"explanation": "No matching songs were found for your request."}

        song_list = ", ".join(
            f'"{m["title"]}" by {m["artist"]}' for m in metadata
        )

        # Template fallback (used when LLM is unavailable)
        fallback = (
            f"Based on your request for \"{query}\", here are songs that match a "
            f"{intent.get('energy_level', 'medium')}-energy, "
            f"{intent.get('mood', 'neutral')} mood: {song_list}."
        )

        if not self._llm:
            return {"explanation": fallback}

        prompt = f"""You are a friendly music recommendation assistant.

The user asked for: "{query}"
Detected intent: mood={intent.get('mood')}, activity={intent.get('activity')}, energy={intent.get('energy_level')}

Top recommended songs: {song_list}

Write a friendly 2-3 sentence explanation of why these songs fit the request.
Focus on the mood, energy, and how they match the context.
Do NOT mention technical terms like BPM, MFCC, or cosine similarity."""

        try:
            from langchain_core.messages import HumanMessage
            response = self._llm.invoke([HumanMessage(content=prompt)])
            explanation = response.content.strip()
            print(f"[Node4] explanation generated ({len(explanation)} chars)")
            return {"explanation": explanation}
        except Exception as e:
            print(f"[Node4] LLM call failed ({e}), using template")
            return {"explanation": fallback}

    # ── Node 5: Feedback Collection ──────────────────────────────────────

    def _node_feedback(self, state: MusicQueryState) -> dict:
        """
        Build a feedback template: {song_id: "pending"} for each recommendation.
        The caller can update values to "like" or "skip" and POST them to
        /api/interact to persist feedback.

        If the state already contains feedback (e.g. second-pass invocation),
        it is preserved and merged with any new recommendations.
        """
        recommendations = state.get("recommendations", {})
        existing_feedback: dict = state.get("feedback", {}) or {}

        feedback_template = {
            sid: existing_feedback.get(sid, "pending")
            for sid in recommendations
        }
        print(f"[Node5] feedback template ready for {len(feedback_template)} songs")
        return {"feedback": feedback_template}

    # ── Public API ───────────────────────────────────────────────────────

    def run(self, query: str, feedback: dict | None = None) -> dict[str, Any]:
        """
        Run the full 5-node pipeline for a natural-language query.

        Args:
            query:    e.g. "calm music for a rainy day"
            feedback: optional prior feedback dict {song_id: "like"|"skip"}

        Returns:
            The final pipeline state dict with keys:
              query, intent, feature_conditions, query_vector,
              recommendations, rec_metadata, explanation, feedback
        """
        initial_state: MusicQueryState = {
            "query":              query,
            "intent":             {},
            "feature_conditions": {},
            "query_vector":       [],
            "recommendations":    {},
            "rec_metadata":       [],
            "explanation":        "",
            "feedback":           feedback or {},
        }
        return dict(self._graph.invoke(initial_state))
