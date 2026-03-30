"""
Offline evaluation script for CBFRecommender.

Evaluation strategy
-------------------
Primary   — Interaction-based:
    For each user who has liked ≥ 2 songs, use each liked song as a query
    and treat the user's other liked songs as the "relevant" set.

Secondary — Genre-based (fallback when interaction data is too sparse):
    For each song with audio features, treat all same-genre songs with
    audio features as the "relevant" set.

Results are printed to stdout and saved to results/cbf_results.json.

Run from the project root:
    python algorithms/cbf_evaluate.py
"""

import json
import os
import sys

# Ensure the project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data.database import get_all_songs, get_all_interactions, init_db
from data.faiss_index import MusicFaissIndex, FEATURE_COLS
from algorithms.cbf_recommender import CBFRecommender
from evaluation.metrics import evaluate_all
from utils.config import EVAL_K_LIST


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_interaction_test_data(
    song_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    songs_with_features: set[str],
) -> list[tuple[str, list[str]]]:
    """
    Build test pairs from the interactions table.
    For each user with ≥ 2 liked songs (that also have audio features):
      → one (query, relevant_list) pair per liked song
    """
    test_data = []

    if interactions_df.empty:
        return test_data

    # Keep only like / unlike events and resolve per-song net status
    like_events = interactions_df[
        interactions_df["action"].isin(["like", "unlike"])
    ].copy()

    if like_events.empty:
        return test_data

    # Most-recent action wins
    like_events = like_events.sort_values("timestamp", ascending=False)
    net = (
        like_events
        .drop_duplicates(subset=["user_id", "song_id"])
        .query("action == 'like'")
    )

    for user_id, group in net.groupby("user_id"):
        liked = [
            sid for sid in group["song_id"].tolist()
            if sid in songs_with_features
        ]
        if len(liked) < 2:
            continue
        for i, query_id in enumerate(liked):
            relevant = [sid for sid in liked if sid != query_id]
            test_data.append((query_id, relevant))

    return test_data


def build_genre_test_data(
    song_df: pd.DataFrame,
    songs_with_features: set[str],
    max_queries: int = 100,
) -> list[tuple[str, list[str]]]:
    """
    Fallback: for each song (with features), treat same-genre songs as relevant.
    Caps at max_queries to keep evaluation fast.
    """
    test_data = []

    eligible = song_df[
        song_df["song_id"].isin(songs_with_features) &
        song_df["genre"].notna() &
        (song_df["genre"] != "")
    ].copy()

    genre_groups = eligible.groupby("genre")["song_id"].apply(list)

    for _, row in eligible.iterrows():
        if len(test_data) >= max_queries:
            break
        sid   = row["song_id"]
        genre = row["genre"]
        relevant = [s for s in genre_groups.get(genre, []) if s != sid]
        if relevant:
            test_data.append((sid, relevant))

    return test_data


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("CBF Offline Evaluation")
    print("=" * 50)

    # 1. Initialise DB and load data
    init_db()
    song_df = get_all_songs()
    print(f"Songs loaded: {len(song_df)}")

    if song_df.empty:
        print("No songs in DB. Run the Melon pipeline first.")
        return

    # 2. Build FAISS index
    faiss_index = MusicFaissIndex()
    faiss_index.build(song_df)

    if not faiss_index.is_built:
        print("FAISS index could not be built (no audio features?).")
        return

    songs_with_features: set[str] = set(faiss_index.id_map)
    print(f"Songs with audio features: {len(songs_with_features)}")

    # 3. Fit recommender
    rec = CBFRecommender(faiss_index)
    rec.fit(song_df)

    # 4. Build test data
    interactions_df = get_all_interactions()
    test_data = build_interaction_test_data(song_df, interactions_df, songs_with_features)

    strategy = "interaction-based"
    if len(test_data) < 5:
        print(
            f"Interaction-based test set too small ({len(test_data)} pairs) "
            "— falling back to genre-based evaluation"
        )
        test_data = build_genre_test_data(song_df, songs_with_features)
        strategy = "genre-based"

    if not test_data:
        print("No test data available. Evaluation aborted.")
        return

    print(f"Test pairs: {len(test_data)}  (strategy: {strategy})")

    # 5. Run evaluation
    avg_metrics = evaluate_all(rec, test_data, k_list=EVAL_K_LIST)

    # 6. Build results dict
    results = {
        "recommender": rec.name,
        "strategy": strategy,
        "num_test_pairs": len(test_data),
        "k_list": EVAL_K_LIST,
        "metrics": avg_metrics,
    }

    # 7. Save JSON
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cbf_results.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved → {out_path}")

    # 8. Summary table
    print("\n── Summary ──────────────────────────────")
    for k in EVAL_K_LIST:
        p = avg_metrics.get(f"Precision@{k}", 0)
        r = avg_metrics.get(f"Recall@{k}", 0)
        n = avg_metrics.get(f"NDCG@{k}", 0)
        print(f"  K={k:2d}  Precision={p:.4f}  Recall={r:.4f}  NDCG={n:.4f}")
    print("─" * 42)


if __name__ == "__main__":
    main()
