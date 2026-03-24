import numpy as np
from utils.config import EVAL_K_LIST


def precision_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    relevant_set = set(relevant)
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, sid in enumerate(recommended[:k])
        if sid in relevant_set
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def skip_rate(interactions: list[dict]) -> float:
    """스킵률: skip / 전체 play"""
    total = len(interactions)
    skips = sum(1 for i in interactions if i["action"] == "skip")
    return skips / total if total > 0 else 0.0


def completion_rate(interactions: list[dict], song_duration: int) -> float:
    """완청률: 90% 이상 들은 비율"""
    plays = [i for i in interactions if i["action"] == "play"]
    if not plays:
        return 0.0
    completed = sum(
        1 for i in plays
        if i.get("play_seconds", 0) >= song_duration * 0.9
    )
    return completed / len(plays)


def evaluate(
    recommended: list[str],
    relevant: list[str],
    k_list: list[int] = None,
) -> dict[str, float]:
    """단일 추천 결과 전체 지표 계산"""
    if k_list is None:
        k_list = EVAL_K_LIST

    results = {}
    for k in k_list:
        results[f"Precision@{k}"] = precision_at_k(recommended, relevant, k)
        results[f"Recall@{k}"]    = recall_at_k(recommended, relevant, k)
        results[f"NDCG@{k}"]      = ndcg_at_k(recommended, relevant, k)
    return results


def evaluate_all(
    recommender,
    test_data: list[tuple[str, list[str]]],
    k_list: list[int] = None,
) -> dict[str, float]:
    """
    여러 쿼리 평균 지표 계산 후 출력
    :param test_data: [(query_song_id, [relevant_song_ids]), ...]
    """
    if k_list is None:
        k_list = EVAL_K_LIST

    all_results = []
    for query_id, relevant in test_data:
        rec_dict = recommender.recommend(query_id, top_k=max(k_list))
        recommended = sorted(rec_dict, key=rec_dict.get, reverse=True)
        all_results.append(evaluate(recommended, relevant, k_list))

    avg = {
        key: float(np.mean([r[key] for r in all_results]))
        for key in all_results[0]
    }

    print(f"\n📊 [{recommender.name}] 평가 결과")
    print("-" * 40)
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")
    print("-" * 40)

    return avg
