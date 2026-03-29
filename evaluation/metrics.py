import numpy as np
from utils.config import EVAL_K_LIST


def precision_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """
    Precision@K: 추천된 상위 K개 중 실제 사용자가 좋아한(relevant) 아이템의 비율
    """
    # 추천된 상위 k개와 실제 정답 셋의 교집합 개수를 구함
    hits = len(set(recommended[:k]) & set(relevant))
    # 정밀도 = 맞춘 개수 / 추천한 개수(k)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """
    Recall@K: 사용자가 좋아하는 전체 아이템 중 추천된 상위 K개에 포함된 비율
    """
    # 추천된 상위 k개와 실제 정답 셋의 교집합 개수
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """
    NDCG@K: 추천 순위를 고려한 지표. 정답이 앞쪽에 나올수록 높은 점수 부여.
    """
    # 정답 셋을 집합으로 변환하여 중복 제거
    relevant_set = set(relevant)
    # DCG(Discounted Cumulative Gain) 계산: 정답이 앞쪽에 나올수록 높은 점수 부여
    # DCG 계산: 정답인 경우 1 / log2(순위+1) 점수를 합산
    dcg = sum(
        1.0 / np.log2(i + 2) # i가 0부터 시작하므로 i+2를 분모로 사용
        for i, sid in enumerate(recommended[:k])
        if sid in relevant_set
    )

    # IDCG(Ideal DCG) 계산: 가장 이상적인 순서(앞부분이 다 정답일 때)의 점수
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    # 실제 DCG를 이상적인 IDCG로 나누어 0~1 사이로 정규화
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
    # 각 테스트 케이스(쿼리 곡, 정답 리스트)를 순회
    for query_id, relevant in test_data:
        # 모델로부터 추천 결과(점수 포함)를 받아옴
        rec_dict = recommender.recommend(query_id, top_k=max(k_list))
        # 점수가 높은 순서대로 아이템 ID만 정렬하여 리스트 생성
        recommended = sorted(rec_dict, key=rec_dict.get, reverse=True)
        # 개별 평가 결과를 리스트에 저장
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
