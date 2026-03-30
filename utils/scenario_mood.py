"""
자연어 질의에서 분위기 힌트 추출 → 임베딩 검색문 보강 + 오디오 특성 재가중에 사용.
"""

from __future__ import annotations

# 부분 문자열 매칭 (예: '신나는' → '신나')
_UPBEAT = (
    "신나",
    "업템포",
    "댄스",
    "클럽",
    "운동",
    "헬스",
    "파티",
    "경쾌",
    "텐션",
    "흥나",
    "흥 ",
    "edm",
    "하이텐션",
    "빠른",
    "뛰",
    "exciting",
    "upbeat",
    "party",
    "dance",
    "workout",
)
_CALM = (
    "발라드",
    "잔잔",
    "조용",
    "감성",
    "새벽",
    "공부",
    "집중",
    "힐링",
    "슬픈",
    "우울",
    "차분",
    "수면",
    "잠들",
    "lofi",
    "로파이",
    "chill",
    "ballad",
    "slow",
    "calm",
    "study",
)


def mood_energy_bias_and_hint(query: str) -> tuple[float, str]:
    """
    :return: (bias, retrieval_hint)
        bias: -1.0(차분/느림 선호) ~ 0(중립) ~ 1.0(활기/빠름 선호)
        retrieval_hint: 임베딩용으로 붙일 짧은 영·한 설명
    """
    q = (query or "").strip().lower()
    if not q:
        return 0.0, ""

    up = sum(1 for k in _UPBEAT if k.lower() in q)
    down = sum(1 for k in _CALM if k.lower() in q)

    if up > down and up > 0:
        return (
            1.0,
            "upbeat energetic dance pop party high tempo exciting k-pop dance music",
        )
    if down > up and down > 0:
        return (
            -1.0,
            "slow ballad acoustic emotional soft quiet calm gentle music",
        )
    return 0.0, ""
