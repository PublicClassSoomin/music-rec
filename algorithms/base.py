from abc import ABC, abstractmethod
import pandas as pd


class BaseRecommender(ABC):
    """
    모든 추천 알고리즘이 상속해야 하는 공통 인터페이스

    ✅ 필수 구현:
        fit(data)                    → 모델 학습
        recommend(song_id, top_k)    → 곡 기반 추천

    📌 반환 형태:
        recommend() → {song_id: score} 딕셔너리
        score 높을수록 추천 우선순위 높음

    💡 등록 방법:
        api/main.py의 startup()에 아래처럼 추가:
        from algorithms.my_algo import MyRecommender
        recommenders["my_algo"] = MyRecommender()
        recommenders["my_algo"].fit(data)
    """

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """
        모델 학습
        :param data: get_all_songs()로 불러온 DataFrame
                     (songs + audio_features 조인 결과)

        구현 마지막에 반드시 self.is_fitted = True 를 설정하세요.
        """
        pass

    @abstractmethod
    def recommend(self, song_id: str, top_k: int = 10) -> dict[str, float]:
        """
        곡 기반 추천
        :param song_id: 기준 곡의 song_id (YouTube video_id)
        :param top_k: 추천할 곡 수
        :return: {song_id: score} 딕셔너리 (score 내림차순)
        """
        pass

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> dict[str, float]:
        """
        유저 기반 추천 (CF / RL 계열에서 오버라이드)
        기본값: 빈 딕셔너리
        """
        return {}

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(f"[{self.name}] fit()을 먼저 호출하세요.")

    def __repr__(self):
        return f"{self.name}(fitted={self.is_fitted})"
