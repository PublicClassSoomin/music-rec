"""
SQLite → FAISS 인덱스 공통 모듈

팀원들이 추천 알고리즘 구현 시 이 모듈을 가져다 쓰면 돼요.

사용 예시:
    from data.faiss_index import MusicFaissIndex

    index = MusicFaissIndex()
    index.build(song_df)

    # song_id로 유사곡 검색
    results = index.search_by_id("video_id_123", top_k=10)
    # → {song_id: score} 딕셔너리 반환
"""

import os
import numpy as np
import pandas as pd
import faiss

FAISS_CACHE_PATH = "./data/faiss.index"
ID_MAP_CACHE_PATH = "./data/faiss_id_map.npy"

# SQLite audio_features에서 사용할 컬럼
FEATURE_COLS = [
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
    "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10",
    "mfcc_11", "mfcc_12", "mfcc_13",
    "bpm", "energy", "spectral_centroid", "zcr",
]


class MusicFaissIndex:
    """
    SQLite audio_features → FAISS 인덱스 빌더 및 검색기

    내부 구조:
        SQLite (숫자 컬럼)
            ↓ numpy 변환
        정규화된 특성 행렬 (n_songs × 17)
            ↓ FAISS IndexFlatIP (내적 = 코사인 유사도)
        빠른 유사도 검색
    """

    def __init__(self):
        self.index      = None          # FAISS 인덱스
        self.id_map     = []            # 인덱스 행 번호 → song_id 매핑
        self.id_to_idx  = {}            # song_id → 인덱스 행 번호 매핑
        self.is_built   = False

    # ── 인덱스 구축 ───────────────────────────────────────

    def build(self, data: pd.DataFrame, use_cache: bool = True) -> None:
        """
        song_df (SQLite 조인 결과)로 FAISS 인덱스 구축

        :param data: get_all_songs()로 불러온 DataFrame
        :param use_cache: True면 캐시 파일 있을 때 재구축 스킵
        """
        if use_cache and self._load_cache():
            return

        print("[FAISS] 인덱스 구축 중...")

        # 오디오 특성 컬럼만 추출 (결측치 있는 행 제거)
        feat_df = data[["song_id"] + FEATURE_COLS].dropna()
        if feat_df.empty:
            print("[FAISS] 오디오 특성 데이터 없음 → 파이프라인 먼저 실행하세요")
            return

        # numpy 변환
        vectors = feat_df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
        vectors = np.ascontiguousarray(vectors)

        # L2 정규화 → 내적 = 코사인 유사도
        faiss.normalize_L2(vectors)

        # FAISS 인덱스 구축 (IndexFlatIP: 정확한 코사인 유사도)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

        # ID 매핑
        self.id_map    = feat_df["song_id"].tolist()
        self.id_to_idx = {sid: i for i, sid in enumerate(self.id_map)}

        self.is_built = True
        self._save_cache()

        print(f"[FAISS] 완료: {self.index.ntotal}곡, 차원={dim}")

    # ── 검색 ──────────────────────────────────────────────

    def search_by_id(self, song_id: str, top_k: int = 10) -> dict[str, float]:
        """
        song_id 기준으로 유사한 곡 검색
        :return: {song_id: cosine_similarity_score} 딕셔너리
        """
        self._check_built()

        if song_id not in self.id_to_idx:
            print(f"[FAISS] song_id={song_id} 인덱스에 없음")
            return {}

        idx = self.id_to_idx[song_id]
        query_vec = self.index.reconstruct(idx).reshape(1, -1)

        # top_k + 1개 검색 (자기 자신 포함되므로)
        scores, indices = self.index.search(query_vec, top_k + 1)

        results = {}
        for score, i in zip(scores[0], indices[0]):
            if i < 0 or i >= len(self.id_map):
                continue
            sid = self.id_map[i]
            if sid == song_id:              # 자기 자신 제외
                continue
            results[sid] = float(score)
            if len(results) >= top_k:
                break

        return results

    def search_by_vector(self, vector: np.ndarray, top_k: int = 10) -> dict[str, float]:
        """
        임의 벡터로 유사한 곡 검색
        LangGraph 노드에서 "조건 벡터"로 검색할 때 사용

        :param vector: (17,) float32 배열 (FEATURE_COLS 순서)
        :return: {song_id: score} 딕셔너리
        """
        self._check_built()

        vec = np.ascontiguousarray(vector.astype(np.float32).reshape(1, -1))
        faiss.normalize_L2(vec)

        scores, indices = self.index.search(vec, top_k)

        results = {}
        for score, i in zip(scores[0], indices[0]):
            if i < 0 or i >= len(self.id_map):
                continue
            results[self.id_map[i]] = float(score)

        return results

    def get_vector(self, song_id: str) -> np.ndarray | None:
        """
        특정 곡의 특성 벡터 반환
        협업 필터링 등에서 보조 특성으로 활용 가능
        """
        self._check_built()
        if song_id not in self.id_to_idx:
            return None
        return self.index.reconstruct(self.id_to_idx[song_id])

    # ── 캐시 ──────────────────────────────────────────────

    def _save_cache(self):
        faiss.write_index(self.index, FAISS_CACHE_PATH)
        np.save(ID_MAP_CACHE_PATH, np.array(self.id_map))
        print(f"[FAISS] 캐시 저장: {FAISS_CACHE_PATH}")

    def _load_cache(self) -> bool:
        if not os.path.exists(FAISS_CACHE_PATH) or not os.path.exists(ID_MAP_CACHE_PATH):
            return False
        self.index   = faiss.read_index(FAISS_CACHE_PATH)
        self.id_map  = np.load(ID_MAP_CACHE_PATH, allow_pickle=True).tolist()
        self.id_to_idx = {sid: i for i, sid in enumerate(self.id_map)}
        self.is_built = True
        print(f"[FAISS] 캐시 로드: {self.index.ntotal}곡")
        return True

    def _check_built(self):
        if not self.is_built:
            raise RuntimeError("[FAISS] build()를 먼저 호출하세요.")
