import numpy as np
import faiss
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional
from algorithms.base import BaseRecommender

class VibeSyncRecommender(BaseRecommender):
    '''
    Vibe-Sync 추천 엔진의 메인 클래스입니다.
    BaseRecommender 인터페이스를 상속받아 시스템의 표준 규격을 준수합니다.
    
    [핵심 로직]
    이 클래스는 곡의 17가지 오디오 특성(MFCC, BPM, Energy 등)을 기반으로,
    사용자가 UI에서 조작하는 6가지 조절바(가수, 분위기, 장르, 노래특성, 나의 감정, 모험의 정도)의 
    값을 수학적으로 해석합니다. 단순히 곡을 찾는 것을 넘어 조절바 값에 따라 타겟 벡터 자체를 왜곡시키며,
    이후 FAISS를 통해 '안주 : 탐험 : 개척'의 하이브리드 비율로 믹싱하여 역동적인 추천 리스트를 반환합니다.
    '''

    def __init__(self, name: str = "vibe_sync"):
        '''
        엔진 초기화 함수입니다.
        부모 클래스(BaseRecommender)의 초기화를 호출하여 이름을 등록하고,
        FAISS 초고속 검색 인덱스, 서로 다른 단위의 데이터를 맞추기 위한 정규화 도구(Scaler), 
        그리고 계산에 사용할 17개의 핵심 오디오 특성 컬럼명을 정의합니다.
        '''
        super().__init__(name)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.scaler: StandardScaler = StandardScaler()
        self.df: Optional[pd.DataFrame] = None
        self.features: List[str] = [
            'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 
            'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 
            'mfcc_11', 'mfcc_12', 'mfcc_13', 'bpm', 'energy', 
            'spectral_centroid', 'zcr'
        ]

    def fit(self, data: pd.DataFrame) -> None:
        '''
        [필수 구현] 데이터베이스에서 가져온 전체 곡 목록을 학습하는 함수입니다.
        
        [로직 흐름]
        1. 전달받은 DataFrame에서 17개의 오디오 특성 데이터만 추출합니다.
        2. StandardScaler를 사용하여 서로 다른 단위(예: BPM 120, Energy 0.5)를 동일한 비중으로 정규화합니다.
        3. 정규화된 32비트 실수형(float32) 데이터를 FAISS의 L2(유클리드 거리) 인덱스에 등록합니다.
        4. BaseRecommender의 규칙에 따라 self.is_fitted를 True로 변경합니다.
        '''
        self.df = data.copy()
        raw_features = self.df[self.features].values.astype('float32')
        
        # 데이터 정규화 및 FAISS 인덱스 빌드
        scaled_features = self.scaler.fit_transform(raw_features)
        
        self.index = faiss.IndexFlatL2(len(self.features))
        self.index.add(np.ascontiguousarray(scaled_features))
        
        self.is_fitted = True

    def _create_target_vector(self, song_id: str, params: Dict[str, Any]) -> np.ndarray:
        '''
        사용자의 6가지 조절바 설정을 적극적으로 반영하여 FAISS가 검색할 '돌연변이 타겟 벡터'를 생성합니다.
        
        [로직 흐름]
        1. 기준이 되는 곡(song_id)의 원본 유전자(특징 벡터)를 추출합니다.
        2. [노래 분위기(Mood)]: 조절바 값(0~100)을 바탕으로 곡의 'energy'와 'spectral_centroid(밝기)'를 조작합니다.
        3. [가수(Artist Affinity)]: '새로운 가수' 쪽으로 갈수록 음의 거칠기인 'ZCR'을 비틀어 다른 스타일을 유도합니다.
        4. [노래특성(Audio Similarity)]: '느낌 일치' 쪽으로 갈수록 음색을 결정하는 13개의 'MFCC' 값 전체에 
           무작위 노이즈를 흩뿌려 비슷한 분위기지만 완전히 다른 소리를 가진 곡을 찾게 만듭니다.
        5. [나의 감정(Emotion)]: '지금 기분' 쪽으로 갈수록 'BPM(템포)'을 흔들어버립니다.
        6. 처참하게(?) 왜곡된 벡터를 학습 시 사용한 Scaler로 다시 정규화하여 반환합니다.
        '''
        base_row = self.df[self.df['song_id'] == song_id]
        if base_row.empty:
            # 기준 곡이 없으면 전체 평균 벡터를 반환하여 뻗는 것을 방지
            return self.scaler.mean_.reshape(1, -1).astype('float32')

        target_vec = base_row[self.features].values.astype('float32').copy()[0]

        # 1. 노래 분위기 (Mood) -> 에너지와 밝기 조절
        mood_factor = (params.get('mood', 50) - 50) / 50.0
        target_vec[self.features.index('energy')] += mood_factor * 0.3
        target_vec[self.features.index('spectral_centroid')] += mood_factor * 800

        # 2. 가수 (Artist Affinity) -> 음의 거칠기(ZCR) 변형으로 다른 스타일 유도
        art_factor = (params.get('artist', 50) - 50) / 50.0
        target_vec[self.features.index('zcr')] += art_factor * 0.05

        # 3. 노래특성 (Audio Similarity) -> MFCC(음색)에 노이즈를 섞어 파격적인 결과 유도
        sim_factor = params.get('similarity', 50) / 100.0
        if sim_factor > 0.3:
            noise = np.random.normal(0, sim_factor * 2, 13)
            for i in range(1, 14):
                target_vec[self.features.index(f'mfcc_{i}')] += noise[i-1]

        # 4. 나의 감정 (Emotion) -> 템포(BPM)를 흔들기
        emo_factor = (params.get('emotion', 50) - 50) / 50.0
        target_vec[self.features.index('bpm')] += emo_factor * 30

        return self.scaler.transform(target_vec.reshape(1, -1)).astype('float32')

    def _sample_songs(self, pool: List[pd.Series], n: int) -> List[str]:
        '''
        후보군 바구니(pool)에서 필요한 개수(n)만큼 무작위로 곡을 안전하게 추출하는 보조 함수입니다.
        추출된 곡들의 고유 ID(song_id)만 리스트 형태로 반환합니다.
        '''
        if not pool or n <= 0: return []
        samples = random.sample(pool, min(len(pool), n))
        return [s['song_id'] for s in samples]

    def recommend(self, song_id: str, top_k: int = 10, params: Optional[Dict[str, Any]] = None) -> dict[str, float]:
        '''
        [필수 구현] 조작된 벡터와 '모험의 정도'를 융합하여 최종 믹싱 리스트를 뱉어내는 메인 함수입니다.
        
        [로직 흐름]
        1. DB에 곡이 존재하는지 검증합니다.
        2. '나만의 믹싱(manual_mode)' 켜짐 여부에 따라 원본 벡터를 쓸지, _create_target_vector로 만든 돌연변이 벡터를 쓸지 결정합니다.
        3. FAISS를 통해 거리가 가장 가까운 상위 200곡의 후보를 아주 넉넉하게 추출합니다.
        4. 후보군을 3가지 바구니(안주-같은 장르 유사곡, 탐험-같은 장르 먼 곡, 개척-다른 장르)로 분류합니다.
        5. [모험의 정도(Adventure)] 파라미터에 따라 비율을 결정합니다.
           - 안전(<30): 안주 100%
           - 중간(<70): 안주 50%, 탐험 50%
           - 극한(>=70): 안주 30%, 탐험 30%, 개척(장르 파괴) 나머지 전부
        6. 추출된 곡들을 매번 무작위(Shuffle)로 섞어 조절바를 만질 때마다 새로운 결과가 눈에 띄게 배치합니다.
        7. BaseRecommender 규격에 맞게 {song_id: score} 형태의 딕셔너리로 반환합니다.
        '''
        self._check_fitted()
        
        base_rows = self.df[self.df['song_id'] == song_id]
        if base_rows.empty:
            return {}

        base_genre = base_rows.iloc[0]['genre']
        p = params or {"adventure": 50, "mood": 50, "emotion": 50, "manual_mode": False}
        
        # 🎯 조절바 값으로 벡터 타겟팅
        if p.get('manual_mode', False):
            target_v = self._create_target_vector(song_id, p)
        else:
            raw_v = base_rows[self.features].values.astype('float32')
            target_v = self.scaler.transform(raw_v)

        # 넉넉하게 200곡 추출 후 필터링
        distances, indices = self.index.search(target_v, 200)
        comfort_pool, explor_pool, pioneer_pool = [], [], []

        for idx in indices[0]:
            song = self.df.iloc[idx]
            if song['song_id'] == song_id: continue
            
            if song['genre'] == base_genre:
                if len(comfort_pool) < 30: 
                    comfort_pool.append(song)
                else: 
                    explor_pool.append(song)
            else:
                pioneer_pool.append(song)

        # 🎯 모험의 정도 (Adventure) 파라미터 적용 로직
        adv = p.get('adventure', 50)
        if adv < 30:   # 안전: 비슷한 곡 위주
            n_c, n_e, n_p = top_k, 0, 0
        elif adv < 70: # 중간: 조금 섞기
            n_c = max(1, int(top_k * 0.5))
            n_e = top_k - n_c
            n_p = 0
        else:          # 극한: 아예 다른 장르(Pioneer) 강제 투입
            n_c = max(1, int(top_k * 0.3))
            n_e = max(1, int(top_k * 0.3))
            n_p = top_k - n_c - n_e

        picked_ids = []
        picked_ids.extend(self._sample_songs(comfort_pool, n_c))
        picked_ids.extend(self._sample_songs(explor_pool, n_e))
        picked_ids.extend(self._sample_songs(pioneer_pool, n_p))

        # 데이터가 부족해 개수가 안 채워지면 비슷한 곡으로 땜빵
        while len(picked_ids) < top_k and comfort_pool:
            extra = self._sample_songs(comfort_pool, 1)
            if extra[0] not in picked_ids:
                picked_ids.extend(extra)
            else:
                break

        # 🎯 핵심: 매번 무작위로 섞어서 맨 앞에 새로운 곡이 오도록 구성
        random.shuffle(picked_ids)
        
        result = {}
        for i, sid in enumerate(picked_ids[:top_k]):
            result[sid] = 1.0 - (i * 0.01)
            
        return result