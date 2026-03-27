import numpy as np
import pandas as pd
from data.database import init_db, get_all_songs
from algorithms.faiss_cbf import FaissContentRecommender
from algorithms.vibe_sync import VibeSyncRecommender
from data.faiss_index import MusicFaissIndex

def calculate_metrics(base_song, rec_dict, df):
    """추천된 곡들의 다양성과 유사도를 계산하는 함수"""
    if not rec_dict:
        return 0.0, 0.0
        
    rec_songs = df[df['song_id'].isin(rec_dict.keys())]
    
    # 1. 장르 다양성 (Diversity): 추천된 곡 중 고유 장르의 비율
    genres = rec_songs['genre'].dropna().unique()
    diversity = len(genres) / len(rec_dict) if len(rec_dict) > 0 else 0
    
    # 2. BPM 분산도 (Variance): 템포가 얼마나 다채로운지
    # 데이터에 'bpm'이 없다면 제외해도 무방합니다.
    bpm_variance = 0.0
    if 'bpm' in rec_songs.columns:
        bpm_variance = np.var(rec_songs['bpm'].values)
        
    return diversity, bpm_variance

def run_evaluation():
    print("📊 추천 모델 정량적 성능 평가 시작...\n")
    
    # 1. 데이터 및 모델 세팅
    init_db()
    df = get_all_songs()
    
    faiss_index = MusicFaissIndex()
    faiss_index.build(df)
    
    model_baseline = FaissContentRecommender(faiss_index)
    model_baseline.fit(df)
    
    model_vibe = VibeSyncRecommender()
    model_vibe.fit(df)
    
    # 2. 테스트할 샘플 곡 5개 무작위 선정
    test_songs = df.sample(5).to_dict('records')
    
    results = []
    
    # 3. 모델별 추천 결과 도출 및 지표 계산
    for song in test_songs:
        sid = song['song_id']
        title = song['title']
        
        # [비교군 A] 일반 CBF 모델 (기존 방식)
        recs_base = model_baseline.recommend(sid, top_k=10)
        div_base, var_base = calculate_metrics(song, recs_base, df)
        
        # [비교군 B] Vibe-Sync (안전 모드: 모험 10)
        recs_vibe_safe = model_vibe.recommend(sid, top_k=10, params={"manual_mode": True, "adventure": 10})
        div_safe, var_safe = calculate_metrics(song, recs_vibe_safe, df)
        
        # [비교군 C] Vibe-Sync (극한 모드: 모험 90, 감정 90)
        recs_vibe_extreme = model_vibe.recommend(sid, top_k=10, params={"manual_mode": True, "adventure": 90, "emotion": 90})
        div_extreme, var_extreme = calculate_metrics(song, recs_vibe_extreme, df)
        
        results.append({
            "Test Song": title,
            "Baseline_Div": div_base,
            "Vibe(Safe)_Div": div_safe,
            "Vibe(Extreme)_Div": div_extreme,
        })
        
    # 4. 결과 출력 (보고서용)
    print("=== 📈 장르 다양성 (Genre Diversity Score) ===")
    print(f"{'Test Song':<25} | {'Baseline (CBF)':<15} | {'Vibe (Safe)':<15} | {'Vibe (Extreme)':<15}")
    print("-" * 75)
    for r in results:
        # 타이틀 길이가 길면 자름
        short_title = r['Test Song'][:20] + "..." if len(r['Test Song']) > 20 else r['Test Song']
        print(f"{short_title:<25} | {r['Baseline_Div']:<15.2f} | {r['Vibe(Safe)_Div']:<15.2f} | {r['Vibe(Extreme)_Div']:<15.2f}")
    
    print("\n💡 결론 해석: Vibe(Extreme)의 다양성 점수가 압도적으로 높아야 우리의 하이브리드 로직이 성공적으로 작동한 것입니다.")

if __name__ == "__main__":
    run_evaluation()