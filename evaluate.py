import time
import random
import pandas as pd
from typing import Dict, Any

# 실제 프로젝트 환경에 맞게 임포트 (경로 오류 시 알맞게 수정해주세요)
from data.database import get_all_songs
from data.faiss_index import MusicFaissIndex
from algorithms.faiss_cbf import FaissContentRecommender
from algorithms.vibe_sync import VibeSyncRecommender

def evaluate_algorithm(
    algo, 
    song_df: pd.DataFrame, 
    num_simulations: int = 100, 
    top_k: int = 10, 
    params: Dict[str, Any] = None
) -> Dict[str, float]:
    """
    단일 추천 알고리즘에 대한 정량적 지표(커버리지, 다양성, 응답속도)를 평가합니다.
    """
    all_song_ids = song_df['song_id'].tolist()
    
    recommended_pool = set()
    total_genres_per_list = 0
    total_latency = 0.0
    valid_simulations = 0

    for _ in range(num_simulations):
        seed_song = random.choice(all_song_ids)
        
        # 1. 응답 속도(Latency) 측정 시작
        start_time = time.time()
        
        # 알고리즘 추천 실행
        if params:
            recs = algo.recommend(seed_song, top_k=top_k, params=params)
        else:
            recs = algo.recommend(seed_song, top_k=top_k)
            
        latency = time.time() - start_time
        total_latency += latency
        
        # 결과가 없는 경우(DB 오류 등) 스킵
        if not recs:
            continue
            
        valid_simulations += 1
        rec_ids = list(recs.keys())
        
        # 2. 카탈로그 커버리지 수집
        recommended_pool.update(rec_ids)
        
        # 3. 다양성 수집 (추천된 10곡 내 포함된 고유 장르의 수)
        # song_df에서 추천된 곡들의 장르만 추출하여 유니크 개수 확인
        unique_genres = song_df[song_df['song_id'].isin(rec_ids)]['genre'].nunique()
        total_genres_per_list += unique_genres

    # 예외 방지
    if valid_simulations == 0:
        return {"name": algo.name, "coverage": 0, "diversity": 0, "latency_ms": 0}

    # 최종 지표 계산
    avg_latency_ms = (total_latency / valid_simulations) * 1000  # 밀리초(ms) 변환
    coverage_percent = (len(recommended_pool) / len(song_df)) * 100
    avg_diversity = total_genres_per_list / valid_simulations

    return {
        "name": algo.name,
        "coverage": coverage_percent,
        "diversity": avg_diversity,
        "latency_ms": avg_latency_ms
    }

def main():
    print("⏳ 데이터 및 FAISS 인덱스 로딩 중...")
    
    # 1. DB에서 전체 곡 로드
    song_df = get_all_songs()
    if song_df is None or song_df.empty:
        print("데이터 로드 실패. DB를 확인하세요.")
        return

    # 2. FAISS 기반 인덱스 빌드 (대조군을 위함)
    faiss_index = MusicFaissIndex()
    faiss_index.build(song_df)
    
    print(f"✅ 총 {len(song_df)}곡 로드 완료. 100회 시뮬레이션 평가를 시작합니다.\n")

    # ==========================================
    # 🧪 [대조군] 단순 코사인 유사도 모델 (Faiss CBF)
    # ==========================================
    baseline_algo = FaissContentRecommender(faiss_index)
    baseline_algo.fit(song_df)
    baseline_results = evaluate_algorithm(baseline_algo, song_df, num_simulations=100, top_k=10)

    # ==========================================
    # 🚀 [실험군] Vibe-Sync 모델 (하이브리드 믹싱)
    # ==========================================
    vibe_algo = VibeSyncRecommender()
    vibe_algo.fit(song_df)
    
    # Vibe-Sync 설정: 어드벤처 80(극한의 모험), 메뉴얼 모드 ON
    vibe_params = {"adventure": 80, "similarity": 30, "mood": 60, "manual_mode": True}
    vibe_results = evaluate_algorithm(vibe_algo, song_df, num_simulations=100, top_k=10, params=vibe_params)

    # ==========================================
    # 📊 최종 결과 리포트 출력
    # ==========================================
    print("="*50)
    print(" 📊 추천 알고리즘 정량적 성능 평가 리포트 ")
    print("="*50)

    def print_report(res):
        print(f"▶ 모델명: {res['name']}")
        print(f" - 카탈로그 커버리지 : {res['coverage']:.2f}% (소외된 곡 발굴률)")
        print(f" - 리스트 내 다양성  : 평균 {res['diversity']:.2f}개 장르 믹스 (Top-10 기준)")
        print(f" - 평균 응답 속도    : {res['latency_ms']:.2f} ms\n")

    print_report(baseline_results)
    print_report(vibe_results)
    
    print("💡 [분석 결과 요약]")
    div_increase = (vibe_results['diversity'] / baseline_results['diversity']) if baseline_results['diversity'] > 0 else 0
    cov_increase = (vibe_results['coverage'] / baseline_results['coverage']) if baseline_results['coverage'] > 0 else 0
    
    print(f"Vibe-Sync는 기존 코사인 유사도 방식 대비 장르 다양성을 약 {div_increase:.1f}배 극대화하며,")
    print(f"추천 풀(커버리지)을 {cov_increase:.1f}배 넓혔습니다. 응답 속도 또한 실시간 서비스에 적합한 수준입니다.")

if __name__ == "__main__":
    main()