import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import MinMaxScaler

# 우리가 만든 모듈 임포트
from data.database import init_db, get_all_songs
from data.faiss_index import MusicFaissIndex
from algorithms.faiss_cbf import FaissContentRecommender
from algorithms.vibe_sync import VibeSyncRecommender

def calculate_diversity(rec_dict, df):
    """추천된 곡들의 장르 다양성 계산"""
    if not rec_dict: return 0.0
    rec_songs = df[df['song_id'].isin(rec_dict.keys())]
    genres = rec_songs['genre'].dropna().unique()
    return len(genres) / len(rec_dict)

def run_evaluation_and_visualize():
    print("🚀 데이터 로딩 및 모델 평가 준비 중...")
    # 1. 실제 DB 데이터 로드 및 모델 세팅
    init_db()
    df = get_all_songs()
    
    faiss_index = MusicFaissIndex()
    faiss_index.build(df)
    
    model_baseline = FaissContentRecommender(faiss_index)
    model_baseline.fit(df)
    
    model_vibe = VibeSyncRecommender()
    model_vibe.fit(df)
    
    # 2. 17개의 특성 중 육각형의 6개 축을 대표할 6개 특성 추출 및 스케일링
    # (시각화를 위해 실제 오디오 데이터를 0.1~1.0 사이의 가중치로 변환)
    features_for_hex = ['energy', 'bpm', 'zcr', 'spectral_centroid', 'mfcc_1', 'mfcc_2']
    scaler = MinMaxScaler(feature_range=(0.1, 1.0))
    song_weights = scaler.fit_transform(df[features_for_hex])

    # 3. 6각 꼭짓점 좌표 계산
    labels = ['Artist (ZCR)', 'Mood (Energy)', 'Genre (Centroid)', 'Similarity (MFCC1)', 'Emotion (BPM)', 'Adventure (MFCC2)']
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    vertices = np.column_stack((np.cos(angles), np.sin(angles)))

    # 6D -> 2D 맵핑 함수
    def to_2d(weights):
        x = np.sum(weights * vertices[:, 0]) / np.sum(weights)
        y = np.sum(weights * vertices[:, 1]) / np.sum(weights)
        return x, y

    # 모든 곡의 2D 좌표 계산 후 DataFrame에 저장
    coords = np.array([to_2d(w) for w in song_weights])
    df['hex_x'] = coords[:, 0]
    df['hex_y'] = coords[:, 1]

    # 4. 기준 곡 선정 (테스트용)
    base_song = df.iloc[0] 
    base_id = base_song['song_id']
    base_x, base_y = base_song['hex_x'], base_song['hex_y']

    print(f"🎯 기준 곡: {base_song['title']} - {base_song['artist']}")

    # 5. 추천 평가 실행 (Baseline vs Vibe-Sync)
    top_k = 10
    # [비교군 A] 일반 CBF 모델
    recs_baseline = model_baseline.recommend(base_id, top_k=top_k)
    div_baseline = calculate_diversity(recs_baseline, df)
    
    # [비교군 B] Vibe-Sync (극한 모드)
    recs_vibe = model_vibe.recommend(base_id, top_k=top_k, params={
        "manual_mode": True, "artist": 80, "mood": 90, "genre": 90, 
        "similarity": 80, "emotion": 90, "adventure": 95
    })
    div_vibe = calculate_diversity(recs_vibe, df)

    print(f"📊 다양성 점수 - Baseline: {div_baseline:.2f} | Vibe-Sync: {div_vibe:.2f}")

    # ==========================================
    # 6. 플롯 시각화 시작
    # ==========================================
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')

    # 육각형 및 라벨 그리기
    hex_polygon = patches.Polygon(vertices, closed=True, fill=False, edgecolor=(1, 1, 1, 0.2), linestyle='--', linewidth=1.5)
    ax.add_patch(hex_polygon)
    
    for r in np.linspace(0.2, 1.0, 5):
        ax.add_patch(patches.Polygon(vertices * r, closed=True, fill=False, edgecolor=(1, 1, 1, 0.08), linewidth=1))
    
    for v in vertices:
        ax.plot([0, v[0]], [0, v[1]], color=(1, 1, 1, 0.1), linewidth=1)

    for angle, label in zip(angles, labels):
        ax.text(np.cos(angle) * 1.2, np.sin(angle) * 1.2, label, color='white', fontsize=11, ha='center', va='center', fontweight='bold')

    # (1) 전체 곡 배경 (회색)
    ax.scatter(df['hex_x'], df['hex_y'], color='#555555', alpha=0.3, s=15, label='Song Universe')

    # (2) 기준 곡 (흰색 별)
    ax.scatter(base_x, base_y, color='white', s=400, marker='*', edgecolor='black', zorder=6, label='Base Song')

    # (3) Baseline 추천 결과 (파란색)
    base_idx = df[df['song_id'].isin(recs_baseline.keys())].index
    ax.scatter(df.loc[base_idx, 'hex_x'], df.loc[base_idx, 'hex_y'], 
               color='#3498db', s=80, alpha=0.9, zorder=5, label=f'Baseline CBF (Div: {div_baseline:.2f})')
    
    # Baseline 연결선
    for _, row in df.loc[base_idx].iterrows():
        ax.plot([base_x, row['hex_x']], [base_y, row['hex_y']], color='#3498db', alpha=0.4, linewidth=1.5, linestyle='--')

    # (4) Vibe-Sync 추천 결과 (네온 그린)
    vibe_idx = df[df['song_id'].isin(recs_vibe.keys())].index
    ax.scatter(df.loc[vibe_idx, 'hex_x'], df.loc[vibe_idx, 'hex_y'], 
               color='#1ed760', s=100, edgecolor='white', zorder=5, label=f'Vibe-Sync Extreme (Div: {div_vibe:.2f})')
    
    # Vibe-Sync 연결선
    for _, row in df.loc[vibe_idx].iterrows():
        ax.plot([base_x, row['hex_x']], [base_y, row['hex_y']], color='#1ed760', alpha=0.6, linewidth=2)

    # 7. 마무리 및 저장
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), facecolor='#121212', edgecolor='#333333', labelcolor='white', fontsize=12)
    plt.title("Model Comparison: Baseline vs Vibe-Sync", color='white', fontsize=18, fontweight='bold', pad=20)
    
    # 하단 텍스트 코멘트 추가
    plt.figtext(0.5, 0.01, "Baseline CBF clusters tightly around the base song (Low Diversity).\nVibe-Sync breaks boundaries and explores the entire audio space (High Diversity).", 
                ha="center", fontsize=12, color='#b3b3b3')

    plt.tight_layout()
    plt.savefig('evaluation_comparison.png', dpi=300, bbox_inches='tight')
    print("✨ 시각화 및 평가 이미지가 'evaluation_comparison.png'로 성공적으로 저장되었습니다!")

if __name__ == "__main__":
    run_evaluation_and_visualize()