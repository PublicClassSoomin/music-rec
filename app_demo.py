import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import os

# --- 0. 페이지 설정 (와이드 모드) ---
st.set_page_config(page_title="Vibe-Sync AI DJ Console", layout="wide")
st.title("🎧 Vibe-Sync AI DJ MIXING CONSOLE (Interactive Demo)")
st.markdown("---")

# --- 1. 더미 데이터 생성 함수 ---
# (실제 프로젝트에서는 DB에서 불러온 17개 특성을 사용합니다.)
@st.cache_data # 데이터를 캐싱하여 속도 향상
def load_dummy_data(n_songs=200):
    np.random.seed(42)
    # 6개 대표 특성 랜덤 생성 (0~100)
    features = np.random.uniform(0, 100, (n_songs, 6))
    
    data = []
    # 한국인에게 익숙한 더미 가수/노래 리스트
    artists = ['BTS', 'aespa', '뉴진스', '아이브', '데이식스', '성시경', '로제', '지코', '르세라핌', '임영웅']
    titles = ['SWIM', 'Supernova', 'Ditto', 'Love Dive', 'Han Page', '너의 모든 순간', 'APT', '지코곡', 'ANTIFRAGILE', '사랑은 늘 도망가']
    genres = ['K-Pop', 'Dance', 'Ballad', 'Rock', 'R&B']

    # 더미 앨범 이미지 (로컬 파일 사용, 실제로는 URL 사용)
    # 스크립트와 같은 폴더에 dummy_album.png 파일이 있어야 합니다.
    img_path = "dummy_album.png" if os.path.exists("dummy_album.png") else None

    for i in range(n_songs):
        art_idx = np.random.randint(0, len(artists))
        data.append({
            'song_id': f'SN{i:03d}',
            'title': f"{titles[art_idx]} (Ver.{i})", # 노래제목 중복 방지
            'artist': artists[art_idx],
            'genre': np.random.choice(genres),
            'image_url': img_path, # 이미지 경로
            # 6개 오디오 특성
            'f_artist': features[i, 0],
            'f_mood': features[i, 1],
            'f_genre': features[i, 2],
            'f_sim': features[i, 3],
            'f_emo': features[i, 4],
            'f_adv': features[i, 5],
        })
    return pd.DataFrame(data)

# 데이터 로드
df = load_dummy_data()

# --- 2. 육각형 맵 좌표 계산 로직 ---
# 6차원 특징을 2차원 육각형 공간으로 평면 투영
features_cols = ['f_artist', 'f_mood', 'f_genre', 'f_sim', 'f_emo', 'f_adv']
labels = ['Artist', 'Mood', 'Genre', 'Similarity', 'Emotion', 'Adventure']

# 정규화 (0.1 ~ 1.0 가중치로 변환)
scaler = MinMaxScaler(feature_range=(0.1, 1.0))
df_scaled = scaler.fit_transform(df[features_cols])

# 6각 꼭짓점 좌표 (각도 설정)
angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
vertices = np.column_stack((np.cos(angles), np.sin(angles)))

# 6D -> 2D 무게중심 매핑 함수
def to_2d(weights):
    # weights: 한 곡의 6개 특성 가중치 배열
    x = np.sum(weights * vertices[:, 0]) / np.sum(weights)
    y = np.sum(weights * vertices[:, 1]) / np.sum(weights)
    return x, y

# 모든 곡의 2D 좌표 계산 및 저장
coords = np.array([to_2d(w) for w in df_scaled])
df['x'] = coords[:, 0]
df['y'] = coords[:, 1]


# ==========================================
# --- 3. 사이드바: 조절바 (User Input) ---
# ==========================================
with st.sidebar:
    st.header("🎛️ Vibe-Sync Controls")
    st.markdown("조절바를 움직여 AI DJ에게 명령을 내리세요.")
    
    manual_mode = st.checkbox("나만의 Mixing (Manual Mode)", value=True)
    
    params = {}
    # 시각화 이미지의 꼭짓점 순서와 매칭
    labels_map = ['Artist', 'Mood', 'Genre', 'Similarity', 'Emotion', 'Adventure']
    
    for label in labels_map:
        params[label.lower()] = st.slider(f"[ {label} ]", 0, 100, 50, disabled=not manual_mode)

    st.markdown("---")
    st.markdown("by Vibe-Sync Team")


# ==========================================
# --- 4. 메인 화면: 육각형 인터랙티브 지도 ---
# ==========================================

# (1) 타겟 벡터 계산 (유저 입력 반영)
# 입력값을 0.1~1.0 가중치로 변환
target_weights = np.array([params[l.lower()] for l in labels_map])
target_weights_scaled = MinMaxScaler(feature_range=(0.1, 1.0)).fit_transform(target_weights.reshape(-1, 1)).flatten()

# 타겟의 2D 좌표
target_x, target_y = to_2d(target_weights_scaled)

# (2) 거리 계산 (모든 곡 vs 타겟)
distances = np.sqrt((df['x'] - target_x)**2 + (df['y'] - target_y)**2)
df['dist'] = distances
closest_indices = np.argsort(distances)

# (3) 바구니 분류 및 색상 지정
# - 기준 곡 (가장 가까운 1곡) -> 별
base_song = df.iloc[closest_indices[0]]

# - 추천될 3개의 점 (초록색)
comfort_recs = df.iloc[closest_indices[1:4]] # 기준 곡 제외 top 3

# - 근처였지만 추천되지 못한 7개의 빨간색 점 (Top 4~10)
near_misses = df.iloc[closest_indices[4:11]] # 그 다음 7곡

# 색상 상태 컬럼 추가
df['status'] = 'Pool' # 기본값: 회색
df.loc[comfort_recs.index, 'status'] = 'Recommended'
df.loc[near_misses.index, 'status'] = 'Near Miss'


# ==========================================
# --- 5. Plotly 육각형 그래프 그리기 ---
# ==========================================
fig = go.Figure()

# (1) 육각형 배경 그리드
# 외곽선
angles_closed = np.append(angles, angles[0])
vertices_closed = np.column_stack((np.cos(angles_closed), np.sin(angles_closed)))
fig.add_trace(go.Scatter(x=vertices_closed[:, 0], y=vertices_closed[:, 1], mode='lines', line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'), showlegend=False, hoverinfo='skip'))

# 내부 거미줄 그리드
for r in np.linspace(0.2, 1.0, 5):
    fig.add_trace(go.Scatter(
        x=vertices_closed[:, 0] * r, 
        y=vertices_closed[:, 1] * r, 
        mode='lines', 
        line=dict(color='rgba(255,255,255,0.08)', width=1), 
        showlegend=False, 
        hoverinfo='skip'
    ))

# 축 및 라벨
for i, label in enumerate(labels_map):
    # 축 선
    fig.add_trace(go.Scatter(x=[0, vertices[i,0]], y=[0, vertices[i,1]], mode='lines', line=dict(color='rgba(255,255,255,0.1)', width=1), showlegend=False, hoverinfo='skip'))
    # 꼭짓점 라벨
    fig.add_trace(go.Scatter(x=[vertices[i,0] * 1.15], y=[vertices[i,1] * 1.15], mode='text', text=[label], textfont=dict(color='white', size=13, family='Arial Black'), showlegend=False, hoverinfo='skip'))


# (2) 전체 곡들 (배경 회색)
# 클릭 이벤트를 받기 위해 customdata에 노래 정보를 심음
pool_df = df[df['status'] == 'Pool']
fig.add_trace(go.Scatter(
    x=pool_df['x'], y=pool_df['y'],
    mode='markers',
    marker=dict(color='#444444', size=7, opacity=0.4),
    name='Song Pool',
    customdata=np.stack((pool_df['title'], pool_df['artist'], pool_df['genre'], pool_df['song_id']), axis=-1),
    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>"
))

# (3) 빨간색 점들 (Near Miss, Top 4~10)
fig.add_trace(go.Scatter(
    x=near_misses['x'], y=near_misses['y'],
    mode='markers',
    marker=dict(color='#e74c3c', size=13, symbol='circle', line=dict(color='white', width=1.5)), # 수정됨
    name='Top 4~10 (Near Miss)',
    customdata=np.stack((near_misses['title'], near_misses['artist'], near_misses['genre'], near_misses['song_id']), axis=-1),
    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>"
))

# (4) 초록색 점들 (Top-K Recommended, Top 1~3)
fig.add_trace(go.Scatter(
    x=comfort_recs['x'], y=comfort_recs['y'],
    mode='markers',
    marker=dict(color='#1ed760', size=15, symbol='circle', line=dict(color='white', width=2)), # 수정됨
    name='Top-K (Comfort Recs)',
    customdata=np.stack((comfort_recs['title'], comfort_recs['artist'], comfort_recs['genre'], comfort_recs['song_id']), axis=-1),
    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>"
))

# (5) 기준 곡 (흰색 별)
fig.add_trace(go.Scatter(
    x=[base_song['x']], y=[base_song['y']], # ✨ 이 부분이 수정되었습니다!
    mode='markers',
    marker=dict(color='white', size=25, symbol='star', line=dict(color='black', width=1.5)),
    name='Base Song (Seed)',
    customdata=np.stack(([base_song['title']], [base_song['artist']], [base_song['genre']], [base_song['song_id']]), axis=-1),
    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>"
))


# 그래프 레이아웃 세팅
fig.update_layout(
    width=900, height=800,
    xaxis=dict(visible=False, range=[-1.4, 1.4]),
    yaxis=dict(visible=False, range=[-1.4, 1.4]),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(font=dict(color='white'), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    clickmode='event+select' # 클릭 이벤트 활성화
)


# --- 6. 레이아웃 배치 및 클릭 이벤트 처리 ---
col_map, col_info = st.columns([2, 1])

with col_map:
    # Plotly 차트를 Streamlit에 출력하고 클릭 이벤트를 변수에 저장
    # ('plotly_click'은 Plotly 차트 안의 점을 클릭했을 때의 데이터)
    click_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="hex_map")


with col_info:
    st.subheader("🎵 Now Playing / Selected Info")
    
    # 클릭 데이터가 있는지 확인
    # (click_data 구조: {'selection': {'point_indices': [...], 'points': [...]}, ...})
    s = st.session_state.hex_map
    selected_song = None

    if s and 'selection' in s and s['selection']['points']:
        # 클릭한 점의 정보를 customdata에서 추출
        pt = s['selection']['points'][0]
        # customdata 순서: title, artist, genre, song_id
        c_data = pt['customdata'] 
        
        # song_id로 원본 DataFrame에서 노래 검색
        selected_song = df[df['song_id'] == c_data[3]].iloc[0]
        st.success(f"노래 '{c_data[0]}'를 선택했습니다.")
    else:
        # 클릭 전에는 기준 곡(Seed) 정보를 기본으로 보여줌
        selected_song = base_song
        st.info(f"지도의 점을 클릭하면 정보를 볼 수 있습니다. (현재: 기준 곡)")

    # 선택된 노래 정보 표시 카드 디자인 (HTML/CSS)
    if selected_song is not None:
        st.markdown(f"""
            <div style="background-color: #1e1e1e; border: 1px solid #333; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(30, 215, 96, 0.2);">
                <img src="app/static/{selected_song['image_url']}" style="width: 100%; max-width: 250px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #1ed760; box-shadow: 0 0 10px #1ed760;">
                <h2 style="color: white; margin: 0; font-size: 1.5rem;">{selected_song['title']}</h2>
                <p style="color: #1ed760; font-size: 1.2rem; font-weight: bold; margin: 5px 0;">{selected_song['artist']}</p>
                <p style="color: #888; font-size: 0.9rem; margin-top: 10px;">ID: {selected_song['song_id']} | Genre: {selected_song['genre']}</p>
                <div style="display:inline-block; margin-top: 15px; padding: 5px 15px; border-radius: 20px; background-color: {'#1ed760' if selected_song['status']=='Recommended' else '#e74c3c' if selected_song['status']=='Near Miss' else '#444'}; color: white; font-weight: bold;">
                    {selected_song['status']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            # 주의: Streamlit에서 로컬 이미지를 표시하려면 정적 경로 설정이 필요합니다. 
            # 위 HTML의 "app/static/" 부분은 더미 이미지 로드를 위한 임시 방편입니다.

st.markdown("---")
st.caption("Vibe-Sync Recommendation Evaluation System | Interactivity Demo")