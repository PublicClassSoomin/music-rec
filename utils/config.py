import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# DB
DB_PATH = os.getenv("DB_PATH", "./music_rec.db")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-change-this-secret")
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

# 멜론 파이프라인 (data/melon_pipeline.py)
# 차트 페이지: tr.lst50 / lst100 구조 (TOP100, HOT100, 일간, 주간 등)
MELON_CHART_URLS = [
    "https://www.melon.com/chart/index.htm",
    "https://www.melon.com/chart/hot100/index.htm",
    "https://www.melon.com/chart/day/index.htm",
    "https://www.melon.com/chart/week/index.htm",
    "https://www.melon.com/chart/month/index.htm",
]
MELON_MAX_SONGS_PER_CHART = 100

# 장르 음악: gnrCode (한국대중 / 해외POP / 기타 인기장르 …)
MELON_GENRE_CODES = [
    "GN0100",
    "GN0900",
    "GN1500",
]
# 장르당 최대 곡 수 (50곡/페이지, song_listPaging.htm)
MELON_MAX_SONGS_PER_GENRE = 300

# 플레이리스트·기타: 곡 리스트 HTML이 차트와 동일하거나 장르 테이블과 동일한 페이지 URL
# (멜론이 JS 전용인 페이지는 실패할 수 있음)
MELON_EXTRA_SONG_PAGE_URLS: list[str] = []

# 데이터 수집 설정
COLLECT_QUERIES = [
    "lo-fi hip hop music",
    "jazz chill music",
    "classical music focus",
    "k-pop playlist",
    "hiphop beats",
    "indie pop music",
    "electronic music chill",
    "acoustic guitar music",
]
MAX_RESULTS_PER_QUERY = 30      # 검색어당 수집할 곡 수
MAX_AUDIO_DURATION    = 30      # librosa 분석 시 앞 몇 초만 사용

# 추천 설정
TOP_K              = 10
COLD_START_LIMIT   = 3          # 인터랙션 N개 미만 → Cold Start

# 평가 설정
EVAL_K_LIST = [5, 10, 20]

# LangGraph LLM 설정
LLM_MODEL = "gemini-1.5-flash"
