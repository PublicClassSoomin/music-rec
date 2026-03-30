import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

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

# 멜론 곡 상세 HTML에서 가사 수집 (약관·저작권은 서비스 운영 시 직접 확인)
MELON_FETCH_LYRICS = os.getenv("MELON_FETCH_LYRICS", "true").lower() in (
    "1",
    "true",
    "yes",
)
MELON_LYRIC_DELAY_SEC = float(os.getenv("MELON_LYRIC_DELAY_SEC", "0.45"))
LYRIC_EMBED_MAX_CHARS = int(os.getenv("LYRIC_EMBED_MAX_CHARS", "1500"))

# yt-dlp (멜론 파이프라인 YouTube 검색·다운로드)
# YouTube가 "봇 확인"을 요구하면 둘 중 하나 필요: https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp
# 파일 우선. 없으면 브라우저 프로필에서 읽기 (로컬에서만 권장).
YOUTUBE_COOKIES_FILE = os.getenv("YOUTUBE_COOKIES_FILE", "").strip()
# 예: chrome / firefox / safari — 또는 chrome:Default 처럼 프로필 지정
YOUTUBE_COOKIES_FROM_BROWSER = os.getenv("YOUTUBE_COOKIES_FROM_BROWSER", "").strip()

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
# 멜론 파이프라인 librosa 분석: 앞 N초만 (기본 30). N<=0 이면 파일 전체 로드 후 분석 (RAM·시간 증가)
MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "30"))

# 추천 설정
TOP_K              = 10
COLD_START_LIMIT   = 3          # 인터랙션 N개 미만 → Cold Start

# 평가 설정
EVAL_K_LIST = [5, 10, 20]

# LangGraph LLM 설정 (질의 보강 + 후보 중 top-3 선별, GEMINI_API_KEY 없으면 생략)
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
# false면 확장·top3 선별 모두 규칙/임베딩만 사용 (키가 있어도 LLM top3 안 씀)
SCENARIO_LLM_TOP3 = os.getenv("SCENARIO_LLM_TOP3", "true").lower() in (
    "1",
    "true",
    "yes",
)
SCENARIO_LLM_CANDIDATES = int(os.getenv("SCENARIO_LLM_CANDIDATES", "18"))

# 시나리오 검색 텍스트 임베딩 (경량 다국어 기본값, 한국어 질의에 적합)
# 더 작은 모델: 영어만 쓸 때 sentence-transformers/all-MiniLM-L6-v2 (한국어 약함)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
