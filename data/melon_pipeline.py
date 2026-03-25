"""
Melon 다중 소스 수집 파이프라인 (단독 실행)

수집:
  - 멜론 차트 여러 종 (TOP100, HOT100, 일간, 주간 …)
  - 장르별 곡 목록 (song_listPaging, 페이지당 50곡)
  - (선택) 추가 URL — HTML이 차트형·장르형 테이블일 때

흐름:
  1. 위 소스에서 (title, artist) 목록 합친 뒤 중복 제거
  2. yt-dlp로 YouTube 검색 → youtube_url 매칭
  3. yt-dlp로 오디오 다운로드 (ffmpeg 필요)
  4. librosa로 오디오 특성 추출
  5. songs + audio_features DB 저장 후 오디오 파일 즉시 삭제

실행:
  python data/melon_pipeline.py

설정:
  utils/config.py — MELON_CHART_URLS, MELON_GENRE_CODES, MELON_EXTRA_SONG_PAGE_URLS 등

사전 조건:
  brew install ffmpeg
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import tempfile
import shutil
import requests
import numpy as np
import librosa
import yt_dlp
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm

from data.database import init_db, upsert_song, upsert_audio_features, get_all_songs

# 크롤링 헤더 설정
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.melon.com",
}


class _YtdlpQuietLogger:
    '''
        유튜브 다운로드 라이브러리(yt-dlp)가 출력하는 불필요한 로그(debug, warning, error)를 무시하여 터미널을 깔끔하게 유지하는 로거 클래스
    '''
    def debug(self, msg):
        return

    def warning(self, msg):
        return

    def error(self, msg):
        return

def _fetch(url: str, referer: str | None = None) -> str | None:
    h = {**HEADERS}
    '''
        멜론 서버에 봇(Bot)이 아닌 척 위장(Headers)하여 접속하고, 해당 URL의 HTML 텍스트를 반환하는 함수
    '''
    if referer:
        h["Referer"] = referer
    try:
        res = requests.get(url, headers=h, timeout=20)
        res.raise_for_status()
        return res.text
    except Exception as e:
        print(f"  ⚠️ 요청 실패: {url}\n     {e}")
        return None


def _parse_tracks_from_chart_soup(soup: BeautifulSoup, max_results: int) -> list[dict]:
    '''
        차트 HTML에서 곡 제목과 가수명을 파싱하여 노래들을 리스트로 반환하는 함수
        [범위 : HTML 1 파일]
    '''
    songs: list[dict] = []
    rows = soup.select("tr.lst50, tr.lst100")
    for row in rows[:max_results]:
        try:
            rank_el = row.select_one(".rank")
            title_el = row.select_one(".rank01 span a")
            artist_el = row.select_one(".rank02 a")
            if not (rank_el and title_el and artist_el):
                continue
            songs.append({
                "rank": int(rank_el.text.strip()),
                "title": title_el.text.strip(),
                "artist": artist_el.text.strip(),
            })
        except Exception:
            continue
    return songs


def scrape_chart_url(url: str, max_results: int) -> list[dict]:
    '''
        URL에서 max_results 만큼 노래들을 수집하여 리스트로 반환하는 함수
        [범위 : HTML 1 파일]
    '''
    html = _fetch(url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    return _parse_tracks_from_chart_soup(soup, max_results)


def _parse_tracks_from_genre_html(html: str, max_results: int) -> list[dict]:
    '''
        장르 HTML에서 곡 제목, 가수명을 파싱하여 노래들을 리스트로 반환하는 함수
        [범위 : HTML 1 파일]
    '''
    # BeautifulSoup으로 HTML을 요리하기 좋게 만듭니다.
    soup = BeautifulSoup(html, "html.parser") 
    songs: list[dict] = []
    for wrap in soup.select("div.wrap_song_info"):
        t_el = wrap.select_one(".rank01 a")
        a_el = wrap.select_one(".rank02 a")
        if not t_el or not a_el:
            continue
        title = t_el.get_text(strip=True)
        artist = a_el.get_text(strip=True)
        if title:
            songs.append({"title": title, "artist": artist})
        if len(songs) >= max_results:
            break
    return songs


def scrape_genre_code(gnr_code: str, max_total: int) -> list[dict]:
    """
        genre/song_listPaging.htm — pageIndex 1, 51, 101, … (50곡/페이지).
        멜론 장르 페이지에서 페이지를 넘기며 곡을 수집하는 함수
        [범위 : HTML 여러 파일]
    """
    ref = f"https://www.melon.com/genre/song_list.htm?gnrCode={gnr_code}"
    all_songs: list[dict] = []
    page_index = 1

    while len(all_songs) < max_total:
        url = (
            "https://www.melon.com/genre/song_listPaging.htm"
            f"?gnrCode={gnr_code}&pageIndex={page_index}"
        )
        html = _fetch(url, referer=ref)
        if not html:
            break
        chunk = _parse_tracks_from_genre_html(html, max_total - len(all_songs))
        if not chunk:
            break
        all_songs.extend(chunk)
        if len(chunk) < 50:
            break
        page_index += 50
        time.sleep(0.25)

    return all_songs[:max_total]


def scrape_extra_page(url: str, max_results: int = 500) -> list[dict]:
    """
    플레이리스트 등 추가 URL.
    차트(tr.lst50) 또는 장르형(wrap_song_info + rank01/rank02) HTML이면 파싱.
    [범위 : HTML 1 파일]
    """
    html = _fetch(url, referer="https://www.melon.com")
    if not html:
        return []
    
    # 리스트의 각행을 제목과 작곡가로 하여 반환
    if "lst50" in html or "lst100" in html:
        soup = BeautifulSoup(html, "html.parser")
        rows = _parse_tracks_from_chart_soup(soup, max_results)
        return [{"title": r["title"], "artist": r["artist"]} for r in rows]
    

    if "wrap_song_info" in html and "rank01" in html:
        return _parse_tracks_from_genre_html(html, max_results)
    print(f"  ⚠️ 파싱 불가 (JS 렌더 전용·오류 페이지일 수 있음): {url[:80]}…")
    return []


def dedupe_tracks(tracks: list[dict]) -> list[dict]:
    '''
        수집된 곡 리스트에서 대소문자를 구분하지 않고 '제목+가수'를 기준으로 중복된 곡을 제거하고 중복되지 않은 제목과 가수를 같은 행으로 하여 리스트를 반환
    '''
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for t in tracks:
        title = (t.get("title") or "").strip()
        artist = (t.get("artist") or "").strip()
        if not title:
            continue
        key = (title.lower(), artist.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": title, "artist": artist})
    return out


def gather_all_melon_tracks() -> list[dict]:
    '''
        설정(config.py)에 등록된 모든 차트와 장르 소스를 순회하며 곡을 긁어모은 뒤, 중복을 제거하여 최종 크롤링 리스트를 반환하는 총괄 매니저 함수
    '''
    from utils.config import (
        MELON_CHART_URLS,
        MELON_MAX_SONGS_PER_CHART,
        MELON_GENRE_CODES,
        MELON_MAX_SONGS_PER_GENRE,
        MELON_EXTRA_SONG_PAGE_URLS,
    )

    raw: list[dict] = []

    print("── 멜론 차트 ──")
    for u in MELON_CHART_URLS:
        short = u.split("/")[-2] if "/" in u else u
        print(f"  ▶ {short}")
        part = scrape_chart_url(u, MELON_MAX_SONGS_PER_CHART)
        print(f"     {len(part)}곡")
        raw.extend(part)
        time.sleep(0.35)

    print("\n── 멜론 장르 (gnrCode) ──")
    for code in MELON_GENRE_CODES:
        print(f"  ▶ {code}")
        part = scrape_genre_code(code, MELON_MAX_SONGS_PER_GENRE)
        print(f"     {len(part)}곡")
        raw.extend(part)
        time.sleep(0.35)

    if MELON_EXTRA_SONG_PAGE_URLS:
        print("\n── 추가 URL (플레이리스트 등) ──")
        for u in MELON_EXTRA_SONG_PAGE_URLS:
            u = u.strip()
            if not u:
                continue
            print(f"  ▶ {u[:70]}{'…' if len(u) > 70 else ''}")
            part = scrape_extra_page(u, 500)
            print(f"     {len(part)}곡")
            raw.extend(part)
            time.sleep(0.35)

    deduped = dedupe_tracks(raw)
    print(f"\n📋 소스 합계 {len(raw)}행 → 중복 제거 후 {len(deduped)}곡\n")
    return deduped


def _check_ffmpeg() -> bool:
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    if ffmpeg_ok and ffprobe_ok:
        return True

    print("\n[환경 오류] ffmpeg/ffprobe를 찾을 수 없습니다.")
    print("macOS: brew install ffmpeg")
    print("설치 후 새 터미널에서 다시 실행하세요: python data/melon_pipeline.py\n")
    return False


def search_youtube_ytdlp(title: str, artist: str) -> dict | None:
    query = f"{title} {artist} official"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "default_search": "ytsearch1",
        "skip_download": True,
        "logger": _YtdlpQuietLogger(),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            res = ydl.extract_info(f"ytsearch1:{query}", download=False)
            entries = res.get("entries", [])
            if not entries:
                return None
            item = entries[0]
            vid = item.get("id")
            if not vid:
                return None
            categories = item.get("categories") or []
            genre = categories[0] if categories else "Music"
            return {
                "song_id": vid,
                "youtube_url": f"https://www.youtube.com/watch?v={vid}",
                "thumbnail_url": item.get("thumbnail", ""),
                "genre": genre,
            }
    except Exception:
        return None


def download_audio(youtube_url: str, output_path: str) -> str | None:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
        "no_warnings": True,
        "logger": _YtdlpQuietLogger(),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        mp3_path = output_path + ".mp3"
        return mp3_path if os.path.exists(mp3_path) else None
    except Exception as e:
        msg = str(e)
        if "Sign in to confirm your age" in msg:
            print(f"[Download] 연령 제한 영상 스킵: {youtube_url}")
        else:
            print(f"[Download] 실패: {youtube_url}")
        return None


def extract_features(audio_path: str, duration: float = 30.0) -> dict | None:
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=duration, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_arr = np.asarray(tempo).reshape(-1)
        bpm = float(tempo_arr[0]) if tempo_arr.size > 0 else 0.0
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        brightness = float(np.mean(centroid))
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))
        return {
            **{f"mfcc_{i+1}": float(mfcc_means[i]) for i in range(13)},
            "bpm": bpm,
            "energy": energy,
            "spectral_centroid": brightness,
            "zcr": zcr_mean,
        }
    except Exception as e:
        print(f"[librosa] 분석 실패: {audio_path} → {e}")
        return None


def run():
    if not _check_ffmpeg():
        return

    init_db()

    existing_df = get_all_songs()
    has_features = (
        set(existing_df.dropna(subset=["bpm"])["song_id"].tolist())
        if not existing_df.empty
        else set()
    )
    print(f"[DB] 기존 오디오 특성 보유 곡: {len(has_features)}곡")

    melon_songs = gather_all_melon_tracks()
    if not melon_songs:
        print("❌ 멜론에서 곡 목록을 가져오지 못했습니다. 네트워크·URL을 확인하세요.")
        return

    print(f"🎵 {len(melon_songs)}곡 → YouTube 매칭 + 다운로드 + 특성 추출 시작\n")

    success = 0
    skip = 0
    fail = 0

    for song in tqdm(melon_songs, desc="Melon pipeline"):
        title = song["title"]
        artist = song["artist"]

        yt_data = search_youtube_ytdlp(title, artist)
        if not yt_data:
            fail += 1
            continue

        sid = yt_data["song_id"]

        if sid in has_features:
            skip += 1
            continue

        upsert_song({
            **yt_data,
            "title": title,
            "artist": artist,
            "duration": 0,
            "collected_at": datetime.now().isoformat(),
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, sid)
            mp3_path = download_audio(yt_data["youtube_url"], output_path)

            if mp3_path:
                features = extract_features(mp3_path)
                if features:
                    upsert_audio_features({"song_id": sid, **features})
                    has_features.add(sid)
                    success += 1
                else:
                    fail += 1
            else:
                fail += 1

        time.sleep(0.3)

    df = get_all_songs()
    print(f"\n{'─'*40}")
    print(f"✅ 특성 추출 성공: {success}곡")
    print(f"⏭️  기존 보유 스킵: {skip}곡")
    print(f"❌ 실패: {fail}곡")
    print(f"📦 DB 총 보유: {len(df)}곡")
    print(f"{'─'*40}")


if __name__ == "__main__":
    run()
