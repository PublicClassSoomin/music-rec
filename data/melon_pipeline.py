"""
Melon 다중 소스 수집 파이프라인 (단독 실행)

수집:
  - 멜론 차트 여러 종 (TOP100, HOT100, 일간, 주간 …)
  - (선택) 추가 URL — HTML이 차트형 또는 wrap_song_info 테이블일 때
  - (선택) 멜론 곡 상세 페이지에서 가사 — MELON_FETCH_LYRICS (utils/config)

흐름:
  1. 위 소스에서 (title, artist, melon_song_id) 목록 합친 뒤 중복 제거
  2. yt-dlp로 YouTube 검색 → youtube_url 매칭
  3. 멜론 상세에서 가사 수집(옵션) → DB songs.lyrics
  4. yt-dlp로 오디오 다운로드 (ffmpeg 필요)
  5. librosa로 오디오 특성 추출
  6. songs + audio_features DB 저장 후 오디오 파일 즉시 삭제

저작권·이용약관: 가사는 멜론 서비스 약관 및 저작권법을 따를 책임이 사용자(운영자)에게 있습니다.

실행:
  python data/melon_pipeline.py

설정:
  utils/config.py — MELON_CHART_URLS, MELON_EXTRA_SONG_PAGE_URLS, MELON_FETCH_LYRICS 등

사전 조건:
  brew install ffmpeg
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import tempfile
import shutil
import re
import requests
import numpy as np
import librosa
import yt_dlp
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm

from data.database import init_db, upsert_song, upsert_audio_features, get_all_songs

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.melon.com",
}


class _YtdlpQuietLogger:
    def debug(self, msg):
        return

    def warning(self, msg):
        return

    def error(self, msg):
        return


_ytdlp_search_error_samples = 0
_ytdlp_download_error_samples = 0
_YTDLP_ERROR_LOG_MAX = 3


def _ytdlp_cookie_opts() -> dict:
    """YouTube 봇 차단 완화: 쿠키 파일(우선) 또는 브라우저 프로필."""
    from pathlib import Path

    from utils.config import YOUTUBE_COOKIES_FILE, YOUTUBE_COOKIES_FROM_BROWSER

    opts: dict = {}
    cf = (YOUTUBE_COOKIES_FILE or "").strip()
    if cf:
        p = Path(cf).expanduser()
        if p.is_file():
            opts["cookiefile"] = str(p)
            return opts
    cfb = (YOUTUBE_COOKIES_FROM_BROWSER or "").strip()
    if cfb:
        parts = [x.strip() for x in cfb.split(":") if x.strip()]
        if parts:
            opts["cookiesfrombrowser"] = tuple(parts) if len(parts) > 1 else (parts[0],)
    return opts


def _melon_song_id_from_href(href: str | None) -> str | None:
    """javascript:melon.play.playSong('…', 601555642) 또는 songId= 쿼리."""
    if not href:
        return None
    m = re.search(r"playSong\(\s*'[^']*'\s*,\s*(\d+)\s*\)", href)
    if m:
        return m.group(1)
    m = re.search(r"songId=(\d+)", href, re.I)
    return m.group(1) if m else None


def fetch_melon_lyrics(melon_song_id: str) -> str | None:
    """
    멜론 곡 상세 HTML에서 가사 영역 파싱.
    '가사 준비중' 이거나 파싱 실패 시 None.
    """
    if not melon_song_id or not str(melon_song_id).isdigit():
        return None
    url = f"https://www.melon.com/song/detail.htm?songId={melon_song_id}"
    html = _fetch(url, referer="https://www.melon.com/chart/index.htm")
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    box = soup.select_one("div.lyric#d_video_summary") or soup.select_one("#d_video_summary")
    if box:
        text = box.get_text("\n", strip=True)
        if len(text) > 40 and "가사 준비중" not in text:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return "\n".join(lines)
    return None


def _fetch(url: str, referer: str | None = None) -> str | None:
    h = {**HEADERS}
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
    songs: list[dict] = []
    rows = soup.select("tr.lst50, tr.lst100")
    for row in rows[:max_results]:
        try:
            rank_el = row.select_one(".rank")
            title_el = row.select_one(".rank01 a") or row.select_one(".rank01 span a")
            artist_el = row.select_one(".rank02 a")
            if not (rank_el and title_el and artist_el):
                continue
            href = title_el.get("href") or ""
            songs.append({
                "rank": int(rank_el.text.strip()),
                "title": title_el.text.strip(),
                "artist": artist_el.text.strip(),
                "melon_song_id": _melon_song_id_from_href(href),
            })
        except Exception:
            continue
    return songs


def scrape_chart_url(url: str, max_results: int) -> list[dict]:
    html = _fetch(url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    return _parse_tracks_from_chart_soup(soup, max_results)


def _parse_tracks_from_genre_html(html: str, max_results: int) -> list[dict]:
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
            href = t_el.get("href") or ""
            songs.append({
                "title": title,
                "artist": artist,
                "melon_song_id": _melon_song_id_from_href(href),
            })
        if len(songs) >= max_results:
            break
    return songs


def scrape_extra_page(url: str, max_results: int = 500) -> list[dict]:
    """
    플레이리스트 등 추가 URL.
    차트(tr.lst50) 또는 wrap_song_info + rank01/rank02 테이블 HTML이면 파싱.
    """
    html = _fetch(url, referer="https://www.melon.com")
    if not html:
        return []
    if "lst50" in html or "lst100" in html:
        soup = BeautifulSoup(html, "html.parser")
        rows = _parse_tracks_from_chart_soup(soup, max_results)
        return [
            {
                "title": r["title"],
                "artist": r["artist"],
                "melon_song_id": r.get("melon_song_id"),
            }
            for r in rows
        ]
    if "wrap_song_info" in html and "rank01" in html:
        return _parse_tracks_from_genre_html(html, max_results)
    print(f"  ⚠️ 파싱 불가 (JS 렌더 전용·오류 페이지일 수 있음): {url[:80]}…")
    return []


def dedupe_tracks(tracks: list[dict]) -> list[dict]:
    best: dict[tuple[str, str], dict] = {}
    for t in tracks:
        title = (t.get("title") or "").strip()
        artist = (t.get("artist") or "").strip()
        if not title:
            continue
        key = (title.lower(), artist.lower())
        cur = best.get(key)
        if cur is None:
            best[key] = {
                "title": title,
                "artist": artist,
                "melon_song_id": t.get("melon_song_id"),
            }
            continue
        if not cur.get("melon_song_id") and t.get("melon_song_id"):
            best[key] = {
                "title": title,
                "artist": artist,
                "melon_song_id": t.get("melon_song_id"),
            }
    return list(best.values())


def gather_all_melon_tracks() -> list[dict]:
    from utils.config import (
        MELON_CHART_URLS,
        MELON_MAX_SONGS_PER_CHART,
        MELON_EXTRA_SONG_PAGE_URLS,
        MELON_FETCH_LYRICS,
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
    print(f"\n📋 소스 합계 {len(raw)}행 → 중복 제거 후 {len(deduped)}곡")
    if MELON_FETCH_LYRICS:
        with_mid = sum(1 for x in deduped if x.get("melon_song_id"))
        print(f"   (멜론 songId 확보: {with_mid}곡 → 가사 수집 시도 가능)\n")
    else:
        print("   (MELON_FETCH_LYRICS=off → 가사 수집 안 함)\n")
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
    global _ytdlp_search_error_samples
    query = f"{title} {artist} official"
    # 플랫 추출: 검색 결과마다 전체 포맷을 풀지 않음 → "Requested format is not available" 회피·속도↑
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "default_search": "ytsearch1",
        "skip_download": True,
        "logger": _YtdlpQuietLogger(),
        **_ytdlp_cookie_opts(),
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
                url = item.get("url") or ""
                m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
                vid = m.group(1) if m else None
            if not vid:
                return None
            categories = item.get("categories") or []
            genre = categories[0] if categories else "Music"
            thumb = (item.get("thumbnail") or "").strip()
            if not thumb:
                thumb = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
            return {
                "song_id": vid,
                "youtube_url": f"https://www.youtube.com/watch?v={vid}",
                "thumbnail_url": thumb,
                "genre": genre,
            }
    except Exception as e:
        if _ytdlp_search_error_samples < _YTDLP_ERROR_LOG_MAX:
            _ytdlp_search_error_samples += 1
            print(f"[YouTube 검색] 실패 예시 ({_ytdlp_search_error_samples}/{_YTDLP_ERROR_LOG_MAX}): {e}")
        return None


def _format_unavailable_msg(msg: str) -> bool:
    return (
        "Requested format is not available" in msg
        or "No video formats" in msg
        or "format is not available" in msg.lower()
    )


def download_audio(youtube_url: str, output_path: str) -> str | None:
    """
    오디오만 받아 MP3로 변환. YouTube는 클라이언트·지역에 따라 DASH만 있거나
    목록이 비는 경우가 있어, 포맷 체인 + player_client 재시도로 완화한다.
    로그인 쿠키가 있어도 일부 응답에서는 포맷 목록이 비는 경우가 있어,
    그때는 쿠키 없이 한 번 더 시도한다(봇 차단이면 그때는 실패할 수 있음).
    """
    global _ytdlp_download_error_samples
    # m4a/webm DASH 오디오 → 순수 오디오 → 영상+오디오 단일 스트림까지 후보
    fmt = (
        "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/ba/"
        "b[vcodec!=none][acodec!=none]/b/best/worst"
    )
    postprocessors = [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "mp3",
        "preferredquality": "128",
    }]
    core = {
        "format": fmt,
        "outtmpl": output_path,
        "postprocessors": postprocessors,
        "quiet": True,
        "no_warnings": True,
        "logger": _YtdlpQuietLogger(),
    }
    cookie_opts = _ytdlp_cookie_opts()
    # 쿠키 O: 기본 → android → ios → tv_embedded
    extractor_tries: list[dict] = [
        {},
        {"extractor_args": {"youtube": {"player_client": ["android"]}}},
        {"extractor_args": {"youtube": {"player_client": ["ios"]}}},
        {"extractor_args": {"youtube": {"player_client": ["tv_embedded"]}}},
    ]

    def try_download(extra: dict, merge_cookie: bool) -> tuple[bool, Exception | None]:
        ydl_opts = {**core, **(cookie_opts if merge_cookie else {}), **extra}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            mp3_path = output_path + ".mp3"
            if os.path.exists(mp3_path):
                return True, None
        except Exception as e:
            return False, e
        return False, None

    last_err: Exception | None = None
    for extra in extractor_tries:
        ok, err = try_download(extra, merge_cookie=True)
        if ok:
            return output_path + ".mp3"
        if err is not None:
            last_err = err
            msg = str(err)
            if "Sign in to confirm your age" in msg:
                print(f"[Download] 연령 제한 영상 스킵: {youtube_url}")
                return None
            if not _format_unavailable_msg(msg):
                break

    # 쿠키가 있는데만 "포맷 없음"이면: 만료·불일치 쿠키로 목록이 비는 경우가 있어 무쿠키 1회
    if (
        cookie_opts
        and last_err is not None
        and _format_unavailable_msg(str(last_err))
    ):
        ok, err = try_download(
            {"extractor_args": {"youtube": {"player_client": ["android"]}}},
            merge_cookie=False,
        )
        if ok:
            return output_path + ".mp3"
        if err is not None:
            last_err = err

    e = last_err
    if e is None:
        return None
    if _ytdlp_download_error_samples < _YTDLP_ERROR_LOG_MAX:
        _ytdlp_download_error_samples += 1
        print(f"[Download] 실패 예시 ({_ytdlp_download_error_samples}/{_YTDLP_ERROR_LOG_MAX}): {youtube_url}\n  → {e}")
    else:
        print(f"[Download] 실패: {youtube_url}")
    return None


def extract_features(audio_path: str, duration: float | None = None) -> dict | None:
    """
    librosa로 MFCC·BPM·energy 등 추출.
    - MAX_AUDIO_DURATION > 0 (기본 30): 앞 N초만 로드
    - MAX_AUDIO_DURATION <= 0 또는 duration=0: 파일 전체 로드 (긴 곡은 메모리·시간 증가)
    """
    from utils.config import MAX_AUDIO_DURATION

    if duration is not None:
        use_full = duration <= 0
        clip_sec = None if use_full else float(duration)
    else:
        use_full = MAX_AUDIO_DURATION <= 0
        clip_sec = None if use_full else float(MAX_AUDIO_DURATION)

    try:
        if clip_sec is None:
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
        else:
            y, sr = librosa.load(audio_path, sr=22050, duration=clip_sec, mono=True)
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

    from pathlib import Path

    from utils.config import (
        MAX_AUDIO_DURATION,
        MELON_FETCH_LYRICS,
        MELON_LYRIC_DELAY_SEC,
        YOUTUBE_COOKIES_FILE,
    )

    if MAX_AUDIO_DURATION <= 0:
        print("[파이프라인] 오디오 특성: 전체 트랙 librosa 분석 (MAX_AUDIO_DURATION<=0)")
    else:
        print(f"[파이프라인] 오디오 특성: 앞 {MAX_AUDIO_DURATION}초만 librosa 분석")

    cf_env = (YOUTUBE_COOKIES_FILE or "").strip()
    if cf_env and not Path(cf_env).expanduser().is_file():
        print(f"[yt-dlp] 경고: YOUTUBE_COOKIES_FILE에 파일 없음 → 무시됨 ({cf_env})")

    co = _ytdlp_cookie_opts()
    if co.get("cookiefile"):
        print(f"[yt-dlp] 쿠키 파일: {co['cookiefile']}")
    elif co.get("cookiesfrombrowser"):
        print(f"[yt-dlp] 브라우저 쿠키: {co['cookiesfrombrowser']}")
    else:
        print(
            "[yt-dlp] 쿠키 미설정 — YouTube가 봇 확인으로 막으면 "
            "YOUTUBE_COOKIES_FILE 또는 YOUTUBE_COOKIES_FROM_BROWSER(.env)를 설정하세요."
        )

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
        melon_id = song.get("melon_song_id")

        lyrics = None
        if MELON_FETCH_LYRICS and melon_id:
            lyrics = fetch_melon_lyrics(melon_id)
            time.sleep(MELON_LYRIC_DELAY_SEC)

        upsert_song({
            **yt_data,
            "title": title,
            "artist": artist,
            "duration": 0,
            "collected_at": datetime.now().isoformat(),
            "melon_song_id": melon_id,
            "lyrics": lyrics,
        })

        if sid in has_features:
            skip += 1
            continue

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
