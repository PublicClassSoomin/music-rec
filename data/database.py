import sqlite3
import hashlib
import hmac
import bcrypt
import pandas as pd
from datetime import datetime
from utils.config import DB_PATH


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """DB 테이블 초기화 (최초 1회)"""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS songs (
            song_id         TEXT PRIMARY KEY,
            title           TEXT NOT NULL,
            artist          TEXT,
            youtube_url     TEXT,
            thumbnail_url   TEXT,
            genre           TEXT,
            duration        INTEGER DEFAULT 0,
            collected_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS audio_features (
            song_id             TEXT PRIMARY KEY,
            mfcc_1              REAL, mfcc_2  REAL, mfcc_3  REAL,
            mfcc_4              REAL, mfcc_5  REAL, mfcc_6  REAL,
            mfcc_7              REAL, mfcc_8  REAL, mfcc_9  REAL,
            mfcc_10             REAL, mfcc_11 REAL, mfcc_12 REAL,
            mfcc_13             REAL,
            bpm                 REAL,
            energy              REAL,
            spectral_centroid   REAL,
            zcr                 REAL,
            FOREIGN KEY (song_id) REFERENCES songs(song_id)
        );

        CREATE TABLE IF NOT EXISTS users (
            user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            created_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS interactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            song_id         TEXT NOT NULL,
            action          TEXT NOT NULL,
            play_seconds    INTEGER DEFAULT 0,
            timestamp       TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (song_id) REFERENCES songs(song_id)
        );
    """)
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "password_hash" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")

    song_cols = [r["name"] for r in conn.execute("PRAGMA table_info(songs)").fetchall()]
    if "lyrics" not in song_cols:
        conn.execute("ALTER TABLE songs ADD COLUMN lyrics TEXT")
    if "melon_song_id" not in song_cols:
        conn.execute("ALTER TABLE songs ADD COLUMN melon_song_id TEXT")

    conn.commit()
    conn.close()
    print(f"[DB] 초기화 완료: {DB_PATH}")


# ── Songs ─────────────────────────────────────────────────

def upsert_song(song: dict):
    conn = get_conn()
    payload = {
        "song_id": song["song_id"],
        "title": song["title"],
        "artist": song.get("artist"),
        "youtube_url": song.get("youtube_url"),
        "thumbnail_url": song.get("thumbnail_url"),
        "genre": song.get("genre"),
        "duration": song.get("duration", 0),
        "collected_at": song.get("collected_at"),
        "lyrics": song.get("lyrics"),
        "melon_song_id": song.get("melon_song_id"),
    }
    conn.execute("""
        INSERT INTO songs (song_id, title, artist, youtube_url,
                           thumbnail_url, genre, duration, collected_at,
                           lyrics, melon_song_id)
        VALUES (:song_id, :title, :artist, :youtube_url,
                :thumbnail_url, :genre, :duration, :collected_at,
                :lyrics, :melon_song_id)
        ON CONFLICT(song_id) DO UPDATE SET
            title           = excluded.title,
            artist          = excluded.artist,
            youtube_url     = excluded.youtube_url,
            thumbnail_url   = COALESCE(excluded.thumbnail_url, songs.thumbnail_url),
            genre           = COALESCE(excluded.genre, songs.genre),
            duration        = CASE WHEN excluded.duration IS NOT NULL AND excluded.duration > 0
                                   THEN excluded.duration ELSE songs.duration END,
            collected_at    = excluded.collected_at,
            melon_song_id   = COALESCE(excluded.melon_song_id, songs.melon_song_id),
            lyrics          = CASE WHEN excluded.lyrics IS NOT NULL
                                        AND length(trim(excluded.lyrics)) > 15
                                   THEN excluded.lyrics ELSE songs.lyrics END
    """, payload)
    conn.commit()
    conn.close()


def upsert_audio_features(features: dict):
    conn = get_conn()
    cols = ", ".join(features.keys())
    vals = ", ".join(f":{k}" for k in features.keys())
    conn.execute(f"""
        INSERT INTO audio_features ({cols}) VALUES ({vals})
        ON CONFLICT(song_id) DO UPDATE SET
            bpm = excluded.bpm,
            energy = excluded.energy
    """, features)
    conn.commit()
    conn.close()


def get_all_songs() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("""
        SELECT s.*, af.mfcc_1, af.mfcc_2, af.mfcc_3, af.mfcc_4,
               af.mfcc_5, af.mfcc_6, af.mfcc_7, af.mfcc_8, af.mfcc_9,
               af.mfcc_10, af.mfcc_11, af.mfcc_12, af.mfcc_13,
               af.bpm, af.energy, af.spectral_centroid, af.zcr
        FROM songs s
        LEFT JOIN audio_features af ON s.song_id = af.song_id
    """, conn)
    conn.close()
    return df


def get_song(song_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM songs WHERE song_id = ?", (song_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Users ─────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(username: str, password: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT user_id FROM users WHERE username = ?", (username,)
    ).fetchone()
    if row:
        conn.close()
        return False

    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    conn.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, pw_hash, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return True


def authenticate_user(username: str, password: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT password_hash FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    if not row:
        return False

    stored = row["password_hash"] or ""
    if not stored:
        return False

    # 신규 포맷: bcrypt
    if stored.startswith("$2"):
        try:
            return bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
        except Exception:
            return False

    # 레거시 포맷(SHA-256) 호환: 로그인 성공 시 bcrypt로 승격
    ok = hmac.compare_digest(stored, _hash_password(password))
    if ok:
        conn = get_conn()
        new_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (new_hash, username)
        )
        conn.commit()
        conn.close()
    return ok


def get_or_create_user(username: str) -> int:
    conn = get_conn()
    row = conn.execute(
        "SELECT user_id FROM users WHERE username = ?", (username,)
    ).fetchone()
    if row:
        user_id = row["user_id"]
    else:
        cur = conn.execute(
            "INSERT INTO users (username, created_at) VALUES (?, ?)",
            (username, datetime.now().isoformat())
        )
        user_id = cur.lastrowid
        conn.commit()
    conn.close()
    return user_id


# ── Interactions ──────────────────────────────────────────

def log_interaction(user_id: int, song_id: str,
                    action: str, play_seconds: int = 0):
    """
    action: "play" | "like" | "skip" | "unlike"
    """
    conn = get_conn()
    conn.execute("""
        INSERT INTO interactions (user_id, song_id, action, play_seconds, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, song_id, action, play_seconds, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_all_interactions() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM interactions", conn)
    conn.close()
    return df


def get_user_interactions(user_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM interactions WHERE user_id = ? ORDER BY timestamp DESC",
        conn, params=(user_id,)
    )
    conn.close()
    return df


def get_user_liked_song_ids(user_id: int) -> list[str]:
    """
    곡별로 가장 최근 인터랙션이 'like'인 song_id만 반환 (이전에 unlike가 있으면 제외).
    """
    conn = get_conn()
    rows = conn.execute(
        """
        WITH last AS (
            SELECT song_id, action,
                   ROW_NUMBER() OVER (
                       PARTITION BY song_id ORDER BY timestamp DESC, id DESC
                   ) AS rn
            FROM interactions
            WHERE user_id = ? AND action IN ('like', 'unlike')
        )
        SELECT song_id FROM last WHERE rn = 1 AND action = 'like'
        ORDER BY song_id
        """,
        (user_id,),
    ).fetchall()
    conn.close()
    return [r["song_id"] for r in rows]


def get_user_liked_songs(user_id: int) -> list[dict]:
    """좋아요한 곡 메타데이터(songs 테이블). 없는 곡은 건너뜀."""
    out: list[dict] = []
    for sid in get_user_liked_song_ids(user_id):
        row = get_song(sid)
        if row:
            out.append(row)
    return out
