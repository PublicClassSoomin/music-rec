from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from embeddings.text_embedder import encode_texts
from utils.config import LYRIC_EMBED_MAX_CHARS

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
META_NAME = "song_text_emb_meta.json"
IDS_NAME = "song_text_emb_ids.npy"
MAT_NAME = "song_text_emb_matrix.npy"


def _fingerprint(song_ids: list[str], texts: list[str]) -> str:
    h = hashlib.sha256()
    for sid, t in zip(song_ids, texts):
        h.update(sid.encode("utf-8", errors="replace"))
        h.update(b"\0")
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()


def _row_text(row: pd.Series) -> str:
    parts = [
        str(row.get("title") or ""),
        str(row.get("artist") or ""),
        str(row.get("genre") or ""),
    ]
    ly = row.get("lyrics")
    if ly is not None and str(ly).strip():
        chunk = " ".join(str(ly).split())
        if len(chunk) > LYRIC_EMBED_MAX_CHARS:
            chunk = chunk[:LYRIC_EMBED_MAX_CHARS]
        parts.append(chunk)
    return " ".join(p for p in parts if p).strip() or str(row.get("song_id") or "")


class SongTextIndex:
    """곡 메타데이터(제목·가수·장르·가사 일부) 텍스트 임베딩 + 코사인 유사도 검색."""

    def __init__(self, model_name: str, batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None  # (n, d), row L2-normalized

    def is_ready(self) -> bool:
        return self._matrix is not None and len(self._ids) > 0

    def build(self, song_df: pd.DataFrame, use_cache: bool = True) -> None:
        if song_df is None or song_df.empty:
            self._ids = []
            self._matrix = None
            return

        df = song_df.dropna(subset=["song_id"]).copy()
        df["song_id"] = df["song_id"].astype(str)
        df = df.drop_duplicates(subset=["song_id"], keep="last")

        song_ids = df["song_id"].tolist()
        texts = [_row_text(row) for _, row in df.iterrows()]
        fp = _fingerprint(song_ids, texts)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = CACHE_DIR / META_NAME
        ids_path = CACHE_DIR / IDS_NAME
        mat_path = CACHE_DIR / MAT_NAME

        if use_cache and meta_path.is_file() and ids_path.is_file() and mat_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (
                    meta.get("fingerprint") == fp
                    and meta.get("model_name") == self.model_name
                    and meta.get("count") == len(song_ids)
                ):
                    loaded_ids = np.load(ids_path, allow_pickle=True).tolist()
                    mat = np.load(mat_path)
                    if len(loaded_ids) == mat.shape[0] == len(song_ids):
                        self._ids = [str(x) for x in loaded_ids]
                        self._matrix = np.ascontiguousarray(mat.astype(np.float32, copy=False))
                        return
            except Exception:
                pass

        if not texts or all(not t for t in texts):
            self._ids = []
            self._matrix = None
            return

        print(f"[SongTextIndex] 임베딩 구축 중… (모델={self.model_name}, 곡 수={len(texts)})")
        mat = encode_texts(self.model_name, texts, batch_size=self.batch_size)
        self._ids = song_ids
        self._matrix = mat

        try:
            meta_path.write_text(
                json.dumps(
                    {
                        "fingerprint": fp,
                        "model_name": self.model_name,
                        "count": len(song_ids),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            np.save(ids_path, np.array(self._ids, dtype=object))
            np.save(mat_path, self._matrix)
        except OSError as e:
            print(f"[SongTextIndex] 캐시 저장 실패(무시): {e}")

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if not self.is_ready() or self._matrix is None:
            return []
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        nrm = np.linalg.norm(q)
        if nrm > 0:
            q = q / nrm
        sims = self._matrix @ q
        k = min(top_k, len(self._ids))
        if k <= 0:
            return []
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self._ids[i], float(sims[i])) for i in idx]
