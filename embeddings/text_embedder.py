from __future__ import annotations

import os
import threading

import numpy as np

_model_lock = threading.Lock()
_model = None
_model_key: str | None = None


def get_sentence_embedder(model_name: str):
    """
    CPU·저사양 환경용 SentenceTransformer 싱글톤.
    TORCH_NUM_THREADS 환경변수로 스레드 수 제한 가능.
    """
    global _model, _model_key
    import torch
    from sentence_transformers import SentenceTransformer

    with _model_lock:
        if _model is not None and _model_key == model_name:
            return _model
        threads = int(os.getenv("TORCH_NUM_THREADS", "2"))
        try:
            torch.set_num_threads(max(1, threads))
        except Exception:
            pass
        _model = SentenceTransformer(model_name, device="cpu")
        _model_key = model_name
        return _model


def encode_texts(model_name: str, texts: list[str], batch_size: int = 16) -> np.ndarray:
    """L2 정규화된 임베딩 행렬 (n × d), float32."""
    model = get_sentence_embedder(model_name)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 200,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)
