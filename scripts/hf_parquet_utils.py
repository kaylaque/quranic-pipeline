"""
hf_parquet_utils.py - Utilities for reading HuggingFace parquet files and
converting embedded MP3 audio to float32 arrays for pipeline consumption.

MP3 decoding uses `miniaudio` (pure Python + C extension, no ffmpeg required).
Falls back to `librosa` + `audioread` (macOS CoreAudio) if miniaudio is absent.
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# Pre-check miniaudio availability to avoid fallback on every decode error
_has_miniaudio = False
try:
    import miniaudio
    _has_miniaudio = True
except ImportError:
    logger.debug("miniaudio not available, will fall back to librosa for MP3 decoding")


# ---------------------------------------------------------------------------
# MP3 → numpy array (no ffmpeg)
# ---------------------------------------------------------------------------

def mp3_bytes_to_array(
    mp3_bytes: bytes,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """
    Decode MP3 binary to a float32 numpy array normalised to [-1, 1].

    Strategy (no ffmpeg required):
      1. miniaudio — pure-Python C-extension, handles MP3 in-memory
      2. librosa fallback — writes to a temp file, uses audioread/CoreAudio

    Returns (audio_array, sample_rate).
    """
    if _has_miniaudio:
        try:
            return _decode_miniaudio(mp3_bytes, target_sr)
        except Exception as exc:
            logger.debug("miniaudio decode failed (%s), trying librosa fallback", exc)

    return _decode_librosa(mp3_bytes, target_sr)


def _decode_miniaudio(mp3_bytes: bytes, target_sr: int) -> tuple[np.ndarray, int]:
    import miniaudio

    decoded = miniaudio.decode(
        mp3_bytes,
        output_format=miniaudio.SampleFormat.FLOAT32,
        nchannels=1,
        sample_rate=target_sr,
    )
    samples = np.frombuffer(decoded.samples, dtype=np.float32).copy()
    return samples, target_sr


def _decode_librosa(mp3_bytes: bytes, target_sr: int) -> tuple[np.ndarray, int]:
    """Fallback: write to a temp file and decode with librosa."""
    import librosa

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(mp3_bytes)
            tmp_path = tmp.name

        audio, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
        return audio.astype(np.float32), sr
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Parquet row iteration
# ---------------------------------------------------------------------------

def iter_parquet_rows(
    parquet_path: Path,
    reciters: list[str] | None = None,
    surahs: list[int] | None = None,
    max_rows: int = 0,
) -> Iterator[dict]:
    """
    Yield rows from a parquet file one at a time to avoid loading all audio
    bytes into memory simultaneously.

    Each yielded dict has keys:
        surah_id, ayah_id, surah_name_en, surah_name_ar,
        ayah_ar, ayah_en, reciter_id, reciter_name,
        audio_bytes, audio_path

    Parameters
    ----------
    reciters : list of str, optional
        Filter by reciter ID (case-insensitive)
    surahs : list of int, optional
        Filter by surah ID
    max_rows : int
        Maximum rows to yield (0 = unlimited)
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    yielded = 0

    # Normalise reciter filter to lowercase
    reciter_filter = set(r.lower() for r in reciters) if reciters else None

    for batch in pf.iter_batches(
        batch_size=50,
        columns=[
            "surah_id", "ayah_id", "surah_name_en", "surah_name_ar",
            "ayah_ar", "ayah_en", "reciter_id", "reciter_name", "audio",
        ],
    ):
        rows = batch.to_pydict()
        n = len(rows["surah_id"])

        for i in range(n):
            if max_rows > 0 and yielded >= max_rows:
                return

            reciter_id = rows["reciter_id"][i]
            surah_id = int(rows["surah_id"][i])

            if reciter_filter and reciter_id.lower() not in reciter_filter:
                continue
            if surahs and surah_id not in surahs:
                continue

            audio_struct = rows["audio"][i]
            if audio_struct is None:
                continue

            yield {
                "surah_id": surah_id,
                "ayah_id": int(rows["ayah_id"][i]),
                "surah_name_en": rows["surah_name_en"][i],
                "surah_name_ar": rows["surah_name_ar"][i],
                "ayah_ar": rows["ayah_ar"][i],
                "ayah_en": rows["ayah_en"][i],
                "reciter_id": reciter_id,
                "reciter_name": rows["reciter_name"][i],
                "audio_bytes": audio_struct.get("bytes") or b"",
                "audio_path": audio_struct.get("path") or "",
            }
            yielded += 1
