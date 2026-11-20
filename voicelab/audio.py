"""Загрузка/сохранение звука через ffmpeg, ресемплинг, простые измерения."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def load(path: str | Path, sr: int, channels: int = 1) -> np.ndarray:
    """Декодирует любой формат (mp4/m4a/ogg/opus/mp3/wav) через ffmpeg.

    Возвращает float32: (samples,) для моно или (channels, samples) для стерео.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-vn", "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).copy()
    if channels > 1:
        x = x.reshape(-1, channels).T
    return x


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    if x.ndim == 2:  # (ch, n)
        return soxr.resample(x.T, sr_in, sr_out, quality="HQ").T.astype(np.float32)
    return soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32)


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0) if x.ndim == 2 else x


def save_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = x.T if x.ndim == 2 else x
    sf.write(str(path), np.clip(data, -1, 1), sr, subtype="PCM_16")


def save_mp3(path: str | Path, x: np.ndarray, sr: int, bitrate: str = "128k") -> None:
    """Для демо-страницы: лёгкий mp3 рядом с html."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(x.T if x.ndim == 2 else x, -1, 1).astype(np.float32)
    ch = 1 if data.ndim == 1 else data.shape[1]
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr),
        "-ac", str(ch), "-i", "-", "-codec:a", "libmp3lame", "-b:a", bitrate, str(path),
    ]
    subprocess.run(cmd, input=data.tobytes(), check=True)


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9))


def snr_estimate(x: np.ndarray, sr: int, frame_ms: float = 25.0) -> float:
    """Грубая оценка SNR без эталона: громкие кадры (речь) против тихих (фон).

    90-й перцентиль энергии кадров минус 10-й. Для фразы с паузами на краях
    даёт адекватную картину «насколько фон тише голоса».
    """
    n = int(sr * frame_ms / 1000)
    if len(x) < n * 4:
        return float("nan")
    frames = x[: len(x) // n * n].reshape(-1, n)
    e = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    return float(np.percentile(e, 90) - np.percentile(e, 10))


def clipping_ratio(x: np.ndarray, thr: float = 0.995) -> float:
    return float(np.mean(np.abs(x) >= thr))


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SDR, дБ — насколько est похож на эталон ref."""
    n = min(len(est), len(ref))
    est, ref = est[:n].astype(np.float64), ref[:n].astype(np.float64)
    est, ref = est - est.mean(), ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12)))


def normalize_rms(x: np.ndarray, target_db: float = -23.0, peak_db: float = -1.0) -> np.ndarray:
    """Выравнивание громкости фраз датасета: RMS к target_db, пик не выше peak_db."""
    gain = 10 ** ((target_db - rms_db(x)) / 20)
    y = x * gain
    peak = np.max(np.abs(y)) + 1e-9
    lim = 10 ** (peak_db / 20)
    if peak > lim:
        y = y * (lim / peak)
    return y.astype(np.float32)
