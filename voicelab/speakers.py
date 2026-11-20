"""Один ли говорящий во фразе + кто есть кто в ролике.

pyannote/speaker-diarization требует токен HF и принятие условий (gated) — токенов
нет, поэтому открытая замена: эмбеддинги голоса SpeechBrain ECAPA-TDNN
(speechbrain/spkrec-ecapa-voxceleb, Apache-2.0, без регистрации).
  * внутри фразы: эмбеддинги окон 1.5 с; если какие-то два окна сильно разные
    (косинус ниже порога) — во фразе вероятно два голоса;
  * по ролику: эмбеддинг фразы, агломеративная кластеризация -> spk0, spk1, ...
Наложение голосов (говорят одновременно) этим ловится хуже, чем pyannote.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

SR = 16000
CACHE = Path(__file__).resolve().parent.parent / "models" / "ecapa"
LOCAL_SRC = Path(__file__).resolve().parent.parent / "models" / "ecapa_src"


class SpeakerChecker:
    def __init__(self, device: str = "cuda"):
        self.enc = EncoderClassifier.from_hparams(
            source=str(LOCAL_SRC) if (LOCAL_SRC / "embedding_model.ckpt").exists()
            else "speechbrain/spkrec-ecapa-voxceleb", savedir=str(CACHE),
            run_opts={"device": device},
        )
        self.device = device

    @torch.inference_mode()
    def embed(self, chunks: list[np.ndarray]) -> np.ndarray:
        n = max(len(c) for c in chunks)
        x = torch.zeros(len(chunks), n)
        lens = torch.zeros(len(chunks))
        for i, c in enumerate(chunks):
            x[i, : len(c)] = torch.from_numpy(c)
            lens[i] = len(c) / n
        e = self.enc.encode_batch(x.to(self.device), lens.to(self.device)).squeeze(1)
        e = torch.nn.functional.normalize(e, dim=-1)
        return e.cpu().numpy()

    def phrase_check(self, audio16k: np.ndarray) -> dict:
        """min_sim — наименьшее сходство между окнами фразы (1 = один голос).

        Окна адаптивные: на 1.5 с ECAPA шумит (один диктор даёт до 0.2), на 3 с —
        один диктор >= 0.5, два разных ~0. Калибровка: scripts/dev/calib_spk.py.
        """
        dur = len(audio16k) / SR
        win_s = 3.0 if dur >= 6 else 2.0 if dur >= 4 else 1.5
        w, h = int(win_s * SR), int(win_s / 2 * SR)
        starts = list(range(0, max(len(audio16k) - w, 0) + 1, h)) or [0]
        chunks = [audio16k[s : s + w] for s in starts]
        # окна почти без звука не участвуют (паузы дают мусорные эмбеддинги)
        rms = np.array([np.sqrt(np.mean(c ** 2)) for c in chunks])
        keep = rms > 0.25 * np.median(rms)
        chunks = [c for c, k in zip(chunks, keep) if k]
        full = self.embed([audio16k])[0]
        if len(chunks) < 2:
            return dict(min_sim=1.0, emb=full)
        E = self.embed(chunks)
        S = E @ E.T
        return dict(min_sim=float(S.min()), emb=full)


def cluster(embs: np.ndarray, thr: float = 0.55) -> np.ndarray:
    """Агломеративная кластеризация по косинусу (average linkage). Метки 0..k."""
    n = len(embs)
    if n == 0:
        return np.zeros(0, dtype=int)
    from scipy.cluster.hierarchy import fcluster, linkage
    if n == 1:
        return np.zeros(1, dtype=int)
    Z = linkage(embs, method="average", metric="cosine")
    lab = fcluster(Z, t=1 - thr, criterion="distance") - 1
    # переименовать: самый частый говорящий = 0
    order = np.argsort(-np.bincount(lab))
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in lab])
