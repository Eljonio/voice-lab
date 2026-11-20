"""Отделение голоса от музыки: Demucs v4 (htdemucs, Meta, MIT).

Модель делит смесь на 4 дорожки: drums, bass, other, vocals. Берём vocals как
«голос», а сумму остальных — как «что осталось музыкой» (mix - vocals).
Ввод/вывод — 44.1 кГц стерео, считаем на видеокарте кусками.
"""
from __future__ import annotations

import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

SR = 44100


class Separator:
    def __init__(self, name: str = "htdemucs", device: str = "cuda"):
        self.device = device
        self.model = get_model(name)
        self.model.to(device).eval()
        self.vocals_idx = self.model.sources.index("vocals")
        self.name = name

    # Длинный ролик целиком: 4 дорожки × 2 канала × float32 = 1,7 ГБ на 20 мин плюс копии — пик RSS 8,4 ГБ (замер 05.10)
    # и своп рядом с браузером. Режем на куски по CHUNK_S с нахлёстом OVL_S и линейной склейкой.
    CHUNK_S = 240
    OVL_S = 2

    def __call__(self, stereo44k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """stereo44k: (2, n) float32. Возвращает (vocals, rest) той же формы."""
        n, step, ov = stereo44k.shape[1], self.CHUNK_S * SR, self.OVL_S * SR
        if n <= step + ov:
            return self._sep(stereo44k)
        voc = np.zeros_like(stereo44k, dtype=np.float32)
        start = 0
        while start < n:
            a, b = max(0, start - ov), min(n, start + step)
            v = self._sep(stereo44k[:, a:b])[0]
            if start == 0:
                voc[:, a:b] = v
            else:
                w = np.linspace(0.0, 1.0, start - a, dtype=np.float32)
                voc[:, a:start] = voc[:, a:start] * (1 - w) + v[:, : start - a] * w
                voc[:, start:b] = v[:, start - a:]
            start = b
        return voc, (stereo44k - voc).astype(np.float32)

    @torch.inference_mode()
    def _sep(self, stereo44k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        wav = torch.from_numpy(np.ascontiguousarray(stereo44k)).float()
        ref = wav.mean(0)
        mu, sd = ref.mean(), ref.std() + 1e-8
        x = ((wav - mu) / sd)[None]
        src = apply_model(
            self.model, x, device=self.device, shifts=0, split=True,
            overlap=0.25, progress=False, segment=None,
        )[0]
        src = src * sd + mu
        vocals = src[self.vocals_idx].cpu().numpy()
        rest = stereo44k - vocals
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return vocals.astype(np.float32), rest.astype(np.float32)
