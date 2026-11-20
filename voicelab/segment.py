"""Нарезка на фразы 2..15 с по паузам. VAD: Silero VAD (MIT, модель внутри pip-пакета)."""
from __future__ import annotations

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

SR = 16000


class Segmenter:
    def __init__(self, min_len=2.0, max_len=15.0, cut_gap=0.45, cut_gap_long=0.25,
                 long_len=8.0, pad=0.15):
        self.vad = load_silero_vad()
        self.min_len, self.max_len = min_len, max_len
        self.cut_gap, self.cut_gap_long, self.long_len = cut_gap, cut_gap_long, long_len
        self.pad = pad

    def speech(self, audio16k: np.ndarray) -> list[tuple[float, float]]:
        ts = get_speech_timestamps(
            torch.from_numpy(audio16k), self.vad, sampling_rate=SR, threshold=0.5,
            min_speech_duration_ms=200, min_silence_duration_ms=120, speech_pad_ms=40,
        )
        return [(t["start"] / SR, t["end"] / SR) for t in ts]

    @staticmethod
    def _split_long(audio16k, s, e, max_len):
        """Отрезок длиннее max_len режем в самых тихих местах (минимум энергии)."""
        out, cur = [], s
        while e - cur > max_len:
            lo, hi = cur + max_len * 0.5, cur + max_len
            a, b = int(lo * SR), int(hi * SR)
            seg = audio16k[a:b]
            fr = int(0.02 * SR)
            if len(seg) < fr * 2:
                break
            en = np.convolve(seg ** 2, np.ones(fr) / fr, mode="valid")
            cut = lo + int(np.argmin(en)) / SR
            out.append((cur, cut))
            cur = cut
        out.append((cur, e))
        return out

    def phrases(self, audio16k: np.ndarray) -> list[dict]:
        segs = self.speech(audio16k)
        dur = len(audio16k) / SR
        groups: list[list[float]] = []
        for s, e in segs:
            if not groups:
                groups.append([s, e]); continue
            g = groups[-1]
            gap, cur_len, new_len = s - g[1], g[1] - g[0], e - g[0]
            cut = (gap >= self.cut_gap and cur_len >= self.min_len) or \
                  (gap >= self.cut_gap_long and cur_len >= self.long_len) or \
                  new_len > self.max_len or gap > 1.5
            if cut:
                groups.append([s, e])
            else:
                g[1] = e
        out = []
        for s, e in groups:
            for a, b in self._split_long(audio16k, s, e, self.max_len):
                out.append([a, b])
        # поля по краям, но не залезая в соседнюю фразу
        res = []
        for i, (a, b) in enumerate(out):
            left = out[i - 1][1] if i > 0 else 0.0
            right = out[i + 1][0] if i + 1 < len(out) else dur
            a2 = max(a - self.pad, (left + a) / 2, 0.0)
            b2 = min(b + self.pad, (b + right) / 2, dur)
            res.append(dict(start=round(a2, 3), end=round(b2, 3), dur=round(b2 - a2, 3),
                            speech_dur=round(b - a, 3)))
        return res
