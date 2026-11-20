#!/usr/bin/env python3
"""streamio.py — кирпичики потокового звука для голосового сервиса.

Общие для tts_server.py (POST /tts/stream), call_server.py (тест звонка) и call_bench.py (замеры):
  Resampler       — потоковая смена частоты 48 кГц → 8/16/24 кГц без щелчков на стыках (soxr, запасной — свой FIR);
  pcm16 / mulaw   — кодирование кусков: PCM 16 бит LE (браузер, Asterisk slin, FreeSWITCH) и G.711 μ-law 8 кГц (Twilio и т. п.);
  SilenceGate     — срезать тишину в начале фразы (раньше слышен первый звук) и держать «хвостовую» тишину,
                    пока за ней не пошла речь (в конце фразы лишняя тишина не уходит в трубку);
  limiter         — мягкий ограничитель пика (громкость в потоке — постоянное усиление, пик не выше −1 dBFS);
  need_prebuffer  — сколько звука надо накопить у слушателя, чтобы при такой скорости синтеза не было провалов.
Только numpy (+ soxr, если есть). Самопроверка: python3 streamio.py
"""
from __future__ import annotations

import math

import numpy as np

RATES = (8000, 16000, 24000, 48000)
ENCODINGS = ("pcm16", "mulaw")


# ---------------------------------------------------------------- смена частоты
class Resampler:
    """Потоковый ресемплер float32 моно. push(x) → готовые отсчёты; push(empty, last=True) — дослать хвост."""

    def __init__(self, sr_in: int, sr_out: int, prefer_soxr: bool = True):
        self.sr_in, self.sr_out = int(sr_in), int(sr_out)
        self.kind = "none" if self.sr_in == self.sr_out else ""
        if self.kind:
            return
        if prefer_soxr:
            try:
                import soxr
                self._rs = soxr.ResampleStream(self.sr_in, self.sr_out, 1, dtype="float32", quality="HQ")
                self.kind = "soxr"
                return
            except Exception:  # noqa: BLE001 — нет soxr: свой FIR
                pass
        if self.sr_in % self.sr_out:
            raise ValueError(f"без soxr умею только кратное понижение: {self.sr_in} → {self.sr_out}")
        self.kind = "fir"
        self.q = self.sr_in // self.sr_out
        n = 32 * self.q + 1                                   # окно-синк, срез 0,45 × новой частоты
        t = np.arange(n) - (n - 1) / 2
        fc = 0.45 / self.q
        h = 2 * fc * np.sinc(2 * fc * t) * np.blackman(n)
        self.h = (h / h.sum()).astype(np.float32)
        self.state = np.zeros(n - 1, np.float32)
        self.phase = 0                                        # индекс следующего выходного отсчёта во входном потоке
        self.delay = (n - 1) // 2                             # задержка фильтра: при last досылаем нули

    def push(self, x: np.ndarray, last: bool = False) -> np.ndarray:
        x = np.asarray(x, np.float32).reshape(-1)
        if self.kind == "none":
            return x
        if self.kind == "soxr":
            return np.asarray(self._rs.resample_chunk(x, last=last), np.float32).reshape(-1)
        if last:
            x = np.concatenate([x, np.zeros(self.delay, np.float32)])
        buf = np.concatenate([self.state, x])
        y = np.convolve(buf, self.h, mode="valid")            # y[i] — выход для входного отсчёта i (с задержкой фильтра)
        idx = np.arange(self.phase, len(y), self.q)
        out = y[idx]
        self.phase = int(idx[-1] + self.q - len(y)) if len(idx) else self.phase - len(y)
        self.state = buf[len(buf) - (len(self.h) - 1):]
        if last and self.delay:
            out = out[self.delay // self.q:]
        return out.astype(np.float32)


# ---------------------------------------------------------------- кодирование
def pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


_EXP = np.array([0, 0, 1, 1, 2, 2, 2, 2] + [3] * 8 + [4] * 16 + [5] * 32 + [6] * 64 + [7] * 128, np.int32)


def mulaw(x: np.ndarray) -> bytes:
    """G.711 μ-law (как в ITU-T G.711 / Sun g711.c): float −1…1 → байты."""
    s = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int32)
    sign = np.where(s < 0, 0x80, 0)
    s = np.minimum(np.abs(s), 32635) + 0x84
    exp = _EXP[(s >> 7) & 0xFF]
    mant = (s >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant) & 0xFF).astype(np.uint8).tobytes()


def mulaw_decode(b: bytes) -> np.ndarray:
    u = ~np.frombuffer(b, np.uint8).astype(np.int32) & 0xFF
    sign, exp, mant = u & 0x80, (u >> 4) & 0x07, u & 0x0F
    s = (((mant << 3) + 0x84) << exp) - 0x84
    return (np.where(sign, -s, s) / 32768.0).astype(np.float32)


def encode(x: np.ndarray, encoding: str) -> bytes:
    return mulaw(x) if encoding == "mulaw" else pcm16(x)


def bytes_per_sample(encoding: str) -> int:
    return 1 if encoding == "mulaw" else 2


# ---------------------------------------------------------------- громкость
def limiter(x: np.ndarray, knee: float = 0.8, ceil: float = 0.89) -> np.ndarray:
    """Мягкий потолок: до knee — без изменений, выше — плавно к ceil (−1 dBFS)."""
    a = np.abs(x)
    over = a > knee
    if not over.any():
        return x
    y = x.copy()
    r = ceil - knee
    y[over] = np.sign(x[over]) * (knee + r * np.tanh((a[over] - knee) / r))
    return y


def speech_gain(x: np.ndarray, sr: int, target_db: float = -20.0, thr_db: float = -45.0, max_gain: float = 8.0) -> float:
    """Усиление, чтобы речь была ≈ target dBFS (по кадрам 20 мс громче thr_db). Для постоянного усиления потока."""
    fr = max(1, int(0.02 * sr))
    n = len(x) // fr
    if n < 3:
        return 1.0
    e = np.sqrt(np.mean(x[: n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    sp = e[20 * np.log10(e) > thr_db]
    if not len(sp):
        return 1.0
    rms = float(np.sqrt(np.mean(sp ** 2)))
    return float(min(max_gain, 10 ** (target_db / 20) / max(rms, 1e-6)))


# ---------------------------------------------------------------- тишина по краям
class SilenceGate:
    """Режет тишину в начале фразы (оставляя поле head) и придерживает тишину, пока за ней не пойдёт речь.

    push(x) → что можно отдавать сейчас; finish() → хвост (тишина конца — не длиннее tail).
    Порог — абсолютный (−45 dBFS по кадрам 20 мс): выход VoxCPM2 — речь ≈ −25…−20 dBFS, тишина < −60.
    """

    def __init__(self, sr: int, thr_db: float = -45.0, head: float = 0.05, tail: float = 0.08, max_lead: float = 1.5,
                 fade: float = 0.015):
        self.sr, self.thr = sr, thr_db
        self.fr = max(1, int(0.02 * sr))
        self.head, self.tail, self.max_lead = int(head * sr), int(tail * sr), int(max_lead * sr)
        self.fade = max(2, int(fade * sr))
        self.lead = True
        self.held = np.zeros(0, np.float32)
        self.dropped = 0                                      # отсчётов тишины срезано в начале

    def _loud(self, x):
        n = len(x) // self.fr
        if n == 0:
            return np.zeros(0, bool)
        e = np.sqrt(np.mean(x[: n * self.fr].reshape(n, self.fr) ** 2, axis=1) + 1e-12)
        return 20 * np.log10(e) > self.thr

    def push(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self.held, np.asarray(x, np.float32).reshape(-1)])
        loud = self._loud(buf)
        idx = np.where(loud)[0]
        if self.lead:
            if not len(idx):
                keep = buf[-self.head:] if self.head else buf[:0]
                self.dropped += len(buf) - len(keep)
                if self.dropped > self.max_lead:              # речи всё нет — сдаёмся, отдаём как есть
                    self.lead, self.held = False, np.zeros(0, np.float32)
                    return keep
                self.held = keep
                return np.zeros(0, np.float32)
            a = max(0, idx[0] * self.fr - self.head)
            self.dropped += a
            buf = buf[a:].copy()
            f = min(self.fade, len(buf))
            buf[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
            self.lead = False
            loud = self._loud(buf)
            idx = np.where(loud)[0]
        if not len(idx):                                      # одна тишина — держим (вдруг дальше речь)
            self.held = buf
            return np.zeros(0, np.float32)
        b = (idx[-1] + 1) * self.fr                           # конец последнего громкого кадра
        self.held = buf[b:].copy()
        return buf[:b]

    def finish(self) -> np.ndarray:
        t = self.held[: self.tail].copy()
        f = min(self.fade, len(t))
        if f > 1:
            t[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
        self.held = np.zeros(0, np.float32)
        return t


# ---------------------------------------------------------------- запас от провалов
def need_prebuffer(emits: list[tuple[float, float]]) -> float:
    """emits: (время готовности куска от первого куска, с; длительность куска, с) по порядку.
    → наименьший запас (с), при котором проигрывание без провалов: max_i(t_i − звук_до_i)."""
    acc, need = 0.0, 0.0
    for t, d in emits:
        need = max(need, t - acc)
        acc += d
    return max(0.0, need)


# ---------------------------------------------------------------- самопроверка
def _selftest() -> int:
    bad = []

    def ok(name, cond, info=""):
        print(("ok   " if cond else "FAIL ") + name + (f" — {info}" if info else ""))
        if not cond:
            bad.append(name)

    sr = 48000
    t = np.arange(int(1.0 * sr)) / sr
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    for kind in ("soxr", "fir"):
        for out in (8000, 16000, 24000):
            try:
                rs = Resampler(sr, out, prefer_soxr=(kind == "soxr"))
            except Exception as e:  # noqa: BLE001
                ok(f"{kind} {out}", False, str(e))
                continue
            if rs.kind != kind:
                ok(f"{kind} {out}: нет soxr — пропуск", True)
                continue
            parts = [rs.push(tone[i:i + 7680]) for i in range(0, len(tone), 7680)] + [rs.push(np.zeros(0, np.float32), last=True)]
            y = np.concatenate(parts)
            whole = Resampler(sr, out, prefer_soxr=(kind == "soxr"))
            yw = np.concatenate([whole.push(tone), whole.push(np.zeros(0, np.float32), last=True)])
            n = min(len(y), len(yw))
            ok(f"{kind} 48к→{out // 1000}к: длина ≈ {out} отсчётов, кусками = целиком",
               abs(len(y) - out) <= 64 and np.max(np.abs(y[:n] - yw[:n])) < 1e-4, f"{len(y)}, разница {np.max(np.abs(y[:n] - yw[:n])):.1e}")
            mid = y[len(y) // 4: 3 * len(y) // 4]
            f = np.fft.rfftfreq(len(mid), 1 / out)[np.argmax(np.abs(np.fft.rfft(mid)))]
            ok(f"{kind} {out // 1000}к: тон 440 Гц на месте", abs(f - 440) < 5, f"{f:.1f} Гц")
    hi = (0.3 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)          # выше Найквиста для 8 кГц → должен уйти
    rs = Resampler(sr, 8000, prefer_soxr=False)
    y = np.concatenate([rs.push(hi), rs.push(np.zeros(0, np.float32), last=True)])
    ok("fir 8к: 6 кГц подавлен (нет наложения)", np.sqrt(np.mean(y[200:-200] ** 2)) < 0.003, f"rms {np.sqrt(np.mean(y[200:-200] ** 2)):.4f}")

    x = np.linspace(-1, 1, 20001).astype(np.float32)
    d = mulaw_decode(mulaw(x))
    ok("μ-law: 0 → 0xFF, +max → 0x80, −max → 0x00", mulaw(np.array([0.0]))[0] == 0xFF and mulaw(np.array([1.0]))[0] == 0x80
       and mulaw(np.array([-1.0]))[0] == 0x00)
    ref = {0x00: -32124, 0x80: 32124, 0xFF: 0, 0x7F: 0, 0x70: -120, 0xF0: 120}   # таблица G.711 (Sun g711.c ulaw2linear)
    got = {k: int(round(float(mulaw_decode(bytes([k]))[0]) * 32768)) for k in ref}
    ok("μ-law декодер = таблица G.711", got == ref, str(got))
    ok("μ-law туда-обратно: ошибка ≤ 1/16 уровня (шаг сегмента)", np.max(np.abs(d - x) / np.maximum(np.abs(x), 0.01)) < 0.0625,
       f"{np.max(np.abs(d - x) / np.maximum(np.abs(x), 0.01)):.3f}")
    ok("pcm16: 2 байта на отсчёт, LE", len(pcm16(x)) == 2 * len(x) and pcm16(np.array([0.5]))[:2] == (16383).to_bytes(2, "little", signed=True))

    sil = np.zeros(int(0.6 * sr), np.float32)
    sp = (0.1 * np.sin(2 * np.pi * 220 * np.arange(int(0.8 * sr)) / sr)).astype(np.float32)
    sig = np.concatenate([sil, sp, sil[: int(0.3 * sr)], sp, sil])
    g = SilenceGate(sr)
    out = [g.push(sig[i:i + 7680]) for i in range(0, len(sig), 7680)] + [g.finish()]
    y = np.concatenate(out)
    lead = np.argmax(np.abs(y) > 1e-3) / sr
    ok("SilenceGate: начало ≤ 70 мс тишины (было 600; поле 50 мс + кадр 20 мс)", lead <= 0.07 + 1e-6, f"{lead * 1000:.0f} мс, срезано {g.dropped / sr * 1000:.0f} мс")
    ok("SilenceGate: пауза внутри сохранена, конец ≤ 100 мс",
       abs(len(y) / sr - (0.05 + 0.8 + 0.3 + 0.8 + 0.08)) < 0.03, f"{len(y) / sr:.3f} с")
    ok("need_prebuffer: RTF 0,5 → 0, RTF 1,25 → растёт",
       need_prebuffer([(i * 0.08, 0.16) for i in range(10)]) == 0.0 and need_prebuffer([(i * 0.2, 0.16) for i in range(10)]) > 0.3)
    yl = limiter(np.array([0.5, 0.9, 2.0, -3.0], np.float32))
    ok("limiter: ниже колена не трогает, пик ≤ 0,89", yl[0] == 0.5 and np.max(np.abs(yl)) <= 0.89 + 1e-6, str(np.round(yl, 3)))
    print("streamio: " + ("ВСЁ ПРОШЛО" if not bad else "НЕ ПРОШЛО: " + ", ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
