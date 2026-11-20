#!/usr/bin/env python3
"""Нарезка v4 (11.10.2026): «пусть говорят полной речью» — самые длинные непрерывные куски, режем только на длинной паузе.

Правило (по итогам прослушивания 08.10 и 11.10: фразы резались в ненужных местах, речь должна идти целиком):
  * НЕ делим речь на предложения: кусок = вся речь между длинными паузами (≥ 0,8 с по энергии исходника); паузы < 0,8 с и вдохи
    внутри куска не трогаем;
  * кусок длиннее предела модели (30 с: официально VoxCPM 3–30 с; тренер режет по 2 048 токенам = далеко) делим на самой длинной
    паузе внутри (≥ 0,6 с, иначе ≥ 0,4 с) на части ≤ 30 с; паузы ≥ 0,4 с нет — кусок выбрасываем, а не режем посреди слова;
  * поля — живая пауза: по 0,3 с (не дальше середины паузы; при паузе ≥ 0,8 с это всегда 0,3 с); затухание 15 мс, края = 0;
  * без обрезки тишины (silenceremove нет), без Demucs и шумодава; громкость — одно статическое усиление;
  * короткая реплика («Иә.», «Жоқ.») — только когда реально стоит отдельно: островок речи 0,3–3 с с длинными паузами с обеих сторон;
  * смена говорящего и грязь (музыка, шум, два голоса) — тоже границы; их знает обработка (ECAPA, DNSMOS, AST) и передаёт сюда
    `hard=[(t0, t1), …]` — интервалы, которые выбрасываются, их края становятся границами кусков. На странице прослушки моделей нет;
  * речь, упирающаяся в край анализируемого окна (за краем неизвестно, кончилась ли), — не берём.
Энергия: кадры 10 мс, сглаживание 30 мс; речь = громче max(p90 − 30 дБ, пол шума p3 + 10 дБ), но не выше p90 − 15 дБ
(как recut_v2.speech_bounds; пол p3, а не p10: в плотной речи пауз меньше 10 % кадров); всплеск < 50 мс — не речь.
Этим же кодом режутся клипы для A/B-прослушивания «исходник ↔ кусок», а в обработке v4 — финальные куски.

  python3 dataprep/cut_v4.py --selftest
"""
from __future__ import annotations

import numpy as np


class CutCfg:
    hop = 0.010          # кадр энергии, с
    smooth = 3           # сглаживание, кадров (30 мс)
    thr_db = 30.0        # речь = громче p90 − 30 дБ …
    noise_db = 10.0      # … и громче пола шума + 10 дБ
    long_pause = 0.80    # граница куска — только пауза не короче (внутри куска паузы и вдохи остаются)
    split_pref = 0.60    # кусок > max_len делим на самой длинной паузе ≥ split_pref …
    split_min = 0.40     # … а если таких нет — ≥ split_min; иначе кусок вон
    min_pause = 0.40     # для speech_units(): паузы короче склеиваются в одну единицу
    lead = 0.30          # поле до речи (≤ половины паузы)
    trail = 0.30         # поле после речи (≤ половины паузы)
    fade = 0.015         # нарастание/затухание
    max_len = 30.0       # предел модели (клип; опору ref build_mix даёт только при клип + ref ≤ 30 с)
    min_speech = 0.30    # кусок короче — не речь
    short_max = 3.0      # отдельная реплика: островок ≤ 3 с между длинными паузами


def frames_db(x: np.ndarray, sr: int, c: CutCfg = CutCfg) -> np.ndarray:
    h = max(1, int(c.hop * sr))
    n = len(x) // h
    if n == 0:
        return np.full(1, -120.0)
    f = x[: n * h].reshape(n, h).astype(np.float64)
    db = 10 * np.log10((f ** 2).mean(1) + 1e-12)
    if n >= c.smooth:                                    # края — повтором крайних кадров, не нулями
        k = c.smooth // 2
        db = np.convolve(np.pad(db, (k, c.smooth - 1 - k), mode="edge"), np.ones(c.smooth) / c.smooth, mode="valid")
    return db


def speech_units(x: np.ndarray, sr: int, c: CutCfg = CutCfg) -> tuple[list[tuple[float, float]], float]:
    """Единицы речи (s, e), разделённые паузами ≥ min_pause; порог речи в дБ."""
    db = frames_db(x, sr, c)
    p90, floor = float(np.percentile(db, 90)), float(np.percentile(db, 3))   # пол — p3: пауз в живой речи бывает < 10 % кадров
    thr = min(max(p90 - c.thr_db, floor + c.noise_db), p90 - 15.0)
    sp = db > thr
    runs, i, n = [], 0, len(sp)
    while i < n:
        if sp[i]:
            j = i
            while j < n and sp[j]:
                j += 1
            if (j - i) * c.hop >= 0.05:                  # всплеск < 50 мс (щелчок, стук) — не речь; на границе его ловит check()
                runs.append([i * c.hop, j * c.hop])
            i = j
        else:
            i += 1
    units = []
    for s, e in runs:
        if units and s - units[-1][1] < c.min_pause:
            units[-1][1] = e
        else:
            units.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in units], thr


def plan(x: np.ndarray, sr: int, c: CutCfg = CutCfg, hard: list[tuple[float, float]] | None = None) -> tuple[list[dict], dict]:
    """Куски окна: {a, b, s, e, lead, trail, kind: chunk|short, pauses_inside}; второе — отказы в секундах.
    hard — интервалы, которые выбрасываются (смена говорящего ±, музыка, шум): их края — тоже границы."""
    dur = len(x) / sr
    units, thr = speech_units(x, sr, c)                # паузы < min_pause уже внутри единиц
    rej = {"край окна": 0.0, "длинно без паузы": 0.0, "грязь": 0.0}
    hard = sorted(hard or [])

    def dirty(s, e):
        return any(s < h1 and e > h0 for h0, h1 in hard)
    # единицы → куски по длинным паузам и границам грязи
    chunks, cur = [], []
    for u in units:
        if dirty(*u):
            if cur:
                chunks.append(cur)
            cur = []
            rej["грязь"] += u[1] - u[0]
            chunks.append(None)                         # граница: соседи не склеиваются через грязь
            continue
        if cur and (u[0] - cur[-1][1] >= c.long_pause):
            chunks.append(cur)
            cur = []
        cur.append(u)
    if cur:
        chunks.append(cur)
    # паузы до/после куска (None = край окна или грязь рядом — неизвестно / нельзя)
    out = []
    flat = [ch for ch in chunks]
    for i, ch in enumerate(flat):
        if ch is None:
            continue
        prev_ch = flat[i - 1] if i > 0 else None
        next_ch = flat[i + 1] if i + 1 < len(flat) else None
        s, e = ch[0][0], ch[-1][1]
        pb = (s - prev_ch[-1][1]) if prev_ch else (s if s >= c.long_pause else None)
        pa = (next_ch[0][0] - e) if next_ch else ((dur - e) if dur - e >= c.long_pause else None)
        if (i > 0 and prev_ch is None and flat[i - 1] is None) or (i + 1 < len(flat) and next_ch is None and flat[i + 1] is None):
            pass                                         # рядом грязь: пауза до неё известна по единицам ниже
        edge_b, edge_a = pb is None, pa is None        # край окна (только у окна страницы; у целого ролика краёв нет)
        if edge_b or edge_a:                           # отрезаем крайний кусок до первой/после последней паузы ≥ split_min, середина — целиком
            gi = [k for k in range(len(ch) - 1) if ch[k + 1][0] - ch[k][1] >= c.split_min]
            i0 = gi[0] if (edge_b and gi) else -1
            i1 = gi[-1] if (edge_a and gi) else len(ch) - 1
            if not gi or i1 <= i0:
                rej["край окна"] += e - s
                continue
            mid = ch[i0 + 1:i1 + 1]
            rej["край окна"] += (e - s) - (mid[-1][1] - mid[0][0])
            pb = ch[i0 + 1][0] - ch[i0][1] if edge_b else pb
            pa = ch[i1 + 1][0] - ch[i1][1] if edge_a else pa
            ch = mid
        for part, ppb, ppa in _split(ch, pb, pa, c, rej):
            ps, pe = part[0][0], part[-1][1]
            sp = sum(u1 - u0 for u0, u1 in part)
            if sp < c.min_speech:
                continue
            lead, trail = min(c.lead, ppb / 2), min(c.trail, ppa / 2)
            inside = [round(part[k + 1][0] - part[k][1], 2) for k in range(len(part) - 1)]
            out.append(dict(a=round(ps - lead, 3), b=round(pe + trail, 3), s=ps, e=pe, lead=round(lead, 3), trail=round(trail, 3),
                            kind="short" if pe - ps <= c.short_max and ppb >= c.long_pause and ppa >= c.long_pause else "chunk",
                            pauses_inside=inside, end_of_thought=bool(ppa >= c.long_pause)))
    return out, rej


def _split(ch, pb, pa, c: CutCfg, rej):
    """Кусок длиннее max_len (с полями) → части на самой длинной внутренней паузе (≥ split_pref, иначе ≥ split_min), рекурсивно."""
    s, e = ch[0][0], ch[-1][1]
    if e - s + min(c.lead, pb / 2) + min(c.trail, pa / 2) <= c.max_len:
        return [(ch, pb, pa)]
    gaps = [(ch[k + 1][0] - ch[k][1], k) for k in range(len(ch) - 1)]
    for lim in (c.split_pref, c.split_min):
        cand = [(g, k) for g, k in gaps if g >= lim and ch[k][1] - s >= 3.0 and e - ch[k + 1][0] >= 3.0]
        if cand:
            g, k = max(cand)
            return _split(ch[:k + 1], pb, g, c, rej) + _split(ch[k + 1:], g, pa, c, rej)
    rej["длинно без паузы"] += e - s
    return []


def fade(y: np.ndarray, sr: int, sec: float) -> np.ndarray:
    n = min(int(sec * sr), len(y) // 4)
    y = y.astype(np.float32).copy()
    if n > 1:
        w = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))).astype(np.float32)
        y[:n] *= w
        y[-n:] *= w[::-1]
        y[0] = y[-1] = 0.0
    return y


def cut(x: np.ndarray, sr: int, a: float, b: float, c: CutCfg = CutCfg) -> np.ndarray:
    return fade(x[max(0, int(round(a * sr))): min(len(x), int(round(b * sr)))], sr, c.fade)


def check(x: np.ndarray, sr: int, p: dict, thr_db: float, c: CutCfg = CutCfg) -> list[str]:
    """Проверка готовой фразы: края в тишине исходника, поля ≥ 0,2 с (или половина паузы), первый/последний отсчёт 0."""
    bad = []
    y = cut(x, sr, p["a"], p["b"], c)
    if abs(float(y[0])) > 1e-6 or abs(float(y[-1])) > 1e-6:
        bad.append("край не ноль")
    db = frames_db(x, sr, c)
    for t in (p["a"], p["b"]):
        k = int(t / c.hop)
        w = db[max(0, k - 2): k + 3]
        if len(w) and float(w.max()) > thr_db:
            bad.append(f"речь на границе {t:.2f} с")
    if p["lead"] < 0.199 or p["trail"] < 0.199:
        bad.append(f"поле < 0,2 с ({p['lead']}/{p['trail']})")
    if p["b"] - p["a"] > c.max_len + 0.01:
        bad.append(f"длиннее {c.max_len:g} с")
    return bad


def _selftest():
    sr = 24000
    rng = np.random.default_rng(1)

    def tone(sec, f=180.0):
        t = np.arange(int(sec * sr)) / sr
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2
        return (0.3 * env * np.sin(2 * np.pi * f * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)

    sil = lambda sec: (0.001 * rng.standard_normal(int(sec * sr))).astype(np.float32)
    long_talk = []
    for k in range(7):                                   # 7 × 5 с речи, паузы 0,3 с, одна 0,65 с посередине → 36 с > 30
        long_talk += [tone(5), sil(0.65 if k == 3 else 0.3)]
    parts = [sil(1.0), tone(4), sil(0.25), tone(5), sil(1.0), tone(0.6, 220), sil(1.0), tone(6), sil(0.5), tone(6), sil(0.9)] \
        + long_talk[:-1] + [sil(1.0), tone(33), sil(1.0)]
    x = np.concatenate(parts)
    ph, rej = plan(x, sr)
    _, thr = speech_units(x, sr)
    kinds = [p["kind"] for p in ph]
    lens = [round(p["b"] - p["a"], 1) for p in ph]
    errs = [e for p in ph for e in check(x, sr, p, thr)]
    ok = kinds == ["chunk", "short", "chunk", "chunk", "chunk"] and not errs and rej["длинно без паузы"] > 32 \
        and all(l <= 30 for l in lens) and ph[2]["pauses_inside"] == [0.5] and lens[0] > 9.5
    print(f"selftest {'ok' if ok else 'FAILED'}: куски {[(p['kind'], p['a'], p['b'], p['lead'], p['trail'], p['pauses_inside']) for p in ph]}, "
          f"длины {lens}, отказ {rej}, ошибки {errs}")
    return ok


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    print(__doc__)
