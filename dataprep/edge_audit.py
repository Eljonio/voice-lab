#!/usr/bin/env python3
"""Аудит звука фраз: резкие края (обрезанное начало/конец слова) и помехи в паузах («пузырьки», щелчки, гейт).

  python dataprep/edge_audit.py FILE.flac ... [--json out.json]        по файлам + сводка
  python dataprep/edge_audit.py --pairs A_DIR B_DIR                       одинаковые имена файлов в двух папках (например Demucs / исходник)

Метрики (кадр 20 мс, шаг 10 мс, ref = p90 громкости кадров фразы):
  lead_ms / trail_ms — тишина до первого / после последнего кадра громче ref − 25 дБ;
  edge_s / edge_e    — громкость первых / последних 20 мс относительно ref, дБ (> −20 дБ — край «в речи», вероятный обрыв);
  pause_s            — секунд пауз (кадры тише ref − 30 дБ);
  clicks_ps          — щелчков на секунду паузы: подблок 2 мс с пиком ≥ 12 дБ над медианой пауз и пик-фактором кадра ≥ 8;
  flux_pause         — «бульканье» в паузах: средний положительный спектральный поток (дБ) между соседними кадрами пауз
                       (музыкальный шум после Demucs / сильного шумодава — пятна спектра включаются и гаснут);
  flat_pause         — спектральная плоскость пауз (шум ~0,3–0,6; «пятна»/тоны — ниже);
  digital_sil        — доля кадров пауз тише −80 дБFS (гейт в исходнике: паузы вырезаны в ноль, на стыках щёлкает);
  clip               — доля отсчётов |x| ≥ 0,99.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

EPS = 1e-10


def frames(x: np.ndarray, sr: int, win: float = 0.02, hop: float = 0.01) -> np.ndarray:
    w, h = int(win * sr), int(hop * sr)
    n = max(0, 1 + (len(x) - w) // h)
    if n == 0:
        return np.zeros((0, w), dtype=np.float32)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    return x[idx]


def audit(x: np.ndarray, sr: int) -> dict:
    if x.ndim > 1:
        x = x.mean(1)
    x = x.astype(np.float32)
    F = frames(x, sr)
    if len(F) < 10:
        return {"dur": round(len(x) / sr, 3), "bad": "короткий"}
    rms = np.sqrt((F ** 2).mean(1) + EPS)
    db = 20 * np.log10(rms + EPS)
    ref = float(np.percentile(db, 90))
    loud = np.where(db > ref - 25)[0]
    lead = int(loud[0]) * 10 if len(loud) else 0
    trail = (len(db) - 1 - int(loud[-1])) * 10 if len(loud) else 0
    w = int(0.02 * sr)
    e_s = 20 * np.log10(np.sqrt((x[:w] ** 2).mean() + EPS)) - ref
    e_e = 20 * np.log10(np.sqrt((x[-w:] ** 2).mean() + EPS)) - ref
    pause = db < ref - 30
    out = dict(dur=round(len(x) / sr, 3), ref_db=round(ref, 1), lead_ms=lead, trail_ms=trail, edge_s=round(float(e_s), 1),
               edge_e=round(float(e_e), 1), pause_s=round(float(pause.sum()) * 0.01, 2),
               clip=round(float((np.abs(x) >= 0.99).mean()), 5))
    if pause.sum() >= 5:
        P = F[pause]
        pr = rms[pause]
        med = float(np.median(pr))
        # щелчки: подблоки 2 мс внутри кадров паузы
        b = int(0.002 * sr)
        sub = P[:, : (P.shape[1] // b) * b].reshape(len(P), -1, b)
        peak = np.abs(sub).max(2).max(1)
        crest = peak / (pr + EPS)
        clicks = int(((20 * np.log10(peak + EPS) - 20 * np.log10(med + EPS) >= 12) & (crest >= 8)).sum())
        out["clicks_ps"] = round(clicks / (pause.sum() * 0.01), 2)
        # спектр пауз: поток и плоскость
        win = np.hanning(F.shape[1])
        S = np.abs(np.fft.rfft(P * win, axis=1)) + 1e-7
        L = 20 * np.log10(S)
        idx = np.where(pause)[0]
        adj = np.where(np.diff(idx) == 1)[0]
        if len(adj):
            d = L[adj + 1] - L[adj]
            out["flux_pause"] = round(float(np.maximum(d, 0).mean()), 2)
        out["flat_pause"] = round(float(np.mean(np.exp(np.log(S).mean(1)) / S.mean(1))), 3)
        out["digital_sil"] = round(float((20 * np.log10(pr + EPS) < -80).mean()), 3)
        out["pause_db"] = round(20 * np.log10(med + EPS) - ref, 1)
    return out


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "bad" not in r]
    s = {"n": len(ok)}
    for k in ("lead_ms", "trail_ms", "edge_s", "edge_e", "clicks_ps", "flux_pause", "flat_pause", "digital_sil", "pause_db"):
        v = [r[k] for r in ok if k in r]
        if v:
            s[k] = dict(med=round(st.median(v), 2), p10=round(float(np.percentile(v, 10)), 2), p90=round(float(np.percentile(v, 90)), 2))
    s["lead_lt50"] = sum(r["lead_ms"] < 50 for r in ok)
    s["trail_lt50"] = sum(r["trail_ms"] < 50 for r in ok)
    s["edge_s_gt20"] = sum(r["edge_s"] > -20 for r in ok)
    s["edge_e_gt20"] = sum(r["edge_e"] > -20 for r in ok)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--pairs", nargs=2, type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    res = {}
    if a.pairs:
        A, B = a.pairs
        for f in sorted(A.glob("*.flac")) + sorted(A.glob("*.wav")):
            g = B / f.name
            if g.exists():
                xa, sa = sf.read(f)
                xb, sb = sf.read(g)
                res[f.name] = {"A": audit(xa, sa), "B": audit(xb, sb)}
        for k in ("A", "B"):
            print(k, json.dumps(summarize([v[k] for v in res.values()]), ensure_ascii=False))
    else:
        for f in a.files:
            x, sr = sf.read(f)
            res[str(f)] = audit(x, sr)
        print(json.dumps(summarize(list(res.values())), ensure_ascii=False))
    if a.json:
        a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
