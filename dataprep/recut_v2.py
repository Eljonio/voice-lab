#!/usr/bin/env python3
"""Пересборка фраз из ИСХОДНОГО звука с чистыми краями — для смеси v3 (CPU, без повторной расшифровки).

  # на CPU-сервере с постоянным диском моделей (сырые ролики докачиваются в $VOICE_HOME/raw, этот скрипт их ест и удаляет):
  python dataprep/recut_v2.py --out "$VOICE_HOME/v2" --raw "$VOICE_HOME/raw" --keep "$VOICE_HOME/v2/manifest/ours_v2.jsonl" \
      --keep-val "$VOICE_HOME/v2/manifest/ours_v2_val.jsonl" --workers 4 [--transfer-flag "$VOICE_HOME/raw/.transfer_end"]

Зачем (08.10, прослушивание выборки сессии 1 и A/B с исходником):
  * «щелчки после каждой фразы» — разрез по живому отсчёту без затухания: последний отсчёт у 15 из 40 фраз громче −40 дБ к уровню
    фразы → при остановке звука щелчок. Теперь: поля только из РЕАЛЬНОЙ паузы исходника + затухание/нарастание 15 мс (raised-cosine);
  * Demucs (сессия 1) — не «чиним» музыку: фраза, где Demucs убрал заметный фон (остаток исходник − Demucs > −20 дБ), — вон;
    остальные берём из исходника без Demucs по тем же таймкодам (текст тот же: звук отличался на −25…−42 дБ);
  * обрезанное слово: речь упирается в край окна И за краем в исходнике речь — фраза вон (раньше поле резало начало согласной);
  * интонация: соседние хорошие фразы одного диктора склеиваем (2–3 предложения, ≤ 25 с), если первая кончается посреди мысли
    (пауза после < 0,6 с или нет . ? ! в конце) и ещё ~35 % остальных — чтобы внутри была межфразовая интонация; финальную точку
    снимаем, если после фразы пауза < 0,6 с (модель: «точка = понижение», «без знака = продолжение»);
  * соседство в манифесте: clip_id, seq, src_start/src_end, prev_audio (предыдущая фраза того же ролика и диктора) — build_mix_v2
    ставит её как ref_audio у части строк (тон и темп продолжаются).
Вход — фразы, прошедшие finalize_v2 (манифест ours_v2*.jsonl) + work/<clip>/result.json (все фразы ролика: соседи, таймкоды).
Выход: <out>/clean_v3/<clip>/*.flac, <out>/manifest/ours_v3.jsonl, ours_v3_val.jsonl, recut_summary.json/.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
SR = 16000
HOP = 160          # 10 мс
END_PUNCT = ".?!…"


class RC:
    lead_max = 0.20     # поле до речи из реальной паузы, с
    trail_max = 0.30    # после речи (итог в конце ≤ 0,3 с — правило «тишина в конце ≤ 0,5 с»)
    min_edge = 0.03     # меньше 30 мс паузы (по энергии) до соседней речи — сплошная речь, фраза вон; иначе режем посередине паузы
    fade = 0.015        # нарастание/затухание 15 мс
    thr_db = 30.0       # речь = кадр громче p90 − 30 дБ (и шумового пола + 6 дБ)
    cut_db = 22.0       # за краем окна ≥ 30 мс громче max(p90 − 22, шум + 12) дБ — там речь (обрезано)
    max_len = 25.0
    merge_gap = 1.5     # склеиваем при паузе между речью ≤ 1,5 с
    merge_frac = 0.35   # доля «законченных» фраз, которые всё равно склеиваем со следующей (2–3 предложения)
    merge_soft_len = 20.0
    sent_gap = 0.6      # пауза после ≥ 0,6 с — конец мысли
    demucs_resid_db = -20.0
    full = False        # 11.10 v4 «полной речью»: склеиваем ВСЕ соседние фразы одного диктора при тихой паузе < long_pause, до max_len;
    long_pause = 0.8    # длиннее — ищем самую длинную паузу внутри (≥ split_pref, иначе любую) — так же, как dataprep/cut_v4.py
    split_pref = 0.6
    min_margin = 0.10   # full: край куска без живой паузы ≥ 0,1 с (смена диктора/грязь впритык) — кусок вон


def full_mode(rc: "RC"):
    """Параметры v4 «полная речь» (dataprep/cut_v4.py): поля по 0,3 с из живой паузы, кусок ≤ 30 с, режем только на паузе ≥ 0,8 с."""
    rc.full, rc.lead_max, rc.trail_max, rc.max_len = True, 0.30, 0.30, 30.0
    return rc


def frames_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP
    if n == 0:
        return np.full(1, -120.0)
    f = x[: n * HOP].reshape(n, HOP).astype(np.float64)
    return 10 * np.log10((f ** 2).mean(1) + 1e-12)


def smooth(db: np.ndarray, k: int = 3) -> np.ndarray:
    return np.convolve(db, np.ones(k) / k, mode="same") if len(db) >= k else db


def speech_bounds(x: np.ndarray, a: float, b: float, rc: RC) -> tuple[float, float, float, float, float, float] | None:
    """Речь внутри [a, b] по энергии: (начало, конец, p90 дБ, пол шума дБ, начало ядра, конец ядра). Шум — p10 кадров окна ±0,3 с;
    «мягкая» граница — громче max(p90 − 30, шум + 10) (с дыханием и слабыми согласными), «ядро» — громче max(p90 − 22, шум + 12)
    (сглаживание 30 мс: щелчок одним кадром — не речь)."""
    i0, i1 = int(a * SR), int(b * SR)
    db = smooth(frames_db(x[i0:i1]))
    if len(db) < 5:
        return None
    ref = float(np.percentile(db, 90))
    wide = frames_db(x[max(0, i0 - int(0.3 * SR)): i1 + int(0.3 * SR)])
    noise = float(np.percentile(wide, 10))
    thr = max(ref - rc.thr_db, noise + 10.0)
    loud = np.where(db > thr)[0]
    core = np.where(db > max(ref - rc.cut_db, noise + 12.0))[0]
    if not len(loud) or not len(core):
        return None
    return (a + loud[0] * HOP / SR, a + (loud[-1] + 1) * HOP / SR, ref, noise,
            a + core[0] * HOP / SR, a + (core[-1] + 1) * HOP / SR)


def loud_between(x: np.ndarray, a: float, b: float, level_db: float, run: int = 3) -> bool:
    """Есть ли в [a, b] речь: ≥ run кадров подряд (30 мс) громче level_db."""
    if b - a < 0.01:
        return False
    db = smooth(frames_db(x[max(0, int(a * SR)): max(0, int(b * SR))]))
    hit = db > level_db
    c = 0
    for h in hit:
        c = c + 1 if h else 0
        if c >= run:
            return True
    return False


def fade(y: np.ndarray, sec: float) -> np.ndarray:
    n = min(int(sec * SR), len(y) // 4)
    if n <= 1:
        return y
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    y = y.copy()
    y[:n] *= w
    y[-n:] *= w[::-1]
    return y


def plan_clip(x: np.ndarray, phrases: list[dict], keep: dict[int, dict], rc: RC, rnd: random.Random,
              demucs_resid: dict[int, float] | None = None) -> tuple[list[dict], Counter]:
    """phrases — все фразы ролика из result.json (idx, start, end, separated, no_pause…); keep — idx → строка манифеста (текст, диктор…).
    → список новых фраз {parts, a, b, s_sp, e_sp, text, …} и счётчик причин отсева (в секундах)."""
    rej = Counter()
    ph = sorted([p for p in phrases if p["idx"] < 5000], key=lambda p: p["start"])
    dur = len(x) / SR
    # речь каждой фразы ролика (в т.ч. отсеянных: это соседи, на чью речь нельзя заходить полями)
    sb = {}
    for p in ph:
        r = speech_bounds(x, p["start"], p["end"], rc)
        if r:
            sb[p["idx"]] = r
    order = [p["idx"] for p in ph if p["idx"] in sb]
    pos = {k: i for i, k in enumerate(order)}

    def nb_end(i):   # конец речи предыдущей фразы ролика (любой)
        return sb[order[i - 1]][1] if i > 0 else 0.0

    def nb_start(i):
        return sb[order[i + 1]][0] if i + 1 < len(order) else dur
    good = []
    for p in ph:
        k = p["idx"]
        if k not in keep:
            continue
        if k not in sb:
            rej["тишина"] += p["end"] - p["start"]
            continue
        s, e, ref, noise, sc, ec = sb[k]
        lvl = max(ref - rc.cut_db, noise + 12.0)
        if demucs_resid is not None and p.get("separated") and demucs_resid.get(k, 0.0) > rc.demucs_resid_db:
            rej["музыка/фон (Demucs)"] += e - s
            continue
        i = pos[k]
        # обрезано: речь упирается в край окна, а за краем в исходнике тоже громко
        if s - p["start"] < 0.03 and loud_between(x, p["start"] - 0.08, p["start"], lvl):
            rej["обрезано начало"] += e - s
            continue
        if p["end"] - e < 0.03 and loud_between(x, p["end"], p["end"] + 0.08, lvl):
            rej["обрезан конец"] += e - s
            continue
        # границы разреза: в паузе, не дальше lead_max/trail_max от речи и не дальше середины паузы до соседа;
        # если по энергии пауза «заполнена» (дыхание, шум) — граница окна конвейера (середина паузы VAD)
        pa = max(s - rc.lead_max, (nb_end(i) + s) / 2) if s - nb_end(i) >= 2 * rc.min_edge else p["start"]
        pb = min(e + rc.trail_max, (e + nb_start(i)) / 2) if nb_start(i) - e >= 2 * rc.min_edge else p["end"]
        pa, pb = max(0.0, min(pa, s)), min(dur, max(pb, e))
        if (sc - pa < 0.03 and sc > 0.03) or (pb - ec < 0.03 and ec < dur - 0.03):   # до ядра речи < 30 мс — разрез по слову
            rej["впритык к соседу"] += e - s
            continue
        # на самой границе (±15 мс) речи быть не должно — иначе разрез по слову
        if loud_between(x, pa - 0.015, pa + 0.015, lvl, run=2) or loud_between(x, pb - 0.015, pb + 0.015, lvl, run=2):
            rej["впритык к соседу"] += e - s
            continue
        good.append(dict(idx=k, i=i, s=s, e=e, pa=pa, pb=pb, ref=ref, lvl=lvl, row=keep[k]))
    if rc.full:
        return _plan_full(x, good, rc, nb_start, nb_end, order), rej
    # склейка соседей: та же речь подряд (соседние фразы ролика), один диктор, пауза между ними тихая
    out, j = [], 0
    while j < len(good):
        chain = [good[j]]
        while j + 1 < len(good):
            A, B = chain[-1], good[j + 1]
            gap = B["s"] - A["e"]
            if B["i"] != A["i"] + 1 or A["row"].get("speaker") != B["row"].get("speaker") or gap > rc.merge_gap \
                    or B["e"] - chain[0]["s"] > rc.max_len or len(chain) >= 3 \
                    or loud_between(x, A["e"] + 0.02, B["s"] - 0.02, min(A["lvl"], B["lvl"])):
                break
            t = A["row"]["text"].rstrip()
            mid = gap < rc.sent_gap or not t or t[-1] not in END_PUNCT
            if not (mid or (B["e"] - chain[0]["s"] <= rc.merge_soft_len and rnd.random() < rc.merge_frac)):
                break
            chain.append(B)
            j += 1
        j += 1
        first, last = chain[0], chain[-1]
        a, b = first["pa"], last["pb"]
        lead, trail = first["s"] - a, b - last["e"]
        gap_after = nb_start(last["i"]) - last["e"] if last["i"] + 1 < len(order) else 9.9
        gap_before = first["s"] - nb_end(first["i"]) if first["i"] > 0 else 9.9
        text = " ".join(c["row"]["text"].strip() for c in chain)
        if gap_after < rc.sent_gap and text.endswith("."):
            text = text[:-1].rstrip()          # посреди мысли — без финальной точки
        out.append(dict(parts=[c["idx"] for c in chain], a=round(a, 3), b=round(b, 3), lead=round(lead, 3), trail=round(trail, 3),
                        text=text, gap_after=round(gap_after, 2), gap_before=round(gap_before, 2),
                        sentence_end=int(gap_after >= rc.sent_gap and text[-1:] in END_PUNCT), row=first["row"]))
    return out, rej


def _plan_full(x, good, rc, nb_start, nb_end, order):
    """v4 «полной речью»: цепочка = соседние хорошие фразы ролика одного диктора, пауза между речью < long_pause и тихая (нет чужой
    речи); граница — пауза ≥ long_pause, смена диктора, отсеянная фраза рядом (грязь). Цепочка > max_len — делим на самой длинной
    паузе внутри (≥ split_pref, иначе самой длинной). Поля — pa/pb фраз (живая пауза, ≤ 0,3 с, не дальше середины)."""
    chains, cur = [], []
    for g in good:
        if cur:
            A = cur[-1]
            gap = g["s"] - A["e"]
            # соседние по ролику (между ними нет другой фразы) → пауза < long_pause, даже «громкая» (вдох, слитная речь), — одна речь
            if g["i"] != A["i"] + 1 or A["row"].get("speaker") != g["row"].get("speaker") or gap >= rc.long_pause:
                chains.append(cur)
                cur = []
        cur.append(g)
    if cur:
        chains.append(cur)

    def split(ch):
        if ch[-1]["pb"] - ch[0]["pa"] <= rc.max_len or len(ch) == 1:
            return [ch] if ch[-1]["pb"] - ch[0]["pa"] <= rc.max_len else []          # одна фраза длиннее предела — вон
        gaps = [(ch[k + 1]["s"] - ch[k]["e"], k) for k in range(len(ch) - 1)]
        pref = [gk for gk in gaps if gk[0] >= rc.split_pref]
        g, k = max(pref or gaps)
        return split(ch[:k + 1]) + split(ch[k + 1:])
    out = []
    for ch in chains:
        for c in split(ch):
            first, last = c[0], c[-1]
            a, b = first["pa"], last["pb"]
            if first["s"] - a < rc.min_margin or b - last["e"] < rc.min_margin:   # край у смены диктора/грязи без паузы — вон
                continue
            gap_after = nb_start(last["i"]) - last["e"] if last["i"] + 1 < len(order) else 9.9
            gap_before = first["s"] - nb_end(first["i"]) if first["i"] > 0 else 9.9
            text = " ".join(q["row"]["text"].strip() for q in c)
            if gap_after < rc.sent_gap and text.endswith("."):
                text = text[:-1].rstrip()
            out.append(dict(parts=[q["idx"] for q in c], a=round(a, 3), b=round(b, 3), lead=round(first["s"] - a, 3),
                            trail=round(b - last["e"], 3), text=text, gap_after=round(gap_after, 2), gap_before=round(gap_before, 2),
                            sentence_end=int(gap_after >= rc.sent_gap and text[-1:] in END_PUNCT), row=first["row"]))
    return out


def cut_audio(x: np.ndarray, a: float, b: float, rc: RC) -> np.ndarray:
    import voicelab.audio as A
    return fade(A.normalize_rms(x[int(a * SR): int(b * SR)]), rc.fade).astype(np.float32)


# ------------------------------------------------------------------------------------------------ сервер
def demucs_residuals(x: np.ndarray, phrases: list[dict], keep: dict[int, dict]) -> dict[int, float]:
    """Для фраз сессии 1, прошедших Demucs: насколько исходник отличается от сохранённого Demucs-звука (дБ к исходнику)."""
    import soundfile as sf
    out = {}
    for p in phrases:
        k = p["idx"]
        if k not in keep or not p.get("separated") or not p.get("file"):
            continue
        try:
            d, _ = sf.read(p["file"], dtype="float32")
        except Exception:  # noqa: BLE001
            out[k] = 0.0
            continue
        r = x[int(p["start"] * SR): int(p["start"] * SR) + len(d)].astype(np.float64)
        n = min(len(r), len(d))
        r, d = r[:n], d[:n].astype(np.float64)
        g = float(r @ d) / float(d @ d + 1e-12)
        out[k] = float(10 * np.log10(((r - g * d) ** 2).sum() / ((r ** 2).sum() + 1e-12) + 1e-12))
    return out


def work_clip(args):
    cid, raw_path, out_dir, keep, rc_kw, seed, delete_raw = args
    import soundfile as sf
    import voicelab.audio as A
    rc = RC()
    tag = rc_kw.pop("_tag", "v3") if isinstance(rc_kw, dict) else "v3"
    if rc_kw.pop("_full", False):
        full_mode(rc)
    for k, v in rc_kw.items():
        setattr(rc, k, v)
    res = json.loads((Path(out_dir) / "work" / cid / "result.json").read_text())
    x = A.load(raw_path, SR, channels=1)
    sep = bool(res.get("separated", res.get("has_music")))
    dres = demucs_residuals(x, res["phrases"], keep) if sep else None
    plan, rej = plan_clip(x, res["phrases"], keep, rc, random.Random(f"{seed}:{cid}"), dres)
    cdir = Path(out_dir) / f"clean_{tag}" / cid
    cdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in plan:
        y = cut_audio(x, q["a"], q["b"], rc)
        nm = f"{q['parts'][0]:04d}" + (f"-{q['parts'][-1]:04d}" if len(q["parts"]) > 1 else "")
        fp = cdir / f"{cid}_{nm}.flac"
        sf.write(str(fp), np.clip(y, -1, 1), SR, format="FLAC", subtype="PCM_16")
        r0 = q["row"]
        rows.append(dict(audio=str(fp), text=q["text"], duration=round(len(y) / SR, 3), speaker=r0.get("speaker"), lang=r0.get("lang"),
                         style=r0.get("style"), source=f"ours_{tag}", clip_id=cid, src_start=q["a"], src_end=q["b"], parts=q["parts"],
                         n_parts=len(q["parts"]), sentence_end=q["sentence_end"], gap_after=q["gap_after"], gap_before=q["gap_before"],
                         lead=q["lead"], trail=q["trail"], separated_src=int(sep),
                         dnsmos_ovrl=r0.get("dnsmos_ovrl"), split=r0.get("_split", "train")))
    # соседство: предыдущая фраза того же ролика и диктора (ref_audio для продолжения тона)
    last_by_spk = {}
    for i, r in enumerate(rows):
        r["seq"] = i
        prev = last_by_spk.get(r["speaker"])
        if prev is not None and r["src_start"] - prev["src_end"] <= 10.0:
            r["prev_audio"], r["prev_duration"] = prev["audio"], prev["duration"]
        last_by_spk[r["speaker"]] = r
    if delete_raw:
        Path(raw_path).unlink(missing_ok=True)
    return cid, rows, dict(rej), sum(r["duration"] for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--keep", type=Path, required=True)
    ap.add_argument("--keep-val", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--transfer-flag", type=Path, default=None)
    ap.add_argument("--delete-raw", type=int, default=1)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--files-out", type=Path, default=None, help="записать список нужных роликов (для перекачки) и выйти")
    ap.add_argument("--mode", choices=("v3", "full"), default="v3", help="full — v4 «полной речью» (dataprep/cut_v4.py)")
    ap.add_argument("--tag", default="v3", help="clean_<tag>/, manifest/ours_<tag>.jsonl, recut_<tag>_summary.*")
    ap.add_argument("--clips", type=Path, default=None, help="только эти ролики (файл: youtube_<id>[.flac] по строке)")
    a = ap.parse_args()
    keep = defaultdict(dict)
    for path, split in ((a.keep, "train"), (a.keep_val, "val")):
        if path and path.exists():
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                m = re.search(r"_(\d{4})\.flac$", r["audio"])
                if m and int(m.group(1)) < 5000:
                    r["_split"] = split
                    keep[r["clip_id"]][int(m.group(1))] = r
    if a.clips:
        want = {l.strip().replace(".flac", "") for l in open(a.clips) if l.strip()}
        for c in list(keep):
            if c not in want:
                del keep[c]
    clips = sorted(keep)
    if a.files_out:
        a.files_out.write_text("".join(f"{c}.flac\n" for c in clips))
        print(f"роликов с фразами: {len(clips)} → {a.files_out}")
        return
    t0 = time.time()
    md = a.out / "manifest"
    md.mkdir(parents=True, exist_ok=True)
    rej_all, n_ok, sec_ok, todo = Counter(), 0, 0.0, list(clips)
    ft = open(md / f"ours_{a.tag}.jsonl", "w", encoding="utf-8")
    fv = open(md / f"ours_{a.tag}_val.jsonl", "w", encoding="utf-8")
    rck = {"_tag": a.tag, "_full": a.mode == "full"}
    stats = Counter()
    missing = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {}
        while todo or futs:
            # подаём ролики по мере появления сырого файла (перекачка идёт параллельно)
            for cid in list(todo):
                f = a.raw / f"{cid}.flac"
                if f.exists() and not (a.raw / ".rsync-partial" / f"{cid}.flac").exists() and len(futs) < a.workers * 2:
                    futs[ex.submit(work_clip, (cid, str(f), str(a.out), keep[cid], dict(rck), a.seed, a.delete_raw))] = cid
                    todo.remove(cid)
            transfer_done = a.transfer_flag is None or a.transfer_flag.exists()
            if transfer_done and todo and not futs:
                # перекачка кончилась, а файлов нет — эти ролики без исходника: в смесь не идут
                missing += todo
                todo = []
            done = [fu for fu in futs if fu.done()]
            for fu in done:
                cid = futs.pop(fu)
                try:
                    _, rows, rej, sec = fu.result()
                except Exception as e:  # noqa: BLE001
                    print(f"ОШИБКА {cid}: {e}", flush=True)
                    stats["ошибок"] += 1
                    continue
                for r in rows:
                    (fv if r.pop("split") == "val" else ft).write(json.dumps(r, ensure_ascii=False) + "\n")
                    stats["фраз"] += 1
                    stats["склеено_из_2_3"] += int(r["n_parts"] > 1)
                    stats["конец_мысли"] += r["sentence_end"]
                    stats["сек"] += r["duration"]
                    stats["сек_склеек"] += r["duration"] * (r["n_parts"] > 1)
                rej_all.update(rej)
                stats["роликов"] += 1
                if stats["роликов"] % 50 == 0:
                    print(f"{time.strftime('%T')} роликов {stats['роликов']}/{len(clips)}, фраз {stats['фраз']}, {stats['сек'] / 3600:.1f} ч", flush=True)
            if not done:
                time.sleep(2)
    ft.close()
    fv.close()
    kept_in = sum(r["duration"] for c in keep.values() for r in c.values())
    summ = dict(clips=stats["роликов"], clips_no_raw=len(missing), errors=stats["ошибок"], phrases=stats["фраз"],
                hours_in=round(kept_in / 3600, 2), hours_out=round(stats["сек"] / 3600, 2),
                merged_share=round(stats["склеено_из_2_3"] / max(1, stats["фраз"]), 3),
                merged_hours=round(stats["сек_склеек"] / 3600, 2),
                sentence_end_share=round(stats["конец_мысли"] / max(1, stats["фраз"]), 3),
                rejected_h={k: round(v / 3600, 2) for k, v in rej_all.items()}, minutes=round((time.time() - t0) / 60, 1))
    sfx = "" if a.tag == "v3" else f"_{a.tag}"
    summ.update(mode=a.mode, tag=a.tag)
    (a.out / f"recut{sfx}_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    txt = (f"пересборка из исходника: роликов {summ['clips']} (без исходника {summ['clips_no_raw']}, ошибок {summ['errors']}); "
           f"вход {summ['hours_in']} ч → выход {summ['hours_out']} ч, фраз {summ['phrases']}; склеек 2–3 фраз {summ['merged_share'] * 100:.0f} % "
           f"({summ['merged_hours']} ч); кончаются концом мысли {summ['sentence_end_share'] * 100:.0f} %; отсев, ч: {summ['rejected_h']}\n")
    (a.out / f"recut{sfx}_summary.txt").write_text(txt)
    print(txt, end="")
    if summ["errors"] > max(5, 0.02 * summ["clips"]) or summ["phrases"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
