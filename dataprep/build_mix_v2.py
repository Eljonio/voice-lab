#!/usr/bin/env python3
"""Смесь v2 для VoxCPM2 (запускать на сервере с постоянным диском данных; CPU-сервера хватает).

  python dataprep/build_mix_v2.py --out "$VOICE_HOME/mix-v3" \
      --ours "$VOICE_HOME/v2/manifest/ours_v2.jsonl" --ours-val "$VOICE_HOME/v2/manifest/ours_v2_val.jsonl" \
      --corpora "$VOICE_HOME/corpora" [--hours ksc2_live=all,ksc2_read=40,kazakhtts2=10,fleurs_kk=all,cv_kk=all,fleurs_ru=all]

Правила (docs/METHOD.md):
  * основа — наша живая речь (ours_v2, после finalize_v2.py): kk / mix / ru, метки стиля из запросов сборщика;
  * KSC2 только «живые» домены целиком (Test: podcasts, radio, talkshow, tv_news — что есть на диске) + чтение дозой
    (KSC2 crowdsourced, KazakhTTS2, FLEURS kk, CV kk) так, чтобы всё чтение было ≤ 25–30 %; русский — наш живой + FLEURS ru;
  * тексты корпусов — через тот же нормализатор (dataprep/textnorm.py): цифры/латиница/символы → строка вон, скобки убраны;
  * края корпусов обрезаются (ffmpeg silenceremove: в начале ≤ 0,15 с, в конце ≤ 0,25 с тишины) — длинная тишина в конце
    учит модель не останавливаться; наши фразы уже нарезаны с полями 0,15 с;
  * диктор ≤ 3 % смеси (у кого есть id), ref_audio — 35 % строк с диктором, ref 3–10 с и ref + клип ≤ 30 с;
  * метки стиля в тексте у 15 % строк: «(живая разговорная речь)» / «(дикторское чтение)» — ручка стиля на инференсе;
  * val: отложенные каналы ours_v2_val + 1 % корпусов (до 100 строк).
Выход: <out>/train.jsonl, val.jsonl, stats.txt, audio/ (16 кГц FLAC корпусов; наши файлы — на месте).

09.10 (mix-v3b):
  * опора (ref_audio) у --ref-frac = 40 % строк train: сначала ХВОСТ 3–8 с предыдущей фразы того же ролика и диктора
    (prev_audio из recut_v2; разрез в самой тихой точке окна, нарастание 15 мс) — до --tail-frac = 30 % строк, остальное —
    другая фраза того же диктора 3–10 с; клип с опорой ≤ 20 с, клип + опора ≤ 30 с;
  * живые домены KSC2 Train (corpora/ksc2_live) берутся только с --ksc2-live-train 1: в них нет знаков препинания;
  * склеенные слова («Күштісосын»): редкое слово (≤ 2 раз) длиной ≥ 8 = два частых слова (≥ 30 раз, каждое ≥ 4 букв) → строка вон;
  * ЖЁСТКИЕ ПРОВЕРКИ после записи (иначе код 3 и «ПРОВЕРКА НЕ ПРОЙДЕНА» в stats.txt): длительность каждого файла заново по
    отсчётам (≠ манифесту > 0,05 с — ошибка), 1–25 с, ≤ 25 симв/с у ВСЕХ строк, опора есть и клип + опора ≤ 30 с; часы ±10 % от
    плана (--expect-hours или сумма наших + заказанных часов корпусов), чтение ≤ --max-read, русский ≥ --min-ru,
    склеенных слов ≤ 2 %. Сводка ещё и в stats.json; выборка для ушей — listen_sample.jsonl (по 20 строк на источник + 20 с хвостом).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from textnorm import normalize  # noqa: E402

DSID = {"ours": 0, "ksc2_live": 1, "ksc2_read": 1, "kazakhtts2": 2, "fleurs_kk": 3, "cv_kk": 3, "fleurs_ru": 4}
READING = {"ksc2_read", "kazakhtts2", "fleurs_kk", "cv_kk", "fleurs_ru"}
AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".opus"}


def dur_of(p: str) -> float:
    try:
        import soundfile as sf
        return float(sf.info(p).duration)
    except Exception:  # noqa: BLE001
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                             capture_output=True, text=True).stdout.strip()
        return float(out or 0)


def stem_pairs(dirs: list[Path], tag: str) -> list[dict]:
    rows = []
    for root in dirs:
        if not root.exists():
            continue
        txt = {p.stem: p for p in root.rglob("*.txt")}
        for p in root.rglob("*"):
            if p.suffix.lower() in AUDIO and p.stem in txt:
                t = txt[p.stem].read_text(encoding="utf-8", errors="ignore").strip()
                if not t:
                    continue
                spk = ""
                if tag == "kazakhtts2":
                    spk = next((q for q in p.parts if re.fullmatch(r"[MF]\d+", q)), p.parent.parent.name)
                rows.append({"audio": str(p), "text": t, "speaker": f"{tag}:{spk}" if spk else "", "domain": root.name})
    return rows


def fleurs(root: Path) -> list[dict]:
    import csv
    rows = []
    for split in ("train", "dev"):
        tsv = root / f"{split}.tsv"
        if tsv.exists():
            for r in csv.reader(open(tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
                p = root / split / r[1]
                if p.exists():
                    rows.append({"audio": str(p), "text": r[2], "speaker": ""})
    return rows


def cv(root: Path) -> list[dict]:
    import csv
    rows = []
    vt = root / "validated.tsv"
    if not vt.exists():
        return rows
    files = {p.name: p for p in root.rglob("*") if p.suffix.lower() in AUDIO}
    for r in csv.DictReader(open(vt, encoding="utf-8"), delimiter="\t"):
        p = files.get(Path(r["path"]).name)
        if p:
            rows.append({"audio": str(p), "text": r["sentence"], "speaker": f"cv:{r['client_id'][:12]}"})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--ours-val", type=Path, default=None)
    ap.add_argument("--corpora", type=Path, default=Path(os.environ.get("VOICE_HOME", ".")) / "corpora")
    ap.add_argument("--hours", default="ksc2_live=all,ksc2_read=40,kazakhtts2=10,fleurs_kk=all,cv_kk=all,fleurs_ru=all")
    ap.add_argument("--spk-cap", type=float, default=0.03)
    ap.add_argument("--ref-frac", type=float, default=0.40, help="доля строк train с опорой (ref_audio)")
    ap.add_argument("--tail-frac", type=float, default=0.30, help="из всех строк — доля с опорой = хвост предыдущей фразы")
    ap.add_argument("--tail-min", type=float, default=3.0)
    ap.add_argument("--tail-max", type=float, default=8.0)
    ap.add_argument("--ref-clip-max", type=float, default=20.0, help="клип с опорой не длиннее, с")
    ap.add_argument("--ksc2-live-train", type=int, default=0, help="1 — брать и Train живых доменов KSC2 (нет пунктуации)")
    ap.add_argument("--expect-hours", type=float, default=0.0, help="план часов train; 0 — наши + заказанные часы корпусов")
    ap.add_argument("--max-read", type=float, default=0.30)
    ap.add_argument("--min-ru", type=float, default=0.03)
    ap.add_argument("--max-glued", type=float, default=0.02)
    ap.add_argument("--style-frac", type=float, default=0.15)
    ap.add_argument("--min-s", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--dup-questions", type=int, default=1, help="строки наших с «?» повторить столько раз (11.10: 2)")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    want = {k: (None if v == "all" else float(v)) for k, v in (x.split("=") for x in a.hours.split(","))}
    K = a.corpora / "ksc2" / "ISSAI_KSC2"
    loaders = {
        "ksc2_live": lambda: stem_pairs([k / s / d for k in (K, a.corpora / "ksc2_live" / "ISSAI_KSC2")
                                         for s in (("Test", "Train") if a.ksc2_live_train else ("Test",))
                                         for d in ("podcasts", "radio", "talkshow", "tv_news")], "ksc2"),
        "ksc2_read": lambda: stem_pairs([K / "Train" / "crowdsourced", K / "Test" / "crowdsourced"], "ksc2"),
        "kazakhtts2": lambda: stem_pairs([a.corpora / "kazakhtts2"], "kazakhtts2"),
        "fleurs_kk": lambda: fleurs(a.corpora / "fleurs_kk"),
        "cv_kk": lambda: cv(a.corpora / "cv_kk"),
        "fleurs_ru": lambda: fleurs(a.corpora / "fleurs_ru"),
    }
    ours = [json.loads(l) for l in open(a.ours, encoding="utf-8") if l.strip()]
    for r in ours:
        r.update(src="ours", dataset_id=0)
    ours = [r for r in ours if a.min_s <= r["duration"] <= a.max_s]
    print(f"ours: {len(ours)} фраз, {sum(r['duration'] for r in ours) / 3600:.1f} ч", flush=True)
    if a.dup_questions > 1:   # 11.10: живых вопросов мало (3,4 % предложений) — строки с «?» повторяем
        qrows = [r for r in ours if "?" in (r.get("text") or "")]
        for _ in range(a.dup_questions - 1):
            ours += [dict(r, dup_q=1) for r in qrows]
        print(f"вопросы: строк с «?» {len(qrows)} ({sum(r['duration'] for r in qrows) / 3600:.1f} ч) × {a.dup_questions}", flush=True)

    def fill(r):
        if not r.get("duration"):
            r["duration"] = round(dur_of(r["audio"]), 3)
        return r
    picked = list(ours)
    dropped = defaultdict(int)
    sel_h = {"ours": sum(r["duration"] for r in ours) / 3600}
    with ThreadPoolExecutor(a.workers) as ex:
        for name, want_h in want.items():
            rows = loaders[name]()
            rnd.shuffle(rows)
            lim = want_h * 3600 if want_h is not None else float("inf")
            s, i, step = 0.0, 0, 2000
            while s < lim and i < len(rows):
                for r in ex.map(fill, rows[i:i + step]):
                    if s >= lim:
                        break
                    if not (a.min_s <= r["duration"] <= a.max_s):
                        dropped[f"{name}:длина"] += 1
                        continue
                    t, probs = normalize(r["text"])
                    if probs:
                        dropped[f"{name}:{probs[0]}"] += 1
                        continue
                    lang = "ru" if name == "fleurs_ru" else "kk"
                    r.update(text=t, dataset_id=DSID[name], src=name, lang=lang)
                    picked.append(r)
                    s += r["duration"]
                i += step
            print(f"{name}: найдено {len(rows)}, отобрано {s / 3600:.2f} ч", flush=True)
            sel_h[name] = s / 3600
    total = sum(r["duration"] for r in picked)
    cap = a.spk_cap * total
    per = defaultdict(float)
    mix = []
    for r in sorted(picked, key=lambda _: rnd.random()):
        spk = r.get("speaker") or ""
        if spk and per[spk] + r["duration"] > cap:
            dropped["потолок_диктора"] += 1
            continue
        per[spk] += r["duration"]
        mix.append(r)
    # 09.10: склеенные слова («Күштісосын»): редкое длинное слово = два частых слова подряд без пробела → строка вон
    def words(t: str) -> list[str]:
        return re.findall(r"[^\W\d_]+", t.lower())

    freq = Counter(w for r in mix for w in words(r["text"]))

    def glued(t: str) -> str:
        for w in words(t):
            if len(w) < 8 or freq[w] > 2:
                continue
            for k in range(4, len(w) - 3):
                if freq[w[:k]] >= 30 and freq[w[k:]] >= 30:
                    return f"{w} = {w[:k]} + {w[k:]}"
        return ""

    glued_ex, kept_g = [], []
    for r in mix:
        g = glued(r["text"])
        if g:
            dropped[f"{r['src']}:склеенные_слова"] += 1
            if len(glued_ex) < 40:
                glued_ex.append(f"{r['src']}: {g}")
            continue
        kept_g.append(r)
    glued_share = (len(mix) - len(kept_g)) / max(1, len(mix))
    mix = kept_g

    # опора: хвост предыдущей фразы (до tail_frac всех строк), остальное до ref_frac — другая фраза того же диктора
    by_spk = defaultdict(list)
    for r in mix:
        if r.get("speaker") and 3 <= r["duration"] <= 10:
            by_spk[r["speaker"]].append(r)
    N = len(mix)
    elig_tail = [r for r in mix if r.get("prev_audio") and (r.get("prev_duration") or 0) >= a.tail_min
                 and r["duration"] <= a.ref_clip_max]
    p_tail = min(1.0, a.tail_frac * N / max(1, len(elig_tail)))
    for r in elig_tail:
        if rnd.random() < p_tail:
            r["_tail_of"] = r["prev_audio"]
    n_tail_plan = sum(1 for r in mix if "_tail_of" in r)
    rest = [r for r in mix if "_tail_of" not in r and r["duration"] <= a.ref_clip_max and len(by_spk.get(r.get("speaker") or "", [])) >= 2]
    p_other = min(1.0, max(0, int(a.ref_frac * N) - n_tail_plan) / max(1, len(rest)))
    for r in rest:
        if rnd.random() >= p_other:
            continue
        c = by_spk[r["speaker"]]
        for _ in range(6):
            x = c[rnd.randrange(len(c))]
            if x is not r and x["duration"] + r["duration"] <= 30:
                r["ref_audio"], r["ref_duration"] = x["audio"], x["duration"]
                break
    print(f"опора: хвостов в плане {n_tail_plan} (из {len(elig_tail)} возможных), другая фраза {sum(1 for r in mix if r.get('ref_audio'))}; "
          f"склеенных слов {glued_share * 100:.2f} %", flush=True)
    for r in mix:
        if rnd.random() < a.style_frac:
            r["text"] = ("(дикторское чтение)" if r["src"] in READING else "(живая разговорная речь)") + r["text"]
    a.out.mkdir(parents=True, exist_ok=True)
    adir = a.out / "audio"
    adir.mkdir(exist_ok=True)
    tdir = a.out / "audio_tail"
    tdir.mkdir(exist_ok=True)

    def conv(src: str):
        """→ (путь, длительность) или None. Наши — как есть. Корпуса: обрезка тишины ПО УРОВНЮ САМОГО ФАЙЛА (кадр 10 мс громче
        max(p90 − 35, шум + 8) дБ), поля 0,15 / 0,25 с, затухание 15 мс. 08.10: ffmpeg silenceremove с порогом −45 дБFS съедал речь
        тихих записей FLEURS — 435 файлов вышли пустыми (длительность «2^63 отсчётов»), ещё ~440 обрезаны до обрывка (338 fleurs_kk > 25 симв/с)."""
        if "/v2/clean/" in src or "/v2/clean_v3/" in src:      # наши уже 16 кГц FLAC (recut_v2: поля из паузы + затухание)
            # (соглашение о путях: process_v2/recut_v2 запускаются с --out <...>/v2)
            return src, None
        dst = adir / (re.sub(r"[^\w.-]", "_", Path(src).parent.name) + "__" + Path(src).stem + ".flac")
        import numpy as np
        import soundfile as sf
        if dst.exists():
            y, _ = sf.read(str(dst), dtype="float32")
            return (str(dst), len(y) / 16000) if len(y) >= 8000 else None
        sys.path.insert(0, str(HERE.parent))
        import voicelab.audio as VA
        try:
            x = VA.load(src, 16000)
        except Exception:  # noqa: BLE001
            return None
        n = len(x) // 160
        if n < 20:
            return None
        db = 10 * np.log10((x[: n * 160].reshape(n, 160).astype(np.float64) ** 2).mean(1) + 1e-12)
        db = np.convolve(db, np.ones(3) / 3, mode="same")
        ref, noise = float(np.percentile(db, 90)), float(np.percentile(db, 10))
        loud = np.where(db > max(ref - 35.0, noise + 8.0))[0]
        if not len(loud):
            return None
        a_ = max(0, loud[0] * 160 - 2400)
        b_ = min(len(x), (loud[-1] + 1) * 160 + 4000)
        y = x[a_:b_].astype(np.float32)
        if len(y) < 8000:
            return None
        k = 240
        w = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))).astype(np.float32)
        y[:k] *= w
        y[-k:] *= w[::-1]
        sf.write(str(dst), np.clip(y, -1, 1), 16000, format="FLAC", subtype="PCM_16")
        return str(dst), len(y) / 16000

    def make_tail(src: str):
        """Хвост предыдущей фразы 3–8 с → (путь, длительность, разрез_в_паузе) или None. Короче tail_max — фраза целиком
        (у неё уже поля и затухание recut_v2); длиннее — разрез в самой тихой точке окна [L − 8, L − 3] с, нарастание 15 мс."""
        import numpy as np
        import soundfile as sf
        dst = tdir / (Path(src).stem + "__tail.flac")
        try:
            x, sr = sf.read(src, dtype="float32")
        except Exception:  # noqa: BLE001
            return None
        if x.ndim > 1:
            x = x.mean(1)
        L = len(x) / max(sr, 1)
        if sr != 16000 or L < a.tail_min:
            return None
        if L <= a.tail_max:
            return src, L, True
        n = len(x) // 160
        db = 10 * np.log10((x[: n * 160].reshape(n, 160).astype(np.float64) ** 2).mean(1) + 1e-12)
        db = np.convolve(db, np.ones(5) / 5, mode="same")
        lo, hi = int((L - a.tail_max) * 100), int((L - a.tail_min) * 100)
        k = lo + int(np.argmin(db[lo:hi]))
        pause = bool(db[k] < float(np.percentile(db, 90)) - 25.0)
        y = x[k * 160:].copy()
        f = 240
        y[:f] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f))).astype(np.float32)
        if not dst.exists():
            sf.write(str(dst), np.clip(y, -1, 1), 16000, format="FLAC", subtype="PCM_16")
        return str(dst), len(y) / 16000, pause

    srcs = sorted({r["audio"] for r in mix} | {r["ref_audio"] for r in mix if r.get("ref_audio")})
    tails_src = sorted({r["_tail_of"] for r in mix if r.get("_tail_of")})
    m, tails = {}, {}
    with ThreadPoolExecutor(a.workers) as ex:
        for s_, d_ in zip(srcs, ex.map(conv, srcs)):
            m[s_] = d_
        for s_, d_ in zip(tails_src, ex.map(make_tail, tails_src)):
            tails[s_] = d_
    kept = []
    n_tail = n_tail_pause = 0
    for r in mix:
        c = m.get(r["audio"])
        if c is None:
            dropped[f"{r['src']}:пусто_после_обрезки"] += 1
            continue
        r["audio"] = c[0]
        if c[1] is not None:
            r["duration"] = round(c[1], 3)
        body = re.sub(r"^\([^)]*\)", "", r["text"])
        if not (a.min_s <= r["duration"] <= a.max_s) or len(body) / r["duration"] > 25:
            dropped[f"{r['src']}:обрывок_или_быстрее_25_симв_с"] += 1
            continue
        t_ = tails.get(r.pop("_tail_of", None) or "")
        if t_ is not None:
            if r["duration"] + t_[1] <= 30:
                r["ref_audio"], r["ref_duration"], r["ref_kind"] = t_[0], round(t_[1], 3), "tail"
                n_tail += 1
                n_tail_pause += int(t_[2])
        elif r.get("ref_audio"):
            cr = m.get(r["ref_audio"])
            if cr is None or r["duration"] + (cr[1] if cr[1] is not None else r["ref_duration"]) > 30:
                r.pop("ref_audio", None)
                r.pop("ref_duration", None)
            else:
                r["ref_audio"], r["ref_kind"] = cr[0], "same_speaker"
                if cr[1] is not None:
                    r["ref_duration"] = round(cr[1], 3)
        kept.append(r)
    mix = kept
    # val: отложенные каналы наших + до 100 строк корпусов
    val = []
    if a.ours_val and a.ours_val.exists():
        val = [dict(json.loads(l), src="ours", dataset_id=0) for l in open(a.ours_val, encoding="utf-8") if l.strip()]
        val = [r for r in val if not glued(r["text"])]
    corp = [r for r in mix if r["src"] != "ours"]
    rnd.shuffle(corp)
    vc = corp[: min(100, len(corp) // 100)]
    vids = {id(r) for r in vc}
    train = [r for r in mix if id(r) not in vids]
    rnd.shuffle(train)
    keys = ["audio", "text", "duration", "dataset_id", "ref_audio", "ref_duration", "speaker", "lang", "src"]
    for fn, part in (("train.jsonl", train), ("val.jsonl", val[:300] + vc)):
        with open(a.out / fn, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({k: r.get(k) for k in keys if r.get(k) not in (None, "")}, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- сводка + жёсткие проверки (по файлам на диске)
    import soundfile as sf

    def measure(r: dict) -> list[str]:
        errs = []
        try:
            i = sf.info(r["audio"])
            d = i.frames / i.samplerate
        except Exception:  # noqa: BLE001
            return ["нет_или_битый_файл"]
        if i.samplerate != 16000:
            errs.append("не_16_кГц")
        if abs(d - r["duration"]) > 0.05:
            errs.append("длительность_не_как_в_манифесте")
        if not (a.min_s - 0.01 <= d <= a.max_s + 0.01):
            errs.append("длина_вне_1_25_с")
        if len(re.sub(r"^\([^)]*\)", "", r["text"])) / max(d, 1e-3) > 25:
            errs.append("быстрее_25_симв_с")
        if r.get("ref_audio"):
            try:
                j = sf.info(r["ref_audio"])
                rd = j.frames / j.samplerate
                if d + rd > 30.01:
                    errs.append("клип_плюс_опора_больше_30_с")
                if rd < 1.0:
                    errs.append("опора_короче_1_с")
            except Exception:  # noqa: BLE001
                errs.append("нет_файла_опоры")
        return errs

    bad = Counter()
    bad_ex = []
    with ThreadPoolExecutor(max(8, a.workers)) as ex:
        for r, errs in zip(train, ex.map(measure, train)):
            for e in errs:
                bad[e] += 1
                if len(bad_ex) < 20:
                    bad_ex.append(f"{e}: {r['audio']}")
    st = defaultdict(lambda: [0, 0.0])
    for r in train:
        st[(r["src"], r["lang"])][0] += 1
        st[(r["src"], r["lang"])][1] += r["duration"]
    H = sum(v[1] for v in st.values()) / 3600
    read_h = sum(v[1] for (s_, _), v in st.items() if s_ in READING) / 3600
    ru_h = sum(v[1] for (_, l), v in st.items() if l == "ru") / 3600
    mix_h = sum(v[1] for (_, l), v in st.items() if l == "mix") / 3600
    n_ref = sum(1 for r in train if r.get("ref_audio"))
    n_tail_tr = sum(1 for r in train if r.get("ref_kind") == "tail")
    plan_h = a.expect_hours or sum(sel_h.values())
    src_h = defaultdict(float)
    for (s_, _), v in st.items():
        src_h[s_] += v[1] / 3600
    lines = [f"смесь {a.out.name}: {len(train)} фраз, {H:.1f} ч (план {plan_h:.1f} ч); чтение {read_h / H * 100:.1f} %, "
             f"живая {100 - read_h / H * 100:.1f} %, ru {ru_h / H * 100:.1f} %, mix {mix_h / H * 100:.1f} %; опора {n_ref} строк "
             f"({n_ref / max(1, len(train)) * 100:.0f} %; хвост предыдущей фразы {n_tail_tr} = {n_tail_tr / max(1, len(train)) * 100:.0f} %, "
             f"разрез хвоста в паузе {n_tail_pause / max(1, n_tail) * 100:.0f} %); val {len(val[:300]) + len(vc)}"]
    lines += [f"  {s_:10s} {l}: {n:7d} фраз {sec / 3600:7.2f} ч ({sec / 36 / max(H, 1e-9):.1f} %)" for (s_, l), (n, sec) in sorted(st.items())]
    lines.append("план → факт по источникам, ч: " + ", ".join(f"{k} {sel_h.get(k, 0):.1f} → {src_h.get(k, 0):.1f}"
                                                         for k in sorted(set(sel_h) | set(src_h))))
    lines.append("выброшено: " + ", ".join(f"{k} {v}" for k, v in sorted(dropped.items())))
    top = max(per.items(), key=lambda kv: kv[1] if kv[0] else 0, default=("", 0))
    lines.append(f"крупнейший диктор {top[0]}: {top[1] / 3600:.2f} ч ({top[1] / max(total, 1) * 100:.1f} %)")
    lines.append(f"склеенные слова: {glued_share * 100:.2f} % строк выброшено; примеры: " + "; ".join(glued_ex[:12]))
    fails = []
    if bad:
        fails.append("файлы: " + ", ".join(f"{k} {v}" for k, v in bad.items()) + " | " + "; ".join(bad_ex[:5]))
    if abs(H / max(plan_h, 1e-9) - 1) > 0.10:
        fails.append(f"часы {H:.1f} вне ±10 % от плана {plan_h:.1f}")
    if read_h / H > a.max_read:
        fails.append(f"чтение {read_h / H * 100:.1f} % > {a.max_read * 100:.0f} %")
    if ru_h / H < a.min_ru:
        fails.append(f"русский {ru_h / H * 100:.1f} % < {a.min_ru * 100:.0f} %")
    if glued_share > a.max_glued:
        fails.append(f"склеенных слов {glued_share * 100:.1f} % > {a.max_glued * 100:.0f} %")
    if len(train) < 1000:
        fails.append(f"строк train всего {len(train)}")
    lines.append("ПРОВЕРКИ ПРОЙДЕНЫ (файлы по отсчётам, ≤ 25 симв/с у всех, клип + опора ≤ 30 с, часы ±10 %, доли)" if not fails
                 else "ПРОВЕРКА НЕ ПРОЙДЕНА: " + " || ".join(fails))
    (a.out / "stats.txt").write_text("\n".join(lines) + "\n")
    (a.out / "stats.json").write_text(json.dumps(dict(
        rows=len(train), hours=round(H, 2), plan_hours=round(plan_h, 2), read_share=round(read_h / H, 4), ru_share=round(ru_h / H, 4),
        mix_share=round(mix_h / H, 4), ref_share=round(n_ref / max(1, len(train)), 4), tail_share=round(n_tail_tr / max(1, len(train)), 4),
        tail_cut_in_pause=round(n_tail_pause / max(1, n_tail), 4), glued_share=round(glued_share, 5),
        by_source={f"{s_}/{l}": dict(rows=n, hours=round(sec / 3600, 2)) for (s_, l), (n, sec) in st.items()},
        planned=sel_h, dropped=dict(dropped), file_errors=dict(bad), fails=fails), ensure_ascii=False, indent=1))
    # выборка для ушей: по 20 строк на источник + 20 строк с хвостом-опорой
    smp = []
    for s_ in sorted(src_h):
        xs = [r for r in train if r["src"] == s_]
        smp += rnd.sample(xs, min(20, len(xs)))
    xs = [r for r in train if r.get("ref_kind") == "tail"]
    smp += [dict(r, src=f"{r['src']}+хвост") for r in rnd.sample(xs, min(20, len(xs)))]
    with open(a.out / "listen_sample.jsonl", "w", encoding="utf-8") as f:
        for r in smp:
            f.write(json.dumps({k: r.get(k) for k in ("src", "audio", "text", "duration", "ref_audio", "ref_duration", "lang")},
                               ensure_ascii=False) + "\n")
    print("\n".join(lines))
    if fails:
        sys.exit(3)


if __name__ == "__main__":
    main()
