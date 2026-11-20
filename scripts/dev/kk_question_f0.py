#!/usr/bin/env python3
"""Мелодия конца вопроса (11.10.2026): живые вопросы и утверждения (v1, kazakh-whisper поставил «?»/«.») против вопросов Аружан
(v3 шаг 6067, голос d20_kk, newvoice-1009/grid). Лёгкое: librosa.pyin (DSP, не модель) по последним 3 с ≤ 200 клипов, nice.

Мера — полутона относительно медианы F0 этого же хвоста (3 с), окна от конца звонкой речи:
  last  — последние 0,2 с (последний слог: частица «ма/бе», «ғой», конец вопр. слова);
  prev  — 0,2–0,6 с до конца (слог/слово перед частицей);
  early — 0,6–1,5 с до конца;
  slope — наклон последних 0,25 с, пт/с; peak — где максимум в последних 1,5 с (0 = начало окна, 1 = конец).
  python3 scripts/dev/kk_question_f0.py --qs questions.json --out f0.json     (questions.json — вывод kk_questions.py --json)
  LIVE_MANIFEST — VoxCPM-манифест живой речи (по умолчанию voxcpm2/data/ours-flac/train.jsonl),
  NV_RUN — папка прогона синтеза с items_grid.jsonl (по умолчанию voxcpm2/runs/newvoice-1009).
"""
import json, os, re, sys, random
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import soundfile as sf
import librosa

ROOT = Path(__file__).resolve().parents[2]
NV = Path(os.environ.get("NV_RUN", ROOT / "voxcpm2/runs/newvoice-1009"))
LIVE = Path(os.environ.get("LIVE_MANIFEST", ROOT / "voxcpm2/data/ours-flac/train.jsonl"))
SR = 16000


def f0_tail(path, tail=3.0):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    x = x[-int(tail * SR):] if len(x) > tail * SR else x
    f0, vf, vp = librosa.pyin(x, fmin=65, fmax=500, sr=SR, frame_length=1024, hop_length=160)
    t = np.arange(len(f0)) * 160 / SR
    v = np.where(vf & ~np.isnan(f0))[0]
    if len(v) < 20:
        return None
    end = t[v[-1]] + 0.01
    st = 12 * np.log2(f0 / np.nanmedian(f0[v]))
    def med(a, b):
        m = (t >= end - a) & (t < end - b) & vf & ~np.isnan(f0)
        return float(np.median(st[m])) if m.sum() >= 3 else None
    last, prev, early = med(0.2, 0.0), med(0.6, 0.2), med(1.5, 0.6)
    m = (t >= end - 0.25) & vf & ~np.isnan(f0)
    slope = float(np.polyfit(t[m], st[m], 1)[0]) if m.sum() >= 5 else None
    w = (t >= end - 1.5) & vf & ~np.isnan(f0)
    peak = float((t[w][np.argmax(st[w])] - (end - 1.5)) / 1.5) if w.sum() else None
    # контур для страницы: 30 точек последних 1,5 с
    grid = np.linspace(end - 1.5, end, 30)
    cont = [None if np.isnan(y) else round(float(y), 2) for y in np.interp(grid, t[w], st[w], left=np.nan, right=np.nan)] if w.sum() > 3 else []
    return dict(last=last, prev=prev, early=early, slope=slope, peak=peak, contour=cont, f0_med=float(np.nanmedian(f0[v])))


def shape(m):
    if not m or None in (m["last"], m["prev"]):
        return "?"
    d_end = m["last"] - m["prev"]
    if d_end >= 1.5 and (m["slope"] or 0) > 0:
        return "подъём на последнем слоге"
    if m["prev"] - m["last"] >= 1.5 and m["prev"] >= (m["early"] if m["early"] is not None else -99):
        return "пик перед концом, на конце спад"
    if (m["early"] is not None) and m["last"] - m["early"] <= -2:
        return "спад к концу"
    return "ровно"


def main():
    rnd = random.Random(11)
    qs = json.load(open(sys.argv[sys.argv.index("--qs") + 1]))
    items = []
    # живые вопросы — только когда вопрос последний в клипе (хвост звука = конец вопроса)
    for q in qs:
        if q["last"]:
            items.append(dict(group="живой вопрос", type=q["type"], text=q["text"], audio=q["audio"], spk=q["spk"]))
    # живые утверждения — последняя фраза клипа с точкой, по 1 на диктора
    R = [json.loads(l) for l in open(LIVE, encoding="utf-8")]
    rnd.shuffle(R)
    seen = set()
    for r in R:
        if r["speaker"] in seen or not r["text"].rstrip().endswith(".") or "?" in r["text"] or r["duration"] < 3:
            continue
        seen.add(r["speaker"])
        items.append(dict(group="живое утверждение", type="утверждение", text=r["text"].split(".")[-2][-120:] + ".", audio=r["audio"], spk=r["speaker"]))
        if len(seen) >= 50:
            break
    # Аружан: d20_kk, 4 настройки
    I = [json.loads(l) for l in open(NV / "items_grid.jsonl", encoding="utf-8")]
    for it in I:
        if not it["key"].startswith("grid/d20_kk/"):
            continue
        t = it["text"].strip()
        last = re.findall(r"[^.!?]+[.!?]", t)[-1].strip()
        g = "Аружан вопрос" if last.endswith("?") else "Аружан утверждение" if last.endswith(".") else "Аружан приветствие/восклицание"
        f = Path(it["file"]) if Path(it["file"]).exists() else NV / it["key"]
        items.append(dict(group=g, type=it["key"].split("/")[2], text=last, audio=str(f), spk="d20_kk", pid=it["pid"]))
    out = []
    for i, it in enumerate(items):
        m = f0_tail(it["audio"])
        if m:
            it.update(m)
            it["shape"] = shape(m)
            out.append(it)
    Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    import collections
    G = collections.defaultdict(list)
    for it in out:
        key = it["group"] if it["group"] != "живой вопрос" else f"живой вопрос: {it['type'].split(' (')[0]}"
        G[key].append(it)
    print(f"{'группа':48} {'n':>3} {'last':>6} {'prev':>6} {'early':>6} {'slope':>7} {'peak':>5}  формы контура")
    for k, L in sorted(G.items()):
        f = lambda a: np.nanmedian([x[a] for x in L if x[a] is not None]) if any(x[a] is not None for x in L) else float("nan")
        sh = collections.Counter(x["shape"] for x in L)
        print(f"{k:48} {len(L):3} {f('last'):6.1f} {f('prev'):6.1f} {f('early'):6.1f} {f('slope'):7.1f} {f('peak'):5.2f}  "
              + ", ".join(f"{a} {b}" for a, b in sh.most_common()))


if __name__ == "__main__":
    main()
