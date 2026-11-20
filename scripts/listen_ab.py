#!/usr/bin/env python3
"""Слепое A/B-сравнение двух версий голоса: одна HTML-страница (file://, без сервера) + подсчёт ответов.

  python3 scripts/listen_ab.py build --a DIR_A --b DIR_B [--phrases eval/phrases_v2.json] [--ids kk_q01,kk_bot03,...]
          [--name-a v4 --name-b v5] [--out listen/ab] [--seed 1]
  python3 scripts/listen_ab.py score ANSWERS.json

DIR_A / DIR_B — папки с <id>.wav одних и тех же фраз (например, вывод voxcpm2/synth_eval.py двух чекпоинтов с одним
эталоном, сидом и cfg). Звук перекладывается во FLAC с выровненной громкостью (статическое усиление, динамику не трогаем,
чтобы громкость не подсказывала). Какая версия A, а какая B — случайно на каждой паре; ключ лежит в meta страницы и уходит
в файл ответов, поэтому подсчёт не требует отдельного файла. Слушатель отмечает «лучше A / B / одинаково» и что не так
у худшей; страница сама сохраняет ответы (localStorage) и скачивает JSON, когда всё отмечено.

Подсчёт: победы каждой версии, «одинаково», двусторонний биномиальный тест без ничьих (p = 0,5), пометки по версиям.
Цифры ASR (GigaAM) смотреть после прослушки: CTC не слышит ударение и мелодию — это решает ухо.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "scripts/listen_page.html"

WHAT = [("melody", "мелодия вопроса"), ("stress", "ударение"), ("pop", "«пуп» / щелчок"), ("letter", "буква / слово"),
        ("ending", "окончание"), ("robot", "робот / плоско")]


def level(x: np.ndarray, target_db: float = -20.0, peak_db: float = -1.0) -> np.ndarray:
    """Статическое усиление: RMS громких кадров → target_db, пик ≤ peak_db. Динамику не трогаем."""
    if x.ndim > 1:
        x = x.mean(1)
    n = 1024
    fr = x[: len(x) // n * n].reshape(-1, n) if len(x) >= n else x[None, :]
    e = (fr ** 2).mean(1) + 1e-12
    act = fr[e >= e.max() * 1e-4]
    rms = np.sqrt((act ** 2).mean() + 1e-12)
    g = 10 ** (target_db / 20) / rms
    g *= min(1.0, 10 ** (peak_db / 20) / (np.abs(x).max() * g + 1e-9))
    return (x * g).astype(np.float32)


def write_page(out: Path, data: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    tpl = TPL.read_text(encoding="utf-8")
    page = out / "index.html"
    page.write_text(tpl.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    return page


def build(a):
    import soundfile as sf
    rnd = random.Random(a.seed)
    texts = {}
    if a.phrases and a.phrases.exists():
        texts = {p["id"]: p for p in json.loads(a.phrases.read_text())["phrases"]}
    ids = [i for i in a.ids.split(",") if i] if a.ids else sorted(f.stem for f in a.a.glob("*.wav"))
    ids = [i for i in ids if (a.a / f"{i}.wav").exists() and (a.b / f"{i}.wav").exists()]
    if not ids:
        raise SystemExit("нет общих <id>.wav в двух папках")
    rnd.shuffle(ids)
    adir = a.out / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    items, key = [], {}
    for k, pid in enumerate(ids):
        iid = f"P{k + 1:02d}"
        swap = rnd.random() < 0.5
        fa, fb = (a.b, a.a) if swap else (a.a, a.b)
        key[iid] = f"A={a.name_b}" if swap else f"A={a.name_a}"
        durs = []
        for w, d in (("a", fa), ("b", fb)):
            x, sr = sf.read(d / f"{pid}.wav", dtype="float32")
            sf.write(adir / f"{iid}_{w}.flac", level(x), sr, format="FLAC", subtype="PCM_16")
            durs.append(len(x) / sr)
        p = texts.get(pid, {})
        items.append(dict(id=iid, text=p.get("text", ""), cat=p.get("cat", ""), a=f"audio/{iid}_a.flac", b=f"audio/{iid}_b.flac",
                          la="A", lb="B", dur=round(max(durs), 1), src=pid))
    data = dict(kind="cmp", ls_key=f"listen-ab-{a.name_a}-{a.name_b}", title=f"Слепое сравнение: {len(items)} пар",
                intro=("Одна и та же фраза двумя версиями модели. Какая из них A, а какая B — случайно, ключ раскроется после "
                       "подсчёта. Отметь, какая лучше, и что не так у худшей. Кнопка A→B играет обе подряд."),
                keyhint="<kbd>A</kbd> / <kbd>B</kbd> — вариант, <kbd>пробел</kbd> — A→B",
                questions=[dict(id="best", label="лучше", ab=1, req=1,
                                options=[dict(id="a", label="A"), dict(id="b", label="B"), dict(id="same", label="одинаково")]),
                           dict(id="what", label="у худшей", ab=1, multi=1,
                                options=[dict(id=i, label=l, bad=1) for i, l in WHAT] + [dict(id="ok", label="обе норм", good=1, excl=1)])],
                items=items, nums_html="",
                meta=dict(a=str(a.a), b=str(a.b), name_a=a.name_a, name_b=a.name_b, seed=a.seed, key=key))
    page = write_page(a.out, data)
    print(f"{len(items)} пар → {page}")


def binom_p(k: int, n: int) -> float:
    """Двусторонний p для k успехов из n при p = 0,5."""
    if n == 0:
        return 1.0
    pk = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(x for x in pk if x <= pk[k] + 1e-12))


def score(a):
    ans = json.loads(a.answers.read_text(encoding="utf-8"))
    meta = ans.get("meta", {})
    key, na, nb = meta.get("key", {}), meta.get("name_a", "A"), meta.get("name_b", "B")
    wins, flags = Counter(), Counter()
    for iid, r in ans.get("items", {}).items():
        best, k = r.get("best"), key.get(iid, "")
        if not best or not k:
            continue
        side_a = k.split("=", 1)[1]
        side_b = nb if side_a == na else na
        if best == "same":
            wins["same"] += 1
            continue
        win, lose = (side_a, side_b) if best == "a" else (side_b, side_a)
        wins[win] += 1
        for w in r.get("what") or []:
            if w != "ok":
                flags[(lose, w)] += 1
    n = wins[na] + wins[nb]
    print(f"{na}: {wins[na]}  {nb}: {wins[nb]}  одинаково: {wins['same']}  (без ничьих n={n}, "
          f"двусторонний биномиальный p={binom_p(wins[na], n):.3f})")
    lab = dict(WHAT)
    for v in (na, nb):
        fl = {w: c for (s, w), c in flags.items() if s == v}
        if fl:
            print(f"  у {v}, когда проиграла: " + ", ".join(f"{lab.get(w, w)} {c}" for w, c in sorted(fl.items(), key=lambda t: -t[1])))
    if ans.get("note"):
        print("  комментарий:", ans["note"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--a", type=Path, required=True)
    b.add_argument("--b", type=Path, required=True)
    b.add_argument("--name-a", default="A")
    b.add_argument("--name-b", default="B")
    b.add_argument("--phrases", type=Path, default=ROOT / "eval/phrases_v2.json")
    b.add_argument("--ids", default="")
    b.add_argument("--out", type=Path, default=ROOT / "listen/ab")
    b.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("score")
    s.add_argument("answers", type=Path)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else score(a)


if __name__ == "__main__":
    main()
