#!/usr/bin/env python3
"""WER/CER расшифровки против эталонных текстов (refs.tsv: имя<TAB>текст). Нормализация: нижний регистр, без пунктуации."""
import csv
import re
import sys
from pathlib import Path

D = Path(__file__).resolve().parent


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower().replace("ё", "е"))).strip()


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


refs = dict(l.rstrip("\n").split("\t", 1) for l in open(D / "refs.tsv", encoding="utf-8") if "\t" in l)
rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else D / "voxcpm_test.asr.csv", encoding="utf-8")))
we = wn = ce = cn = 0
for r in rows:
    name = Path(r["file"]).stem
    if name not in refs:
        continue
    ref, hyp = norm(refs[name]), norm(r["text"])
    w, c = lev(ref.split(), hyp.split()), lev(ref, hyp)
    we, wn, ce, cn = we + w, wn + len(ref.split()), ce + c, cn + len(ref)
    print(f"{name:12s} WER {w / max(1, len(ref.split())):5.0%} CER {c / max(1, len(ref)):5.0%} ok={r['ok']} {r['why']}\n   эталон: {refs[name]}\n   ASR:    {r['text']}")
print(f"\nИТОГО {len(rows)} фраз: WER {we / max(1, wn):.1%}, CER {ce / max(1, cn):.1%}")
