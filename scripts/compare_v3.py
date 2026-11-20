#!/usr/bin/env python3
"""Сводка v3 по готовым JSON (модели не грузит): снимки v3 (runs/lora-v3-<TAG>/out) рядом с v0 и v2 из пробы (runs/probe-1010/out).
Протокол один: eval/phrases_v2.json (170 фраз), казахский эталон eval/ref_kk.wav, сиды 1 и 2, cfg 2,0.

  python3 scripts/compare_v3.py voxcpm2/runs/<RUN>   → печать + <run>/summary.md + <run>/best.txt
  PROBE_OUT=<папка out пробы A/B/C> — откуда брать v0/v2 для сравнения (по умолчанию voxcpm2/runs/probe-1010/out)

Лучший снимок по автомату: меньше доля провалов kk (CER > 10 %), потом меньше средний CER kk, при равенстве — более поздний.
Это только кандидат на слепое сравнение с v2: CER у всех у пола, решает ухо.
"""
from __future__ import annotations

import json
import os
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE = Path(os.environ.get("PROBE_OUT", ROOT / "voxcpm2/runs/probe-1010/out"))


def sets(d: Path) -> tuple[dict, dict]:
    g = json.loads((d / "gigaam.json").read_text())["sets"] if (d / "gigaam.json").exists() else {}
    s = json.loads((d / "sim_dnsmos.json").read_text())["versions"] if (d / "sim_dnsmos.json").exists() else {}
    return g, s


def agg(items: dict) -> dict:
    out = {}
    for lang in ("kk", "mix", "ru"):
        xs = [v for v in items.values() if v["lang"] == lang]
        if xs:
            c = [v["cer"] for v in xs]
            n_sp = sum(v.get("sp_n", 0) for v in xs)
            out[lang] = dict(cer=st.mean(c), fail=sum(x > 0.10 for x in c) / len(c), last=sum(not v["last_ok"] for v in xs),
                             sp=sum(v.get("sp_err", 0) for v in xs) / max(1, n_sp), short=st.mean(
                                 [v["cer"] for k, v in items.items() if k.startswith("kk_short")] or [0]))
    return out


def main():
    run = Path(sys.argv[1]).expanduser()
    g3, s3 = sets(run / "out")
    gp, sp = sets(PROBE)
    rows = []

    def add(name, g, s, keys):
        got = [agg(g[k]["items"]) for k in keys if k in g]
        if not got:
            return
        m = lambda lang, f: st.mean(x[lang][f] for x in got if lang in x)  # noqa: E731
        sims = [s[k]["ref"]["sim_mean"] for k in keys if k in s and "ref" in s[k]]
        ov = [s[k]["ref"]["ovrl"] for k in keys if k in s and "ref" in s[k]]
        rows.append(dict(name=name, kk=m("kk", "cer"), fail=m("kk", "fail"), last=sum(x[l]["last"] for x in got for l in x),
                         sp=m("kk", "sp"), short=m("kk", "short"), mix=m("mix", "cer"), ru=m("ru", "cer"),
                         sim=st.mean(sims) if sims else None, ovrl=st.mean(ov) if ov else None, n=len(got)))
    add("v0 (база)", gp, sp, ["v0_s1", "v0_s2"])
    add("v2 (LoRA r=64, 6164)", gp, sp, ["A6164_s1", "A6164_s2"])
    add("B проба (r=128, 2000)", gp, sp, ["B2000_s1", "B2000_s2"])
    snaps = sorted({int(m.group(1)) for k in g3 for m in [re.match(r"v3_(\d+)_s\d", k)] if m})
    for n in snaps:
        add(f"v3 шаг {n}", g3, s3, [f"v3_{n}_s1", f"v3_{n}_s2"])
    f = lambda x: "—" if x is None else f"{x * 100:.1f} %"  # noqa: E731
    lines = ["# v3 — сводка (GigaAM CTC; phrases_v2 × 2 сида; казахский эталон)", "",
             "| версия | kk CER | kk провалы | нет посл. слова | особые буквы | короткие реплики CER | mix CER | ru CER | SIM | OVRL |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sim = "—" if r["sim"] is None else f"{r['sim']:.3f}"
        ovrl = "—" if r["ovrl"] is None else f"{r['ovrl']:.2f}"
        lines.append(f"| {r['name']} | {f(r['kk'])} | {f(r['fail'])} | {r['last']} | {f(r['sp'])} | {f(r['short'])} | {f(r['mix'])} | "
                     f"{f(r['ru'])} | {sim} | {ovrl} |")
    v3 = [r for r in rows if r["name"].startswith("v3")]
    best = None
    if v3:
        best = min(v3, key=lambda r: (round(r["fail"], 3), round(r["kk"], 4), -int(r["name"].split()[-1])))
        lines += ["", f"Кандидат на слепое сравнение с v2 (меньше провалов, потом CER): **{best['name']}**.",
                  "Разница меньше ~2 п.п. CER — шум (у v2 разброс между сидами 2,0 п.п.); решает слепое прослушивание."]
        (run / "best.txt").write_text(best["name"].split()[-1] + "\n")
    txt = "\n".join(lines)
    (run / "summary.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
