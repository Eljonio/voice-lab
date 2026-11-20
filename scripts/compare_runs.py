#!/usr/bin/env python3
"""Сводная таблица версий по готовым JSON оценки (модели не грузит — можно на ноуте):
eval/results/<RUN>_gigaam.json (score_run.py) и <RUN>_sim_dnsmos.json (sim_dnsmos.py).

  python3 scripts/compare_runs.py v0=lora-v2-1008:base v1=lora-1006-1600:step_2950 v2-3000=lora-v2-1008:step_3000 \
      v2=lora-v2-1008:step_6164 [--md]

Колонки: CER kk/mix/ru (среднее, медиана), провалы (CER > 10 %), нет последнего слова, ошибки на особых буквах
(ә ғ қ ң ө ұ ү һ і) и доля «обрусевших» (замена русским аналогом), SIM к эталону (среднее/мин), DNSMOS OVRL.
Фраз мало (kk 11, mix 4, ru 4–5 на версию, один сид): разница меньше ~2 п.п. CER и ~0,05 SIM — шум (шаги 6163 и 6164 v2 —
одни и те же веса, а SIM 0,751 и 0,695).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gigaam_eval as G  # noqa: E402
from score_run import pooled  # noqa: E402


def sim_of(run: str, ver: str) -> dict:
    f = ROOT / "eval/results" / f"{run}_sim_dnsmos.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    v = d.get("versions", d).get(ver, {})
    return v.get("ref", v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", help="ИМЯ=RUN:ВЕРСИЯ")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    rows = []
    for spec in a.specs:
        name, rv = spec.split("=", 1)
        run, ver = rv.split(":", 1)
        d = json.loads((ROOT / "eval/results" / f"{run}_gigaam.json").read_text())
        v = d["versions"][ver]
        items = [x for k in ("ref", "samples") if k in v["kinds"] for x in G.add_special(v["kinds"][k]["items"]).values()]
        p = pooled(items)
        s = sim_of(run, ver)
        rows.append((name, p, s))
    hdr = ["версия", "kk CER ср/мед", "kk пров", "kk особые ош/рус", "mix CER", "ru CER ср/мед", "нет посл.", "SIM ср/мин", "OVRL"]
    out = []
    for name, p, s in rows:
        kk, mx, ru = p.get("kk", {}), p.get("mix", {}), p.get("ru", {})
        last = sum(x.get("last_missing", 0) for x in (kk, mx, ru))
        out.append([name, f"{kk.get('cer_mean', 0) * 100:.1f} / {kk.get('cer_median', 0) * 100:.1f} %", f"{kk.get('fail', 0) * 100:.0f} %",
                    f"{kk.get('sp_err', 0) * 100:.1f} / {kk.get('sp_rus', 0) * 100:.1f} %", f"{mx.get('cer_mean', 0) * 100:.1f} %",
                    f"{ru.get('cer_mean', 0) * 100:.1f} / {ru.get('cer_median', 0) * 100:.1f} %", str(last),
                    (f"{s['sim_mean']:.3f} / {s['sim_min']:.3f}" if "sim_mean" in s else "—"),
                    (f"{s['ovrl']:.2f}" if "ovrl" in s else "—")])
    if a.md:
        print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
        for r in out:
            print("| " + " | ".join(r) + " |")
    else:
        w = [max(len(h), *(len(r[i]) for r in out)) for i, h in enumerate(hdr)]
        print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
        for r in out:
            print("  ".join(c.ljust(w[i]) for i, c in enumerate(r)))


if __name__ == "__main__":
    main()
