#!/usr/bin/env python3
"""Сводка пробы A/B/C по готовым JSON (модели не грузит — ноут): <probe>/out/gigaam.json (gigaam_eval sets) и sim_dnsmos.json.

  python3 scripts/compare_probe.py voxcpm2/runs/<probe>     → печать + <probe>/summary.md

Таблица 1 — финалы (v0, A 6164, B 2000, C 2000), среднее по сидам 1 и 2: CER kk (ср/мед), провалы (CER > 10 %), нет последнего слова,
ошибки особых букв (ә ғ қ ң ө ұ ү һ і) и «обрусевшие», mix и ru CER, длительность на символ (выбросы = не остановилась/обрыв), SIM к
казахскому эталону, DNSMOS OVRL; разброс сид 1 ↔ сид 2 — оценка шума. Таблица 2 — kk CER по категориям phrases_v2. Таблица 3 —
середины (A 1500, B 1000, C 1000) против финалов на kk-подмножестве. Таблица 4 — эталон ru/kk × cfg у A 6164 на «особых буквах».
Правило решения: C заметно чище по акценту, ru CER хуже не больше 1 п.п., Voice Design жив → полное; B ≈ C и оба
лучше A → расширенная LoRA; A ≈ B ≈ C → большой v3 не запускаем. Цифры — только половина ответа, вторая — слепое прослушивание.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PH = {p["id"]: p for p in json.loads((ROOT / "eval/phrases_v2.json").read_text())["phrases"]}


def agg(items: list[dict]) -> dict:
    out = {}
    for lang in ("kk", "mix", "ru"):
        xs = [v for v in items if v["lang"] == lang]
        if not xs:
            continue
        c = [v["cer"] for v in xs]
        n_sp = sum(v.get("sp_n", 0) for v in xs)
        out[lang] = dict(n=len(xs), cer=st.mean(c), med=st.median(c), fail=sum(x > 0.10 for x in c) / len(c),
                         last=sum(not v["last_ok"] for v in xs), sp_err=sum(v.get("sp_err", 0) for v in xs) / max(1, n_sp),
                         sp_rus=sum(v.get("sp_rus", 0) for v in xs) / max(1, n_sp),
                         spc=st.median(v["dur"] / max(1, len(v["text"])) for v in xs))
    return out


def main():
    pr = Path(sys.argv[1]).expanduser()
    g = json.loads((pr / "out/gigaam.json").read_text())["sets"]
    sd_f = pr / "out/sim_dnsmos.json"
    sd = json.loads(sd_f.read_text())["versions"] if sd_f.exists() else {}
    items = {name: [dict(v, id=k) for k, v in s["items"].items()] for name, s in g.items()}
    lines = ["# Проба A/B/C — сводка (GigaAM CTC, независимый ASR; эталон — казахский ref_kk)", ""]

    def f(x, pct=True):
        return "—" if x is None else (f"{x * 100:.1f} %" if pct else f"{x:.3f}")
    # 1) финалы
    lines += ["## 1. Финалы, среднее по сидам 1 и 2", "",
              "| версия | kk CER ср / мед | kk провалы | нет посл. слова | особые ош / рус | mix CER | ru CER | с/симв kk | SIM | OVRL | шум s1↔s2 kk CER |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in ("v0", "A6164", "B2000", "C2000"):
        seeds = [f"{v}_s1", f"{v}_s2"]
        got = [agg(items[s]) for s in seeds if s in items]
        if not got:
            continue
        m = lambda lang, k: st.mean(x[lang][k] for x in got if lang in x) if any(lang in x for x in got) else None  # noqa: E731
        sims = [sd[s]["ref"].get("sim_mean") for s in seeds if s in sd and "ref" in sd[s]]
        ovrl = [sd[s]["ref"].get("ovrl") for s in seeds if s in sd and "ref" in sd[s]]
        noise = abs(got[0]["kk"]["cer"] - got[1]["kk"]["cer"]) if len(got) == 2 else None
        last = sum(x[l_]["last"] for x in got for l_ in x)
        lines.append(f"| {v} | {f(m('kk', 'cer'))} / {f(m('kk', 'med'))} | {f(m('kk', 'fail'))} | {last} | "
                     f"{f(m('kk', 'sp_err'))} / {f(m('kk', 'sp_rus'))} | {f(m('mix', 'cer'))} | {f(m('ru', 'cer'))} | "
                     f"{m('kk', 'spc'):.3f} | {f(st.mean(sims), False) if sims else '—'} | {f(st.mean(ovrl), False) if ovrl else '—'} | {f(noise)} |")
    # 2) по категориям
    cats = ["sp", "q", "num", "short", "long", "bot", "emo", "brand", "dialog"]
    lines += ["", "## 2. kk CER по категориям (сид 1)", "", "| версия | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for v in ("v0_s1", "A6164_s1", "B2000_s1", "C2000_s1"):
        if v not in items:
            continue
        row = []
        for c in cats:
            xs = [x["cer"] for x in items[v] if PH.get(x["id"], {}).get("cat") == c and x["lang"] in ("kk", "mix")]
            row.append(f"{st.mean(xs) * 100:.1f}" if xs else "—")
        lines.append(f"| {v} | " + " | ".join(row) + " |")
    # 3) середины против финалов на подмножестве
    lines += ["", "## 3. Середина против финала (kk-подмножество sp,q,num,short,bot; сид 1)", "", "| пара | kk CER | особые ош | провалы |", "|---|---|---|---|"]
    for mid, fin in (("A1500_s1", "A6164_s1"), ("B1000_s1", "B2000_s1"), ("C1000_s1", "C2000_s1")):
        if mid not in items or fin not in items:
            continue
        ids = {x["id"] for x in items[mid]}
        for nm in (mid, fin):
            a = agg([x for x in items[nm] if x["id"] in ids]).get("kk", {})
            lines.append(f"| {nm} | {f(a.get('cer'))} | {f(a.get('sp_err'))} | {f(a.get('fail'))} |")
    # 4) эталон × cfg
    lines += ["", "## 4. A 6164: эталон ru/kk × cfg (20 фраз «особые буквы», сид 1)", "", "| эталон | cfg | kk CER | особые ош / рус | провалы | SIM к ref_kk |", "|---|---|---|---|---|---|"]
    for r in ("ru", "kk"):
        for c in ("1.5", "2.0", "2.5"):
            nm = f"A6164_ref{r}_cfg{c}"
            if nm not in items:
                continue
            a = agg(items[nm]).get("kk", {})
            sim = sd.get(nm, {}).get("ref", {}).get("sim_mean")
            lines.append(f"| {r} | {c} | {f(a.get('cer'))} | {f(a.get('sp_err'))} / {f(a.get('sp_rus'))} | {f(a.get('fail'))} | {f(sim, False) if sim else '—'} |")
    # 5) диалоги: по одной реплике против контекстного режима (реплика i с репликой i−1 как опорой)
    lines += ["", "## 5. Диалоги (20 реплик): отдельно против контекста (сид 1)", "", "| версия | CER отдельно | CER в контексте | провалы в контексте | нет посл. слова |",
              "|---|---|---|---|---|"]
    for v in ("v0", "A6164", "B2000", "C2000"):
        a_ = [x for x in items.get(f"{v}_s1", []) if PH.get(x["id"], {}).get("cat") == "dialog"]
        b_ = items.get(f"{v}_dlg", [])
        if not b_:
            continue
        ca = st.mean(x["cer"] for x in a_) if a_ else None
        cb = st.mean(x["cer"] for x in b_)
        lines.append(f"| {v} | {f(ca)} | {f(cb)} | {f(sum(x['cer'] > 0.10 for x in b_) / len(b_))} | {sum(not x['last_ok'] for x in b_)} |")
    lines += ["", "Шум: 170 фраз × 2 сида; разница меньше колонки «шум s1↔s2» — не различие. Решение — вместе со слепым прослушиванием."]
    txt = "\n".join(lines)
    (pr / "summary.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
