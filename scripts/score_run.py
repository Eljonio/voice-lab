#!/usr/bin/env python3
"""Честная оценка прогона независимым ASR (GigaAM-Multilingual large_ctc) — все чекпоинты за одну загрузку модели.

  .venv-asr/bin/python scripts/score_run.py <RUN> [--extra v0hf=<папка с {ref,design}/<id>.wav>]

Что озвучено прогоном (remote_job.sh) и что оцениваем:
  eval/base/{ref,design}      — исходная модель на наборе eval/phrases.json (тот же сервер и сид, что у чекпоинтов);
  base/                       — исходная модель, фразы synth_baseline (report.json);
  step_N/eval/{ref,design}    — чекпоинт N на наборе eval (design — только у последнего);
  step_N/samples/             — чекпоинт N, фразы synth_baseline;
  --extra NAME=DIR            — сторонняя версия с DIR/{ref,design}/<id>.wav (например v0 из песочницы HF).
Сводка на версию: набор eval (ref) + samples вместе по языкам kk/mix/ru: CER ср/медиана, доля провалов (CER > 10 %),
WER, «нет последнего слова». Результат: eval/results/<RUN>_gigaam.json (по фразам + сводки).
⚠ Фраз мало (kk 11, mix 4, ru 4 на чекпоинт, один сид) — разница в 1–2 % CER между чекпоинтами = шум.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gigaam_eval as G  # noqa: E402

ROOT = G.ROOT


def pooled(items: list[dict]) -> dict:
    s = {}
    for lang in ("kk", "mix", "ru"):
        xs = [v for v in items if v["lang"] == lang]
        if not xs:
            continue
        c = [v["cer"] for v in xs]
        s[lang] = dict(n=len(xs), cer_mean=round(st.mean(c), 4), cer_median=round(st.median(c), 4),
                       fail=round(sum(x > G.FAIL for x in c) / len(c), 3), wer_mean=round(st.mean(v["wer"] for v in xs), 4),
                       last_missing=sum(not v["last_ok"] for v in xs), **G.special_summary(xs))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--extra", action="append", default=[], help="NAME=DIR, где DIR/{ref,design}/<id>.wav")
    ap.add_argument("--phrases", type=Path, default=ROOT / "eval/phrases.json")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_dir() else ROOT / "voxcpm2/runs" / a.run
    ph = {p["id"]: p for p in json.loads(a.phrases.read_text())["phrases"]}

    # версия → {вид → папка}; вид: ref/design (набор eval) или samples (report.json)
    versions: dict[str, dict[str, Path]] = {}
    for spec in a.extra:
        name, d = spec.split("=", 1)
        versions[name] = {m: Path(d).expanduser() / m for m in ("ref", "design")}
    versions["base"] = {"ref": run / "eval/base/ref", "design": run / "eval/base/design", "samples": run / "base"}
    for s in sorted(run.glob("step_*"), key=lambda p: int(re.sub(r"\D", "", p.name))):
        n = int(re.sub(r"\D", "", s.name))
        versions[f"step_{n}"] = {"ref": s / "eval/ref", "design": s / "eval/design", "samples": s / "samples"}
    if (run / "design/report.json").exists():
        versions["design_variants"] = {"samples": run / "design"}

    asr = G.ASR()
    res = {"model": f"{G.MODEL_ID}@large_ctc:{G.REVISION[:8]}", "run": run.name, "made": time.strftime("%Y-%m-%d %H:%M"),
           "fail_threshold": G.FAIL, "versions": {}}
    for name, kinds in versions.items():
        out = {}
        for kind, d in kinds.items():
            if not d.is_dir():
                continue
            items = G.items_from_dir(d, ph if kind != "samples" else None)
            if kind == "samples" and name == "design_variants":  # 24 тембра: русская фраза с описанием голоса
                for it in items:
                    it["lang"] = G.lang_of("x", it["text"])
            if items:
                r = G.score(asr, items)
                out[kind] = dict(dir=str(d), summary=G.summary(r), items=r)
        if not out:
            continue
        main_items = [v for k in ("ref", "samples") if k in out for v in out[k]["items"].values()]
        res["versions"][name] = dict(kinds=out, pooled=pooled(main_items))
        p = res["versions"][name]["pooled"]
        line = "  ".join(f"{lg} n={v['n']:2} CER ср {v['cer_mean']*100:5.1f} мед {v['cer_median']*100:4.1f} пров {v['fail']*100:3.0f}%"
                         f" WER {v['wer_mean']*100:5.1f} нет посл {v['last_missing']}"
                         + (f" особ {v['sp_err']*100:4.1f}%/рус {v['sp_rus']*100:4.1f}%" if lg != "ru" else "") for lg, v in p.items())
        print(f"{name:16} {line}", flush=True)
    out = a.out or ROOT / "eval/results" / f"{run.name}_gigaam.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"→ {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
