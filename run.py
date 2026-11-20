#!/usr/bin/env python3
"""Запуск конвейера очистки.

  ./run.sh --input "$VOICE_DS/raw" --clean "$VOICE_DS/clean" --work "$VOICE_DS/work"

Уже обработанные ролики (есть work/<clip>/result.json) пропускаются, поэтому
можно запускать повторно по мере докачки. В конце собирается manifest.csv
(по фразам) и clips.csv (по роликам) в папке --clean.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from voicelab.pipeline import Config, Models, clip_id_of, collect, find_inputs, load_index, process_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--clean", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--set", default="", help="метка набора в манифесте (raw, testmix ...)")
    ap.add_argument("--ref-dir", type=Path, default=None, help="эталоны чистой речи <clip>.wav")
    ap.add_argument("--separate", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--demucs", default="htdemucs")
    ap.add_argument("--whisper", default="medium")
    ap.add_argument("--demo-stems", action="store_true")
    ap.add_argument("--keep-rejected", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="обработать только эти clip_id")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ast-win", type=float, default=3.0)
    ap.add_argument("--ast-hop", type=float, default=1.5)
    ap.add_argument("--shard", default="", help="k/n: обработать каждый n-й файл начиная с k (параллельные процессы)")
    ap.add_argument("--order", type=Path, default=None, help="CSV с колонкой file: обрабатывать в этом порядке (plan.csv поднабора)")
    ap.add_argument("--until", default="", help="ЧЧ:ММ — новый ролик после этого времени не начинать (сохраняется всё готовое)")
    a = ap.parse_args()

    cfg = Config(separate=a.separate, demucs_model=a.demucs, whisper=a.whisper,
                 demo_stems=a.demo_stems, keep_rejected=a.keep_rejected,
                 ast_win=a.ast_win, ast_hop=a.ast_hop)
    files = find_inputs(a.input)
    if a.only:
        files = [f for f in files if clip_id_of(f) in set(a.only)]
    todo = [f for f in files if a.redo or not (a.work / clip_id_of(f) / "result.json").exists()]
    if a.order:
        import csv
        rank = {Path(r["file"]).stem: i for i, r in enumerate(csv.DictReader(open(a.order)))}
        todo = sorted((f for f in todo if f.stem in rank), key=lambda f: rank[f.stem])
    if a.limit:
        todo = todo[: a.limit]
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        todo = todo[k::n]
    print(f"файлов: {len(files)}, к обработке: {len(todo)}", flush=True)
    if todo:
        index = load_index(a.input)
        m = Models(cfg, a.device)
        print(f"модели загружены за {m.load_s:.1f} с", flush=True)
        deadline = None
        if a.until:
            import datetime as dt
            hh, mm = map(int, a.until.split(":"))
            now = dt.datetime.now()
            deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if deadline <= now:
                deadline += dt.timedelta(days=1)
        for k, f in enumerate(todo, 1):
            if deadline is not None:
                import datetime as dt
                if dt.datetime.now() >= deadline:
                    print(f"стоп по времени {a.until}: обработано {k - 1}/{len(todo)}, остальное — повторным запуском", flush=True)
                    break
            t = time.time()
            try:
                ref = a.ref_dir / f"{f.stem}.wav" if a.ref_dir else None
                meta = index.get(f.stem) or index.get(f.name) or index.get(f.stem.split("_")[-1]) or {}
                r = process_clip(f, m, cfg, a.clean, a.work, a.set, meta, ref)
                print(f"[{k}/{len(todo)}] {r['clip_id']}: {r['duration']:.0f} с, музыка={int(r['has_music'])}, "
                      f"фраз {r['n_passed']}/{r['n_phrases']}, чисто {r['clean_sec']:.0f} с, "
                      f"{time.time() - t:.1f} с (RTF {r['rtf']:.2f})", flush=True)
            except Exception as e:  # один битый файл не должен ронять весь прогон
                print(f"[{k}/{len(todo)}] ОШИБКА {f}: {e}", file=sys.stderr, flush=True)
                traceback.print_exc()
    res = collect(a.work, a.clean / "manifest.csv", a.clean / "clips.csv")
    print(f"манифест: {a.clean / 'manifest.csv'} ({len(res)} роликов)")


if __name__ == "__main__":
    main()
