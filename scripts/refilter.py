#!/usr/bin/env python3
"""Пересчёт отсева по уже посчитанным метрикам (без повторной обработки звука).

Берёт пороги из текущего Config, перерешает passed/reject_reason для каждой фразы
в work/*/result.json и переносит файлы фраз между clean/<clip>/ и work/<clip>/rejected/
(файлы отсеянных есть, если прогон шёл с --keep-rejected). Потом собирает manifest.csv.
  ./run.sh scripts/refilter.py --clean DIR --work DIR
"""
import argparse, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voicelab.pipeline import REASON_ORDER, Config, collect  # noqa: E402

def reasons(r, c):
    if r["duration"] < c.min_len:
        return ["too_short"]
    out = []
    if r["music_after"] > c.max_music_after: out.append("music_left")
    if r["sing_after"] > c.max_sing_after: out.append("singing")
    if r["spk_min_sim"] < c.min_spk_sim: out.append("multi_speaker")
    if r["dnsmos_bak"] < c.min_bak or (r["snr_db"] == r["snr_db"] and r["snr_db"] < c.min_snr): out.append("noise")
    if r["dnsmos_ovrl"] < c.min_ovrl or r["dnsmos_sig"] < c.min_sig: out.append("low_quality")
    if r["clip_ratio"] > c.max_clip: out.append("clipping")
    if r.get("lang") not in c.langs_ok: out.append("lang_other")
    return sorted(out, key=REASON_ORDER.index)

ap = argparse.ArgumentParser(); ap.add_argument("--clean", type=Path, required=True); ap.add_argument("--work", type=Path, required=True)
a = ap.parse_args(); c = Config(); changed = 0
for f in sorted(a.work.glob("*/result.json")):
    res = json.loads(f.read_text()); cid = res["clip_id"]; kept = 0.0
    for r in res["phrases"]:
        rs = reasons(r, c); was = r["passed"]; now = int(not rs)
        r["passed"], r["reject_reason"], r["reject_all"] = now, (rs[0] if rs else ""), ";".join(rs)
        name = f"{cid}_{r['idx']:03d}.wav"
        good, bad = a.clean / cid / name, a.work / cid / "rejected" / name
        if now and not was and bad.exists():
            good.parent.mkdir(parents=True, exist_ok=True); shutil.move(bad, good); r["file"] = str(good); changed += 1
        elif was and not now and good.exists():
            bad.parent.mkdir(parents=True, exist_ok=True); shutil.move(good, bad); r["file"] = str(bad); changed += 1
        kept += r["duration"] if now else 0
    res["clean_sec"], res["n_passed"] = kept, sum(p["passed"] for p in res["phrases"])
    f.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
collect(a.work, a.clean / "manifest.csv", a.clean / "clips.csv")
print(f"фраз поменяли решение: {changed}")
