#!/usr/bin/env python3
"""Сборка обучающей смеси для VoxCPM2: наша речь + открытые корпуса ISSAI/FLEURS/CV + немного русского.

Источники:
  --ours  voxcpm_train.jsonl от transcribe.py (можно несколько раз)
  --corpus имя=папка  (kazakhtts2, ksc2, fleurs_kk, fleurs_ru, cv_kk — как раскладывает download_corpora.sh)
Отбор: --hours ours=all,ksc2=12,kazakhtts2=4,fleurs_kk=all,cv_kk=all,fleurs_ru=4
Правила смеси:
  * один диктор — не больше --spk-cap (3 %) всей смеси: модель учит язык и манеру, а не чужой тембр;
  * 40 % строк получают ref_audio — другую фразу того же диктора (докум. VoxCPM2: 30–50 %, чтобы не потерять
    клонирование голоса — через него потом подаём синтетический эталон голоса);
  * dataset_id: 0 наша речь, 1 ksc2, 2 kazakhtts2, 3 fleurs/cv kk, 4 русский;
  * 1 % (до 200 строк) уходит в val.jsonl — по нему тренер пишет пробы в TensorBoard.
Аудио: --to-flac пересобирает отобранное в 16 кГц моно FLAC (модель всё равно учится на 16 кГц) — меньше заливать.

  .venv/bin/python build_manifest.py --ours clean/voxcpm_train.jsonl --corpus ksc2=$VOICE_HOME/corpora/ksc2 \
      --hours ours=all,ksc2=12 --out $VOICE_HOME/mix-v1 --to-flac
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DSID = {"ours": 0, "ksc2": 1, "kazakhtts2": 2, "fleurs_kk": 3, "cv_kk": 3, "fleurs_ru": 4}
LANG = {"fleurs_ru": "ru"}
AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".opus"}


def dur_of(p: str) -> float:
    try:
        import soundfile as sf
        return float(sf.info(p).duration)
    except Exception:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                             capture_output=True, text=True).stdout.strip()
        return float(out or 0)


def load_stem_pairs(name: str, root: Path) -> list[dict]:
    """KazakhTTS2 / KSC2: аудио и .txt с тем же именем. Структура архивов не сверялась — парим по имени файла,
    диктор = папка-родитель аудио (для KazakhTTS2 это диктор, для KSC2 — источник/сессия)."""
    txt = {p.stem: p for p in root.rglob("*.txt")}
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() in AUDIO and p.stem in txt:
            t = txt[p.stem].read_text(encoding="utf-8", errors="ignore").strip()
            if t:
                spk = p.parent.name if name == "kazakhtts2" else ""
                if name == "kazakhtts2":  # ISSAI_KazakhTTS2/<диктор>/Audios/x.wav
                    spk = next((q for q in p.parts if re.fullmatch(r"[MF]\d+", q)), p.parent.parent.name)
                rows.append({"audio": str(p), "text": t, "speaker": f"{name}:{spk}" if spk else ""})
    return rows


def load_fleurs(name: str, root: Path) -> list[dict]:
    rows = []
    for split in ("train", "dev"):
        tsv = root / f"{split}.tsv"
        if not tsv.exists():
            continue
        for r in csv.reader(open(tsv, encoding="utf-8"), delimiter="\t"):
            # id, file, raw transcription, normalized, words, n_samples, gender
            p = root / split / r[1]
            if p.exists():
                rows.append({"audio": str(p), "text": r[2], "speaker": ""})
    return rows


def load_cv(name: str, root: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(root / "validated.tsv", encoding="utf-8"), delimiter="\t"):
        hits = list(root.rglob(r["path"]))
        if hits:
            rows.append({"audio": str(hits[0]), "text": r["sentence"], "speaker": f"cv:{r['client_id'][:12]}"})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", action="append", default=[])
    ap.add_argument("--corpus", action="append", default=[], help="имя=папка")
    ap.add_argument("--hours", default="ours=all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spk-cap", type=float, default=0.03)
    ap.add_argument("--ref-frac", type=float, default=0.4)
    ap.add_argument("--min-s", type=float, default=2.0)
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--to-flac", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    want = {k: (None if v == "all" else float(v)) for k, v in (x.split("=") for x in a.hours.split(","))}

    pool: dict[str, list[dict]] = {}
    for j in a.ours:
        pool.setdefault("ours", []).extend(json.loads(l) for l in open(j, encoding="utf-8") if l.strip())
    for spec in a.corpus:
        name, d = spec.split("=", 1)
        loader = load_fleurs if name.startswith("fleurs") else load_cv if name == "cv_kk" else load_stem_pairs
        pool[name] = loader(name, Path(d))
        print(f"{name}: найдено {len(pool[name])} пар аудио+текст")

    # длительности (у наших уже есть). 06.10: считаем лениво — после перемешивания, порциями, пока не набраны часы:
    # раньше sf.info шёл по всем 645 тыс. файлам KSC2 ради 8 ч и не успевал до потолка CPU-сервера.
    def fill(r):
        if not r.get("duration"):
            r["duration"] = round(dur_of(r["audio"]), 3)
        return r

    # отбор часов по источникам
    picked = []
    with ThreadPoolExecutor(8) as ex:
        for name, rows in pool.items():
            if name not in want:
                continue
            rnd.shuffle(rows)
            lim = want[name] * 3600 if want[name] is not None else float("inf")
            s, i, step = 0.0, 0, 2000
            while s < lim and i < len(rows):
                for r in ex.map(fill, rows[i:i + step]):
                    if s >= lim:
                        break
                    if not (a.min_s <= r["duration"] <= a.max_s):
                        continue
                    r.update(dataset_id=DSID.get(name, 0), src=name, lang=r.get("lang") or LANG.get(name, "kk"))
                    picked.append(r)
                    s += r["duration"]
                i += step
            print(f"{name}: отобрано {s / 3600:.2f} ч (просмотрено {min(i, len(rows))} из {len(rows)})", flush=True)
    total = sum(r["duration"] for r in picked)

    # потолок на диктора
    cap = a.spk_cap * total
    per = defaultdict(float)
    mix = []
    for r in picked:
        spk = r.get("speaker") or ""
        if spk and per[spk] + r["duration"] > cap:
            continue
        per[spk] += r["duration"]
        mix.append(r)

    # ref_audio: другая фраза того же диктора, 3–12 с
    by_spk = defaultdict(list)
    for r in mix:
        if r.get("speaker") and 3 <= r["duration"] <= 12:
            by_spk[r["speaker"]].append(r)
    for r in mix:
        cands = [c for c in by_spk.get(r.get("speaker") or "", []) if c is not r]
        if cands and rnd.random() < a.ref_frac:
            c = rnd.choice(cands)
            r["ref_audio"], r["ref_duration"] = c["audio"], c["duration"]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.to_flac:
        adir = out / "audio"
        adir.mkdir(exist_ok=True)
        conv = {}

        def flac(src: str) -> str:
            dst = adir / (re.sub(r"[^\w.-]", "_", Path(src).parent.name) + "__" + Path(src).stem + ".flac")
            if not dst.exists():
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", str(dst)], check=True)
            return str(dst)
        srcs = sorted({r["audio"] for r in mix} | {r["ref_audio"] for r in mix if r.get("ref_audio")})
        with ThreadPoolExecutor(8) as ex:
            for s, d in zip(srcs, ex.map(flac, srcs)):
                conv[s] = d
        for r in mix:
            r["audio"] = conv[r["audio"]]
            if r.get("ref_audio"):
                r["ref_audio"] = conv[r["ref_audio"]]

    rnd.shuffle(mix)
    n_val = min(200, max(1, len(mix) // 100))
    keys = ["audio", "text", "duration", "dataset_id", "ref_audio", "ref_duration", "speaker", "lang", "src"]
    for fn, part in (("val.jsonl", mix[:n_val]), ("train.jsonl", mix[n_val:])):
        with open(out / fn, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({k: r.get(k) for k in keys if r.get(k) not in (None, "")}, ensure_ascii=False) + "\n")

    st = defaultdict(lambda: [0, 0.0])
    for r in mix:
        st[(r["src"], r["lang"])][0] += 1
        st[(r["src"], r["lang"])][1] += r["duration"]
    hours = sum(v[1] for v in st.values()) / 3600
    lines = [f"смесь: {len(mix)} фраз, {hours:.1f} ч, дикторов с id: {len(per) - (1 if '' in per else 0)}, "
             f"с ref_audio: {sum(1 for r in mix if r.get('ref_audio'))}"]
    lines += [f"  {s:12s} {l}: {n:7d} фраз {sec / 3600:6.2f} ч ({sec / 36 / max(hours, 1e-9):.0f} %)"
              for (s, l), (n, sec) in sorted(st.items())]
    (out / "stats.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"→ {out}/train.jsonl, val.jsonl")


if __name__ == "__main__":
    main()
