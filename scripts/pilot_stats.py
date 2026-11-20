#!/usr/bin/env python3
"""Цифры пилота по work/*/result.json: сколько чистой речи вышло и почему отсеялось."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REASON_RU = {
    "too_short": "кусок короче 2 с",
    "music_left": "музыка не отделилась",
    "singing": "осталось пение",
    "multi_speaker": "несколько говорящих",
    "noise": "шум / слабый сигнал",
    "low_quality": "плохое качество голоса",
    "clipping": "перегруз (клиппинг)",
    "lang_other": "не казахский / не русский",
}


def stats(work: Path) -> dict:
    res = [json.loads(p.read_text()) for p in sorted(work.glob("*/result.json"))]
    if not res:
        return {}
    audio = sum(r["duration"] for r in res)
    speech = sum(r["speech_sec"] for r in res)
    clean = sum(r["clean_sec"] for r in res)
    proc = sum(r["timings"]["total"] for r in res)
    rej = Counter()
    lang_clean = Counter()
    lang_all = Counter()
    for r in res:
        for p in r["phrases"]:
            lang_all[p.get("lang", "?")] += p["duration"]
            if p["passed"]:
                lang_clean[p.get("lang", "?")] += p["duration"]
            else:
                rej[p["reject_reason"]] += p["duration"]
    stage = defaultdict(float)
    for r in res:
        for k, v in r["timings"].items():
            if k != "total":
                stage[k] += v
    sep = [r for r in res if r["separated"]]
    zero = [r for r in res if r["clean_sec"] == 0]
    clean_per_clip = clean / len(res)
    return dict(
        n=len(res), n_music=sum(r["has_music"] for r in res), n_sep=len(sep), n_zero=len(zero),
        audio_min=audio / 60, speech_min=speech / 60, clean_min=clean / 60,
        clean_share_audio=clean / audio if audio else 0, clean_share_speech=clean / speech if speech else 0,
        rejected_min={k: v / 60 for k, v in rej.most_common()},
        lang_clean_min={k: v / 60 for k, v in lang_clean.most_common()},
        lang_all_min={k: v / 60 for k, v in lang_all.most_common()},
        clean_min_per_clip=clean_per_clip / 60,
        clips_for_100h=(100 * 3600 / clean_per_clip) if clean_per_clip else float("inf"),
        clips_for_100h_kk=(100 * 3600 / (lang_clean.get("kk", 0) / len(res))) if lang_clean.get("kk") else float("inf"),
        proc_s=proc, sec_per_audio_min=proc / (audio / 60) if audio else 0,
        stage_s_per_audio_min={k: v / (audio / 60) for k, v in stage.items()},
        avg_clip_s=audio / len(res),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    s = stats(ap.parse_args().work)
    if not s:
        print("нет результатов"); return
    print(f"роликов: {s['n']} (с музыкой {s['n_music']}, отделяли {s['n_sep']}, без единой чистой фразы {s['n_zero']})")
    print(f"аудио {s['audio_min']:.1f} мин, речь (фразы) {s['speech_min']:.1f} мин, чистая речь {s['clean_min']:.1f} мин "
          f"({100 * s['clean_share_audio']:.0f}% от аудио, {100 * s['clean_share_speech']:.0f}% от речи)")
    print("отсев (мин): " + ", ".join(f"{REASON_RU.get(k, k)} {v:.1f}" for k, v in s["rejected_min"].items()))
    print("язык чистой речи (мин): " + ", ".join(f"{k} {v:.1f}" for k, v in s["lang_clean_min"].items()))
    print(f"чистой речи на ролик: {s['clean_min_per_clip']:.2f} мин -> на 100 ч нужно ~{s['clips_for_100h']:.0f} роликов "
          f"(только казахский: ~{s['clips_for_100h_kk']:.0f})")
    print(f"скорость: {s['sec_per_audio_min']:.1f} с обработки на 1 мин аудио; по шагам: " +
          ", ".join(f"{k} {v:.1f}" for k, v in s["stage_s_per_audio_min"].items()))


if __name__ == "__main__":
    main()
