#!/usr/bin/env python3
"""Сводка обработки v2 по work/*/result.json и «ворота» пробного куска.

  python dataprep/gate_v2.py --out "$VOICE_HOME/v2" [--since UNIX_TS] [--check]

Печатает: роликов, часы сырого / речи / чисто после очистки / с надёжным текстом, отсев по причинам (в часах), медианы
CER(GigaAM vs второй ASR) и уверенности CTC, скорость (ч сырого на час работы), ошибки. Последняя строка — коротко.
--check: код 1, если пробный кусок плохой (конвейер сломан, а не «данные похуже»):
  < 5 роликов готово; ошибок > 30 %; чисто < 10 % сырого; надёжного текста < 35 % чистого; медиана CER > 0,30.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--since", type=float, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    res, errors = [], 0
    for d in sorted((a.out / "work").glob("*")):
        f = d / "result.json"
        if f.exists():
            try:
                r = json.loads(f.read_text())
                rs = d / "respan.json"
                if rs.exists():
                    x = json.loads(rs.read_text())
                    r["phrases"] = [p for p in r["phrases"] if not p.get("no_pause")] + x["phrases"]
                    r["clean_sec"] += x.get("clean_sec", 0.0)
                    r["asr_sec"] += x.get("asr_sec", 0.0)
                res.append(r)
                continue
            except Exception:  # noqa: BLE001
                pass
        if (d / "error.txt").exists():
            errors += 1
    H = 3600.0
    raw = sum(r["duration"] for r in res) / H
    speech = sum(r["speech_sec"] for r in res) / H
    clean = sum(r["clean_sec"] for r in res) / H
    ok = sum(r["asr_sec"] for r in res) / H
    rej, asr_rej, lang_ok = Counter(), Counter(), Counter()
    cers, confs = [], []
    for r in res:
        for p in r["phrases"]:
            if not p.get("clean_ok"):
                rej[(p.get("clean_why") or "?").split(";")[0]] += p["duration"] / H
                continue
            if p.get("cer_giga") is not None:
                cers.append(p["cer_giga"])
            if p.get("ctc_conf") is not None:
                confs.append(p["ctc_conf"])
            if p.get("asr_ok"):
                lang_ok[p.get("lang") or "?"] += p["duration"] / H
            else:
                for w in (p.get("asr_why") or "?").split(";"):
                    asr_rej[w] += p["duration"] / H
    el = (time.time() - a.since) / H if a.since else 0

    def made_ts(r):
        try:
            return time.mktime(time.strptime(r.get("made", ""), "%Y-%m-%d %H:%M:%S"))
        except Exception:  # noqa: BLE001
            return 0
    raw_new = sum(r["duration"] for r in res if made_ts(r) >= a.since) / H if a.since else raw   # сессия 2: только свои ролики
    speed = raw_new / el if el > 0 else 0
    med_cer = st.median(cers) if cers else None
    print(f"роликов {len(res)}, ошибок {errors}; сырого {raw:.1f} ч, речи {speech:.1f} ч, чисто {clean:.1f} ч "
          f"({clean / raw * 100 if raw else 0:.0f} % сырого), надёжный текст {ok:.1f} ч ({ok / clean * 100 if clean else 0:.0f} % чистого)")
    print("отсев очистки, ч:", ", ".join(f"{k} {v:.1f}" for k, v in rej.most_common()))
    print("отсев текста, ч (причины могут совпадать):", ", ".join(f"{k} {v:.1f}" for k, v in asr_rej.most_common()))
    print("надёжный текст по языкам, ч:", ", ".join(f"{k} {v:.1f}" for k, v in lang_ok.most_common()))
    if cers:
        print(f"CER GigaAM vs второй ASR: медиана {med_cer:.3f}, p90 {sorted(cers)[int(0.9 * len(cers))]:.3f}; "
              f"уверенность CTC медиана {st.median(confs):.3f}")
    if el:
        print(f"работа {el:.2f} ч, скорость {speed:.1f} ч сырого/ч")
    short = (f"{len(res)} роликов, сырого {raw:.1f} ч → чисто {clean:.1f} → текст ок {ok:.1f} ч; "
             f"CER мед {med_cer if med_cer is None else round(med_cer, 3)}; {speed:.0f} ч/ч; ошибок {errors}")
    if a.json:
        a.json.write_text(json.dumps(dict(clips=len(res), errors=errors, raw_h=raw, speech_h=speech, clean_h=clean, ok_h=ok,
                                          reject_clean_h=dict(rej), reject_text_h=dict(asr_rej), ok_by_lang_h=dict(lang_ok),
                                          cer_median=med_cer, speed_h_per_h=speed), ensure_ascii=False, indent=1))
    bad = []
    if a.check:
        if len(res) < 5:
            bad.append(f"готово всего {len(res)} роликов")
        if errors > 0.3 * max(1, len(res) + errors):
            bad.append(f"ошибок {errors} из {len(res) + errors}")
        if raw and clean / raw < 0.10:
            bad.append(f"чисто {clean / raw * 100:.0f} % сырого < 10 %")
        if clean and ok / clean < 0.35:
            bad.append(f"надёжного текста {ok / clean * 100:.0f} % чистого < 35 %")
        if med_cer is not None and med_cer > 0.30:
            bad.append(f"медиана CER {med_cer:.2f} > 0,30")
    print(("ПЛОХО: " + "; ".join(bad) + " | ") if bad else "", short, sep="")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
