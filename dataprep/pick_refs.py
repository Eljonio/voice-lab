#!/usr/bin/env python3
"""Кандидаты в КАЗАХСКИЙ эталон голоса для оценки (на арендованном CPU-сервере с постоянным диском данных; ноут не грузим).

  python dataprep/pick_refs.py --out "$VOICE_HOME/refs" --val "$VOICE_HOME/v2/manifest/ours_v3_val.jsonl" \
      --ksc2 "$VOICE_HOME/corpora/ksc2/ISSAI_KSC2/Test" [--per-source 4]

Зачем (08.10): все пробы v1/v2 озвучены по eval/ref.wav — это РУССКАЯ синтетическая фраза ru01; часть «акцента»
может идти от неё. Нужен живой казахский женский эталон 8–12 с, законченная фраза, точный текст.
Отбор: 8–12 с; женщина (медиана F0 по librosa.pyin ≥ 170 Гц); DNSMOS OVRL ≥ 3,1; GigaAM CTC CER к тексту ≤ 5 %;
у наших — фраза кончается концом мысли (sentence_end, знак в конце); один диктор — один кандидат.
09.10: без заполнителей в тексте («ыыы», «ээ», «мм», «жаңағы», «ну», «вот») — эталон задаёт манеру, «эканье» в нём не нужно;
у KSC2 знаков нет — законченность не проверить, поэтому и у них текст должен начинаться не с союза («а», «и», «ал», «бірақ»).
Источники: (1) KSC2 Test, живые домены (ISSAI, MIT — можно показывать и использовать дальше; в тексте нет знаков);
(2) отложенные каналы наших (YouTube, ours_v3_val) — ТОЛЬКО для внутренней оценки, наружу не отдавать.
Выход: <out>/<id>.wav (16 кГц) + refs.json (текст, длительность, F0, DNSMOS, CER, лицензия).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "scripts"))


FILL = re.compile(r"(?<!\w)(ы+|э+|мм+|м|хм+|әә+|аа+|жаңағы|ну|вот)(?!\w)", re.I)
CONJ = re.compile(r"^(а|и|ал|бірақ|және|ну|но|сосын|содан)\b", re.I)
RUW = re.compile(r"(?<!\w)(просто|максимально|вообще|короче|значит|конечно|потому|сейчас|даже|уже|тоже|очень|только|когда|если|что|это|как|так|"
                 r"вот|ладно|давай|типа|кстати|блин|который|которые|можно|нужно|надо)(?!\w)", re.I)


def kk_share(t: str) -> float:
    """Доля слов (≥ 3 букв) с казахскими буквами ә ғ қ ң ө ұ ү һ і — в живом казахском обычно 0,4–0,6, у русских слов 0."""
    w = [x for x in re.findall(r"[^\W\d_]{3,}", t.lower())]
    return sum(bool(re.search("[әғқңөұүһі]", x)) for x in w) / max(1, len(w))


def f0_median(x, sr=16000) -> float:
    import librosa
    import numpy as np
    f0, vflag, _ = librosa.pyin(x, fmin=70, fmax=420, sr=sr, frame_length=1024)
    v = f0[vflag & ~np.isnan(f0)]
    return float(np.median(v)) if len(v) > 20 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--ksc2", type=Path, required=True)
    ap.add_argument("--per-source", type=int, default=4)
    ap.add_argument("--max-try", type=int, default=600)
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import voicelab.audio as A
    from voicelab.quality import DNSMOS
    import gigaam_eval as G

    rnd = random.Random(a.seed)
    cands = []
    for line in open(a.val, encoding="utf-8"):
        r = json.loads(line)
        t = r.get("text", "")
        if r.get("lang") == "kk" and 8 <= r.get("duration", 0) <= 12 and r.get("sentence_end", 1) and t[-1:] in ".!?" \
                and not re.search(r"[0-9A-Za-z]", t):
            cands.append(dict(src="ours_val", audio=r["audio"], text=t, speaker=r.get("speaker") or r["audio"],
                              license="YouTube, только внутренняя оценка"))
    for d in ("podcasts", "radio", "talkshow"):
        root = a.ksc2 / d
        if not root.exists():
            continue
        txt = {p.stem: p for p in root.rglob("*.txt")}   # как stem_pairs в build_mix_v2: звук и текст могут лежать в разных папках
        for p in root.rglob("*"):
            if p.suffix.lower() in (".flac", ".wav") and p.stem in txt:
                t = txt[p.stem].read_text(encoding="utf-8", errors="ignore").strip()
                if t:
                    cands.append(dict(src=f"ksc2_{d}", audio=str(p), text=t, speaker=f"ksc2_{d}:{p.stem.rsplit('_', 1)[0]}",
                                      license="KSC2 ISSAI, MIT"))
    rnd.shuffle(cands)
    asr = G.ASR("cpu")
    dns = DNSMOS()
    a.out.mkdir(parents=True, exist_ok=True)
    picked, tried, seen_spk = [], {}, set()
    for c in cands:
        grp = "ours" if c["src"] == "ours_val" else "ksc2"
        if sum(1 for x in picked if x["group"] == grp) >= a.per_source or tried.get(grp, 0) >= a.max_try:
            continue
        if c["speaker"] in seen_spk or FILL.search(c["text"]) or CONJ.search(c["text"].strip()) or RUW.search(c["text"]) \
                or kk_share(c["text"]) < 0.3:
            continue
        try:
            x = A.load(c["audio"], 16000).astype(np.float32)
        except Exception:  # noqa: BLE001
            continue
        dur = len(x) / 16000
        if not 8 <= dur <= 12:
            continue
        tried[grp] = tried.get(grp, 0) + 1
        f0 = f0_median(x)
        if f0 < 170:
            continue
        q = dns(x)
        if q["ovrl"] < 3.1:
            continue
        hyp, _ = asr(c["audio"])
        cer = G.cer(G.norm(c["text"]), G.norm(hyp))
        if cer > 0.05:
            continue
        seen_spk.add(c["speaker"])
        cid = f"{grp}_{len([x for x in picked if x['group'] == grp]) + 1}"
        sf.write(str(a.out / f"{cid}.wav"), x, 16000)
        picked.append(dict(id=cid, group=grp, src=c["src"], text=c["text"], hyp=hyp, cer=round(cer, 4), duration=round(dur, 2),
                           f0_median=round(f0, 1), dnsmos_ovrl=round(q["ovrl"], 3), dnsmos_sig=round(q["sig"], 3),
                           orig=c["audio"], license=c["license"]))
        print(f"{cid}: {dur:.1f} с F0 {f0:.0f} OVRL {q['ovrl']:.2f} CER {cer * 100:.1f} % | {c['text'][:90]}", flush=True)
        if len(picked) >= 2 * a.per_source:
            break
    (a.out / "refs.json").write_text(json.dumps(dict(tried=tried, picked=picked), ensure_ascii=False, indent=1))
    print(f"кандидатов {len(picked)} (проверено {tried}) → {a.out}")


if __name__ == "__main__":
    main()
