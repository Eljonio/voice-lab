#!/usr/bin/env python3
"""SIM к эталону (ECAPA, speechbrain spkrec-ecapa-voxceleb) и DNSMOS P.835 по всем версиям прогона — как в
eval/results/<RUN>_sim_dnsmos.json.

  python scripts/sim_dnsmos.py $VOICE_HOME/runs/<RUN> --ref $VOICE_HOME/eval/ref.wav --out X_sim_dnsmos.json [--device cpu]

Версии: eval/base (исходная модель), step_N/eval; режим ref/ (клон eval/ref.wav). SIM — косинус эмбеддинга фразы и эталона
(среднее, минимум); DNSMOS OVRL/SIG — среднее по фразам режима ref. Плюс design/ (голос по описанию) — только DNSMOS:
SIM к эталону там не имеет смысла (тембр свой у каждой версии).
Тяжёлое (ECAPA, DNSMOS) — лучше на сервере, а не на ноутбуке.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--extra", action="append", default=[], help="NAME=DIR, где DIR/{ref,design}/*.wav")
    a = ap.parse_args()

    import numpy as np
    import voicelab.audio as A
    from voicelab.quality import DNSMOS
    from voicelab.speakers import SpeakerChecker

    t0 = time.time()
    spk = SpeakerChecker(device=a.device)
    dns = DNSMOS()
    ref_emb = spk.embed([A.load(a.ref, 16000)])[0]
    versions: dict[str, Path] = {}
    for spec in a.extra:
        n, d = spec.split("=", 1)
        versions[n] = Path(d)
    versions["base"] = a.run / "eval/base"
    for s in sorted(a.run.glob("step_*"), key=lambda p: int(re.sub(r"\D", "", p.name))):
        versions[f"step_{int(re.sub(r'[^0-9]', '', s.name))}"] = s / "eval"
    res = {"run": a.run.name, "ref": str(a.ref), "made": time.strftime("%Y-%m-%d %H:%M"), "versions": {}}
    for name, d in versions.items():
        out = {}
        for mode in ("ref", "design"):
            files = sorted((d / mode).glob("*.wav"))
            if not files:
                continue
            sims, ov, sg, per = [], [], [], {}
            for f in files:
                x = A.load(f, 16000).astype(np.float32)
                q = dns(x)
                ov.append(q["ovrl"])
                sg.append(q["sig"])
                item = dict(ovrl=round(q["ovrl"], 3), sig=round(q["sig"], 3), bak=round(q["bak"], 3))
                if mode == "ref":
                    e = spk.embed([x])[0]
                    sim = float(e @ ref_emb)
                    sims.append(sim)
                    item["sim"] = round(sim, 4)
                per[f.stem] = item
            m = dict(n=len(files), ovrl=round(st.mean(ov), 3), sig=round(st.mean(sg), 3), items=per)
            if sims:
                m.update(sim_mean=round(st.mean(sims), 3), sim_min=round(min(sims), 3))
            out[mode] = m
        if out:
            res["versions"][name] = out
            r = out.get("ref", {})
            dz = out.get("design", {})
            print(f"{name:12} ref: SIM ср {r.get('sim_mean', float('nan')):.3f} мин {r.get('sim_min', float('nan')):.3f} "
                  f"OVRL {r.get('ovrl', float('nan')):.2f} SIG {r.get('sig', float('nan')):.2f}"
                  + (f" | design: OVRL {dz['ovrl']:.2f} SIG {dz['sig']:.2f}" if dz else ""), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"→ {a.out} за {time.time() - t0:.0f} с", file=sys.stderr)


if __name__ == "__main__":
    main()
