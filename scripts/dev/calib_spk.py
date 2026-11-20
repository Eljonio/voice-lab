"""Калибровка проверки «один голос»: варианты метрик на одиночных и диалоговых фразах."""
import os, sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from voicelab import audio as A
from voicelab.segment import Segmenter
from voicelab.speakers import SpeakerChecker
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
spk, seg = SpeakerChecker(dev), Segmenter()
T = Path(os.environ.get("VOICE_DS", "data")) / "testmix"   # scripts/make_testmix.py
def windows(x, w, h):
    W, H = int(w*16000), int(h*16000)
    st = list(range(0, max(len(x)-W, 0)+1, H)) or [0]
    ch = [x[s:s+W] for s in st]; r = np.array([np.sqrt(np.mean(c**2)) for c in ch])
    return [c for c, k in zip(ch, r > 0.25*np.median(r)) if k]
for clip in ["c01_kazemo-akzhol_clean", "c03_ksd_kk", "m01_kaztts2-f1_lofi_-10dB", "c04_dialog_two_voices", "c02_commonvoice_kk"]:
    x = A.load(T/"ref"/f"{clip}.wav", 16000, 1)
    for p in seg.phrases(x):
        s = x[int(p["start"]*16000):int(p["end"]*16000)]
        out = [f"{clip[:12]} {p['start']:5.1f}-{p['end']:5.1f}"]
        for w, h in [(1.5, .75), (2.0, 1.0), (3.0, 1.0)]:
            ch = windows(s, w, h)
            if len(ch) < 2: out.append(f"w{w}: -"); continue
            E = spk.embed(ch); S = E @ E.T; m = E.mean(0); m /= np.linalg.norm(m)
            out.append(f"w{w}: pair_min {S.min():.2f} to_mean_min {(E@m).min():.2f}")
        print(" | ".join(out), flush=True)
