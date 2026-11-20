"""Проверка определения языка на фразах с известным языком."""
import os, sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from voicelab import audio as A
from voicelab.segment import Segmenter
from voicelab.lang import LangID
dev, size = (sys.argv[1:3] + ["cpu", "small"])[:2] if len(sys.argv) > 2 else ("cpu", "small")
lid, seg = LangID(size, dev), Segmenter()
T = Path(os.environ.get("VOICE_DS", "data")) / "testmix"   # scripts/make_testmix.py
items = [(T/"ref"/f"{c}.wav", "kk") for c in ["c01_kazemo-akzhol_clean", "c02_commonvoice_kk", "m01_kaztts2-f1_lofi_-10dB", "m05_kaztts2-f3_vocalpop_-10dB", "m08_fleurs_lofi_-3dB"]]
items += [(T/"input"/"r02_drama_sorry_ru.m4a", "ru")]
ok = n = 0
for f, truth in items:
    x = A.load(f, 16000, 1)
    for p in seg.phrases(x):
        if p["dur"] < 2: continue
        r = lid(x[int(p["start"]*16000):int(p["end"]*16000)], with_text=False)
        n += 1; ok += r["lang"] == truth
        print(f"{f.stem[:14]} {p['dur']:5.1f}s truth={truth} -> {r['lang']:5s} top={r['lang_top']:7s} kk={r['p_kk_like']:.2f} ru={r['p_ru']:.2f} vox_kk={r['vox_kk']:.2f}", flush=True)
print(f"верно {ok}/{n}")
