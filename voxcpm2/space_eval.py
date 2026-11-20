#!/usr/bin/env python3
"""v0 без своей GPU: набор eval/phrases.json через официальную песочницу OpenBMB/VoxCPM-Demo на Hugging Face.

Те же веса VoxCPM2, их GPU, ~50 с на фразу, без регистрации и токена. В песочницу уходят только тексты набора
(реплики бота) и синтетический эталон, который она же и сделала. Сида и числа шагов в песочнице нет — cfg 2.0,
остальное по умолчанию; поэтому v0 из песочницы и v0 из платной сессии (synth_eval.py, seed 42, 10 шагов) — чуть разные.

  .venv/bin/python space_eval.py [--out runs/space-v0]
Выход: <out>/eval/ref.wav (эталон: описание + ru01; копируется в ../eval/ref.wav, если его ещё нет — общий для всех версий),
       <out>/eval/base/design/<id>.wav (голос по описанию), <out>/eval/base/ref/<id>.wav (клон эталона), report.json.
Продолжает с места: готовые файлы не перегенерирует.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "runs/space-v0"))
    ap.add_argument("--space", default="OpenBMB/VoxCPM-Demo")
    ap.add_argument("--modes", default="design,ref")
    a = ap.parse_args()
    from gradio_client import Client, handle_file
    import soundfile as sf

    P = json.loads((ROOT / "eval/phrases.json").read_text())
    control = P["voice"]["design"].strip("()")
    out = Path(a.out)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    rep_p = out / "eval/base/report.json"
    rep = json.loads(rep_p.read_text()) if rep_p.exists() else {
        "version": "base", "where": f"HF Space {a.space}", "control": control, "cfg": 2.0, "items": {}}
    cl = Client(a.space, verbose=False)

    def gen(text: str, dst: Path, ref: Path | None = None, ctrl: str = "") -> dict:
        if dst.exists():
            return rep["items"].get(str(dst.relative_to(out / "eval/base")) if "base" in dst.parts else dst.name, {})
        t = time.time()
        for attempt in range(3):
            try:
                wav = cl.predict(text, ctrl, handle_file(str(ref)) if ref else None, False, "", 2.0, False, False,
                                 api_name="/generate")
                break
            except Exception as e:  # очередь / перезапуск песочницы
                print(f"  попытка {attempt + 1}: {e}", flush=True)
                time.sleep(20)
        else:
            raise RuntimeError(f"песочница не ответила для {dst.name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        x, sr = sf.read(wav)
        sf.write(dst, x, sr)
        r = dict(audio_s=round(len(x) / sr, 2), sr=sr, wall_s=round(time.time() - t, 1))
        print(f"{dst.relative_to(out)} {r}", flush=True)
        return r

    ref = out / "eval/ref.wav"
    ru01 = next(p for p in P["phrases"] if p["id"] == "ru01")
    rep["ref"] = gen(ru01["text"], ref, ctrl=control) or rep.get("ref", {})
    common = ROOT / "eval/ref.wav"
    if not common.exists():
        shutil.copy2(ref, common)
        print(f"общий эталон набора: {common}")
    for mode in a.modes.split(","):
        for p in P["phrases"]:
            dst = out / "eval/base" / mode / f"{p['id']}.wav"
            r = gen(p["text"], dst, ref=common if mode == "ref" else None, ctrl=control if mode == "design" else "")
            if r:
                rep["items"][f"{mode}/{p['id']}"] = r
            rep_p.parent.mkdir(parents=True, exist_ok=True)
            rep_p.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    rep["made"] = time.strftime("%Y-%m-%d %H:%M")
    rep_p.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print("готово:", out)


if __name__ == "__main__":
    main()
