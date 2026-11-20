#!/usr/bin/env python3
"""Озвучка фиксированного набора eval/phrases.json: исходная модель и чекпоинты LoRA за ОДНУ загрузку.

Для страницы сравнения версий: одни и те же 12 фраз у всех версий.
Два режима на фразу:
  ref/<id>.wav    — клон одного синтетического эталона eval/ref.wav. Основной режим сравнения: тембр у всех версий один,
                    слышна разница в произношении;
  design/<id>.wav — голос по описанию из phrases.json (после LoRA может ослабнуть, это тоже надо слышать).
Эталон: если --ref не существует, его создаёт исходная модель (описание + ru01, seed 42) — один раз, дальше не меняется.

  python synth_eval.py --phrases eval/phrases.json --ref eval/ref.wav --base runs/R/eval/base \
      --lora runs/R/step_0500=runs/R/step_0500/eval --lora runs/R/step_1000=runs/R/step_1000/eval --device cuda

09.10: --ckpt ПАПКА=ВЫВОД — чекпоинт любого вида: LoRA (lora_weights.*) или ПОЛНЫЙ (model.safetensors / pytorch_model.bin,
как пишет train_voxcpm_finetune.py при полном дообучении). Полные веса грузятся в ту же модель поверх базы (load_state_dict)
с проверкой ключей: лишние ключи или пропуск чего-то кроме audio_vae.* — ошибка (иначе strict=False молча оставил бы базу).
Порядок: base первой, потом чекпоинты. LoRA и полные в одном запуске не смешиваются.
  --ids kk01,kk02 — только эти фразы (проверка на CPU одной фразой); --design-for none — без режима design.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("VOXCPM2_WEIGHTS", HERE / "weights" / "VoxCPM2"))


def ckpt_kind(path: str) -> str:
    p = Path(path)
    if (p / "lora_weights.safetensors").exists() or (p / "lora_weights.ckpt").exists():
        return "lora"
    if (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists():
        return "full"
    raise SystemExit(f"{p}: ни lora_weights.*, ни model.safetensors/pytorch_model.bin")


def load_full(model, ckpt: str) -> None:
    """Полные веса чекпоинта (без audio_vae — тренер его не сохраняет) поверх загруженной модели, с проверкой ключей."""
    import torch
    p = Path(ckpt)
    t = time.time()
    if (p / "model.safetensors").exists():
        from safetensors.torch import load_file
        sd = load_file(str(p / "model.safetensors"))
    else:
        c = torch.load(p / "pytorch_model.bin", map_location="cpu", weights_only=True)
        sd = c.get("state_dict", c)
    tm = model.tts_model
    own = tm.state_dict()
    unexpected = [k for k in sd if k not in own]
    missing = [k for k in own if k not in sd and not k.startswith("audio_vae.") and "lora_" not in k]
    shape = [k for k in sd if k in own and tuple(sd[k].shape) != tuple(own[k].shape)]
    if unexpected or missing or shape:
        raise RuntimeError(f"{p}: ключи не совпадают с моделью — лишних {len(unexpected)} {unexpected[:3]}, "
                           f"нет {len(missing)} {missing[:3]}, другая форма {len(shape)} {shape[:3]}")
    tm.load_state_dict(sd, strict=False)
    dt = {str(v.dtype) for v in sd.values()}
    n = sum(v.numel() for v in sd.values())
    del sd
    print(f"полные веса {p.name}: {n / 1e9:.2f} млрд параметров ({', '.join(sorted(dt))}) за {time.time() - t:.0f} с", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases", required=True)
    ap.add_argument("--ref", required=True, help="эталон голоса; создаётся исходной моделью, если нет")
    ap.add_argument("--base", default="", help="папка для исходной модели (v0); пусто — не озвучивать")
    ap.add_argument("--lora", action="append", default=[], help="чекпоинт=папка_вывода (можно несколько)")
    ap.add_argument("--ckpt", action="append", default=[], help="чекпоинт LoRA или полный = папка_вывода")
    ap.add_argument("--ids", default="", help="только эти id фраз через запятую")
    ap.add_argument("--cats", default="", help="только эти категории (поле cat в phrases_v2.json) через запятую")
    ap.add_argument("--seed", type=int, default=None, help="сид вместо voice.seed (оценка на 2–3 сидах)")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--dialog-only", action="store_true",
                    help="только диалоги phrases_v2 в контекстном режиме: реплика i с репликой i−1 как prompt (dialog/<id>.wav)")
    ap.add_argument("--design-for", default="base,last", help="для каких версий ещё и режим design: base,last,all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--timesteps", type=int, default=10)
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from voxcpm import VoxCPM

    P = json.loads(Path(a.phrases).read_text())
    phrases, voice = P["phrases"], P["voice"]
    if a.ids:
        want = set(a.ids.split(","))
        phrases = [p for p in phrases if p["id"] in want]
    if a.cats:
        want = set(a.cats.split(","))
        phrases = [p for p in phrases if p.get("cat") in want]
    if a.seed is not None:
        voice = dict(voice, seed=a.seed)
    loras = [x.split("=", 1) for x in a.lora]
    fulls = []
    for x in a.ckpt:
        c, o = x.split("=", 1)
        (loras if ckpt_kind(c) == "lora" else fulls).append([c, o])
    if loras and fulls:
        raise SystemExit("LoRA и полные чекпоинты в одном запуске не смешиваю — два запуска")
    t0 = time.time()
    kw = dict(load_denoiser=False, optimize=False, device=a.device)
    if loras:  # модель строится с LoRA-слоями (конфиг из первого чекпоинта), база = LoRA выключена
        kw["lora_weights_path"] = loras[0][0]
    model = VoxCPM.from_pretrained(str(WEIGHTS), **kw)
    sr = model.tts_model.sample_rate
    print(f"модель загружена за {time.time() - t0:.0f} с", flush=True)

    def gen(text: str, out: Path, ref: str | None = None, **extra) -> dict:
        torch.manual_seed(voice["seed"])
        np.random.seed(voice["seed"])
        t = time.time()
        # 09.10: сид — в сам generate (иначе VoxCPM пересевает генератор torch.seed() и оценка невоспроизводима)
        gk = dict(text=text, cfg_value=a.cfg, inference_timesteps=a.timesteps, seed=voice["seed"])
        if ref:
            gk["reference_wav_path"] = ref
        gk.update(extra)
        wav = model.generate(**gk)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, wav, sr)
        return dict(audio_s=round(len(wav) / sr, 2), compute_s=round(time.time() - t, 1))

    ref = Path(a.ref)
    if not ref.exists():
        if fulls:
            raise SystemExit(f"нет эталона {ref}: с полными чекпоинтами эталон не создаю (модель уже не базовая)")
        if loras:
            model.set_lora_enabled(False)
        ru01 = next(p for p in P["phrases"] if p["id"] == "ru01")
        r = gen(voice["design"] + ru01["text"], ref)
        print(f"эталон создан: {ref} {r}", flush=True)

    jobs = []
    if a.base:
        jobs.append(("base", None, Path(a.base)))
    jobs += [(Path(c).name, c, Path(o)) for c, o in loras + fulls]
    full_set = {c for c, _ in fulls}
    want_design = set(a.design_for.split(","))
    for k, (name, ckpt, out) in enumerate(jobs):
        if ckpt is None:
            if loras:
                model.set_lora_enabled(False)
        elif ckpt in full_set:
            load_full(model, ckpt)
        else:
            model.load_lora(ckpt)
            model.set_lora_enabled(True)
        is_last = k == len(jobs) - 1
        modes = ["ref"]
        if "all" in want_design or (ckpt is None and "base" in want_design) or (ckpt and is_last and "last" in want_design):
            modes.append("design")
        rep = {"version": name, "checkpoint": ckpt, "ref": str(ref), "seed": voice["seed"], "cfg": a.cfg,
               "made": time.strftime("%Y-%m-%d %H:%M"), "items": {}}
        if a.dialog_only:   # продолжение: тот же голос говорит следующую реплику после своей предыдущей (тон и темп)
            byid = {q["id"]: q for q in P["phrases"]}
            for dlg in P.get("dialogs", []):
                prev = None
                for tid in dlg["turns"]:
                    fp = out / "dialog" / f"{tid}.wav"
                    kw = dict(prompt_wav_path=str(prev[0]), prompt_text=prev[1]) if prev else {}
                    rep["items"][f"dialog/{tid}"] = gen(byid[tid]["text"], fp, str(ref), **kw)
                    prev = (fp, byid[tid]["text"])
            (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
            print(f"{name}: диалоги {len(P.get('dialogs', []))} в контекстном режиме → {out}", flush=True)
            continue
        for p in phrases:
            for m in modes:
                fp = out / m / f"{p['id']}.wav"
                r = gen(p["text"], fp, str(ref)) if m == "ref" else gen(voice["design"] + p["text"], fp)
                rep["items"][f"{m}/{p['id']}"] = r
        (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        print(f"{name}: {len(phrases)} фраз × {modes} → {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
