#!/usr/bin/env python3
"""nano_bench.py — проба Nano-vLLM-VoxCPM (github.com/a710128/nanovllm-voxcpm, MIT, PyPI nano-vllm-voxcpm 2.0.4) на ВЛИТЫХ весах
(merge_lora.py): скорость по одному и параллельно, файлы для GigaAM. Запуск — на GPU, в отдельном venv.

Что известно по коду репо (прочитан 11.10, коммит c702631): движок грузит *.safetensors + audiovae.pth из папки модели, q/k/v и
gate/up сливает сам; шаги диффузии задаются на весь пул (inference_timesteps), cfg — на запрос; эталон — ref_audio_latents
(encode_latents(wav)), продолжение — prompt_latents+prompt_text; выход — куски float32 потоком. Нужен flash-attn (импорт при старте).
Официальный Dockerfile собран под CUDA 12.6 и архитектуры 8.0–9.0 — Blackwell (sm_120, наша PRO 6000) там не заявлен: это и проверяем.

  python nano_bench.py --model $VOICE_HOME/merged/<имя> --ref voices/aruzhan_f_kk.wav --phrases eval/phrases_v2.json --out OUT/N10
Итог: OUT/N10/<id>.wav, OUT/N10/nano.json (TTFB, RTF по одному; параллельно 4 и 8: TTFB p50/p90, RTF на запрос).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from call_bench import DEFAULT_IDS, pick_phrases  # noqa: E402


async def one(srv, text, ref, sr, cfg, seed):
    t0 = time.perf_counter()
    t1, parts = None, []
    async for c in srv.generate(target_text=text, ref_audio_latents=ref, cfg_value=cfg, seed=seed):
        if t1 is None:
            t1 = time.perf_counter()
        parts.append(np.asarray(c, np.float32).reshape(-1))
    dt = time.perf_counter() - t0
    w = np.concatenate(parts) if parts else np.zeros(1, np.float32)
    aud = len(w) / sr
    return w, dict(ttfb_ms=round((t1 - t0) * 1000) if t1 else None, synth_s=round(dt, 3), audio_s=round(aud, 3),
                   rtf=round(dt / max(aud, 1e-6), 3), rtf_after_ttfb=round((dt - ((t1 or t0) - t0)) / max(aud, 1e-6), 3))


async def main_async(a):
    import soundfile as sf
    from nanovllm_voxcpm import VoxCPM
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(made=time.strftime("%F %T"), model=a.model, steps=a.steps, cfg=a.cfg)
    t = time.time()
    srv = VoxCPM.from_pretrained(model=a.model, inference_timesteps=a.steps, max_num_seqs=16, max_model_len=4096,
                                 gpu_memory_utilization=a.mem, devices=[0])
    await srv.wait_for_ready()
    info = await srv.get_model_info()
    sr = int(info["sample_rate"])
    res["load_s"] = round(time.time() - t, 1)
    print(f"nano: загрузка {res['load_s']} с, sr {sr}", flush=True)
    ref = await srv.encode_latents(Path(a.ref).read_bytes(), "wav")
    phrases = pick_phrases(Path(a.phrases), DEFAULT_IDS, 3, 2)
    for p in phrases[:2]:
        await one(srv, p["text"], ref, sr, a.cfg, a.seed)              # прогрев
    items = {}
    for p in phrases:
        w, m = await one(srv, p["text"], ref, sr, a.cfg, a.seed)
        sf.write(out / f"{p['id']}.wav", w, sr)
        items[p["id"]] = dict(m, lang=p["lang"])
    v = list(items.values())
    res["seq"] = dict(n=len(v), ttfb_med=statistics.median(x["ttfb_ms"] for x in v), rtf_med=statistics.median(x["rtf"] for x in v),
                      rtf_p90=sorted(x["rtf"] for x in v)[int(0.9 * (len(v) - 1))], items=items)
    print(f"nano по одному: TTFB мед {res['seq']['ttfb_med']} мс, RTF мед {res['seq']['rtf_med']} p90 {res['seq']['rtf_p90']}", flush=True)
    for n in (4, 8):
        batch = [phrases[i % len(phrases)] for i in range(n)]
        rs = await asyncio.gather(*[one(srv, p["text"], ref, sr, a.cfg, a.seed + i) for i, p in enumerate(batch)])
        ms = [m for _, m in rs]
        res[f"par{n}"] = dict(ttfb_p50=statistics.median(m["ttfb_ms"] for m in ms), ttfb_max=max(m["ttfb_ms"] for m in ms),
                              rtf_med=statistics.median(m["rtf"] for m in ms), rtf_max=max(m["rtf"] for m in ms))
        print(f"nano параллельно {n}: TTFB p50 {res[f'par{n}']['ttfb_p50']} мс (макс {res[f'par{n}']['ttfb_max']}), "
              f"RTF на запрос мед {res[f'par{n}']['rtf_med']} (макс {res[f'par{n}']['rtf_max']})", flush=True)
    (out / "nano.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    await srv.stop()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--phrases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mem", type=float, default=0.5)
    a = ap.parse_args()
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    sys.exit(main())
