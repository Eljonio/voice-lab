#!/usr/bin/env python3
"""Точка отсчёта VoxCPM2 без дообучения (CPU): русская фраза, казахская фраза, голос по описанию.

Запуск:  .venv/bin/python synth_baseline.py [--lora путь/к/step_XXXX] [--only ru,kk,design] [--out samples/base]
Звук только в файлы (samples/), в колонки ничего не выводится.
С --lora тот же набор фраз гоняется через дообученную модель — так сравниваем чекпоинты на слух.
"""
import argparse
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("VOXCPM2_WEIGHTS", HERE / "weights" / "VoxCPM2"))

KK = "Сәлеметсіз бе! Мен Аружанмын, HelpLine компаниясының AI-менеджерімін. Қалай көмектесе аламын?"
RU = "Добрый день! Меня зовут Аружан, я AI-менеджер HelpLine. Чем помочь?"
# Voice Design: описание в скобках перед текстом (так в карточке модели)
DESIGN_RU = "(молодая казахская девушка, тёплый спокойный голос администратора)" + RU
DESIGN_KK = "(молодая казахская девушка, тёплый спокойный голос администратора)" + KK

JOBS = {
    "ru": RU,
    "kk": KK,
    "design": DESIGN_RU,
    "design_kk": DESIGN_KK,
    # для прослушивания чекпоинтов: живые фразы администратора, казахский со смешением
    "kk2": "Иә, жазып алдым. Ертең сағат онда шебер келеді, сізге ыңғайлы ма?",
    "kk3": "Ээ, кешіріңіз, нақты бағасын менеджер айтады. Нөміріңізді қалдырасыз ба?",
    "mix": "Жақсы, онда заявкаңызды қабылдадым, ну, бір сағат ішінде хабарласамыз.",
    "ru2": "Поняла вас. Мастер приедет завтра с десяти до двенадцати, вам удобно?",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="ru,kk,design", help="через запятую: " + ",".join(JOBS))
    ap.add_argument("--out", default=str(HERE / "samples" / "base"))
    ap.add_argument("--lora", default="")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default="", help="диапазон сидов, напр. 1-24: варианты голоса для выбора тембра (Voice Design)")
    ap.add_argument("--timesteps", type=int, default=10)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--ref", default="", help="эталон голоса (wav) — клонирование, например синтетический эталон из Voice Design")
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch

    from voxcpm import VoxCPM

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
    kw = dict(load_denoiser=False, optimize=False, device=a.device)
    if a.lora:
        kw["lora_weights_path"] = a.lora
    # Модель сначала строится в float32 (2B × 4 байта = 8 ГБ) и только потом приводится к bf16 — на ноуте с 15 ГБ
    # и параллельными задачами это убивал OOM. Строим сразу в bf16, потом возвращаем float32 по умолчанию.
    if a.device == "cpu":
        torch.set_default_dtype(torch.bfloat16)
    model = VoxCPM.from_pretrained(str(WEIGHTS), **kw)
    torch.set_default_dtype(torch.float32)
    print(f"модель загружена за {time.time() - t0:.0f} с", flush=True)
    sr = model.tts_model.sample_rate
    report = {}
    seeds = [a.seed]
    if a.seeds:
        lo, hi = (int(x) for x in a.seeds.split("-"))
        seeds = list(range(lo, hi + 1))
    for name, seed in [(n, sd) for n in a.only.split(",") for sd in seeds]:
        text = JOBS[name]
        torch.manual_seed(seed)
        np.random.seed(seed)
        t = time.time()
        # 09.10: сид — в сам generate: без него VoxCPM зовёт torch.seed() (случайный) и manual_seed выше ничего не фиксирует —
        # до 09.10 все пробы и 24 варианта тембра «по сидам» были невоспроизводимы
        gk = dict(text=text, cfg_value=2.0, inference_timesteps=a.timesteps, seed=seed)
        if a.ref:
            gk["reference_wav_path"] = a.ref
        wav = model.generate(**gk)
        dt = time.time() - t
        dur = len(wav) / sr
        tag = name if len(seeds) == 1 else f"{name}_s{seed:02d}"
        fp = out / f"voxcpm2_{tag}.wav"
        sf.write(fp, wav, sr)
        report[tag] = {"file": str(fp), "text": text, "audio_s": round(dur, 2), "compute_s": round(dt, 1),
                        "rtf": round(dt / max(dur, 1e-6), 1), "sr": sr}
        print(json.dumps({tag: report[tag]}, ensure_ascii=False), flush=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
