#!/usr/bin/env python3
"""Дословная расшифровка нарезанных фраз для обучения VoxCPM2.

Вход:  clean/manifest.csv конвейера voicelab (колонки file, duration, lang, speaker, passed, text_whisper)
       или --dir папка с .wav (для проверки).
Выход: <фраза>.txt рядом с каждым .wav  +  JSONL-манифест в формате VoxCPM2
       ({"audio", "text", "duration", "speaker", "lang", ...}) + CSV с метриками качества.

Модель по умолчанию — shyngys879/kazakh-whisper-large-v3-turbo (Apache-2.0, FLEURS kk WER 11,8 %
против 19,8 % у исходного turbo и 31,1 % у large-v3), сконвертированная в CTranslate2 для faster-whisper.
Русские фразы (lang=ru) идут той же моделью с language=ru (или --ru-model отдельной).

Дословность: модели ASR по умолчанию «чистят» речь (выкидывают «ээ», «мм», повторы). Против этого —
подсказка (initial_prompt) с заполнителями в живом стиле: Whisper подстраивает стиль расшифровки под неё.
Фразы с цифрами помечаются (has_digits): VoxCPM учится на тексте «как сказано», числа должны быть прописью.

Отсев (колонка ok=0 и причина): avg_logprob < -1.0, compression_ratio > 2.4, no_speech_prob > 0.6,
скорость < 2,5 или > 25 символов/с, расхождение двух ASR (CER > 0,5), пустой текст, цифры. В VoxCPM-манифест идут только ok=1.

  .venv/bin/python transcribe.py --manifest "$VOICE_DS/clean/manifest.csv"
  .venv/bin/python transcribe.py --dir test_asr/wavs --limit 10 --device cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HF_DIR = HERE / "weights" / "kazakh-whisper-large-v3-turbo"
CT2_DIR = HERE / "weights" / "kazakh-whisper-large-v3-turbo-ct2"

PROMPT = {
    # стиль живой речи с заполнителями и русскими вкраплениями, как говорят в Казахстане
    "kk": "Ээ, сәлеметсіз бе. Мм, иә, жаңағы, ну, мен айтайын, вот, яғни, қазір, ээ, айтайыншы.",
    "ru": "Ээ, ну, здравствуйте. Мм, да, короче, вот, это самое, ну, как бы, ээ.",
}
DIGITS = re.compile(r"\d")
FILLERS = {"э", "ээ", "эээ", "эм", "эмм", "м", "мм", "ммм", "хм", "аа", "ыы", "әә"}
PUNCT_CT2 = HERE / "weights" / "whisper-large-v3-turbo-ct2"


def _core(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower().replace("ё", "е"))


def transfer_punct(kk: str, base: str) -> str:
    """Казахская модель пишет точнее, но строчными и без знаков (училась на нормализованных текстах ISSAI).
    Исходный turbo ошибается в словах чаще, зато ставит знаки, заглавные и чаще оставляет «ээ/мм».
    Выравниваем слова (difflib) и переносим со второго на первый: знаки, заглавные, вставленные заполнители."""
    import difflib
    a, b = kk.split(), base.split()
    if not a or not b:
        return kk
    sm = difflib.SequenceMatcher(a=[_core(w) for w in a], b=[_core(w) for w in b], autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out += b[j1:j2]
        elif op == "insert":
            out += [w for w in b[j1:j2] if _core(w) in FILLERS]
        elif op == "replace":
            seg = list(a[i1:i2])
            if b[j1][:1].isupper():
                seg[0] = seg[0][:1].upper() + seg[0][1:]
            tail = re.search(r"[^\w]+$", b[j2 - 1])
            if tail:
                seg[-1] += tail.group(0)
            out += seg
        else:  # delete: слово есть только у казахской модели — оставляем
            out += a[i1:i2]
    t = " ".join(out)
    return t[:1].upper() + t[1:]


def cer(a: str, b: str) -> float:
    a, b = _core(a.replace(" ", "_")), _core(b.replace(" ", "_"))
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / max(1, len(a))


def load16k(path: str):
    """Своё чтение аудио в 16 кГц моно float32: faster-whisper 1.2 падает на новом PyAV (metadata_errors)."""
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        import soxr
        x = soxr.resample(x, sr, 16000)
    return np.ascontiguousarray(x, dtype=np.float32)


def ensure_ct2(src: Path, dst: Path, quant: str):
    if (dst / "model.bin").exists():
        return
    print(f"конвертирую {src.name} → CTranslate2 ({quant}) ...", flush=True)
    cmd = [str(Path(sys.executable).parent / "ct2-transformers-converter"), "--model", str(src),
           "--output_dir", str(dst), "--quantization", quant,
           "--copy_files", "tokenizer.json", "preprocessor_config.json"]
    subprocess.run(cmd, check=True)


def read_rows(a) -> list[dict]:
    rows = []
    if a.manifest:
        with open(a.manifest, newline="") as f:
            for r in csv.DictReader(f):
                if str(r.get("passed", "1")) not in ("1", "True", "true") or not r.get("file"):
                    continue
                rows.append(r)
    else:
        for p in sorted(Path(a.dir).rglob("*.wav")):
            rows.append({"file": str(p), "lang": a.lang, "speaker": p.stem.rsplit("-", 1)[0], "duration": ""})
    return rows[: a.limit] if a.limit else rows


def judge(text: str, segs, dur: float) -> tuple[int, str, dict]:
    m = {"avg_logprob": min((s.avg_logprob for s in segs), default=-9.0),
         "compression_ratio": max((s.compression_ratio for s in segs), default=0.0),
         "no_speech_prob": max((s.no_speech_prob for s in segs), default=1.0)}
    m["cps"] = round(len(text) / dur, 1) if dur else 0.0
    why = []
    if not text.strip():
        why.append("empty")
    if m["avg_logprob"] < -1.0:
        why.append("low_logprob")
    w = [_core(x) for x in text.split()]
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    if m["compression_ratio"] > 2.4 or len(grams) != len(set(grams)):
        why.append("repetition")   # галлюцинация Whisper: та же фраза дважды (на тесте — fleurs-kk-4)
    if m["no_speech_prob"] > 0.6:
        why.append("no_speech")
    if dur and not (2.5 <= m["cps"] <= 25):
        why.append("speed")
    if DIGITS.search(text):
        why.append("has_digits")
    return int(not why), ";".join(why), {k: round(v, 3) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="clean/manifest.csv от voicelab")
    src.add_argument("--dir", help="папка с .wav (проверка)")
    ap.add_argument("--lang", default="kk", help="язык для --dir: kk | ru")
    ap.add_argument("--out-jsonl", help="VoxCPM-манифест (по умолчанию рядом с входом: voxcpm_train.jsonl)")
    ap.add_argument("--device", default="auto", help="cuda | cpu | auto")
    ap.add_argument("--compute-type", default="", help="int8 (CPU) / float16 (GPU); пусто — автоматически")
    ap.add_argument("--model", default=str(CT2_DIR), help="папка CTranslate2-модели")
    ap.add_argument("--ru-model", default="", help="отдельная модель для русских фраз (например large-v3-turbo)")
    ap.add_argument("--no-prompt", action="store_true", help="без подсказки с заполнителями")
    ap.add_argument("--punct-model", default=str(PUNCT_CT2) if PUNCT_CT2.joinpath("model.bin").exists() else "",
                    help="второй проход исходным turbo: знаки, заглавные, заполнители + сверка (пусто — выкл.)")
    ap.add_argument("--max-disagree", type=float, default=0.5, help="CER между двумя ASR, выше — отсев (исходный turbo слаб в казахском, 0.25 выкидывал верные фразы)")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--dataset-id", type=int, default=0, help="dataset_id в VoxCPM-манифесте")
    ap.add_argument("--no-txt", action="store_true", help="не писать .txt рядом с wav")
    ap.add_argument("--resume", action="store_true", help="дописывать: пропустить фразы, уже записанные в .asr.csv")
    ap.add_argument("--until", default="", help="ЧЧ:ММ — после этого времени новые фразы не начинать (продолжить с --resume)")
    a = ap.parse_args()

    from faster_whisper import WhisperModel
    import soundfile as sf

    if a.model == str(CT2_DIR):
        ensure_ct2(HF_DIR, CT2_DIR, "float16")
    dev = a.device
    if dev == "auto":
        import ctranslate2
        dev = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    ctype = a.compute_type or ("float16" if dev == "cuda" else "int8")
    t0 = time.time()
    models = {"kk": WhisperModel(a.model, device=dev, compute_type=ctype, cpu_threads=a.threads)}
    models["ru"] = WhisperModel(a.ru_model, device=dev, compute_type=ctype, cpu_threads=a.threads) if a.ru_model else models["kk"]
    if a.punct_model:
        models["punct"] = WhisperModel(a.punct_model, device=dev, compute_type=ctype, cpu_threads=a.threads)
    print(f"ASR загружена за {time.time() - t0:.0f} с ({dev}, {ctype})", flush=True)

    rows = read_rows(a)
    base = Path(a.manifest).parent if a.manifest else Path(a.dir)
    out_jsonl = Path(a.out_jsonl) if a.out_jsonl else base / "voxcpm_train.jsonl"
    out_csv = out_jsonl.with_suffix(".asr.csv")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    sec_ok = 0.0
    t_all = time.time()
    mode = "w"
    if a.resume and out_csv.exists() and out_jsonl.exists():
        done = {r["file"] for r in csv.DictReader(open(out_csv, newline=""))}
        rows = [r for r in rows if str(Path(r["file"])) not in done]
        mode = "a"
        print(f"продолжение: уже расшифровано {len(done)}, осталось {len(rows)}", flush=True)
    deadline = None
    if a.until:
        import datetime as dt
        hh, mm = map(int, a.until.split(":"))
        now = dt.datetime.now()
        deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if deadline <= now:
            deadline += dt.timedelta(days=1)
    with open(out_jsonl, mode) as fj, open(out_csv, mode, newline="") as fc:
        cols = ["file", "lang", "speaker", "duration", "ok", "why", "text", "text_kk_raw", "text_base",
                "cer_kk_vs_base", "text_whisper_draft",
                "avg_logprob", "compression_ratio", "no_speech_prob", "cps", "asr_s"]
        wc = csv.DictWriter(fc, fieldnames=cols)
        if mode == "w":
            wc.writeheader()
        for i, r in enumerate(rows, 1):
            if deadline is not None:
                import datetime as dt
                if dt.datetime.now() >= deadline:
                    print(f"стоп по времени {a.until}: {i - 1}/{len(rows)}, дальше — с --resume", flush=True)
                    break
            wav = Path(r["file"])
            lang = "ru" if r.get("lang") == "ru" else "kk"   # kk и mix — казахской моделью
            dur = float(r["duration"]) if r.get("duration") else sf.info(str(wav)).duration
            t = time.time()
            segs, _ = models[lang].transcribe(
                load16k(str(wav)), language=lang, task="transcribe", beam_size=a.beam,
                initial_prompt=None if a.no_prompt else PROMPT[lang],
                condition_on_previous_text=False, vad_filter=False, temperature=[0.0, 0.2, 0.4])
            segs = list(segs)
            text = " ".join(s.text.strip() for s in segs).strip()
            raw_kk, base, dis = text, "", None
            if "punct" in models:
                audio = load16k(str(wav))
                bs, _ = models["punct"].transcribe(audio, language=lang, beam_size=a.beam,
                                                    initial_prompt=None if a.no_prompt else PROMPT[lang],
                                                    condition_on_previous_text=False, temperature=[0.0, 0.2])
                base = " ".join(x.text.strip() for x in bs).strip()
                dis = round(cer(raw_kk, base), 3)
                text = transfer_punct(raw_kk, base)
            ok, why, m = judge(text, segs, dur)
            if dis is not None and dis > a.max_disagree:
                why = ";".join(x for x in (why, "asr_disagree") if x)
                ok = 0
            rec = {"file": str(wav), "lang": r.get("lang", lang), "speaker": r.get("speaker", ""),
                   "duration": round(dur, 3), "ok": ok, "why": why, "text": text, "text_kk_raw": raw_kk,
                   "text_base": base, "cer_kk_vs_base": dis,
                   "text_whisper_draft": r.get("text_whisper", ""), **m, "asr_s": round(time.time() - t, 1)}
            wc.writerow(rec)
            if not a.no_txt:
                wav.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
            if ok:
                n_ok += 1
                sec_ok += dur
                fj.write(json.dumps({"audio": str(wav.resolve()), "text": text, "duration": round(dur, 3),
                                     "dataset_id": a.dataset_id, "speaker": rec["speaker"], "lang": rec["lang"]},
                                    ensure_ascii=False) + "\n")
            fc.flush()
            fj.flush()
            print(f"[{i}/{len(rows)}] {wav.name} {dur:.1f}s ok={ok} {why or ''} | {text[:110]}", flush=True)
    el = time.time() - t_all
    print(f"\nготово: {n_ok}/{len(rows)} фраз прошли, {sec_ok / 3600:.2f} ч; {el:.0f} с на всё")
    print(f"VoxCPM-манифест: {out_jsonl}\nметрики: {out_csv}")


if __name__ == "__main__":
    main()
