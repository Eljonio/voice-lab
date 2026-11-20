#!/usr/bin/env python3
"""Конвейер v2: ролик (FLAC) → чистые фразы 16 кГц FLAC + дословный текст + метрики. Один процесс = один воркер.

Что поменялось против v1 (docs/METHOD.md):
  * нарезка ТОЛЬКО по паузам VAD ≥ 0,3 с; кусок речи > 20 с без такой паузы не режется, а отбрасывается (no_pause);
  * расшифровка без подсказки (в v1 подсказка дописывала «жаңағы»): основной ASR — казахский turbo (kk/mix) или исходный turbo (ru),
    знаки и заглавные — с исходного turbo; второй, независимый ASR — GigaAM-Multilingual CTC (MIT): CER между ними,
    таймкоды слов (обрезано ли слово на краю), уверенность CTC;
  * нормализатор текста (dataprep/textnorm.py): цифры/латиница/символы → строка не в обучение; скобки, кавычки, КАПС, гомоглифы — чистятся;
  * на выходе 16 кГц FLAC (частота энкодера VoxCPM2), всё возобновляемо по роликам: work/<clip>/result.json.

Очередь — общая для всех воркеров: файл берётся «захватом» work/<clip>/.claim (O_EXCL), порядок — order.txt
(лучшие каналы первыми). Файла ещё нет на диске (перекачка идёт) — ждём, пока не появится флаг конца перекачки.

  python dataprep/process_v2.py --raw "$VOICE_HOME/raw" --out "$VOICE_HOME/v2" --order order.txt --device cuda [--until-ts EPOCH]
  python dataprep/process_v2.py --raw DIR --out DIR --order order.txt --device cuda --asr-device cuda --giga-device cpu --max-sec 180  # проверка на ноуте
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "voxcpm2", ROOT / "dataprep", ROOT / "scripts"):
    sys.path.insert(0, str(p))

from voicelab import audio as A  # noqa: E402
from voicelab.music import span_stats, summarize  # noqa: E402
from textnorm import count_fillers, normalize  # noqa: E402

SR = 16000
GIGA_ID = "ai-sage/GigaAM-Multilingual"
GIGA_REV = os.environ.get("GIGAAM_REV", "3905cd51c3ed4e88c8edf33f3302969ba480a327")  # ветка large_ctc
HARD = {"empty", "digits", "latin", "symbols", "low_logprob", "repetition", "no_speech", "speed", "asr_disagree",
        "cut_end", "cut_start", "has_digits"}


class Cfg:
    # нарезка
    min_len = 1.5          # короткие законченные реплики тоже нужны; DNSMOS на коротком зацикливает звук
    max_len = 20.0         # склейка единиц в фразу — не длиннее (клип + ref ≤ 30 с)
    unit_max = 25.0        # 08.10: одна единица без паузы ≥ 0,3 с — до 25 с целиком (официально VoxCPM до 30 с; ref тогда не даём —
                           # build_mix даёт ref только при клип + ref ≤ 30 с). Было 20 → «без пауз» 44 ч отсева сессии 1 = быстрая живая речь
    cut_gap = 0.30         # резать можно только на паузе ≥ 0,3 с (по VAD с полями 40 мс: реальная ≥ 0,38 с)
    split_gap = 0.20       # кусок > 20 с без такой паузы — делим по самой длинной паузе ≥ 0,2 с (реальная ≥ 0,28 с)
    fine_gap = 0.0         # 08.10: тонкий VAD (микропаузы 0,12 с, без полей) выключен — режет посреди фразы, поля ≤ 60 мс, обрезает
                           # начала слов (на слух: обрезается интонация). Вернуть: 0.12. Используется и в --respan.
    sent_gap = 0.80        # пауза ≥ 0,8 с — конец мысли: режем, если набралось ≥ min_len
    soft_len = 12.0        # 08.10: было 8 с / 0,5 с — резали посреди мысли на вдохе; теперь после 12 с на паузе ≥ 0,6 с
    soft_gap = 0.60
    pad = 0.15             # поля по краям (в конце тишины ≤ 0,15 + 0,04 с)
    # музыка (AST)
    music_thr = 0.25
    ast_win = 3.0
    ast_hop = 1.5
    clip_music_frac = 0.10
    use_demucs = False      # сессия 2: Demucs выключен (замер сессии 1: у 91 % фраз с текстом из «музыкальных» роликов музыки в исходнике
                            # < 0,30 — медиана 0,02; Demucs = 28 % времени; по методу звук после Demucs в финал и так не берём)
    clip_music_peak = 1.01  # сессия 2: триггер «хоть одно окно ≥ 0,5» выключен. Замер сессии 1 (163 ролика): у 92 роликов музыки
                            # < 10 % окон (заставки) — 67 % сырого, выход текста 30 %, но Demucs шёл на ~половине из них (23 % их времени).
                            # Без Demucs фразы под музыкой всё равно отсеет music_after (= музыка исходника) > 0,30.
    # отсев фраз (пороги v1, voicelab/pipeline.py)
    max_music_after = 0.30
    max_sing_after = 0.30
    # 08.10: в обработке пороги мягче — пограничные фразы тоже расшифровываются и сохраняются, строгие пороги v1 (2,7 / 3,0 / 2,8 / 12 дБ)
    # ставит finalize_v2.py (--strict). Так ослабить фильтр можно позже без новой обработки (живость: эмоции/смех DNSMOS ценит ниже).
    min_ovrl = 2.50
    min_sig = 2.80
    min_bak = 2.60
    min_snr = 10.0
    min_spk_sim = 0.15
    max_clip = 0.005
    langs_ok = ("kk", "ru", "mix")
    spk_cluster_thr = 0.35
    # расшифровка
    max_disagree = 0.25    # CER(основной ASR, GigaAM) — выше в обучение не идёт (калибровать на 200 фразах носителем)
    min_tail = 0.08        # последнее слово кончается ≥ 80 мс до конца клипа
    min_head = 0.04        # первое начинается ≥ 40 мс от начала (кадр GigaAM 40 мс)
    beam = 5


# ---------------------------------------------------------------- нарезка только по паузам
class PauseSegmenter:
    def __init__(self, cfg: Cfg):
        from silero_vad import load_silero_vad
        self.vad = load_silero_vad()
        self.c = cfg

    def speech(self, x: np.ndarray) -> list[tuple[float, float]]:
        import torch
        from silero_vad import get_speech_timestamps
        ts = get_speech_timestamps(torch.from_numpy(x), self.vad, sampling_rate=SR, threshold=0.5,
                                   min_speech_duration_ms=200, min_silence_duration_ms=120, speech_pad_ms=40)
        return [(t["start"] / SR, t["end"] / SR) for t in ts]

    def _fine(self, x: np.ndarray, s: float, e: float) -> list[tuple[float, float]]:
        """Тонкий VAD внутри длинного куска: без полей, тишина от 60 мс (сессия 1: 70 % кусков «без пауз» имеют паузы ≥ 0,12 с)."""
        import torch
        from silero_vad import get_speech_timestamps
        ts = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(x[int(s * SR):int(e * SR)])), self.vad, sampling_rate=SR,
                                   threshold=0.5, min_speech_duration_ms=60, min_silence_duration_ms=60, speech_pad_ms=0)
        fs = [(s + t["start"] / SR, s + t["end"] / SR) for t in ts]
        if fs:
            fs[0], fs[-1] = (s, fs[0][1]), (fs[-1][0], e)
        return fs

    def _split(self, segs: list[tuple[float, float]], x: np.ndarray | None = None, fine: bool = False) -> list[tuple[float, float, bool]]:
        """Единица длиннее max_len: делим по самой длинной внутренней паузе (≥ split_gap по обычному VAD; если такой нет —
        по тонкому VAD, пауза ≥ fine_gap), пока куски не влезут. Нет и такой — кусок в отсев (no_pause): внутри слова не режем."""
        s, e = segs[0][0], segs[-1][1]
        if e - s <= self.c.unit_max:
            return [(s, e, False)]
        gaps = [(segs[i + 1][0] - segs[i][1], i) for i in range(len(segs) - 1)]
        g, i = max(gaps, default=(0.0, -1))
        if g < (self.c.fine_gap if fine else self.c.split_gap):
            if not fine and x is not None and self.c.fine_gap > 0:
                fs = self._fine(x, s, e)
                if len(fs) > 1:
                    return self._split(fs, x, fine=True)
            return [(s, e, True)]
        return self._split(segs[: i + 1], x, fine) + self._split(segs[i + 1:], x, fine)

    def span_phrases(self, x: np.ndarray, spans: list[tuple[float, float]]) -> list[dict]:
        """Фразы только из заданных кусков (переобработка «без пауз» сессии 1): тонкое деление + сборка до max_len."""
        out = []
        for s, e in spans:
            units = self._split(self._fine(x, s, e) or [(s, e)], x, fine=True)
            out += self._assemble(units, x, lo=s, hi=e)
        return out

    def phrases(self, x: np.ndarray) -> list[dict]:
        c = self.c
        segs = self.speech(x)
        # 1) «единицы»: всё, что разделено паузой < 0,3 с, — вместе (длинные делятся в _split)
        groups: list[list[tuple[float, float]]] = []
        for s, e in segs:
            if groups and s - groups[-1][-1][1] < c.cut_gap:
                groups[-1].append((s, e))
            else:
                groups.append([(s, e)])
        units = [u for g in groups for u in self._split(g, x)]
        return self._assemble(units, x, lo=0.0, hi=len(x) / SR)

    def _assemble(self, units, x, lo: float, hi: float) -> list[dict]:
        """Склейка единиц (между ними паузы ≥ cut_gap) во фразы ≤ max_len. Режем на конце мысли (пауза ≥ sent_gap), после soft_len —
        на паузе ≥ soft_gap. 08.10: если фраза переполняется, режем не на последней границе, а на САМОЙ ДЛИННОЙ паузе внутри неё
        (скорее конец предложения) — раньше резали где придётся, и фраза кончалась посреди мысли."""
        c = self.c
        out: list[dict] = []
        cur: list[tuple[float, float]] = []

        def emit(part):
            out.append(dict(s=part[0][0], e=part[-1][1], no_pause=False))
        for s, e, nop in units:
            if nop:
                if cur:
                    emit(cur)
                    cur = []
                out.append(dict(s=s, e=e, no_pause=True))
                continue
            if not cur:
                cur = [(s, e)]
                continue
            gap, cur_len, new_len = s - cur[-1][1], cur[-1][1] - cur[0][0], e - cur[0][0]
            if gap > 1.5 or (gap >= c.sent_gap and cur_len >= c.min_len) or (gap >= c.soft_gap and cur_len >= c.soft_len):
                emit(cur)
                cur = [(s, e)]
                continue
            cur.append((s, e))
            while len(cur) > 1 and cur[-1][1] - cur[0][0] > c.max_len:
                # границы k|k+1: левая часть ≥ min_len и ≤ max_len; берём самую длинную паузу (при равенстве — позднюю)
                cand = [(cur[k + 1][0] - cur[k][1], k) for k in range(len(cur) - 1)
                        if c.min_len <= cur[k][1] - cur[0][0] <= c.max_len]
                k = max(cand)[1] if cand else len(cur) - 2
                emit(cur[: k + 1])
                cur = cur[k + 1:]
        if cur:
            emit(cur)
        res = []
        for i, p in enumerate(out):
            left = out[i - 1]["e"] if i > 0 else lo
            right = out[i + 1]["s"] if i + 1 < len(out) else hi
            a = max(p["s"] - c.pad, (left + p["s"]) / 2, lo)
            b = min(p["e"] + c.pad, (p["e"] + right) / 2, hi)
            res.append(dict(start=round(a, 3), end=round(b, 3), dur=round(b - a, 3),
                            speech_dur=round(p["e"] - p["s"], 3), no_pause=p["no_pause"]))
        return res


# ---------------------------------------------------------------- модели
class GigaCTC:
    """GigaAM-Multilingual large_ctc: текст, слова с таймкодами, уверенность CTC (средняя макс. вероятность не-пустых кадров)."""

    def __init__(self, device: str):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.m = AutoModel.from_pretrained(GIGA_ID, revision=GIGA_REV, trust_remote_code=True).to(device).eval()
        self.inner = self.m.model
        self.dev = device

    def __call__(self, x: np.ndarray) -> dict:
        torch, inner = self.torch, self.inner
        with torch.inference_mode():
            wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.dev)[None]
            length = torch.full([1], wav.shape[-1], device=self.dev)
            enc, enc_len = inner.forward(wav, length)
            lp = inner.head(encoder_output=enc)[0, : int(enc_len[0])]
            text, words = inner._decode(enc, enc_len, length, word_timestamps=True)[0]
            mx = lp.max(-1)
            nonblank = mx.indices != lp.shape[-1] - 1
            conf = float(mx.values[nonblank].exp().mean()) if bool(nonblank.any()) else 0.0
        words = [(w.text, float(w.start), float(w.end)) for w in (words or [])]
        return dict(text=text.strip(), words=words, conf=round(conf, 4))


class FastDNSMOS:
    """DNSMOS P.835 (sig/bak/ovrl) без P.808 (в v2 не нужен, а его мел-спектр на CPU — самое дорогое). GPU, если есть onnxruntime-gpu."""

    def __init__(self, gpu: bool):
        import onnxruntime as ort
        from voicelab.quality import MODEL_DIR, _P_BAK, _P_OVR, _P_SIG
        self.P = (_P_SIG, _P_BAK, _P_OVR)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.ort, self.opts, self.path = ort, opts, str(MODEL_DIR / "sig_bak_ovr.onnx")
        self.sess = None
        if gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            try:
                self.sess = ort.InferenceSession(self.path, opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                self.sess.run(None, {"input_1": np.zeros((1, int(9.01 * SR)), dtype=np.float32)})
            except Exception as e:  # noqa: BLE001 — Blackwell/драйвер: тогда CPU
                print(f"DNSMOS на GPU не пошёл ({e}); CPU", file=sys.stderr)
                self.sess = None
        if self.sess is None:
            self.sess = ort.InferenceSession(self.path, opts, providers=["CPUExecutionProvider"])
        self.provider = self.sess.get_providers()[0]

    def __call__(self, x: np.ndarray, hop_s: float = 3.0) -> dict:
        need = int(9.01 * SR)
        x = x.astype(np.float32)
        if len(x) == 0:
            return dict(sig=np.nan, bak=np.nan, ovrl=np.nan)
        while len(x) < need:
            x = np.append(x, x)
        starts = np.arange(0, len(x) - need + 1, int(hop_s * SR))
        batch = np.stack([x[s: s + need] for s in starts])
        raw = np.concatenate([self.sess.run(None, {"input_1": batch[i: i + 1]})[0] for i in range(len(batch))])
        sig, bak, ovr = (self.P[k](raw[:, k]) for k in range(3))
        return dict(sig=float(np.mean(sig)), bak=float(np.mean(bak)), ovrl=float(np.mean(ovr)))


class FastLID:
    """Язык фразы как в v1 (voicelab/lang.py: Whisper + VoxLingua107, тюркские близкие = «kk-подобный»), но Whisper только
    определяет язык (detect_language, без расшифровки) — расшифровка дальше делается только для прошедших фраз."""

    def __init__(self, whisper, device: str):
        from voicelab.lang import TURKIC_NEAR, LangID
        self.TURKIC = TURKIC_NEAR
        self.w = whisper
        tmp = LangID.__new__(LangID)
        from speechbrain.inference.classifiers import EncoderClassifier
        from voicelab.lang import VOX_CACHE, VOX_SRC
        src = str(VOX_SRC) if (VOX_SRC / "embedding_model.ckpt").exists() else "speechbrain/lang-id-voxlingua107-ecapa"
        tmp.vox = EncoderClassifier.from_hparams(source=src, savedir=str(VOX_CACHE), run_opts={"device": device})
        codes = {i: lab.split(":")[0].strip() for i, lab in tmp.vox.hparams.label_encoder.ind2lab.items()}
        tmp.vox_codes = codes
        tmp.vox_turkic = [i for i, c in codes.items() if c in TURKIC_NEAR]
        tmp.vox_ru = [i for i, c in codes.items() if c == "ru"][0]
        tmp.vox_kk = [i for i, c in codes.items() if c == "kk"][0]
        tmp.device = device
        self.vox = tmp

    KK_LETTERS = re.compile(r"[әғқңөұүһі]")
    CYR = re.compile(r"[а-яё]")

    def __call__(self, x: np.ndarray, giga_text: str = "") -> dict:
        """08.10 (проверка на слух: 39 из 40 роликов с низким p_kk Whisper — чистый казахский): Whisper-LID казахский не узнаёт.
        Решает текст GigaAM CTC (обучен на kk/ru/ky/uz/en): есть буквы ә ғ қ ң ө ұ ү һ і — казахский (или mix, если
        русский заметен); кириллица без них — русский или короткая казахская фраза; не кириллица — other."""
        lang, prob, allp = self.w.detect_language(x[: 30 * SR])
        probs = dict(allp or [])
        w_kk = sum(probs.get(c, 0.0) for c in self.TURKIC)
        w_ru = probs.get("ru", 0.0)
        v = self.vox._vox(x)
        kk, ru = (w_kk + v["kk_like"]) / 2, (w_ru + v["ru"]) / 2
        g = (giga_text or "").lower()
        if self.KK_LETTERS.search(g):
            out = "mix" if ru >= 0.35 else "kk"
        elif self.CYR.search(g):
            out = "kk" if kk > ru else "ru"
        elif not g.strip():
            out = "kk" if kk >= 0.5 else "ru" if ru >= 0.5 else "other"
        else:
            out = "other"
        return dict(lang=out, lang_top=f"{lang}/{v['top']}", p_kk_like=kk, p_ru=ru)


class Models:
    def __init__(self, a, cfg: Cfg):
        from faster_whisper import WhisperModel
        from voicelab.music import MusicTagger
        from voicelab.separate import Separator
        from voicelab.speakers import SpeakerChecker
        t = time.time()
        dev, adev = a.device, a.asr_device or a.device
        self.tagger = MusicTagger(dev)
        self.sep = Separator("htdemucs", dev)
        self.dnsmos = FastDNSMOS(dev == "cuda")
        self.seg = PauseSegmenter(cfg)
        self.spk = SpeakerChecker(dev)
        ct = "float16" if adev == "cuda" else "int8"
        if adev == "cuda" and a.int8:
            ct = "int8_float16"
        self.base = WhisperModel(a.base_ct2, device=adev, compute_type=ct, cpu_threads=a.threads)   # язык + знаки (+ ru)
        self.kk = WhisperModel(a.kk_ct2, device=adev, compute_type=ct, cpu_threads=a.threads)       # второе мнение kk
        self.lid = FastLID(self.base, dev)
        self.giga = GigaCTC(a.giga_device or adev)
        self.load_s = time.time() - t


# ---------------------------------------------------------------- один ролик
def whisper_text(model, x, lang: str, beam: int):
    segs, info = model.transcribe(x, language=lang, task="transcribe", beam_size=beam, initial_prompt=None,
                                  condition_on_previous_text=False, vad_filter=False, without_timestamps=True,
                                  temperature=[0.0, 0.2, 0.4])
    segs = list(segs)
    return " ".join(s.text.strip() for s in segs).strip(), segs


def asr_phrase(m: Models, cfg: Cfg, x: np.ndarray, lid: dict, g: dict | None = None) -> dict:
    """Слова — GigaAM CTC (FLEURS kk WER 4,4 % против 11,8 % у казахского turbo; CTC без языковой модели не «досочиняет»),
    знаки, заглавные и «ээ/мм» — переносом с исходного turbo; второе мнение — казахский turbo (kk) или исходный turbo (ru).
    Расхождение GigaAM и второго мнения > max_disagree — текст ненадёжен, в обучение не идёт."""
    from gigaam_eval import cer as cer_chars, norm as norm_cmp
    from transcribe import judge, transfer_punct
    lang = "ru" if lid.get("lang") == "ru" else "kk"
    dur = len(x) / SR
    t = time.time()
    g = g or m.giga(x)
    if lang == "kk":
        second, segs = whisper_text(m.kk, x, "kk", cfg.beam)
        punct, _ = whisper_text(m.base, x, "kk", 1)
    else:
        second, segs = whisper_text(m.base, x, "ru", cfg.beam)
        punct = second
    words = g["text"] or second
    text = transfer_punct(words, punct) if punct else words
    clean, probs = normalize(text)
    ok, why, wm = judge(clean, segs, dur)
    why = [w for w in why.split(";") if w] + probs
    r_g, r_s = norm_cmp(g["text"]), norm_cmp(second)
    dis = round(cer_chars(r_g, r_s), 4) if r_g else 1.0
    if dis > cfg.max_disagree:
        why.append("asr_disagree")
    head = g["words"][0][1] if g["words"] else None
    tail = dur - g["words"][-1][2] if g["words"] else None
    if tail is not None and tail < cfg.min_tail:
        why.append("cut_end")
    if head is not None and head < cfg.min_head:
        why.append("cut_start")
    fill = count_fillers(clean)
    if fill and fill / max(dur, 1.0) * 10 > 2:
        why.append("fillers_dense")          # мягкая причина: в основной объём можно, в финальный поднабор — нет
    why = sorted(set(why) - {""})
    return dict(asr_lang=lang, text=clean, text_giga=g["text"], text_second=second, text_punct=punct, cer_giga=dis,
                ctc_conf=g["conf"], head_gap=None if head is None else round(head, 3),
                tail_gap=None if tail is None else round(tail, 3), n_fillers=fill,
                asr_ok=int(not (set(why) & HARD)), asr_why=";".join(why), asr_s=round(time.time() - t, 2),
                **{k: wm[k] for k in ("avg_logprob", "compression_ratio", "no_speech_prob", "cps")})


def prepare(path: Path, m: Models, cfg: Cfg, T: dict, max_sec: float = 0, force_sep: bool | None = None):
    """Декодирование, детектор музыки, Demucs (если музыка). → x16 (исходный), v16 (голос), tl_b, tl_a, has_music."""
    t = time.time()
    x16 = A.load(path, SR, channels=1)
    if max_sec:
        x16 = x16[: int(max_sec * SR)]
    T["decode"] = time.time() - t
    t = time.time()
    tl_b = m.tagger.timeline(x16, cfg.ast_win, cfg.ast_hop)
    s_b = summarize(tl_b, cfg.music_thr)
    has_music = s_b["music_frac"] >= cfg.clip_music_frac or float(np.max(tl_b["music"])) >= cfg.clip_music_peak
    sep = has_music and cfg.use_demucs
    if force_sep is not None:
        sep = force_sep
    T["music"] = time.time() - t
    v16, tl_a = x16, tl_b
    if sep:
        t = time.time()
        x44 = A.load(path, 44100, channels=2)
        if max_sec:
            x44 = x44[:, : int(max_sec * 44100)]
        voc44, _ = m.sep(x44)
        del x44
        v16 = A.resample(A.to_mono(voc44), 44100, SR)
        del voc44
        T["demucs"] = time.time() - t
        t = time.time()
        tl_a = m.tagger.timeline(v16, cfg.ast_win, cfg.ast_hop)
        T["music"] += time.time() - t
    return x16, v16, tl_b, tl_a, s_b, has_music, sep


def analyze(phrases: list[dict], x16, v16, tl_a, has_music: bool, m: Models, cfg: Cfg, cid: str, cdir: Path, T: dict,
            idx0: int = 0) -> tuple[list[dict], dict, float, float]:
    """Проверки качества, диктор, язык, расшифровка, запись FLAC по каждой фразе. → rows, центры голосов, чисто с, текст ок с."""
    T.update({k: T.get(k, 0.0) for k in ("dnsmos", "speaker", "lid", "asr")})
    rows, embs = [], []
    for j, p in enumerate(phrases):
        i = idx0 + j
        a, b = int(p["start"] * SR), int(p["end"] * SR)
        seg = v16[a:b]
        r = dict(clip_id=cid, idx=i, start=p["start"], end=p["end"], duration=p["dur"], speech_dur=p["speech_dur"],
                 no_pause=int(p["no_pause"]), separated=int(has_music))
        sa = span_stats(tl_a, p["start"], p["end"])
        r.update(music_after=round(sa["music"], 3), sing_after=round(sa["sing"], 3))
        if not p["no_pause"] and p["dur"] >= cfg.min_len and r["music_after"] <= cfg.max_music_after \
                and r["sing_after"] <= cfg.max_sing_after:
            t1 = time.time()
            q = m.dnsmos(seg, hop_s=3.0)
            T["dnsmos"] += time.time() - t1
            r.update(dnsmos_sig=round(q["sig"], 3), dnsmos_bak=round(q["bak"], 3), dnsmos_ovrl=round(q["ovrl"], 3))
            r["snr_db"] = round(A.snr_estimate(seg, SR), 1)
            r["clip_ratio"] = round(A.clipping_ratio(x16[a:b]), 5)
            q_ok = r["dnsmos_bak"] >= cfg.min_bak and r["dnsmos_ovrl"] >= cfg.min_ovrl and r["dnsmos_sig"] >= cfg.min_sig \
                and not (r["snr_db"] == r["snr_db"] and r["snr_db"] < cfg.min_snr) and r["clip_ratio"] <= cfg.max_clip
            if q_ok:   # дорогие проверки — только для прошедших по качеству (причина отсева = первая сработавшая)
                t1 = time.time()
                sc = m.spk.phrase_check(seg)
                T["speaker"] += time.time() - t1
                r["spk_min_sim"] = round(sc["min_sim"], 3)
                embs.append((i, sc["emb"]))
                if r["spk_min_sim"] >= cfg.min_spk_sim:
                    t1 = time.time()
                    g = m.giga(A.normalize_rms(seg))       # нужен и для языка, и потом для текста
                    li = m.lid(seg, g["text"])
                    T["lid"] += time.time() - t1
                    r.update(lang=li["lang"], lang_top=li["lang_top"], p_kk_like=round(li["p_kk_like"], 3),
                             p_ru=round(li["p_ru"], 3), _giga=g)
        rows.append(r)
    from voicelab.speakers import cluster
    labels, cents = {}, {}
    if embs:
        E = np.stack([e for _, e in embs])
        lab = cluster(E, cfg.spk_cluster_thr) if len(embs) > 1 else np.zeros(1, dtype=int)
        labels = {i: int(l) for (i, _), l in zip(embs, lab)}
        for l in set(labels.values()):
            v = E[[k for k, (i, _) in enumerate(embs) if labels[i] == l]].mean(0)
            cents[int(l)] = (v / (np.linalg.norm(v) + 1e-9)).round(4).tolist()
    clean_sec = asr_sec = 0.0
    import soundfile as sf
    for r in rows:
        why = []
        if r["no_pause"]:
            why.append("no_pause")
        elif r["duration"] < cfg.min_len:
            why.append("too_short")
        else:
            if r["music_after"] > cfg.max_music_after:
                why.append("music_left")
            elif r["sing_after"] > cfg.max_sing_after:
                why.append("singing")
            elif r["dnsmos_bak"] < cfg.min_bak or (r["snr_db"] == r["snr_db"] and r["snr_db"] < cfg.min_snr):
                why.append("noise")
            elif r["dnsmos_ovrl"] < cfg.min_ovrl or r["dnsmos_sig"] < cfg.min_sig:
                why.append("low_quality")
            elif r["clip_ratio"] > cfg.max_clip:
                why.append("clipping")
            elif r.get("spk_min_sim", 0) < cfg.min_spk_sim:
                why.append("multi_speaker")
            elif r.get("lang") not in cfg.langs_ok:
                why.append("lang_other")
        r["speaker"] = f"{cid}_spk{labels[r['idx']]}" if r["idx"] in labels else ""
        r["clean_ok"] = int(not why)
        r["clean_why"] = ";".join(why)
        lid = {"lang": r.get("lang")}
        g_pre = r.pop("_giga", None)
        if r["clean_ok"]:
            a, b = int(r["start"] * SR), int(r["end"] * SR)
            seg = A.normalize_rms(v16[a:b])
            t1 = time.time()
            try:
                r.update(asr_phrase(m, cfg, seg, lid, g_pre))
            except Exception as e:  # noqa: BLE001 — одна фраза не роняет ролик
                r.update(asr_ok=0, asr_why=f"asr_error:{type(e).__name__}", text="")
            T["asr"] += time.time() - t1
            fp = cdir / f"{cid}_{r['idx']:04d}.flac"
            fp.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(fp), np.clip(seg, -1, 1), SR, format="FLAC", subtype="PCM_16")
            r["file"] = str(fp)
            clean_sec += r["duration"]
            if r.get("asr_ok"):
                asr_sec += r["duration"]
    return rows, cents, clean_sec, asr_sec


def write_json(fp: Path, obj: dict):
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=float))
    os.replace(tmp, fp)


def process_clip(path: Path, m: Models, cfg: Cfg, out: Path, meta: dict, max_sec: float = 0) -> dict:
    T: dict[str, float] = {}
    t_all = time.time()
    cid = path.stem
    x16, v16, tl_b, tl_a, s_b, has_music, sep = prepare(path, m, cfg, T, max_sec)
    s_a = summarize(tl_a, cfg.music_thr)
    t = time.time()
    phrases = m.seg.phrases(v16)
    T["vad"] = time.time() - t
    rows, cents, clean_sec, asr_sec = analyze(phrases, x16, v16, tl_a, sep, m, cfg, cid, out / "clean" / cid, T)
    T["total"] = time.time() - t_all
    dur = len(x16) / SR
    res = dict(clip_id=cid, source=str(path), meta=meta, duration=dur, has_music=bool(has_music), separated=bool(sep),
               music_before=s_b, music_after=s_a, n_phrases=len(rows),
               speech_sec=float(sum(r["speech_dur"] for r in rows)), clean_sec=clean_sec, asr_sec=asr_sec,
               n_speakers=len(cents), speaker_centroids=cents, timings={k: round(v, 2) for k, v in T.items()},
               rtf=round(T["total"] / max(dur, 1e-6), 4), phrases=rows, made=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(out / "work" / cid / "result.json", res)
    return res


def process_respan(path: Path, m: Models, cfg: Cfg, out: Path) -> dict | None:
    """Сессия 2: куски «без пауз» из result.json сессии 1 — тонкое деление и полный разбор только их (respan.json рядом;
    finalize_v2 подставляет эти фразы вместо no_pause). Музыка/Demucs считаются только по самим кускам (±0,3 с), не по ролику."""
    cid = path.stem
    base = json.loads((out / "work" / cid / "result.json").read_text())
    spans = [(p["start"], p["end"]) for p in base["phrases"] if p.get("no_pause")]
    if not spans:
        return None
    T: dict[str, float] = {}
    t_all = time.time()
    sep = cfg.use_demucs and bool(base.get("separated", base.get("has_music")))   # 08.10: Demucs только если включён (--demucs)
    x16 = A.load(path, SR, channels=1)
    x44 = A.load(path, 44100, channels=2) if sep else None
    rows_all, cents_all, clean_sec, asr_sec, k = [], {}, 0.0, 0.0, 0
    for s0, e0 in spans:
        o = max(0.0, s0 - 0.3)
        e1 = min(len(x16) / SR, e0 + 0.3)
        xc = x16[int(o * SR):int(e1 * SR)]
        if sep:
            t = time.time()
            voc, _ = m.sep(x44[:, int(o * 44100):int(e1 * 44100)])
            vc = A.resample(A.to_mono(voc), 44100, SR)[: len(xc)]
            T["demucs"] = T.get("demucs", 0.0) + time.time() - t
        else:
            vc = xc
        t = time.time()
        tl = m.tagger.timeline(vc, cfg.ast_win, cfg.ast_hop)
        T["music"] = T.get("music", 0.0) + time.time() - t
        t = time.time()
        ph = m.seg.span_phrases(vc, [(s0 - o, min(e0, e1) - o)])
        T["vad"] = T.get("vad", 0.0) + time.time() - t
        rows, cents, cs, asr = analyze(ph, xc, vc, tl, sep, m, cfg, cid, out / "clean" / cid, T, idx0=5000 + k)
        k += len(rows)
        for r in rows:   # время — от начала ролика; голоса переобработки — свои номера (100+span*10+…), не смешивать с сессией 1
            r["start"], r["end"] = round(r["start"] + o, 3), round(r["end"] + o, 3)
            if r.get("speaker"):
                r["speaker"] = f"{cid}_spk{100 + len(cents_all) + int(r['speaker'].rsplit('spk', 1)[1])}"
        cents_all.update({100 + len(cents_all) + int(kk): v for kk, v in cents.items()})
        rows_all += rows
        clean_sec += cs
        asr_sec += asr
    T["total"] = time.time() - t_all
    res = dict(clip_id=cid, spans=spans, spans_sec=float(sum(e - s for s, e in spans)), n_phrases=len(rows_all),
               clean_sec=clean_sec, asr_sec=asr_sec, speaker_centroids_respan=cents_all, timings={k_: round(v, 2) for k_, v in T.items()},
               phrases=rows_all, made=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(out / "work" / cid / "respan.json", res)
    return res


# ---------------------------------------------------------------- очередь
def done(out: Path, cid: str) -> bool:
    f = out / "work" / cid / "result.json"
    try:
        return f.stat().st_size > 0 and json.loads(f.read_text()).get("clip_id") == cid
    except Exception:  # noqa: BLE001 — битый/пустой после жёсткого выключения → переделать
        return False


def need_respan(out: Path, cid: str) -> bool:
    d = out / "work" / cid
    if (d / "respan.json").exists() or not done(out, cid):
        return False
    try:
        return any(p.get("no_pause") for p in json.loads((d / "result.json").read_text())["phrases"])
    except Exception:  # noqa: BLE001
        return False


def claim(out: Path, cid: str, name: str = ".claim") -> bool:
    d = out / "work" / cid
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(d / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--order", type=Path, required=True, help="файл: по строке «<id>\\t<json meta>» в порядке обработки")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--asr-device", default="")
    ap.add_argument("--giga-device", default="")
    ap.add_argument("--int8", action="store_true", help="whisper int8_float16 (ноут, 4 ГБ)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--kk-ct2", default=str(ROOT / "voxcpm2/weights/kazakh-whisper-large-v3-turbo-ct2"))
    ap.add_argument("--base-ct2", default=str(ROOT / "voxcpm2/weights/whisper-large-v3-turbo-ct2"))
    ap.add_argument("--until-ts", type=float, default=0, help="новый ролик после этого времени (unix) не начинать")
    ap.add_argument("--transfer-flag", type=Path, default=None, help="файл-флаг «перекачка закончена»")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-sec", type=float, default=0, help="только первые N секунд ролика (проверка)")
    ap.add_argument("--worker", default="w0")
    ap.add_argument("--prefix", default="youtube_", help="имя файла ролика в --raw: <prefix><id>.flac")
    ap.add_argument("--beam", type=int, default=3, help="лучи второго мнения (kk-turbo / ru turbo)")
    ap.add_argument("--respan", action="store_true", help="сессия 2: ещё и переобработать куски «без пауз» уже готовых роликов")
    ap.add_argument("--demucs", action="store_true", help="включить Demucs для роликов с музыкой (сессия 1); по умолчанию выключен")
    a = ap.parse_args()
    cfg = Cfg()
    cfg.beam = a.beam
    cfg.use_demucs = a.demucs
    if a.device == "cpu":
        cfg.ast_win, cfg.ast_hop = 6.0, 6.0
    order = []
    for line in a.order.read_text().splitlines():
        if line.strip():
            vid, _, js = line.partition("\t")
            order.append((vid.strip(), json.loads(js) if js.strip() else {}))
    m = Models(a, cfg)
    print(f"[{a.worker}] модели за {m.load_s:.0f} с (DNSMOS: {m.dnsmos.provider}), в очереди {len(order)}", flush=True)
    n = 0
    for vid, meta in order:
        if a.until_ts and time.time() >= a.until_ts:
            print(f"[{a.worker}] стоп по времени", flush=True)
            break
        cid = f"{a.prefix}{vid}"
        mode = "clip"
        if done(a.out, cid):
            if not (a.respan and need_respan(a.out, cid) and claim(a.out, cid, ".claim_respan")):
                continue
            mode = "respan"
        elif not claim(a.out, cid):
            continue
        f = a.raw / f"{cid}.flac"
        while not f.exists():
            if a.transfer_flag is None or a.transfer_flag.exists():
                break
            if a.until_ts and time.time() >= a.until_ts:
                break
            time.sleep(15)
        if not f.exists():
            (a.out / "work" / cid / (".claim" if mode == "clip" else ".claim_respan")).unlink(missing_ok=True)
            continue
        t = time.time()
        try:
            if mode == "respan":
                r = process_respan(f, m, cfg, a.out)
                n += 1
                if r:
                    print(f"[{a.worker}] {cid} respan: кусков {len(r['spans'])} / {r['spans_sec']/60:.1f} мин → фраз {r['n_phrases']}, "
                          f"чисто {r['clean_sec']/60:.1f} мин, текст ок {r['asr_sec']/60:.1f} мин, {time.time() - t:.0f} с", flush=True)
                continue
            r = process_clip(f, m, cfg, a.out, meta, a.max_sec)
            n += 1
            print(f"[{a.worker}] {cid}: {r['duration']/60:.1f} мин, фраз {r['n_phrases']}, чисто {r['clean_sec']/60:.1f} мин, "
                  f"текст ок {r['asr_sec']/60:.1f} мин, {time.time() - t:.0f} с (RTF {r['rtf']:.3f}) {r['timings']}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{a.worker}] ОШИБКА {cid}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
            (a.out / "work" / cid / "error.txt").write_text(traceback.format_exc())
        if a.limit and n >= a.limit:
            break
    print(f"[{a.worker}] готово: {n} роликов", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()
