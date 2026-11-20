#!/usr/bin/env python3
"""tts_server.py — голос (VoxCPM2 + LoRA) по HTTP: голосовые сообщения для мессенджера и поток для звонка.

Модель: VoxCPM2 (TTS_WEIGHTS) + LoRA-снимок (TTS_CKPT, например $VOICE_HOME/runs/<run>/step_XXXXXXX), грузится один раз при старте,
прогрев одной фразой.
Вызовы VoxCPM — те же, что в voxcpm2/nv_synth.py (прогон «новый голос» 09.10): from_pretrained(lora_weights_path=…, load_denoiser=False,
optimize=False), generate(text, reference_wav_path, prompt_wav_path, prompt_text, cfg_value, inference_timesteps, seed).

  POST /tts     Authorization: Bearer <токен>; JSON {"text": "...", "voice": "aruzhan_f"|"male_m"|"aruzhan_f_ru",
                необязательно: "mode": "chain"|"ref"|"clone", "cfg": 1.0–3.0, "steps": 4–30, "seed": int, "format": "ogg"|"wav"}
                → 200 audio/ogg (OGG/Opus, моно, 48 кГц — голосовое WhatsApp) + заголовки X-Audio-Seconds, X-Synth-Seconds, X-RTF, X-Chunks
                ошибки — JSON {"error": код, "detail": …}: 400 / 401 / 413 / 503 (busy, not_ready; Retry-After) / 504 / 500
  GET  /health  без токена: готовность, голоса, настройки по умолчанию, очередь
  GET  /metrics только с 127.0.0.1: счётчики под именами vLLM — их читает внешний сторож простоя GPU (простой = нет запросов)

Режимы (текст режется на куски ≤ 180 знаков по предложениям, склейка с паузой TTS_PAUSE):
  chain (по умолчанию) — кусок 1: только эталон-тембр (reference_wav_path) — ровно как замерено 09.10 (d20_kk/d22_kk: CER 0/0,3 %);
                         кусок k > 1: эталон + продолжение предыдущего сгенерированного куска (prompt_wav + его текст);
  clone — кусок 1: эталон + продолжение самого эталона с его текстом (Ultimate Cloning из README VoxCPM2), дальше как chain;
  ref   — каждый кусок только по эталону-тембру, без продолжения.
Очередь: одна генерация за раз, ждущих ≤ TTS_QUEUE_MAX. В журнал — только длины и время, текст клиента не пишется.
TTS_MOCK=1 — без модели (синтетический сигнал), для проверки обвязки на ноуте.

Поток для звонка:
  POST /tts/stream  тот же JSON, что /tts, плюс "sr": 8000|16000|24000|48000 (по умолчанию TTS_STREAM_SR = 24000),
                    "encoding": "pcm16" (по умолчанию) | "mulaw" (только 8000, G.711 для телефонии).
                    → 200, Transfer-Encoding: chunked, тело — сырой звук моно (pcm16 = int16 little-endian) по мере синтеза
                    (кусок ≈ 160 мс: один патч VoxCPM2 = 4 латента × 40 мс). Заголовки уходят, когда готов ПЕРВЫЙ звук:
                    X-TTFA-Ms (от приёма запроса до первого звука, с очередью), X-TTFA-Model-Ms (до первого патча модели),
                    X-Queue-Ms, X-Sample-Rate, X-Encoding, X-Request-Id. Ошибки до первого звука — те же JSON-коды, что у /tts.
                    Модель: VoxCPM.generate_streaming (тот же _generate_with_prompt_cache, что и generate; LoRA — слои модели,
                    работают в обоих путях; retry_badcase в потоке библиотека выключает → свой сторож «не остановилась»).
  GET  /tts/stats?rid=…  (токен) итог потока: ttfa, длительность, синтез, RTF, нужный запас буфера у слушателя (prebuffer_ms), обрывы.
  TTS_CKPT=none — без LoRA (влитые веса: TTS_WEIGHTS=$VOICE_HOME/merged/<имя>, см. merge_lora.py).
  TTS_REF_CACHE=1 — латенты эталона считаются один раз на голос (тот же _encode_wav, что в build_prompt_cache: детерминирован).
"""
from __future__ import annotations

import collections
import hashlib
import hmac
import io
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import streamio  # noqa: E402
import textprep  # noqa: E402


def env(k, d):
    return os.environ.get(k, d)


VOICE_HOME = Path(env("VOICE_HOME", str(Path.home() / "voice")))   # веса, снимки LoRA, влитые модели


class C:  # настройки — из окружения (на сервере удобно через systemd EnvironmentFile=$TTS_HOME/env)
    host = env("TTS_HOST", "0.0.0.0")
    port = int(env("TTS_PORT", "8000"))
    home = Path(env("TTS_HOME", str(Path.home() / "tts")))
    token_file = Path(env("TTS_TOKEN_FILE", str(home / "token")))
    voices_json = Path(env("TTS_VOICES_JSON", str(HERE / "voices.json")))
    voices_dir = Path(env("TTS_VOICES_DIR", str(home / "voices")))
    weights = env("TTS_WEIGHTS", str(VOICE_HOME / "weights" / "VoxCPM2"))
    ckpt = env("TTS_CKPT", "none")                 # папка снимка LoRA (lora_weights.safetensors); none — без LoRA
    device = env("TTS_DEVICE", "cuda")
    optimize = env("TTS_OPTIMIZE", "0") == "1"
    mock = env("TTS_MOCK", "0") == "1"
    mode = env("TTS_MODE", "chain")
    cfg = float(env("TTS_CFG", "2.0"))
    steps = int(env("TTS_STEPS", "10"))
    seed = int(env("TTS_SEED", "1"))
    max_chars = int(env("TTS_MAX_CHARS", "1000"))
    chunk_chars = int(env("TTS_CHUNK_CHARS", str(textprep.MAX_CHARS)))
    queue_max = int(env("TTS_QUEUE_MAX", "4"))
    queue_wait = float(env("TTS_QUEUE_WAIT", "90"))
    deadline = float(env("TTS_DEADLINE", "150"))
    pause = float(env("TTS_PAUSE", "0.25"))
    bitrate = env("TTS_BITRATE", "32k")
    keepalive = env("TTS_KEEPALIVE", "/run/gpu-keepalive")
    log = Path(env("TTS_LOG", str(home / "logs" / "tts.log")))
    cache_n = int(env("TTS_CACHE", "200"))
    mock_rtf = float(env("TTS_MOCK_RTF", "0.02"))
    mock_load = float(env("TTS_MOCK_LOAD", "1.0"))
    debug_calls = env("TTS_DEBUG_CALLS", "")      # только mock: jsonl вызовов синтеза (без текста)
    # поток для звонка
    stream_sr = int(env("TTS_STREAM_SR", "24000"))
    ref_cache = env("TTS_REF_CACHE", "1") == "1"
    stream_gain = env("TTS_STREAM_GAIN", "auto")   # auto — по прогреву; число — постоянное усиление
    stream_timeout = float(env("TTS_STREAM_SEND_TIMEOUT", "15"))
    stats_n = int(env("TTS_STATS", "500"))
    mock_prefill = float(env("TTS_MOCK_PREFILL", "0.08"))    # mock: время до первого патча (префилл), с
    mock_stream_rtf = float(env("TTS_MOCK_STREAM_RTF", "0.5"))  # mock: скорость патчей в потоке
    mock_lead = float(env("TTS_MOCK_LEAD", "0.3"))           # mock: тишина в начале фразы, с (проверка SilenceGate)


def use_lora():
    return C.ckpt.strip().lower() not in ("", "none", "merged")


MODES = ("chain", "ref", "clone")
WARMUP_TEXT = "Сәлеметсіз бе! Тыңдап тұрмын, сізге қалай көмектесе аламын?"
OGG_SOFT_LIMIT = 500 * 1024          # WhatsApp: значок «play» у голосового только при файле ≤ 512 КБ (доки Meta, audio messages)

log = logging.getLogger("tts")


def setup_logging():
    C.log.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(C.log), logging.StreamHandler(sys.stderr)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def touch_keepalive():
    try:
        Path(C.keepalive).touch()
    except OSError:
        pass


# ---------------------------------------------------------------- голоса
class Voice:
    def __init__(self, vid, d):
        self.id, self.label, self.gender = vid, d.get("label", vid), d.get("gender", "")
        self.wav = str(C.voices_dir / d["wav"])
        self.text = (C.voices_dir / d["text"]).read_text().strip() if d.get("text") else None


def load_voices():
    j = json.loads(C.voices_json.read_text())
    vs = {k: Voice(k, v) for k, v in j["voices"].items()}
    miss = [v.wav for v in vs.values() if not Path(v.wav).is_file()]
    if miss:
        raise FileNotFoundError(f"нет эталонов: {miss}")
    return vs, j.get("default") or next(iter(vs))


# ---------------------------------------------------------------- звук
def trim(wav, sr, thr_db=-45.0, head=0.05, tail=0.08):
    """Срезать тишину по краям куска, оставив поля head/tail и затухание 15 мс (не «край в край»)."""
    fr = max(1, int(0.02 * sr))
    n = len(wav) // fr
    if n < 3:
        return wav
    e = np.sqrt(np.mean(wav[: n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    on = np.where(20 * np.log10(e / e.max()) > thr_db)[0]
    if not len(on):
        return wav
    a = max(0, on[0] * fr - int(head * sr))
    b = min(len(wav), (on[-1] + 1) * fr + int(tail * sr))
    out = wav[a:b].astype(np.float32).copy()
    f = min(len(out) // 4, int(0.015 * sr))
    if f > 1:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        out[:f] *= ramp
        out[-f:] *= ramp[::-1]
    return out


def join(pieces, sr):
    gap = np.zeros(int(C.pause * sr), np.float32)
    parts = [np.zeros(int(0.1 * sr), np.float32)]
    for k, p in enumerate(pieces):
        if k:
            parts.append(gap)
        parts.append(p)
    parts.append(np.zeros(int(0.2 * sr), np.float32))
    x = np.concatenate(parts)
    speech = x[np.abs(x) > 1e-4]
    rms = float(np.sqrt(np.mean(speech ** 2))) if len(speech) else 0.0
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if rms > 0 and peak > 0:
        x = x * min(10 ** (-20 / 20) / rms, 0.89 / peak)       # речь ≈ −20 dBFS, пик не выше −1 dBFS
    return x.astype(np.float32)


def resample(x, sr, to):
    if sr == to:
        return x
    n = int(round(len(x) * to / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def _ffmpeg_ogg(x, sr, bitrate):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           "-ar", "48000", "-ac", "1", "-c:a", "libopus", "-b:a", bitrate, "-vbr", "on", "-application", "audio",
           "-frame_duration", "20", "-f", "ogg", "pipe:1"]
    r = subprocess.run(cmd, input=x.astype("<f4").tobytes(), capture_output=True, timeout=60)
    if r.returncode or not r.stdout:
        raise RuntimeError("ffmpeg: " + r.stderr.decode("utf-8", "replace")[-200:])
    return r.stdout


def encoder_name():
    return "ffmpeg-libopus" if shutil.which("ffmpeg") else "soundfile-opus"


def encode(x, sr, fmt):
    """→ (bytes, content-type). OGG: контейнер Ogg, кодек Opus, моно, 48 кГц (Meta Cloud API: audio/ogg, OPUS only, mono)."""
    import soundfile as sf
    if fmt == "wav":
        buf = io.BytesIO()
        sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "audio/wav"
    if shutil.which("ffmpeg"):
        b = _ffmpeg_ogg(x, sr, C.bitrate)
        if len(b) > OGG_SOFT_LIMIT:
            b = _ffmpeg_ogg(x, sr, "16k")
        return b, "audio/ogg"
    buf = io.BytesIO()
    sf.write(buf, resample(x, sr, 48000), 48000, format="OGG", subtype="OPUS")
    return buf.getvalue(), "audio/ogg"


# ---------------------------------------------------------------- модель
class Engine:
    def __init__(self):
        self.model, self.sr = None, 48000
        self.active = self.max_active = 0
        self.ref_feats = {}          # голос → латенты эталона (TTS_REF_CACHE)

    def load(self):
        t = time.time()
        if C.mock:
            time.sleep(C.mock_load)
            log.info("модель: MOCK (без VoxCPM), sr %d", self.sr)
            return
        import torch  # noqa: F401
        from voxcpm import VoxCPM
        if use_lora():
            self.model = VoxCPM.from_pretrained(C.weights, load_denoiser=False, optimize=C.optimize, device=C.device,
                                                lora_weights_path=C.ckpt)
            self.model.set_lora_enabled(True)
        else:                       # влитые веса (merge_lora.py): обычная модель без LoRA-слоёв
            self.model = VoxCPM.from_pretrained(C.weights, load_denoiser=False, optimize=C.optimize, device=C.device)
        self.sr = self.model.tts_model.sample_rate
        log.info("модель VoxCPM2 %s за %.0f с, sr %d, optimize=%s", f"+ LoRA {C.ckpt}" if use_lora() else f"(влитые веса {C.weights})",
                 time.time() - t, self.sr, C.optimize)

    def synth(self, text, ref, prompt_wav, prompt_text, cfg, steps, seed, mock_delay=None, dbg=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if C.mock:
                return self._mock(text, seed, mock_delay, dbg)
            import torch
            torch.manual_seed(seed)
            np.random.seed(seed)
            wav = self.model.generate(text=text, reference_wav_path=ref, prompt_wav_path=prompt_wav, prompt_text=prompt_text,
                                      cfg_value=cfg, inference_timesteps=steps, seed=seed)
            return np.asarray(wav, dtype=np.float32).reshape(-1)
        finally:
            self.active -= 1

    def stream(self, text, ref, prompt_wav, prompt_text, cfg, steps, seed, mock_delay=None, dbg=None):
        """Генератор кусков float32 (частота self.sr) по мере синтеза. Закрывать через .close() — иначе VAE модели
        останется в потоковом режиме (StreamingVAEDecoder снимает свои заплатки в __exit__)."""
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if C.mock:
                yield from self._mock_stream(text, seed, mock_delay, dbg)
                return
            import torch
            torch.manual_seed(seed)
            np.random.seed(seed)
            if C.ref_cache:
                gen = self._stream_cached(text, ref, prompt_wav, prompt_text, cfg, steps, seed)
            else:
                gen = self.model.generate_streaming(text=text, reference_wav_path=ref, prompt_wav_path=prompt_wav,
                                                    prompt_text=prompt_text, cfg_value=cfg, inference_timesteps=steps, seed=seed)
            try:
                for w in gen:
                    yield np.asarray(w, dtype=np.float32).reshape(-1)
            finally:
                gen.close()
        finally:
            self.active -= 1

    def _stream_cached(self, text, ref, prompt_wav, prompt_text, cfg, steps, seed):
        """То же, что VoxCPM._generate(streaming=True) (voxcpm core.py, коммит f0c787f), но латенты эталона — из кэша.
        build_prompt_cache: ref → _encode_wav(padding_mode="right"), prompt → _encode_wav(padding_mode="left")."""
        import re
        tm = self.model.tts_model
        if ref not in self.ref_feats:
            self.ref_feats[ref] = tm._encode_wav(ref, padding_mode="right")
        cache = {"ref_audio_feat": self.ref_feats[ref], "mode": "reference"}
        if prompt_wav is not None:
            cache.update(prompt_text=prompt_text, audio_feat=tm._encode_wav(prompt_wav, padding_mode="left"), mode="ref_continuation")
        text = re.sub(r"\s+", " ", text.replace("\n", " "))
        g = tm._generate_with_prompt_cache(target_text=text, prompt_cache=cache, min_len=2, max_len=4096, inference_timesteps=steps,
                                           cfg_value=cfg, retry_badcase=False, streaming=True, seed=seed)
        try:
            for wav, _, _ in g:
                yield wav.squeeze(0).cpu().numpy()
        finally:
            g.close()

    def _mock_stream(self, text, seed, mock_delay, dbg):
        letters = sum(ch.isalpha() for ch in text)
        dur = 0.3 + 0.075 * letters
        sr, patch = self.sr, 0.16
        if C.debug_calls and dbg is not None:
            with open(C.debug_calls, "a") as f:
                f.write(json.dumps(dict(dbg, stream=True)) + "\n")
        time.sleep(C.mock_prefill)
        f0 = 200 + 20 * (seed % 5)
        lead, tail = int(C.mock_lead * sr), int(0.3 * sr)
        total = lead + int(dur * sr) + tail
        step = int(patch * sr)
        rtf = C.mock_stream_rtf if mock_delay is None else mock_delay / max(dur, 1e-3)
        for a in range(0, total, step):
            n = np.arange(a, min(total, a + step))
            t = n / sr
            x = sum(0.2 / k * np.sin(2 * np.pi * f0 * k * t) for k in (1, 2, 3)) * (0.55 + 0.45 * np.sin(2 * np.pi * 4 * t))
            x = np.where((n >= lead) & (n < total - tail), x, 0.0)
            time.sleep(len(n) / sr * rtf)
            yield x.astype(np.float32)

    def _mock(self, text, seed, mock_delay, dbg):
        letters = sum(ch.isalpha() for ch in text)
        dur = 0.3 + 0.075 * letters
        time.sleep(mock_delay if mock_delay is not None else dur * C.mock_rtf)
        sr = self.sr
        t = np.arange(int(dur * sr)) / sr
        f0 = 200 + 20 * (seed % 5)
        x = sum(0.2 / k * np.sin(2 * np.pi * f0 * k * t) for k in (1, 2, 3)) * (0.55 + 0.45 * np.sin(2 * np.pi * 4 * t))
        sil = np.zeros(int(0.4 * sr))
        if C.debug_calls and dbg is not None:
            with open(C.debug_calls, "a") as f:
                f.write(json.dumps(dbg) + "\n")
        return np.concatenate([sil, x, sil]).astype(np.float32)


NORM_SPL = 0.09     # с на букву: эталон aruzhan_f_kk — 12,5 с на 133 буквы ≈ 0,094


def plausible(wav, sr, text):
    """Грубая проверка куска по длительности на букву → (годен, с/букву). Ловит обрыв и «не остановилась»."""
    letters = max(1, sum(ch.isalpha() for ch in text))
    dur = len(wav) / sr
    spl = dur / letters
    if dur < 0.3:
        return False, spl
    return (0.025 <= spl <= 0.25) if letters >= 8 else True, spl


# ---------------------------------------------------------------- приложение
class Busy(Exception):
    pass


class Deadline(Exception):
    pass


class App:
    def __init__(self):
        self.engine = Engine()
        self.voices, self.default_voice = load_voices()
        self.token = C.token_file.read_text().strip()
        if len(self.token) < 20:
            raise SystemExit("токен короче 20 знаков")
        self.ready, self.error, self.t0 = False, None, time.time()
        self.gate = threading.Lock()
        self.m = threading.Lock()
        self.waiting = self.running = 0
        self.n_req = self.n_ok = self.n_err = 0
        self.audio_s = self.synth_s = 0.0
        self.cache = collections.OrderedDict()
        self.stats = collections.OrderedDict()      # rid → итог потока (GET /tts/stats)
        self.gain = 1.0                              # постоянное усиление потока (по прогреву)
        (C.home / "tmp").mkdir(parents=True, exist_ok=True)

    # очередь: одна генерация за раз
    def enter(self):
        with self.m:
            if self.waiting >= C.queue_max:
                raise Busy()
            self.waiting += 1
        ok = self.gate.acquire(timeout=C.queue_wait)
        with self.m:
            self.waiting -= 1
            if ok:
                self.running = 1
        if not ok:
            raise Busy()

    def leave(self):
        with self.m:
            self.running = 0
        self.gate.release()

    def start(self):
        try:
            self.engine.load()
            t = time.time()
            self.enter()
            try:
                x, info = self.synthesize([dict(text=WARMUP_TEXT, lang="kk")], self.voices[self.default_voice], C.mode,
                                          C.cfg, C.steps, C.seed, time.time())
            finally:
                self.leave()
            log.info("прогрев: %.1f с звука за %.1f с (RTF %.2f), кодировщик %s", info["audio_s"], time.time() - t,
                     (time.time() - t) / max(info["audio_s"], 1e-6), encoder_name())
            try:
                self.warm_stream()
            except Exception as e:  # noqa: BLE001 — /tts (голосовые WhatsApp) работает и без потока
                log.error("прогрев потока не удался: %s: %s", type(e).__name__, str(e)[:300])
            self.ready = True
            touch_keepalive()
            log.info("ГОТОВ: http://%s:%d, голоса %s, по умолчанию %s · %s · cfg %.1f · %d шагов",
                     C.host, C.port, ",".join(self.voices), self.default_voice, C.mode, C.cfg, C.steps)
        except Exception as e:  # noqa: BLE001 — сервис живёт и отвечает /health с ошибкой
            self.error = f"{type(e).__name__}: {str(e)[:300]}"
            log.error("СТАРТ НЕ УДАЛСЯ: %s", self.error)

    def warm_stream(self):
        """Прогрев потокового пути (VAE в потоковом режиме, кэш эталона) и постоянное усиление потока."""
        t = time.time()
        self.enter()
        try:
            v = self.voices[self.default_voice]
            g = self.engine.stream(WARMUP_TEXT, v.wav, None, None, C.cfg, C.steps, C.seed)
            try:
                raw = np.concatenate(list(g) or [np.zeros(1, np.float32)])
            finally:
                g.close()
        finally:
            self.leave()
        auto = streamio.speech_gain(raw, self.engine.sr)
        try:
            self.gain = auto if C.stream_gain == "auto" else float(C.stream_gain)
        except ValueError:
            self.gain = auto
        log.info("прогрев потока: %.1f с звука за %.1f с, усиление потока %.2f (по прогреву %.2f), ресемплер %s",
                 len(raw) / self.engine.sr, time.time() - t, self.gain, auto, streamio.Resampler(self.engine.sr, C.stream_sr).kind)

    def stream(self, chunks, voice, mode, cfg, steps, seed, t_start, mock_delay=None, st=None):
        """Поток: куски float32 (частота модели) по мере синтеза; режимы chain/clone/ref — как synthesize().
        st — словарь итогов (заполняется по ходу). Тишина в начале срезается, между кусками текста — пауза TTS_PAUSE."""
        eng, sr = self.engine, self.engine.sr
        import soundfile as sf
        st = st if st is not None else {}
        st.update(patches=0, runaway=0, lead_cut_ms=0.0, model_first=None, chunks=len(chunks))
        tmp = Path(tempfile.mkdtemp(dir=C.home / "tmp"))
        prev = None
        try:
            for k, ch in enumerate(chunks):
                if time.time() - t_start > C.deadline:
                    raise Deadline()
                pw = pt = None
                if mode in ("chain", "clone") and k > 0 and prev:
                    pw, pt = prev
                elif mode == "clone" and k == 0 and voice.text:
                    pw, pt = voice.wav, voice.text
                dbg = dict(chunk=k, mode=mode, ref=voice.id, prompt=("voice" if pw == voice.wav else "prev" if pw else None),
                           cfg=cfg, steps=steps, seed=seed + k, chars=len(ch["text"]))
                letters = max(1, sum(c.isalpha() for c in ch["text"]))
                cap = max(2.5, 0.25 * letters) + 0.5          # дольше — «не остановилась»: глушим кусок
                gate = streamio.SilenceGate(sr)
                whole, made = [], 0
                if k:
                    yield np.zeros(int(C.pause * sr), np.float32)
                g = eng.stream(ch["text"], voice.wav, pw, pt, cfg, steps, seed + k, mock_delay, dbg)
                try:
                    for w in g:
                        if st["model_first"] is None:
                            st["model_first"] = time.time()
                        st["patches"] += 1
                        made += len(w)
                        whole.append(w)
                        out = gate.push(w)
                        if len(out):
                            yield streamio.limiter(out * self.gain)
                        if made / sr > cap:
                            st["runaway"] += 1
                            log.info("поток: кусок %d длиннее %.1f с на %d букв — глушу (не остановилась)", k, cap, letters)
                            break
                        if time.time() - t_start > C.deadline:
                            raise Deadline()
                finally:
                    g.close()
                tail = gate.finish()
                if len(tail):
                    yield streamio.limiter(tail * self.gain)
                if k == 0:
                    st["lead_cut_ms"] = round(gate.dropped / sr * 1000)
                prev = None
                if mode in ("chain", "clone") and k + 1 < len(chunks) and whole:
                    wav = trim(np.concatenate(whole), sr)
                    ok, _ = plausible(wav, sr, ch["text"])
                    if ok:
                        p = tmp / f"prev_{k}.wav"
                        sf.write(p, wav, sr)
                        prev = (str(p), ch["text"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def synthesize(self, chunks, voice, mode, cfg, steps, seed, t_start, mock_delay=None):
        eng, sr = self.engine, self.engine.sr
        import soundfile as sf
        tmp = Path(tempfile.mkdtemp(dir=C.home / "tmp"))
        pieces, prev, retries = [], None, 0
        try:
            for k, ch in enumerate(chunks):
                if time.time() - t_start > C.deadline:
                    raise Deadline()
                pw = pt = None
                if mode in ("chain", "clone") and k > 0 and prev:
                    pw, pt = prev
                elif mode == "clone" and k == 0 and voice.text:
                    pw, pt = voice.wav, voice.text
                dbg = dict(chunk=k, mode=mode, ref=voice.id, prompt=("voice" if pw == voice.wav else "prev" if pw else None),
                           cfg=cfg, steps=steps, seed=seed + k, chars=len(ch["text"]))
                wav = trim(eng.synth(ch["text"], voice.wav, pw, pt, cfg, steps, seed + k, mock_delay, dbg), sr)
                ok, spl = plausible(wav, sr, ch["text"])
                if not ok:              # одна повторная попытка: только эталон, другой сид; берём ту, что ближе к норме
                    retries += 1
                    dbg = dict(dbg, prompt=None, seed=seed + 1000 + k, retry=True)
                    w2 = trim(eng.synth(ch["text"], voice.wav, None, None, cfg, steps, seed + 1000 + k, mock_delay, dbg), sr)
                    ok2, spl2 = plausible(w2, sr, ch["text"])
                    if ok2 or abs(spl2 - NORM_SPL) < abs(spl - NORM_SPL):
                        wav, ok = w2, ok2
                pieces.append(wav)
                prev = None
                if ok and mode in ("chain", "clone"):
                    p = tmp / f"prev_{k}.wav"
                    sf.write(p, wav, sr)
                    prev = (str(p), ch["text"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        x = join(pieces, sr)
        return x, dict(audio_s=len(x) / sr, chunks=len(chunks), retries=retries)

    def health(self):
        h = dict(ok=True, ready=self.ready, error=self.error, mock=C.mock,
                 model=(f"VoxCPM2 + LoRA {Path(C.ckpt).parent.name}/{Path(C.ckpt).name}" if use_lora()
                        else f"VoxCPM2, влитые веса {Path(C.weights).name}"), sample_rate=self.engine.sr,
                 stream=dict(sr_default=C.stream_sr, rates=list(streamio.RATES), encodings=list(streamio.ENCODINGS),
                             ref_cache=C.ref_cache, gain=round(self.gain, 3), optimize=C.optimize),
                 encoder=encoder_name(), voices={k: v.label for k, v in self.voices.items()}, default_voice=self.default_voice,
                 defaults=dict(mode=C.mode, cfg=C.cfg, steps=C.steps, seed=C.seed), modes=list(MODES),
                 limits=dict(max_chars=C.max_chars, queue_max=C.queue_max, deadline_s=C.deadline),
                 queue=self.waiting, busy=self.running, served=self.n_req, uptime_s=int(time.time() - self.t0))
        if C.mock:
            h["mock_max_parallel"] = self.engine.max_active
        return h

    def metrics(self):
        return ("# tts: имена как у vLLM — их читает внешний сторож простоя GPU (простой = счётчик не растёт)\n"
                f"vllm:request_success_total {self.n_req}\n"
                f"vllm:num_requests_running {self.running}\n"
                f"vllm:num_requests_waiting {self.waiting}\n"
                f"tts_ok_total {self.n_ok}\ntts_error_total {self.n_err}\n"
                f"tts_audio_seconds_total {self.audio_s:.1f}\ntts_synth_seconds_total {self.synth_s:.1f}\n")


APP: App | None = None


class Bad(Exception):
    def __init__(self, status, code, detail=""):
        super().__init__(code)
        self.status, self.code, self.detail = status, code, detail


def parse(body: dict):
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise Bad(400, "empty_text", "поле text — непустая строка")
    if len(text) > C.max_chars:
        raise Bad(413, "text_too_long", f"не больше {C.max_chars} знаков, пришло {len(text)}")
    vid = body.get("voice") or APP.default_voice
    if vid not in APP.voices:
        raise Bad(400, "bad_voice", f"голоса: {', '.join(APP.voices)}")
    mode = body.get("mode") or C.mode
    if mode not in MODES:
        raise Bad(400, "bad_mode", f"режимы: {', '.join(MODES)}")
    try:
        cfg = float(body.get("cfg", C.cfg))
        steps = int(body.get("steps", C.steps))
        seed = int(body.get("seed", C.seed))
    except (TypeError, ValueError):
        raise Bad(400, "bad_param", "cfg — число, steps и seed — целые") from None
    if not (1.0 <= cfg <= 3.0) or not (4 <= steps <= 30) or not (0 <= seed < 2 ** 31):
        raise Bad(400, "bad_param", "cfg 1,0–3,0; steps 4–30; seed ≥ 0")
    fmt = body.get("format", "ogg")
    if fmt not in ("ogg", "wav"):
        raise Bad(400, "bad_format", "format: ogg | wav")
    chunks = textprep.prepare(text, C.chunk_chars)
    if not chunks:
        raise Bad(400, "empty_text", "после нормализации нечего произносить (ссылки, эмодзи, символы)")
    mock_delay = body.get("_mock_delay") if C.mock else None
    return dict(voice=APP.voices[vid], mode=mode, cfg=cfg, steps=steps, seed=seed, fmt=fmt, chunks=chunks, chars=len(text),
                mock_delay=mock_delay)


def parse_stream(body: dict):
    """Доп. поля потока: sr, encoding → (sr, encoding)."""
    try:
        sr = int(body.get("sr", C.stream_sr))
    except (TypeError, ValueError):
        raise Bad(400, "bad_param", "sr — целое") from None
    enc = body.get("encoding", "pcm16")
    if sr not in streamio.RATES:
        raise Bad(400, "bad_param", f"sr: {', '.join(map(str, streamio.RATES))}")
    if enc not in streamio.ENCODINGS:
        raise Bad(400, "bad_format", "encoding: pcm16 | mulaw")
    if enc == "mulaw" and sr != 8000:
        raise Bad(400, "bad_format", "mulaw — только sr 8000 (G.711)")
    return sr, enc


class Handler(BaseHTTPRequestHandler):
    server_version = "voice-tts/1.0"
    sys_version = ""
    timeout = 30                     # чтение запроса; синтез идёт отдельно от этого

    def log_message(self, *a):       # стандартный журнал http.server не пишем: только свой, без текста
        pass

    def _send(self, status, body: bytes, ctype, extra=None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status, obj, extra=None):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", extra)

    def _auth(self):
        h = self.headers.get("Authorization", "")
        got = h[7:].strip() if h[:7].lower() == "bearer " else ""
        return bool(got) and hmac.compare_digest(got.encode(), APP.token.encode())

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            h = APP.health()
            return self._json(200 if APP.ready else 503, h)
        if path == "/tts/stats":
            if not self._auth():
                return self._json(401, dict(error="unauthorized"), {"WWW-Authenticate": "Bearer"})
            from urllib.parse import parse_qs, urlsplit
            rid = (parse_qs(urlsplit(self.path).query).get("rid") or [""])[0]
            with APP.m:
                x = APP.stats.get(rid)
            return self._json(200, x) if x else self._json(404, dict(error="not_found", detail="нет такого rid (или ещё идёт)"))
        if path == "/metrics" and self.client_address[0] in ("127.0.0.1", "::1"):
            return self._send(200, APP.metrics().encode(), "text/plain; version=0.0.4")
        return self._json(404, dict(error="not_found"))

    def do_POST(self):
        t0 = time.time()
        rid = (self.headers.get("X-Request-Id") or uuid.uuid4().hex[:10])[:40]
        path = self.path.split("?", 1)[0]
        if path not in ("/tts", "/tts/stream"):
            return self._json(404, dict(error="not_found"))
        if not self._auth():
            log.info("tts 401 rid=%s from=%s", rid, self.client_address[0])
            return self._json(401, dict(error="unauthorized"), {"WWW-Authenticate": "Bearer"})
        with APP.m:
            APP.n_req += 1                      # активность для сторожей — только запросы с верным токеном
        touch_keepalive()
        if path == "/tts/stream":
            return self._stream(t0, rid)
        status, code, info, p = 200, "ok", {}, None
        try:
            if not APP.ready:
                raise Bad(503, "not_ready", APP.error or "модель грузится")
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 64 * 1024:
                raise Bad(413 if n > 0 else 400, "bad_body", "JSON до 64 КБ с Content-Length")
            try:
                body = json.loads(self.rfile.read(n))
                assert isinstance(body, dict)
            except (ValueError, AssertionError):
                raise Bad(400, "bad_json", "тело — JSON-объект") from None
            p = parse(body)
            key = hashlib.sha256(json.dumps([p["voice"].id, p["mode"], p["cfg"], p["steps"], p["seed"], p["fmt"],
                                             [c["text"] for c in p["chunks"]]]).encode()).hexdigest()
            with APP.m:
                hit = APP.cache.get(key)
                if hit:
                    APP.cache.move_to_end(key)
            if hit:
                audio, ctype, info = hit
                info = dict(info, cache="hit", wait_s=0.0, synth_s=0.0)
            else:
                try:
                    APP.enter()
                except Busy:
                    raise Bad(503, "busy", "очередь полна, повторите позже") from None
                t1 = time.time()
                try:
                    x, info = APP.synthesize(p["chunks"], p["voice"], p["mode"], p["cfg"], p["steps"], p["seed"], t0,
                                             p["mock_delay"])
                finally:
                    APP.leave()
                t2 = time.time()
                audio, ctype = encode(x, APP.engine.sr, p["fmt"])
                info.update(cache="miss", wait_s=t1 - t0, synth_s=t2 - t1, bytes=len(audio))
                with APP.m:
                    APP.cache[key] = (audio, ctype, info)
                    while len(APP.cache) > C.cache_n:
                        APP.cache.popitem(last=False)
                    APP.audio_s += info["audio_s"]
                    APP.synth_s += info["synth_s"]
            rtf = info["synth_s"] / max(info["audio_s"], 1e-6)
            hdr = {"X-Request-Id": rid, "X-Audio-Seconds": f"{info['audio_s']:.2f}", "X-Synth-Seconds": f"{info['synth_s']:.2f}",
                   "X-RTF": f"{rtf:.2f}", "X-Chunks": info["chunks"], "X-Cache": info["cache"], "X-Voice": p["voice"].id,
                   "X-Mode": p["mode"], "Cache-Control": "no-store"}
            self._send(200, audio, ctype, hdr)
            with APP.m:
                APP.n_ok += 1
        except Bad as e:
            status, code = e.status, e.code
            extra = {"Retry-After": 20 if e.code == "busy" else 30} if e.status == 503 else None
            self._json(e.status, dict(error=e.code, detail=e.detail), extra)
        except Deadline:
            status, code = 504, "timeout"
            self._json(504, dict(error="timeout", detail=f"синтез дольше {C.deadline:.0f} с — сократите текст"))
        except Exception as e:  # noqa: BLE001
            status, code = 500, "synth_failed"
            log.error("tts 500 rid=%s %s: %s", rid, type(e).__name__, str(e)[:200])
            self._json(500, dict(error="synth_failed", detail=type(e).__name__))
        finally:
            if status != 200:
                with APP.m:
                    APP.n_err += 1
            touch_keepalive()
            # журнал: только длины, время и настройки — текст клиента не пишем
            log.info("tts %d %s rid=%s voice=%s mode=%s cfg=%s steps=%s chars=%s chunks=%s retries=%s audio=%.1fs synth=%.1fs "
                     "wait=%.1fs rtf=%.2f bytes=%s cache=%s total=%.1fs",
                     status, code, rid, p and p["voice"].id, p and p["mode"], p and p["cfg"], p and p["steps"],
                     p and p["chars"], info.get("chunks") or (p and len(p["chunks"])), info.get("retries"), info.get("audio_s", 0), info.get("synth_s", 0),
                     info.get("wait_s", 0), info.get("synth_s", 0) / max(info.get("audio_s", 0), 1e-6), info.get("bytes"),
                     info.get("cache"), time.time() - t0)


    # ------------------------------------------------------------ поток
    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 64 * 1024:
            raise Bad(413 if n > 0 else 400, "bad_body", "JSON до 64 КБ с Content-Length")
        try:
            body = json.loads(self.rfile.read(n))
            assert isinstance(body, dict)
        except (ValueError, AssertionError):
            raise Bad(400, "bad_json", "тело — JSON-объект") from None
        return body

    def _write(self, data: bytes):
        if data:
            self.wfile.write(b"%X\r\n%s\r\n" % (len(data), data) if self._chunked else data)

    def _stream(self, t0, rid):
        status, code, p, st = 200, "ok", None, {}
        gen, entered, sent, t_q, t_first = None, False, False, None, None
        sr_out, enc, n_out, nbytes, emits = C.stream_sr, "pcm16", 0, 0, []
        self._chunked = False
        try:
            if not APP.ready:
                raise Bad(503, "not_ready", APP.error or "модель грузится")
            body = self._read_json()
            p = parse(body)
            sr_out, enc = parse_stream(body)
            try:
                APP.enter()
            except Busy:
                raise Bad(503, "busy", "очередь полна, повторите позже") from None
            entered, t_q = True, time.time()
            gen = APP.stream(p["chunks"], p["voice"], p["mode"], p["cfg"], p["steps"], p["seed"], t0, p["mock_delay"], st)
            rs = streamio.Resampler(APP.engine.sr, sr_out)
            try:
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            self.connection.settimeout(C.stream_timeout)
            for w in gen:
                y = rs.push(w)
                now = time.time()
                if not sent:
                    if not len(y):
                        continue
                    t_first = now
                    self._chunked = self.request_version == "HTTP/1.1"
                    if self._chunked:
                        self.protocol_version = "HTTP/1.1"
                    self.send_response(200)
                    ttfa_model = (st["model_first"] - t0) * 1000 if st.get("model_first") else -1
                    for k, v in {"Content-Type": "application/octet-stream", "X-Request-Id": rid, "X-Sample-Rate": sr_out,
                                 "X-Encoding": enc, "X-Channels": 1, "X-TTFA-Ms": f"{(t_first - t0) * 1000:.0f}",
                                 "X-TTFA-Model-Ms": f"{ttfa_model:.0f}", "X-Queue-Ms": f"{(t_q - t0) * 1000:.0f}",
                                 "X-Chunks": len(p["chunks"]), "X-Voice": p["voice"].id, "X-Mode": p["mode"], "X-Steps": p["steps"],
                                 "X-CFG": p["cfg"], "Cache-Control": "no-store", "Connection": "close"}.items():
                        self.send_header(k, str(v))
                    if self._chunked:
                        self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    sent = True
                emits.append((now - t_first, len(w) / APP.engine.sr))
                if len(y):
                    b = streamio.encode(y, enc)
                    self._write(b)
                    n_out += len(y)
                    nbytes += len(b)
            if not sent:
                raise Bad(500, "empty_audio", "модель не дала звука")
            y = rs.push(np.zeros(0, np.float32), last=True)
            if len(y):
                b = streamio.encode(y, enc)
                self._write(b)
                n_out += len(y)
                nbytes += len(b)
            if self._chunked:
                self.wfile.write(b"0\r\n\r\n")
            with APP.m:
                APP.n_ok += 1
        except Bad as e:
            status, code = e.status, e.code
            if not sent:
                extra = {"Retry-After": 20 if e.code == "busy" else 30} if e.status == 503 else None
                self._json(e.status, dict(error=e.code, detail=e.detail), extra)
        except Deadline:
            status, code = 504, "timeout"
            if not sent:
                self._json(504, dict(error="timeout", detail=f"синтез дольше {C.deadline:.0f} с — сократите текст"))
        except (BrokenPipeError, ConnectionResetError, socket.timeout, TimeoutError):
            status, code = 499, "client_gone"
        except Exception as e:  # noqa: BLE001
            status, code = 500, "synth_failed"
            log.error("stream 500 rid=%s %s: %s", rid, type(e).__name__, str(e)[:200])
            if not sent:
                self._json(500, dict(error="synth_failed", detail=type(e).__name__))
        finally:
            if gen is not None:
                gen.close()                 # закрыть генераторы модели: VAE выходит из потокового режима
            if entered:
                APP.leave()
            self.close_connection = True
            t_end = time.time()
            audio_s = n_out / sr_out
            synth_s = (t_end - t_q) if t_q else 0.0
            x = dict(rid=rid, status=status, code=code, sr=sr_out, encoding=enc, chunks=st.get("chunks"),
                     patches=st.get("patches", 0), runaway=st.get("runaway", 0), lead_cut_ms=st.get("lead_cut_ms", 0),
                     queue_ms=round((t_q - t0) * 1000) if t_q else None,
                     ttfa_ms=round((t_first - t0) * 1000) if t_first else None,
                     ttfa_model_ms=round((st["model_first"] - t0) * 1000) if st.get("model_first") else None,
                     audio_s=round(audio_s, 3), synth_s=round(synth_s, 3), rtf=round(synth_s / audio_s, 3) if audio_s else None,
                     prebuffer_ms=round(streamio.need_prebuffer(emits) * 1000), bytes=nbytes,
                     voice=p and p["voice"].id, mode=p and p["mode"], cfg=p and p["cfg"], steps=p and p["steps"],
                     chars=p and p["chars"], total_s=round(t_end - t0, 3))
            with APP.m:
                if status != 200:
                    APP.n_err += 1
                else:
                    APP.audio_s += audio_s
                    APP.synth_s += synth_s
                APP.stats[rid] = x
                while len(APP.stats) > C.stats_n:
                    APP.stats.popitem(last=False)
            touch_keepalive()
            log.info("stream %d %s rid=%s voice=%s mode=%s cfg=%s steps=%s chars=%s chunks=%s patches=%s ttfa=%sms ttfa_model=%sms "
                     "queue=%sms audio=%.1fs synth=%.1fs rtf=%s prebuf=%sms runaway=%s lead_cut=%sms sr=%s enc=%s total=%.1fs",
                     status, code, rid, x["voice"], x["mode"], x["cfg"], x["steps"], x["chars"], x["chunks"], x["patches"],
                     x["ttfa_ms"], x["ttfa_model_ms"], x["queue_ms"], audio_s, synth_s, x["rtf"], x["prebuffer_ms"], x["runaway"],
                     x["lead_cut_ms"], sr_out, enc, x["total_s"])


def main():
    global APP
    setup_logging()
    APP = App()
    srv = ThreadingHTTPServer((C.host, C.port), Handler)
    srv.daemon_threads = True
    log.info("слушаю %s:%d (mock=%s), гружу модель…", C.host, C.port, C.mock)
    touch_keepalive()
    threading.Thread(target=APP.start, daemon=True).start()
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
