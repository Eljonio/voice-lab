#!/usr/bin/env python3
"""local_tts.py — голос (VoxCPM2 + LoRA v5 шаг 3479) локально на ноутбуке, без сервера.

Режимы модели:
  gpu — int8-веса (serve/lowvram.py, папка voxcpm2/weights/v5-3479-q8) на RTX 3050 4 ГБ;
  cpu — запасной: база bf16 + LoRA на процессоре (как synth_baseline.py; RTF ≈ 28 — фраза идёт минуты);
  auto — gpu, если карта видна и свободно ≥ 3,6 ГиБ, иначе cpu.
Текст — как на сервере (tts_server.py): textprep.prepare (числа, бренды, prosody_kk) → куски ≤ 180 знаков → режим chain
(кусок 1 по эталону-тембру d20_kk, следующие — с продолжением предыдущего) → обрезка краёв, проверка длины, склейка, громкость.

  local_tts.py say "текст" [--voice aruzhan_f] [--seed 1] [--out файл.wav] [--no-play]
        если тёплый процесс уже работает — через него (секунды), иначе запускает его в фоне и ждёт загрузки (~1 мин)
  local_tts.py serve [--mode auto|gpu|cpu] [--idle 15]   тёплый процесс (unix-сокет ~/.cache/voice-lab/local.sock), сам гаснет
                                                          после --idle минут без запросов
  local_tts.py stop | status
  local_tts.py eval --phrases eval/phrases_v2.json --ref eval/ref_kk.wav --seed 1 --out DIR [--ids a,b] [--mode gpu]
        ровно как voxcpm2/synth_eval.py на сервере (сырой текст без textprep, cfg 2,0, 10 шагов, тот же сид) — для A/B с out-1011b
  local_tts.py bench [--mode gpu] [--n 6]   поток: первый звук, RTF, пик VRAM/ОЗУ по фразам бота
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = Path(os.environ.get("LOCAL_TTS_CACHE", Path.home() / ".cache" / "voice-lab"))
SOCK = CACHE / "local.sock"
Q8 = Path(os.environ.get("LOCAL_Q8", ROOT / "voxcpm2/weights/v5-3479-q8"))
BASE = Path(os.environ.get("LOCAL_BASE", ROOT / "voxcpm2/weights/VoxCPM2"))
LORA = Path(os.environ.get("LOCAL_LORA", ROOT / "voxcpm2/runs/lora-v5/step_0003479"))
OUTD = Path(os.environ.get("LOCAL_OUT", Path(os.environ.get("VOICE_DS", Path.home() / "voice-dataset")) / "local-tts"))
NEED_VRAM = 3.6 * 2 ** 30

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # 3,68 ГиБ карты — без дробления кусков
os.environ.setdefault("TTS_HOME", str(CACHE))
os.environ.setdefault("TTS_VOICES_DIR", str(ROOT / "voices"))
os.environ.setdefault("TTS_DEADLINE", "3600")          # на CPU фраза идёт минуты — серверный срок 150 с тут не годится


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rss_peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def pick_mode(mode: str) -> str:
    if mode in ("gpu", "cpu"):
        return mode
    try:
        import torch
        if torch.cuda.is_available() and (Q8 / "model_q8.safetensors").exists():
            free, _ = torch.cuda.mem_get_info()
            if free >= NEED_VRAM:
                return "gpu"
            log(f"на карте свободно {free / 2 ** 30:.2f} ГиБ (< 3,6) — режим cpu")
    except Exception as e:  # noqa: BLE001
        log("CUDA недоступна:", e)
    return "cpu"


def load_model(mode: str):
    """→ (объект VoxCPM, режим)."""
    import torch
    mode = pick_mode(mode)
    t = time.time()
    if mode == "gpu":
        sys.path.insert(0, str(HERE))
        from lowvram import load_q8
        tts = load_q8(Q8, device="cuda")
    else:
        from voxcpm import VoxCPM
        torch.set_num_threads(int(os.environ.get("LOCAL_THREADS", "4")))
        torch.set_default_dtype(torch.bfloat16)       # сборка сразу в bf16 (иначе fp32 8 ГБ и OOM, замер 03.10)
        tts = VoxCPM.from_pretrained(str(BASE), load_denoiser=False, optimize=False, device="cpu", lora_weights_path=str(LORA))
        torch.set_default_dtype(torch.float32)
        tts.set_lora_enabled(True)
    log(f"модель готова: режим {mode}, {time.time() - t:.0f} с, пик ОЗУ процесса {rss_peak_gb():.1f} ГиБ")
    return tts, mode


class Synth:
    """Тот же путь, что tts_server.App.synthesize (импорт оттуда же — одна логика на сервере и на ноуте)."""

    def __init__(self, tts, mode):
        sys.path.insert(0, str(HERE))
        import tts_server as T
        self.T = T
        (Path(os.environ["TTS_HOME"]) / "tmp").mkdir(parents=True, exist_ok=True)
        self.eng = T.Engine()
        self.eng.model, self.eng.sr = tts, tts.tts_model.sample_rate
        self.engine = self.eng
        self.voices, self.default_voice = T.load_voices()
        self.mode = mode

    def say(self, text, voice=None, seed=None, cfg=None, steps=None, out=None):
        import numpy as np  # noqa: F401
        import soundfile as sf
        import textprep
        T = self.T
        chunks = textprep.prepare(text, T.C.chunk_chars)
        if not chunks:
            raise ValueError("после нормализации нечего произносить")
        v = self.voices[voice or self.default_voice]
        t = time.time()
        x, info = T.App.synthesize(self, chunks, v, T.C.mode, cfg or T.C.cfg, steps or T.C.steps,
                                   T.C.seed if seed is None else seed, t)
        dt = time.time() - t
        out = Path(out) if out else OUTD / f"{time.strftime('%m%d-%H%M%S')}-{v.id}-{self.mode}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, x, self.eng.sr)
        return dict(out=str(out), audio_s=round(info["audio_s"], 2), synth_s=round(dt, 1), rtf=round(dt / max(info["audio_s"], 1e-6), 2),
                    chunks=info["chunks"], retries=info["retries"], mode=self.mode,
                    text=" | ".join(c["text"] for c in chunks))


# ---------------------------------------------------------------- тёплый процесс
def cmd_serve(a):
    CACHE.mkdir(parents=True, exist_ok=True)
    if SOCK.exists():
        try:
            _ask({"cmd": "ping"}, timeout=3)
            log("уже работает:", SOCK)
            return 0
        except OSError:
            SOCK.unlink()
    tts, mode = load_model(a.mode)
    s = Synth(tts, mode)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCK))
    os.chmod(SOCK, 0o600)
    srv.listen(4)
    srv.settimeout(30)
    last = time.time()
    log(f"ГОТОВ: {SOCK}, режим {mode}, гасну после {a.idle:g} мин простоя")
    try:
        while time.time() - last < a.idle * 60:
            try:
                c, _ = srv.accept()
            except socket.timeout:
                continue
            with c:
                c.settimeout(None)
                req = json.loads(_recv_line(c))
                if req.get("cmd") == "ping":
                    c.sendall((json.dumps({"ok": True, "mode": mode}) + "\n").encode())
                    continue
                if req.get("cmd") == "stop":
                    c.sendall(b'{"ok": true}\n')
                    break
                try:
                    r = s.say(req["text"], req.get("voice"), req.get("seed"), req.get("cfg"), req.get("steps"), req.get("out"))
                    r["ok"] = True
                except Exception as e:  # noqa: BLE001
                    r = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
                log(json.dumps({k: v for k, v in r.items() if k != "text"}, ensure_ascii=False))
                c.sendall((json.dumps(r, ensure_ascii=False) + "\n").encode())
                last = time.time()
    finally:
        srv.close()
        SOCK.unlink(missing_ok=True)
        log("погас")
    return 0


def _recv_line(c):
    buf = b""
    while not buf.endswith(b"\n"):
        b = c.recv(65536)
        if not b:
            break
        buf += b
    return buf.decode()


def _ask(req, timeout=None):
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.settimeout(timeout)
    c.connect(str(SOCK))
    with c:
        c.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode())
        return json.loads(_recv_line(c))


def cmd_say(a):
    up = SOCK.exists()
    if up:
        try:
            _ask({"cmd": "ping"}, timeout=3)
        except OSError:
            up = False
    if not up:
        CACHE.mkdir(parents=True, exist_ok=True)
        lf = open(CACHE / "serve.log", "a")
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "serve", "--mode", a.mode, "--idle", str(a.idle)],
                         stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        log(f"гружу модель (первый раз ~1 мин на GPU, на CPU дольше); журнал {CACHE / 'serve.log'}")
        t = time.time()
        while time.time() - t < 900:
            time.sleep(2)
            try:
                _ask({"cmd": "ping"}, timeout=3)
                break
            except OSError:
                continue
        else:
            log("модель не поднялась за 15 мин — смотри", CACHE / "serve.log")
            return 1
    log("синтезирую…")
    r = _ask({"text": a.text, "voice": a.voice, "seed": a.seed, "out": a.out})
    if not r.get("ok"):
        log("ОШИБКА:", r.get("error"))
        return 1
    log(f"{r['audio_s']} с звука за {r['synth_s']} с (RTF {r['rtf']}, режим {r['mode']}, кусков {r['chunks']}, повторов {r['retries']})")
    log("текст модели:", r["text"])
    log("файл:", r["out"])
    if not a.no_play:
        for p in (["pw-play", r["out"]], ["paplay", r["out"]], ["aplay", "-q", r["out"]]):
            try:
                subprocess.run(p, check=True)
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
    return 0


def cmd_stop(a):
    try:
        _ask({"cmd": "stop"}, timeout=10)
        log("остановлен")
    except OSError:
        log("не запущен")
    return 0


def cmd_status(a):
    try:
        r = _ask({"cmd": "ping"}, timeout=3)
        log("работает, режим", r.get("mode"))
    except OSError:
        log("не запущен")
    return 0


# ---------------------------------------------------------------- оценка как synth_eval.py
def cmd_eval(a):
    import numpy as np
    import soundfile as sf
    import torch
    P = json.loads(Path(a.phrases).read_text())
    phrases = P["phrases"]
    if a.ids:
        want = set(a.ids.split(","))
        phrases = [p for p in phrases if p["id"] in want]
    if a.cats:
        want = set(a.cats.split(","))
        phrases = [p for p in phrases if p.get("cat") in want]
    tts, mode = load_model(a.mode)
    sr = tts.tts_model.sample_rate
    out = Path(a.out)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rep = {"version": f"local-{mode}", "checkpoint": str(Q8 if mode == "gpu" else LORA), "ref": a.ref, "seed": a.seed, "cfg": a.cfg,
           "made": time.strftime("%Y-%m-%d %H:%M"), "items": {}}
    for k, p in enumerate(phrases):
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        t = time.time()
        wav = tts.generate(text=p["text"], cfg_value=a.cfg, inference_timesteps=a.timesteps, seed=a.seed, reference_wav_path=a.ref)
        fp = out / "ref" / f"{p['id']}.wav"
        fp.parent.mkdir(parents=True, exist_ok=True)
        sf.write(fp, wav, sr)
        it = dict(audio_s=round(len(wav) / sr, 2), compute_s=round(time.time() - t, 1))
        rep["items"][f"ref/{p['id']}"] = it
        log(f"[{k + 1}/{len(phrases)}] {p['id']}: {it['audio_s']} с за {it['compute_s']} с")
        (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    au = sum(v["audio_s"] for v in rep["items"].values())
    cs = sum(v["compute_s"] for v in rep["items"].values())
    rep["summary"] = dict(phrases=len(phrases), audio_s=round(au, 1), compute_s=round(cs, 1), rtf=round(cs / max(au, 1e-6), 3),
                          vram_peak_torch_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None,
                          rss_peak_gib=round(rss_peak_gb(), 2), mode=mode)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    log("итог:", json.dumps(rep["summary"], ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- поток: первый звук
BENCH = [
    "Сәлеметсіз бе. Тыңдап тұрмын, сізге қалай көмектесе аламын?",
    "Иә, жазып алдым. Ертең сағат онда шебер келеді, сізге ыңғайлы ма?",
    "Жақсы, онда заявкаңызды қабылдадым, бір сағат ішінде хабарласамыз.",
    "Здравствуйте! Подскажите, пожалуйста, на какой адрес нужен мастер?",
    "Кешіріңіз, нақты бағасын менеджер айтады. Нөміріңізді қалдырасыз ба?",
    "Рахмет, күтеміз.",
]


def cmd_bench(a):
    import numpy as np
    import torch
    sys.path.insert(0, str(HERE))
    import textprep
    tts, mode = load_model(a.mode)
    sr = tts.tts_model.sample_rate
    ref = str(ROOT / "voices/aruzhan_f_kk.wav")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rows = []
    texts = (BENCH * 3)[: a.n]
    for k, raw in enumerate(texts):
        text = " ".join(c["text"] for c in textprep.prepare(raw))
        torch.manual_seed(1)
        np.random.seed(1)
        t = time.time()
        first, n = None, 0
        for w in tts.generate_streaming(text=text, reference_wav_path=ref, cfg_value=2.0, inference_timesteps=10, seed=1):
            if first is None:
                first = time.time() - t
            n += np.asarray(w).reshape(-1).shape[0]
        dt = time.time() - t
        au = n / sr
        rows.append(dict(i=k, chars=len(text), audio_s=round(au, 2), first_s=round(first or 0, 3), synth_s=round(dt, 2),
                         rtf=round(dt / max(au, 1e-6), 3), realtime_ok=bool(dt - first <= au)))
        log(json.dumps(rows[-1], ensure_ascii=False))
    warm = rows[1:] or rows
    s = dict(mode=mode, n=len(rows), first_s_median=sorted(r["first_s"] for r in warm)[len(warm) // 2],
             rtf_median=sorted(r["rtf"] for r in warm)[len(warm) // 2],
             vram_peak_torch_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None,
             rss_peak_gib=round(rss_peak_gb(), 2))
    log("итог (без первой, холодной):", json.dumps(s, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(summary=s, rows=rows), ensure_ascii=False, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("say")
    p.add_argument("text")
    p.add_argument("--voice", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--mode", default=os.environ.get("LOCAL_MODE", "auto"))
    p.add_argument("--idle", type=float, default=float(os.environ.get("LOCAL_IDLE", "15")))
    p = sub.add_parser("serve")
    p.add_argument("--mode", default="auto")
    p.add_argument("--idle", type=float, default=15)
    sub.add_parser("stop")
    sub.add_parser("status")
    p = sub.add_parser("eval")
    p.add_argument("--phrases", default=str(ROOT / "eval/phrases_v2.json"))
    p.add_argument("--ref", default=str(ROOT / "eval/ref_kk.wav"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--ids", default="")
    p.add_argument("--cats", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="gpu")
    p = sub.add_parser("bench")
    p.add_argument("--mode", default="gpu")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--out", default="")
    a = ap.parse_args()
    return {"say": cmd_say, "serve": cmd_serve, "stop": cmd_stop, "status": cmd_status, "eval": cmd_eval, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
