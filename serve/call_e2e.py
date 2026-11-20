#!/usr/bin/env python3
"""call_e2e.py — сквозная проверка страницы звонка в headless Chromium с поддельным микрофоном (файл WAV вместо голоса).
Только headless (рабочий стол не трогаем). Нужен playwright с Chromium.

  python3 call_e2e.py --url https://127.0.0.1:PORT --token-file CALL_TOKEN --out DIR [--turns 3]
Микрофон: 1,5 с тишины → 1,2 с «речи» (гармоники с модуляцией) → 2,5 с тишины, по кругу. Страница сама ловит конец фразы,
шлёт реплику, проигрывает поток; итог — window.__call.turns (этапы в мс). Выход 0 — голос ответил ≥ turns раз, этапы на месте.
"""
import argparse
import json
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np


def fake_mic(path: Path, sr=48000):
    t1 = np.arange(int(1.2 * sr)) / sr
    sp = sum(0.12 / k * np.sin(2 * np.pi * 160 * k * t1) for k in (1, 2, 3, 4)) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t1))
    noise = np.random.default_rng(0).normal(0, 0.0015, int(5.2 * sr))
    x = noise.copy()
    a = int(1.5 * sr)
    x[a:a + len(sp)] += sp
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--turns", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mic = out / "fake_mic.wav"
    fake_mic(mic)
    tok = Path(a.token_file).read_text().strip()
    from playwright.sync_api import sync_playwright
    fails = []
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                                                   f"--use-file-for-fake-audio-capture={mic}", "--autoplay-policy=no-user-gesture-required"])
        ctx = b.new_context(ignore_https_errors=True, viewport={"width": 390, "height": 900})   # ширина телефона
        page = ctx.new_page()
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(f"{a.url}/call#k={tok}")
        page.wait_for_timeout(500)
        if "#k=" in page.url:
            fails.append("токен остался в адресной строке")
        page.click("#call")
        t0 = time.time()
        n = 0
        while time.time() - t0 < 60:
            n = page.evaluate("window.__call.turns.length")
            if n >= a.turns:
                break
            page.wait_for_timeout(300)
        turns = page.evaluate("window.__call.turns")
        page.screenshot(path=str(out / "call_page.png"), full_page=True)
        page.click("#hangup")
        b.close()
    (out / "e2e_turns.json").write_text(json.dumps(turns, ensure_ascii=False, indent=1))
    if errs:
        fails.append("ошибки JS: " + "; ".join(errs)[:300])
    greet = [t for t in turns if t.get("settings", {}).get("greet")]
    real = [t for t in turns if not t.get("settings", {}).get("greet") and not t.get("error")]
    if not greet or not greet[0].get("reply"):
        fails.append("нет приветствия")
    if len(real) < a.turns - 1:
        fails.append(f"реплик {len(real)} < {a.turns - 1}")
    for t in real:
        miss = [k for k in ("total_ms", "vad_ms", "net_ms", "asr_ms", "brain_ms", "tts_ms", "buf_ms", "rtf") if t.get(k) is None]
        if miss:
            fails.append(f"реплика {t.get('n')}: нет {miss}")
    med = lambda k: statistics.median([t[k] for t in real if isinstance(t.get(k), (int, float))] or [float("nan")])  # noqa: E731
    print(f"страница: приветствие {greet[0].get('greet_total_ms') if greet else '—'} мс от запроса; реплик {len(real)}; "
          f"медианы: тишина→звук {med('total_ms'):.0f} мс = конец фразы {med('vad_ms'):.0f} + сеть {med('net_ms'):.0f} + ASR {med('asr_ms'):.0f} "
          f"+ мозг {med('brain_ms'):.0f} + голос {med('tts_ms'):.0f} + буфер/вывод {med('buf_ms') - med('first_byte_ms'):.0f} мс; "
          f"RTF {med('rtf'):.2f}, провалов {sum(1 for t in real if t.get('under'))}")
    for t in real[:3]:
        print(f"  #{t['n']}: «{t.get('asr')}» → «{t.get('reply')}» · {t.get('total_ms')} мс")
    print("e2e: " + ("ВСЁ ПРОШЛО" if not fails else "НЕ ПРОШЛО: " + "; ".join(fails)) + f"; снимок {out / 'call_page.png'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
