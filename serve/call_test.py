#!/usr/bin/env python3
"""call_test.py — проверка call_server.py в песочнице (tts_server.py с TTS_MOCK=1 + call_server.py с CALL_ASR=mock, CALL_BRAIN=script):
TLS, токен, реплика, приветствие, поток, этапы, занятость, «тест закончен», журнал без текста.

  python call_test.py --url https://127.0.0.1:PORT --token-file CALL_TOKEN --log CALL_LOG --done-file CALL_DONE
"""
import argparse
import http.client
import json
import ssl
import threading
import time
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import numpy as np

fails = []


def check(name, ok, info=""):
    print(("ok   " if ok else "FAIL ") + name + (f" — {info}" if info else ""))
    if not ok:
        fails.append(name)


CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE                 # самоподписанный — как «всё равно открыть» в браузере


def call(url, method, path, token=None, body=b"", headers=None, timeout=30):
    u = urlsplit(url)
    c = (http.client.HTTPSConnection(u.hostname, u.port, timeout=timeout, context=CTX) if u.scheme == "https"
         else http.client.HTTPConnection(u.hostname, u.port, timeout=timeout))
    h = dict(headers or {})
    if token:
        h["Authorization"] = "Bearer " + token
    t0 = time.time()
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    t_head = time.time() - t0
    times, parts = [], []
    while True:
        b = r.read1(65536)
        if not b:
            break
        parts.append(b)
        times.append(time.time() - t0)
    c.close()
    return r.status, {k.lower(): v for k, v in r.getheaders()}, b"".join(parts), t_head, times


def speech(sec=1.2, sr=16000):
    t = np.arange(int(sec * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    return (x * 32767).astype("<i2").tobytes()


def main():
    ap = argparse.ArgumentParser()
    for k in ("url", "token-file", "log", "done-file"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    url, tok = a.url, Path(a.token_file).read_text().strip()

    st, h, b, _, _ = call(url, "GET", "/call")
    check("GET /call по https: страница", st == 200 and b"<title>" in b and h.get("content-type", "").startswith("text/html"), str(st))
    st, h, b, _, _ = call(url, "GET", "/call/health")
    j = json.loads(b or b"{}")
    check("/call/health: готов, голос готов, поток объявлен", st == 200 and j.get("ready") and j.get("tts", {}).get("ready")
          and j["tts"].get("stream", {}).get("sr_default") == 24000, json.dumps({k: j.get(k) for k in ("ready", "asr", "brain")}, ensure_ascii=False))
    st, *_ = call(url, "POST", "/call/turn?call=test01", None, speech())
    check("реплика без токена → 401", st == 401, str(st))
    st, *_ = call(url, "POST", "/call/turn?call=test01", "x" * 40, speech())
    check("чужой токен → 401", st == 401, str(st))
    # приветствие: только голос (звонок начинается с greet=1 — занимает линию)
    st, h, b, t_head, times = call(url, "POST", "/call/turn?greet=1&lang=kk&steps=6&call=test01", tok, b"", {"X-Sample-Rate": "16000"})
    check("приветствие: 200, PCM16 24 кГц, текст в X-Reply-Text, ASR не звался", st == 200 and h.get("x-sample-rate") == "24000"
          and "көмектесе" in unquote(h.get("x-reply-text", "")) and h.get("x-asr-text") == "" and h.get("x-steps") == "6",
          f"{st} «{unquote(h.get('x-reply-text', ''))[:40]}…», байт {len(b)}")
    st, *_ = call(url, "POST", "/call/turn?call=test01", tok, b"")
    check("пустой звук (не приветствие) → 400", st == 400, str(st))

    # обычная реплика: заглушка ASR («…бағасы қанша?») → правило «цена» → голос
    st, h, b, t_head, times = call(url, "POST", "/call/turn?brain=script&steps=6&cfg=2.0&voice=aruzhan_f&call=test01", tok, speech(),
                                   {"X-Sample-Rate": "16000"})
    reply, asr = unquote(h.get("x-reply-text", "")), unquote(h.get("x-asr-text", ""))
    check("реплика: 200 chunked, ASR → мозг (правило «цена») → голос", st == 200 and h.get("transfer-encoding") == "chunked"
          and "бағасы" in asr and "бағасын" in reply and h.get("x-brain") == "script" and h.get("x-lang") == "kk", f"«{asr}» → «{reply}»")
    stages = {k: int(h.get(k, -1)) for k in ("x-recv-ms", "x-asr-ms", "x-brain-ms", "x-tts-head-ms", "x-tts-ttfa-ms", "x-server-ms")}
    check("этапы в заголовках, сумма ≈ X-Server-Ms", all(v >= 0 for v in stages.values())
          and abs(stages["x-recv-ms"] + stages["x-asr-ms"] + stages["x-brain-ms"] + stages["x-tts-head-ms"] - stages["x-server-ms"]) <= 3,
          str(stages))
    check("звук идёт потоком (первые байты сильно раньше конца)", len(times) > 3 and times[-1] - times[0] > 0.5,
          f"кусков {len(times)}, первый {times[0]:.2f} с, последний {times[-1]:.2f} с")
    rid, reply_audio = h.get("x-request-id", ""), b
    time.sleep(0.3)
    st, _, b, _, _ = call(url, "GET", f"/call/stats?rid={rid}", tok)
    js = json.loads(b or b"{}")
    ts = js.get("tts") or {}
    check("GET /call/stats: этапы + итог голоса (RTF, запас, длительность)", st == 200 and js.get("status") == 200 and ts.get("rtf")
          and ts.get("prebuffer_ms") is not None and js.get("asr_ms") is not None, f"tts {ts}")
    got = len(reply_audio) / 2 / 24000
    check("звук, дошедший до клиента = audio_s голоса", abs(got - ts.get("audio_s", -1)) < 0.02, f"{got:.2f} с против {ts.get('audio_s')} с")

    # эхо, мозг с задержкой, русская фраза
    st, h, b, *_ = call(url, "POST", "/call/turn?brain=echo&brain_delay=300&call=test01", tok, speech(),
                        {"X-Sample-Rate": "16000", "X-Mock-Text": quote("Здравствуйте, сколько стоит?")})
    check("эхо + задержка мозга 300 мс, русский", st == 200 and unquote(h.get("x-reply-text", "")).startswith("Вы сказали")
          and int(h.get("x-brain-ms", 0)) >= 300 and h.get("x-lang") == "ru", f"{unquote(h.get('x-reply-text', ''))[:50]} · мозг {h.get('x-brain-ms')} мс")
    st, h, b, *_ = call(url, "POST", "/call/turn?call=test01", tok, speech(0.5), {"X-Sample-Rate": "16000", "X-Mock-Text": ""})
    check("пустое распознавание (эхо/тишина) → 422 no_speech, без ответа по кругу", st == 422, str(st))
    st, h, b, *_ = call(url, "POST", "/call/turn?call=test01", tok, speech(1.0, 48000), {"X-Sample-Rate": "48000"})
    check("микрофон 48 кГц → сервер сам понижает до 16 кГц", st == 200, str(st))

    # занятость: вторая реплика, пока идёт первая → 503
    res = []

    def one():
        res.append(call(url, "POST", "/call/turn?brain=script&call=test01", tok, speech(), {"X-Sample-Rate": "16000"})[0])
    th = [threading.Thread(target=one) for _ in range(2)]
    for t in th:
        t.start()
        time.sleep(0.02)
    for t in th:
        t.join()
    check("две реплики разом: одна 200, вторая 503 busy", sorted(res) == [200, 503], str(res))

    # закончить тест
    Path(a.done_file).unlink(missing_ok=True)
    st, *_ = call(url, "POST", "/call/done", tok)
    check("POST /call/done → файл CALL_DONE (сеанс идёт к замерам)", st == 200 and Path(a.done_file).exists(), str(st))
    time.sleep(0.2)
    lg = Path(a.log).read_text()
    check("журнал: этапы есть, текстов реплик и токена нет", "turn 200 ok" in lg and "бағасы" not in lg and "сколько" not in lg and tok not in lg)
    print(f"\nзвонок (HTTP): {'ВСЁ ПРОШЛО' if not fails else 'НЕ ПРОШЛО: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
