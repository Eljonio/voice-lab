#!/usr/bin/env python3
"""stream_test.py — проверка POST /tts/stream и GET /tts/stats в песочнице (tts_server.py с TTS_MOCK=1, без модели и GPU).

  python stream_test.py --url http://127.0.0.1:PORT --token-file T --log L --calls CALLS.jsonl --out DIR
Выход 0 — всё прошло. Сервис: TTS_MOCK=1, TTS_QUEUE_MAX=2, TTS_DEADLINE=6, TTS_MOCK_STREAM_RTF=0.2, TTS_MOCK_LEAD=0.3.
"""
import argparse
import http.client
import json
import sys
import threading
import time
import wave
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamio  # noqa: E402

MARK = "Қызылжарлықтарға"
KK = (f"Сәлеметсіз бе, {MARK}! Шебер ертең сағат 10:30-да келеді. Жұмыс шамамен екі сағатқа созылады, бағасы 15 000 ₸. "
      "Төлемді жұмыс аяқталғаннан кейін жасайсыз. Сұрақтарыңыз болса, маған қоңырау шалыңыз, мен әрқашан көмектесемін. Рахмет!")
SHORT = "Иә, жазып қойдым. Шебер ертең келеді."
fails = []


def check(name, ok, info=""):
    print(("ok   " if ok else "FAIL ") + name + (f" — {info}" if info else ""))
    if not ok:
        fails.append(name)


class Stream:
    """Один поток: время заголовков, время каждого куска, байты."""

    def __init__(self, url, token, body, path="/tts/stream", stop_after=None, timeout=60):
        u = urlsplit(url)
        self.t0 = time.time()
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
        h = {"Content-Type": "application/json"}
        if token:
            h["Authorization"] = "Bearer " + token
        c.request("POST", path, body=json.dumps(body, ensure_ascii=False).encode(), headers=h)
        r = c.getresponse()
        self.t_head = time.time() - self.t0
        self.status, self.headers = r.status, {k.lower(): v for k, v in r.getheaders()}
        self.parts, self.times, self.truncated = [], [], False
        while True:
            try:
                b = r.read1(65536) if hasattr(r, "read1") else r.read(4096)
            except http.client.IncompleteRead as e:     # сервер оборвал поток (срок) — без завершающего куска
                self.parts.append(e.partial)
                self.truncated = True
                break
            if not b:
                break
            self.parts.append(b)
            self.times.append(time.time() - self.t0)
            if stop_after and sum(map(len, self.parts)) >= stop_after:
                c.close()               # клиент «положил трубку»
                break
        self.t_end = time.time() - self.t0
        self.data = b"".join(self.parts)
        c.close()

    def json(self):
        try:
            return json.loads(self.data)
        except ValueError:
            return {}


def get_stats(url, token, rid):
    u = urlsplit(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    c.request("GET", f"/tts/stats?rid={rid}", headers={"Authorization": "Bearer " + token})
    r = c.getresponse()
    b = r.read()
    c.close()
    return r.status, (json.loads(b) if b else {})


def calls_since(path, pos):
    p = Path(path)
    if not p.exists():
        return [], pos
    lines = p.read_text().splitlines()
    return [json.loads(x) for x in lines[pos:]], len(lines)


def main():
    ap = argparse.ArgumentParser()
    for k in ("url", "token-file", "log", "calls", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    url, token = a.url, Path(a.token_file).read_text().strip()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. защита и проверка полей
    s = Stream(url, None, {"text": SHORT})
    check("без токена → 401", s.status == 401, str(s.status))
    for name, body, want, code in [("sr 22050", {"text": SHORT, "sr": 22050}, 400, "bad_param"),
                                   ("encoding opus", {"text": SHORT, "encoding": "opus"}, 400, "bad_format"),
                                   ("mulaw на 24 кГц", {"text": SHORT, "encoding": "mulaw", "sr": 24000}, 400, "bad_format"),
                                   ("пустой текст", {"text": " "}, 400, "empty_text")]:
        s = Stream(url, token, body)
        check(f"{name} → {want} {code}", s.status == want and s.json().get("error") == code, f"{s.status} {s.data[:80]!r}")

    # 2. основной путь: 24 кГц PCM16, куски приходят по ходу синтеза, тишина в начале срезана
    _, pos = calls_since(a.calls, 0)
    s = Stream(url, token, {"text": KK, "sr": 24000})
    h = s.headers
    x = np.frombuffer(s.data[: len(s.data) // 2 * 2], "<i2").astype(np.float32) / 32768
    with wave.open(str(out / "stream_kk_24k.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(s.data[: len(s.data) // 2 * 2])
    check("200, chunked, octet-stream, X-Sample-Rate 24000, X-Encoding pcm16",
          s.status == 200 and h.get("transfer-encoding") == "chunked" and h.get("x-sample-rate") == "24000"
          and h.get("x-encoding") == "pcm16" and h.get("content-type") == "application/octet-stream", str({k: h.get(k) for k in
                                                                                                           ("transfer-encoding", "x-sample-rate")}))
    ttfa = float(h.get("x-ttfa-ms", -1))
    check("X-TTFA-Ms есть и ≈ времени заголовков у клиента", ttfa > 0 and abs(ttfa / 1000 - s.t_head) < 0.15,
          f"сервер {ttfa:.0f} мс, клиент {s.t_head * 1000:.0f} мс, X-TTFA-Model-Ms {h.get('x-ttfa-model-ms')}")
    check("звук идёт кусками: первый байт раньше конца ≥ чем на 1 с", len(s.times) >= 5 and s.t_end - s.times[0] > 1.0 and not s.truncated,
          f"кусков {len(s.times)}, первый {s.times[0]:.2f} с, конец {s.t_end:.2f} с")
    nz = np.where(np.abs(x) > 1e-3)[0]
    lead = nz[0] / 24000 if len(nz) else 9
    check("тишина в начале срезана (≤ 70 мс; в заглушке 300 мс)", lead <= 0.07, f"{lead * 1000:.0f} мс")
    dur = len(x) / 24000
    check("длительность разумная (2 куска текста + пауза)", 8 < dur < 20, f"{dur:.2f} с")
    peak = float(np.max(np.abs(x))) if len(x) else 0
    check("пик ≤ −1 dBFS (ограничитель)", 0.05 < peak <= 0.9, f"{peak:.3f}")
    cl, pos = calls_since(a.calls, pos)
    cl = [c for c in cl if c.get("stream")]
    check("chain в потоке: кусок 1 только эталон, кусок 2 — продолжение предыдущего",
          len(cl) >= 2 and cl[0]["prompt"] is None and all(c["prompt"] == "prev" for c in cl[1:]), str([(c["chunk"], c["prompt"]) for c in cl]))
    rid = h.get("x-request-id", "")
    st, js = get_stats(url, token, rid)
    check("GET /tts/stats: ttfa, audio, synth, rtf, prebuffer",
          st == 200 and js.get("status") == 200 and abs(js.get("audio_s", 0) - dur) < 0.05 and js.get("rtf") and js.get("prebuffer_ms") is not None,
          f"{st} ttfa {js.get('ttfa_ms')} мс, звук {js.get('audio_s')} с, RTF {js.get('rtf')}, запас {js.get('prebuffer_ms')} мс, "
          f"срезано в начале {js.get('lead_cut_ms')} мс")
    check("RTF заглушки 0,2 → запас буфера ≈ 0", js.get("prebuffer_ms", 999) <= 60, f"{js.get('prebuffer_ms')} мс")

    # 3. 8 кГц μ-law (телефония) и 16 кГц
    s = Stream(url, token, {"text": SHORT, "sr": 8000, "encoding": "mulaw"})
    y = streamio.mulaw_decode(s.data)
    js = get_stats(url, token, s.headers.get("x-request-id", ""))[1]
    check("8 кГц μ-law: 1 байт на отсчёт, длина = audio_s", s.status == 200 and abs(len(s.data) / 8000 - js.get("audio_s", 0)) < 0.01
          and float(np.max(np.abs(y))) > 0.05, f"{len(s.data)} байт = {len(s.data) / 8000:.2f} с")
    s = Stream(url, token, {"text": SHORT, "sr": 16000, "voice": "male_m", "mode": "ref", "steps": 6, "cfg": 1.5})
    check("16 кГц, male_m, ref, 6 шагов, cfg 1,5 → 200 и заголовки", s.status == 200 and s.headers.get("x-steps") == "6"
          and s.headers.get("x-voice") == "male_m" and s.headers.get("x-cfg") == "1.5", str({k: s.headers.get(k) for k in ("x-steps", "x-voice", "x-cfg")}))

    # 4. медленный синтез (RTF > 1) → stats показывает нужный запас буфера
    s = Stream(url, token, {"text": SHORT, "_mock_delay": 5.0})
    js = get_stats(url, token, s.headers.get("x-request-id", ""))[1]
    check("RTF > 1 → prebuffer_ms > 0 (провалы без запаса)", (js.get("rtf") or 0) > 1 and js.get("prebuffer_ms", 0) > 300,
          f"RTF {js.get('rtf')}, запас {js.get('prebuffer_ms')} мс")

    # 5. клиент бросил трубку посреди потока → генерация останавливается, очередь свободна
    t = time.time()
    s = Stream(url, token, {"text": KK, "_mock_delay": 6.0}, stop_after=20000)
    time.sleep(1.5)
    s2 = Stream(url, token, {"text": SHORT})
    check("после брошенной трубки следующий поток идёт сразу (≤ 2,5 с)", s2.status == 200 and time.time() - t < 6,
          f"{s2.status}, X-Queue-Ms {s2.headers.get('x-queue-ms')}")
    time.sleep(0.3)
    lg = Path(a.log).read_text()
    check("в журнале брошенный поток = 499 client_gone", "stream 499 client_gone" in lg)

    # 6. очередь: 1 идёт + 2 ждут, 4-й и 5-й → 503 busy
    res = []

    def one(i):
        res.append(Stream(url, token, {"text": SHORT, "seed": 50 + i, "_mock_delay": 1.0}))
    th = [threading.Thread(target=one, args=(i,)) for i in range(5)]
    for t_ in th:
        t_.start()
        time.sleep(0.05)
    for t_ in th:
        t_.join()
    codes = sorted(r.status for r in res)
    check("5 одновременных потоков: 3 × 200, 2 × 503 busy", codes.count(200) == 3 and codes.count(503) == 2, str(codes))
    hh = json.loads(__import__("urllib.request").request.urlopen(url + "/health", timeout=5).read())
    check("модель ни разу не работала параллельно", hh.get("mock_max_parallel") == 1, f"максимум {hh.get('mock_max_parallel')}")
    check("/health: блок stream (sr, кодировки, усиление)", hh.get("stream", {}).get("sr_default") == 24000 and "mulaw" in hh["stream"]["encodings"],
          str(hh.get("stream")))

    # 7. срок (TTS_DEADLINE=6) посреди потока → поток обрывается, журнал 504
    s = Stream(url, token, {"text": KK * 3, "_mock_delay": 3.0}, timeout=60)
    time.sleep(0.3)
    lg = Path(a.log).read_text()
    check("поток дольше срока → обрыв без завершающего куска, в журнале stream 504 timeout", "stream 504 timeout" in lg and s.truncated,
          f"{s.status}, {len(s.data)} байт до обрыва")
    check("в журнале нет текста клиента и токена", MARK not in lg and "Шебер" not in lg and token not in lg)
    print(f"\nпоток: {'ВСЁ ПРОШЛО' if not fails else 'НЕ ПРОШЛО: ' + ', '.join(fails)}; образец: {out / 'stream_kk_24k.wav'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
