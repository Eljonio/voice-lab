#!/usr/bin/env python3
"""call_test_bot.py — звонок с мозгом «внешний бот» (bot) на заглушках, без GPU и без моделей.

Сам поднимает песочницу: голос tts_server.py (TTS_MOCK=1, эталоны — тишина), бот mock_bot.py (контракт /voice/turn), call_server.py
(ASR-заглушка, CALL_BRAIN=bot, без TLS — как за туннелем). Проверяет полный круг start → реплики → end и поведение при сбоях.

  python3 call_test_bot.py [--keep]           (заглушки, секунды CPU)
  python3 call_test_bot.py --url URL --token-file F --secret-file S --bot-log L [--mode-url U]
                                              — только проверки против уже поднятого сервера с заглушкой бота
Выход 0 — всё прошло.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import guestlink  # noqa: E402

fails: list[str] = []


def check(name, ok, info=""):
    print(("ok   " if ok else "FAIL ") + name + (f" — {info}" if info else ""), flush=True)
    if not ok:
        fails.append(name)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def req(url, method, path, token=None, body=b"", headers=None, timeout=30):
    u = urlsplit(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    h = dict(headers or {})
    if token:
        h["Authorization"] = "Bearer " + token
    t0 = time.time()
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    t_head = time.time() - t0
    parts, times = [], []
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


def dummy_voices(d: Path):
    """Эталоны голосов в репозиторий не входят — заглушке голоса (TTS_MOCK=1) хватает тишины с теми же именами файлов."""
    import wave
    d.mkdir(parents=True, exist_ok=True)
    for v in json.loads((HERE / "voices.json").read_text())["voices"].values():
        with wave.open(str(d / v["wav"]), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\0\0" * 16000)
        if v.get("text"):
            (d / v["text"]).write_text("Сәлеметсіз бе.\n", encoding="utf-8")


class Sandbox:
    def __init__(self, keep=False, idle="4"):
        self.d = Path(tempfile.mkdtemp(prefix="call-bot-"))
        self.keep, self.procs = keep, {}
        h = self.d
        (h / "logs").mkdir()
        dummy_voices(h / "voices")
        for f in ("token", "call_token", "bot_key"):
            (h / f).write_text(secrets.token_urlsafe(32) + "\n")
        guestlink.init_secret(h / "guest_secret")
        self.tp, self.gp, self.cp = free_port(), free_port(), free_port()
        self.url = f"http://127.0.0.1:{self.cp}"
        self.events = h / "bot_events.jsonl"
        py = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable
        self.py = py
        base = dict(os.environ, PYTHONUNBUFFERED="1")
        self.start("tts", [py, str(HERE / "tts_server.py")], dict(base, TTS_MOCK="1", TTS_HOST="127.0.0.1", TTS_PORT=str(self.tp), TTS_HOME=str(h),
                   TTS_TOKEN_FILE=str(h / "token"), TTS_VOICES_JSON=str(HERE / "voices.json"), TTS_VOICES_DIR=str(h / "voices"),
                   TTS_LOG=str(h / "logs" / "tts.log"), TTS_KEEPALIVE=str(h / "keepalive"), TTS_MOCK_LOAD="0.2", TTS_MOCK_STREAM_RTF="0.5",
                   TTS_MOCK_PREFILL="0.08", TTS_MOCK_LEAD="0.1"))
        self.start("bot", [sys.executable, str(HERE / "mock_bot.py"), "--port", str(self.gp), "--key-file", str(h / "bot_key"),
                             "--log", str(self.events), "--delay-ms", "200"], base)
        self.call_env = dict(base, CALL_HOST="127.0.0.1", CALL_PORT=str(self.cp), CALL_HOME=str(h), CALL_TOKEN_FILE=str(h / "call_token"),
                             TTS_URL=f"http://127.0.0.1:{self.tp}", TTS_TOKEN_FILE=str(h / "token"), CALL_ASR="mock", CALL_BRAIN="bot",
                             BOT_TURN_URL=f"http://127.0.0.1:{self.gp}/voice/turn", BOT_VOICE_KEY="", BOT_VOICE_KEY_FILE=str(h / "bot_key"),
                             BOT_TENANT="demo", BOT_STREAM="1",
                             CALL_GUEST_SECRET_FILE=str(h / "guest_secret"), CALL_IDLE_S=idle, CALL_FILLER_MS="700", CALL_FILLER2_MS="2500",
                             BOT_TIMEOUT_S="5", CALL_KEEPALIVE=str(h / "keepalive"), CALL_DONE_FILE=str(h / "CALL_DONE"),
                             CALL_LOG=str(h / "logs" / "call.log"), CALL_PUBLIC_URL_FILE=str(h / "public_url"),
                             CALL_LAB_GUEST_PHONE="77015550000", CALL_LAB_GUEST_NAME="Оператор")
        self.start_call()

    def start(self, name, cmd, env_):
        self.procs[name] = subprocess.Popen(cmd, env=env_, cwd=str(self.d), stdout=open(self.d / f"{name}.out", "ab"), stderr=subprocess.STDOUT)

    def start_call(self):
        self.start("call", [self.py, str(HERE / "call_server.py")], self.call_env)
        for _ in range(150):
            try:
                st, _, b, _, _ = req(self.url, "GET", "/call/health", timeout=3)
                if st == 200 and json.loads(b).get("ready"):
                    return
            except OSError:
                pass
            time.sleep(0.2)
        tail = (self.d / "call.out").read_text()[-2000:]
        self.stop()                                       # не оставлять сирот (голос и бот-заглушка), если звонок не поднялся
        raise SystemExit("call_server не поднялся: " + tail)

    def stop(self):
        import shutil
        for p in self.procs.values():
            if p.poll() is None:
                p.terminate()
        for p in self.procs.values():
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
        if not self.keep:
            shutil.rmtree(self.d, ignore_errors=True)


def events(path):
    try:
        return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    except OSError:
        return []


def run_checks(url, lab, secret, bot_log, mode_url=None, sb: Sandbox | None = None):
    def mode(**kw):
        if mode_url:
            u = urlsplit(mode_url)
            c = http.client.HTTPConnection(u.hostname, u.port, timeout=5)
            c.request("POST", "/mock/mode", body=json.dumps(kw).encode())
            c.getresponse().read()
            c.close()

    def turn(tok, call, text=None, greet=False, extra=""):
        path = f"/call/turn?call={call}&lang=kk" + ("&greet=1" if greet else "") + extra
        hd = {"X-Sample-Rate": "16000"}
        if text is not None:
            hd["X-Mock-Text"] = quote(text)
        return req(url, "POST", path, tok, b"" if greet else speech(), hd)

    def end(tok, call, reason="hangup"):
        st, _, b, _, _ = req(url, "POST", "/call/end", tok, json.dumps({"call": call, "reason": reason}).encode(), {"Content-Type": "application/json"})
        return st, json.loads(b or b"{}")

    def stats(rid, tok=lab):
        st, _, b, _, _ = req(url, "GET", f"/call/stats?rid={rid}", tok)
        return st, json.loads(b or b"{}")

    def evs(call):
        return [e for e in events(bot_log) if e.get("call") == call]

    # ---- страницы
    st, h, b, _, _ = req(url, "GET", "/")
    if (HERE / "client.html").is_file():
        check("GET / — страница клиента, CSP, микрофон разрешён только себе", st == 200
              and "content-security-policy" in h and "microphone=(self)" in h.get("permissions-policy", ""), str(st))
    else:
        check("GET / — страницы клиента нет в репозитории → 404, сервер не падает", st == 404, str(st))
    st, h, b, _, _ = req(url, "GET", "/call")
    check("GET /call — лабораторная страница, CSP, микрофон разрешён только себе", st == 200 and "Тест звонка".encode() in b
          and "content-security-policy" in h and "microphone=(self)" in h.get("permissions-policy", ""), str(st))
    if (HERE / "assets" / "mark.png").is_file():
        st, h, b, _, _ = req(url, "GET", "/assets/mark.png")
        check("картинка /assets/mark.png", st == 200 and h.get("content-type") == "image/png" and b[:4] == b"\x89PNG", str(st))
    st, _, _, _, _ = req(url, "GET", "/assets/../call_server.py")
    check("из /assets ничего, кроме картинок", st == 404, str(st))
    st, _, b, _, _ = req(url, "GET", "/call/health")
    hj = json.loads(b)
    check("health: готов, мозг bot, ключ есть, гости есть, линия свободна", hj.get("ready") and hj.get("brain") == "bot" and hj.get("bot")
          and hj.get("guests") and hj.get("busy") is False, json.dumps({k: hj.get(k) for k in ("ready", "brain", "bot", "guests", "busy")}))

    # ---- доступ
    st, *_ = turn(None, "nocall01", greet=True)
    check("без токена → 401", st == 401, str(st))
    tok = guestlink.make(secret, "Айбике", "8 701 234 56 78", hours=1, lang="ru")
    head, body, sig = tok.split(".")
    pay = json.loads(guestlink._b64d(body))
    pay["p"] = "77770000000"
    forged = ".".join([head, guestlink._b64e(json.dumps(pay, ensure_ascii=False, separators=(",", ":")).encode()), sig])
    st, *_ = turn(forged, "forge001", greet=True)
    check("поддельный номер в ссылке → 401", st == 401, str(st))
    old = guestlink.make(secret, "Айбике", "77012345678", hours=1, now=time.time() - 7200)
    st, _, b, _, _ = turn(old, "old00001", greet=True)
    check("ссылка с вышедшим сроком → 401 link_expired", st == 401 and b"link_expired" in b, f"{st} {b[:60]!r}")

    # ---- гость: start → turn → end
    call = "guestcall1"
    mode(delay_ms=200, stream=0, fail=None, fail_n=0)
    st, h, b, _, _ = turn(tok, call, greet=True)
    rep = stats(h.get("x-request-id"))[1].get("_reply", "")
    e = evs(call)
    check("start: бот здоровается сам (текст бота, не зашитый GREET), звук 24 кГц, без «секунду»",
          st == 200 and "Айбике" in rep and h.get("x-sample-rate") == "24000" and len(b) > 24000 and h.get("x-filler") == "0",
          f"{st} «{rep[:50]}» {len(b)} байт")
    check("start дошёл до бота: tenant demo, guest {phone, name} из ссылки, stream: true, без сжатия, тело строго по контракту",
          len(e) == 1 and e[0]["ev"] == "start" and e[0]["tenant"] == "demo" and e[0]["guest"] == {"phone": "77012345678", "name": "Айбике"}
          and e[0]["stream"] is True and e[0]["enc"] == "identity" and not e[0]["errs"],
          json.dumps(e[-1:] and {k: e[-1][k] for k in ("ev", "guest", "stream", "enc", "errs")}, ensure_ascii=False))
    st, h, b, _, _ = req(url, "GET", "/call/health")
    check("health: линия занята во время звонка", json.loads(b).get("busy") is True)
    st, h, b, _, _ = turn(tok, call, "Здравствуйте, сколько стоит подключение?")
    e = evs(call)
    js = stats(h.get("x-request-id"))[1]
    check("turn: речь клиента → бот (text, lang ru) → ответ бота голосом на русском, по предложениям",
          st == 200 and e[-1]["ev"] == "turn" and e[-1]["text"] == "Здравствуйте, сколько стоит подключение?" and e[-1]["lang"] == "ru"
          and h.get("x-lang") == "ru" and "Записать" in js.get("_reply", "") and js.get("segments", 0) >= 1, f"{st} {h.get('x-lang')} «{js.get('_reply', '')[:40]}»")
    st, h, b, _, _ = turn(tok, call, "Сәлеметсіз бе, ертеңге жазылайын деп едім")
    check("turn по-казахски → lang kk, ответ на казахском", st == 200 and evs(call)[-1]["lang"] == "kk" and h.get("x-lang") == "kk", h.get("x-lang", ""))
    rid = h.get("x-request-id")
    st, js = stats(rid, tok)
    check("гость видит в итоге реплики только status/end", st == 200 and set(js) == {"status", "end"}, json.dumps(js))
    st, js = stats(rid)
    check("лаборатория видит этапы и тексты реплики", st == 200 and js.get("_reply") and js.get("brain_ms") is not None and js.get("lang_in") == "kk",
          f"brain_ms {js.get('brain_ms')}")
    st, _, b, _, _ = turn(lab, "labcall01", greet=True)
    check("второй звонок, пока идёт первый → 409 line_busy", st == 409 and b"line_busy" in b, f"{st} {b[:60]!r}")
    st, js = end(tok, call)
    time.sleep(0.5)
    e = evs(call)
    check("end: кнопка «завершить» → бот получил end один раз (X-End-Reason hangup)", st == 200 and js.get("ended")
          and [x["ev"] for x in e] == ["start", "turn", "turn", "end"] and e[-1]["reason"] == "hangup", str([x["ev"] for x in e]))
    st, js = end(tok, call)
    time.sleep(0.3)
    check("повторный end не дублируется у бота", st == 200 and not js.get("ended") and [x["ev"] for x in evs(call)].count("end") == 1)
    st, _, b, _, _ = turn(tok, call, "ещё вопрос")
    check("реплика в закрытый звонок → 409 call_over", st == 409 and b"call_over" in b, str(st))
    st, h, b, _, _ = turn(lab, "labvoice01", greet=True)
    st, h, b, _, _ = turn(lab, "labvoice01", "Здравствуйте, сколько стоит?", extra="&voice_ru=" + quote("base:aruzhan_f_ru"))
    check("лаборатория: «база + русский эталон» без второго голоса — откат на v5 с тем же эталоном, без сбоя",
          st == 200 and h.get("x-voice") == "aruzhan_f_ru", f"{st} {h.get('x-voice')}")
    end(lab, "labvoice01")

    # ---- лаборатория: гость для проверки истории, ссылка гостя
    st, h, b, _, _ = turn(lab, "labcall02", greet=True, extra="&guest_phone=77770000000&guest_name=" + quote("Чужой"))
    e = evs("labcall02")
    check("лаборатория: гость — из настроек сервера (CALL_LAB_GUEST_PHONE), номер со страницы игнорируется",
          st == 200 and e and e[0]["guest"] == {"phone": "77015550000", "name": "Оператор"}, str(e[:1]))
    st, _, b, _, _ = req(url, "POST", "/call/guestlink", lab, json.dumps({"name": "Айбике", "phone": "+7 701 234 56 78", "hours": 24}).encode(),
                         {"Content-Type": "application/json"})
    check("выпуск ссылок гостей со страницы выключен (номер — только из выдачи оператора)", st == 404, str(st))
    for p in ("/call/guestlink", "/call/done", "/call/note"):
        st, *_ = req(url, "POST", p, tok, b'{"x":1}', {"Content-Type": "application/json"})
        check(f"гость не может {p} → 403", st == 403, str(st))
    end(lab, "labcall02")

    # ---- бот думает долго: «Секунду.» раньше ответа
    call = "slowcall01"
    turn(tok, call, greet=True)
    mode(delay_ms=1800)
    st, h, b, t_head, times = turn(tok, call, "Скажите, а когда мастер приедет?")
    rid = h.get("x-request-id")
    _, js = stats(rid)
    check("бот думает 1,8 с: первый звук («секунду») раньше ответа бота, потом ответ", st == 200 and h.get("x-filler") == "1" and t_head < 1.6
          and js.get("filler") == 1 and js.get("_reply") and js.get("brain_ms", 0) >= 1700, f"первый звук {t_head:.2f} с, мозг {js.get('brain_ms')} мс")
    check("заголовок ответа без текста бота (его ещё нет), текст — в итоге реплики", h.get("x-reply-text") == "" and "Записать" in js.get("_reply", ""))

    # ---- по предложениям (ndjson): голос начинает с первого
    mode(delay_ms=300, sent_gap_ms=1500)
    st, h, b, t_head, times = turn(tok, call, "Можно на завтра?")
    rid = h.get("x-request-id")
    _, js = stats(rid)
    e = evs(call)
    check("бот отдаёт по предложениям: звук первого раньше, чем бот договорил", st == 200 and t_head < 1.2 and js.get("segments") == 2
          and js.get("brain_total_ms", 0) >= 1700 and e[-1]["stream"] is True, f"первый звук {t_head:.2f} с, бот всё за {js.get('brain_total_ms')} мс, кусков {js.get('segments')}")
    mode(delay_ms=300, sent_gap_ms=200, search_ms=2500)
    st, h, b, t_head, times = turn(tok, call, "Найди, пожалуйста, курс доллара")
    _, js = stats(h.get("x-request-id"))
    check("бот ищет в интернете ({type:wait}) → сразу «Сейчас посмотрю, минутку», потом ответ", st == 200 and t_head < 1.2
          and js.get("bot_wait") == "search" and js.get("filler") == 1 and "Нацбанка" in js.get("_reply", ""),
          f"первый звук {t_head:.2f} с, ответ через {js.get('brain_ms')} мс")
    mode(delay_ms=200, json_only=1)
    st, h, b, t_head, times = turn(tok, call, "Хорошо, а адрес какой?")
    check("бот без потока (обычный JSON) — тоже работает, текст ответа в заголовке", st == 200 and "Записать" in unquote(h.get("x-reply-text", ""))
          and h.get("x-filler") == "0", unquote(h.get("x-reply-text", ""))[:40])
    mode(json_only=0, delay_ms=200)

    # ---- сбои бота
    mode(fail="500", fail_n=1)
    st, h, b, _, _ = turn(tok, call, "А цена какая?")
    _, js = stats(h.get("x-request-id"))
    check("бот ответил 500 → «связь прервалась, повторите» голосом, не тишина и не ошибка", st == 200 and js.get("brain") == "fallback"
          and "Повторите" in js.get("_reply", "") and js.get("bot_err") == "http_500" and not js.get("end"), js.get("_reply", ""))
    mode(fail="close", fail_n=1)
    st, h, b, _, _ = turn(tok, call, "А цена какая?")
    _, js = stats(h.get("x-request-id"))
    check("второй сбой подряд → «сбой связи, позвоните позже» и конец звонка", st == 200 and "позже" in js.get("_reply", "") and js.get("end") is True,
          f"{js.get('_reply')} end={js.get('end')}")
    st, h, b, _, _ = turn(tok, call, "А цена какая?")
    _, js = stats(h.get("x-request-id"))
    check("бот ожил → обычный ответ, счётчик сбоев сброшен", st == 200 and js.get("brain") == "bot" and not js.get("bot_err"), js.get("brain", ""))
    mode(fail="brain_down", fail_n=1)
    st, h, b, _, _ = turn(tok, call, "А цена какая?")
    _, js = stats(h.get("x-request-id"))
    check("бот ответил error: brain_down (reply пустой) → «связь прервалась, повторите»", st == 200 and js.get("bot_err") == "bot_brain_down"
          and "Повторите" in js.get("_reply", ""), js.get("_reply", ""))
    mode(fail="stream_error", fail_n=1)
    st, h, b, _, _ = turn(tok, call, "А цена какая?")
    _, js = stats(h.get("x-request-id"))
    check("поток оборвался {type:error} второй раз подряд → «позвоните позже» и конец", st == 200 and js.get("bot_err") == "bot_turn_failed"
          and js.get("end") is True, f"{js.get('bot_err')} end={js.get('end')}")
    turn(tok, call, "Алло?")
    mode(fail="hang", fail_n=1)
    t0 = time.time()
    st, h, b, t_head, _ = turn(tok, call, "Вы тут?")
    _, js = stats(h.get("x-request-id"))
    check("бот завис: «секунду» → «ещё секунду» → «повторите» по сроку (5 с), а не тишина", st == 200 and js.get("filler") == 2
          and js.get("bot_err") == "timeout" and t_head < 1.2 and time.time() - t0 < 9, f"{time.time() - t0:.1f} с, заполнителей {js.get('filler')}")
    mode(fail=None, fail_n=0)
    st, h, b, _, _ = turn(tok, call, "Ладно, молчи")
    check("бот решил промолчать → 204, страница снова слушает", st == 204, str(st))
    st, h, b, _, _ = turn(tok, call, "")
    check("тишина/эхо (пустое распознавание) → 422 no_speech, бот не дёргается", st == 422 and evs(call)[-1]["text"] != "", str(st))
    st, h, b, _, _ = turn(tok, call, "Спасибо, пока")
    _, js = stats(h.get("x-request-id"), tok)
    check("бот закончил разговор (end:true) → страница узнаёт из итога реплики и кладёт трубку", st == 200 and js.get("end") is True, json.dumps(js))
    end(tok, call, "bot_end")
    time.sleep(0.4)
    check("после end:true бота end всё равно уходит (бот просил слать всегда)", [x["ev"] for x in evs(call)][-1:] == ["end"])

    # ---- start: бот думает долго — без «секунду» (страница даёт гудок)
    mode(delay_ms=1500)
    st, h, b, t_head, _ = turn(tok, "slowstart1", greet=True)
    check("медленное приветствие бота: без нашего «секунду», первый звук — сам бот", st == 200 and h.get("x-filler") == "0" and t_head >= 1.4,
          f"первый звук {t_head:.2f} с")
    mode(delay_ms=200)
    end(tok, "slowstart1")

    # ---- бот не ответил на start → своё приветствие
    mode(fail="500", fail_n=1)
    st, h, b, _, _ = turn(tok, "startfail1", greet=True)
    _, js = stats(h.get("x-request-id"))
    check("start упал → Аружан всё равно здоровается (своё приветствие)", st == 200 and js.get("brain") == "greet_local", js.get("brain", ""))
    mode(fail=None, fail_n=0)
    st, js = end(tok, "startfail1")
    time.sleep(0.4)
    check("end после упавшего start всё равно уходит боту", [x["ev"] for x in evs("startfail1")][-1:] == ["end"])

    # ---- прогрев: страница открылась → приветствие и его звук заранее
    def warm(tok_, call_):
        return req(url, "POST", f"/call/warm?call={call_}&lang=ru", tok_)
    mode(start_ms=2500, delay_ms=200, prep_stale=0)
    st, h, b, t_no, _ = turn(tok, "nowarm001", greet=True)
    end(tok, "nowarm001")
    check("замер: без прогрева — от кнопки до первого звука приветствия", st == 200 and h.get("x-warm") == "0" and t_no >= 2.4, f"{t_no:.2f} с")
    st, _, wb, _, _ = warm(tok, "warm000001")
    check("warm отвечает сразу, на линию звонок не встаёт", st == 200 and json.loads(wb).get("ok") is True
          and json.loads(req(url, "GET", "/call/health")[2]).get("busy") is False, wb.decode()[:80])
    time.sleep(6.5)
    st, h, b, t_w, _ = turn(tok, "warm000001", greet=True)
    _, js = stats(h.get("x-request-id"))
    e = evs("warm000001")
    check("замер: с прогревом (кнопка через 6 с) — приветствие сразу из готового звука", st == 200 and h.get("x-warm") == "1" and t_w < 0.4
          and js.get("warm") == "hit" and js.get("warm_state") == "ready" and len(b) > 120000, f"{t_w:.2f} с (без прогрева {t_no:.2f} с), звук {len(b)} байт")
    check("бот: warm (wait: true, тот же guest и call) → start", [x["ev"] for x in e][:2] == ["warm", "start"] and e[0]["wait"] is True
          and e[0]["guest"] == e[1]["guest"] and not e[0]["errs"], str([(x["ev"], x.get("wait")) for x in e]))
    end(tok, "warm000001")
    st, _, _, _, _ = warm(tok, "warm000002")
    time.sleep(0.1)
    st, h, b, t_i, _ = turn(tok, "warm000002", greet=True)
    _, js = stats(h.get("x-request-id"))
    check("замер: кнопка сразу после открытия — бот дожидает прогрев, звук идёт хвостом того же синтеза, не хуже, чем без прогрева",
          st == 200 and h.get("x-warm") == "1" and js.get("warm_state") == "synth" and t_i <= t_no + 0.3, f"{t_i:.2f} с (без прогрева {t_no:.2f} с)")
    end(tok, "warm000002")
    warm(tok, "closed0001")
    time.sleep(0.3)
    st, js_ = end(tok, "closed0001", "page_closed")            # бот ещё готовит (2,5 с) — end уйдёт, когда он подтвердит
    time.sleep(3.0)
    e = evs("closed0001")
    check("открыли и закрыли без звонка → боту end (page_closed), start не было, линия свободна", js_.get("warm_closed") is True
          and [x["ev"] for x in e] == ["warm", "end"] and e[-1]["reason"] == "page_closed"
          and json.loads(req(url, "GET", "/call/health")[2]).get("busy") is False, str([(x["ev"], x.get("reason")) for x in e]))
    warm(tok, "stale00001")
    time.sleep(6.5)
    mode(prep_stale=1)
    st, h, b, t_s, _ = turn(tok, "stale00001", greet=True)
    _, js = stats(h.get("x-request-id"))
    check("беседа изменилась между прогревом и звонком → бот дал новый текст, звучит новый, а не прогретый", st == 200 and h.get("x-warm") == "0"
          and js.get("warm") == "miss:text" and "Снова" in js.get("_reply", ""), f"{js.get('warm')} «{js.get('_reply', '')[:30]}»")
    mode(prep_stale=0)
    end(tok, "stale00001")
    tok_b = guestlink.make(secret, "Другой", "77010000002", hours=1, lang="ru")
    warm(tok, "other00001")
    time.sleep(6.5)
    st, h, b, _, _ = turn(tok_b, "other00001", greet=True)
    _, js = stats(h.get("x-request-id"))
    check("прогрев одного гостя не достаётся другому (тот же call, чужая ссылка)", st == 200 and h.get("x-warm") == "0" and js.get("warm") == "none",
          str(js.get("warm")))
    end(tok_b, "other00001")
    end(tok, "other00001", "page_closed")
    mode(no_warm=1)                                            # бот без поддержки warm
    warm(tok, "nowarm0060")
    time.sleep(0.5)
    end(tok, "nowarm0060", "page_closed")
    time.sleep(0.4)
    st, h, b, t_6, _ = turn(tok, "nowarm0061", greet=True)
    e = evs("nowarm0060")
    check("бот без warm (warm → 400): звонок как раньше, end по «прогреву» не уходит (пустой беседы в базе не будет)",
          st == 200 and h.get("x-warm") == "0" and [x["ev"] for x in e] == ["warm"], f"{[x['ev'] for x in e]} · {t_6:.2f} с")
    end(tok, "nowarm0061")
    mode(start_ms=None, delay_ms=200, no_warm=0)

    # ---- тишина в звонке → сервер сам закрывает (end боту), без страницы
    call = "idlecall01"
    turn(tok, call, greet=True)
    time.sleep(9)
    e = evs(call)
    check("страница пропала (без end) → через CALL_IDLE_S сервер сам шлёт end (reason idle), линия свободна",
          e and e[-1]["ev"] == "end" and e[-1]["reason"] == "idle" and json.loads(req(url, "GET", "/call/health")[2]).get("busy") is False,
          str([(x["ev"], x.get("reason")) for x in e]))

    # ---- sendBeacon: токен в теле, без заголовка
    call = "beacon0001"
    turn(tok, call, greet=True)
    st, _, b, _, _ = req(url, "POST", "/call/end", None, json.dumps({"token": tok, "call": call, "reason": "page_closed"}).encode(),
                         {"Content-Type": "text/plain;charset=UTF-8"})
    time.sleep(0.4)
    check("закрытие страницы (sendBeacon, токен в теле) → end боту", st == 200 and evs(call)[-1]["reason"] == "page_closed", str(st))

    # ---- остановка сервиса посреди звонка
    if sb is not None:
        call = "stopcall01"
        turn(tok, call, greet=True)
        sb.procs["call"].send_signal(signal.SIGTERM)
        sb.procs["call"].wait(10)
        time.sleep(0.3)
        e = evs(call)
        check("systemctl stop посреди звонка → end боту (server_stop)", e and e[-1]["ev"] == "end" and e[-1]["reason"] == "server_stop", str(e[-1:]))
        sb.start_call()

    # ---- журнал и строгость контракта
    allev = events(bot_log)
    bad = [x for x in allev if x.get("errs")]
    check("все запросы к боту строго по контракту (поля, типы, lang, guest, stream только не для end)", not bad and len(allev) > 15
          and all(x.get("stream") is None for x in allev if x["ev"] == "end"), f"{len(allev)} событий, плохих {len(bad)}")
    if sb is not None:
        lg = (sb.d / "logs" / "call.log").read_text()
        check("журнал звонка: этапы есть, текстов реплик, имён и токенов нет", "turn 200 ok" in lg and "сколько стоит" not in lg
              and "Айбике" not in lg and tok not in lg and lab not in lg)


def lang_checks():
    import call_server as cs
    cases = [("Здравствуйте, сколько стоит?", "ru"), ("Сәлеметсіз бе, бағасы қанша?", "kk"), ("Иә", "kk"), ("Да", "ru"),
             ("Салем, как дела, можно записаться?", "ru"), ("ок", "kk"), ("Мен ертең келемін", "kk"), ("Подскажите адрес", "ru"),
             ("Ну, мен осы нөмірде", "kk"), ("бүгін можно?", "kk")]
    got = [(t, cs.lang_of(t, "kk")) for t, _ in cases]
    bad = [(t, g, w) for (t, g), (_, w) in zip(got, cases) if g != w]
    check("язык реплики по тексту ASR (kk/ru, ничья — язык прошлой реплики)", not bad, str(bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--url", help="проверить уже поднятый звонок (макет сеанса), а не свою песочницу")
    ap.add_argument("--token-file")
    ap.add_argument("--secret-file")
    ap.add_argument("--bot-log")
    ap.add_argument("--mode-url")
    a = ap.parse_args()
    lang_checks()
    if a.url:
        run_checks(a.url, Path(a.token_file).read_text().strip(), guestlink.load_secret(a.secret_file), a.bot_log, a.mode_url, None)
        print(f"\nзвонок с мозгом bot (макет сеанса): {'ВСЁ ПРОШЛО' if not fails else 'НЕ ПРОШЛО: ' + ', '.join(fails)}")
        return 1 if fails else 0
    sb = Sandbox(a.keep)
    try:
        run_checks(sb.url, (sb.d / "call_token").read_text().strip(), guestlink.load_secret(sb.d / "guest_secret"), sb.events,
                   f"http://127.0.0.1:{sb.gp}", sb)
    finally:
        sb.stop()
        print(f"песочница: {sb.d}" + ("" if a.keep else " (удалена; --keep — оставить)"))
    print(f"\nзвонок с мозгом bot (заглушки): {'ВСЁ ПРОШЛО' if not fails else 'НЕ ПРОШЛО: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
