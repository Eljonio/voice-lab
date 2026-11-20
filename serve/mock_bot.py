#!/usr/bin/env python3
"""mock_bot.py — заглушка внешнего бота диалога (мозг bot в call_server.py) по тому же HTTP-протоколу.

Протокол:
  POST /voice/turn  Bearer K; тело {tenant, call, event: start|turn|end, text, lang: ru|kk, guest: {phone,name}|null [, stream: true]}
    stream: true (не для end) → application/x-ndjson: {"type":"say","i","text"} по предложениям, {"type":"wait","what":"search"},
      последняя {"type":"done","reply","lang","end","ms","msTotal"[,"error"]}; сбой — {"type":"error","error":"turn_failed"}
    иначе JSON {"reply","lang","end","ms","msTotal"[,"error"]}; пустой turn → {"skipped":"empty"}; ход после end → 409; end после end → 200.
  GET  /voice/health Bearer K → {"ok":true}
  POST /mock/mode   (без токена, только для тестов) {"delay_ms","sent_gap_ms","fail": null|"500"|"close"|"hang"|"brain_down"|"stream_error",
                    "fail_n", "json_only": 0|1 (бот не умеет поток), "search_ms": задержка после wait,
                    "start_ms": сколько бот готовит приветствие (по умолчанию delay_ms), "prep_stale": 1 — беседа изменилась, start даёт другой текст}
  event warm (wait: true → после подготовки {"warm":"ready","prepared"}; без wait → {"warm":"started"}); start с тем же call —
    готовый текст сразу (ждёт остаток, если прогрев не кончился); end до звонка — прогрев забыт.
Ответы: start — приветствие (с именем гостя), «пока/сау бол» → end:true, «молчи» → пустой ответ, «курс/ищи» → wait + поиск.
Журнал событий (с текстом — это заглушка) — jsonl: проверка, что start/turn/end, guest и stream дошли и тело строго по контракту.
  python3 mock_bot.py --port P --key-file K --log events.jsonl [--delay-ms 300]
"""
import argparse
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODE = dict(delay_ms=300, sent_gap_ms=600, fail=None, fail_n=0, json_only=0, search_ms=2500, start_ms=None, prep_stale=0, no_warm=0)
LOCK = threading.Lock()
ENDED: set[str] = set()
PREP: dict[str, dict] = {}          # прогрев — call → {text, lang, phone, ready: Event}
A = None


def log_ev(rec):
    with LOCK, open(A.log, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(rec, t=round(time.time(), 3)), ensure_ascii=False) + "\n")


def check_body(b):
    errs = []
    keys = set(b)
    if not {"tenant", "call", "event", "text", "lang", "guest"} <= keys or keys - {"tenant", "call", "event", "text", "lang", "guest", "stream", "wait"}:
        errs.append(f"поля {sorted(b)}")
    if b.get("event") not in ("warm", "start", "turn", "end"):
        errs.append(f"event {b.get('event')!r}")
    if b.get("lang") not in ("kk", "ru"):
        errs.append(f"lang {b.get('lang')!r}")
    if not isinstance(b.get("text"), str) or not isinstance(b.get("call"), str) or not b.get("call"):
        errs.append("text/call")
    if "stream" in b and (b["stream"] is not True or b.get("event") in ("end", "warm")):
        errs.append(f"stream {b.get('stream')!r} при {b.get('event')}")
    if "wait" in b and (b["wait"] is not True or b.get("event") != "warm"):
        errs.append(f"wait {b.get('wait')!r} при {b.get('event')}")
    g = b.get("guest")
    if g is not None and (not isinstance(g, dict) or set(g) != {"phone", "name"} or not str(g.get("phone", "")).startswith("7")):
        errs.append(f"guest {g!r}")
    return errs


def answer(b):
    """→ (предложения, язык, end, ищет ли)."""
    ru = b["lang"] == "ru"
    if b["event"] in ("start", "warm"):
        name = (b.get("guest") or {}).get("name", "")
        if ru:
            return [f"Здравствуйте{', ' + name if name else ''}!", "Слушаю вас, чем могу помочь?"], "ru", False, False
        return [f"Сәлеметсіз бе{', ' + name if name else ''}!", "Тыңдап тұрмын, сізге қалай көмектесе аламын?"], "kk", False, False
    tl = b["text"].lower()
    if "молчи" in tl:
        return [], b["lang"], False, False
    if any(w in tl for w in ("пока", "до свидания", "сау бол")):
        return (["Спасибо за звонок!", "Всего доброго."], "ru", True, False) if ru else (["Қоңырауыңызға рахмет!", "Сау болыңыз."], "kk", True, False)
    if any(w in tl for w in ("курс", "ищи", "найди")):
        return ["По данным Нацбанка, один доллар стоит четыреста пятьдесят тенге."], "ru", False, True
    if ru:
        return ["Поняла вас.", "Записать вас на завтра на десять утра?"], "ru", False, False
    return ["Түсіндім.", "Ертең таңғы оныншы жазып қояйын ба?"], "kk", False, False


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, st, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(st)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _auth(self):
        h = self.headers.get("Authorization", "")
        return h[:7] == "Bearer " and hmac.compare_digest(h[7:].encode(), A.key.encode())

    def do_GET(self):
        if self.path == "/voice/health":
            return self._json(200 if self._auth() else 401, {"ok": self._auth()})
        return self._json(404, {"error": "not_found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if self.path == "/mock/mode":
            with LOCK:
                MODE.update(json.loads(raw or b"{}"))
            return self._json(200, dict(MODE))
        if self.path != "/voice/turn":
            return self._json(404, {"error": "not_found"})
        if not self._auth():
            return self._json(401, {"error": "unauthorized"})
        try:
            b = json.loads(raw.decode("utf-8"))
        except ValueError:
            return self._json(400, {"error": "bad_json"})
        errs = check_body(b)
        log_ev(dict(ev=b.get("event"), call=b.get("call"), text=b.get("text"), lang=b.get("lang"), guest=b.get("guest"), tenant=b.get("tenant"),
                    stream=b.get("stream"), wait=b.get("wait"), errs=errs, enc=self.headers.get("Accept-Encoding", ""), reason=self.headers.get("X-End-Reason")))
        if errs:
            return self._json(400, {"error": "bad_body", "detail": errs})
        t0 = time.time()
        call = b["call"]
        if call in ENDED:
            if b["event"] == "end":
                return self._json(200, {"reply": "", "lang": b["lang"], "end": True, "ms": 0, "msTotal": 0})
            return self._json(409, {"error": "звонок уже завершён"})
        if b["event"] == "end":
            ENDED.add(call)
            PREP.pop(call, None)
            return self._json(200, {"reply": "", "lang": b["lang"], "end": True, "ms": 1, "msTotal": 1})
        phone = (b.get("guest") or {}).get("phone")
        start_s = (MODE["start_ms"] if MODE["start_ms"] is not None else MODE["delay_ms"]) / 1000
        if b["event"] == "warm" and MODE["no_warm"]:      # бот без поддержки события warm
            return self._json(400, {"error": "event"})
        if b["event"] == "warm":
            pr = PREP.get(call)
            if not pr:
                sents, lang, _, _ = answer(b)
                pr = PREP[call] = dict(text=" ".join(sents), lang=lang, phone=phone, ready=threading.Event())
                threading.Timer(start_s, pr["ready"].set).start()          # готовит «мозг»
            if b.get("wait") is not True:
                return self._json(200, {"reply": "", "lang": b["lang"], "end": False, "ms": 0, "warm": "started", "msTotal": 0})
            pr["ready"].wait(60)
            return self._json(200, {"reply": "", "lang": pr["lang"], "end": False, "ms": round((time.time() - t0) * 1000), "warm": "ready",
                                    "prepared": pr["text"], "msTotal": round((time.time() - t0) * 1000)})
        if b["event"] == "start" and call in PREP:
            pr = PREP.pop(call)
            if pr["phone"] == phone and not MODE["prep_stale"]:
                pr["ready"].wait(60)                                       # кнопку нажали раньше — ждём остаток
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for o in ({"type": "say", "i": 0, "text": pr["text"]},
                          {"type": "done", "reply": pr["text"], "lang": pr["lang"], "end": False, "ms": 0, "warmed": True,
                           "msTotal": round((time.time() - t0) * 1000)}):
                    d = (json.dumps(o, ensure_ascii=False) + "\n").encode()
                    self.wfile.write(b"%X\r\n%s\r\n" % (len(d), d))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                return
        if b["event"] == "turn" and not b["text"].strip():
            return self._json(200, {"reply": "", "lang": b["lang"], "end": False, "ms": 0, "msTotal": 0, "skipped": "empty"})
        with LOCK:
            mode = dict(MODE)
            if mode["fail"] and mode["fail_n"] > 0:
                MODE["fail_n"] -= 1
            else:
                mode["fail"] = None
        if mode["fail"] == "500":
            return self._json(500, {"error": "turn_failed"})
        if mode["fail"] == "close":
            self.close_connection = True
            return
        if mode["fail"] == "hang":
            time.sleep(60)
            return
        sents, lang, end, search = answer(b)
        if b["event"] == "start":
            mode["delay_ms"] = start_s * 1000
            if MODE["prep_stale"]:
                sents = ["Снова здравствуйте!"] + sents[1:]
        if mode["fail"] == "brain_down":
            sents = []
        stream = b.get("stream") is True and not mode["json_only"]
        if not stream:
            time.sleep(mode["delay_ms"] / 1000 + (mode["search_ms"] / 1000 if search else 0))
            out = {"reply": " ".join(sents), "lang": lang, "end": end, "ms": round((time.time() - t0) * 1000)}
            if mode["fail"] == "brain_down":
                out["error"] = "brain_down"
            out["msTotal"] = out["ms"]
            return self._json(200, out)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(o):
            d = (json.dumps(o, ensure_ascii=False) + "\n").encode()
            self.wfile.write(b"%X\r\n%s\r\n" % (len(d), d))
            self.wfile.flush()
        time.sleep(mode["delay_ms"] / 1000)
        if mode["fail"] == "stream_error":
            chunk({"type": "error", "error": "turn_failed"})
            self.wfile.write(b"0\r\n\r\n")
            return
        if search:
            chunk({"type": "wait", "what": "search"})
            time.sleep(mode["search_ms"] / 1000)
        for i, s in enumerate(sents):
            if i:
                time.sleep(mode["sent_gap_ms"] / 1000)
            chunk({"type": "say", "i": i, "text": s})
        done = {"type": "done", "reply": " ".join(sents), "lang": lang, "end": end, "ms": round((time.time() - t0) * 1000)}
        if mode["fail"] == "brain_down":
            done["error"] = "brain_down"
        done["msTotal"] = done["ms"]
        chunk(done)
        self.wfile.write(b"0\r\n\r\n")


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--delay-ms", type=int, default=300)
    A = ap.parse_args()
    A.key = open(A.key_file).read().strip()
    MODE.update(delay_ms=A.delay_ms)
    srv = ThreadingHTTPServer(("127.0.0.1", A.port), H)
    srv.daemon_threads = True
    print(f"mock bot: http://127.0.0.1:{A.port}/voice/turn", flush=True)
    srv.serve_forever(poll_interval=0.3)


if __name__ == "__main__":
    main()
