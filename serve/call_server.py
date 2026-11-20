"""call_server.py — тестовый звонок: микрофон → распознавание → мозг → голос потоком.

Живёт на том же GPU-сервере, что tts_server.py (голос), отдельным процессом (свой venv: GigaAM — transformers 4.57 + hydra,
VoxCPM — свой venv; так их зависимости не мешают друг другу).

Страницы:
  GET  /, /c              страница для клиента (CALL_CLIENT_PAGE; в репозиторий не входит): личная ссылка https://<адрес>/#g=<токен гостя>
                          (guestlink.py)
  GET  /call              лабораторная страница (call.html): токен звонка в #k=…, переключатели, замеры, замечания
  GET  /assets/<файл>     картинки страниц (CALL_ASSETS; в репозиторий не входят)
API (Authorization: Bearer <токен звонка | токен гостя>):
  GET  /call/health       готовность: ASR, голос, мозг, занята ли линия (без токена)
  POST /call/turn         тело — PCM16 моно (X-Sample-Rate, по умолчанию 16000), до 20 с; в строке: call (id звонка, обязателен),
                          greet=1 (начало звонка: тела нет, бот здоровается сам), lang (подсказка kk|ru);
                          только для лабораторного токена — voice, voice_ru, steps, cfg, mode, sr, brain (bot|llm|script|echo),
                          brain_delay, guest_name, guest_phone (проверить историю гостя).
                          → 200 chunked PCM16; заголовки уходят с ПЕРВЫМ звуком (это может быть «Секунду…», если бот думает долго):
                          X-Request-Id, X-Sample-Rate, X-ASR-Text, X-Reply-Text (пусто, если ответ ещё не известен), X-Filler, X-End,
                          этапы X-Recv-Ms / X-ASR-Ms / X-Brain-Ms / X-TTS-Head-Ms / X-Server-Ms (−1 — неизвестно на момент заголовков).
                          204 — бот решил промолчать; 409 line_busy — идёт чужой звонок; 409 call_over — звонок уже закрыт;
                          422 no_speech — речи не слышно; 401 link_expired — срок ссылки гостя вышел.
  POST /call/warm?call=…  прогрев: страница клиента открылась — бот заранее готовит приветствие (event warm),
                          сервер звонка сразу синтезирует его в буфер; по кнопке start бот отдаёт тот же текст, и звук идёт из буфера
                          (без гудков). Кнопку нажали раньше — start ждёт остаток у бота, звук — хвост того же синтеза. Ответ сразу.
  POST /call/end          конец звонка (кнопка, закрытие страницы — sendBeacon с {"token","call","reason"} в теле) → бот получает end
  GET  /call/stats?rid=…  итог реплики (лаборатория — всё, гость — только end своей реплики)
  POST /call/guestlink    (только CALL_LAB_MINT=1) выпуск ссылки гостя со страницы — по умолчанию ВЫКЛЮЧЕН: номер гостя берётся только
                          из выдачи оператора (guestlink.py make), иначе держатель лабораторной ссылки назовёт чужой номер и услышит
                          чужую переписку
  POST /call/done         (лаборатория) «сеанс закончен» → файл CALL_DONE: сервер сворачивается
  POST /call/note         (лаборатория) замечание → logs/notes.jsonl

Мозг bot — внешний HTTP-бот диалога: POST BOT_TURN_URL, Bearer — BOT_VOICE_KEY (или файл BOT_VOICE_KEY_FILE);
  тело {"tenant","call","event":"start"|"turn"|"end","text","lang","guest":{"phone","name"}|null, "stream": true};
  stream → NDJSON: {"type":"say","i","text"} — предложение сразу в голос; {"type":"wait","what":"search"} — бот ищет в интернете
  (10–30 с) → наша фраза «Сейчас посмотрю, минутку.»; последняя {"type":"done","reply","lang","end","ms","msTotal","error"?};
  {"type":"error"} — сбой. Без stream — JSON {"reply","lang","end","ms","msTotal","error"?,"skipped"?}. error (brain_down|empty_reply)
  = reply пустой: что сказать на линии, решаем мы. lang первого события — подсказка языка для бота.
  Бот молчит дольше CALL_FILLER_MS (3 с; первая фраза у бота 1,3–2,5 с) — «Секунду.» (готовый звук из кэша); новой речи нет ещё
  CALL_FILLER2_MS — «Ещё секунду, уточняю.»; нет ответа BOT_TIMEOUT_S (55 с; поиск у бота до 30 с) или сбой — «Извините, связь
  прервалась, повторите» (второй сбой подряд — «сбой связи, позвоните позже» и конец звонка). На start заполнителей нет: страница
  клиента сама даёт гудок. Человек не слышит ни долгой тишины, ни технического текста.
  end уходит боту ровно один раз: кнопка/закрытие страницы, тишина дольше CALL_IDLE_S (сторож на сервере), остановка сервиса.
  Заглушка бота с тем же протоколом — mock_bot.py, проверки — call_test_bot.py.
Линия одна: пока идёт звонок, второй получает 409 line_busy (страница пишет «линия занята»).
Журнал — только время этапов и длины; тексты реплик в журнал не пишутся (лабораторная страница видит их через /call/stats).
"""
from __future__ import annotations

import collections
import hmac
import http.client
import json
import logging
import os
import queue
import re
import signal
import socket
import ssl
import sys
import tempfile
import threading
import time
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guestlink  # noqa: E402


def env(k, d):
    return os.environ.get(k, d)


class C:
    host = env("CALL_HOST", "0.0.0.0")
    port = int(env("CALL_PORT", "8443"))
    home = Path(env("CALL_HOME", env("TTS_HOME", str(Path.home() / "tts"))))
    token_file = Path(env("CALL_TOKEN_FILE", str(home / "call_token")))
    cert, key = env("CALL_TLS_CERT", ""), env("CALL_TLS_KEY", "")
    tts_url = env("TTS_URL", "http://127.0.0.1:8000")
    tts_url_ru = env("CALL_TTS_URL_RU", "")                      # второй голос — база VoxCPM2 без LoRA; голос «base:<id>»
    tts_token_file = Path(env("TTS_TOKEN_FILE", str(home / "token")))
    asr = env("CALL_ASR", "gigaam")
    brain = env("CALL_BRAIN", "script")
    llm_url, llm_model, llm_key_file = env("CALL_LLM_URL", ""), env("CALL_LLM_MODEL", ""), env("CALL_LLM_KEY_FILE", "")
    bot_url = env("BOT_TURN_URL", "")                            # внешний бот (мозг bot): URL /voice/turn; пусто — мозг bot выключен
    bot_key = env("BOT_VOICE_KEY", "")                           # ключ бота (Bearer); или файл BOT_VOICE_KEY_FILE
    bot_key_file = Path(env("BOT_VOICE_KEY_FILE", str(home / "bot_voice_key")))
    tenant = env("BOT_TENANT", "demo")
    bot_stream = env("BOT_STREAM", "1") == "1"                   # просить ndjson (по предложениям); бот может ответить обычным JSON
    bot_timeout = float(env("BOT_TIMEOUT_S", "55"))              # таймаут клиента не меньше 50 с (поиск в интернете 10–30 с)
    filler_ms = int(env("CALL_FILLER_MS", "3000"))               # 0 — без «Секунду.»; у бота первая фраза 1,3–2,5 с — раньше 3 с не мешаем
    filler2_ms = int(env("CALL_FILLER2_MS", "12000"))            # от последнего звука
    lab_guest_phone = env("CALL_LAB_GUEST_PHONE", "")             # гость лабораторных звонков — задаёт оператор при запуске, не страница
    lab_guest_name = env("CALL_LAB_GUEST_NAME", "")
    lab_mint = env("CALL_LAB_MINT", "0") == "1"                   # выпуск ссылок гостей со страницы лаборатории (по умолчанию нет)
    idle_s = float(env("CALL_IDLE_S", "90"))                      # тишина в звонке дольше — сервер сам закрывает звонок (end боту)
    voice_kk = env("CALL_VOICE_KK", "aruzhan_f")
    voice_ru = env("CALL_VOICE_RU", "aruzhan_f")                  # русский — тем же казахским эталоном; aruzhan_f_ru — русский
                                                                  # эталон на v5; base:aruzhan_f_ru — база VoxCPM2 с русским эталоном (нужен CALL_TTS_URL_RU)
    steps = env("CALL_STEPS", "")                                 # шаги голоса для гостей (пусто — как у голоса по умолчанию)
    sr_out = int(env("CALL_SR", "24000"))
    phrases_file = env("CALL_PHRASES_FILE", "")
    guest_secret_file = Path(env("CALL_GUEST_SECRET_FILE", str(home / "guest_secret")))
    public_base = env("CALL_PUBLIC_BASE", "")
    public_url_file = Path(env("CALL_PUBLIC_URL_FILE", str(home / "public_url")))
    keepalive = env("CALL_KEEPALIVE", "/run/gpu-keepalive")
    done_file = Path(env("CALL_DONE_FILE", str(home / "CALL_DONE")))
    log = Path(env("CALL_LOG", str(home / "logs" / "call.log")))
    page = Path(env("CALL_PAGE", str(HERE / "call.html")))
    client_page = Path(env("CALL_CLIENT_PAGE", str(HERE / "client.html")))
    assets = Path(env("CALL_ASSETS", str(HERE / "assets")))
    max_audio_s = float(env("CALL_MAX_AUDIO_S", "20"))
    plain_port = int(env("CALL_PLAIN_PORT", "0"))   # >0: ещё и простой HTTP на 127.0.0.1 — под ssh-туннель/cloudflared
    mock_asr_ms = float(env("CALL_MOCK_ASR_MS", "40"))


log = logging.getLogger("call")
KK_LET = set("әғқңөұүһі")
CALL_ID = re.compile(r"^[A-Za-z0-9_-]{4,40}$")


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


# ---------------------------------------------------------------- распознавание
class ASR:
    def __init__(self):
        self.kind, self.m, self.err = C.asr, None, None

    def load(self):
        t = time.time()
        if self.kind == "mock":
            log.info("ASR: заглушка (%d мс)", C.mock_asr_ms)
            return
        import gigaam_eval
        self.m = gigaam_eval.ASR()
        sil = np.zeros(16000, np.float32)
        self(sil)                                              # прогрев (CUDA, ядра)
        log.info("ASR: GigaAM-Multilingual CTC (%s) за %.0f с", gigaam_eval.REVISION[:8], time.time() - t)

    def __call__(self, x16: np.ndarray, mock_text: str | None = None) -> str:
        if self.kind == "mock":
            time.sleep(C.mock_asr_ms / 1000)
            return mock_text if mock_text is not None else ("Сәлеметсіз бе, бағасы қанша?" if len(x16) > 8000 else "")
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=C.home / "tmp") as f:
            with wave.open(f.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes((np.clip(x16, -1, 1) * 32767).astype("<i2").tobytes())
            text, _ = self.m(f.name)
        return text


# ---------------------------------------------------------------- язык
KK_WORDS = set("сәлем салем иә жоқ қанша рахмет керек қашан ертең бүгін мен сіз сен бұл осы қалай неше жақсы болады ма ме ба бе па пе "
               "ғой деп үшін және емес бар жоқпа иа ия келеді қайда кім не неге айтыңыз ыңғайлы".split())
RU_WORDS = set("что это как мне можно сколько пожалуйста здравствуйте привет спасибо да нет хорошо когда где почему если или есть будет "
               "надо нужно вы я мы он она не ну вот так вас нас меня тебя добрый день алло слушаю подскажите скажите".split())


def lang_of(t: str, fallback: str | None = "kk") -> str | None:
    """Язык реплики по распознанному тексту: казахские буквы и слова против русских слов; ничья — fallback (язык прошлой реплики)."""
    tl = (t or "").lower()
    words = re.findall(r"[а-яёәғқңөұүһі]+", tl)
    if not words:
        return fallback
    kk = sum(c in KK_LET for c in tl) + 2 * sum(w in KK_WORDS for w in words)
    ru = 2 * sum(w in RU_WORDS for w in words) + sum(c in "ёъщ" for c in tl)
    if kk > ru:
        return "kk"
    if ru > kk:
        return "ru"
    return fallback or ("ru" if re.search(r"[а-яё]", tl) else "kk")


# ---------------------------------------------------------------- фразы самого звонка (не бота): приветствие на случай сбоя, «секунду», сбой
SAY = {
    "greet": {"kk": ["Сәлеметсіз бе. Тыңдап тұрмын, сізге қалай көмектесе аламын?"],
              "ru": ["Здравствуйте! Слушаю вас, чем могу помочь?"]},
    "filler": {"kk": ["Бір сәт.", "Бір минут.", "Қазір, бір сәт."], "ru": ["Секунду.", "Минутку.", "Так, секунду."]},
    "wait": {"kk": ["Қазір қарап көрейін, бір минут."], "ru": ["Сейчас посмотрю, минутку."]},
    "filler2": {"kk": ["Тағы бір сәт, нақтылап жатырмын."], "ru": ["Ещё секунду, уточняю."]},
    "fail": {"kk": ["Кешіріңіз, байланыс үзіліп кетті. Қайталап жібересіз бе?"], "ru": ["Извините, связь прервалась. Повторите, пожалуйста?"]},
    "fail_many": {"kk": ["Кешіріңіз, қазір байланыста ақау бар. Сәл кейінірек қайта қоңырау шалыңызшы."],
                  "ru": ["Извините, у меня сейчас сбой связи. Попробуйте позвонить чуть позже."]},
}


def load_phrases():
    if C.phrases_file and Path(C.phrases_file).is_file():
        try:
            j = json.loads(Path(C.phrases_file).read_text(encoding="utf-8"))
            for k, v in j.items():
                if k in SAY and isinstance(v, dict):
                    for lg in ("kk", "ru"):
                        if isinstance(v.get(lg), list) and v[lg]:
                            SAY[k][lg] = [str(x) for x in v[lg]]
            log.info("фразы звонка: из %s", C.phrases_file)
        except (OSError, ValueError) as e:
            log.info("фразы звонка: %s не прочитан (%s) — встроенные", C.phrases_file, type(e).__name__)


# ---------------------------------------------------------------- старые мозги лаборатории (llm / правила / эхо)
RULES = [
    (r"баға|қанша|сколько|цен|стоит|стоимост",
     "Нақты бағасын шебер келіп көргеннен кейін айтады. Сізге қай күні ыңғайлы?",
     "Точную стоимость мастер скажет на месте. Когда вам удобно?"),
    (r"қашан|уақыт|ертең|бүгін|когда|завтра|сегодня|время",
     "Жақсы, ертеңге жазып қойдым. Мекенжайыңызды айтып жібересіз бе?",
     "Хорошо, записала на завтра. Подскажите адрес?"),
    (r"көше|мекенжай|үй|пәтер|улиц|адрес|дом|квартир",
     "Рахмет, жазып алдым. Шебер барар алдында сізге қоңырау шалады.",
     "Спасибо, записала. Мастер позвонит перед выездом."),
    (r"рахмет|сау бол|спасибо|до свидан|пока",
     "Сізге де рахмет! Сау болыңыз.",
     "Спасибо вам! Всего доброго."),
    (r"сәлем|салем|здравств|привет|добрый",
     "Сәлеметсіз бе. Тыңдап тұрмын, айта беріңіз.",
     "Здравствуйте! Слушаю вас."),
    (r"\bиә\b|\bжоқ\b|\bда\b|\bнет\b",
     "Түсіндім. Тағы қандай сұрағыңыз бар?",
     "Поняла. Что-то ещё подсказать?"),
]
FALLBACK = ("Түсіндім. Толығырақ айтып берсеңіз?", "Поняла. Расскажите чуть подробнее?")


class Brain:
    def __init__(self):
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=12))
        self.llm_key = ""
        if C.llm_key_file and Path(C.llm_key_file).is_file():
            self.llm_key = Path(C.llm_key_file).read_text().strip()

    def reply(self, kind: str, text: str, call_id: str, lang: str) -> tuple[str, str, str]:
        """→ (ответ, язык, какой мозг ответил)"""
        i = 0 if lang == "kk" else 1
        if kind == "echo":
            t = text.strip().rstrip(".!?…")
            return (f"Сіз айттыңыз: {t}." if lang == "kk" else f"Вы сказали: {t}."), lang, "echo"
        if kind == "llm" and C.llm_url and C.llm_model:
            try:
                return self._llm(text, call_id), lang, "llm"
            except Exception as e:  # noqa: BLE001 — тест: падение мозга не роняет звонок
                log.info("llm ошибка %s — отвечаю правилами", type(e).__name__)
        tl = text.lower()
        for pat, kk, ru in RULES:
            if re.search(pat, tl):
                return (kk, ru)[i], lang, "script"
        return FALLBACK[i], lang, "script"

    def _llm(self, text, call_id):
        sys_p = ("Ты администратор сервисной компании, говоришь по телефону. Отвечай одной-двумя короткими фразами, "
                 "разговорно, без списков и цифр (числа словами). Отвечай на языке клиента: казахский или русский.")
        h = self.hist[call_id]
        msgs = [{"role": "system", "content": sys_p}, *h, {"role": "user", "content": text}]
        u = urlsplit(C.llm_url)
        c = (http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection)(u.hostname, u.port, timeout=10)
        hd = {"Content-Type": "application/json"}
        if self.llm_key:
            hd["Authorization"] = "Bearer " + self.llm_key
        c.request("POST", (u.path.rstrip("/") or "") + "/v1/chat/completions",
                  body=json.dumps(dict(model=C.llm_model, messages=msgs, max_tokens=120, temperature=0.3)).encode(), headers=hd)
        r = c.getresponse()
        b = json.loads(r.read())
        c.close()
        if r.status != 200:
            raise RuntimeError(f"llm {r.status}")
        out = b["choices"][0]["message"]["content"].strip()
        h.append({"role": "user", "content": text})
        h.append({"role": "assistant", "content": out})
        return out


# ---------------------------------------------------------------- мозг = внешний бот по HTTP
class BotBrain:
    def __init__(self):
        self.key = C.bot_key.strip()
        if not self.key and C.bot_key_file.is_file():
            self.key = C.bot_key_file.read_text().strip()

    def ok(self) -> bool:
        return bool(C.bot_url and self.key)

    def _conn(self, timeout):
        u = urlsplit(C.bot_url)
        if u.scheme == "https":
            return http.client.HTTPSConnection(u.hostname, u.port, timeout=timeout), u
        return http.client.HTTPConnection(u.hostname, u.port, timeout=timeout), u

    def body(self, event, call, text, lang, guest, stream=False) -> bytes:
        b = {"tenant": C.tenant, "call": call, "event": event, "text": text or "", "lang": lang if lang in ("kk", "ru") else "kk",
             "guest": guest or None}
        if stream and event != "end":
            b["stream"] = True
        return json.dumps(b, ensure_ascii=False).encode()

    def request(self, event, call, text, lang, guest, rid) -> queue.Queue:
        """start/turn: ответ кусками в очередь — ("say", текст, язык) … ("final", ответ, язык, end, ms) | ("error", код)."""
        q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, args=(q, self.body(event, call, text, lang, guest, C.bot_stream), rid), daemon=True).start()
        return q

    def _run(self, q, body, rid):
        conn = None
        try:
            if not self.ok():
                q.put(("error", "no_key"))
                return
            conn, u = self._conn(C.bot_timeout)
            hd = {"Authorization": "Bearer " + self.key, "Content-Type": "application/json; charset=utf-8", "X-Request-Id": rid,
                  "Accept": "application/x-ndjson, application/json;q=0.9" if C.bot_stream else "application/json",
                  "Accept-Encoding": "identity",          # без сжатия: Cloudflare не копит поток ради gzip
                  "User-Agent": "voice-call/2"}
            conn.request("POST", u.path + (("?" + u.query) if u.query else ""), body=body, headers=hd)
            r = conn.getresponse()
            if r.status != 200:
                r.read()
                q.put(("error", f"http_{r.status}"))
                return
            if "ndjson" in (r.getheader("Content-Type") or ""):
                said = []
                while True:
                    line = r.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                    except ValueError:
                        continue
                    typ = o.get("type")
                    s = str(o.get("text") if typ == "say" else (o.get("say") or "")).strip()
                    if s:
                        said.append(s)
                        q.put(("say", s, o.get("lang")))
                    elif typ == "wait":
                        q.put(("wait", str(o.get("what") or "")))
                    elif typ == "error":
                        q.put(("error", "bot_" + str(o.get("error") or "error")))
                        return
                    if typ == "done" or o.get("done") is True:
                        q.put(("final", str(o.get("reply") or " ".join(said)), o.get("lang"), bool(o.get("end")), o.get("ms"), o.get("error")))
                        return
                q.put(("final", " ".join(said), None, False, None, None if said else "no_done"))   # поток кончился без последней строки
            else:
                b = json.loads(r.read() or b"{}")
                q.put(("final", str(b.get("reply") or ""), b.get("lang"), bool(b.get("end")), b.get("ms"),
                       b.get("error") or None))
        except (socket.timeout, TimeoutError):
            q.put(("error", "timeout"))
        except Exception as e:  # noqa: BLE001 — сбой мозга не роняет звонок
            q.put(("error", type(e).__name__))
        finally:
            if conn is not None:
                conn.close()

    def warm(self, call, lang, guest, rid) -> dict:
        """event=warm, wait=true: бот готовит приветствие заранее → {"warm": "ready", "prepared": текст} (или started/skipped/failed)."""
        conn = None
        try:
            if not self.ok():
                return {"warm": "no_key"}
            conn, u = self._conn(60)
            b = json.loads(self.body("warm", call, "", lang, guest))
            b["wait"] = True
            conn.request("POST", u.path + (("?" + u.query) if u.query else ""), body=json.dumps(b, ensure_ascii=False).encode(),
                         headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json; charset=utf-8",
                                  "Accept": "application/json", "Accept-Encoding": "identity", "X-Request-Id": rid, "User-Agent": "voice-call/2"})
            r = conn.getresponse()
            raw = r.read()
            if r.status != 200:
                return {"warm": f"http_{r.status}"}
            return json.loads(raw or b"{}")
        except Exception as e:  # noqa: BLE001
            return {"warm": type(e).__name__}
        finally:
            if conn is not None:
                conn.close()

    def end(self, call, lang, guest, reason) -> str:
        """event=end: ответа голосом нет, бот по итогам делает свою работу. → «200» / код ошибки (для журнала)."""
        conn = None
        try:
            if not self.ok():
                return "no_key"
            conn, u = self._conn(10)
            conn.request("POST", u.path + (("?" + u.query) if u.query else ""), body=self.body("end", call, "", lang, guest),
                         headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json; charset=utf-8",
                                  "Accept": "application/json", "X-End-Reason": reason, "User-Agent": "voice-call/2"})
            r = conn.getresponse()
            r.read()
            return str(r.status)
        except Exception as e:  # noqa: BLE001
            return type(e).__name__
        finally:
            if conn is not None:
                conn.close()


# ---------------------------------------------------------------- линия: один звонок за раз
def loose(t: str) -> str:
    """Ключ сравнения текста приветствия: только буквы/цифры (бот отдаёт prepared «как есть», а в start — после speakable)."""
    return re.sub(r"[^0-9a-zа-яёәғқңөұүһі]+", "", (t or "").lower())


class Warm:
    """Прогрев одного звонка: текст приветствия от бота и его звук, синтезируемый потоком (читатель может идти следом)."""

    def __init__(self, call, who, gid, lang, guest):
        self.call, self.who, self.gid, self.lang, self.guest = call, who, gid, lang, guest
        self.t0, self.t_ready, self.state = time.time(), None, "warming"       # warming → synth → ready | failed
        self.text, self.key, self.voice = "", "", ""
        self.chunks: list[bytes] = []
        self.done, self.err = False, None
        self.closed = ""                     # страницу закрыли, пока бот готовил: end уйдёт, как только бот подтвердит прогрев
        self.cv = threading.Condition()

    def push(self, b: bytes):
        with self.cv:
            self.chunks.append(b)
            self.cv.notify_all()

    def finish(self, err=None):
        with self.cv:
            self.done, self.err = True, err
            self.cv.notify_all()

    def stream(self, timeout=30.0):
        """Куски звука по мере синтеза (уже готовые — сразу); синтез встал дольше timeout — TimeoutError."""
        i = 0
        while True:
            with self.cv:
                if i >= len(self.chunks) and not self.done and not self.cv.wait(timeout):
                    raise TimeoutError("прогрев: синтез встал")
                if i >= len(self.chunks):
                    if self.done:
                        return
                    continue
                part = self.chunks[i:]
            yield from part
            i += len(part)


class Line:
    def __init__(self):
        self.m = threading.Lock()
        self.cur: dict | None = None

    def start(self, call, who, guest, lang, brain) -> tuple[dict, dict | None]:
        """→ (звонок, вытесненный зависший звонок или None). Чужой живой звонок → 409 line_busy."""
        now, old = time.time(), None
        with self.m:
            c = self.cur
            if c and c["call"] == call:
                c["t_last"] = now
                return c, None
            if c and (now - c["t_last"] < C.idle_s or c.get("in_turn")):
                raise Bad(409, "line_busy", "идёт другой звонок")
            old = c
            self.cur = dict(call=call, who=who, guest=guest, lang=lang, brain=brain, t0=now, t_last=now, turns=0, fails=0,
                            fill_i=0, in_turn=False, bot=False)
            return self.cur, old

    def get(self, call) -> dict:
        with self.m:
            if self.cur and self.cur["call"] == call:
                return self.cur
        raise Bad(409, "call_over", "звонок уже закончен")

    def take(self, call=None) -> dict | None:
        """Снять звонок с линии (call=None — любой текущий)."""
        with self.m:
            c = self.cur
            if c and (call is None or c["call"] == call):
                self.cur = None
                return c
        return None

    def idle(self) -> dict | None:
        with self.m:
            c = self.cur
            if c and not c.get("in_turn") and time.time() - c["t_last"] > C.idle_s:
                self.cur = None
                return c
        return None

    def busy(self) -> bool:
        with self.m:
            c = self.cur
            return bool(c and (time.time() - c["t_last"] < C.idle_s or c.get("in_turn")))


# ---------------------------------------------------------------- приложение
class App:
    def __init__(self):
        self.token = C.token_file.read_text().strip()
        self.tts_token = C.tts_token_file.read_text().strip()
        if len(self.token) < 20 or len(self.tts_token) < 20:
            raise SystemExit("токен короче 20 знаков")
        self.guest_secret = b""
        if C.guest_secret_file.is_file():
            try:
                self.guest_secret = guestlink.load_secret(C.guest_secret_file)
            except ValueError as e:
                log.info("секрет гостей не годится: %s — ссылки гостей выключены", e)
        self.asr, self.brain, self.bot, self.line = ASR(), Brain(), BotBrain(), Line()
        self.ready, self.error = False, None
        self.lock = threading.Lock()
        self.stats = collections.OrderedDict()
        self.m = threading.Lock()
        self.n_turns = 0
        self.clips: dict[tuple, bytes] = {}
        self.clips_m = threading.Lock()
        self.warms: dict[str, Warm] = {}
        self.warms_m = threading.Lock()
        (C.home / "tmp").mkdir(parents=True, exist_ok=True)

    def start(self):
        try:
            load_phrases()
            self.asr.load()
            self.wait_tts()
            self.warm_clips()
            self.ready = True
            threading.Thread(target=self.warm_clips, kwargs=dict(extra=True), daemon=True).start()
            log.info("ГОТОВ: %s://%s:%d, ASR %s, мозг %s%s, голос %s, гости %s", "https" if C.cert else "http", C.host, C.port, C.asr, C.brain,
                     (" (" + C.bot_url + (", ключ есть" if self.bot.ok() else ", КЛЮЧА НЕТ") + ")") if C.brain == "bot" else "",
                     C.tts_url, "да" if self.guest_secret else "нет")
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {str(e)[:300]}"
            log.error("СТАРТ НЕ УДАЛСЯ: %s", self.error)
        threading.Thread(target=self.reaper, daemon=True).start()

    def wait_tts(self, limit=900):
        """Голос грузится дольше ASR: «готов» только вместе с голосом (иначе первые «Секунду.» синтезируются на лету)."""
        t = time.time()
        while time.time() - t < limit:
            try:
                st, th = self.tts_get("/health")
                if st == 200 and th.get("ready") is True:
                    if time.time() - t > 1:
                        log.info("голос готов через %.0f с ожидания", time.time() - t)
                    return
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise RuntimeError(f"голос не готов за {limit} с")

    def tts_ru_ready(self) -> bool:
        if not C.tts_url_ru:
            return False
        try:
            st, th = self.tts_get("/health", C.tts_url_ru)
            return st == 200 and th.get("ready") is True
        except (OSError, ValueError):
            return False

    # ---- готовый звук коротких фраз («Секунду.», сбой, запасное приветствие): без ожидания синтеза
    def warm_clips(self, extra: bool = False):
        """Готовый звук коротких фраз. До «готов» — голоса гостей (kk и C.voice_ru); extra=True (в фоне после) — остальные русские
        голоса лаборатории: v5 + казахский/русский эталон и база (когда второй голос поднимется) — переключение без задержки."""
        t, n = time.time(), 0
        if extra:
            for _ in range(60):                          # второй голос грузится в фоне — ждём до 5 мин
                if not C.tts_url_ru or self.tts_ru_ready():
                    break
                time.sleep(5)
            voices = {"ru": [v for v in ("aruzhan_f", "aruzhan_f_ru") + (("base:aruzhan_f_ru",) if self.tts_ru_ready() else ())
                             if v != voice_for("ru", {})]}
        else:
            voices = {"kk": [voice_for("kk", {})], "ru": [voice_for("ru", {})]}
        for lg, vs in voices.items():
            for v in vs:
                for key in ("filler", "wait", "filler2", "fail", "fail_many", "greet"):
                    for txt in SAY[key][lg]:
                        if time.time() - t > 150:
                            log.info("кэш фраз: не успел за 150 с (%d готово) — остальные синтезируются по ходу", n)
                            return
                        try:
                            if not self.lock.locked():   # идёт реплика — не мешаем голосу, фраза синтезируется по ходу
                                self.clip(txt, v, C.sr_out)
                                n += 1
                        except Exception as e:  # noqa: BLE001
                            log.info("кэш фраз: %s/%s — %s", v, key, type(e).__name__)
        log.info("кэш фраз%s: %d за %.1f с (%s)", " (лаборатория)" if extra else "", n, time.time() - t, voices)

    def clip(self, text, voice, sr) -> bytes:
        k = (text, voice, sr)
        with self.clips_m:
            if k in self.clips:
                return self.clips[k]
        url, vid = tts_route(voice)
        conn = http.client.HTTPConnection(*hostport(url), timeout=60)
        try:
            conn.request("POST", "/tts/stream", body=json.dumps(dict(text=text, voice=vid, mode="chain", sr=sr), ensure_ascii=False).encode(),
                         headers={"Authorization": "Bearer " + self.tts_token, "Content-Type": "application/json"})
            r = conn.getresponse()
            b = r.read()
            if r.status != 200:
                raise RuntimeError(f"tts {r.status}")
        finally:
            conn.close()
        with self.clips_m:
            self.clips[k] = b
        return b

    def tts_get(self, path, url=None):
        c = http.client.HTTPConnection(*hostport(url or C.tts_url), timeout=10)
        c.request("GET", path, headers={"Authorization": "Bearer " + self.tts_token})
        r = c.getresponse()
        b = r.read()
        c.close()
        return r.status, (json.loads(b) if b else {})

    # ---- конец звонка: end боту ровно один раз
    def end_call(self, c: dict | None, reason: str, sync: bool = False):
        if not c or c.get("end_sent"):
            return
        c["end_sent"] = True
        dur = round(time.time() - c["t0"])

        def go():
            res = self.bot.end(c["call"], c.get("lang"), c.get("guest"), reason) if c.get("bot") else "-"
            log.info("звонок закончен: %s, %s, %d с, реплик %d, end боту: %s", reason, c["who"], dur, c["turns"], res)
        if sync:
            go()
        else:
            threading.Thread(target=go, daemon=True).start()

    def reaper(self):
        while True:
            time.sleep(3)
            c = self.line.idle()
            if c:
                self.end_call(c, "idle")
            with self.warms_m:                      # бот забывает прогрев через 3 мин — и мы
                for k in [k for k, w in self.warms.items() if time.time() - w.t0 > 200]:
                    del self.warms[k]

    # ---- прогрев: страница открылась — приветствие заранее
    def warm(self, call, who, gid, lang, guest, rid) -> Warm:
        with self.warms_m:
            w = self.warms.get(call)
            if w and w.who == who and w.gid == gid and time.time() - w.t0 < 170:
                return w
            w = Warm(call, who, gid, lang, guest)
            self.warms[call] = w
            while len(self.warms) > 20:
                self.warms.pop(next(iter(self.warms)))
        threading.Thread(target=self._warm_run, args=(w, rid), daemon=True).start()
        return w

    def _warm_run(self, w: Warm, rid: str):
        t = time.time()
        r = self.bot.warm(w.call, w.lang, w.guest, rid)
        w.text = str(r.get("prepared") or "").strip()
        if r.get("warm") != "ready" or not w.text:
            w.state = "failed"
            w.finish(f"bot_{r.get('warm')}")
            log.info("прогрев %s: бот не подготовил (%s) за %.1f с", w.who, r.get("warm"), time.time() - t)
            return
        w.key = loose(w.text)
        if w.closed:
            log.info("прогрев без звонка закрыт (%s, после подготовки): end боту %s", w.closed, self.bot.end(w.call, w.lang, w.guest, w.closed))
            w.finish("closed")
            return
        if self.line.busy() and (self.line.cur or {}).get("call") != w.call:   # идёт чужой звонок: голос ему нужнее, а этот гость
            w.state = "deferred"                                                # всё равно получит «линия занята»; текст бота пригодится
            w.finish("busy")
            log.info("прогрев %s: приветствие у бота за %.1f с, звук не синтезирую — линия занята", w.who, time.time() - t)
            return
        lg = r.get("lang") if r.get("lang") in ("kk", "ru") else lang_of(w.text, w.lang)
        w.voice, w.state, w.t_ready = voice_for(lg, {}), "synth", time.time()
        url, vid = tts_route(w.voice)
        conn = None
        try:
            conn = http.client.HTTPConnection(*hostport(url), timeout=60)
            conn.request("POST", "/tts/stream", body=json.dumps(dict(text=w.text, voice=vid, mode="chain", sr=C.sr_out), ensure_ascii=False).encode(),
                         headers={"Authorization": "Bearer " + self.tts_token, "Content-Type": "application/json", "X-Request-Id": rid + "w"})
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f"tts {resp.status}")
            while True:
                b = resp.read1(65536)
                if not b:
                    break
                w.push(b)
            w.state = "ready"
            w.finish()
            log.info("прогрев %s: приветствие у бота %.1f с, звук готов ещё через %.1f с (%d КБ)", w.who, w.t_ready - t, time.time() - w.t_ready,
                     sum(map(len, w.chunks)) // 1024)
        except Exception as e:  # noqa: BLE001
            w.state = "failed"
            w.finish(type(e).__name__)
            log.info("прогрев: синтез не удался (%s)", type(e).__name__)
        finally:
            if conn is not None:
                conn.close()

    def take_warm(self, call, who, gid) -> Warm | None:
        with self.warms_m:
            w = self.warms.get(call)
            if w and w.who == who and w.gid == gid:
                del self.warms[call]
                return w
        return None


def hostport(url):
    u = urlsplit(url)
    return u.hostname, u.port


_RU_OK = {"t": 0.0, "ok": False}


def tts_route(voice: str) -> tuple[str, str]:
    """«base:aruzhan_f_ru» → (второй голос, «aruzhan_f_ru»); второй не поднят/не готов — первый с тем же эталоном (v5), без сбоя."""
    if voice.startswith("base:"):
        if C.tts_url_ru and APP is not None:
            if time.time() - _RU_OK["t"] > 10:
                _RU_OK.update(t=time.time(), ok=APP.tts_ru_ready())
            if _RU_OK["ok"]:
                return C.tts_url_ru, voice[5:]
        return C.tts_url, voice[5:]
    return C.tts_url, voice


def voice_for(lang: str, ov: dict) -> str:
    if lang == "ru":
        return ov.get("voice_ru") or C.voice_ru
    return ov.get("voice") or C.voice_kk


APP: App | None = None


class Bad(Exception):
    def __init__(self, status, code, detail=""):
        super().__init__(code)
        self.status, self.code, self.detail = status, code, detail


class Out:
    """Ответ одной реплики: chunked PCM16; заголовки уходят с первым звуком. Несколько кусков подряд (заполнитель, предложения бота)."""

    def __init__(self, h: "Handler", st: dict, base: dict, sr: int, t0: float):
        self.h, self.st, self.base, self.sr, self.t0 = h, st, base, sr, t0
        self.sent, self.nb, self.t_first, self.t_last = False, 0, None, None

    def head(self, extra: dict):
        if self.sent:
            return
        h = self.h
        h.protocol_version = "HTTP/1.1"
        h.send_response(200)
        hd = dict(self.base)
        hd.update(extra)
        hd["X-Server-Ms"] = round((time.time() - self.t0) * 1000)       # от приёма запроса до первого звука
        hd.update({"Content-Type": "application/octet-stream", "Cache-Control": "no-store", "Connection": "close",
                   "Transfer-Encoding": "chunked", "X-Sample-Rate": self.sr, "X-Encoding": "pcm16"})
        for k, v in hd.items():
            h.send_header(k, str(v))
        h.end_headers()
        self.sent, self.t_first = True, time.time()
        try:
            h.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def write(self, b: bytes, extra: dict):
        if not b:
            return
        self.head(extra)
        self.h._write(b)
        self.nb += len(b)
        self.t_last = time.time()

    def pause(self, ms: int):
        if self.sent:
            self.h._write(bytes(int(self.sr * ms / 1000) * 2))

    def clip(self, text: str, voice: str, extra: dict):
        self.write(APP.clip(text, voice, self.sr), extra)

    def tts(self, text: str, voice: str, params: dict, extra: dict, rid: str) -> dict:
        """Синтез потоком и сразу клиенту. → итог куска (rid голоса, время до первого звука голоса)."""
        url, vid = tts_route(voice)
        body = dict(text=text, voice=vid, mode=params.get("mode", "chain"), sr=self.sr)
        for k in ("steps", "cfg", "seed"):
            if params.get(k) is not None:
                body[k] = params[k]
        t0 = time.time()
        conn = http.client.HTTPConnection(*hostport(url), timeout=60)
        try:
            conn.request("POST", "/tts/stream", body=json.dumps(body, ensure_ascii=False).encode(),
                         headers={"Authorization": "Bearer " + APP.tts_token, "Content-Type": "application/json", "X-Request-Id": rid})
            r = conn.getresponse()
            t_head = time.time()
            if r.status != 200:
                b = r.read()
                if self.sent:
                    log.info("голос %d посреди ответа — обрываю реплику", r.status)
                    return dict(error=r.status)
                raise Bad(r.status if r.status in (400, 503, 504) else 502,
                          "tts_" + ((json.loads(b).get("error") or "error") if b else "error"), "голос не ответил")
            info = dict(tts_rid=r.getheader("X-Request-Id"), tts_base=url != C.tts_url, tts_head_ms=round((t_head - t0) * 1000), tts_ttfa_ms=int(r.getheader("X-TTFA-Ms") or -1),
                        steps=r.getheader("X-Steps"), cfg=r.getheader("X-CFG"), voice=r.getheader("X-Voice"))
            ex = dict(extra)
            ex.update({"X-TTS-Head-Ms": info["tts_head_ms"], "X-TTS-TTFA-Ms": info["tts_ttfa_ms"], "X-Steps": info["steps"], "X-CFG": info["cfg"],
                       "X-Voice": info["voice"]})
            while True:
                b = r.read1(65536)
                if not b:
                    break
                self.write(b, ex)
            return info
        finally:
            conn.close()

    def close(self):
        if self.sent:
            self.h.wfile.write(b"0\r\n\r\n")


class Handler(BaseHTTPRequestHandler):
    server_version = "voice-call/2.0"
    sys_version = ""
    timeout = 30

    def log_message(self, *a):
        pass

    def _send(self, status, body: bytes, ctype, extra=None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
            pass

    def _json(self, status, obj, extra=None):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", extra)

    def _who(self, token: str | None = None):
        """→ ("lab", None) | ("guest", данные ссылки); нет — Bad 401."""
        if token is None:
            h = self.headers.get("Authorization", "")
            token = h[7:].strip() if h[:7].lower() == "bearer " else ""
        if token and hmac.compare_digest(token.encode(), APP.token.encode()):
            return "lab", None
        if token and APP.guest_secret:
            pay, err = guestlink.verify(APP.guest_secret, token)
            if pay:
                return "guest", pay
            if err == "expired":
                raise Bad(401, "link_expired", "срок ссылки вышел")
        raise Bad(401, "unauthorized")

    def _page(self, path: Path):
        if not path.is_file():                     # страница клиента в репозиторий не входит (CALL_CLIENT_PAGE — своя)
            return self._json(404, dict(error="not_found", detail="страницы нет"))
        csp = ("default-src 'self'; script-src 'self' 'unsafe-inline' blob:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
               "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; media-src 'self' blob:; worker-src 'self' blob:; "
               "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
        return self._send(200, path.read_bytes(), "text/html; charset=utf-8",
                          {"Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff", "Content-Security-Policy": csp,
                           "Permissions-Policy": "microphone=(self), camera=(), geolocation=()"})

    def _body_json(self, limit=20000) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > limit:
            raise Bad(400, "bad_body")
        try:
            j = json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise Bad(400, "bad_json") from None
        if not isinstance(j, dict):
            raise Bad(400, "bad_json")
        return j

    def do_GET(self):
        u = urlsplit(self.path)
        try:
            if u.path in ("/", "/c", "/c/"):
                return self._page(C.client_page)
            if u.path in ("/call", "/call/"):
                return self._page(C.page)
            if u.path.startswith("/assets/"):
                name = u.path[len("/assets/"):]
                f = C.assets / name
                if re.fullmatch(r"[a-z0-9_-]+\.(png|svg)", name) and f.is_file():
                    return self._send(200, f.read_bytes(), "image/png" if name.endswith(".png") else "image/svg+xml",
                                      {"Cache-Control": "public, max-age=86400"})
                return self._json(404, dict(error="not_found"))
            if u.path == "/favicon.ico":
                f = C.assets / "favicon.png"
                return self._send(200, f.read_bytes(), "image/png") if f.is_file() else self._json(404, dict(error="not_found"))
            if u.path == "/call/health":
                try:
                    st, th = APP.tts_get("/health")
                except OSError as e:
                    st, th = 0, {"error": type(e).__name__}
                return self._json(200, dict(ok=True, ready=APP.ready, error=APP.error, asr=C.asr, brain=C.brain, llm=bool(C.llm_url),
                                            bot=APP.bot.ok() if C.brain == "bot" else None, guests=bool(APP.guest_secret),
                                            busy=APP.line.busy(), clips=len(APP.clips), lab_guest=bool(guestlink.norm_phone(C.lab_guest_phone)),
                                            tts_ru=APP.tts_ru_ready(), voice_ru=C.voice_ru,
                                            tts=dict(status=st, ready=th.get("ready"), model=th.get("model"), voices=th.get("voices"),
                                                     defaults=th.get("defaults"), stream=th.get("stream"))))
            if u.path == "/call/stats":
                who, pay = self._who()
                rid = (parse_qs(u.query).get("rid") or [""])[0]
                with APP.m:
                    x = APP.stats.get(rid)
                if not x:
                    return self._json(404, dict(error="not_found"))
                if who == "lab":
                    return self._json(200, x)
                if x.get("gid") != pay.get("i"):
                    return self._json(404, dict(error="not_found"))
                return self._json(200, dict(status=x.get("status"), end=bool(x.get("end"))))
            return self._json(404, dict(error="not_found"))
        except Bad as e:
            return self._json(e.status, dict(error=e.code, detail=e.detail))

    def do_POST(self):
        u = urlsplit(self.path)
        try:
            if u.path == "/call/turn":
                return self._turn(parse_qs(u.query))
            if u.path == "/call/end":
                return self._end()
            if u.path == "/call/warm":
                return self._warm(parse_qs(u.query))
            who, pay = self._who()
            if who != "lab":
                raise Bad(403, "forbidden")
            if u.path == "/call/done":
                C.done_file.write_text(time.strftime("%F %T") + "\n")
                log.info("сеанс закончен со страницы")
                return self._json(200, dict(ok=True))
            if u.path == "/call/note":     # замечания во время звонка → logs/notes.jsonl (забирается с результатами)
                rec = self._body_json()
                rec = dict(rec, t=time.strftime("%F %T"))
                with open(C.log.parent / "notes.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log.info("замечание к реплике %s", rec.get("turn"))
                return self._json(200, dict(ok=True))
            if u.path == "/call/guestlink" and C.lab_mint:
                return self._guestlink()
            return self._json(404, dict(error="not_found"))
        except Bad as e:
            return self._json(e.status, dict(error=e.code, detail=e.detail))

    def _base(self) -> str:
        if C.public_base:
            return C.public_base
        try:
            s = C.public_url_file.read_text().strip()
            if s.startswith("https://"):
                return s
        except OSError:
            pass
        host = self.headers.get("X-Forwarded-Host") or self.headers.get("Host") or f"127.0.0.1:{C.port}"
        local = host.split(":")[0] in ("127.0.0.1", "localhost")
        proto = "https" if (self.headers.get("X-Forwarded-Proto") == "https" or not local or C.cert) else "http"
        return f"{proto}://{host}"

    def _guestlink(self):
        if not APP.guest_secret:
            raise Bad(503, "no_guest_secret", "на сервере нет guest_secret")
        j = self._body_json()
        try:
            hours = float(j.get("hours") or 72)
            tok = guestlink.make(APP.guest_secret, str(j.get("name") or ""), str(j.get("phone") or ""), max(0.1, min(hours, 24 * 14)),
                                 str(j.get("lang") or ""))
        except ValueError as e:
            raise Bad(400, "bad_guest", str(e)) from None
        pay, _ = guestlink.verify(APP.guest_secret, tok)
        log.info("ссылка гостя выпущена: id %s, срок %s", pay["i"], time.strftime("%F %T", time.localtime(pay["e"])))
        return self._json(200, dict(link=guestlink.link(self._base(), tok), expires=pay["e"], id=pay["i"]))

    def _warm(self, qs):
        """Страница открылась: бот готовит приветствие заранее, мы — его звук. Ответ сразу; на линию звонок не встаёт."""
        who, pay = self._who()
        call = (qs.get("call") or [""])[0]
        if not CALL_ID.match(call):
            raise Bad(400, "bad_call", "нужен call — id звонка")
        if not APP.ready or C.brain != "bot" or not APP.bot.ok():
            return self._json(200, dict(ok=False, warm="off"))
        lang = (qs.get("lang") or ["kk"])[0]
        lang = lang if lang in ("kk", "ru") else "kk"
        if who == "lab":
            gp = guestlink.norm_phone(C.lab_guest_phone)
            guest = {"phone": gp, "name": C.lab_guest_name[:60]} if gp else None
        else:
            guest = guestlink.guest_of(pay)
        w = APP.warm(call, who, (pay or {}).get("i"), lang, guest, uuid.uuid4().hex[:10])
        touch_keepalive()
        return self._json(200, dict(ok=True, warm=w.state))

    def _end(self):
        h = self.headers.get("Authorization", "")
        j = {}
        if self.headers.get("Content-Length"):
            try:
                j = self._body_json()
            except Bad:
                j = {}
        tok = h[7:].strip() if h[:7].lower() == "bearer " else str(j.get("token") or "")
        call = str(j.get("call") or (parse_qs(urlsplit(self.path).query).get("call") or [""])[0])
        reason = re.sub(r"[^a-z_]", "", str(j.get("reason") or "hangup"))[:20] or "hangup"
        who, pay = self._who(tok)
        c = APP.line.take(call) if CALL_ID.match(call or "") else None
        if c is None and CALL_ID.match(call or ""):         # прогрев без звонка (страницу закрыли) — боту end, он закроет живой процесс
            w = APP.take_warm(call, who, (pay or {}).get("i"))
            # end — только по подтверждённому прогреву: бот без поддержки warm не подтвердит, а end без звонка создал бы пустую беседу
            if w and w.key:
                threading.Thread(target=lambda: log.info("прогрев без звонка закрыт (%s): end боту %s", reason,
                                                         APP.bot.end(call, w.lang, w.guest, reason)), daemon=True).start()
                return self._json(200, dict(ok=True, ended=False, warm_closed=True))
            if w and w.state == "warming":                  # бот ещё готовит — end уйдёт из потока прогрева, когда он ответит
                w.closed = reason
                return self._json(200, dict(ok=True, ended=False, warm_closed=True))
        APP.end_call(c, reason)
        return self._json(200, dict(ok=True, ended=bool(c)))

    def _write(self, b):
        if b:
            self.wfile.write(b"%X\r\n%s\r\n" % (len(b), b))

    # ------------------------------------------------------------ реплика
    def _turn(self, qs):
        t0 = time.time()
        rid = uuid.uuid4().hex[:10]
        g = lambda k, d: (qs.get(k) or [d])[0]  # noqa: E731
        who, pay = self._who()
        lab = who == "lab"
        touch_keepalive()
        if not APP.ready:
            raise Bad(503, "not_ready", APP.error or "ASR грузится")
        call = g("call", "")
        if not CALL_ID.match(call):
            raise Bad(400, "bad_call", "нужен call — id звонка")
        greet = g("greet", "0") == "1"
        hint = g("lang", "kk") if g("lang", "kk") in ("kk", "ru") else "kk"
        if lab:
            guest = None                                   # номер гостя — только из выдачи оператора, не со страницы
            gp = guestlink.norm_phone(C.lab_guest_phone)
            if gp:
                guest = {"phone": gp, "name": C.lab_guest_name[:60]}
            brain = g("brain", C.brain)                  # страница выбор мозга не запоминает — по умолчанию мозг сервера
            if brain not in ("bot", "llm", "script", "echo"):
                brain = C.brain
            ov = {k: g(k, "") for k in ("voice", "voice_ru")}
            params = {"mode": g("mode", "chain")}
            for k, cast in (("steps", int), ("cfg", float), ("seed", int)):
                if g(k, ""):
                    params[k] = cast(g(k, ""))
            sr = int(g("sr", str(C.sr_out)))
        else:
            guest, brain, ov, sr = guestlink.guest_of(pay), C.brain, {}, C.sr_out
            params = {"mode": "chain"}
            if C.steps:
                params["steps"] = int(C.steps)
        if greet:
            c, old = APP.line.start(call, who, guest, hint, brain)
            APP.end_call(old, "replaced")
        else:
            c = APP.line.get(call)
        if not APP.lock.acquire(blocking=False):
            raise Bad(503, "busy", "идёт другая реплика")
        st = dict(rid=rid, status=0, code="", t=time.strftime("%H:%M:%S"), call=call, who=who, gid=(pay or {}).get("i"), end=False, filler=0)
        with APP.m:                    # сразу: страница спрашивает итог, как только дочитала звук (раньше, чем отработает finally)
            APP.stats[rid] = st
            while len(APP.stats) > 300:
                APP.stats.popitem(last=False)
        out = Out(self, st, {"X-Request-Id": rid}, sr, t0)
        c["in_turn"] = True
        try:
            n = int(self.headers.get("Content-Length") or 0)
            sr_in = int(self.headers.get("X-Sample-Rate") or 16000)
            if not greet and (n <= 0 or n > C.max_audio_s * sr_in * 2):
                raise Bad(413 if n > 0 else 400, "bad_audio", f"PCM16 моно до {C.max_audio_s:.0f} с")
            raw = self.rfile.read(n) if n > 0 else b""
            t_recv = time.time()
            x = np.frombuffer(raw[: len(raw) // 2 * 2], "<i2").astype(np.float32) / 32768
            if sr_in != 16000 and len(x):
                import streamio
                rs = streamio.Resampler(sr_in, 16000)
                x = np.concatenate([rs.push(x), rs.push(np.zeros(0, np.float32), last=True)])
            st.update(audio_in_s=round(len(x) / 16000, 2), recv_ms=round((t_recv - t0) * 1000), brain=brain)
            if greet:
                text, t_asr, lang = "", time.time(), c.get("lang") or hint
            else:
                mt = self.headers.get("X-Mock-Text")
                text = APP.asr(x, mock_text=(None if mt is None else unquote(mt)))
                t_asr = time.time()
                if not text.strip():   # тишина/эхо своего же голоса → не отвечаем «естімей қалдым» по кругу
                    raise Bad(422, "no_speech", "речи не слышно — говорите ближе к микрофону")
                lang = lang_of(text, c.get("lang") or hint)
                c["lang"] = lang
            st.update(asr_ms=round((t_asr - t_recv) * 1000), lang_in=lang, chars_in=len(text), _asr=text)
            base = {"X-Recv-Ms": st["recv_ms"], "X-ASR-Ms": st["asr_ms"], "X-ASR-Text": quote(text, safe=""), "X-Brain": brain}
            if brain == "bot":
                w = APP.take_warm(call, who, (pay or {}).get("i")) if greet else None
                st["warm"] = "none" if greet and w is None else None
                self._speak_bot(out, c, st, "start" if greet else "turn", text, lang, ov, params, base, rid, t0, t_asr, w)
            else:
                if greet:
                    reply, lang_out, who_b = SAY["greet"][lang][0], lang, "greet"
                else:
                    delay = min(5000, max(0, int(g("brain_delay", "0") or 0))) if lab else 0
                    reply, lang_out, who_b = APP.brain.reply(brain, text, call, lang)
                    if delay:
                        time.sleep(delay / 1000)
                t_b = time.time()
                st.update(brain_ms=round((t_b - t_asr) * 1000), brain=who_b, lang=lang_out, chars_out=len(reply), _reply=reply)
                ex = dict(base, **{"X-Brain-Ms": st["brain_ms"], "X-Reply-Text": quote(reply, safe=""), "X-Lang": lang_out, "X-Brain": who_b,
                                   "X-Filler": 0, "X-End": 0, "X-Server-Ms": round((t_b - t0) * 1000)})
                info = out.tts(reply, voice_for(lang_out, ov), params, ex, rid)
                st.update(info)
                st["server_ms"] = round(((out.t_first or time.time()) - t0) * 1000)
            if not out.sent:
                st.update(status=204, code="silent")
                return self._send(204, b"", "text/plain", {"X-Request-Id": rid})
            st.update(status=200, code="ok", bytes_out=out.nb, total_ms=round((time.time() - t0) * 1000),
                      first_audio_ms=round((out.t_first - t0) * 1000))
            out.close()
        except Bad as e:
            st.update(status=e.status, code=e.code)
            if not out.sent:
                self._json(e.status, dict(error=e.code, detail=e.detail), {"Retry-After": 2} if e.status == 503 else None)
        except (BrokenPipeError, ConnectionResetError, socket.timeout, TimeoutError, ssl.SSLError):
            st.update(status=499, code="client_gone")
        except (http.client.HTTPException, OSError) as e:
            st.update(status=502, code="tts_unreachable")
            log.info("голос недоступен: %s", type(e).__name__)
            if not out.sent:
                self._json(502, dict(error="tts_unreachable", detail=type(e).__name__))
        except Exception as e:  # noqa: BLE001
            st.update(status=500, code="call_failed")
            log.error("turn 500 %s: %s", type(e).__name__, str(e)[:200])
            if not out.sent:
                self._json(500, dict(error="call_failed", detail=type(e).__name__))
        finally:
            APP.lock.release()
            c["in_turn"] = False
            c["t_last"] = time.time()
            c["turns"] += 1
            self.close_connection = True
            touch_keepalive()
            if st.get("tts_rid") and st["status"] == 200:
                try:
                    _, ts = APP.tts_get(f"/tts/stats?rid={st['tts_rid']}", C.tts_url_ru if st.get("tts_base") and C.tts_url_ru else None)
                    st["tts"] = {k: ts.get(k) for k in ("ttfa_ms", "ttfa_model_ms", "queue_ms", "audio_s", "synth_s", "rtf", "prebuffer_ms",
                                                        "patches", "runaway", "lead_cut_ms", "chunks")}
                except (OSError, ValueError):
                    pass
            with APP.m:
                APP.n_turns += 1
            ts = st.get("tts") or {}
            log.info("turn %s %s rid=%s %s %s in=%.1fs recv=%s asr=%s brain=%s(%s) first=%s filler=%s seg=%s bot=%s end=%s warm=%s tts_head=%s "
                     "audio=%s rtf=%s steps=%s", st["status"], st["code"], rid, who, "greet" if greet else "turn", st.get("audio_in_s", 0),
                     st.get("recv_ms"), st.get("asr_ms"), st.get("brain_ms"), st.get("brain"), st.get("first_audio_ms"), st.get("filler"),
                     st.get("segments"), st.get("bot_err") or st.get("bot_ms") or "-", int(bool(st.get("end"))),
                     (st.get("warm") or "-") + (f"/{st['warm_state']}" if st.get("warm_state") else ""), st.get("tts_head_ms"),
                     ts.get("audio_s"), ts.get("rtf"), st.get("steps"))
            if st.get("end"):      # бот сам закончил разговор (или «сбой связи») — страница положит трубку и пришлёт end; запасной путь — сторож
                c["bot_end"] = True

    def _play_warm(self, out: Out, w: Warm, text: str, st: dict, extra: dict) -> bool:
        """Приветствие бота совпало с прогретым — звук из буфера прогрева (готовый — сразу, недосинтезированный — следом)."""
        t = time.time()
        while w.state == "warming" and time.time() - t < 1.5:      # ответы warm и start приходят от бота почти одновременно
            time.sleep(0.02)
        if w.state not in ("synth", "ready") or w.key != loose(text):
            st["warm"] = "miss:" + (w.state if w.state not in ("synth", "ready") else "text")
            return False
        st["warm_wait_ms"] = round((time.time() - t) * 1000)
        st["warm_state"] = w.state                                  # ready — звук был готов к нажатию; synth — шёл хвостом
        n = 0
        try:
            for b in w.stream():
                out.write(b, dict(extra, **{"X-Warm": 1}))
                n += 1
        except TimeoutError:
            log.info("прогрев: синтез встал посреди приветствия")
        if n:
            st["warm"] = "hit"
            return True
        st["warm"] = "miss:" + str(w.err or "empty")
        return False

    def _speak_bot(self, out: Out, c: dict, st: dict, event: str, text: str, lang: str, ov: dict, params: dict, base: dict,
                     rid: str, t0: float, t_asr: float, warm: Warm | None = None):
        """Ответ внешнего бота голосом: предложения по мере готовности, «Секунду.», если бот молчит, «Сейчас посмотрю» на поиск,
        запасные фразы при сбое. Порядок ходов одного звонка — строго по очереди (APP.lock + полудуплекс страницы)."""
        if event == "start":
            c["bot"] = True
        q = APP.bot.request(event, c["call"], text, lang, c.get("guest"), rid)
        t_req = time.time()
        deadline = t_req + C.bot_timeout + 1
        said: list[str] = []
        pend: collections.deque = collections.deque()
        f1 = f2 = wt = False                       # «Секунду.», «Ещё секунду», «Сейчас посмотрю» — каждая не больше раза за ход

        def ex(**kw):
            d = dict(base)
            d.update({"X-Brain-Ms": kw.pop("brain_ms", -1), "X-Reply-Text": quote(kw.pop("reply", ""), safe=""), "X-Lang": kw.pop("lang", lang),
                      "X-Filler": int(f1 or f2 or wt), "X-End": int(bool(st.get("end"))), "X-Warm": 0})
            return d

        def say(key, lg):
            arr = SAY[key][lg]
            txt = arr[c["fill_i"] % len(arr)] if key == "filler" else arr[0]
            if key == "filler":
                c["fill_i"] += 1
            st["filler"] = st.get("filler", 0) + (key in ("filler", "filler2", "wait"))
            if out.sent:
                out.pause(150)
            out.clip(txt, voice_for(lg, ov), ex())
            return txt

        def fail(why):
            c["fails"] += 1
            st["bot_err"] = why
            log.info("бот не ответил (%s, %s, сбоев подряд %d)%s", event, why, c["fails"], " — посреди ответа" if said else "")
            if said:
                return
            if event == "start":
                st.update(_reply=say("greet", lang), brain="greet_local")
                return
            key = "fail_many" if c["fails"] >= 2 else "fail"
            if key == "fail_many":
                st["end"] = True
            st.update(_reply=say(key, lang), brain="fallback")

        while True:
            if pend:
                ev = pend.popleft()
            else:
                when, nxt = deadline, None
                if event != "start":
                    if not (said or f1 or wt) and C.filler_ms > 0:
                        when, nxt = min(deadline, t_req + C.filler_ms / 1000), "filler"
                    elif not f2 and C.filler2_ms > 0 and (said or f1 or wt):
                        when, nxt = min(deadline, (out.t_last or t_req) + C.filler2_ms / 1000), "filler2"
                try:
                    ev = q.get(timeout=max(0.0, when - time.time()))
                except queue.Empty:
                    if time.time() >= deadline or nxt is None:
                        ev = ("error", "timeout")
                    else:
                        if nxt == "filler":
                            f1 = True
                        else:
                            f2 = True
                        say(nxt, lang)                 # флаг — до звука: X-Filler уходит в заголовках вместе с ним
                        continue
            kind = ev[0]
            if "brain_ms" not in st and kind in ("say", "final"):
                st["brain_ms"] = round((time.time() - t_asr) * 1000)
            if kind == "wait":                         # бот ищет в интернете (10–30 с): одна короткая фраза, если своё ещё не говорили
                st["bot_wait"] = ev[1] or "search"
                if not (f1 or wt):
                    wt = True
                    say("wait", lang)
                continue
            if kind == "say":
                texts, lg = [ev[1]], ev[2] if ev[2] in ("kk", "ru") else lang_of(ev[1], lang)
                while True:                            # всё, что бот уже успел прислать, — одним куском (интонация связнее)
                    try:
                        e2 = q.get_nowait()
                    except queue.Empty:
                        break
                    if e2[0] == "say":
                        texts.append(e2[1])
                    else:
                        pend.append(e2)
                        break
                if out.sent:
                    out.pause(150)
                if warm is not None and event == "start" and not said and self._play_warm(out, warm, " ".join(texts), st,
                                                                                           ex(brain_ms=st["brain_ms"], lang=lg)):
                    warm = None
                else:
                    info = out.tts(" ".join(texts), voice_for(lg, ov), params, ex(brain_ms=st["brain_ms"], lang=lg), rid)
                    st.update({k: v for k, v in info.items() if k not in st})
                said += texts
                st["segments"] = st.get("segments", 0) + 1
                st["lang"] = lg
                continue
            if kind == "final":
                reply, lg, end, ms, err = ev[1].strip(), ev[2], ev[3], ev[4], ev[5] if len(ev) > 5 else None
                st.update(end=bool(end), bot_ms=ms if isinstance(ms, (int, float)) else None,
                          brain_total_ms=round((time.time() - t_asr) * 1000), _reply=reply or " ".join(said), chars_out=len(reply))
                if err and not said and not reply:     # brain_down / empty_reply: говорить нечего — что сказать на линии, решаем мы
                    fail("bot_" + str(err))
                    return
                c["fails"] = 0
                lg = lg if lg in ("kk", "ru") else lang_of(reply, lang)
                st["lang"] = lg
                if not said:
                    if reply:
                        if out.sent:
                            out.pause(150)
                        if not (warm is not None and event == "start" and self._play_warm(out, warm, reply, st,
                                                                                         ex(brain_ms=st["brain_ms"], reply=reply, lang=lg))):
                            info = out.tts(reply, voice_for(lg, ov), params, ex(brain_ms=st["brain_ms"], reply=reply, lang=lg), rid)
                            st.update(info)
                        st["segments"] = 1
                    elif event == "start":
                        st.update(_reply=say("greet", lang), brain="greet_local")
                return
            fail(ev[1])                                # ("error", код): сеть, тайм-аут, HTTP ≠ 200, {"type":"error"}
            return


class Server(ThreadingHTTPServer):
    daemon_threads = True
    ctx: ssl.SSLContext | None = None

    def process_request_thread(self, request, client_address):   # TLS-рукопожатие — в потоке запроса, не в главном
        if self.ctx is not None:
            try:
                request.settimeout(15)
                request = self.ctx.wrap_socket(request, server_side=True)
            except (ssl.SSLError, OSError):
                self.shutdown_request(request)
                return
        super().process_request_thread(request, client_address)


def main():
    global APP
    setup_logging()
    APP = App()
    srv = Server((C.host, C.port), Handler)
    if C.cert and C.key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(C.cert, C.key)
        srv.ctx = ctx
    log.info("слушаю %s:%d (%s), гружу ASR…", C.host, C.port, "TLS" if srv.ctx else "без TLS")
    if C.plain_port:
        lo = Server(("127.0.0.1", C.plain_port), Handler)
        threading.Thread(target=lo.serve_forever, kwargs=dict(poll_interval=0.5), daemon=True).start()
        log.info("и 127.0.0.1:%d без TLS — для туннеля (http://127.0.0.1:%d/call)", C.plain_port, C.plain_port)

    def stop(signum, frame):   # systemctl stop / конец сеанса: идущий звонок → end боту, потом выход
        c = APP.line.take()
        APP.end_call(c, "server_stop", sync=True)
        log.info("стоп по сигналу %d", signum)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    threading.Thread(target=APP.start, daemon=True).start()
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
