#!/usr/bin/env python3
"""mock_test.py — проверка голосового сервиса в песочнице (tts_server.py с TTS_MOCK=1): API, защита, очередь, срок, режимы, OGG/Opus, журнал.

  python mock_test.py --url http://127.0.0.1:PORT --token-file T --log L --keepalive K --calls CALLS.jsonl --out DIR
Сервис должен быть запущен с TTS_MOCK=1, TTS_QUEUE_MAX=2, TTS_DEADLINE=6, TTS_DEBUG_CALLS=CALLS.jsonl. Выход 0 — всё прошло.
"""
import argparse
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MARK_KK = "Қызылжарлықтар"            # слово-метка: в журнале его быть не должно (текст клиента не пишем)
MARK_RU = "кадастровыйномер"
KK = (f"Сәлеметсіз бе, {MARK_KK}! Шебер ертең сағат 10:30-да келеді. Жұмыс шамамен екі сағатқа созылады, бағасы 15 000 ₸. "
      "Төлемді жұмыс аяқталғаннан кейін жасайсыз. Сұрақтарыңыз болса, маған жазыңыз, мен әрқашан көмектесемін. Рахмет!")
RU = f"Здравствуйте! Ваш {MARK_RU} принят. Перезвоним по номеру +7 701 234 56 78 до 18:00, скидка 10%."

fails = []


def check(name, ok, info=""):
    print(("ok   " if ok else "FAIL ") + name + (f" — {info}" if info else ""))
    if not ok:
        fails.append(name)


def req(url, path, body=None, token=None, raw=None, method=None, timeout=60):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    r = urllib.request.Request(url + path, data=data, method=method or ("POST" if data is not None else "GET"))
    if data is not None:
        r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=format_name,duration:stream=codec_name,channels,sample_rate",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    j = json.loads(out or "{}")
    s = (j.get("streams") or [{}])[0]
    return dict(fmt=j.get("format", {}).get("format_name"), dur=float(j.get("format", {}).get("duration") or 0),
                codec=s.get("codec_name"), ch=s.get("channels"), sr=int(s.get("sample_rate") or 0))


def main():
    ap = argparse.ArgumentParser()
    for k in ("url", "token-file", "log", "keepalive", "calls", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    url, token = a.url, Path(a.token_file).read_text().strip()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    n_auth = 0          # запросов /tts с верным токеном — столько должен показать счётчик для сторожа

    def tts(body, **kw):
        nonlocal n_auth
        n_auth += 1
        return req(url, "/tts", body, token, **kw)

    def calls_since(pos):
        p = Path(a.calls)
        if not p.exists():
            return [], 0
        b = p.read_bytes()
        return [json.loads(x) for x in b[pos:].decode().splitlines() if x.strip()], len(b)

    # 1. готовность
    for _ in range(60):
        st, _, b = req(url, "/health")
        if st == 200 and json.loads(b).get("ready"):
            break
        time.sleep(0.5)
    h = json.loads(b)
    check("health: ready", st == 200 and h.get("ready"), f"голоса {list(h.get('voices', {}))}, по умолчанию {h.get('default_voice')}, "
          f"{h.get('defaults')}, кодировщик {h.get('encoder')}")
    check("health: три голоса", set(h.get("voices", {})) == {"aruzhan_f", "male_m", "aruzhan_f_ru"})
    check("health: по умолчанию aruzhan_f · chain · cfg 2,0 · 10 шагов",
          h.get("default_voice") == "aruzhan_f" and h.get("defaults") == {"mode": "chain", "cfg": 2.0, "steps": 10, "seed": 1})

    # 2. защита и ошибки
    st, hd, _ = req(url, "/tts", {"text": "Сәлем"})
    check("без токена → 401", st == 401 and hd.get("WWW-Authenticate") == "Bearer", str(st))
    st, _, _ = req(url, "/tts", {"text": "Сәлем"}, token="x" * 43)
    check("чужой токен → 401", st == 401, str(st))
    st, _, _ = req(url, "/nope")
    check("неизвестный путь → 404", st == 404, str(st))
    st, _, b = req(url, "/metrics")
    check("/metrics с 127.0.0.1 отдаётся", st == 200 and b"vllm:request_success_total 0" in b, b.decode()[:60].replace("\n", " | "))
    for name, body, want, code in [
        ("пустой текст", {"text": "  "}, 400, "empty_text"),
        ("только ссылка и эмодзи", {"text": "😊 https://example.kz"}, 400, "empty_text"),
        ("1001 знак", {"text": "а" * 1001}, 413, "text_too_long"),
        ("чужой голос", {"text": "Сәлем", "voice": "nope"}, 400, "bad_voice"),
        ("чужой режим", {"text": "Сәлем", "mode": "fast"}, 400, "bad_mode"),
        ("cfg 9", {"text": "Сәлем", "cfg": 9}, 400, "bad_param"),
        ("steps строкой", {"text": "Сәлем", "steps": "много"}, 400, "bad_param"),
        ("format mp3", {"text": "Сәлем", "format": "mp3"}, 400, "bad_format"),
    ]:
        st, _, b = tts(body)
        check(f"{name} → {want} {code}", st == want and json.loads(b).get("error") == code, f"{st} {b.decode()[:80]}")
    st, _, b = tts(None, raw=b"{not json")
    check("битый JSON → 400 bad_json", st == 400 and json.loads(b).get("error") == "bad_json", str(st))

    # 3. основной путь: казахский текст с числами, 2+ куска, OGG/Opus моно 48 кГц ≤ 512 КБ
    pos = calls_since(0)[1]
    t = time.time()
    st, hd, b = tts({"text": KK})
    f = out / "kk_chain.ogg"
    f.write_bytes(b)
    p = ffprobe(f) if st == 200 else {}
    check("kk → 200 audio/ogg", st == 200 and hd.get("Content-Type") == "audio/ogg", f"{st} {hd.get('Content-Type')} за {time.time() - t:.1f} с")
    check("OGG/Opus, моно, 48 кГц (WhatsApp)", p.get("fmt") == "ogg" and p.get("codec") == "opus" and p.get("ch") == 1 and p.get("sr") == 48000,
          str(p))
    check("≤ 512 КБ (значок «play» в WhatsApp)", len(b) < 512 * 1024, f"{len(b)} байт")
    check("заголовки: X-Chunks ≥ 2, X-Audio-Seconds ≈ ffprobe",
          int(hd.get("X-Chunks", 0)) >= 2 and abs(float(hd.get("X-Audio-Seconds", 0)) - p.get("dur", 0)) < 0.2,
          f"кусков {hd.get('X-Chunks')}, звук {hd.get('X-Audio-Seconds')} с, ffprobe {p.get('dur', 0):.2f} с, RTF {hd.get('X-RTF')}")
    cl, pos = calls_since(pos)
    check("chain: кусок 1 только эталон, дальше — продолжение предыдущего",
          len(cl) >= 2 and cl[0]["prompt"] is None and all(c["prompt"] == "prev" for c in cl[1:])
          and all(c["cfg"] == 2.0 and c["steps"] == 10 for c in cl), str([(c["chunk"], c["prompt"]) for c in cl]))

    # 4. режимы, голоса, параметры
    for mode, voice, want0 in (("ref", "aruzhan_f", None), ("clone", "aruzhan_f", "voice"), ("clone", "aruzhan_f_ru", None),
                               ("chain", "male_m", None)):
        st, hd, b = tts({"text": KK, "mode": mode, "voice": voice})
        cl, pos = calls_since(pos)
        rest = [c["prompt"] for c in cl[1:]]
        want_rest = None if mode == "ref" else "prev"
        check(f"{mode} · {voice}: кусок 1 = {want0 or 'только эталон'}, дальше {want_rest or 'только эталон'}",
              st == 200 and cl and cl[0]["prompt"] == want0 and all(r == want_rest for r in rest) and all(c["ref"] == voice for c in cl),
              f"{st} {[(c['chunk'], c['prompt']) for c in cl]}")
    st, hd, b = tts({"text": RU, "voice": "aruzhan_f_ru", "cfg": 1.5, "steps": 25, "seed": 7})
    cl, pos = calls_since(pos)
    (out / "ru_ref_voice.ogg").write_bytes(b)
    check("ru + cfg 1,5 · 25 шагов · сид 7 доходят до модели", st == 200 and cl and all(c["cfg"] == 1.5 and c["steps"] == 25 for c in cl)
          and cl[0]["seed"] == 7, f"{st} {[(c['cfg'], c['steps'], c['seed']) for c in cl]}")
    st, hd, b = tts({"text": RU, "voice": "aruzhan_f_ru", "cfg": 1.5, "steps": 25, "seed": 7})
    cl, pos = calls_since(pos)
    check("повтор того же запроса — из кэша, модель не зовётся", st == 200 and hd.get("X-Cache") == "hit" and not cl, f"{hd.get('X-Cache')}")
    st, hd, b = tts({"text": "Иә, жазып қойдым.", "format": "wav"})
    check("format wav → audio/wav", st == 200 and hd.get("Content-Type") == "audio/wav" and b[:4] == b"RIFF", f"{st}")
    long_txt = ("Шебер ертең таңертең келеді, жұмыс шамамен екі сағатқа созылады, бағасын орнында айтады. " * 11)[:1000]
    st, hd, b = tts({"text": long_txt, "seed": 3})
    (out / "kk_1000.ogg").write_bytes(b)
    p = ffprobe(out / "kk_1000.ogg") if st == 200 else {}
    check("1000 знаков → 200, ≤ 512 КБ", st == 200 and len(b) < 512 * 1024 and p.get("codec") == "opus",
          f"{st}, {len(b)} байт, {p.get('dur', 0):.1f} с звука, кусков {hd.get('X-Chunks')}")

    # 5. очередь: одна генерация за раз, ждущих ≤ 2, лишние — 503 busy с Retry-After
    res = []

    def one(i):
        res.append(tts({"text": f"Сәлеметсіз бе! Жазбаңыз қабылданды.", "seed": 100 + i, "_mock_delay": 1.0}))
    th = [threading.Thread(target=one, args=(i,)) for i in range(5)]
    for x in th:
        x.start()
        time.sleep(0.05)
    for x in th:
        x.join()
    codes = sorted(r[0] for r in res)
    busy = [r for r in res if r[0] == 503]
    check("5 одновременных: 3 × 200, 2 × 503 busy с Retry-After", codes.count(200) == 3 and codes.count(503) == 2
          and all(json.loads(r[2]).get("error") == "busy" and r[1].get("Retry-After") for r in busy), str(codes))
    h = json.loads(req(url, "/health")[2])
    check("модель ни разу не работала параллельно", h.get("mock_max_parallel") == 1, f"максимум {h.get('mock_max_parallel')}")

    # 6. срок: 4 куска по 2,5 с при TTS_DEADLINE=6 → 504
    st, _, b = tts({"text": long_txt[:700], "seed": 9, "_mock_delay": 2.5}, timeout=60)
    check("синтез дольше срока → 504 timeout", st == 504 and json.loads(b).get("error") == "timeout", f"{st} {b.decode()[:80]}")

    # 7. сторожа: счётчик и keepalive, журнал без текста и токена
    st, _, b = req(url, "/metrics")
    m = dict(x.split(" ", 1) for x in b.decode().splitlines() if x and not x.startswith("#"))
    check("счётчик для сторожа gpu = числу запросов с токеном", int(float(m.get("vllm:request_success_total", -1))) == n_auth
          and m.get("vllm:num_requests_running") == "0", f"{m.get('vllm:request_success_total')} из {n_auth}")
    ka = os.path.getmtime(a.keepalive) if os.path.exists(a.keepalive) else 0
    check("keepalive обновлён запросами (сторож простоя)", ka > t_start, f"{time.time() - ka:.1f} с назад")
    lg = Path(a.log).read_text()
    check("в журнале нет текста клиента", MARK_KK not in lg and MARK_RU not in lg and "Шебер" not in lg)
    check("в журнале нет токена", token not in lg)
    ok_lines = [x for x in lg.splitlines() if " tts 200 ok " in x]
    check("в журнале — длины, время, RTF", ok_lines and "chars=" in ok_lines[-1] and "rtf=" in ok_lines[-1], ok_lines[-1][20:170] if ok_lines else "")
    print(f"\nитог: {'ВСЁ ПРОШЛО' if not fails else 'НЕ ПРОШЛО: ' + ', '.join(fails)}; образцы OGG (синус-заглушка): {out}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
