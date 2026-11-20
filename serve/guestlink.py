#!/usr/bin/env python3
"""guestlink.py — личная ссылка гостя на тестовый звонок. Только стандартная библиотека.

Ссылка: https://<адрес звонка>/#g=<токен>. Токен = g1.<данные>.<подпись>:
  данные — base64url JSON {"n": имя, "p": "7XXXXXXXXXX", "e": срок (unix), "i": случайный id ссылки, "l": "kk"|"ru" (язык страницы)};
  подпись — HMAC-SHA256 секретом сеанса (файл guest_secret, на сервере — $TTS_HOME/guest_secret; в репозиторий не входит).
Зачем подпись: если бот по номеру гостя продолжает его историю переписки, поддельный номер в ссылке = чужая переписка голосом.
Токен идёт во фрагменте (#g=…): браузер не шлёт его ни серверу при загрузке страницы, ни Cloudflare в журнал; страница
отдаёт его как Bearer в запросах звонка.

  guestlink.py init   --secret-file F                       создать секрет (если нет), права 600
  guestlink.py make   --secret-file F --base URL --name N --phone P [--hours 72] [--lang kk|ru]   → ссылка
  guestlink.py verify --secret-file F TOKEN                 → данные или причина отказа
  guestlink.py                                              самопроверка
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

VER = "g1"
SIG_BYTES = 18          # 144 бита — подбор невозможен, ссылка короче


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def norm_phone(p: str | None) -> str | None:
    """«+7 (701) 234-56-78», «87012345678», «7012345678» → «77012345678»; не казахстанский/российский формат → None."""
    d = re.sub(r"\D", "", p or "")
    if len(d) == 11 and d[0] == "8":
        d = "7" + d[1:]
    elif len(d) == 10 and d[0] in "67":
        d = "7" + d
    return d if re.fullmatch(r"7\d{10}", d) else None


def load_secret(path: str | Path) -> bytes:
    s = Path(path).read_text().strip()
    if len(s) < 32:
        raise ValueError("секрет короче 32 знаков")
    return s.encode()


def make(secret: bytes, name: str, phone: str, hours: float = 72, lang: str = "", now: float | None = None) -> str:
    p = norm_phone(phone)
    if not p:
        raise ValueError(f"номер не похож на казахстанский: {phone!r} (нужно 11 цифр, 7XXXXXXXXXX)")
    name = re.sub(r"\s+", " ", (name or "").strip())[:60]
    if not name:
        raise ValueError("пустое имя")
    pay = {"n": name, "p": p, "e": int((now or time.time()) + hours * 3600), "i": secrets.token_hex(3)}
    if lang in ("kk", "ru"):
        pay["l"] = lang
    body = _b64e(json.dumps(pay, ensure_ascii=False, separators=(",", ":")).encode())
    sig = _b64e(hmac.new(secret, f"{VER}.{body}".encode(), hashlib.sha256).digest()[:SIG_BYTES])
    return f"{VER}.{body}.{sig}"


def verify(secret: bytes, token: str, now: float | None = None) -> tuple[dict | None, str]:
    """→ (данные, "") или (None, "bad" | "expired")."""
    parts = (token or "").strip().split(".")
    if len(parts) != 3 or parts[0] != VER or not secret:
        return None, "bad"
    want = _b64e(hmac.new(secret, f"{VER}.{parts[1]}".encode(), hashlib.sha256).digest()[:SIG_BYTES])
    if not hmac.compare_digest(want.encode(), parts[2].encode()):
        return None, "bad"
    try:
        pay = json.loads(_b64d(parts[1]).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, "bad"
    if not isinstance(pay, dict) or not norm_phone(pay.get("p")) or not isinstance(pay.get("e"), int):
        return None, "bad"
    if pay["e"] < (now or time.time()):
        return None, "expired"
    return pay, ""


def guest_of(pay: dict) -> dict:
    """Данные токена → поле guest контракта /voice/turn."""
    return {"phone": pay["p"], "name": pay.get("n") or ""}


def link(base: str, token: str) -> str:
    return base.rstrip("/") + "/#g=" + token


def init_secret(path: str | Path) -> bool:
    p = Path(path)
    if p.is_file() and len(p.read_text().strip()) >= 32:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    old = os.umask(0o077)
    try:
        p.write_text(secrets.token_urlsafe(32) + "\n")
    finally:
        os.umask(old)
    os.chmod(p, 0o600)
    return True


def _selftest() -> int:
    fails = []

    def check(name, got, want):
        ok = got == want
        print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {got!r} != {want!r}"))
        if not ok:
            fails.append(name)
    s = secrets.token_urlsafe(32).encode()
    for raw, want in (("+7 (701) 234-56-78", "77012345678"), ("87012345678", "77012345678"), ("7012345678", "77012345678"),
                      ("77012345678", "77012345678"), ("12345", None), ("+1 555 010 0000", None)):
        check(f"номер {raw}", norm_phone(raw), want)
    t = make(s, "Тест Қонақ", "8 701 234 56 78", hours=1, lang="kk")
    pay, err = verify(s, t)
    check("свой токен проходит", (err, pay and pay["p"], pay and pay["n"], pay and pay["l"]), ("", "77012345678", "Тест Қонақ", "kk"))
    check("guest контракта", guest_of(pay), {"phone": "77012345678", "name": "Тест Қонақ"})
    head, body, sig = t.split(".")
    fake = json.loads(_b64d(body))
    fake["p"] = "77770000000"
    t2 = ".".join([head, _b64e(json.dumps(fake, ensure_ascii=False, separators=(",", ":")).encode()), sig])
    check("подменённый номер не проходит", verify(s, t2), (None, "bad"))
    check("чужой секрет не проходит", verify(secrets.token_urlsafe(32).encode(), t), (None, "bad"))
    check("срок вышел", verify(s, make(s, "А", "77012345678", hours=1), now=time.time() + 7200), (None, "expired"))
    check("мусор", verify(s, "abc"), (None, "bad"))
    check("длина токена ≤ 200", len(t) <= 200, True)
    print(f"guestlink: {'ВСЁ ПРОШЛО' if not fails else 'НЕ ПРОШЛО: ' + ', '.join(fails)} (токен {len(t)} знаков)")
    return 1 if fails else 0


def main() -> int:
    if len(sys.argv) == 1:
        return _selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_init = sub.add_parser("init")
    a_make = sub.add_parser("make")
    a_ver = sub.add_parser("verify")
    for p in (a_init, a_make, a_ver):
        p.add_argument("--secret-file", required=True)
    a_make.add_argument("--base", required=True, help="адрес звонка, например https://xxx.trycloudflare.com")
    a_make.add_argument("--name", required=True)
    a_make.add_argument("--phone", required=True)
    a_make.add_argument("--hours", type=float, default=72)
    a_make.add_argument("--lang", default="", choices=["", "kk", "ru"])
    a_ver.add_argument("token")
    a = ap.parse_args()
    if a.cmd == "init":
        print("секрет создан" if init_secret(a.secret_file) else "секрет уже есть", a.secret_file)
        return 0
    s = load_secret(a.secret_file)
    if a.cmd == "make":
        try:
            print(link(a.base, make(s, a.name, a.phone, a.hours, a.lang)))
        except ValueError as e:
            print("ошибка:", e, file=sys.stderr)
            return 2
        return 0
    pay, err = verify(s, a.token)
    print(json.dumps(pay, ensure_ascii=False) if pay else f"отказ: {err}")
    return 0 if pay else 1


if __name__ == "__main__":
    raise SystemExit(main())
