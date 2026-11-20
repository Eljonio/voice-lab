#!/usr/bin/env python3
"""textprep.py — текст бота → «произносимые» куски для VoxCPM2 (сервис tts_server.py).

prepare(text) -> list[dict(text, lang)]
  1. ссылки, почты, домены — вон (голосом их не читают); маркеры списков и переносы строк → границы предложений;
  2. бренды и латинские сокращения — по словарю lexicon.json («WhatsApp» → «уатсап», «AI» → «эй-ай»);
  3. числа словами по языку предложения: казахский — dataprep/textnorm.numbers_kk (тот же слой, что в оценке phrases_v2),
     русский — numbers_ru ниже (именительный падеж; падежи и порядковые бот пишет словами сам); остаток цифр — числом;
  4. textnorm.normalize: скобки (у VoxCPM2 скобка в начале = команда Voice Design), кавычки, markdown, эмодзи, КАПС, гомоглифы;
  5. деление на предложения и упаковка в куски ≤ MAX_CHARS (≈ 12 с речи): короткие реплики («Иә.») не синтезируются в одиночку,
     длинное предложение режется по запятой ближе к середине.
Только стандартная библиотека.  Самопроверка: python3 textprep.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "dataprep"):    # на сервере textnorm.py может лежать рядом, в репо — в dataprep/
    if (p / "textnorm.py").exists():
        sys.path.insert(0, str(p))
        break
from textnorm import _ordinal, normalize, num_kk, numbers_kk  # noqa: E402

MAX_CHARS = 180      # ≈ 14 с казахской речи (эталон aruzhan_f_kk: 159 знаков = 12,5 с)
MIN_CHARS = 25       # короче — приклеиваем к соседнему
KK_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
CYR = re.compile(r"[а-яёА-ЯЁәғқңөұүһіӘҒҚҢӨҰҮҺІ]")

# ---------------------------------------------------------------- числа по-русски (именительный падеж)
_R_ONES_M = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_R_ONES_F = ["", "одна", "две"] + _R_ONES_M[3:]
_R_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
            "семнадцать", "восемнадцать", "девятнадцать"]
_R_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
_R_HUND = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]
_R_BIG = [(10 ** 9, ("миллиард", "миллиарда", "миллиардов"), False), (10 ** 6, ("миллион", "миллиона", "миллионов"), False),
          (1000, ("тысяча", "тысячи", "тысяч"), True)]


def plural_ru(n: int, forms: tuple[str, str, str]) -> str:
    n = abs(n)
    if 11 <= n % 100 <= 14:
        return forms[2]
    return forms[0] if n % 10 == 1 else forms[1] if 2 <= n % 10 <= 4 else forms[2]


def _r_u1000(n: int, fem: bool) -> list[str]:
    out = []
    h, r = divmod(n, 100)
    if h:
        out.append(_R_HUND[h])
    if 10 <= r <= 19:
        out.append(_R_TEENS[r - 10])
    else:
        t, o = divmod(r, 10)
        if t:
            out.append(_R_TENS[t])
        if o:
            out.append((_R_ONES_F if fem else _R_ONES_M)[o])
    return out


def num_ru(n: int) -> str:
    """15000 → «пятнадцать тысяч», 2001 → «две тысячи один», 0 → «ноль»."""
    if n == 0:
        return "ноль"
    out = []
    for v, forms, fem in _R_BIG:
        k, n = divmod(n, v)
        if k:
            out += _r_u1000(k, fem) + [plural_ru(k, forms)]
    out += _r_u1000(n, False)
    return " ".join(out)


def _digits_ru(g: str) -> str:
    lead = len(g) - len(g.lstrip("0"))
    rest = g[lead:]
    return " ".join(["ноль"] * lead + ([num_ru(int(rest))] if rest else []))


_PHONE = re.compile(r"(\+7|(?<!\d)8)[\s-]?\(?(\d{3})\)?[\s-]?(\d{3})[\s-]?(\d{2})[\s-]?(\d{2})(?!\d)")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_NUM_RU = re.compile(r"(?<![\w-])(\d{1,3}(?:[   ]\d{3})+|\d+)(?:-[а-яё]+)?(\s*(?:%|₸))?")


# ---------------------------------------------------------------- даты (16.10: бот пишет «17.10» — голос читал «семнадцать.десять»)
_DATE = re.compile(r"(?<![\d.,])(\d{1,2})\.(\d{2})(?:\.(\d{4}))?(?![\d]|\.\d)")
_ORD_GEN_RU = ("", "первого", "второго", "третьего", "четвёртого", "пятого", "шестого", "седьмого", "восьмого", "девятого", "десятого",
               "одиннадцатого", "двенадцатого", "тринадцатого", "четырнадцатого", "пятнадцатого", "шестнадцатого", "семнадцатого",
               "восемнадцатого", "девятнадцатого")
_TENS_RU = {20: ("двадцать", "двадцатого"), 30: ("тридцать", "тридцатого"), 40: ("сорок", "сорокового"), 50: ("пятьдесят", "пятидесятого"),
            60: ("шестьдесят", "шестидесятого"), 70: ("семьдесят", "семидесятого"), 80: ("восемьдесят", "восьмидесятого"),
            90: ("девяносто", "девяностого")}
_MONTHS_RU = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря")
_MONTHS_KK = ("қаңтар", "ақпан", "наурыз", "сәуір", "мамыр", "маусым", "шілде", "тамыз", "қыркүйек", "қазан", "қараша", "желтоқсан")


def _ord_gen_ru(n: int) -> str:
    """1–99 → порядковое в родительном: 17 → «семнадцатого», 26 → «двадцать шестого»."""
    if n < 20:
        return _ORD_GEN_RU[n]
    t, u = _TENS_RU[n // 10 * 10]
    return u if n % 10 == 0 else t + " " + _ORD_GEN_RU[n % 10]


def dates(text: str, lang: str) -> str:
    """«17.10» → «семнадцатого октября» / «он жетінші қазан»; «05.11.2026» — с годом. Не дата (32.13) — как было."""
    def one(m: re.Match) -> str:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m.group(0)
        if lang == "kk":
            out = _ordinal(num_kk(d)) + " " + _MONTHS_KK[mo - 1]
            return (_ordinal(num_kk(int(y))) + " жылғы " + out) if y else out
        out = _ord_gen_ru(d) + " " + _MONTHS_RU[mo - 1]
        if y and 2000 < int(y) < 2100:
            out += " две тысячи " + _ord_gen_ru(int(y) - 2000) + " года"
        return out
    text = _DATE.sub(one, text)
    return _DOT_TIME.sub(r"\1:\2", text)            # не дата: «9.00-де», «10.30» — время через точку → через двоеточие


_DOT_TIME = re.compile(r"(?<![\d.,])([01]?\d|2[0-3])\.([0-5]\d)(?![\d]|\.\d)")

_KK_TIME_SUF = re.compile(r"(?<![\d])(\d{1,2}):(\d{2})-([а-яәғқңөұүһі]{1,4})(?![\w])", re.I)


def _harmony(word: str, suf: str) -> str:
    """Окончание местного/дательного/исходного падежа под произносимое слово: «тоғыз» + «де» → «тоғызда», «он бес» + «да» → «он бесте»."""
    w, sl = word.lower(), suf.lower()
    back = next((ch in "аоұы" for ch in reversed(w) if ch in "аоұыәөүіеиэу"), True)
    last = w[-1]
    voiceless, nasal = last in "пфкқтсшщхцч", last in "мнң"
    if sl in ("да", "де", "та", "те"):
        return ("т" if voiceless else "д") + ("а" if back else "е")
    if sl in ("ға", "ге", "қа", "ке"):
        return ("қа" if back else "ке") if voiceless else ("ға" if back else "ге")
    if sl in ("дан", "ден", "тан", "тен", "нан", "нен"):
        return ("т" if voiceless else "н" if nasal else "д") + ("ан" if back else "ен")
    return suf


def _kk_time(text: str) -> str:
    """«9:00-де» → «тоғызда» (numbers_kk приклеил бы окончание как написано: «тоғызде»)."""
    def one(m):
        h, mm = int(m.group(1)), m.group(2)
        w = num_kk(h) + ("" if mm == "00" else " " + num_kk(int(mm)))
        return w + _harmony(w.split()[-1], m.group(3))
    return _KK_TIME_SUF.sub(one, text)


# ---------------------------------------------------------------- бренд с казахским окончанием через дефис: «WhatsApp-қа» → «уатсапқа»
_KK_SUFFIX = ("қа|ке|ға|ге|та|те|да|де|на|не|дан|ден|тан|тен|нан|нен|ның|нің|дың|дің|тың|тің|ны|ні|ды|ді|ты|ті|мен|бен|пен|"
              "дағы|дегі|тағы|тегі|ы|і|сы|сі")


def numbers_ru(text: str) -> str:
    """Цифры → русские слова: «15 000 ₸» → «пятнадцать тысяч тенге», «50%» → «пятьдесят процентов», «10:30» → «десять тридцать»,
    «+7 701 234 56 78» → «плюс семь, семьсот один, …». Окончание через дефис («2-го») отбрасывается: падеж не угадываем."""
    t = _PHONE.sub(lambda m: ("плюс семь" if m.group(1) == "+7" else "восемь") + ", "
                   + ", ".join(_digits_ru(g) for g in m.groups()[1:]), text)
    t = _TIME.sub(lambda m: num_ru(int(m.group(1))) + ("" if m.group(2) == "00" else " " + _digits_ru(m.group(2))), t)

    def one(m: re.Match) -> str:
        n = int(re.sub(r"\D", "", m.group(1)))
        unit = (m.group(2) or "").strip()
        tail = {"%": " " + plural_ru(n, ("процент", "процента", "процентов")), "₸": " тенге", "": ""}[unit]
        return num_ru(n) + tail
    return _NUM_RU.sub(one, t)


def _phone_kk(text: str) -> str:   # «8 707 …» — numbers_kk знает только «+7»
    return re.sub(r"(?<!\d)8([\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2})(?!\d)",
                  lambda m: "сегіз, " + numbers_kk("+7" + m.group(1)).replace("плюс жеті, ", "", 1), text)


# ---------------------------------------------------------------- словарь, ссылки, язык
def load_lexicon(path: Path | None = None) -> list[tuple[re.Pattern, str]]:
    p = path or HERE / "lexicon.json"
    try:
        lex = json.loads(Path(p).read_text())["words"]
    except (OSError, ValueError, KeyError):
        return []
    items = sorted(lex.items(), key=lambda kv: -len(kv[0]))          # длинные сначала: «Kaspi.kz» раньше «Kaspi»
    # 16.10: казахское окончание через дефис прирастает к слову («WhatsApp-қа» → «уатсапқа»); русское «WhatsApp-чат» не трогаем
    return [(re.compile(r"(?<![\w])" + re.escape(k) + r"(?:-(" + _KK_SUFFIX + r"))?(?![\w])", re.I),
             (lambda m, v=v: v + (m.group(1) or ""))) for k, v in items]


LEXICON = load_lexicon()
_URL = re.compile(r"https?://\S+|www\.\S+|\b[\w-]+\.(?:kz|com|ru|net|org|io|me|app)\b(?:/\S*)?", re.I)
# почту бот проговаривает словами сам; проскочил «@» — читаем, а не вырезаем: «info собака example точка кей-зет»
_EMAIL = re.compile(r"(?<![\w.+-])([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)(?![\w-])")
_TLD = {"kz": "кей-зет", "com": "ком", "ru": "ру", "net": "нет", "org": "орг", "io": "ай-оу", "me": "ми", "app": "апп", "ua": "юэй"}


def _email(m: re.Match) -> str:
    local = m.group(1).replace(".", " точка ")
    parts = m.group(2).split(".")
    dom = " точка ".join(parts[:-1] + [_TLD.get(parts[-1].lower(), parts[-1])])
    return f"{local} собака {dom}"
_BULLET = re.compile(r"^\s*(?:[-•*·–—]+|\d{1,2}[.)])\s+", re.M)
_SENT = re.compile(r"(?<=[.!?…])\s+")
_TG = re.compile(r"(\d)\s*(?:тг|тнг|тенге|теңге)(?![\w])", re.I)     # точку не трогаем: она может быть концом предложения


_GREET_KK = re.compile(r"(?<![\w])(Сәлеметсіз|Сәлеметсіздер)(\s+)(бе|ме)!", re.I)
_TAG_DA = re.compile(r"(?<=[^\s,])\s+(да)\?", re.I)


def prosody_kk(s: str) -> str:
    """Знаки под живую мелодию (слепая прослушка 11.10, tricks.html): «Сәлеметсіз бе!» с «!» модель роняет и звучит роботом —
    с точкой лучше 9 из 10 пар; хвост «… да?» люди говорят после паузы — ставим запятую. «?» → «.» НЕ делаем (3:3, «не звучит как вопрос»)."""
    s = _GREET_KK.sub(r"\1\2\3.", s)
    return _TAG_DA.sub(r", \1?", s)


def lang_of(s: str, fallback: str = "ru") -> str:
    if any(ch in KK_LETTERS for ch in s):
        return "kk"
    return "ru" if CYR.search(s) else fallback


def _lines_to_sentences(text: str) -> str:
    text = _BULLET.sub("", text.replace("\r", ""))
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if not re.search(r"[.!?…:;,]$", line):
            line += "."
        out.append(line)
    return " ".join(out)


def speakable(text: str) -> str:
    """Весь текст → одна «произносимая» строка (без цифр, ссылок, скобок, латинских брендов из словаря)."""
    t = _URL.sub(" ", _EMAIL.sub(_email, text or ""))
    t = _lines_to_sentences(t)
    for rx, v in LEXICON:
        t = rx.sub(v, t)
    t = _TG.sub(r"\1 ₸", t)
    whole = lang_of(t, "ru")
    parts = []
    for piece in re.split(r"(?<=[.!?…])\s+", t):
        lg = lang_of(piece, whole)
        piece = dates(piece, lg)
        if lg == "kk":
            piece = prosody_kk(piece)
            piece = numbers_kk(_phone_kk(_kk_time(piece)))
            piece = re.sub(r"\d+", lambda m: num_kk(int(m.group(0))), piece)
        else:
            piece = numbers_ru(piece)
            piece = re.sub(r"\d+", lambda m: num_ru(int(m.group(0))), piece)
        parts.append(piece)
    t = " ".join(parts)
    t = re.sub(r"[%₸$€№@&§©®™°+]", " ", t)
    t, _ = normalize(t)
    t = re.sub(r"\s*[:;]\s*([.!?])", r"\1", t)          # «Сайт: .» после вырезанной ссылки
    t = re.sub(r"([.!?,])(?:\s*[.,])+", r"\1", t)
    t = re.sub(r"\s*[:;,]\s*$", ".", t.strip())                 # «Сайт:» в конце после вырезанной ссылки
    return t.strip()


def _split_long(s: str, limit: int) -> list[str]:
    if len(s) <= limit:
        return [s]
    mid = len(s) // 2
    cuts = [m.end() for m in re.finditer(r"[,;:]\s+", s)] or [m.end() for m in re.finditer(r"\s+", s)]
    if not cuts:
        return [s]
    cut = min(cuts, key=lambda c: abs(c - mid))
    return _split_long(s[:cut].strip(), limit) + _split_long(s[cut:].strip(), limit)


def pack(sentences: list[str], max_chars: int = MAX_CHARS, min_chars: int = MIN_CHARS) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        for part in _split_long(s, max_chars + 40):
            if not cur:
                cur = part
            elif len(cur) + 1 + len(part) <= max_chars or len(cur) < min_chars:
                cur += " " + part
            else:
                chunks.append(cur)
                cur = part
    if cur:
        if chunks and len(cur) < min_chars and len(chunks[-1]) + 1 + len(cur) <= max_chars + 60:
            chunks[-1] += " " + cur
        else:
            chunks.append(cur)
    return chunks


def prepare(text: str, max_chars: int = MAX_CHARS) -> list[dict]:
    t = speakable(text)
    if not CYR.search(t) and not re.search(r"[A-Za-z]", t):
        return []
    sents = [s.strip() for s in _SENT.split(t) if s.strip()]
    whole = lang_of(t, "ru")
    return [dict(text=c, lang=lang_of(c, whole)) for c in pack(sents, max_chars)]


if __name__ == "__main__":
    bad = 0

    def check(name, got, want):
        global bad
        ok = got == want
        bad += not ok
        print("ok " if ok else "BAD", name, "→", repr(got), "" if ok else f"(ждали {want!r})")

    check("ru 15000", numbers_ru("15 000 ₸"), "пятнадцать тысяч тенге")
    check("ru 2001", num_ru(2001), "две тысячи один")
    check("ru 21%", numbers_ru("21%"), "двадцать один процент")
    check("ru 3%", numbers_ru("3%"), "три процента")
    check("ru 50%", numbers_ru("50%"), "пятьдесят процентов")
    check("ru time", numbers_ru("в 10:30 и 9:00"), "в десять тридцать и девять")
    check("ru time 05", numbers_ru("18:05"), "восемнадцать ноль пять")
    check("ru phone", numbers_ru("+7 701 234 56 78"), "плюс семь, семьсот один, двести тридцать четыре, пятьдесят шесть, семьдесят восемь")
    check("ru phone 8", numbers_ru("8 (700) 000-00-07"), "восемь, семьсот, ноль ноль ноль, ноль ноль, ноль семь")
    check("ru 1 000 000", num_ru(1_000_000), "один миллион")
    check("kk phone 8", _phone_kk("8 707 123 45 67"), "сегіз, жеті жүз жеті, жүз жиырма үш, қырық бес, алпыс жеті")
    s = speakable("Сәлеметсіз бе! Шебер ертең 10:30-да келеді, бағасы 15 000 тг, Kaspi-де төлеуге болады. Сайт: https://example.kz")
    check("kk speakable", s, "Сәлеметсіз бе. Шебер ертең он отызда келеді, бағасы он бес мың теңге, Каспиде төлеуге болады. Сайт.")
    check("kk бренд + окончание", speakable("Клиенттеріңіз WhatsApp-қа жазғанда, AI-менеджер жауап береді."),
          "Клиенттеріңіз уатсапқа жазғанда, эй-ай-менеджер жауап береді.")
    check("ru бренд-составное не склеиваем", speakable("Напишите в WhatsApp-чат."), "Напишите в уатсап-чат.")
    check("ru дата", speakable("Записала вас на 15:30, завтра 17.10. Мастер перезвонит."),
          "Записала вас на пятнадцать тридцать, завтра семнадцатого октября. Мастер перезвонит.")
    check("ru дата с годом", speakable("Договор до 05.11.2026."), "Договор до пятого ноября две тысячи двадцать шестого года.")
    check("kk дата", speakable("Шебер 21.10 келеді."), "Шебер жиырма бірінші қазан келеді.")
    check("kk время через точку", speakable("Ертеңге 9.00-де жазып қойдым."), "Ертеңге тоғызда жазып қойдым.")
    check("ru время через точку", speakable("Приедем в 10.30."), "Приедем в десять тридцать.")
    check("ru почта с @", speakable("Пришлю на info@example.kz, проверьте."), "Пришлю на info собака example точка кей-зет, проверьте.")
    check("сайт по-прежнему не читаем", speakable("Сайт: https://example.kz"), "Сайт.")
    check("kk время + окончание", speakable("Шебер 15:00-та, 10:30-да немесе 9:00-ге дейін келеді."),
          "Шебер он бесте, он отызда немесе тоғызға дейін келеді.")
    check("kk приветствие", speakable("Сәлеметсіз бе! Тыңдап тұрмын."), "Сәлеметсіз бе. Тыңдап тұрмын.")
    check("kk хвост да", speakable("Бізде қазір сондай болмай тұр да?"), "Бізде қазір сондай болмай тұр, да?")
    check("kk хвост да с запятой", speakable("Ертең келесіз, да?"), "Ертең келесіз, да?")
    check("kk вопрос не трогаем", speakable("Төлемді картамен жасайсыз ба, қолма-қол ма?"), "Төлемді картамен жасайсыз ба, қолма-қол ма?")
    s = speakable("(весело) Здравствуйте! Я AI-менеджер, пишите в **WhatsApp** 😊\n- Цена: 5 000 ₸\n- Мастер приедет в 14:00")
    check("ru speakable", s, "Здравствуйте! Я эй-ай-менеджер, пишите в уатсап. Цена: пять тысяч тенге. Мастер приедет в четырнадцать.")
    ch = prepare("Иә. " + "Шебер ертең таңертең келеді, жұмыс шамамен екі сағатқа созылады. " * 4 + "Рахмет!")
    check("pack: кусков", len(ch), 2)
    check("pack: «Иә.» не одна", ch[0]["text"].startswith("Иә. Шебер"), True)
    check("pack: «Рахмет!» приклеено", ch[-1]["text"].endswith("Рахмет!"), True)
    check("pack: ≤ лимита", all(len(c["text"]) <= MAX_CHARS + 60 for c in ch), True)
    check("pack: язык kk", {c["lang"] for c in ch}, {"kk"})
    long = "Бұл өте ұзақ сөйлем, " * 20 + "соңы."
    check("long split", all(len(c["text"]) <= MAX_CHARS + 60 for c in prepare(long)), True)
    check("empty", prepare("😊 https://x.kz"), [])
    mixed = prepare("Сәлеметсіз бе! Стоимость 12 000 ₸, мастер приедет в 9:00.")
    check("mixed: нет цифр", any(re.search(r"\d", c["text"]) for c in mixed), False)
    check("mixed: ru число", "двенадцать тысяч тенге" in mixed[0]["text"], True)
    check("no brackets", any("(" in c["text"] for c in prepare("(шёпотом)Привет (смеётся) как дела")), False)
    raise SystemExit(bad)
