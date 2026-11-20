"""Нормализатор текста для обучения (и потом для бота): одна функция на все источники.

normalize(text) -> (clean_text, problems)
  clean_text — текст «как сказано»: без скобок/кавычек/markdown/эмодзи, без пробела перед знаком, гомоглифы исправлены,
               КАПС-слова строчными, многоточия и тире упрощены;
  problems   — список причин, по которым строку в обучение НЕ брать: digits (цифры — читаются как попало),
               latin (латиница: бренды/сокращения без словаря), empty, symbols (₸ % № @ # и т.п.).
Числа словами здесь не раскрываем: для обучения строки с цифрами выбрасываем (их мало), для бота — отдельный
слой с падежами (на стороне приложения), чтобы не учить модель на кривых раскрытиях.
Скобки в начале строки у VoxCPM2 = команда Voice Design → в тексте обучения их быть не должно вообще.
"""
from __future__ import annotations

import re
import unicodedata

# латинские буквы, похожие на кириллицу, внутри кириллического слова → кириллица
_HOMO = str.maketrans({"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у", "i": "і", "h": "һ",
                       "A": "А", "C": "С", "E": "Е", "O": "О", "P": "Р", "X": "Х", "B": "В", "H": "Н", "K": "К",
                       "M": "М", "T": "Т", "I": "І",
                       "ä": "ә", "ö": "ө", "ü": "ү", "ñ": "ң", "Ä": "Ә", "Ö": "Ө", "Ü": "Ү", "Ñ": "Ң"})
_CYR = "а-яёәғқңөұүһіА-ЯЁӘҒҚҢӨҰҮҺІ"
_MIXED_WORD = re.compile(rf"\b(?=\w*[{_CYR}])(?=\w*[A-Za-zäöüñÄÖÜÑ])\w+\b")
_BRACKETS = re.compile(r"[\(\[\{（][^\)\]\}）]*[\)\]\}）]")
_DROP_CHARS = re.compile(r"[\"«»„“”‘’`*_#~^<>|\\/=+\[\]{}()（）]")
_SYMBOLS = re.compile(r"[%₸$€№@&§©®™°]")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:…])")
_CAPS_WORD = re.compile(r"\b[А-ЯЁӘҒҚҢӨҰҮҺІ]{3,}\b")


def _fix_mixed(m: re.Match) -> str:
    return m.group(0).translate(_HOMO)


def normalize(text: str) -> tuple[str, list[str]]:
    probs: list[str] = []
    t = unicodedata.normalize("NFC", text or "")
    t = _EMOJI.sub(" ", t)
    t = _BRACKETS.sub(" ", t)                      # (смех), [музыка], (живой разговор) — вон целиком
    t = _MIXED_WORD.sub(_fix_mixed, t)             # «сәлем» с латинской «e» → кириллица
    t = t.replace("…", "...").replace("—", ",").replace("–", ",").replace(" - ", ", ")
    t = _DROP_CHARS.sub(" ", t)
    t = re.sub(r"\.{2,}", ".", t)
    t = re.sub(r"([,!?;:])\1+", r"\1", t)
    t = _CAPS_WORD.sub(lambda m: m.group(0).lower(), t)  # КАПС читается по буквам → обычный регистр
    t = re.sub(r"\s+", " ", t).strip()
    t = _SPACE_BEFORE_PUNCT.sub(r"\1", t)
    t = re.sub(r"^[,.;:!?\s]+", "", t)
    t = re.sub(r",\s*([.!?])", r"\1", t)
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    t = re.sub(r"([.!?]\s+)(\w)", lambda m: m.group(1) + m.group(2).upper(), t)   # заглавная после конца предложения
    if not re.search(rf"[{_CYR}]", t):
        probs.append("empty")
    if re.search(r"\d", t):
        probs.append("digits")
    if re.search(r"[A-Za-z]", t):
        probs.append("latin")
    if _SYMBOLS.search(t):
        probs.append("symbols")
    return t, probs


FILLER_RE = re.compile(r"(?<!\w)(э+|ээ+|эм+|м+|мм+|хм+|ы+|әә+|аа+)(?!\w)", re.I)


def count_fillers(text: str) -> int:
    return len(FILLER_RE.findall(text or ""))


# ---------------------------------------------------------------- числа словами (казахский) — 09.10
# Для набора оценки phrases_v2 и для бота (тот же слой): «15 000 ₸» → «он бес мың теңге», «10-да» → «онда» (окончание пишет автор,
# оно уже согласовано с произношением числа), «2-ші» → «екінші», «+7 700 123 45 67» → «плюс жеті, жеті жүз, …», «50%» →
# «елу пайыз», «10:30» → «он отыз». Даты и падежи без дефиса не угадываем — такие места автор пишет словами.
_ONES = ["", "бір", "екі", "үш", "төрт", "бес", "алты", "жеті", "сегіз", "тоғыз"]
_TENS = ["", "он", "жиырма", "отыз", "қырық", "елу", "алпыс", "жетпіс", "сексен", "тоқсан"]
_BIG = [(10 ** 9, "миллиард"), (10 ** 6, "миллион"), (1000, "мың")]
_BACK, _FRONT = set("аоұы"), set("әөүіеиу")


def _u1000(n: int) -> list[str]:
    out = []
    h, r = divmod(n, 100)
    if h:
        out += (["жүз"] if h == 1 else [_ONES[h], "жүз"])
    t, o = divmod(r, 10)
    if t:
        out.append(_TENS[t])
    if o:
        out.append(_ONES[o])
    return out


def num_kk(n: int) -> str:
    """Количественное число словами: 2026 → «екі мың жиырма алты», 1500 → «мың бес жүз», 0 → «нөл»."""
    if n == 0:
        return "нөл"
    out = []
    for v, w in _BIG:
        k, n = divmod(n, v)
        if k:
            out += ([w] if (k == 1 and v == 1000) else _u1000(k) + [w])
    out += _u1000(n)
    return " ".join(out)


def _ordinal(words: str) -> str:
    w = words.split()
    last = w[-1]
    if last == "қырық":
        last = "қырқ"
    back = next((ch in _BACK for ch in reversed(last) if ch in _BACK | _FRONT), True)
    if last[-1] in "ыі":            # екі → екінші, алты → алтыншы, жеті → жетінші
        last += "ншы" if back else "нші"
    elif last[-1] in "аә":          # жиырма → жиырмасыншы
        last += "сыншы" if back else "сінші"
    else:                           # бір → бірінші, елу → елуінші, он → оныншы, қырық → қырқыншы
        last += "ыншы" if back else "інші"
    return " ".join(w[:-1] + [last])


def _digits_kk(g: str) -> str:
    """Группа цифр телефона: ведущие нули по одному («05» → «нөл бес»), дальше числом."""
    lead = len(g) - len(g.lstrip("0"))
    rest = g[lead:]
    return " ".join(["нөл"] * lead + ([num_kk(int(rest))] if rest else []))


_PHONE = re.compile(r"\+7[\s-]?\(?(\d{3})\)?[\s-]?(\d{3})[\s-]?(\d{2})[\s-]?(\d{2})")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b(?:-([а-яёәғқңөұүһі]+))?")
_NUM = re.compile(r"(?<![\w-])(\d{1,3}(?:[   ]\d{3})+|\d+)(?:-(ші|шы|нші|ншы|інші|ыншы)\b|-([а-яёәғқңөұүһі]+))?(\s*(?:%|₸))?")


def numbers_kk(text: str) -> str:
    """Цифры → казахские слова (см. выше). Что не разобрано — останется цифрами и normalize() пометит digits."""
    t = _PHONE.sub(lambda m: "плюс жеті, " + ", ".join(_digits_kk(g) for g in m.groups()), text)
    t = _TIME.sub(lambda m: num_kk(int(m.group(1))) + ("" if m.group(2) == "00" else " " + num_kk(int(m.group(2))))
                  + (m.group(3) or ""), t)

    def one(m: re.Match) -> str:
        w = num_kk(int(re.sub(r"\D", "", m.group(1))))
        if m.group(2):
            w = _ordinal(w)
        elif m.group(3):
            w = w + m.group(3)
        unit = (m.group(4) or "").strip()
        return w + {"%": " пайыз", "₸": " теңге", "": ""}[unit]
    return _NUM.sub(one, t)


if __name__ == "__main__":  # мини-тесты: python dataprep/textnorm.py
    cases = [
        ("Сәлеметсіз бе ! Мен (күледі) Аружанмын .", "Сәлеметсіз бе! Мен Аружанмын.", []),
        ("сәлем, қалайсыз?", "Сәлем, қалайсыз?", []),
        ("Бағасы 15 мың теңге", "Бағасы 15 мың теңге", ["digits"]),
        ("(молодая девушка)Добрый день", "Добрый день", []),
        ("Ceнiң үйің", "Сенің үйің", []),
        ("Бұл «өте» жақсы — иә...", "Бұл өте жақсы, иә.", []),
        ("WhatsApp арқылы", "WhatsApp арқылы", ["latin"]),
        ("ЖАҚСЫ болды", "Жақсы болды", []),
        ("50% жеңілдік", "50% жеңілдік", ["digits", "symbols"]),
        ("10 ₸", "10 ₸", ["digits", "symbols"]),
        ("болды. сондықтан иә", "Болды. Сондықтан иә", []),
    ]
    bad = 0
    for src, want, wprob in cases:
        got, pr = normalize(src)
        ok = got == want and set(pr) >= set(wprob)
        bad += not ok
        print("ok " if ok else "BAD", repr(src), "→", repr(got), pr)
    print("fillers:", count_fillers("Ээ, ну, мм, жаңағы, эм иә"))
    ncases = [("Ертең сағат 10-да келеді", "Ертең сағат онда келеді"), ("15 000 ₸", "он бес мың теңге"),
              ("2026 жылы", "екі мың жиырма алты жылы"), ("1500", "мың бес жүз"), ("2-ші қабат", "екінші қабат"),
              ("40-шы үй", "қырқыншы үй"), ("6-шы", "алтыншы"), ("20-шы", "жиырмасыншы"), ("50-ші", "елуінші"), ("7-ші", "жетінші"),
              ("1-ші", "бірінші"), ("10-шы", "оныншы"), ("50%", "елу пайыз"),
              ("10:30", "он отыз"), ("09:00-дан 18:00-ге", "тоғыздан он сегізге"), ("+7 700 123 45 67", "плюс жеті, жеті жүз, жүз жиырма үш, қырық бес, алпыс жеті"),
              ("+7 700 005 00 07", "плюс жеті, жеті жүз, нөл нөл бес, нөл нөл, нөл жеті"), ("100", "жүз"), ("101", "жүз бір"),
              ("15-ке дейін", "он беске дейін"), ("3 бөлме", "үш бөлме")]
    for src, want in ncases:
        got = numbers_kk(src)
        bad += got != want
        print("ok " if got == want else "BAD", repr(src), "→", repr(got))
    raise SystemExit(bad)
