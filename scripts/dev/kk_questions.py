#!/usr/bin/env python3
"""Вопросы в живой казахской речи (11.10.2026): формы, частоты, примеры.
Вход — тексты с пунктуацией kazakh-whisper-turbo (по умолчанию v1-поднабор voxcpm2/data/ours-flac/{train,val}.jsonl, 7,9 ч, 98 дикторов;
другие манифесты — KK_TEXTS=a.jsonl,b.jsonl).
  python3 scripts/dev/kk_questions.py [--json out.json]
Без моделей: только текст. «?» ставил Whisper (по звуку и смыслу) — это и есть разметка «вопрос» (вместе с ошибками Whisper).
"""
import collections, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = [Path(p) for p in os.environ.get("KK_TEXTS", "").split(",") if p] or \
      [ROOT / "voxcpm2/data/ours-flac/train.jsonl", ROOT / "voxcpm2/data/ours-flac/val.jsonl"]
PART = r"(?:ма|ме|ба|бе|па|пе)"
TAILS = r"(?:енді|ә|иә|еді|екен|ғой|деймін|шы|ші)"
WH = ["қалай", "қалайша", "қайда", "қайдан", "қайдағы", "қашан", "кім", "кімге", "кімді", "кімнің", "кімдер", "не", "неге", "несі", "нені",
      "неде", "немене", "неліктен", "неше", "нешеде", "нешінші", "қандай", "қанша", "қаншама", "қай", "қайсы", "қайсысы", "нелер", "нелерді",
      "нені", "несімен", "ше"]
WH_MULTI = ["не үшін", "не себепті", "не себептен", "не істеп", "не болды", "не болып", "не деген", "не ғып", "не қылып"]
TAG_END = [r"ғой", r"да", r"де", r"иә", r"ә", r"рас", r"солай\s+ма", r"солай\s+ғой", r"емес\s+пе", r"дұрыс\s+па", r"шығар", r"ия", r"а"]
FILL = re.compile(r"^(ы+|м+|ммм|ыыы|ііі|э+|енді|жаңағы|мысалы|яғни|ал|а|и|иә|мхм|мм|ну|вот|сонымен|негізі|қазір|сонда)$", re.I)
RU = set("уже если что потому даже то есть оказывается давай брат но чтобы выше можете какой ценой ответственности незаменима вымышленный "
         "персонаж дальше выбор подругам подруга на как так это просто короче типа вообще кстати значит например пожалуйста "
         "сколько можно хорошо ладно ок окей скажете насчет первых пяти минуты рейтинг удобно наверное конечно серьезно реально прям "
         "братан братцы сразу если что потому опаздываю должна доехать можете подписать".split())
KKX = re.compile(r"[әғқңөұүһі]", re.I)


def words(s):
    return re.findall(r"[а-яёәғқңөұүһіА-ЯЁӘҒҚҢӨҰҮҺІa-zA-Z\-]+", s)


def classify(q):
    """q — вопросительное предложение (с «?»). → тип и признаки."""
    s = q.strip()
    core = re.sub(r"[?.!…\s]+$", "", s)
    low = core.lower()
    w = words(low)
    f = {}
    f["n_words"] = len(w)
    m_tag = next((t for t in TAG_END if re.search(rf"(^|[\s,])({t})$", low)), None)
    m_part_end = re.search(rf"(^|[\s\-]){PART}(\s+{TAILS})*$", low)
    m_part_any = re.search(rf"(^|[\s\-]){PART}([\s,]|$)", low)
    wh_hit = [x for x in WH if re.search(rf"(^|[\s,]){x}([\s,]|$)", low)] + [x for x in WH_MULTI if x in low]
    alt = bool(re.search(rf"{PART}[\s,]+(жоқ[\s,]+)?(әлде|немесе|или)\b", low)) or bool("әлде" in low and m_part_any)
    f["alt"] = alt
    f["wh"] = wh_hit
    # тип
    if alt:
        t = "альтернатива (…ма, әлде …ма)"
    elif m_part_end:
        t = "частица в конце (…ма/бе?)"
    elif m_tag and m_tag not in ("а",):
        t = "хвост-переспрос (…ғой? / да? / иә? / емес пе?)"
    elif wh_hit:
        t = "вопросительное слово (неге/қалай/не/қанша…)"
    elif m_part_any:
        t = "частица в середине (…ма, + хвост)"
    else:
        t = "только интонация (без частицы и слова)"
    f["type"] = t
    f["short"] = len(w) <= 2
    # вопрос в конце длинной фразы: слов до последней запятой
    parts = re.split(r"[,—:]", core)
    f["pre_words"] = len(words(",".join(parts[:-1]))) if len(parts) > 1 else 0
    f["fillers"] = [x for x in w if FILL.match(x)]
    f["ru"] = [x for x in w if x in RU and not KKX.search(x)]
    f["latin"] = [x for x in w if re.match(r"[a-z]", x)]
    f["prog"] = re.findall(r"\w+(?:ватыр\w*|ватқан\w*|ват\b|ватсың\w*|ватыр|пжатыр\w*|ғып|қып)", low)
    f["tag"] = m_tag
    f["wh_pos"] = ("начало" if w and w[0] in wh_hit else "конец" if w and w[-1] in wh_hit else "середина") if wh_hit else None
    return f


def main():
    R = []
    for p in SRC:
        R += [json.loads(l) for l in open(p, encoding="utf-8")]
    allsent, qs = 0, []
    for i, r in enumerate(R):
        sents = [x.strip() for x in re.findall(r"[^.!?…]+[.!?…]*", r["text"]) if x.strip()]
        allsent += len(sents)
        for k, s in enumerate(sents):
            if "?" in s:
                qs.append(dict(row=i, k=k, last=k == len(sents) - 1, first=k == 0, text=s, audio=r["audio"], dur=r["duration"],
                               spk=r["speaker"], **classify(s)))
    n = len(qs)
    print(f"строк {len(R)}, предложений {allsent}, вопросов «?» {n} ({n / allsent * 100:.1f} %); вопрос в конце клипа {sum(q['last'] for q in qs)}")
    ty = collections.Counter(q["type"] for q in qs)
    for k, v in ty.most_common():
        print(f"  {v:4} ({v / n * 100:4.1f} %)  {k}")
    print(f"короткие (≤ 2 слова) {sum(q['short'] for q in qs)}; с заполнителями {sum(bool(q['fillers']) for q in qs)}; "
          f"с русскими словами {sum(bool(q['ru']) for q in qs)}; с «предисловием» ≥ 5 слов до последней запятой {sum(q['pre_words'] >= 5 for q in qs)}")
    lens = sorted(q["n_words"] for q in qs)
    print(f"длина, слов: медиана {lens[len(lens) // 2]}, p25 {lens[len(lens) // 4]}, p75 {lens[3 * len(lens) // 4]}")
    wh = collections.Counter(x for q in qs for x in q["wh"])
    print("вопросительные слова: " + ", ".join(f"{k} {v}" for k, v in wh.most_common(20)))
    whp = collections.Counter(q["wh_pos"] for q in qs if q["wh_pos"])
    print("позиция вопр. слова: " + ", ".join(f"{k} {v}" for k, v in whp.most_common()))
    tags = collections.Counter(q["tag"] for q in qs if q["tag"])
    print("хвосты: " + ", ".join(f"{k} {v}" for k, v in tags.most_common()))
    fill = collections.Counter(x for q in qs for x in q["fillers"])
    print("заполнители: " + ", ".join(f"{k} {v}" for k, v in fill.most_common(12)))
    ru = collections.Counter(x for q in qs for x in q["ru"])
    print("русские слова: " + ", ".join(f"{k} {v}" for k, v in ru.most_common(15)))
    parts = collections.Counter(m.group(1) for q in qs for m in re.finditer(rf"(?:^|[\s\-])({PART})(?=[\s,?]|$)", q["text"].lower()))
    print("частицы: " + ", ".join(f"{k} {v}" for k, v in parts.most_common()))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(qs, ensure_ascii=False, indent=0), encoding="utf-8")
    if "--examples" in sys.argv:
        for t in ty:
            print(f"\n== {t}")
            for q in [q for q in qs if q["type"] == t][:14]:
                print(f"   [{q['n_words']}] {q['text'][:150]}")


if __name__ == "__main__":
    main()
