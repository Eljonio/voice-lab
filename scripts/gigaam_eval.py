#!/usr/bin/env python3
"""Независимый оценщик озвучки: GigaAM-Multilingual Large CTC (ai-sage, MIT) — не та модель, что размечала данные.

CTC без языковой модели и без подсказки: пишет то, что слышит, не «исправляет» акцент, как Whisper.
Запуск — в оверлей-venv (torch/transformers из .venv + hydra-core):
  .venv-asr/bin/python scripts/gigaam_eval.py <режим> ...

Режимы:
  sets  NAME=DIR [NAME=DIR ...] [--phrases eval/phrases.json] [--out eval/results/X.json]
        DIR — папка с <id>.wav. Тексты: из --phrases (id → text, lang) или из DIR/report.json (формат synth_baseline:
        {tag: {file, text}}). Язык тега: ru*, mix*, остальное kk.
  manifest FILE.jsonl|FILE.tsv [--lang kk] [--limit N] [--out ...]
        jsonl: {"audio", "text", "lang"?}; tsv FLEURS: id<TAB>file<TAB>raw<TAB>norm..., аудио рядом в audio/dev/.

Метрики на фразу: CER/WER после нормализации (строчные, ё→е, без знаков, бренды из EVAL_SKIP_RE вырезаны), длительность,
символов в секунду, «последнее слово на месте». Сводка по языку: среднее и медиана CER, доля провалов (CER > 10 %).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "ai-sage/GigaAM-Multilingual"
# ветка large_ctc, коммит проверен 06.10.2026 (код modeling_gigaam.py прочитан: сеть только huggingface_hub)
REVISION = os.environ.get("GIGAAM_REV", "3905cd51c3ed4e88c8edf33f3302969ba480a327")
FAIL = 0.10

# названия брендов в репликах бота не считаем: модель читает их по-разному, а ASR пишет латиницу кириллицей как попало.
# Регулярное выражение — из EVAL_SKIP_RE (например r"(acme\s*-?\s*corp|эк\w{0,3}\s*корп)"); по умолчанию ничего не вырезаем.
BRAND = re.compile(os.environ.get("EVAL_SKIP_RE") or r"(?!x)x", re.I)
# «AI-менеджер» (фразы samples): ASR пишет «эй ай / э ай / а ай / эайй / әи / эйай» — 1–2 слова только из этих букв
# перед «менеджер» убираем и в эталоне, и в ответе (только если в эталоне есть AI)
AI_REF = re.compile(r"\bAI\b")
BEFORE_MANAGER = re.compile(r"(?:\b[эеаәйиia]{1,4}\s+){1,2}(?=менеджер)")
PAREN = re.compile(r"^\s*[(（][^)）]*[)）]")  # команда Voice Design в начале текста


def norm(t: str) -> str:
    t = PAREN.sub(" ", t)
    t = BRAND.sub(" ", t)
    t = t.lower().replace("ё", "е").replace("-", " ")
    t = re.sub(r"[^\w\s]|_|\d", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def edit(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    return edit(ref, hyp) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    return edit(r, hyp.split()) / max(1, len(r))


# Заменитель оценки акцента — как CTC слышит особые казахские буквы. Ошибка = буква эталона выровнена
# не с собой (замена или пропуск); «обрусела» = заменена русским аналогом (қ→к, ң→н, ө→о, ұ/ү→у, і→и/ы, ә→а, ғ→г, һ→х).
SPECIAL = {"ә": "а", "ғ": "г", "қ": "к", "ң": "н", "ө": "о", "ұ": "у", "ү": "у", "һ": "х", "і": "иы"}


def align(a: str, b: str) -> list[tuple[str | None, str | None]]:
    """Посимвольное выравнивание Левенштейна (эталон a, распознанное b) → пары (символ a | None, символ b | None)."""
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            out.append((a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            out.append((a[i - 1], None))
            i -= 1
        else:
            out.append((None, b[j - 1]))
            j -= 1
    return out[::-1]


def special_errors(ref: str, hyp: str) -> dict:
    """{sp_n: особых букв в эталоне, sp_err: из них распознано не так, sp_rus: из них — русским аналогом, by: по буквам}."""
    n = err = rus = 0
    by: dict[str, list[int]] = {}
    for x, y in align(ref, hyp):
        if x not in SPECIAL:
            continue
        n += 1
        st_ = by.setdefault(x, [0, 0, 0])
        st_[0] += 1
        if y != x:
            err += 1
            st_[1] += 1
            if y is not None and y in SPECIAL[x]:
                rus += 1
                st_[2] += 1
    return dict(sp_n=n, sp_err=err, sp_rus=rus, sp_by=by)


def last_word_ok(ref: str, hyp: str) -> bool:
    """Последнее слово эталона есть среди двух последних слов распознанного (допуск — 1/3 символов слова)."""
    r, h = ref.split(), hyp.split()
    if not r:
        return True
    if not h:
        return False
    w = r[-1]
    return any(edit(w, x) <= max(1, len(w) // 3) for x in h[-2:])


class ASR:
    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoModel
        self.torch = torch
        dev = device or ("cuda" if torch.cuda.is_available() and not os.environ.get("GIGAAM_CPU") else "cpu")
        t = time.time()
        self.m = AutoModel.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True).to(dev).eval()
        self.dev = dev
        print(f"GigaAM large_ctc на {dev}, загрузка {time.time() - t:.0f} с", file=sys.stderr, flush=True)

    def __call__(self, wav_path: str) -> tuple[str, float]:
        """(текст, длительность с). Длинные (> 25 с, «не остановилась») — целиком, без отказа."""
        inner = self.m.model
        with self.torch.inference_mode():
            wav, length = inner.prepare_wav(str(wav_path))
            enc, enc_len = inner.forward(wav, length)
            text = inner._decode(enc, enc_len, length)[0][0]
        return text.strip(), length.item() / 16000


def lang_of(tag: str, text: str = "") -> str:
    """Язык по тегу (ru, ru2, design_ru, mix, kk3, design_kk); тег без языка («design») — по буквам текста."""
    parts = re.split(r"[_/]", tag.split("/")[-1])
    for lang in ("ru", "mix", "kk"):
        if any(p.startswith(lang) for p in parts):
            return lang
    return "kk" if re.search(r"[әғқңөұүһі]", PAREN.sub("", text).lower()) else "ru"


def score(asr: ASR, items: list[dict]) -> dict:
    out = {}
    for it in items:
        hyp, dur = asr(it["audio"])
        r, h = norm(it["text"]), norm(hyp)
        if AI_REF.search(it["text"]):
            r, h = BEFORE_MANAGER.sub("", r), BEFORE_MANAGER.sub("", h)
        out[it["id"]] = dict(lang=it["lang"], text=it["text"], hyp=hyp, cer=round(cer(r, h), 4), wer=round(wer(r, h), 4),
                             dur=round(dur, 2), chars_per_s=round(len(r) / max(dur, 0.01), 1), last_ok=last_word_ok(r, h),
                             **special_errors(r, h))
    return out


def add_special(res: dict) -> dict:
    """Досчитать метрику особых букв по уже сохранённым hyp (старые json без sp_*): ASR не нужен."""
    for v in res.values():
        if "sp_n" not in v:
            r, h = norm(v["text"]), norm(v["hyp"])
            if AI_REF.search(v["text"]):
                r, h = BEFORE_MANAGER.sub("", r), BEFORE_MANAGER.sub("", h)
            v.update(special_errors(r, h))
    return res


def special_summary(xs: list[dict]) -> dict:
    n = sum(v.get("sp_n", 0) for v in xs)
    return dict(sp_n=n, sp_err=round(sum(v.get("sp_err", 0) for v in xs) / max(1, n), 4),
                sp_rus=round(sum(v.get("sp_rus", 0) for v in xs) / max(1, n), 4))


def summary(res: dict) -> dict:
    s = {}
    for lang in ("kk", "mix", "ru", "all"):
        xs = [v for v in res.values() if lang == "all" or v["lang"] == lang]
        if not xs:
            continue
        c = [v["cer"] for v in xs]
        s[lang] = dict(n=len(xs), cer_mean=round(statistics.mean(c), 4), cer_median=round(statistics.median(c), 4),
                       fail=round(sum(x > FAIL for x in c) / len(c), 3), wer_mean=round(statistics.mean(v["wer"] for v in xs), 4),
                       last_missing=sum(not v["last_ok"] for v in xs),
                       cps_median=round(statistics.median(v["chars_per_s"] for v in xs), 1), **special_summary(xs))
    return s


def items_from_dir(d: Path, phrases: dict | None) -> list[dict]:
    rep = d / "report.json"
    items = []
    if phrases:
        for pid, p in phrases.items():
            f = d / f"{pid}.wav"
            if f.exists():
                items.append(dict(id=pid, audio=str(f), text=p["text"], lang=p["lang"]))
    elif rep.exists():
        for tag, x in json.loads(rep.read_text()).items():
            if not isinstance(x, dict) or "text" not in x:
                continue
            f = d / Path(x.get("file", f"{tag}.wav")).name
            if f.exists():
                items.append(dict(id=tag, audio=str(f), text=x["text"], lang=lang_of(tag, x["text"])))
    return items


def items_from_manifest(fp: Path, lang: str, limit: int) -> list[dict]:
    items = []
    if fp.suffix == ".tsv":  # FLEURS: id, file, raw, norm, chars, n_samples, gender
        with fp.open() as f:
            for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                a = fp.parent / "audio" / "dev" / row[1]
                if a.exists() and not re.search(r"\d", row[2]):  # как в README GigaAM: эталоны с цифрами не берём
                    items.append(dict(id=row[1], audio=str(a), text=row[2], lang=lang))
    else:
        for i, line in enumerate(fp.open()):
            x = json.loads(line)
            items.append(dict(id=Path(x["audio"]).stem, audio=x["audio"], text=x["text"], lang=x.get("lang", lang)))
    return items[:limit] if limit else items


def table(name: str, s: dict) -> str:
    rows = [f"{name}:"]
    for lang, v in s.items():
        rows.append(f"  {lang:4} n={v['n']:3}  CER ср {v['cer_mean']*100:5.1f} %  мед {v['cer_median']*100:5.1f} %  "
                    f"провалы {v['fail']*100:4.0f} %  WER {v['wer_mean']*100:5.1f} %  нет посл. слова {v['last_missing']}  "
                    f"симв/с {v['cps_median']}  особые буквы: ошибок {v['sp_err']*100:4.1f} % (рус. аналог {v['sp_rus']*100:4.1f} %, n={v['sp_n']})")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sets", "manifest"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--phrases", type=Path, default=None)
    ap.add_argument("--lang", default="kk")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    asr = ASR(a.device)
    result = {"model": f"{MODEL_ID}@large_ctc:{REVISION[:8]}", "made": time.strftime("%Y-%m-%d %H:%M"), "sets": {}}
    if a.mode == "sets":
        ph = None
        if a.phrases:
            ph = {p["id"]: p for p in json.loads(a.phrases.read_text())["phrases"]}
        for spec in a.args:
            name, d = spec.split("=", 1) if "=" in spec else (Path(spec).name, spec)
            items = items_from_dir(Path(d), ph)
            if not items:
                print(f"{name}: нет файлов в {d}", file=sys.stderr)
                continue
            res = score(asr, items)
            s = summary(res)
            result["sets"][name] = dict(dir=str(d), summary=s, items=res)
            print(table(name, s), flush=True)
    else:
        items = items_from_manifest(Path(a.args[0]), a.lang, a.limit)
        t = time.time()
        res = score(asr, items)
        s = summary(res)
        result["sets"][Path(a.args[0]).stem] = dict(dir=a.args[0], summary=s, items=res)
        print(table(a.args[0], s), f"\n  {len(items)} файлов за {time.time() - t:.0f} с", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(f"→ {a.out}", file=sys.stderr)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()
