#!/usr/bin/env python3
"""call_bench.py — замеры скорости голоса для звонка за ОДНУ загрузку модели (запуск — на GPU).

Конфигурации (по порядку, итог пишется после каждой — обрыв по сроку не теряет сделанное):
  A    LoRA, целиком (model.generate, retry_badcase — как /tts у бота сейчас), 10 шагов   — база качества и скорости
  B    LoRA, поток (тот же путь, что POST /tts/stream: латенты эталона из кэша), 10 шагов — поток = целиком? TTFA
  M10 M8 M6 M4   влитые веса (LoRALinear → nn.Linear с W' из merge_lora.py), поток, 10/8/6/4 шагов
  M6c15          влитые, 6 шагов, cfg 1,5
  O10 O6         влитые + torch.compile (VoxCPM2Model.optimize: mode="reduce-overhead", CUDA graphs), поток — если хватает времени
Эталон — aruzhan_f_kk.wav (голос бота), режим «только эталон» (= кусок 1 в chain). Сид один на фразу (--seed).
На фразу: ttfa_model (до первого патча), ttfa (до первого слышимого звука после среза тишины), synth, звук, RTF,
нужный запас буфера у слушателя (prebuffer), «не остановилась». WAV 48 кГц — <out>/<конф>/<id>.wav (для GigaAM и прослушки).

  call_bench.py run  --weights W --lora L --merged M --ref voices/aruzhan_f_kk.wav --phrases eval/phrases_v2.json --out OUT [--deadline TS]
  call_bench.py report OUT [--gigaam OUT/gigaam.json]     → OUT/bench.txt (таблица: RTF, TTFA, запас, CER GigaAM kk, провалы)
  call_bench.py page OUT                                  → OUT/index.html (прослушка: фразы × конфигурации)
  --mock — без модели (заглушка tts_server), проверка обвязки на ноуте.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VOICE_HOME = Path(os.environ.get("VOICE_HOME", str(Path.home() / "voice")))

DEFAULT_IDS = ([f"kk_bot{i:02d}" for i in range(1, 13)] + ["kk_q01", "kk_q05", "kk_q09", "kk_q13", "kk_short03", "kk_short05",
               "kk_short08"])
ORDER = ["A", "B", "M10", "M8", "M6", "M4", "M6c15", "O10", "O6"]
SPEC = {"A": dict(kind="full", steps=10, cfg=2.0, w="lora"), "B": dict(kind="stream", steps=10, cfg=2.0, w="lora"),
        "M10": dict(kind="stream", steps=10, cfg=2.0, w="merged"), "M8": dict(kind="stream", steps=8, cfg=2.0, w="merged"),
        "M6": dict(kind="stream", steps=6, cfg=2.0, w="merged"), "M4": dict(kind="stream", steps=4, cfg=2.0, w="merged"),
        "M6c15": dict(kind="stream", steps=6, cfg=1.5, w="merged"),
        "O10": dict(kind="stream", steps=10, cfg=2.0, w="compiled"), "O6": dict(kind="stream", steps=6, cfg=2.0, w="compiled")}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pick_phrases(path: Path, ids: list[str], n_mix: int, n_ru: int) -> list[dict]:
    ph = json.loads(path.read_text())["phrases"]
    by = {p["id"]: p for p in ph}
    out = [by[i] for i in ids if i in by]
    out += [p for p in ph if p["lang"] == "mix" and p["cat"] == "mix"][:n_mix]
    out += [p for p in ph if p["lang"] == "ru" and p["cat"] == "ru"][:n_ru]
    return out


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return xs[k]


def summarize(items: dict) -> dict:
    v = list(items.values())
    f = lambda k, p: (round(q([x.get(k) for x in v], p), 3) if q([x.get(k) for x in v], p) is not None else None)  # noqa: E731
    return dict(n=len(v), audio_s=round(sum(x["audio_s"] for x in v), 1), rtf_med=f("rtf", 0.5), rtf_p90=f("rtf", 0.9),
                ttfa_model_med=f("ttfa_model_ms", 0.5), ttfa_med=f("ttfa_ms", 0.5), ttfa_p90=f("ttfa_ms", 0.9),
                prebuf_med=f("prebuffer_ms", 0.5), prebuf_p90=f("prebuffer_ms", 0.9), prebuf_max=f("prebuffer_ms", 1.0),
                runaway=sum(1 for x in v if x.get("runaway")))


# ---------------------------------------------------------------- прогон
def run(a):
    import soundfile as sf
    import streamio
    import tts_server as T

    T.C.mock = a.mock
    T.C.weights, T.C.ckpt, T.C.ref_cache, T.C.optimize = a.weights, a.lora, True, False
    T.C.mock_stream_rtf, T.C.mock_lead = 0.05, 0.2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    phrases = pick_phrases(Path(a.phrases), a.ids.split(",") if a.ids else DEFAULT_IDS, a.n_mix, a.n_ru)
    configs = [c for c in (a.configs.split(",") if a.configs else ORDER) if c in SPEC]
    res_path = out / "bench.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    res.setdefault("_meta", {}).update(made=time.strftime("%Y-%m-%d %H:%M:%S"), weights=a.weights, lora=a.lora, merged=a.merged,
                                       ref=a.ref, seed=a.seed, phrases=[p["id"] for p in phrases], mock=a.mock)
    deadline = a.deadline or (time.time() + 3600)

    def save():
        tmp = res_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        os.replace(tmp, res_path)

    def watchdog():                      # срок сеанса: записать сделанное и выйти, даже если torch.compile завис
        while time.time() < deadline:
            time.sleep(5)
        log("СРОК СЕАНСА — выхожу, сделанное в", res_path)
        try:
            save()
        finally:
            os._exit(3)
    threading.Thread(target=watchdog, daemon=True).start()

    t = time.time()
    eng = T.Engine()
    eng.load()
    sr = eng.sr
    res["_meta"]["load_s"] = round(time.time() - t, 1)
    gpu = None
    if not a.mock:
        import torch
        gpu = torch.cuda.get_device_name(0)
    res["_meta"]["gpu"] = gpu
    log(f"модель загружена за {res['_meta']['load_s']} с, sr {sr}, GPU {gpu}, фраз {len(phrases)}, конфигураций {configs}")

    def one_full(p, steps, cfg):
        t0 = time.perf_counter()
        w = eng.synth(p["text"], a.ref, None, None, cfg, steps, a.seed)
        dt = time.perf_counter() - t0
        return w, dict(synth_s=round(dt, 3), audio_s=round(len(w) / sr, 3), rtf=round(dt / max(len(w) / sr, 1e-6), 3), ttfa_ms=round(dt * 1000),
                       ttfa_model_ms=round(dt * 1000), prebuffer_ms=None, runaway=False)

    def one_stream(p, steps, cfg):
        letters = max(1, sum(ch.isalpha() for ch in p["text"]))
        cap = max(2.5, 0.25 * letters) + 0.5
        gate = streamio.SilenceGate(sr)
        raw, emits = [], []
        t0 = time.perf_counter()
        t_model = t_aud = None
        made, runaway = 0, False
        g = eng.stream(p["text"], a.ref, None, None, cfg, steps, a.seed)
        try:
            for w in g:
                now = time.perf_counter()
                if t_model is None:
                    t_model = now
                raw.append(w)
                made += len(w)
                o = gate.push(w)
                if len(o):
                    if t_aud is None:
                        t_aud = now
                    emits.append((now - t_aud, len(o) / sr))
                if made / sr > cap:
                    runaway = True
                    break
        finally:
            g.close()
        dt = time.perf_counter() - t0
        x = np.concatenate(raw) if raw else np.zeros(1, np.float32)
        aud = len(x) / sr
        return x, dict(synth_s=round(dt, 3), audio_s=round(aud, 3), rtf=round(dt / max(aud, 1e-6), 3),
                       ttfa_model_ms=round((t_model - t0) * 1000) if t_model else None,
                       ttfa_ms=round((t_aud - t0) * 1000) if t_aud else None, lead_cut_ms=round(gate.dropped / sr * 1000),
                       prebuffer_ms=round(streamio.need_prebuffer(emits) * 1000), patches=len(raw), runaway=runaway)

    state = "lora"
    for c in configs:
        sp = SPEC[c]
        left = deadline - time.time()
        if left < (420 if sp["w"] == "compiled" and state != "compiled" else 90):
            log(f"{c}: пропуск — до срока {left:.0f} с")
            res[c] = dict(skipped=f"мало времени ({left:.0f} с)")
            save()
            continue
        try:
            if sp["w"] in ("merged", "compiled") and state == "lora":
                t = time.time()
                n = merge_in_memory(eng, a.merged, a.mock)
                state = "merged"
                res["_meta"]["merge_swap"] = dict(modules=n, s=round(time.time() - t, 1))
                log(f"веса заменены на влитые ({a.merged}): {n} слоёв LoRALinear → nn.Linear за {time.time() - t:.1f} с")
            if sp["w"] == "compiled" and state != "compiled":
                t = time.time()
                if not a.mock:
                    eng.model.tts_model.optimize()
                for p in phrases[:2]:                 # компиляция на первых вызовах
                    one_stream(p, sp["steps"], sp["cfg"])
                state = "compiled"
                res["_meta"]["compile_s"] = round(time.time() - t, 1)
                log(f"torch.compile + прогрев: {time.time() - t:.0f} с")
            d = out / c
            d.mkdir(exist_ok=True)
            for p in phrases[:2]:                     # прогрев режима (не в счёт)
                (one_full if sp["kind"] == "full" else one_stream)(p, sp["steps"], sp["cfg"])
            items = {}
            for p in phrases:
                w, m = (one_full if sp["kind"] == "full" else one_stream)(p, sp["steps"], sp["cfg"])
                sf.write(d / f"{p['id']}.wav", w, sr)
                items[p["id"]] = dict(m, lang=p["lang"], chars=len(p["text"]))
            res[c] = dict(spec=sp, summary=summarize(items), items=items)
            s = res[c]["summary"]
            log(f"{c}: RTF мед {s['rtf_med']} p90 {s['rtf_p90']}, TTFA мед {s['ttfa_med']} мс p90 {s['ttfa_p90']} "
                f"(до патча {s['ttfa_model_med']}), запас p90 {s['prebuf_p90']} мс, не остановилась {s['runaway']}")
        except Exception as e:  # noqa: BLE001 — сломался режим (например, compile) — остальное уже записано
            res[c] = dict(error=f"{type(e).__name__}: {str(e)[:400]}")
            log(f"{c}: ОШИБКА {res[c]['error']}")
            if sp["w"] == "compiled":
                state = "broken"
        save()
    ids = [p["id"] for p in phrases]
    for key, x, y in (("stream_vs_full", "A", "B"), ("merged_vs_lora", "B", "M10")):
        if "items" in res.get(x, {}) and "items" in res.get(y, {}):
            res["_meta"][key] = compare(out / x, out / y, ids)
            log(f"{x}↔{y} (тот же сид, корреляция волн):", res["_meta"][key])
    save()
    log("готово:", res_path)
    return 0


def merge_in_memory(eng, merged_dir, mock):
    """LoRALinear → nn.Linear с весами из влитого model.safetensors (то же, что загрузить влитую модель с диска)."""
    if mock:
        return 0
    import torch
    from safetensors.torch import load_file
    from voxcpm.modules.layers.lora import LoRALinear
    tm = eng.model.tts_model
    sd = load_file(str(Path(merged_dir) / "model.safetensors"))
    n = 0
    for name, mod in list(tm.named_modules()):
        if not isinstance(mod, LoRALinear):
            continue
        lin = torch.nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None, device=mod.weight.device, dtype=mod.weight.dtype)
        with torch.no_grad():
            lin.weight.copy_(sd[name + ".weight"].to(lin.weight.device))
            if mod.bias is not None:
                lin.bias.copy_(sd[name + ".bias"].to(lin.bias.device))
        parent = tm.get_submodule(name.rsplit(".", 1)[0]) if "." in name else tm
        setattr(parent, name.rsplit(".", 1)[-1], lin)
        n += 1
    left = sum(1 for m in tm.modules() if isinstance(m, LoRALinear))
    if left:
        raise RuntimeError(f"после замены осталось LoRALinear: {left}")
    torch.cuda.empty_cache()
    return n


def compare(da: Path, db: Path, ids):
    import soundfile as sf
    cs, ds = [], []
    for i in ids:
        try:
            x, _ = sf.read(da / f"{i}.wav")
            y, _ = sf.read(db / f"{i}.wav")
        except Exception:  # noqa: BLE001
            continue
        n = min(len(x), len(y))
        if n < 1000:
            continue
        x, y = x[:n], y[:n]
        cs.append(float(np.corrcoef(x, y)[0, 1]))
        ds.append(abs(len(x) - len(y)))
    return dict(n=len(cs), corr_med=round(statistics.median(cs), 4) if cs else None, corr_min=round(min(cs), 4) if cs else None)


# ---------------------------------------------------------------- отчёт и страница
def report(a):
    out = Path(a.out)
    res = json.loads((out / "bench.json").read_text())
    gg = {}
    gp = Path(a.gigaam) if a.gigaam else out / "gigaam.json"
    if gp.exists():
        gg = json.loads(gp.read_text()).get("sets", {})
    m = res.get("_meta", {})
    rows = [f"Замер скорости голоса для звонка — {m.get('made')} · GPU {m.get('gpu')} · загрузка {m.get('load_s')} с · "
            f"фраз {len(m.get('phrases', []))} · сид {m.get('seed')}",
            f"LoRA {m.get('lora')} · влитые {m.get('merged')} · замена слоёв {m.get('merge_swap')} · compile {m.get('compile_s')} с",
            f"поток против целиком (A↔B, тот же сид, корреляция волн): {m.get('stream_vs_full')}; "
            f"влитые против LoRA (B↔M10): {m.get('merged_vs_lora')}", "",
            f"{'конф':6} {'шаги':>4} {'cfg':>4} {'RTF мед':>8} {'p90':>6} {'TTFA мед':>9} {'p90':>6} {'до патча':>9} {'запас p90':>10} "
            f"{'макс':>6} {'не ост.':>7} {'CER kk мед':>10} {'ср':>6} {'провалы':>8} {'CER все':>8}"]
    for c in ORDER:
        x = res.get(c)
        if not x:
            continue
        if "summary" not in x:
            rows.append(f"{c:6} {x.get('skipped') or x.get('error')}")
            continue
        s, sp = x["summary"], x["spec"]
        g = gg.get(c, {}).get("summary", {})
        kk, al = g.get("kk", {}), g.get("all", {})
        pct = lambda v: f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "—"  # noqa: E731
        rows.append(f"{c:6} {sp['steps']:>4} {sp['cfg']:>4} {s['rtf_med']!s:>8} {s['rtf_p90']!s:>6} {s['ttfa_med']!s:>9} {s['ttfa_p90']!s:>6} "
                    f"{s['ttfa_model_med']!s:>9} {s['prebuf_p90']!s:>10} {s['prebuf_max']!s:>6} {s['runaway']:>7} "
                    f"{pct(kk.get('cer_median')):>10} {pct(kk.get('cer_mean')):>6} {pct(kk.get('fail')):>8} {pct(al.get('cer_mean')):>8}")
    rows += ["", "RTF — время синтеза / длительность звука (< 1 — быстрее реального времени); TTFA — от вызова до первого слышимого звука",
             "(в звонке к нему прибавятся распознавание, мозг и сеть); запас — сколько звука копить у слушателя, чтобы не было провалов;",
             "CER — GigaAM CTC без подсказки (независимый ASR), провалы — доля фраз с CER > 10 %."]
    txt = "\n".join(rows)
    (out / "bench.txt").write_text(txt + "\n")
    short = []
    for c in ORDER:
        x = res.get(c) or {}
        if "summary" in x:
            kk = gg.get(c, {}).get("summary", {}).get("kk", {})
            cer = f", CER kk {kk['cer_median'] * 100:.1f}% (провалы {kk['fail'] * 100:.0f}%)" if kk else ""
            short.append(f"{c}: RTF {x['summary']['rtf_med']}, TTFA {x['summary']['ttfa_med']} мс{cer}")
        elif x:
            short.append(f"{c}: {(x.get('skipped') or x.get('error') or '')[:60]}")
    (out / "bench_short.txt").write_text("; ".join(short) + "\n")
    print(txt)
    return 0


def page(a):
    out = Path(a.out)
    res = json.loads((out / "bench.json").read_text())
    phr = {}
    try:
        phr = {p["id"]: p for p in json.loads(Path(a.phrases).read_text())["phrases"]}
    except Exception:  # noqa: BLE001
        pass
    gg = {}
    if (out / "gigaam.json").exists():
        gg = json.loads((out / "gigaam.json").read_text()).get("sets", {})
    cs = [c for c in ORDER if c in res and "items" in res[c]]
    head = "".join(f"<th>{c}<br><small>{res[c]['spec']['steps']} ш · cfg {res[c]['spec']['cfg']}<br>RTF {res[c]['summary']['rtf_med']}</small></th>"
                   for c in cs)
    body = []
    for pid in res.get("_meta", {}).get("phrases", []):
        cells = []
        for c in cs:
            it = res[c]["items"].get(pid, {})
            cer = gg.get(c, {}).get("items", {}).get(pid, {}).get("cer")
            cells.append(f"<td><audio controls preload='none' src='{c}/{pid}.wav'></audio><br><small>RTF {it.get('rtf')} · "
                         f"TTFA {it.get('ttfa_ms')} мс{' · CER ' + format(cer * 100, '.0f') + '%' if cer is not None else ''}</small></td>")
        text = html.escape(phr.get(pid, {}).get("text", ""))
        body.append(f"<tr><td><b>{pid}</b><br>{text}</td>{''.join(cells)}</tr>")
    doc = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Скорость голоса</title><style>
:root{{--bg:#fff;--fg:#1e293b;--mut:#5b6b85;--line:#e3e8f0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1726;--fg:#e8eef8;--mut:#9fb0c8;--line:#26324a}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.4 Inter,system-ui,sans-serif;margin:16px}}
table{{border-collapse:collapse}} td,th{{border:1px solid var(--line);padding:6px;vertical-align:top}} small{{color:var(--mut)}}
audio{{width:180px}} pre{{white-space:pre-wrap;font-size:12px}}
</style></head><body><h1>Скорость голоса для звонка</h1>
<p>Одна фраза — один сид во всех столбцах. A — как сейчас у бота (LoRA, целиком); B — LoRA потоком; M — влитые веса потоком
(число — шаги диффузии); O — влитые + torch.compile. Слушать: одинаково ли звучит B и A, где начинает портиться при меньших шагах.</p>
<pre>{html.escape((out / 'bench.txt').read_text()) if (out / 'bench.txt').exists() else ''}</pre>
<div style="overflow-x:auto"><table><tr><th>фраза</th>{head}</tr>{''.join(body)}</table></div></body></html>"""
    (out / "index.html").write_text(doc)
    print(out / "index.html")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "page"])
    ap.add_argument("out", nargs="?")
    ap.add_argument("--out", dest="out_opt")
    ap.add_argument("--weights", default=os.environ.get("TTS_WEIGHTS", str(VOICE_HOME / "weights" / "VoxCPM2")))
    ap.add_argument("--lora", default=os.environ.get("TTS_CKPT", ""), help="папка снимка LoRA (пусто — без LoRA)")
    ap.add_argument("--merged", default="")
    ap.add_argument("--ref", default=str(HERE.parent / "voices" / "aruzhan_f_kk.wav"))
    ap.add_argument("--phrases", default=str(HERE.parent / "eval" / "phrases_v2.json"))
    ap.add_argument("--ids", default="")
    ap.add_argument("--n-mix", type=int, default=3)
    ap.add_argument("--n-ru", type=int, default=2)
    ap.add_argument("--configs", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--deadline", type=float, default=0, help="unix-время: к нему всё записать и выйти")
    ap.add_argument("--gigaam", default="")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    a.out = a.out_opt or a.out
    if not a.out:
        ap.error("нужна папка out")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return {"run": run, "report": report, "page": page}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
