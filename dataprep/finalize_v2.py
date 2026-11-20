#!/usr/bin/env python3
"""Итог обработки v2: манифест VoxCPM для смеси + сводка. Можно запускать сколько угодно раз (читает только work/*/result.json).

  python dataprep/finalize_v2.py --out "$VOICE_HOME/v2" --refs tts_refs.json

Что делает поверх фильтров process_v2.py:
  * TTS-голоса: центр голоса ролика (ECAPA) против эталонов TTS (открытые TTS + голоса, найденные 06.10) ≥ 0,70 → вон;
    один и тот же голос у ≥ 3 разных каналов (≥ 0,75) — стоковый голос озвучки → вон;
  * уверенность CTC: нижние 7 % фраз → вон; скорость речи (симв/с) — выбросы за 1,5 IQR внутри канала → вон;
  * диктор: главный голос ролика = yt:<канал> (как в v1), остальные — yt:<канал>:<ролик>:<голос>;
  * стиль по запросу сборщика (подкаст / сұхбат / влог / шолу / әңгіме / прочее) — для меток стиля в смеси;
  * val: 8 каналов целиком (по хэшу имени, ≥ 0,3 ч каждый) — не случайный 1 %.
Выход: manifest/ours_v2.jsonl (train), manifest/ours_v2_val.jsonl, summary.json, summary.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

STYLE = [("подкаст", "подкаст"), ("podcast", "подкаст"), ("сұхбат", "сұхбат"), ("интервью", "сұхбат"), ("влог", "влог"),
         ("vlog", "влог"), ("шолу", "шолу"), ("обзор", "шолу"), ("әңгіме", "әңгіме")]


def style_of(q: str) -> str:
    q = (q or "").lower()
    return next((s for k, s in STYLE if k in q), "прочее")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--refs", type=Path, default=None)
    ap.add_argument("--tts-thr", type=float, default=0.70)
    ap.add_argument("--stock-thr", type=float, default=0.75)
    ap.add_argument("--conf-pct", type=float, default=7.0)
    ap.add_argument("--val-channels", type=int, default=8)
    ap.add_argument("--recheck-cuts", type=int, default=1, help="1 — фразы, отсеянные только за cut_start/cut_end, проверить по звуку краёв")
    ap.add_argument("--edge-db", type=float, default=-20.0, help="край (20 мс) тише этого уровня к p90 фразы — не обрезан")
    # 08.10: обработка сессии 2 хранит и пограничные фразы (DNSMOS OVRL ≥ 2,5, SIG ≥ 2,8, BAK ≥ 2,6, SNR ≥ 10) — строгие пороги v1 здесь
    ap.add_argument("--min-ovrl", type=float, default=2.70)
    ap.add_argument("--min-sig", type=float, default=3.00)
    ap.add_argument("--min-bak", type=float, default=2.80)
    ap.add_argument("--min-snr", type=float, default=12.0)
    ap.add_argument("--s2-since", default="2026-10-08 04:00:00", help="result.json с made не раньше (UTC сервера) — сессия 2")
    a = ap.parse_args()
    H = 3600.0
    res = []
    for f in sorted((a.out / "work").glob("*/result.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        rs = f.with_name("respan.json")   # сессия 2: куски «без пауз» переобработаны — их фразы вместо no_pause
        if rs.exists():
            try:
                x = json.loads(rs.read_text())
                r["phrases"] = [p for p in r["phrases"] if not p.get("no_pause")] + x["phrases"]
                r["speaker_centroids"] = {**(r.get("speaker_centroids") or {}), **{str(k): v for k, v in (x.get("speaker_centroids_respan") or {}).items()}}
                r["clean_sec"] += x.get("clean_sec", 0.0)
                r["asr_sec"] += x.get("asr_sec", 0.0)
            except Exception:  # noqa: BLE001
                pass
        res.append(r)
    # --- голоса роликов
    voices = []   # (clip_idx, spk, author, emb, clean_sec)
    for ci, r in enumerate(res):
        sec = Counter()
        for p in r["phrases"]:
            if p.get("clean_ok") and p.get("speaker"):
                sec[int(p["speaker"].rsplit("spk", 1)[1])] += p["duration"]
        r["_main"] = sec.most_common(1)[0][0] if sec else None
        for k, e in (r.get("speaker_centroids") or {}).items():
            voices.append((ci, int(k), r["meta"].get("author", ""), np.asarray(e, dtype=np.float32), sec.get(int(k), 0.0)))
    tts_bad: dict[tuple[int, int], str] = {}
    if voices:
        E = np.stack([v[3] for v in voices])
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        if a.refs and a.refs.exists():
            refs = json.loads(a.refs.read_text())
            R = np.asarray(list(refs.values()), dtype=np.float32)
            R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-9
            S = E @ R.T
            names = list(refs)
            for i, v in enumerate(voices):
                j = int(S[i].argmax())
                if S[i, j] >= a.tts_thr:
                    tts_bad[(v[0], v[1])] = f"tts_ref:{names[j]}:{S[i, j]:.2f}"
        main_idx = [i for i, v in enumerate(voices) if res[v[0]]["_main"] == v[1] and v[4] >= 30]
        if main_idx:
            M = E[main_idx]
            S = M @ M.T
            auth = [voices[i][2] for i in main_idx]
            par = list(range(len(main_idx)))

            def f(x):
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                return x
            ii, jj = np.where(np.triu(S, 1) >= a.stock_thr)
            for i, j in zip(ii, jj):
                if auth[i] != auth[j]:
                    par[f(i)] = f(j)
            groups = defaultdict(list)
            for i in range(len(main_idx)):
                groups[f(i)].append(i)
            for g in groups.values():
                if len({auth[i] for i in g}) >= 3:
                    for i in g:
                        v = voices[main_idx[i]]
                        tts_bad.setdefault((v[0], v[1]), f"stock_voice:{len({auth[k] for k in g})}_каналов")
    # --- фразы-кандидаты
    # 08.10 (замер на 3 000 фразах): «обрезано» по таймкодам GigaAM — ложная тревога: у фраз cut_start/cut_end энергия первых/последних
    # 20 мс та же, что у принятых (медиана −33…−35 дБ к p90 фразы; громче −20 дБ — 5–7 % в обеих группах), а head_gap = 0 —
    # особенность таймкодов CTC. Поэтому такие фразы перепроверяем по звуку: края тише edge_db — фраза целая.
    HARD = {"empty", "digits", "latin", "symbols", "low_logprob", "repetition", "no_speech", "speed", "asr_disagree", "has_digits",
            "cut_end", "cut_start"}
    CUTS = {"cut_start", "cut_end"}

    def edges_quiet(fp: str) -> bool:
        import soundfile as sf
        try:
            x, sr = sf.read(fp)
        except Exception:  # noqa: BLE001
            return False
        w = max(1, int(0.02 * sr))
        n = len(x) // w
        if n < 5:
            return False
        fr = np.sqrt((x[: n * w].reshape(n, w) ** 2).mean(1) + 1e-12)
        ref = np.percentile(fr, 90)
        return bool(20 * np.log10(fr[0] / ref) <= a.edge_db and 20 * np.log10(fr[-1] / ref) <= a.edge_db)
    recovered = 0.0
    cand = []
    for ci, r in enumerate(res):
        au = r["meta"].get("author", "?")
        for p in r["phrases"]:
            if not (p.get("clean_ok") and p.get("file")):
                continue
            if not p.get("asr_ok"):
                why = set((p.get("asr_why") or "").split(";")) & HARD
                if not (a.recheck_cuts and why and why <= CUTS and edges_quiet(p["file"])):
                    continue
                recovered += p["duration"]
            spk = int(p["speaker"].rsplit("spk", 1)[1]) if p.get("speaker") else -1
            p["_why"] = tts_bad.get((ci, spk), "")
            snr = p.get("snr_db")
            if not p["_why"] and ((p.get("dnsmos_ovrl") or 0) < a.min_ovrl or (p.get("dnsmos_sig") or 0) < a.min_sig
                                  or (p.get("dnsmos_bak") or 0) < a.min_bak or (snr is not None and snr == snr and snr < a.min_snr)):
                p["_why"] = "borderline_quality"   # пограничная (сессия 2 хранит) — строгий порог v1
            p["_author"], p["_clip"], p["_spk"], p["_ci"] = au, r["clip_id"], spk, ci
            cand.append(p)
    confs = np.array([p.get("ctc_conf") or 0 for p in cand]) if cand else np.zeros(0)
    conf_thr = float(np.percentile(confs, a.conf_pct)) if len(confs) else 0.0
    by_auth = defaultdict(list)
    for p in cand:
        by_auth[p["_author"]].append(p.get("cps") or 0)
    allc = np.array([p.get("cps") or 0 for p in cand]) if cand else np.zeros(1)

    def fences(xs):
        q1, q3 = np.percentile(xs, [25, 75])
        return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    g_lo, g_hi = fences(allc)
    au_f = {au: (fences(np.array(xs)) if len(xs) >= 20 else (g_lo, g_hi)) for au, xs in by_auth.items()}
    for p in cand:
        if p["_why"]:
            continue
        if (p.get("ctc_conf") or 0) < conf_thr:
            p["_why"] = "low_ctc_conf"
        else:
            lo, hi = au_f[p["_author"]]
            if not lo <= (p.get("cps") or 0) <= hi:
                p["_why"] = "speed_outlier"
    keep = [p for p in cand if not p["_why"]]
    # --- val: целые каналы
    hours_au = Counter()
    for p in keep:
        hours_au[p["_author"]] += p["duration"] / H
    val_auth = [au for au in sorted(hours_au, key=lambda x: hashlib.md5(x.encode()).hexdigest()) if hours_au[au] >= 0.3][: a.val_channels]
    md = a.out / "manifest"
    md.mkdir(parents=True, exist_ok=True)
    n_tr = n_va = 0
    sec_lang, sec_style, spk_sec = Counter(), Counter(), Counter()
    with open(md / "ours_v2.jsonl", "w") as ft, open(md / "ours_v2_val.jsonl", "w") as fv:
        for p in keep:
            r = res[p["_ci"]]
            main = r["_main"] == p["_spk"]
            spk = f"yt:{p['_author']}" if main else f"yt:{p['_author']}:{p['_clip']}:{p['_spk']}"
            lang = "mix" if p.get("lang") == "mix" else p.get("asr_lang", "kk")
            st_ = style_of(r["meta"].get("query", ""))
            rec = dict(audio=p["file"], text=p["text"], duration=round(p["duration"], 3), speaker=spk, lang=lang, style=st_,
                       source="ours_v2", clip_id=p["_clip"], fillers_dense=int("fillers_dense" in (p.get("asr_why") or "")),
                       dnsmos_ovrl=p.get("dnsmos_ovrl"), ctc_conf=p.get("ctc_conf"), cer_giga=p.get("cer_giga"),
                       session="s2" if (r.get("made") or "") >= a.s2_since else "s1", separated=int(bool(p.get("separated"))),
                       cps=p.get("cps"))
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if p["_author"] in val_auth:
                fv.write(line)
                n_va += 1
            else:
                ft.write(line)
                n_tr += 1
                sec_lang[lang] += p["duration"]
                sec_style[st_] += p["duration"]
                spk_sec[spk] += p["duration"]
    fin_rej = Counter()
    for p in cand:
        if p["_why"]:
            fin_rej[p["_why"].split(":")[0]] += p["duration"] / H
    tot = sum(sec_lang.values())
    summ = dict(
        clips=len(res), raw_h=round(sum(r["duration"] for r in res) / H, 2),
        speech_h=round(sum(r["speech_sec"] for r in res) / H, 2), clean_h=round(sum(r["clean_sec"] for r in res) / H, 2),
        asr_ok_h=round(sum(r["asr_sec"] for r in res) / H, 2), final_reject_h={k: round(v, 2) for k, v in fin_rej.items()},
        train_h=round(tot / H, 2), train_n=n_tr, val_n=n_va, val_channels=val_auth,
        by_lang_h={k: round(v / H, 2) for k, v in sec_lang.items()}, by_style_h={k: round(v / H, 2) for k, v in sec_style.items()},
        speakers=len(spk_sec), top_speaker_share=round(max(spk_sec.values()) / tot, 4) if tot else 0,
        ctc_conf_threshold=round(conf_thr, 4), tts_voices_dropped=len(tts_bad), recovered_cut_h=round(recovered / H, 2))
    # 08.10 «живость»: что отсеивается по стилю и темпу (на слух фильтры срезали живость голоса)
    H_ = 3600.0
    by_st = defaultdict(Counter)
    for r in res:
        st_ = style_of(r["meta"].get("query", ""))
        by_st[st_]["сырого"] += r["duration"] / H_
        for p in r["phrases"]:
            d = p["duration"] / H_
            by_st[st_]["речи"] += p.get("speech_dur", p["duration"]) / H_
            if not p.get("clean_ok"):
                by_st[st_]["отсев:" + ((p.get("clean_why") or "?").split(";")[0])] += d
            elif not p.get("asr_ok"):
                by_st[st_]["отсев_текст"] += d
    for p in cand:
        st_ = style_of(res[p["_ci"]]["meta"].get("query", ""))
        if p["_why"]:
            by_st[st_]["отсев_финал:" + p["_why"].split(":")[0]] += p["duration"] / H_
        else:
            by_st[st_]["в обучение"] += p["duration"] / H_
    cps_keep = [p.get("cps") for p in keep if p.get("cps")]
    cps_rej = [p.get("cps") for p in cand if p["_why"] and p.get("cps")]
    q = lambda xs: [round(float(v), 1) for v in np.percentile(xs, [10, 50, 90])] if xs else None
    summ["liveness"] = dict(by_style_h={k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in by_st.items()},
                            cps_p10_50_90_kept=q(cps_keep), cps_p10_50_90_rejected_final=q(cps_rej))
    (a.out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    txt = (f"v2: {summ['clips']} роликов; сырого {summ['raw_h']} ч → речи {summ['speech_h']} → чисто {summ['clean_h']} → "
           f"текст ок {summ['asr_ok_h']} → в обучение {summ['train_h']} ч ({n_tr} фраз, val {n_va} фраз, {len(val_auth)} каналов)\n"
           f"финальный отсев, ч: {summ['final_reject_h']}\nпо языкам, ч: {summ['by_lang_h']}\nпо стилю, ч: {summ['by_style_h']}\n"
           f"дикторов {summ['speakers']}, крупнейший {summ['top_speaker_share'] * 100:.1f} %; голосов TTS выброшено {len(tts_bad)}; "
           f"возвращено из «обрезано» после проверки краёв по звуку {summ['recovered_cut_h']} ч\n")
    (a.out / "summary.txt").write_text(txt)
    print(txt, end="")


if __name__ == "__main__":
    main()
