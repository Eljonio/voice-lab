"""Конвейер очистки: ролик -> чистые фразы + строки манифеста.

Шаги на ролик:
 1. декодирование (44.1 кГц стерео для Demucs, 16 кГц моно для анализа);
 2. детектор музыки по окнам 3 с (AST) -> есть ли музыкальный фон;
 3. если музыка есть -> Demucs htdemucs: vocals = голос, mix - vocals = музыка;
    повторный детектор по дорожке голоса (сколько музыки/пения осталось);
 4. нарезка на фразы 2..15 с по паузам (Silero VAD) по дорожке голоса;
 5. на каждую фразу: DNSMOS (SIG/BAK/OVRL), SNR, остаток музыки/пения,
    один ли голос (ECAPA), язык + черновой текст (Whisper medium);
 6. отсев по порогам, запись прошедших фраз (24 кГц моно, выровненная громкость).
Результат ролика -> work/<clip>/result.json; общий manifest.csv собирает collect().
"""
from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import audio as A
from .music import span_stats, summarize

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".opus", ".webm", ".flac", ".mkv", ".mov"}


@dataclass
class Config:
    out_sr: int = 24000
    separate: str = "auto"          # auto | always | never
    demucs_model: str = "htdemucs"
    whisper: str = "medium"
    # детектор музыки (вероятности AST по окнам 3 с)
    music_thr: float = 0.25          # окно считается «с музыкой»
    ast_win: float = 3.0             # окно детектора, с (на CPU берём 6 — в 4 раза быстрее)
    ast_hop: float = 1.5
    clip_music_frac: float = 0.10    # доля таких окон, чтобы включить отделение
    clip_music_peak: float = 0.50    # или хотя бы одно уверенное окно
    # отсев фраз
    min_len: float = 2.0
    max_len: float = 15.0
    max_music_after: float = 0.30
    max_sing_after: float = 0.30
    min_ovrl: float = 2.70
    min_sig: float = 3.00
    min_bak: float = 2.80
    min_snr: float = 12.0
    min_spk_sim: float = 0.15
    max_clip: float = 0.005          # доля сэмплов у пика; TikTok громко сведён лимитером, 0.1% — слишком строго
    langs_ok: tuple = ("kk", "ru", "mix")
    spk_cluster_thr: float = 0.35
    keep_rejected: bool = False
    demo_stems: bool = False         # сохранить mp3 «было/голос/музыка» для демо


REASON_ORDER = ["too_short", "music_left", "singing", "multi_speaker", "noise",
                "low_quality", "clipping", "lang_other"]


class Models:
    """Все модели грузятся один раз и живут на видеокарте вместе (влезают в 4 ГБ)."""

    def __init__(self, cfg: Config, device: str = "cuda"):
        from .lang import LangID
        from .music import MusicTagger
        from .quality import DNSMOS
        from .segment import Segmenter
        from .separate import Separator
        from .speakers import SpeakerChecker
        t = time.time()
        self.tagger = MusicTagger(device)
        self.sep = Separator(cfg.demucs_model, device)
        self.dnsmos = DNSMOS()
        self.seg = Segmenter(min_len=cfg.min_len, max_len=cfg.max_len)
        self.spk = SpeakerChecker(device)
        self.lang = LangID(cfg.whisper, device)
        self.load_s = time.time() - t


def clip_id_of(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)[:80]


def _dnsmos_clip(m: Models, x16: np.ndarray) -> dict:
    return m.dnsmos(x16, hop_s=9.0)


def process_clip(path: Path, m: Models, cfg: Config, out_clean: Path, work: Path,
                 source_set: str = "", meta: dict | None = None,
                 ref_path: Path | None = None) -> dict:
    T = {}
    t_all = time.time()
    cid = clip_id_of(path)
    wdir = work / cid
    wdir.mkdir(parents=True, exist_ok=True)

    t = time.time()
    x44 = A.load(path, 44100, channels=2)
    x16 = A.load(path, 16000, channels=1)
    dur = len(x16) / 16000
    T["decode"] = time.time() - t

    t = time.time()
    tl_before = m.tagger.timeline(x16, cfg.ast_win, cfg.ast_hop)
    s_before = summarize(tl_before, cfg.music_thr)
    T["music_detect"] = time.time() - t

    has_music = s_before["music_frac"] >= cfg.clip_music_frac or \
        float(np.max(tl_before["music"])) >= cfg.clip_music_peak
    do_sep = cfg.separate == "always" or (cfg.separate == "auto" and has_music)

    t = time.time()
    q_before = _dnsmos_clip(m, x16)
    T["dnsmos_clip"] = time.time() - t

    rest44 = None
    if do_sep:
        t = time.time()
        voc44, rest44 = m.sep(x44)
        T["demucs"] = time.time() - t
        v16 = A.resample(A.to_mono(voc44), 44100, 16000)
        t = time.time()
        tl_after = m.tagger.timeline(v16, cfg.ast_win, cfg.ast_hop)
        T["music_detect"] += time.time() - t
        t = time.time()
        q_after = _dnsmos_clip(m, v16)
        T["dnsmos_clip"] += time.time() - t
        hi = A.resample(A.to_mono(voc44), 44100, cfg.out_sr)
    else:
        voc44, v16, tl_after, q_after = None, x16, tl_before, q_before
        hi = A.resample(A.to_mono(x44), 44100, cfg.out_sr)
    s_after = summarize(tl_after, cfg.music_thr)

    # эталон (только для синтетических смесей): насколько голос похож на чистую запись
    ref_metrics = {}
    if ref_path is not None and ref_path.exists():
        ref16 = A.load(ref_path, 16000, 1)
        ref_metrics = dict(si_sdr_mix=A.si_sdr(x16, ref16), si_sdr_out=A.si_sdr(v16, ref16),
                           dnsmos_ref=_dnsmos_clip(m, ref16))

    if cfg.demo_stems:
        A.save_mp3(wdir / "orig.mp3", x44, 44100)
        if voc44 is not None:
            A.save_mp3(wdir / "vocals.mp3", voc44, 44100)
            A.save_mp3(wdir / "music.mp3", rest44, 44100)

    # --- фразы ---
    t = time.time()
    phrases = m.seg.phrases(v16)
    T["vad"] = time.time() - t
    T.update(dnsmos_phr=0.0, speaker=0.0, whisper=0.0)
    rows, embs = [], []
    for i, p in enumerate(phrases):
        a, b = int(p["start"] * 16000), int(p["end"] * 16000)
        seg16 = v16[a:b]
        r = dict(clip_id=cid, source=str(path), source_set=source_set, idx=i,
                 start=p["start"], end=p["end"], duration=p["dur"], separated=int(do_sep),
                 clip_had_music=int(has_music))
        sb = span_stats(tl_before, p["start"], p["end"])
        sa = span_stats(tl_after, p["start"], p["end"])
        r.update(music_before=round(sb["music"], 3), music_after=round(sa["music"], 3),
                 sing_after=round(sa["sing"], 3))
        if p["dur"] >= cfg.min_len:
            t1 = time.time()
            q = m.dnsmos(seg16, hop_s=3.0)
            T["dnsmos_phr"] += time.time() - t1
            r.update(dnsmos_sig=round(q["sig"], 3), dnsmos_bak=round(q["bak"], 3),
                     dnsmos_ovrl=round(q["ovrl"], 3), p808=round(q["p808"], 3))
            r["snr_db"] = round(A.snr_estimate(seg16, 16000), 1)
            r["clip_ratio"] = round(A.clipping_ratio(x16[a:b]), 5)
            t1 = time.time()
            sc = m.spk.phrase_check(seg16)
            T["speaker"] += time.time() - t1
            r["spk_min_sim"] = round(sc["min_sim"], 3)
            embs.append((i, sc["emb"]))
            t1 = time.time()
            li = m.lang(seg16)
            T["whisper"] += time.time() - t1
            r.update(lang=li["lang"], lang_top=li["lang_top"], lang_prob=round(li["lang_prob"], 3),
                     p_kk=round(li["p_kk"], 3), p_kk_like=round(li["p_kk_like"], 3),
                     p_ru=round(li["p_ru"], 3), vox_kk=round(li["vox_kk"], 3), text_whisper=li["text"])
        rows.append(r)

    # говорящие по ролику
    from .speakers import cluster
    labels = {}
    if embs:
        lab = cluster(np.stack([e for _, e in embs]), cfg.spk_cluster_thr)
        labels = {i: int(l) for (i, _), l in zip(embs, lab)}
    n_spk = len(set(labels.values())) if labels else 0

    # отсев
    kept_sec = 0.0
    for r in rows:
        reasons = []
        if r["duration"] < cfg.min_len:
            reasons.append("too_short")
        else:
            if r["music_after"] > cfg.max_music_after:
                reasons.append("music_left")
            if r["sing_after"] > cfg.max_sing_after:
                reasons.append("singing")
            if r["spk_min_sim"] < cfg.min_spk_sim:
                reasons.append("multi_speaker")
            if r["dnsmos_bak"] < cfg.min_bak or (r["snr_db"] == r["snr_db"] and r["snr_db"] < cfg.min_snr):
                reasons.append("noise")
            if r["dnsmos_ovrl"] < cfg.min_ovrl or r["dnsmos_sig"] < cfg.min_sig:
                reasons.append("low_quality")
            if r["clip_ratio"] > cfg.max_clip:
                reasons.append("clipping")
            if r.get("lang") not in cfg.langs_ok:
                reasons.append("lang_other")
        reasons.sort(key=REASON_ORDER.index)
        r["speaker"] = f"{cid}_spk{labels[r['idx']]}" if r["idx"] in labels else ""
        r["n_speakers_clip"] = n_spk
        r["passed"] = int(not reasons)
        r["reject_reason"] = reasons[0] if reasons else ""
        r["reject_all"] = ";".join(reasons)
        a, b = int(r["start"] * cfg.out_sr), int(r["end"] * cfg.out_sr)
        name = f"{cid}_{r['idx']:03d}.wav"
        if r["passed"]:
            fp = out_clean / cid / name
            A.save_wav(fp, A.normalize_rms(hi[a:b]), cfg.out_sr)
            r["file"] = str(fp)
            kept_sec += r["duration"]
        elif cfg.keep_rejected:
            fp = wdir / "rejected" / name
            A.save_wav(fp, A.normalize_rms(hi[a:b]), cfg.out_sr)
            r["file"] = str(fp)
        else:
            r["file"] = ""

    T["total"] = time.time() - t_all
    res = dict(
        clip_id=cid, source=str(path), source_set=source_set, meta=meta or {},
        duration=dur, has_music=bool(has_music), separated=bool(do_sep),
        music_before=s_before, music_after=s_after,
        dnsmos_before=q_before, dnsmos_after=q_after, ref=ref_metrics,
        n_phrases=len(rows), n_passed=sum(r["passed"] for r in rows),
        clean_sec=kept_sec, speech_sec=float(sum(r["duration"] for r in rows)),
        n_speakers=n_spk, timings=T, rtf=T["total"] / max(dur, 1e-6),
        timeline_before=dict(t0=tl_before["t0"].tolist(), music=tl_before["music"].tolist(),
                             sing=tl_before["sing"].tolist(), speech=tl_before["speech"].tolist()),
        timeline_after=dict(t0=tl_after["t0"].tolist(), music=tl_after["music"].tolist(),
                            sing=tl_after["sing"].tolist(), speech=tl_after["speech"].tolist()),
        phrases=rows, config=asdict(cfg),
    )
    (wdir / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return res


MANIFEST_COLS = [
    "file", "source", "source_set", "clip_id", "idx", "start", "end", "duration", "lang",
    "lang_top", "lang_prob", "p_kk", "p_kk_like", "p_ru", "vox_kk", "dnsmos_ovrl", "dnsmos_sig",
    "dnsmos_bak", "p808", "snr_db", "clip_had_music", "separated", "music_before",
    "music_after", "sing_after", "spk_min_sim", "speaker", "n_speakers_clip", "clip_ratio",
    "passed", "reject_reason", "reject_all", "text_whisper",
]


def collect(work: Path, manifest: Path, clips_csv: Path | None = None) -> list[dict]:
    """Собирает work/*/result.json в manifest.csv (по фразам) и clips.csv (по роликам)."""
    results = [json.loads(p.read_text()) for p in sorted(work.glob("*/result.json"))]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS, extrasaction="ignore")
        w.writeheader()
        for res in results:
            for r in res["phrases"]:
                w.writerow(r)
    if clips_csv:
        cols = ["clip_id", "source", "source_set", "duration", "has_music", "separated",
                "music_frac_before", "music_frac_after", "sing_frac_after", "ovrl_before",
                "ovrl_after", "n_phrases", "n_passed", "speech_sec", "clean_sec", "n_speakers",
                "rtf", "total_s"]
        with open(clips_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in results:
                w.writerow([r["clip_id"], r["source"], r["source_set"], round(r["duration"], 2),
                            int(r["has_music"]), int(r["separated"]),
                            round(r["music_before"]["music_frac"], 3), round(r["music_after"]["music_frac"], 3),
                            round(r["music_after"]["sing_frac"], 3), round(r["dnsmos_before"]["ovrl"], 2),
                            round(r["dnsmos_after"]["ovrl"], 2), r["n_phrases"], r["n_passed"],
                            round(r["speech_sec"], 1), round(r["clean_sec"], 1), r["n_speakers"],
                            round(r["rtf"], 3), round(r["timings"]["total"], 1)])
    return results


def find_inputs(inp: Path) -> list[Path]:
    return sorted(p for p in inp.rglob("*") if p.suffix.lower() in AUDIO_EXT and p.is_file()
                  and not p.name.startswith("."))


def load_index(inp: Path) -> dict[str, dict]:
    """index.csv от сборщика: пытаемся сопоставить строку с файлом по любой колонке."""
    idx = inp / "index.csv"
    if not idx.exists():
        return {}
    out = {}
    with open(idx, newline="") as f:
        for row in csv.DictReader(f):
            for v in row.values():
                if v:
                    out[Path(v).stem] = row
                    out[Path(v).name] = row
    return out
