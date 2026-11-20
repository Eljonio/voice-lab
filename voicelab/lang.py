"""Язык фразы + черновой текст.

Два мнения, итог — среднее:
  * faster-whisper (Whisper medium, MIT) — ещё и черновой текст;
  * VoxLingua107 ECAPA (speechbrain/lang-id-voxlingua107-ecapa, Apache-2.0) —
    специализированный определитель 107 языков, казахский в нём есть.
Whisper путает казахский с близкими тюркскими (az, tr, uz, ky, tt, ba) — на KazakhTTS
бывает top=az. Поэтому вероятности этих языков складываются в «kk-подобный» счёт.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from faster_whisper import WhisperModel

SR = 16000
TURKIC_NEAR = ("kk", "az", "tr", "uz", "ky", "tt", "ba", "tk", "crh")
VOX_SRC = Path(__file__).resolve().parent.parent / "models" / "voxlingua_src"
VOX_CACHE = Path(__file__).resolve().parent.parent / "models" / "voxlingua"


class LangID:
    def __init__(self, size: str = "medium", device: str = "cuda"):
        ct = "int8_float16" if device == "cuda" else "int8"
        self.model = WhisperModel(size, device=device, compute_type=ct)
        from speechbrain.inference.classifiers import EncoderClassifier
        src = str(VOX_SRC) if (VOX_SRC / "embedding_model.ckpt").exists() else "speechbrain/lang-id-voxlingua107-ecapa"
        self.vox = EncoderClassifier.from_hparams(source=src, savedir=str(VOX_CACHE), run_opts={"device": device})
        labels = self.vox.hparams.label_encoder.ind2lab
        codes = {i: lab.split(":")[0].strip() for i, lab in labels.items()}
        self.vox_codes = codes
        self.vox_turkic = [i for i, c in codes.items() if c in TURKIC_NEAR]
        self.vox_ru = [i for i, c in codes.items() if c == "ru"][0]
        self.vox_kk = [i for i, c in codes.items() if c == "kk"][0]
        self.device = device

    @torch.inference_mode()
    def _vox(self, audio16k: np.ndarray) -> dict:
        x = torch.from_numpy(audio16k[: 20 * SR]).float()[None].to(self.device)
        logp = self.vox.classify_batch(x)[0][0]
        p = torch.softmax(logp.float(), dim=-1).cpu().numpy()  # на случай ненормированных логитов
        top = self.vox_codes[int(p.argmax())]
        return dict(kk_like=float(p[self.vox_turkic].sum()), ru=float(p[self.vox_ru]),
                    kk=float(p[self.vox_kk]), top=top)

    def __call__(self, audio16k: np.ndarray, with_text: bool = True) -> dict:
        segs, info = self.model.transcribe(
            audio16k, beam_size=1, vad_filter=False, condition_on_previous_text=False,
            without_timestamps=True, language=None,
        )
        text = " ".join(s.text.strip() for s in segs) if with_text else ""
        probs = dict(info.all_language_probs or [])
        w_kk = sum(probs.get(c, 0.0) for c in TURKIC_NEAR)
        w_ru = probs.get("ru", 0.0)
        v = self._vox(audio16k)
        kk = (w_kk + v["kk_like"]) / 2
        ru = (w_ru + v["ru"]) / 2
        other = max(0.0, 1.0 - kk - ru)
        if kk >= 0.25 and ru >= 0.25:
            lang = "mix"   # казахский и русский оба заметны
        elif kk > ru and kk > other:
            lang = "kk"
        elif ru > kk and ru > other:
            lang = "ru"
        else:
            lang = "other"
        return dict(lang=lang, lang_top=f"{info.language}/{v['top']}", lang_prob=float(info.language_probability),
                    p_kk=float(probs.get("kk", 0.0)), p_kk_like=float(kk), p_ru=float(ru),
                    vox_kk=v["kk"], text=text)
