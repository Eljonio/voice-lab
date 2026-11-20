"""DNSMOS P.835 (Microsoft DNS Challenge): оценка качества речи без эталона.

SIG — чистота самого голоса (искажения, артефакты), BAK — насколько тихий фон
(5 = фона нет), OVRL — общая оценка. Шкала 1..5, как MOS у слушателей.
Портировано из DNS-Challenge/DNSMOS/dnsmos_local.py (модели ONNX там же).
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort

SR = 16000
INPUT_LENGTH = 9.01
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "dnsmos"

_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])


class DNSMOS:
    def __init__(self, model_dir: Path = MODEL_DIR):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        prov = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(model_dir / "sig_bak_ovr.onnx"), opts, providers=prov)
        self.p808 = ort.InferenceSession(str(model_dir / "model_v8.onnx"), opts, providers=prov)

    @staticmethod
    def _melspec(audio: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=321, hop_length=160, n_mels=120)
        return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T

    def __call__(self, audio16k: np.ndarray, hop_s: float = 1.0) -> dict:
        """audio16k — моно float32 16 кГц. hop_s — шаг окна 9 с (1 с как в оригинале)."""
        audio = audio16k.astype(np.float32)
        need = int(INPUT_LENGTH * SR)
        if len(audio) == 0:
            return dict(sig=np.nan, bak=np.nan, ovrl=np.nan, p808=np.nan)
        while len(audio) < need:  # короткие фразы зацикливаются, как в эталонном коде
            audio = np.append(audio, audio)
        starts = np.arange(0, len(audio) - need + 1, int(hop_s * SR))
        sig, bak, ovr, p808 = [], [], [], []
        for s in starts:
            seg = audio[s : s + need]
            raw = self.sess.run(None, {"input_1": seg[None, :]})[0][0]
            sig.append(_P_SIG(raw[0]))
            bak.append(_P_BAK(raw[1]))
            ovr.append(_P_OVR(raw[2]))
            feat = self._melspec(seg[:-160]).astype(np.float32)[None]
            p808.append(self.p808.run(None, {"input_1": feat})[0][0][0])
        return dict(
            sig=float(np.mean(sig)), bak=float(np.mean(bak)),
            ovrl=float(np.mean(ovr)), p808=float(np.mean(p808)),
        )
