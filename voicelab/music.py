"""Детектор музыки / речи / пения по окнам.

Модель: MIT/ast-finetuned-audioset-10-10-0.4593 (Audio Spectrogram Transformer,
обучен на AudioSet, 527 классов, BSD-3). Ставится через transformers, без TensorFlow.
Для каждого окна получаем вероятности трёх групп классов:
  music  — «Music» и музыкальные инструменты/жанры (макс. по группе);
  sing   — пение/рэп/хор (важно: Demucs оставляет пение в дорожке голоса);
  speech — речь, разговор, повествование.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from pathlib import Path

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
LOCAL = Path(__file__).resolve().parent.parent / "models" / "ast"
SR = 16000

MUSIC_LABELS = {
    "Music", "Musical instrument", "Pop music", "Hip hop music", "Electronic music",
    "Rock music", "Drum", "Drum kit", "Drum machine", "Bass drum", "Snare drum",
    "Hi-hat", "Cymbal", "Bass guitar", "Guitar", "Electric guitar", "Acoustic guitar",
    "Piano", "Keyboard (musical)", "Synthesizer", "Electronic dance music", "Techno",
    "House music", "Dubstep", "Trap music", "Beatboxing", "Orchestra", "Violin, fiddle",
    "Strum", "Plucked string instrument", "Percussion", "Music of Asia",
    "Background music", "Theme music", "Soundtrack music", "Ambient music",
    "Dance music", "Rhythm and blues", "Soul music", "Reggae", "Funk", "Jazz",
    "Folk music", "Middle Eastern music", "Traditional music", "Pop music",
    "Christian music", "Music of Latin America", "Electronica", "Lofi",
}
SING_LABELS = {
    "Singing", "Choir", "Rapping", "Male singing", "Female singing", "Child singing",
    "Synthetic singing", "Humming", "Yodeling", "Chant", "Mantra", "A capella",
    "Vocal music", "Song",
}
SPEECH_LABELS = {
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue",
}


class MusicTagger:
    def __init__(self, device: str = "cuda"):
        self.device = device
        src = str(LOCAL) if (LOCAL / "model.safetensors").exists() else MODEL_ID
        self.fe = ASTFeatureExtractor.from_pretrained(src)
        self.model = ASTForAudioClassification.from_pretrained(src).to(device).eval()
        if device == "cuda":
            self.model = self.model.half()
        id2label = self.model.config.id2label
        self.idx_music = [i for i, l in id2label.items() if l in MUSIC_LABELS]
        self.idx_sing = [i for i, l in id2label.items() if l in SING_LABELS]
        self.idx_speech = [i for i, l in id2label.items() if l in SPEECH_LABELS]
        self.labels = id2label

    @torch.inference_mode()
    def probs(self, chunks: list[np.ndarray], batch: int = 8) -> np.ndarray:
        out = []
        for i in range(0, len(chunks), batch):
            feats = self.fe(chunks[i : i + batch], sampling_rate=SR, return_tensors="pt")
            x = feats["input_values"].to(self.device)
            if self.device == "cuda":
                x = x.half()
            logits = self.model(input_values=x).logits.float()
            out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def timeline(self, audio16k: np.ndarray, win_s: float = 3.0, hop_s: float = 1.5) -> dict:
        """Вероятности по окнам. Возвращает dict(t0, t1, music, sing, speech, top)."""
        n, w, h = len(audio16k), int(win_s * SR), int(hop_s * SR)
        starts = list(range(0, max(n - w, 0) + 1, h)) or [0]
        if starts[-1] + w < n:
            starts.append(max(n - w, 0))
        chunks = [audio16k[s : s + w] for s in starts]
        p = self.probs(chunks)
        top = []
        for row in p:
            j = np.argsort(row)[::-1][:3]
            top.append([(self.labels[int(k)], float(row[k])) for k in j])
        return dict(
            t0=np.array(starts) / SR,
            t1=np.minimum(np.array(starts) + w, n) / SR,
            music=p[:, self.idx_music].max(axis=1),
            sing=p[:, self.idx_sing].max(axis=1),
            speech=p[:, self.idx_speech].max(axis=1),
            top=top,
        )


def summarize(tl: dict, thr: float = 0.25) -> dict:
    """Итог по клипу: доля окон с музыкой, средние и максимум."""
    return dict(
        music_frac=float(np.mean(tl["music"] > thr)),
        music_mean=float(np.mean(tl["music"])),
        music_p90=float(np.percentile(tl["music"], 90)),
        sing_frac=float(np.mean(tl["sing"] > thr)),
        sing_p90=float(np.percentile(tl["sing"], 90)),
        speech_mean=float(np.mean(tl["speech"])),
    )


def span_stats(tl: dict, t0: float, t1: float) -> dict:
    """Максимальные вероятности музыки/пения в окнах, пересекающих отрезок [t0, t1]."""
    m = (tl["t1"] > t0 + 0.3) & (tl["t0"] < t1 - 0.3)
    if not m.any():
        m = (tl["t1"] > t0) & (tl["t0"] < t1)
    if not m.any():
        return dict(music=np.nan, sing=np.nan, speech=np.nan)
    return dict(
        music=float(np.max(tl["music"][m])),
        sing=float(np.max(tl["sing"][m])),
        speech=float(np.mean(tl["speech"][m])),
    )
