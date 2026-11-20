#!/usr/bin/env python3
"""Тестовый набор из открытых данных: речь (KazakhTTS2, KazEmoTTS, FLEURS, Common Voice,
KSD — CC BY 4.0 / CC0) + музыка с Wikimedia Commons (CC BY / CC BY-SA / PD).

Для синтетических смесей известен эталон — чистая речь до смешивания (ref/<clip>.wav),
поэтому качество отделения можно измерить честно (SI-SDR), а не на слух.
Смеси кодируются в AAC 128k (.m4a), как звук в TikTok.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voicelab import audio as A  # noqa: E402

SR = 44100
SPEECH = Path(os.environ.get("TESTMIX_SPEECH", "data/testmix-speech"))   # короткие клипы открытых корпусов: kaztts2-f1-1.wav, fleurs-kk-1.wav, ...
MUSIC = Path(__file__).resolve().parent.parent / "testdata/music"   # файлы скачать по testdata/music/LICENSES.csv
OUT = Path(os.environ.get("VOICE_DS", "data")) / "testmix"
rng = np.random.default_rng(7)


def speech(names, gap=(0.5, 0.8)):
    parts = []
    for n in names:
        x = A.load(SPEECH / n, SR, 1)
        parts.append(x)
        parts.append(np.zeros(int(rng.uniform(*gap) * SR), np.float32))
    y = np.concatenate([np.zeros(int(0.6 * SR), np.float32)] + parts)
    return A.normalize_rms(y, -22.0, -3.0)


def active_rms(x):
    fr = int(0.03 * SR)
    f = x[: len(x) // fr * fr].reshape(-1, fr)
    e = np.sqrt(np.mean(f ** 2, axis=1))
    return float(np.sqrt(np.mean(e[e > 0.1 * e.max()] ** 2)))


def music(name, offset, n):
    m = A.load(MUSIC / name, SR, 2)[:, int(offset * SR):]
    m = m[:, :n]
    fade = int(0.5 * SR)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    m[:, :fade] *= ramp
    m[:, -fade:] *= ramp[::-1]
    return m


def pink(n):
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    f /= np.sqrt(np.arange(1, len(f) + 1))
    p = np.fft.irfft(f, n)
    return (p / np.std(p)).astype(np.float32)


def write(name, stereo, ref=None, desc=""):
    (OUT / "input").mkdir(parents=True, exist_ok=True)
    (OUT / "ref").mkdir(parents=True, exist_ok=True)
    stereo = stereo / max(1.0, float(np.max(np.abs(stereo))) / 0.98)
    tmp = OUT / "input" / f"{name}.tmp.wav"
    A.save_wav(tmp, stereo, SR)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(tmp), "-c:a", "aac",
                    "-b:a", "128k", str(OUT / "input" / f"{name}.m4a")], check=True)
    tmp.unlink()
    if ref is not None:
        A.save_wav(OUT / "ref" / f"{name}.wav", ref, SR)
    return dict(clip=name, desc=desc, has_ref=int(ref is not None))


def mix(name, spk_files, mfile, offset, music_db, desc):
    s = speech(spk_files)
    floor = 10 ** (-62 / 20) * pink(len(s))      # еле слышный шум комнаты, как у живой записи
    m = music(mfile, offset, len(s))
    gain = active_rms(s) * 10 ** (music_db / 20) / (np.sqrt(np.mean(m ** 2)) + 1e-9)
    st = np.stack([s, s]) + gain * m + floor
    return write(name, st, s, desc)


def main():
    kz = lambda p, n=4: [f"{p}-{i}.wav" for i in range(1, n + 1)]  # noqa: E731
    rows = [
        mix("m01_kaztts2-f1_lofi_-10dB", kz("kaztts2-f1"), "raspberry_lofi.ogg", 20, -10,
            "KazakhTTS2 F1 + lo-fi инструментал, музыка на 10 дБ тише голоса"),
        mix("m02_kaztts2-m1_lofi_-6dB", kz("kaztts2-m1"), "sappheiros_perspective.ogg", 30, -6,
            "KazakhTTS2 M1 + lo-fi, громкая музыка (-6 дБ)"),
        mix("m03_kaztts2-f2_pop_-12dB", kz("kaztts2-f2"), "macleod_betyoucan.mp3", 40, -12,
            "KazakhTTS2 F2 + поп-инструментал Kevin MacLeod (-12 дБ)"),
        mix("m04_kazemo-madi_edm_-8dB",
            ["kazemo-madi-happy-1.wav", "kazemo-madi-neutral-1.wav", "kazemo-madi-happy-2.wav",
             "kazemo-madi-neutral-2.wav"], "audial_arabian.ogg", 60, -8,
            "KazEmoTTS Madi + EDM/трэп с тяжёлым басом (-8 дБ)"),
        mix("m05_kaztts2-f3_vocalpop_-10dB", kz("kaztts2-f3"), "cg5_fish.ogg", 48, -10,
            "KazakhTTS2 F3 + поп-песня С ПЕНИЕМ (англ.), -10 дБ — трудный случай"),
        mix("m06_kazemo-marzhan_rap_-10dB",
            ["kazemo-marzhan-neutral-1.wav", "kazemo-marzhan-happy-1.wav",
             "kazemo-marzhan-neutral-2.wav", "kazemo-marzhan-happy-2.wav"],
            "kidmc_patrao.wav", 8, -10,
            "KazEmoTTS Marzhan + рэп С ГОЛОСОМ (порт.), -10 дБ — самый трудный: рэп похож на речь"),
        mix("m07_kaztts2-m2_rap_-14dB", kz("kaztts2-m2"), "bobbyjimmy.ogg", 40, -14,
            "KazakhTTS2 M2 + рэп с голосом, тихо (-14 дБ)"),
        mix("m08_fleurs_lofi_-3dB", kz("fleurs-kk", 3), "raspberry_lofi.ogg", 70, -3,
            "FLEURS kk (чтение) + lo-fi, музыка почти как голос (-3 дБ)"),
    ]
    # реальные смеси без эталона: аудиопостановка на русском (CC BY-SA 4.0)
    for src, name in [("drama_plodi.opus", "r01_drama_plodi_ru"), ("drama_sorry.opus", "r02_drama_sorry_ru")]:
        x = A.load(MUSIC / src, SR, 2)
        rows.append(write(name, x, None, "Реальная аудиопостановка (рус.): речь поверх музыки, несколько персонажей"))

    # без музыки
    s = speech(["kazemo-akzhol-neutral-1.wav", "kazemo-akzhol-happy-1.wav",
                "kazemo-akzhol-neutral-2.wav", "kazemo-akzhol-happy-2.wav"])
    s = s + 10 ** (-62 / 20) * pink(len(s))
    rows.append(write("c01_kazemo-akzhol_clean", np.stack([s, s]), s, "Студийная запись, музыки нет"))
    s = speech(["cv-kk-1.mp3", "cv-kk-3.mp3", "cv-kk-4.mp3", "cv-kk-5.mp3"])
    rows.append(write("c02_commonvoice_kk", np.stack([s, s]), s, "Common Voice kk: бытовые микрофоны, разные люди"))
    s = speech(kz("ksd", 3))
    rows.append(write("c03_ksd_kk", np.stack([s, s]), s, "Kazakh Speech Dataset: телефон/ноутбук"))
    # диалог: два голоса впритык (паузы 0.15 с) + наложение 1 с
    a1, a2 = A.load(SPEECH / "kazemo-akzhol-neutral-1.wav", SR, 1), A.load(SPEECH / "kazemo-akzhol-neutral-2.wav", SR, 1)
    b1, b2 = A.load(SPEECH / "kazemo-marzhan-neutral-1.wav", SR, 1), A.load(SPEECH / "kazemo-marzhan-happy-1.wav", SR, 1)
    g = np.zeros(int(0.15 * SR), np.float32)
    d = np.concatenate([np.zeros(int(0.5 * SR), np.float32), a1, g, b1, g, a2])
    ov = int(1.0 * SR)
    d = np.concatenate([d, np.zeros(len(b2) - ov, np.float32)])
    d[-len(b2):] += b2
    d = np.concatenate([d, np.zeros(int(0.5 * SR), np.float32)])
    d = A.normalize_rms(d, -22, -3) + 10 ** (-62 / 20) * pink(len(d))
    rows.append(write("c04_dialog_two_voices", np.stack([d, d]), d,
                      "Диалог двух голосов впритык + 1 с наложения — должен отсеяться как «несколько говорящих»"))
    s = speech(kz("fleurs-kk", 4)[1:])
    nz = pink(len(s)) * active_rms(s) * 10 ** (-5 / 20)
    rows.append(write("n01_fleurs_noise_5dB", np.stack([s + nz, s + nz]), s,
                      "FLEURS kk + сильный шум (SNR 5 дБ) — должен отсеяться как «шум»"))

    with open(OUT / "testmix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"готово: {len(rows)} роликов в {OUT}")


if __name__ == "__main__":
    main()
