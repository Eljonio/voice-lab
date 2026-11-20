# Method: fine-tuning VoxCPM2 for conversational Kazakh (and Russian)

This document describes what was built, what worked, what did not, and the numbers behind those
conclusions. All figures come from the project's own run summaries, evaluation outputs and logs
(October 2026). Where a number is a plan or an estimate rather than a measurement, it says so.

## 1. Goal and constraints

- **Goal:** a natural-sounding, accent-free Kazakh voice (with Kazakh–Russian code-switching and
  plain Russian) for a phone/messenger assistant, built on an open model that can be used commercially.
- **Base model:** [VoxCPM2](https://github.com/OpenBMB/VoxCPM) (OpenBMB, Apache-2.0, ~2B parameters,
  48 kHz output, 16 kHz AudioVAE encoder), code pinned at commit `f0c787f`.
- **Voice:** the target timbre is *synthetic* — created with VoxCPM2 Voice Design (a text description
  of the voice), not cloned from a real person. The training mix caps every real speaker at ≤ 3 % so
  the model learns the language and speaking manner, not someone's identity.
- **Budget:** hourly-rented single GPUs (RTX PRO 6000 96 GB; RTX A5000 24 GB as a fallback) plus a
  laptop with a 4 GB RTX 3050 for data cleaning and light evaluation.

## 2. Baseline: what the base model does with Kazakh

| measurement | result |
|---|---|
| Russian, base model | clean (CER ≈ 0 %) |
| Kazakh, base model, 12-phrase set, CER by a Kazakh Whisper | **29.4 %** |
| Kazakh, base model, 170-phrase set × 2 seeds, CER by an independent CTC ASR (GigaAM) | **35.0 % mean / 24.2 % median**, 82 % of phrases with CER > 10 % |
| Kazakh-specific letters (ә ғ қ ң ө ұ ү һ і) recognised wrongly | 62.5 % |

Typical errors: Kazakh-only sounds replaced (`Қалай көмектесе` → `Ялай көмектісі`: Қ→Я, е→і),
endings changed (`-мін` → `-мен`). Voice Design itself worked out of the box.
Conclusion: Kazakh needs fine-tuning; Russian must not be broken by it.

## 3. Data pipeline

```
raw recordings ──► voicelab/  (stage 1 cleaning: music → [separation] → VAD → quality → one speaker → language)
               ──► voxcpm2/transcribe.py  or  dataprep/process_v2.py  (verbatim text + two-ASR agreement)
               ──► dataprep/recut_v2.py, cut_v4.py  (re-cut from the ORIGINAL audio with clean edges)
               ──► dataprep/finalize_v2.py  (speaker ids, synthetic-voice filter, CTC-confidence and speed outliers, held-out channels)
               ──► dataprep/build_mix_v2.py  (mix with open corpora, speaker cap, reference audio, style tags)
               ──► voxcpm2/remote_job.sh  (LoRA training on a rented GPU + synthesis of the eval set per checkpoint)
               ──► scripts/gigaam_eval.py, sim_dnsmos.py, compare_*.py, listen_ab.py  (independent evaluation)
```

### 3.1 Sources

- **Own conversational speech:** publicly available Kazakh-language videos (podcasts, interviews,
  vlogs, street polls), used for internal research only. Neither the audio nor the transcripts are
  redistributed (see the README).
- **Open corpora** (downloaded by `voxcpm2/download_corpora.sh`): KSC2 (ISSAI, MIT), KazakhTTS2
  (ISSAI, CC-BY-4.0), FLEURS kk/ru (Google, CC-BY-4.0), Common Voice kk (CC0). KSD (CC-BY-SA) and
  non-commercial sets were deliberately not used.

### 3.2 Stage-1 cleaning (`voicelab/`, thresholds in `voicelab/pipeline.py`)

| step | model | rule |
|---|---|---|
| music / singing detection | AST (AudioSet) on 3 s windows, 1.5 s hop | window "musical" at p ≥ 0.25; clip has music if ≥ 10 % of windows or any window ≥ 0.5 |
| separation (optional) | Demucs htdemucs, 4-min chunks with 2 s overlap | phrase rejected if music or singing left > 0.30 |
| segmentation | Silero VAD | phrases 2–15 s |
| quality | DNSMOS P.835 (ONNX) | OVRL ≥ 2.7, SIG ≥ 3.0, BAK ≥ 2.8 |
| noise | percentile SNR estimate | ≥ 12 dB; clipping ≤ 0.5 % of samples |
| one speaker | ECAPA embeddings of 1.5–3 s windows | min pairwise cosine ≥ 0.15; speakers per clip by agglomerative clustering (0.35) |
| language | Whisper + VoxLingua107 averaged; near-Turkic languages pooled into a "Kazakh-like" score | keep kk / ru / mix |

A synthetic test set (`scripts/make_testmix.py`: open speech + Creative-Commons music at known SNR)
makes separation measurable with SI-SDR instead of by ear.

**First subset, measured:** 14.0 h raw → 12.0 h speech (VAD) → 8.53 h after cleaning (61 % of raw;
rejected: DNSMOS 11 %, noise 8 %, two voices 4 %, music 4 %) → **7.9 h, 3,604 phrases, 98 speakers**
after transcription filters (two-ASR disagreement 5 %, repetitions 2 %). Cleaning ran at RTF 0.23–0.45
on the laptop GPU; transcription at ≈ 36 phrases/min (int8_float16).

### 3.3 Transcription

- Main ASR: `shyngys879/kazakh-whisper-large-v3-turbo` (Apache-2.0; author-reported FLEURS kk WER 11.8 %
  vs 19.8 % for the original turbo). On 10 held-out Kazakh phrases we measured WER 20.4 % / CER 9.9 %,
  of which one phrase was a Whisper hallucination (the sentence repeated twice); without it
  **WER 10.3 % / CER 1.8 %**.
- The Kazakh model outputs lowercase text without punctuation, which TTS needs for pauses and questions.
  A second pass with the original turbo is word-aligned (difflib) and only punctuation, capitalisation
  and fillers are transferred.
- Hallucination filters: `avg_logprob < −1`, `compression_ratio > 2.4` or a repeated 4-gram,
  `no_speech_prob > 0.6`, speed outside 2.5–25 chars/s, any digits (text must be "as spoken").
- **Two-ASR agreement.** v1: CER between the two Whisper passes > 0.5 → reject (0.25 rejected correct
  phrases because the original turbo is weak in Kazakh). v2 (`dataprep/process_v2.py`): words from
  GigaAM-Multilingual CTC, second opinion from the Kazakh turbo, **CER > 0.25 → reject**; CTC word
  timestamps also reject clipped words (last word must end ≥ 80 ms before the clip end, first word start
  ≥ 40 ms after the start); the bottom 7 % by CTC confidence and per-channel speed outliers (1.5 IQR)
  are dropped in `finalize_v2.py`.
- One text normaliser (`dataprep/textnorm.py`) is shared by training data and inference: digits,
  Latin and symbols make a row ineligible for training; brackets are removed (a leading bracket is a
  Voice Design command in VoxCPM2), CAPS and Latin/Cyrillic homoglyphs are fixed.

### 3.4 Cutting rules (the most frequent source of defects)

| | v1 | v3 (`recut_v2.py`) | v4–v5 (`cut_v4.py`, "full speech") |
|---|---|---|---|
| unit | VAD phrase, forced split > 15 s at the quietest point | VAD phrases, sometimes 2–3 glued | **all speech between pauses ≥ 0.8 s**; shorter pauses and breaths kept inside |
| max length | 15 s | 25 s | **30 s** (model limit); longer → split at the longest pause ≥ 0.6 s (else ≥ 0.4 s), otherwise dropped |
| margins | 0.15 s | from the real pause, ≤ 0.2 / 0.3 s | **0.3 s of the real pause** |
| edges | no fade (clicks) | 15 ms fade | 15 ms fade, edge samples = 0; speech touching the boundary → dropped |
| separation | Demucs on about half the clips | Demucs audio not used | off |

### 3.5 Training mix (`dataprep/build_mix_v2.py`)

- Speaker cap ≤ 3 % of the mix; clip 1–25 s, ≤ 25 chars/s; clip + reference ≤ 30 s.
- `ref_audio` on 40 % of rows (VoxCPM2 docs recommend 30–50 % to keep cloning ability); in v4–v5,
  30 % of rows use the **tail of the previous phrase** of the same speaker as the reference, which
  teaches continuation across sentences.
- Style tags in brackets on 15 % of rows ("(live conversational speech)" / "(read speech)") as an
  inference-time style handle.
- Corpus silences trimmed to ≤ 0.15 s at the start and ≤ 0.25 s at the end (long trailing silence
  teaches the model not to stop).
- Validation: whole held-out channels, not a random 1 %.

| mix | hours | rows | conversational / read | Russian | code-switch |
|---|---|---|---|---|---|
| v1 (trial) | 26.3 | 11,801 | 30 % / 70 % | 8 % (read) | 1 % |
| v3 (mix-v3b) | 148.9 | 48,542 | own 111.2 h + 37.9 h corpora | | |
| v4 | 183.2 | 57,286 | 81.7 % / 18.3 % | 4.2 % | 2.8 % |
| v5 | 188.9 | 55,667 | 91.3 % / 8.7 % | 4.1 % | 3.1 % |

The bulk processing kept 19–25 % of raw audio for training (measured on v2/v3).

## 4. Training runs

All runs use the official trainer (`scripts/train_voxcpm_finetune.py` of VoxCPM) with
`sample_rate 16000` (the AudioVAE encoder rate — 48000 produces garbage), effective batch 16,
weight decay 0.01, grad-norm 1.0. Configs: `voxcpm2/conf/`.

| run | what | lr | steps |
|---|---|---|---|
| v1 | LoRA r=64/α=64 on LM + DiT attention, 26 h, 4 epochs | 1e-4, warmup 200 | 2,950 |
| v2 | same LoRA, ~147 h, 2 epochs | 1e-4, warmup 300 | 6,164 |
| probe A/B/C | A = v2; **B = extended LoRA r=128/α=128 on attention + MLP + LM↔DiT projections (~210 M trainable params, ~9 %)**; C = full fine-tune | B 1e-4, C 1e-5 | 2,000 each |
| v3 | extended LoRA (B) on mix-v3b, 2 epochs | 1e-4 | 6,067 |
| v4 | warm start from v3 LoRA weights, new cutting rule | 5e-5 | 3,580 |
| v5 | warm start from v4, less read speech, questions up-weighted ×3, batch 4×4 | 1e-4 | 3,479 |

Checkpoints are chosen by evaluation and blind listening, not by validation loss (the diffusion loss
barely moves and does not separate checkpoints).

## 5. Evaluation protocol

- **Independent ASR:** GigaAM-Multilingual large CTC (MIT) — not the model that labelled the data, no
  language model, no prompt. Whisper "corrects" accented speech with its language model; CTC writes
  what it hears. The first evaluation with the labelling Whisper was biased and was replaced.
- **Eval set** `eval/phrases_v2.json`: 170 phrases — 100 Kazakh (special letters and vowel harmony,
  questions, numbers spoken as words, short replies, long 25–40-word sentences, assistant replies,
  emotional lines), 20 code-switched, 20 Russian, 10 names/brands, 5 four-turn dialogues; each synthesised
  with 2 seeds, cfg 2.0, 10 diffusion steps, one Kazakh reference voice.
- **Metrics per checkpoint:** mean/median CER per language, **fail rate (share of phrases with
  CER > 10 %)**, missing last word, errors on Kazakh-specific letters and the share replaced by the
  Russian look-alike (қ→к, ң→н, ө→о, …), duration per character (runaway / truncated), speaker
  similarity to the reference (ECAPA), DNSMOS (relative only).
- **Noise floor:** with ~120 Kazakh phrases, a difference below ~2 pp CER between checkpoints is noise
  (the same weights differ by up to 2 pp between seeds).
- **Blind A/B listening** (`scripts/listen_ab.py`): same phrase, two versions, sides randomised, key
  hidden until scoring, plus trap pairs (identical clips). Native-speaker ears decide stress and melody,
  which CTC cannot hear.

## 6. Results

GigaAM CTC on `phrases_v2`, Kazakh reference, mean of seeds 1 and 2:

| version | kk CER (mean) | kk fail rate | special-letter errors | short replies CER | mix CER | ru CER | SIM | DNSMOS OVRL |
|---|---|---|---|---|---|---|---|---|
| base | 35.0 % | 82.1 % | 62.5 % | 55.3 % | 23.4 % | 0.1 % | 0.645 | 2.82 |
| v2 (LoRA r=64) | 2.1 % | 2.1 % | 1.2 % | 16.0 % | 1.1 % | 0.5 % | 0.651 | 3.02 |
| probe B (r=128 + MLP, 2k steps) | 1.1 % | 1.2 % | 0.9 % | 4.4 % | 1.1 % | 0.2 % | 0.657 | 3.04 |
| probe C (full FT, 2k steps) | 2.0 % | 1.7 % | 1.5 % | — | 1.9 % | 0.1 % | 0.661 | 3.07 |
| v3, step 6067 | 1.0 % | 1.2 % | 0.6 % | 3.3 % | 1.2 % | 0.5 % | 0.651 | 3.01 |

Later runs (same protocol, seed 1 / seed 2): v4 step 3580 — kk CER 3.2 % / 0.8 %, ru 0.5 % / 0.1 %;
v5 step 3479 — kk CER 3.7 % / 0.7 %, ru 0.1 % / 0.3 %, fail rate 2.5 % / 0.8 %. Median kk CER is 0 %
for the final checkpoints of v2–v5 and the probes, so the mean is driven by a few failures.

**Kazakh CER dropped from ~29–35 % to roughly 1–4 % (median 0 %) after fine-tuning, while Russian
stayed at ≤ 0.5 %.**

Blind listening decided between versions where CER could not: extended LoRA B beat v2 6/6 in the
probe; v3 vs v4 was a tie (8 : 6, 19 equal); v5 was preferred over v4 (14 : 9, 11 equal) and became the
serving checkpoint.

### 6.1 Streaming latency (serving stand, `serve/`)

RTX PRO 6000, v5 LoRA merged into the base weights (`serve/merge_lora.py`: W' = W + α/r·B·A),
chunked PCM streaming (`POST /tts/stream`):

| configuration | time to first audio (median) | RTF (median) | kk CER (mean) |
|---|---|---|---|
| whole-utterance synthesis, LoRA | 3.0 s | 0.90 | 0.5 % |
| streaming, LoRA layers | 303 ms | 0.93 | 0.2 % |
| **streaming, merged weights, 10 steps** | **209 ms** | **0.65** | 0.0 % |
| streaming, merged, 6 steps | 153 ms | 0.47 | 1.1 % (5 % fails) |

Live test calls (two sessions, 74 synthesised replies, merged weights): first audio median
**0.23–0.26 s** (p90 0.27–0.31 s), RTF median 0.68–0.73 (max 0.86). In a call this is TTS only — ASR,
the dialogue backend and the network add to it.

## 7. What did not work (and what replaced it)

1. **Whisper prompt with real filler words.** An `initial_prompt` containing words like «жаңағы»
   made the ASR insert them: 348 occurrences in the v1 subset vs 21 in a prompt-less draft. → No prompt
   (or fillers only), a CTC model as the second opinion.
2. **Evaluating with the labelling ASR.** Biased CER (the language model hides accent; the prompt adds
   fillers), only 12 phrases. → Independent CTC evaluator, 170 phrases × 2 seeds, fail rate as the
   headline metric.
3. **Forced cuts inside speech.** Splitting long segments at the quietest point produced truncated
   words; ~7 % of v1 clips ended mid-thought. The model learns to "eat" word endings. → Cut only at real
   pauses, drop what cannot be cut, later "full speech" units between ≥ 0.8 s pauses.
4. **Hard edges.** Cutting without a fade left an audible click at 15 of 40 sampled phrase ends. →
   Margins from the real pause + 15 ms raised-cosine fade.
5. **Source separation.** Demucs leaves artifacts the model can learn; and it was rarely needed — in
   clips flagged as "musical", 91 % of phrases had music probability < 0.30 (median 0.02), while Demucs
   took 28 % of processing time. → Off by default; phrases with real background music are dropped, not
   repaired.
6. **Read-speech-heavy trial mix.** v1 was 70 % read speech, corpus rows without speaker ids escaped
   the 3 % cap, and only 21 % of rows had a reference. → Live speech first, speaker ids for corpora,
   40 % references.
7. **Lower learning rate on continuation.** v4 at lr 5e-5 was audibly no different from v3 (blind tie).
   → v5 back to 1e-4.
8. **Cutting read speech too far.** v5 reduced read speech from 18.3 % to 8.7 %. On 60 assistant-style
   phrases and questions in the serving voice, the Kazakh fail rate rose from 0 % (v4) to 5.2 % (v5), with
   Russian-like substitutions on some Kazakh sounds. Read Kazakh speech seems to anchor phonemes;
   our working hypothesis is not to go below ~12–15 % without replacing it with something equivalent.
9. **Question intonation by data weighting.** Up-weighting question sentences ×3 (v5) did not fix the
   main remaining defect by ear (see §8).
10. **Full fine-tuning vs LoRA at this data size.** The full fine-tune probe was not better than the
    extended LoRA at 2,000 steps and costs much more disk and memory (a full checkpoint with optimizer
    state ≈ 27.5 GB). Extended LoRA was kept.

Engineering lessons from rented GPUs (provider-independent):

- Blackwell cards need the driver with open kernel modules; check `nvidia-smi -L` before paying for
  anything else.
- Deleting a cloud server is a power cut: files written seconds earlier were empty. → `sync` and
  unmount the persistent disk before every teardown; the job writes an `EXIT` file last.
- Re-issued IP addresses break SSH host keys → remove the old key on every new server.
- Never edit a bash script that is currently running (bash reads it by offset) — replace it atomically.
- `load_state_dict(strict=False)` in the trainer silently skips mismatched keys, so a warm start can
  quietly become training from scratch → `voxcpm2/lora_keys_check.py` verifies every key first.
- VoxCPM re-seeds its generator unless the seed is passed to `generate()`; until that was fixed,
  "seeded" samples were not reproducible.
- On OOM the job halves the batch and doubles accumulation (same effective batch, same lr) and resumes
  from `latest/`; optimizer state is pruned from all but the newest checkpoint.

## 8. Status and open problems

- **Not in production.** Serving was tested on an hourly GPU (HTTP API for voice messages, streaming
  endpoint, browser call stand with ASR → dialogue backend → streaming TTS). TLS/tunnelling, a
  permanent GPU and multi-stream serving are not done.
- **Question intonation is still wrong in places:** stress drifts onto the question particle
  («…ма/ме/ба/бе?») and the last syllable is pushed up, where native speakers keep the pitch and put the
  peak earlier. `scripts/dev/kk_question_f0.py` measures the end-of-question F0 contour against live
  speech; the fix is expected from data (more natural questions), not from text tricks.
- **Short standalone replies** («Иә.», «Рахмет!») are unstable between checkpoints (CER 3.3–31 % in v3
  snapshots): the conversational data contains few of them as separate utterances (~1.9 per hour).
- **CPU inference is not viable** for VoxCPM2 (base model on a laptop CPU: RTF ≈ 28). A distilled small
  student model (e.g. Kokoro-82M trained on filtered synthetic speech) was planned, not built.
- Code-switched and Russian speech are measured on small sets (20 + 20 phrases).

## 9. References

- VoxCPM / VoxCPM2: https://github.com/OpenBMB/VoxCPM, https://huggingface.co/openbmb/VoxCPM2,
  fine-tuning guide https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html
- KSC2 (ISSAI): https://huggingface.co/datasets/issai/Kazakh_Speech_Corpus_2 · KazakhTTS2: https://github.com/IS2AI/Kazakh_TTS
- FLEURS: https://huggingface.co/datasets/google/fleurs · Common Voice: https://commonvoice.mozilla.org
- Kazakh Whisper turbo: https://huggingface.co/shyngys879/kazakh-whisper-large-v3-turbo
- GigaAM: https://github.com/salute-developers/GigaAM
- Emilia data pipeline (filtering ideas): https://arxiv.org/abs/2407.05361
- DNSMOS P.835: https://github.com/microsoft/DNS-Challenge
- Demucs: https://github.com/facebookresearch/demucs · Silero VAD: https://github.com/snakers4/silero-vad ·
  SpeechBrain ECAPA / VoxLingua107: https://speechbrain.github.io
- Nano-vLLM-VoxCPM: https://github.com/a710128/nanovllm-voxcpm
