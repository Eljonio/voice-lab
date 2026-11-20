# voice-lab — Kazakh / Russian TTS fine-tuning for VoxCPM2

[![tests](https://github.com/Eljonio/voice-lab/actions/workflows/test.yml/badge.svg)](https://github.com/Eljonio/voice-lab/actions/workflows/test.yml)

Pipeline for teaching an open multilingual TTS model ([VoxCPM2](https://github.com/OpenBMB/VoxCPM), ~2B,
Apache-2.0) to speak natural conversational **Kazakh**, with Kazakh–Russian code-switching and Russian,
using LoRA on a single rented GPU. It covers the whole loop: cleaning raw speech, verbatim transcription
with two ASR systems, quality filters, building the training mix, training, independent evaluation and a
streaming serving stand for voice calls.

The base model mispronounces Kazakh-specific sounds (CER ~29–35 %, 82 % of phrases with CER > 10 %).
After fine-tuning, Kazakh CER is ~1–4 % mean / 0 % median on a 170-phrase set measured by an independent
CTC recogniser, with Russian kept at ≤ 0.5 %. Streaming synthesis gives first audio in ~0.2–0.3 s at
RTF 0.65–0.86 on an RTX PRO 6000. Method, numbers and failures: **[docs/METHOD.md](docs/METHOD.md)**.

> Status: research prototype. Not deployed to production. Question intonation in Kazakh is still wrong
> in places (stress drifts to the question particle, the last syllable goes up). Data and model weights
> are not part of this repository — see [What is not included and why](#what-is-not-included-and-why).

## Where to look first

| to check | open |
|---|---|
| the method, thresholds, numbers and what did not work | [`docs/METHOD.md`](docs/METHOD.md), sections 5–7 (evaluation protocol, results, failures) |
| two-ASR transcription and the filters | [`dataprep/process_v2.py`](dataprep/process_v2.py) |
| Kazakh / Russian text normalisation | [`dataprep/textnorm.py`](dataprep/textnorm.py), [`serve/textprep.py`](serve/textprep.py) |
| evaluation that does not depend on the labelling model | [`scripts/gigaam_eval.py`](scripts/gigaam_eval.py), [`eval/phrases_v2.json`](eval/phrases_v2.json) |
| the streaming TTS server and the call stand | [`serve/tts_server.py`](serve/tts_server.py), [`serve/call_server.py`](serve/call_server.py) |
| tests that run on CPU without any model | the commands under [Serving and the call stand](#serving-and-the-call-stand) |

**Provenance.** Exported on 2026-11-20 from a private working copy. Audio, transcripts, weights, logs and data collectors were removed, so the public history starts with the export.

## Pipeline

```
 raw audio (your own, with rights)          open corpora (KSC2, KazakhTTS2, FLEURS, Common Voice)
        │                                              │  voxcpm2/download_corpora.sh
        ▼                                              │
 1. clean        voicelab/ + run.py                    │
    AST music detector → (Demucs) → Silero VAD →       │
    DNSMOS / SNR / clipping → ECAPA one-speaker →      │
    Whisper + VoxLingua107 language ID                 │
        ▼                                              │
 2. transcribe   voxcpm2/transcribe.py (v1)            │
                 dataprep/process_v2.py (v2)           │
    Kazakh Whisper turbo + GigaAM CTC, punctuation     │
    transfer, hallucination and two-ASR filters,       │
    text normaliser dataprep/textnorm.py               │
        ▼                                              │
 3. re-cut       dataprep/recut_v2.py, cut_v4.py       │
    from the original audio, cut only at real pauses   │
        ▼                                              │
 4. finalize     dataprep/finalize_v2.py               │
    speaker ids, synthetic-voice filter, CTC           │
    confidence & speed outliers, held-out channels     │
        ▼                                              ▼
 5. mix          dataprep/build_mix_v2.py  (voxcpm2/build_manifest.py for v1)
    speaker cap 3 %, reference audio on 40 % of rows, style tags, corpus silence trimming
        ▼
 6. train        voxcpm2/remote_job.sh + voxcpm2/conf/*.yaml  (on the GPU server)
    LoRA (r=64 → r=128 on attention + MLP + LM↔DiT projections), eval-set synthesis per checkpoint
        ▼
 7. evaluate     scripts/gigaam_eval.py, score_run.py, sim_dnsmos.py, compare_*.py, listen_ab.py
    CER/WER by an independent CTC ASR, fail rate, Kazakh-letter errors, speaker similarity, DNSMOS,
    blind A/B listening
        ▼
 8. serve        serve/  — HTTP TTS (voice messages + chunked PCM streaming), LoRA merge,
                 browser call stand: mic → GigaAM ASR → dialogue backend → streaming TTS
```

## Repository layout

| path | what |
|---|---|
| `run.py`, `run.sh`, `voicelab/` | stage-1 cleaning pipeline (music detection, separation, VAD, DNSMOS, SNR, one-speaker check, language ID) |
| `scripts/` | stats and re-filtering of stage 1, synthetic test mixes, evaluation (GigaAM CER/WER, SIM + DNSMOS, run comparisons), blind A/B page |
| `scripts/dev/` | calibration of the language / speaker checks, analysis of Kazakh questions (forms and end-of-question F0) |
| `dataprep/` | v2 data processing: two-ASR transcription, re-cutting, finalisation, training-mix builder, text normaliser |
| `voxcpm2/` | v1 transcription and mix, training configs, on-GPU training job, baseline / eval synthesis, LoRA key check, corpus download |
| `eval/` | frozen evaluation phrase sets (`phrases.json` — 12 phrases, `phrases_v2.json` — 170 phrases + 5 dialogues) and their generator |
| `serve/` | TTS server, streaming helpers, text preparation for TTS, LoRA merge, low-VRAM loader, call stand and its tests |
| `testdata/music/LICENSES.csv` | Creative-Commons music used for synthetic test mixes (files are downloaded separately) |
| `docs/METHOD.md` | the method, thresholds, hyper-parameters, results and what did not work |

Code comments and console messages are mostly in Russian; documentation is in English.

## What you need

- **Python 3.12**, `ffmpeg`, an NVIDIA GPU. Stage-1 cleaning and evaluation fit a 4 GB laptop GPU
  (models are loaded one at a time); LoRA training used an RTX PRO 6000 96 GB (an A5000 24 GB works with
  batch 2 × 8 for r=64).
- **VoxCPM2** code and weights — [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) (pinned at commit
  `f0c787f`) and [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), both under the
  [Apache-2.0 license](https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE). Check the model card for
  usage terms before deploying a voice.
- **Other models** (downloaded from their hubs on first use): AST AudioSet classifier, Demucs htdemucs,
  Silero VAD, SpeechBrain ECAPA and VoxLingua107, faster-whisper (`medium`, `large-v3-turbo`),
  `shyngys879/kazakh-whisper-large-v3-turbo`, `ai-sage/GigaAM-Multilingual` (CTC), DNSMOS P.835 ONNX
  files from [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge) placed in
  `models/dnsmos/` (`sig_bak_ovr.onnx`, `model_v8.onnx`).
- **Your own data** that you have the right to use for training, and the open corpora downloaded by you
  under their licenses (KSC2 — MIT; KazakhTTS2 and FLEURS — CC-BY-4.0 with attribution; Common Voice — CC0).

Configuration is done with command-line arguments and environment variables; copy `.env.example` and
fill in what you use. Main variables: `VOICE_DS` (local dataset root, default `./data`), `VOICE_HOME`
(working root on the GPU/CPU server: weights, corpora, mixes, runs), `VOXCPM2_WEIGHTS`.

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
# VoxCPM (training / synthesis) and the evaluation stack go to separate venvs, see the steps below
```

## Running the stages

```bash
# 1. clean raw recordings (resumable; writes work/<clip>/result.json, clean/manifest.csv, clean/clips.csv)
./run.sh --input "$VOICE_DS/raw" --clean "$VOICE_DS/clean" --work "$VOICE_DS/work" --device cuda
./run.sh scripts/pilot_stats.py "$VOICE_DS/work"        # yield and rejection reasons
./run.sh scripts/refilter.py --clean "$VOICE_DS/clean" --work "$VOICE_DS/work"   # re-apply new thresholds without reprocessing

# 2a. v1 transcription → VoxCPM manifest (in a venv with voxcpm + faster-whisper)
voxcpm2/.venv/bin/python voxcpm2/transcribe.py --manifest "$VOICE_DS/clean/manifest.csv"

# 2b. v2 processing of raw clips (GPU worker; several workers share one queue)
python dataprep/process_v2.py --raw "$VOICE_HOME/raw" --out "$VOICE_HOME/v2" --order order.txt --device cuda
python dataprep/gate_v2.py --out "$VOICE_HOME/v2" --check
python dataprep/recut_v2.py --out "$VOICE_HOME/v2" --raw "$VOICE_HOME/raw" --keep "$VOICE_HOME/v2/manifest/ours_v2.jsonl" ...
python dataprep/finalize_v2.py --out "$VOICE_HOME/v2" --refs tts_refs.json

# 3. corpora and the training mix (on a CPU server with a persistent disk)
DATA="$VOICE_HOME/corpora" voxcpm2/download_corpora.sh --sizes
DATA="$VOICE_HOME/corpora" voxcpm2/download_corpora.sh ksc2 kazakhtts2 fleurs_kk fleurs_ru cv_kk
python dataprep/build_mix_v2.py --out "$VOICE_HOME/mix-v5" --ours ... --corpora "$VOICE_HOME/corpora"

# 4. training on the GPU server (VoxCPM code in $VOICE_HOME/code/VoxCPM, weights in $VOICE_HOME/weights/VoxCPM2)
python voxcpm2/lora_keys_check.py --pretrained "$VOICE_HOME/weights/VoxCPM2" --conf voxcpm2/conf/lora_v5_cont_pro6000.yaml --weights <warm-start lora_weights.safetensors>
V="$VOICE_HOME" RUN=lora-v5 MIX=mix-v5 CONF="$PWD/voxcpm2/conf/lora_v5_cont_pro6000.yaml" EPOCHS=1 bash voxcpm2/remote_job.sh

# 5. evaluation (GigaAM venv: torch + transformers + hydra-core)
python voxcpm2/synth_eval.py --phrases eval/phrases_v2.json --ref eval/ref_kk.wav --ckpt runs/lora-v5/step_0003479=out/v5_s1 --seed 1
.venv-asr/bin/python scripts/gigaam_eval.py sets v5_s1=out/v5_s1/ref --phrases eval/phrases_v2.json --out out/gigaam.json
python scripts/sim_dnsmos.py "$VOICE_HOME/runs/lora-v5" --ref eval/ref_kk.wav --out out/sim_dnsmos.json
python scripts/listen_ab.py build --a out/v4_s1/ref --b out/v5_s1/ref --name-a v4 --name-b v5   # blind A/B page
python scripts/listen_ab.py score listen-ab-v4-v5-<date>.json
```

Renting and tearing down GPU servers is provider-specific and is **not** part of this repository: the
scripts assume a Linux GPU box with the persistent working directory mounted at `$VOICE_HOME`.
`voxcpm2/remote_job.sh` is self-contained on that box (training → per-checkpoint synthesis → `EXIT`
file) so a disconnected laptop does not lose a run.

### Serving and the call stand

See the module docstrings in `serve/`. In short: `serve/tts_server.py` loads VoxCPM2 + LoRA (or weights
merged by `serve/merge_lora.py`) once and exposes `POST /tts` (Ogg/Opus voice message) and
`POST /tts/stream` (chunked PCM16 / μ-law); `serve/textprep.py` turns chat text into speakable chunks
(numbers to words in Kazakh/Russian, brand lexicon, sentence chunking). `serve/call_server.py` +
`serve/call.html` is a browser call stand (microphone → end-of-utterance detection → GigaAM CTC →
dialogue backend → streaming TTS) with per-turn latency logging. The dialogue backend is pluggable:
a fixed script, an echo, an OpenAI-compatible LLM endpoint, or an external HTTP bot (`BOT_TURN_URL`,
NDJSON protocol described in the `call_server.py` docstring; `serve/mock_bot.py` implements it).

Tests that need no GPU and no model:

```bash
cd serve
python3 textprep.py && python3 streamio.py && python3 guestlink.py   # self-tests
python3 merge_test.py                                               # LoRA merge on a toy model (torch, CPU)
python3 call_test_bot.py         # starts tts_server (TTS_MOCK=1) + mock_bot + call_server in a sandbox and checks the call contract
# mock_test.py / stream_test.py / call_test.py expect an already running server, e.g.:
#   TTS_MOCK=1 TTS_HOST=127.0.0.1 TTS_PORT=8011 TTS_HOME=/tmp/tts python3 tts_server.py &
#   python3 stream_test.py --url http://127.0.0.1:8011 --token-file /tmp/tts/token --log /tmp/tts/logs/tts.log ...
```

The plain HTTP setup is for testing only — put it behind TLS or a tunnel before exposing it.

## What is not included and why

| not included | why |
|---|---|
| raw and cleaned audio, transcripts, training manifests | the conversational speech came from publicly available videos of real people; publishing it would redistribute their voices and words without consent (a voice is personal data) and violate the platforms' terms |
| reference voices (`voices/*.wav`, `eval/ref*.wav`) and evaluation audio | derived from the model and the data above |
| fine-tuned LoRA / merged weights | trained on data that cannot be redistributed |
| open corpora | download them yourself from the original publishers to accept and honour their licenses and attribution requirements |
| logs, listening-test answers, run outputs | contain transcripts, local paths and personal notes |
| GPU rental / deployment scripts | provider- and account-specific |

To reproduce, bring your own speech data with clear rights (or use only the open corpora), run the
stages above and evaluate with the provided protocol. If you build a voice from a real person, get their
written consent first.

## License

Code: MIT, see [LICENSE](LICENSE). VoxCPM2 and every third-party model and corpus keep their own licenses.
