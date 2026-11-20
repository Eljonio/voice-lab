#!/usr/bin/env python3
"""lowvram.py — VoxCPM2 + LoRA на ноутбуке с 4 ГБ видеопамяти (RTX 3050 Laptop).

Зачем: база VoxCPM2 в bf16 — 4,58 ГБ (2,29 млрд параметров), в 4 ГБ карты не влезает. Линейные слои (2,13 млрд из 2,29) храним
в int8 «по строке» (симметрично, scale = max|W_строки| / 127), считаем в bf16: y = (x · Wq_bf16ᵀ) · scale. Это weight-only int8,
как torchao int8_weight_only, но без зависимостей. Эмбеддинги, нормы, мелкие проекции и AudioVAE — как в обычной загрузке.

LoRA v5 вливается ДО квантования, в fp32 (W' = W + alpha/r · B·A, как serve/merge_lora.py), — квантуем уже влитые веса.

  lowvram.py prep  --base voxcpm2/weights/VoxCPM2 --lora voxcpm2/runs/lora-v5/step_0003479 --out voxcpm2/weights/v5-3479-q8
  lowvram.py check --dir voxcpm2/weights/v5-3479-q8          # ошибка квантования по слоям (из q8_info.json)

Загрузка в коде:  from lowvram import load_q8;  tts = load_q8(dir, device="cuda")  → объект VoxCPM (tts.generate(...) как обычно).
Варианты (переменные окружения или аргументы): keep — группы, которые НЕ квантовать при загрузке, если они сохранены и в bf16 (нет);
max_length — длина KV-кэша (по умолчанию из config.json, 8192).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# где квантуем nn.Linear. DiT (feat_decoder, 212 млн) по умолчанию НЕ квантуем: он идёт 10 раз за шаг (шаги диффузии), распаковка
# int8 → bf16 на каждом вызове — главная трата времени; в bf16 он +0,21 ГБ, после VAE кусками место есть (замер 12.10).
GROUPS = ("base_lm.", "residual_lm.", "feat_encoder.")
ALL_GROUPS = ("base_lm.", "residual_lm.", "feat_encoder.", "feat_decoder.")
MIN_NUMEL = 1_000_000                                                     # мелкие проекции (in_proj 64→1024 и т. п.) — bf16
QFILE = "model_q8.safetensors"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def is_qtarget(key: str, shape, groups=GROUPS) -> bool:
    return (key.endswith(".weight") and key.startswith(tuple(groups)) and len(shape) == 2 and "embed" not in key
            and shape[0] * shape[1] >= MIN_NUMEL)


def sha256(p: Path, n=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(n):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- запись safetensors потоком (не держим 2,4 ГБ в памяти)
_DT = {"I8": 1, "BF16": 2, "F16": 2, "F32": 4}


def _st_dtype(t):
    import torch
    return {torch.int8: "I8", torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}[t.dtype]


def write_stream(path: Path, plan: list[tuple[str, str, list[int]]], produce):
    """plan: [(ключ, dtype, форма)] в порядке записи; produce(ключ) → тензор ровно этой формы/типа. Формат — safetensors 0.x."""
    import torch
    off, hdr = 0, {"__metadata__": {"format": "pt"}}
    for k, dt, shp in plan:
        n = _DT[dt]
        for s in shp:
            n *= s
        hdr[k] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for k, dt, shp in plan:
            t = produce(k).contiguous()
            assert _st_dtype(t) == dt and list(t.shape) == shp, (k, t.dtype, t.shape, dt, shp)
            f.write(t.view(-1).view(torch.uint8).numpy().tobytes())
    return off


def cmd_prep(a):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(int(os.environ.get("PREP_THREADS", "4")))
    base, lora, out = Path(a.base), Path(a.lora) if a.lora else None, Path(a.out)
    groups = tuple(g if g.endswith(".") else g + "." for g in a.groups.split(","))
    bf = base / "model.safetensors"
    sb = safe_open(str(bf), "pt")
    bkeys = list(sb.keys())
    scaling, mods, sl, lc = 0.0, {}, None, None
    if lora:
        lc = json.loads((lora / "lora_config.json").read_text())["lora_config"]
        scaling = float(lc["alpha"]) / int(lc["r"])
        sl = safe_open(str(lora / "lora_weights.safetensors"), "pt")
        for k in sl.keys():
            for suf in (".lora_A", ".lora_B"):
                if k.endswith(suf):
                    mods.setdefault(k[: -len(suf)], set()).add(suf)
        bad = [m for m, s in mods.items() if s != {".lora_A", ".lora_B"} or m + ".weight" not in set(bkeys)]
        if bad:
            log("ОШИБКА: модули LoRA без пары или без базового веса:", bad[:5])
            return 1
        log(f"LoRA {lora}: r={lc['r']} alpha={lc['alpha']} scaling={scaling:g}, модулей {len(mods)}")
    plan, qkeys = [], []
    for k in bkeys:
        shp = list(sb.get_slice(k).get_shape())
        if is_qtarget(k, shp, groups):
            m = k[: -len(".weight")]
            plan += [(m + ".weight_q", "I8", shp), (m + ".weight_scale", "F32", [shp[0]])]
            qkeys.append(k)
        else:
            plan.append((k, sb.get_slice(k).get_dtype(), shp))
    lora_not_q = [m for m in mods if m + ".weight" not in qkeys]
    log(f"база {len(bkeys)} ключей; в int8 — {len(qkeys)} матриц; LoRA ложится на неквантуемые: {len(lora_not_q)} {lora_not_q[:3]}")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    stats, cache = [], {}
    gen = torch.Generator().manual_seed(0)

    def merged(k):
        W = sb.get_tensor(k)
        m = k[: -len(".weight")]
        if m in mods:
            A = sl.get_tensor(m + ".lora_A").float()
            B = sl.get_tensor(m + ".lora_B").float()
            return W, W.float() + scaling * (B @ A), True
        return W, W.float(), False

    def produce(key):
        if key.endswith(".weight_scale"):
            return cache.pop(key)
        if key.endswith(".weight_q"):
            m = key[: -len(".weight_q")]
            Wb, W32, had = merged(m + ".weight")
            s = (W32.abs().amax(dim=1) / 127.0).clamp(min=1e-12)
            q = torch.round(W32 / s[:, None]).clamp_(-127, 127).to(torch.int8)
            cache[m + ".weight_scale"] = s.float()
            # ошибка: влитые bf16 (как сервер) против int8, к fp32-эталону, на случайном входе
            x = torch.randn(8, W32.shape[1], generator=gen)
            y = x @ W32.t()
            yb = x.bfloat16().float() @ W32.bfloat16().float().t()
            yq = (x.bfloat16().float() @ q.float().t()) * s
            n = y.norm().item() + 1e-12
            stats.append(dict(m=m, lora=had, err_bf16=(yb - y).norm().item() / n, err_q8=(yq - y).norm().item() / n))
            return q
        if key.endswith(".weight") and key[: -len(".weight")] in mods:   # LoRA на неквантуемом слое — влить, оставить bf16
            W, W32, _ = merged(key)
            return W32.to(W.dtype)
        return sb.get_tensor(key)

    t = time.time()
    nbytes = write_stream(tmp / QFILE, plan, produce)
    for f in base.iterdir():
        if f.is_file() and f.name != "model.safetensors" and not f.name.startswith("."):
            shutil.copy2(f, tmp / f.name)
    eb = [s["err_bf16"] for s in stats]
    eq = [s["err_q8"] for s in stats]
    worst = sorted(stats, key=lambda s: -s["err_q8"])[:5]
    info = dict(made=time.strftime("%Y-%m-%d %H:%M:%S"), base=str(base), lora=str(lora) if lora else None, lora_config=lc,
                scaling=scaling, scheme="int8 weight-only, per-row symmetric, scale=max|W|/127 (fp32), compute bf16",
                groups=list(groups), min_numel=MIN_NUMEL, q_matrices=len(qkeys), bytes=nbytes,
                lora_sha256=sha256(lora / "lora_weights.safetensors") if lora else None,
                err_bf16_median=statistics.median(eb), err_q8_median=statistics.median(eq), err_q8_max=max(eq),
                worst=worst, per_layer=stats)
    (tmp / "q8_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    log(f"записано {out}/{QFILE}: {nbytes / 1e9:.2f} ГБ за {time.time() - t:.0f} с; ошибка выхода слоя к fp32: "
        f"bf16 медиана {statistics.median(eb) * 100:.3f} %, int8 медиана {statistics.median(eq) * 100:.3f} % (макс {max(eq) * 100:.3f} %)")
    return 0


# ---------------------------------------------------------------- модель
def _qlinear_cls():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class QLinear(nn.Module):
        """int8-веса по строке; вход bf16/fp32 → выход того же типа."""

        def __init__(self, in_features, out_features, bias):
            super().__init__()
            self.in_features, self.out_features = in_features, out_features
            self.register_buffer("weight_q", torch.empty(out_features, in_features, dtype=torch.int8))
            self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16), requires_grad=False) if bias else None
            self._s = None

        def forward(self, x):
            if self._s is None or self._s.dtype != x.dtype or self._s.device != x.device:
                self._s = self.weight_scale.to(device=x.device, dtype=x.dtype)
            y = F.linear(x, self.weight_q.to(x.dtype)) * self._s
            if self.bias is not None:
                y = y + self.bias.to(x.dtype)
            return y

        def extra_repr(self):
            return f"in={self.in_features}, out={self.out_features}, int8"

    return QLinear


def _cpu_embedding_cls():
    import torch.nn as nn
    import torch.nn.functional as F

    class CPUEmbedding(nn.Module):
        """Таблица текстовых эмбеддингов (73 448 × 2048 bf16 = 300 МБ) остаётся в ОЗУ: она нужна один раз на фразу (префилл)."""

        def __init__(self, weight):
            super().__init__()
            self.w = weight            # не параметр и не буфер — model.to(cuda) её не трогает

        def forward(self, ids):
            return F.embedding(ids.to("cpu"), self.w).to(ids.device)

    return CPUEmbedding


def chunked_vae_decode(vae, frames=25):
    """VAE-декод не всей фразы разом (активации на 48 кГц растут с длиной: 4,6 с → OOM на 3,68 ГиБ), а по `frames` латентов
    (25 = 1 с) через StreamingVAEDecoder библиотеки (каузальные свёртки с переносом состояния). Проверено 12.10 на CPU:
    кусками 25/13/4 против целиком — макс. разница 2e-7…1,4e-6 (округление fp32), длина та же."""
    import torch
    from voxcpm.modules.audiovae.audio_vae_v2 import StreamingVAEDecoder
    orig = vae.decode

    def decode(z, sr_cond=None):
        if z.shape[-1] <= frames or getattr(vae, "_chunk_streaming", False):
            return orig(z, sr_cond)
        outs = []
        vae._chunk_streaming = True
        try:
            with StreamingVAEDecoder(vae):
                for i in range(0, z.shape[-1], frames):
                    outs.append(orig(z[..., i:i + frames], sr_cond))
        finally:
            vae._chunk_streaming = False
        return torch.cat(outs, dim=-1)

    vae.decode = decode


def chunked_vae_encode(vae, samples=640 * 16):
    """VAE-кодирование эталона/продолжения не целиком (12,5 с эталона d20_kk = +0,75 ГиБ на пике — с моделью это 3,38 из 3,68),
    а кусками по `samples` (кратно hop 640 и патчу 2560) с переносом состояния каузальных свёрток — тот же приём, что
    StreamingVAEDecoder библиотеки, только для кодера (у него тоже все свёртки каузальные: левое дополнение = (K−1)·dil,
    у страйдовых дополнение = шаг, ядро = 2·шаг). Проверено 12.10 на CPU: кусками 1280/5120/16000 против целиком —
    относительная разница 3e-7…6e-6 (шум fp32), форма та же."""
    import torch
    from voxcpm.modules.audiovae.audio_vae_v2 import CausalConv1d, StreamingVAEDecoder

    class _EncStream(StreamingVAEDecoder):
        def _install(self):
            for _, mod in self._vae.encoder.named_modules():
                if isinstance(mod, CausalConv1d):
                    pad = mod._CausalConv1d__padding * 2 - mod._CausalConv1d__output_padding
                    if pad > 0:
                        self._patch_causal_conv(mod, pad)

    def encode(audio_data, sample_rate):
        if audio_data.ndim == 2:
            audio_data = audio_data.unsqueeze(1)
        x = vae.preprocess(audio_data, sample_rate)
        if x.shape[-1] <= samples:
            return vae.encoder(x)["mu"]
        with _EncStream(vae):
            mus = [vae.encoder(x[..., i:i + samples])["mu"] for i in range(0, x.shape[-1], samples)]
        return torch.cat(mus, dim=-1)

    vae.encode = encode


def cache_ref_latents(tm, roots):
    """Латенты эталонов голоса (файлы из roots: voices/, eval/) считаются один раз: _encode_wav детерминирован (mu, без шума)."""
    orig = tm._encode_wav
    memo = {}
    roots = [str(Path(r).resolve()) for r in roots]

    def enc(wav_path, padding_mode="right", trim_silence_vad=False):
        p = Path(wav_path).resolve()
        if not any(str(p).startswith(r + os.sep) for r in roots):
            return orig(wav_path, padding_mode=padding_mode, trim_silence_vad=trim_silence_vad)
        st = p.stat()
        k = (str(p), st.st_size, st.st_mtime_ns, padding_mode, trim_silence_vad)
        if k not in memo:
            memo[k] = orig(wav_path, padding_mode=padding_mode, trim_silence_vad=trim_silence_vad)
        return memo[k].clone()

    tm._encode_wav = enc


def load_q8(d, device="cuda", max_length=2048, embed_cpu=True, vae_frames=25, enc_samples=640 * 16, verbose=True):
    """Собрать VoxCPM2 из папки prep (int8) → объект voxcpm.VoxCPM (generate/generate_streaming как обычно).
    Сборка — на устройстве meta (ОЗУ не тратится; иначе пик 7,2 ГиБ и своп), веса подставляются из файла как есть (assign).
    max_length — длина статического KV-кэша: 2048 вместо 8192 (−226 МБ видеопамяти); внимание идёт по маске, длина кэша на
    результат не влияет; наши куски ≤ 180 знаков занимают ≤ ~1 500 позиций (эталон + продолжение + текст + звук)."""
    import torch
    from safetensors import safe_open
    from transformers import LlamaTokenizerFast
    from voxcpm import VoxCPM
    from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
    from voxcpm.modules.audiovae import AudioVAEV2
    from voxcpm.modules.minicpm4.model import MiniCPMLongRoPE

    d = Path(d)
    t0 = time.time()
    config = VoxCPMConfig.model_validate_json((d / "config.json").read_text())
    if max_length:
        config.max_length = int(max_length)
    tok = LlamaTokenizerFast.from_pretrained(str(d))
    vae = AudioVAEV2(config=config.audio_vae_config)          # настоящий, fp32 (как from_local)
    ck = torch.load(d / "audiovae.pth", map_location="cpu", weights_only=True)
    r = vae.load_state_dict(ck.get("state_dict", ck), strict=True)
    del ck
    QLinear = _qlinear_cls()
    so = safe_open(str(d / QFILE), "pt")
    keys = set(so.keys())
    qmods = sorted(k[: -len(".weight_q")] for k in keys if k.endswith(".weight_q"))
    with torch.device("meta"):                                 # KV-кэш setup_cache кладёт на device явно — он настоящий
        model = VoxCPM2Model(config, tok, vae, None, device=device)
        for m in qmods:
            parent, _, name = m.rpartition(".")
            pm = model.get_submodule(parent)
            old_l = getattr(pm, name)
            setattr(pm, name, QLinear(old_l.in_features, old_l.out_features, old_l.bias is not None))
    own = model.state_dict()
    want = {k for k in own if not k.startswith("audio_vae.")}
    miss, extra = sorted(want - keys), sorted(keys - want)
    if miss or extra:
        raise RuntimeError(f"ключи q8 не сходятся с моделью: нет {len(miss)} {miss[:3]}, лишние {len(extra)} {extra[:3]}")
    sd = {k: so.get_tensor(k) for k in want}
    bad = [k for k in sd if tuple(sd[k].shape) != tuple(own[k].shape)]
    if bad:
        raise RuntimeError(f"форма не та: {bad[:3]}")
    model.load_state_dict(sd, strict=False, assign=True)
    del sd, own
    # буферы RoPE не хранятся в файле (persistent=False) — считаем заново, как в __init__ (fp32), и в bf16, как model.to(bf16)
    for mod in model.modules():
        if isinstance(mod, MiniCPMLongRoPE):
            fresh = MiniCPMLongRoPE(mod.config)
            for b in ("inv_freq", "cos_cached", "sin_cached"):
                setattr(mod, b, getattr(fresh, b).to(torch.bfloat16))
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    left += [f"{type(m).__name__}.{k}" for m in model.modules() for k, v in vars(m).items() if isinstance(v, torch.Tensor) and v.is_meta]
    if left:
        raise RuntimeError(f"остались тензоры без данных (meta): {left[:5]}")
    dts = sorted({str(t.dtype) for n, t in model.named_parameters() if not n.startswith("audio_vae.")})
    model.audio_vae = model.audio_vae.to(torch.float32)
    # weight_norm (старый, на хуке) пересчитывает вес g·v/‖v‖ перед КАЖДЫМ вызовом свёртки и держит вторую fp32-копию весов VAE
    # (кодер 48 млн + декодер 46 млн параметров): +0,18 ГиБ на кодер и столько же на декодер. Сливаем один раз — та же формула.
    # Замер 12.10: пик кодирования 12,5 с +0,36 → +0,18 ГиБ, декодирования +0,58 → +0,14; разница звука 1e-4 (TF32 cuDNN).
    from torch.nn.utils import remove_weight_norm
    n_wn = 0
    for m in model.audio_vae.modules():
        if hasattr(m, "weight_g"):
            remove_weight_norm(m)
            n_wn += 1
    emb = None
    if embed_cpu:
        emb = model.base_lm.embed_tokens.weight.data
        model.base_lm.embed_tokens = None
    model = model.to(device).eval()
    if embed_cpu:
        model.base_lm.embed_tokens = _cpu_embedding_cls()(emb)
    if vae_frames:
        chunked_vae_decode(model.audio_vae, vae_frames)
    if enc_samples:
        chunked_vae_encode(model.audio_vae, enc_samples)
    cache_ref_latents(model, [ROOT / "voices", ROOT / "eval"])
    tts = VoxCPM.__new__(VoxCPM)
    tts.tts_model, tts.text_normalizer, tts.denoiser = model, None, None
    if verbose:
        nq = sum(1 for m in model.modules() if isinstance(m, QLinear))
        mem = torch.cuda.memory_allocated() / 2 ** 30 if str(device).startswith("cuda") else 0
        log(f"q8: {nq} слоёв int8, типы {dts}, KV {config.max_length}, эмбеддинги {'ОЗУ' if embed_cpu else 'GPU'}, "
            f"VAE: декод по {vae_frames} лат., кодер по {enc_samples} отсч., кеш эталонов, weight_norm слит в {n_wn}; загрузка {time.time() - t0:.0f} с, VRAM torch {mem:.2f} ГиБ, VAE {r}")
    return tts


def cmd_check(a):
    info = json.loads((Path(a.dir) / "q8_info.json").read_text())
    for k in ("made", "lora", "scheme", "q_matrices", "bytes", "err_bf16_median", "err_q8_median", "err_q8_max"):
        print(f"{k:16} {info.get(k)}")
    for w in info["worst"]:
        print(f"  худший {w['m']:<48} int8 {w['err_q8'] * 100:.3f} %  bf16 {w['err_bf16'] * 100:.3f} %  lora={w['lora']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--base", default=str(ROOT / "voxcpm2/weights/VoxCPM2"))
    p.add_argument("--lora", default=str(ROOT / "voxcpm2/runs/lora-v5/step_0003479"), help="'' — без LoRA (база)")
    p.add_argument("--out", default=str(ROOT / "voxcpm2/weights/v5-3479-q8"))
    p.add_argument("--groups", default=",".join(GROUPS), help="что квантовать; всё: " + ",".join(ALL_GROUPS))
    p = sub.add_parser("check")
    p.add_argument("--dir", default=str(ROOT / "voxcpm2/weights/v5-3479-q8"))
    a = ap.parse_args()
    return {"prep": cmd_prep, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
