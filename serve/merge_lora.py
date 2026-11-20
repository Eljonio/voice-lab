#!/usr/bin/env python3
"""merge_lora.py — «влить» LoRA в базовые веса VoxCPM2.

Зачем: в звонке важна скорость. LoRA r=128 на 340 слоях — это ещё два матричных умножения и сложение на каждом слое
на каждом шаге (на каждом из 10 шагов диффузии DiT ×2 из-за cfg). Влитые веса W' = W + (alpha/r)·B·A — та же сеть,
но без лишних операций; и Nano-vLLM-VoxCPM грузит их как обычную модель (без runtime-LoRA).

Как устроено в VoxCPM (коммит f0c787f, modules/layers/lora.py): LoRALinear.forward = F.linear(x, W, b) + (x·Aᵀ·Bᵀ)·scaling,
scaling = alpha/r; ключи файла LoRA — «<модуль>.lora_A» [r, in] и «<модуль>.lora_B» [out, r] (F32), база — «<модуль>.weight» (BF16)
в model.safetensors (577 ключей, без audio_vae — он в audiovae.pth). Итог: та же папка, что база, только model.safetensors другой.

  merge_lora.py --base $VOICE_HOME/weights/VoxCPM2 --lora $VOICE_HOME/runs/lora-v3/step_0006067 \
                --out $VOICE_HOME/merged/v3-6067              # влить и записать (GPU, если есть)
  merge_lora.py --base … --lora … --plan                      # только заголовки: ключи, формы, scaling (без загрузки тензоров)
  merge_lora.py --base … --lora … --limit 8                   # влить 8 слоёв и проверить точность, ничего не писать (ноут)

Проверка точности на каждом влитом слое: случайный вход x (bf16) → три выхода: путь LoRA как в модели (bf16: W, A, B в bf16),
влитый (bf16 W'), эталон fp32. Влитые годны, если их ошибка к эталону не больше max(1,5 × ошибки пути LoRA, 1 %).
Звук до/после (GigaAM CER) — call_bench.py на GPU. Выход 0 — всё ок.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def lora_modules(keys):
    mods = {}
    for k in keys:
        for suf in (".lora_A", ".lora_B"):
            if k.endswith(suf):
                mods.setdefault(k[: -len(suf)], set()).add(suf)
    return mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="папка VoxCPM2 (config.json, model.safetensors, audiovae.*, токенизатор)")
    ap.add_argument("--lora", required=True, help="папка снимка LoRA (lora_weights.safetensors + lora_config.json)")
    ap.add_argument("--out", help="куда записать влитую модель (папка; создаётся через .tmp)")
    ap.add_argument("--device", default=None, help="cuda | cpu (по умолчанию cuda, если есть)")
    ap.add_argument("--plan", action="store_true", help="только заголовки")
    ap.add_argument("--limit", type=int, default=0, help="влить первые N слоёв и проверить, без записи")
    ap.add_argument("--check-rows", type=int, default=8, help="строк случайного входа на слой")
    ap.add_argument("--force", action="store_true", help="перезаписать --out, если есть")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import save_file

    base, lora = Path(a.base), Path(a.lora)
    bf, lf, cf = base / "model.safetensors", lora / "lora_weights.safetensors", lora / "lora_config.json"
    for f in (base / "config.json", bf, lf, cf):
        if not f.is_file():
            log("НЕТ ФАЙЛА:", f)
            return 2
    arch = json.loads((base / "config.json").read_text()).get("architecture", "").lower()
    if arch != "voxcpm2":
        log("архитектура не voxcpm2:", arch)
        return 2
    lc = json.loads(cf.read_text())["lora_config"]
    r, alpha = int(lc["r"]), float(lc["alpha"])
    scaling = alpha / r
    sb, sl = safe_open(str(bf), "pt"), safe_open(str(lf), "pt")
    bkeys, lkeys = list(sb.keys()), list(sl.keys())
    mods = lora_modules(lkeys)
    bset = set(bkeys)
    bad_pair = [m for m, s in mods.items() if s != {".lora_A", ".lora_B"}]
    no_base = [m for m in mods if m + ".weight" not in bset]
    other = [k for k in lkeys if not (k.endswith(".lora_A") or k.endswith(".lora_B"))]
    shape_bad = []
    for m in mods:
        if m + ".weight" not in bset:
            continue
        A, B, W = sl.get_slice(m + ".lora_A").get_shape(), sl.get_slice(m + ".lora_B").get_shape(), sb.get_slice(m + ".weight").get_shape()
        if not (len(A) == 2 and len(B) == 2 and A[0] == r and B[1] == r and B[0] == W[0] and A[1] == W[1]):
            shape_bad.append((m, A, B, W))
    groups = {}
    for m in mods:
        g = m.split(".")[0] + ("" if "." not in m else "." + m.split(".")[-1])
        groups[g] = groups.get(g, 0) + 1
    log(f"база: {len(bkeys)} ключей ({bf.stat().st_size / 1e9:.2f} ГБ), LoRA: r={r}, alpha={alpha:g}, scaling={scaling:g}, "
        f"модулей {len(mods)} (ключей {len(lkeys)}); lm={lc.get('enable_lm')} dit={lc.get('enable_dit')} proj={lc.get('enable_proj')}")
    log("по группам:", ", ".join(f"{k} {v}" for k, v in sorted(groups.items())))
    if bad_pair or no_base or other or shape_bad:
        log(f"ОШИБКА: без пары A/B {len(bad_pair)}, нет базового веса {len(no_base)}, чужие ключи {len(other)}, форма не та {len(shape_bad)}")
        for name, xs in (("без пары", bad_pair), ("нет базы", no_base), ("чужие", other), ("форма", shape_bad)):
            if xs:
                log("  ", name, xs[:4])
        return 1
    log("ключи и формы сходятся: каждый «X.lora_A/B» ложится на «X.weight»")
    if a.plan:
        return 0

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = sorted(mods)
    if a.limit:
        names = names[: a.limit]
    gen = torch.Generator(device="cpu").manual_seed(0)
    errs = []
    merged = {}
    t = time.time()
    for m in names:
        W = sb.get_tensor(m + ".weight")
        A32, B32 = sl.get_tensor(m + ".lora_A").to(dev, torch.float32), sl.get_tensor(m + ".lora_B").to(dev, torch.float32)
        Wd = W.to(dev)
        W32 = Wd.float() + scaling * (B32 @ A32)
        Wm = W32.to(W.dtype)
        x = torch.randn(a.check_rows, W.shape[1], generator=gen).to(dev)
        xb = x.to(W.dtype)
        y_ref = x @ W32.t()
        y_lora = F.linear(xb, Wd) + F.linear(F.linear(xb, A32.to(W.dtype)), B32.to(W.dtype)) * scaling   # как LoRALinear в bf16
        y_m = F.linear(xb, Wm)
        n = y_ref.norm().item() + 1e-12
        e_l, e_m = (y_lora.float() - y_ref).norm().item() / n, (y_m.float() - y_ref).norm().item() / n
        d_rel = (scaling * (B32 @ A32)).norm().item() / (Wd.float().norm().item() + 1e-12)
        errs.append((m, e_l, e_m, d_rel))
        merged[m + ".weight"] = Wm.cpu().contiguous()
    worst = max(errs, key=lambda e: e[2] / max(1.5 * e[1], 0.01))
    import statistics
    log(f"влито {len(names)} слоёв за {time.time() - t:.1f} с на {dev}; ошибка к fp32-эталону: путь LoRA медиана "
        f"{statistics.median(e[1] for e in errs) * 100:.3f} %, влитые медиана {statistics.median(e[2] for e in errs) * 100:.3f} % "
        f"(макс {max(e[2] for e in errs) * 100:.3f} %); вклад LoRA |sBA|/|W| медиана {statistics.median(e[3] for e in errs) * 100:.2f} %")
    bad = [e for e in errs if e[2] > max(1.5 * e[1], 0.01)]
    if bad:
        log(f"ОШИБКА точности: {len(bad)} слоёв, худший {worst[0]}: LoRA {worst[1] * 100:.3f} %, влитые {worst[2] * 100:.3f} %")
        return 1
    log("точность ок: влитые не хуже пути LoRA (bf16) на всех слоях")
    if a.limit or not a.out:
        log("без записи (--limit или нет --out)")
        return 0

    out = Path(a.out)
    tmp = out.with_name(out.name + ".tmp")
    if out.exists() and not a.force:
        log("уже есть:", out, "(--force — перезаписать)")
        return 2
    need = bf.stat().st_size * 1.05 + 5e8
    free = shutil.disk_usage(out.parent if out.parent.exists() else Path("/")).free
    if free < need:
        log(f"мало места: свободно {free / 1e9:.1f} ГБ, нужно {need / 1e9:.1f}")
        return 3
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t = time.time()
    sd = {}
    for k in bkeys:
        sd[k] = merged[k] if k in merged else sb.get_tensor(k)
    save_file(sd, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    for f in base.iterdir():                     # всё остальное — как в базе (config, токенизатор, audiovae.*)
        if f.is_file() and f.name != "model.safetensors" and not f.name.startswith("."):
            shutil.copy2(f, tmp / f.name)
    so = safe_open(str(tmp / "model.safetensors"), "pt")
    okeys = set(so.keys())
    same_shape = all(so.get_slice(k).get_shape() == sb.get_slice(k).get_shape() and so.get_slice(k).get_dtype() == sb.get_slice(k).get_dtype()
                     for k in bkeys)
    import random
    rnd = random.Random(0)
    untouched = [k for k in bkeys if k not in merged]
    eq = all(torch.equal(so.get_tensor(k), sb.get_tensor(k)) for k in rnd.sample(untouched, min(10, len(untouched))))
    changed = all(not torch.equal(so.get_tensor(k), sb.get_tensor(k)) for k in rnd.sample(sorted(merged), min(10, len(merged))))
    if okeys != set(bkeys) or not same_shape or not eq or not changed:
        log(f"ОШИБКА записи: ключи {'ок' if okeys == set(bkeys) else 'НЕ те'}, формы/типы {'ок' if same_shape else 'НЕ те'}, "
            f"нетронутые {'равны' if eq else 'ИЗМЕНИЛИСЬ'}, влитые {'другие' if changed else 'НЕ ИЗМЕНИЛИСЬ'}")
        return 1
    info = dict(made=time.strftime("%Y-%m-%d %H:%M:%S"), base=str(base), lora=str(lora), r=r, alpha=alpha, scaling=scaling,
                merged_modules=len(merged), keys=len(bkeys), lora_config=lc,
                err_lora_median=statistics.median(e[1] for e in errs), err_merged_median=statistics.median(e[2] for e in errs),
                err_merged_max=max(e[2] for e in errs), lora_delta_rel_median=statistics.median(e[3] for e in errs))
    (tmp / "merge_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    log(f"записано {out} за {time.time() - t:.0f} с: {len(bkeys)} ключей (влито {len(merged)}), "
        f"{(out / 'model.safetensors').stat().st_size / 1e9:.2f} ГБ; ключи/формы/типы = база, нетронутые байт в байт")
    return 0


if __name__ == "__main__":
    sys.exit(main())
