#!/usr/bin/env python3
"""merge_test.py — проверка merge_lora.py на игрушечной модели (ключи как у VoxCPM2, размеры маленькие).

  python merge_test.py [папка]     → выход 0, если влитые веса = W + (alpha/r)·B·A, остальное байт в байт, файлы скопированы,
                                     а сломанная LoRA (нет базового веса) отклоняется.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="merge-test-"))
    base, lora, out = d / "base", d / "lora", d / "merged"
    base.mkdir(parents=True, exist_ok=True)
    lora.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(1)
    sd = {}
    mods = ["base_lm.layers.0.self_attn.q_proj", "base_lm.layers.0.mlp.gate_proj", "residual_lm.layers.0.self_attn.o_proj",
            "feat_decoder.estimator.decoder.layers.0.mlp.down_proj", "enc_to_lm_proj"]
    for m in mods:
        sd[m + ".weight"] = torch.randn(48, 32, generator=g).to(torch.bfloat16)
    sd["enc_to_lm_proj.bias"] = torch.randn(48, generator=g).to(torch.bfloat16)
    sd["base_lm.embed_tokens.weight"] = torch.randn(100, 32, generator=g).to(torch.bfloat16)
    sd["stop_head.weight"] = torch.randn(2, 32, generator=g).to(torch.bfloat16)
    save_file(sd, str(base / "model.safetensors"))
    (base / "config.json").write_text(json.dumps({"architecture": "voxcpm2"}))
    (base / "tokenizer.json").write_text("{}")
    (base / "audiovae.pth").write_bytes(b"vae")
    r, alpha = 8, 16
    ld = {}
    for m in mods:
        ld[m + ".lora_A"] = torch.randn(r, 32, generator=g) * 0.1
        ld[m + ".lora_B"] = torch.randn(48, r, generator=g) * 0.1
    save_file(ld, str(lora / "lora_weights.safetensors"))
    (lora / "lora_config.json").write_text(json.dumps({"lora_config": {"r": r, "alpha": alpha, "enable_lm": True, "enable_dit": True,
                                                                       "enable_proj": True}}))
    py = sys.executable
    rc = subprocess.run([py, str(HERE / "merge_lora.py"), "--base", str(base), "--lora", str(lora), "--out", str(out), "--device", "cpu",
                         "--force"]).returncode
    fails = []
    if rc:
        fails.append(f"merge_lora код {rc}")
    else:
        m_sd = load_file(str(out / "model.safetensors"))
        if set(m_sd) != set(sd):
            fails.append("ключи не те")
        for m in mods:
            want = (sd[m + ".weight"].float() + (alpha / r) * ld[m + ".lora_B"] @ ld[m + ".lora_A"]).to(torch.bfloat16)
            if not torch.equal(m_sd[m + ".weight"], want):
                fails.append(f"{m}: не W + s·B·A")
        for k in ("enc_to_lm_proj.bias", "base_lm.embed_tokens.weight", "stop_head.weight"):
            if not torch.equal(m_sd[k], sd[k]):
                fails.append(f"{k} изменился")
        for f in ("config.json", "tokenizer.json", "audiovae.pth", "merge_info.json"):
            if not (out / f).is_file():
                fails.append(f"нет {f}")
        info = json.loads((out / "merge_info.json").read_text())
        if info.get("merged_modules") != len(mods) or info.get("scaling") != alpha / r:
            fails.append(f"merge_info: {info.get('merged_modules')} модулей, scaling {info.get('scaling')}")
    bad = dict(ld)
    bad["base_lm.layers.9.self_attn.q_proj.lora_A"] = torch.zeros(r, 32)
    bad["base_lm.layers.9.self_attn.q_proj.lora_B"] = torch.zeros(48, r)
    (d / "lora_bad").mkdir(exist_ok=True)
    save_file(bad, str(d / "lora_bad" / "lora_weights.safetensors"))
    (d / "lora_bad" / "lora_config.json").write_text((lora / "lora_config.json").read_text())
    rc2 = subprocess.run([py, str(HERE / "merge_lora.py"), "--base", str(base), "--lora", str(d / "lora_bad"), "--plan"],
                         capture_output=True, text=True).returncode
    if rc2 == 0:
        fails.append("LoRA с лишним слоем не отклонена")
    print("merge_test: " + ("ВСЁ ПРОШЛО (W' = W + s·B·A, остальное байт в байт, лишний слой LoRA — отказ)" if not fails else "НЕ ПРОШЛО: " + "; ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
