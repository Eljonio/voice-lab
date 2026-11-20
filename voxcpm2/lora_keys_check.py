#!/usr/bin/env python3
"""Тёплый старт v4 (11.10.2026): проверить, что LoRA-веса v3 РЕАЛЬНО лягут в модель тренера, до платного обучения.

Тренер (train_voxcpm_finetune.py, load_checkpoint) делает `load_state_dict(state_dict, strict=False)` — несовпавшие ключи
пропускаются молча, и «продолжение v3» тихо превратилось бы в обучение с нуля. Здесь модель строится ровно как у тренера
(`VoxCPM2Model.from_local(..., training=True, lora_config=LoRAConfig(**conf.lora))`), веса грузятся так же, и проверяется:
  * все ключи файла есть в модели (unexpected = 0);
  * все LoRA-параметры модели есть в файле (missing среди "lora_" = 0);
  * формы совпадают; после загрузки тензоры модели = тензоры файла (сверка 5 случайных);
  * веса не нулевые (lora_B у обученной LoRA ≠ 0).
Код выхода 0 — можно учить; иначе 1 и причина.
  python voxcpm2/lora_keys_check.py --pretrained $VOICE_HOME/weights/VoxCPM2 --conf voxcpm2/conf/lora_v4_cont_pro6000.yaml \
      --weights $VOICE_HOME/runs/lora-v4/latest/lora_weights.safetensors
"""
import argparse
import json
import random
import sys

import torch
import yaml
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--conf", required=True)
    ap.add_argument("--weights", required=True)
    a = ap.parse_args()
    conf = yaml.safe_load(open(a.conf, encoding="utf-8"))
    from voxcpm.model import VoxCPM2Model
    from voxcpm.model.voxcpm2 import LoRAConfig
    arch = json.load(open(f"{a.pretrained}/config.json")).get("architecture", "voxcpm").lower()
    if arch != "voxcpm2":
        print(f"ОШИБКА: архитектура {arch}, ожидалась voxcpm2")
        return 1
    model = VoxCPM2Model.from_local(a.pretrained, optimize=False, training=True, lora_config=LoRAConfig(**conf["lora"]))
    sd = load_file(a.weights)
    mk = model.state_dict()
    lora_model = {k for k in mk if "lora_" in k}
    unexpected = [k for k in sd if k not in mk]
    missing = sorted(lora_model - set(sd))
    shape_bad = [k for k in sd if k in mk and tuple(sd[k].shape) != tuple(mk[k].shape)]
    try:
        res = model.load_state_dict(sd, strict=False)
        bad_load = list(res.unexpected_keys)
    except RuntimeError as e:                          # несовпадение форм тренер тоже не переживёт
        print(f"ОШИБКА load_state_dict: {str(e)[:300]}")
        bad_load = ["RuntimeError"]
    after = model.state_dict()
    rnd = random.Random(0)
    both = sorted(k for k in sd if k in after)
    sample = rnd.sample(both, min(5, len(both)))
    eq = bool(sample) and all(torch.equal(after[k].detach().cpu().to(sd[k].dtype), sd[k]) for k in sample)
    nz = sum(1 for k in sd if "lora_B" in k and float(sd[k].abs().sum()) > 0)
    nb = sum(1 for k in sd if "lora_B" in k)
    print(f"ключей в файле {len(sd)}, LoRA-параметров в модели {len(lora_model)}; лишних {len(unexpected)}, недостающих LoRA {len(missing)}, "
          f"форма не та {len(shape_bad)}; load_state_dict: ошибок/лишних {len(bad_load)}; сверка 5 тензоров {'ок' if eq else 'НЕ СОВПАЛИ'}; "
          f"lora_B ненулевых {nz}/{nb}")
    ok = not unexpected and not missing and not shape_bad and not bad_load and eq and nb > 0 and nz == nb
    if not ok:
        for name, xs in (("лишние", unexpected), ("недостающие", missing), ("форма", shape_bad)):
            if xs:
                print(f"  {name}: {xs[:5]}")
        print("ОШИБКА: веса v3 не лягут в модель — обучение не запускать")
        return 1
    print("ок: тёплый старт v3 → v4 загрузится полностью")
    return 0


if __name__ == "__main__":
    sys.exit(main())
