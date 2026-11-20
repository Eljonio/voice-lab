#!/usr/bin/env bash
# remote_job.sh — выполняется НА арендованном GPU-сервере (его заливает и запускает оркестрация аренды — она провайдер-зависимая
# и в репозиторий не входит). Всё самодостаточно: обучение → пробы звука по каждому чекпоинту → файл EXIT. Ноутбук для этого
# не нужен: если он выключится, результат останется на постоянном диске ($V), а сервер без нагрузки на GPU удалит сторож
# простоя (25 мин) — лишняя оплата максимум за эти 25 мин.
# Раскладка $V (VOICE_HOME на сервере): weights/VoxCPM2, venv/, code/VoxCPM (OpenBMB/VoxCPM @ f0c787f), synth_baseline.py,
#   synth_eval.py, eval/phrases.json, <MIX>/{train,val}.jsonl,stats.txt (build_manifest.py / dataprep/build_mix_v2.py), runs/.
# Переменные: V, RUN, MIX (папка смеси внутри $V), CONF (voxcpm2/conf/*.yaml; плейсхолдеры VOICE_HOME, MIX, RUN, RUNDIR, STEPS
#   подставляются здесь), EPOCHS, MAX_STEPS (потолок шагов), RUNDIR (папка прогона; по умолч. $V/runs/$RUN — для
#   полного дообучения — загрузочный диск сервера, потому что полный чекпоинт с optimizer ≈ 27,5 ГБ),
#   STOP_AT_STEP (09.10: остановить, как только чекпоинт этого шага целиком лёг в latest/ — косинус не трогаем, продолжение
#   тем же RUN/CONF с latest/; EXIT пишется кодом 0 и строкой «остановлено на шаге N»), SKIP_EVAL=1 (пробы делает вызывающий).
# 09.10: в фоне чистит optimizer.pth/scheduler.pth во всех step_* кроме самого свежего (его может ещё копировать тренер) и latest/.
set -uo pipefail
cd "$V"
R="${RUNDIR:-$V/runs/$RUN}"
mkdir -p "$R"; rm -f "$R/EXIT" "$R/STOPPED_AT"
n=$(wc -l < "$MIX/train.jsonl")
bs=$(awk '/^batch_size:/{print $2}' "$CONF"); ga=$(awk '/^grad_accum_steps:/{print $2}' "$CONF")
steps=$(( EPOCHS * n / (bs * ga) )); [ "$steps" -lt 500 ] && steps=500
# 08.10: потолок шагов (цепочка v2 урезает, если эпохи не влезают в оплаченные часы)
if [ "${MAX_STEPS:-0}" -gt 0 ] && [ "$steps" -gt "${MAX_STEPS:-0}" ]; then steps=$MAX_STEPS; fi
sed -e "s#RUNDIR#$R#g" -e "s/STEPS/$steps/g" -e "s/RUN/$RUN/g" -e "s#/MIX/#/$MIX/#g" -e "s#VOICE_HOME#$V#g" "$CONF" > "$R/config.yaml"   # RUNDIR — до RUN
echo "фраз $n, шагов $steps (эпох $EPOCHS, батч $bs×$ga)${STOP_AT_STEP:+, стоп на шаге $STOP_AT_STEP}" | tee "$R/plan.txt"
cat "$MIX/stats.txt" >> "$R/plan.txt"
SAVE=$(awk '/^save_path:/{print $2}' "$R/config.yaml")
[ -n "$SAVE" ] || SAVE="$R"

prune_opt() {   # optimizer/scheduler — только в latest/ и в самом свежем step_* (его может ещё копировать тренер); all — и в нём
  local newest; newest=$(ls -d "$SAVE"/step_* 2>/dev/null | sort | tail -1); [ "${1:-}" = all ] && newest=""
  for d in "$SAVE"/step_*; do
    [ -d "$d" ] && [ "$d" != "$newest" ] && rm -f "$d/optimizer.pth" "$d/scheduler.pth"
  done
}
latest_step() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['step'])" "$SAVE/latest/training_state.json" 2>/dev/null || echo -1; }

if [ "${SKIP_EVAL:-0}" != 1 ]; then
# точка отсчёта и подбор тембра на базовой модели (на GPU — секунды на фразу), один раз на прогон:
#   base/ — те же фразы без LoRA; design/ — 24 варианта голоса по описанию, из них на слух выбирается эталон голоса
if [ ! -d "$R/base" ]; then
  VOXCPM2_WEIGHTS="$V/weights/VoxCPM2" "$V/venv/bin/python" "$V/synth_baseline.py" --device cuda \
      --only ru,kk,kk2,kk3,mix,ru2,design,design_kk --out "$R/base" >> "$R/samples.log" 2>&1
  VOXCPM2_WEIGHTS="$V/weights/VoxCPM2" "$V/venv/bin/python" "$V/synth_baseline.py" --device cuda \
      --only design --seeds 1-24 --out "$R/design" >> "$R/samples.log" 2>&1
  echo "базовые пробы и 24 варианта тембра готовы $(date -Is)" >> "$R/plan.txt"
fi
# страница версий: фиксированный набор eval/phrases.json исходной моделью (v0). Эталон eval/ref.wav — общий
# для всех версий; если его ещё нет на диске, synth_eval.py создаст его исходной моделью один раз.
EVAL_ARGS=(--phrases "$V/eval/phrases.json" --ref "$V/eval/ref.wav" --device cuda)
if [ -f "$V/eval/phrases.json" ] && [ ! -f "$R/eval/base/report.json" ]; then
  VOXCPM2_WEIGHTS="$V/weights/VoxCPM2" "$V/venv/bin/python" "$V/synth_eval.py" "${EVAL_ARGS[@]}" --base "$R/eval/base" \
      >> "$R/samples.log" 2>&1 && echo "набор eval исходной моделью готов $(date -Is)" >> "$R/plan.txt"
  mkdir -p "$R/eval"; cp -f "$V/eval/ref.wav" "$R/eval/ref.wav" 2>/dev/null || true
fi
fi
# keepalive от этапа подготовки больше не нужен: дальше GPU занят обучением
pkill -f 'touch /run/gpu-keepalive' || true
run_trainer() {   # обучение + сторож шагов; код выхода тренера в $code
( cd code/VoxCPM && exec "$V/venv/bin/python" scripts/train_voxcpm_finetune.py --config_path "$R/config.yaml" ) >> "$R/stdout.log" 2>&1 &
TPID=$!
# сторож шагов: чистка optimizer.pth раз в 2 мин; STOP_AT_STEP — убить тренер, когда latest/ = шаг N целиком
# (SIGKILL, не SIGTERM: по SIGTERM тренер сохраняет ещё один чекпоинт текущего шага — лишние 27 ГБ и минуты у полного)
while kill -0 "$TPID" 2>/dev/null; do
  sleep "${WATCH_S:-120}"
  prune_opt
  if [ "${STOP_AT_STEP:-0}" -gt 0 ] && [ "$(latest_step)" -ge "$STOP_AT_STEP" ]; then
    s=$(printf '%s/step_%07d' "$SAVE" "$(latest_step)")
    a=$(du -sb "$SAVE/latest" 2>/dev/null | cut -f1); b=$(du -sb "$s" 2>/dev/null | cut -f1)
    if [ -f "$SAVE/latest/optimizer.pth" ] && [ -f "$SAVE/latest/scheduler.pth" ] && [ "${a:-0}" -ge "${b:-1}" ]; then
      sleep 20; sync
      kill -9 "$TPID" 2>/dev/null; pkill -9 -f "config_path $R/config.yaml" 2>/dev/null
      echo "$(latest_step)" > "$R/STOPPED_AT"
      echo "остановлено на шаге $(latest_step) (STOP_AT_STEP=$STOP_AT_STEP) $(date -Is)" >> "$R/plan.txt"
      break
    fi
  fi
done
wait "$TPID" 2>/dev/null; code=$?
}
: > "$R/stdout.log"
run_trainer
# 09.10: нехватка памяти (полное дообучение на батче 8 у PRO 6000 упирается в 96 ГБ) → батч вдвое меньше, накопление вдвое больше
# (эффективный батч и lr те же), продолжение с latest/ если он есть; до 2 повторов
for try in 1 2; do
  [ "$code" -ne 0 ] && [ ! -f "$R/STOPPED_AT" ] && grep -qE "OutOfMemoryError|CUDA out of memory" "$R/stdout.log" || break
  b0=$(awk '/^batch_size:/{print $2}' "$R/config.yaml"); g0=$(awk '/^grad_accum_steps:/{print $2}' "$R/config.yaml")
  [ "$b0" -ge 2 ] || break
  sed -i -e "s/^batch_size: .*/batch_size: $((b0 / 2))/" -e "s/^grad_accum_steps: .*/grad_accum_steps: $((g0 * 2))/" "$R/config.yaml"
  echo "нехватка памяти GPU на батче ${b0}×${g0} → ${b0}/2 × ${g0}·2 = $((b0 / 2))×$((g0 * 2)), повтор $try (с latest/: шаг $(latest_step)) $(date -Is)" | tee -a "$R/plan.txt"
  mv -f "$R/stdout.log" "$R/stdout_oom$try.log"; : > "$R/stdout.log"
  run_trainer
done
[ -f "$R/STOPPED_AT" ] && code=0
prune_opt all     # тренер стоит: optimizer нужен только в latest/ (продолжение)
sync
echo "train exit $code $(date -Is)" >> "$R/plan.txt"
if [ "${SKIP_EVAL:-0}" != 1 ]; then
# пробы: одни и те же фразы по каждому чекпоинту (на GPU — секунды на фразу). Полный чекпоинт — самодостаточная папка весов
# (model.safetensors + config/tokenizer/audiovae от тренера) → грузим её как исходную модель, без --lora
for c in "$SAVE"/step_*; do
  if [ -f "$c/lora_weights.safetensors" ]; then
    VOXCPM2_WEIGHTS="$V/weights/VoxCPM2" "$V/venv/bin/python" "$V/synth_baseline.py" --device cuda --lora "$c" \
        --only kk,kk2,kk3,mix,ru,ru2,design_kk --out "$c/samples" >> "$R/samples.log" 2>&1 && echo "пробы $c" >> "$R/plan.txt"
  elif [ -f "$c/model.safetensors" ]; then
    VOXCPM2_WEIGHTS="$c" "$V/venv/bin/python" "$V/synth_baseline.py" --device cuda \
        --only kk,kk2,kk3,mix,ru,ru2,design_kk --out "$c/samples" >> "$R/samples.log" 2>&1 && echo "пробы $c" >> "$R/plan.txt"
  fi
done
# набор eval по всем чекпоинтам за одну загрузку модели (design — только у последнего); --ckpt сам различает LoRA и полные
lo=(); for c in "$SAVE"/step_*; do { [ -f "$c/lora_weights.safetensors" ] || [ -f "$c/model.safetensors" ]; } && lo+=(--ckpt "$c=$c/eval"); done
if [ -f "$V/eval/phrases.json" ] && [ ${#lo[@]} -gt 0 ]; then
  VOXCPM2_WEIGHTS="$V/weights/VoxCPM2" "$V/venv/bin/python" "$V/synth_eval.py" --phrases "$V/eval/phrases.json" \
      --ref "$V/eval/ref.wav" --device cuda "${lo[@]}" >> "$R/samples.log" 2>&1 && echo "набор eval по чекпоинтам готов $(date -Is)" >> "$R/plan.txt"
fi
fi
sync   # 06.10: сервер удаляется как выключение питания — без sync свежие файлы на постоянном диске теряются
echo "$code" > "$R/EXIT"
sync
# 08.10: keepalive этапа подготовки трогал файл вечно → после EXIT сервер жил до потолка часов (лишний оплачиваемый час).
# Гасим его: дальше сторож простоя удалит сервер сам, если ноутбук не успел его удалить.
[ "${KEEP_KEEPALIVE:-0}" = 1 ] || pkill -f "gpu-keepalive" || true
