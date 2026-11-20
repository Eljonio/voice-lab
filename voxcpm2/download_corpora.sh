#!/usr/bin/env bash
# download_corpora.sh — открытые казахские (и немного русских) корпуса для базы дообучения VoxCPM2.
# НИЧЕГО НЕ КАЧАЕТ БЕЗ ЯВНОГО ЗАПУСКА. Публичные файлы Hugging Face, токен не нужен, регистрация не нужна.
#
# ГДЕ ЗАПУСКАТЬ. На ноутбуке НЕ надо: KSC2 = 81 ГБ архива (+ столько же распакованного), KazakhTTS2 = 38 ГБ,
# домашний канал ~5–8 МБ/с → 4–6 часов. Правильно — на дешёвом облачном CPU-сервере с постоянным диском (150 ГБ),
# тем же, который потом монтирует GPU-сервер обучения.
# В корпуса уходят только отобранные часы в 16 кГц FLAC (это 2–5 ГБ), архивы удаляются.
#
#   DATA=$VOICE_HOME/corpora ./download_corpora.sh kazakhtts2 ksc2 fleurs_kk fleurs_ru cv_kk
#   ./download_corpora.sh --sizes          # только показать, что и сколько весит
#
# | корпус        | что внутри                                   | архив     | лицензия                         |
# |---------------|----------------------------------------------|-----------|----------------------------------|
# | kazakhtts2    | 271 ч, 5 дикторов (3 ж, 2 м), чтение         | 37,7 ГБ   | CC-BY-4.0 (GitHub IS2AI/Kazakh_TTS) |
# | ksc2          | ~1 200 ч: ТВ, радио, сенат, подкасты + KSC + KazakhTTS2 | 80,8 ГБ | MIT (HF и GitHub IS2AI/ksc2) |
# | fleurs_kk     | ~10 ч чтения Википедии, много дикторов       | 2,9 ГБ    | CC-BY-4.0                        |
# | fleurs_ru     | ~10 ч русского (держать русский, 10–20 % смеси) | 2,0 ГБ | CC-BY-4.0                        |
# | cv_kk         | Common Voice 22 kk, ~2–3 ч, много дикторов   | 0,08 ГБ   | CC0-1.0 (неофициальное зеркало fsicoli; оригинал у Mozilla теперь только после регистрации в Data Collective) |
set -euo pipefail

DATA="${DATA:-${VOICE_HOME:-.}/corpora}"
HF="https://huggingface.co/datasets"
# Частичная загрузка (06.10): KSC2 с HF шёл 5 МБ/с → 81 ГБ ≈ 4,5 ч, а для пробы нужно 8 ч из 1 200.
# В начале архива KSC2 — весь Test (talkshow, tv_news, radio, podcasts, parliament, crowdsourced), затем Train/crowdsourced:
# первые 2 части (16 ГБ) = ~135 тыс. фраз FLAC (> 100 ч). Хвост оборванного архива tar не дочитает — это ожидаемо.
KTTS_PARTS="${KTTS_PARTS:-aa ab ac ad ae af}"
KSC2_PARTS="${KSC2_PARTS:-aa ab ac ad ae af ag ah ai aj}"
KTTS_ALL="aa ab ac ad ae af"; KSC2_ALL="aa ab ac ad ae af ag ah ai aj"

sizes() {
  cat <<'EOF'
kazakhtts2  37.7 GB  issai/KazakhTTS  ISSAI_KazakhTTS2_by_parts/ISSAI_KazakhTTS2.tar.gz.part{aa..af}
ksc2        80.8 GB  issai/Kazakh_Speech_Corpus_2  ISSAI_KSC2.tar.gz.part{aa..aj}
fleurs_kk    2.9 GB  google/fleurs  data/kk_kz/audio/{train,dev,test}.tar.gz + *.tsv
fleurs_ru    2.0 GB  google/fleurs  data/ru_ru/audio/{train,dev,test}.tar.gz + *.tsv
cv_kk        0.08 GB fsicoli/common_voice_22_0  audio/kk/*/*.tar + transcript/kk/*.tsv
Пик диска при потоковой распаковке = размер распакованного корпуса (архив не хранится).
EOF
}

# Потоковая распаковка многотомного tar.gz: части идут curl подряд прямо в tar, архив на диск не пишется.
# Обрыв = перезапуск корпуса целиком (докачки нет), поэтому сначала проверяем свободное место.
stream_parts() {  # $1 = URL-префикс до .part, $2 = список суффиксов, $3 = папка
  local url="$1" parts="$2" out="$3"
  mkdir -p "$out"
  [ -f "$out/.done" ] && { echo "  уже есть: $out"; return; }
  if [ "${4:-}" = partial ]; then   # оборванный многотомный архив: tar/gzip ругнутся на конец — берём распакованное
    { for p in $parts; do curl -fsSL --retry 5 --retry-delay 10 "${url}.part${p}"; done | tar -xz -C "$out"; } \
      || echo "  частичная загрузка ($parts): архив оборван намеренно, распаковано $(du -sh "$out" | cut -f1)"
  else
    for p in $parts; do curl -fsSL --retry 5 --retry-delay 10 "${url}.part${p}"; done | tar -xz -C "$out"
  fi
  touch "$out/.done"
}

need_gb() {  # $1 = нужно ГБ
  local free; free=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc 0-9)
  if [ "$free" -lt "$1" ]; then echo "мало места в $DATA: свободно ${free} ГБ, нужно ~$1 ГБ" >&2; exit 1; fi
}

get() {  # $1 = URL, $2 = файл
  [ -s "$2" ] && return
  mkdir -p "$(dirname "$2")"
  if command -v aria2c >/dev/null; then aria2c -q -x8 -s8 -c -d "$(dirname "$2")" -o "$(basename "$2")" "$1"
  else curl -fsSL --retry 5 -o "$2" "$1"; fi
}

[ "${1:-}" = "--sizes" ] && { sizes; exit 0; }
[ $# -eq 0 ] && { echo "укажи корпуса: kazakhtts2 ksc2 fleurs_kk fleurs_ru cv_kk (или --sizes)"; exit 1; }
mkdir -p "$DATA"
for c in "$@"; do
  echo "== $c → $DATA/$c  ($(date +%H:%M))"
  case "$c" in
    kazakhtts2) need_gb 45
      stream_parts "$HF/issai/KazakhTTS/resolve/main/ISSAI_KazakhTTS2_by_parts/ISSAI_KazakhTTS2.tar.gz" "$KTTS_PARTS" "$DATA/kazakhtts2" \
          "$([ "$KTTS_PARTS" != "$KTTS_ALL" ] && echo partial)" ;;
    ksc2) need_gb 95
      stream_parts "$HF/issai/Kazakh_Speech_Corpus_2/resolve/main/ISSAI_KSC2.tar.gz" "$KSC2_PARTS" "$DATA/ksc2" \
          "$([ "$KSC2_PARTS" != "$KSC2_ALL" ] && echo partial)" ;;
    fleurs_kk|fleurs_ru)
      l=$([ "$c" = fleurs_kk ] && echo kk_kz || echo ru_ru)
      for s in train dev test; do
        get "$HF/google/fleurs/resolve/main/data/$l/$s.tsv" "$DATA/$c/$s.tsv"
        [ -d "$DATA/$c/$s" ] || { get "$HF/google/fleurs/resolve/main/data/$l/audio/$s.tar.gz" "$DATA/$c/$s.tar.gz"
                                  tar -xzf "$DATA/$c/$s.tar.gz" -C "$DATA/$c" && rm -f "$DATA/$c/$s.tar.gz"; }
      done ;;
    cv_kk)
      for t in validated train dev test other clip_durations; do
        get "$HF/fsicoli/common_voice_22_0/resolve/main/transcript/kk/$t.tsv" "$DATA/cv_kk/$t.tsv"; done
      for s in train dev test other; do
        get "$HF/fsicoli/common_voice_22_0/resolve/main/audio/kk/$s/kk_${s}_0.tar" "$DATA/cv_kk/kk_${s}_0.tar"
        tar -xf "$DATA/cv_kk/kk_${s}_0.tar" -C "$DATA/cv_kk" && rm -f "$DATA/cv_kk/kk_${s}_0.tar"
      done ;;
    *) echo "неизвестный корпус: $c" >&2; exit 1 ;;
  esac
  du -sh "$DATA/$c" | sed 's/^/  /'
done
echo "готово ($(date +%H:%M)). Дальше: build_manifest.py / dataprep/build_mix_v2.py — отбор часов, 16 кГц FLAC, VoxCPM-манифест."
