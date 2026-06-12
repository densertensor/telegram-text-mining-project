#!/usr/bin/env bash
# ============================================================================
# reproduce.sh — единая точка входа для полного воспроизведения результатов
# проекта (динамика тем и тональности в «патриотических» TG-каналах).
#
# Протокол и пояснения: REPRODUCE.md (рядом с этим файлом).
# Автопроверка результата: ./reproduce.sh verify
#
# Использование:
#   ./reproduce.sh check      — проверка окружения и входных данных
#   ./reproduce.sh filters    — этап 1: фильтрация корпусов из SQLite (CPU)
#   ./reproduce.sh topics     — этап 2: эмбеддинги + BERTopic (GPU)
#   ./reproduce.sh sentiment  — этап 3: ансамбль тональности 3 модели (GPU)
#   ./reproduce.sh stage3     — этап 4: агрегаты «до/после» + sensitivity
#   ./reproduce.sh package    — этап 5: сборка пакета results/
#   ./reproduce.sh verify     — этап 6: сверка с эталоном reproduce_expected.json
#   ./reproduce.sh all        — этапы 1-6 подряд
#   ./reproduce.sh import     — (для держателей сырых выгрузок) пересборка SQLite из inputs/*.jsonl
#
# Переопределяемые переменные окружения:
#   PY            python из env "topic-sentiment" (BERTopic, transformers 4.4x)
#   GERACL_PY     python из env "geracl" (transformers>=4.49 для GeRaCl)
#   GERACL_MODEL  путь к локальному чекпойнту deepvk/GeRaCl-USER2-base
#   GPUS          список GPU, напр. "0 1 2 3" (по умолчанию все видимые)
#
# Замечание о детерминизме: этапы 1, 4, 5 детерминированы бит-в-бит.
# Этапы 2-3 (GPU-инференс) воспроизводятся с точностью до float-шума;
# verify сравнивает их с допусками (см. REPRODUCE.md, раздел «Детерминизм»).
# ============================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PY="${PY:-python}"
GERACL_PY="${GERACL_PY:-python}"
GERACL_MODEL="${GERACL_MODEL:-models/GeRaCl-USER2-base}"
if [[ -z "${GPUS:-}" ]]; then
    n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
    # Исторические прогоны шли на 4 GPU — повторяем ту же раскладку.
    (( n_gpu > 4 )) && n_gpu=4
    if (( n_gpu > 0 )); then GPUS=$(seq -s' ' 0 $((n_gpu - 1))); else GPUS=""; fi
fi

DB=patriot_channels_posts_20260423_233414.sqlite
OUT=outputs
V2_DIR=topic_model_outputs_dynamic_putin_v2_user_bge
EXE_DIR=topic_model_outputs_dynamic_executors_user_bge
FOOTER_URLS=$OUT/rkn_register_audit/footer_crosspromo_rkn_urls.txt

V2_STOP=(ввп главковерх главковерха главковерху нацлидер нацлидера нацлидеру putin)
EXE_STOP=(
    роскомнадзор роскомнадзора роскомнадзору роскомнадзором роскомнадзоре роскомпозор ркн
    минцифры минцифра минцифре минцифрой минцифру минсвязи
    шадаев шадаева шадаеву шадаевым шадаеве
    кремль кремля кремлю кремлем кремле кремлевский кремлевская кремлевские кремлевского кремлевской кремлевском
    песков пескова пескову песковым пескове
    медведев медведева медведеву медведевым медведеве
    совбез совбеза совбезу совбезом совбезе
    мишустин мишустина мишустину мишустиным мишустине
    кабмин кабмина кабмину кабмином кабмине
    володин володина володину володиным володине матвиенко
    патрушев патрушева патрушеву патрушевым патрушеве
    кириенко вайно
    клишас клишаса клишасу клишасом клишасе
    горелкин горелкина горелкину горелкиным горелкине
    свинцов свинцова свинцову свинцовым свинцове
    боярский боярского боярскому боярским боярском
    мизулина мизулиной мизулину
    генпрокуратура генпрокуратуры генпрокуратуре генпрокуратурой генпрокуратуру
    фас тспу грчц ссоп
    администрация администрации администрацию администрацией
)

log() { printf '\n=== [%s] %s ===\n' "$(date -Is)" "$*"; }

# ----------------------------------------------------------------------------
stage_check() {
    log "Проверка окружения"
    "$PY" -c "import bertopic, sentence_transformers, transformers, pymorphy3, razdel, plotly; print('env topic-sentiment: OK')"
    "$GERACL_PY" -c "import geracl, transformers; print('env geracl: OK (transformers', transformers.__version__, ')')" \
        || echo "ВНИМАНИЕ: env geracl недоступен — этап sentiment (zero-shot) не выполнится"
    [[ -f "$DB" ]] && echo "БД найдена: $DB ($(du -h "$DB" | cut -f1))" \
        || echo "ВНИМАНИЕ: нет $DB — скачайте и распакуйте архив базы (README, раздел «Получение данных»)"
    [[ -d "$GERACL_MODEL" ]] && echo "Чекпойнт GeRaCl найден: $GERACL_MODEL" \
        || echo "ВНИМАНИЕ: нет чекпойнта GeRaCl ($GERACL_MODEL)"
    [[ -f "$FOOTER_URLS" ]] && echo "Замороженный аудит футеров найден: $FOOTER_URLS" \
        || echo "ВНИМАНИЕ: нет $FOOTER_URLS — это замороженный артефакт, он НЕ генерируется скриптами"
    echo "GPU: ${GPUS:-нет (CPU; этапы topics/sentiment будут очень медленными)}"
}

# ----------------------------------------------------------------------------
stage_import() {
    log "Этап 0 (опционально): импорт сырых JSONL в SQLite"
    "$PY" import_patriot_data_to_sqlite.py
}

# ----------------------------------------------------------------------------
stage_filters() {
    log "Этап 1а: корпус президента v2 (207 776 постов; ~40-60 мин CPU)"
    "$PY" filter_president_v2_from_sqlite.py

    log "Этап 1б: корпус исполнителей (85 992 поста)"
    "$PY" filter_executors_from_sqlite.py

    log "Этап 1в: legacy-корпус президента v1 (нужен только для справочной линии PRES в stage3)"
    "$PY" filter_putin_posts_from_sqlite.py

    log "Этап 1г: разведочная выборка кандидатов-исполнителей (outputs/executor_candidates.sqlite)"
    "$PY" scan_executor_candidates.py

    log "Этап 1д: подготовка входов для моделирования (_triggers -> строка; убрать dict-колонку _objects)"
    "$PY" - <<'PYEOF'
import json
src = "outputs/president_putin_selection_v2.jsonl"
dst = "outputs/president_putin_selection_v2_for_model.jsonl"
n = 0
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for line in f:
        r = json.loads(line)
        r["_triggers"] = ",".join(r.get("_triggers") or [])
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1
print("president_v2 prepared docs:", n)
PYEOF
    "$PY" - <<'PYEOF'
import json
src = "outputs/executors_selection_20260423_233414.jsonl"
dst = "outputs/executors_selection_for_model.jsonl"
n = 0
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for line in f:
        r = json.loads(line)
        r.pop("_objects", None)
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1
print("executors prepared docs:", n)
PYEOF
}

# ----------------------------------------------------------------------------
stage_topics() {
    log "Этап 2а: BERTopic для корпуса президента v2 (161 599 док.; GPU, ~1-2 ч)"
    "$PY" dynamic_topics_sentiments.py \
        --in-jsonl "$PWD/$OUT/president_putin_selection_v2_for_model.jsonl" \
        --out-dir "$PWD/$V2_DIR" \
        --disable-sentiment \
        ${GPUS:+--gpu-ids $GPUS} \
        --extra-stop-terms "${V2_STOP[@]}"

    log "Этап 2б: BERTopic для корпуса исполнителей (65 698 док.; GPU, ~30-60 мин)"
    "$PY" dynamic_topics_sentiments.py \
        --in-jsonl "$PWD/$OUT/executors_selection_for_model.jsonl" \
        --out-dir "$PWD/$EXE_DIR" \
        --min-topic-size 50 \
        --disable-sentiment \
        ${GPUS:+--gpu-ids $GPUS} \
        --extra-stop-terms "${EXE_STOP[@]}"
}

# ----------------------------------------------------------------------------
stage_sentiment() {
    log "Этап 3а: 2-модельный ансамбль для исполнителей (бэкап для sensitivity-анализа)"
    "$PY" dynamic_topics_sentiments.py \
        --in-jsonl "$PWD/$OUT/executors_selection_for_model.jsonl" \
        --out-dir "$PWD/$EXE_DIR" \
        --recompute-sentiment-only \
        --zero-shot-sentiment-model "" \
        --min-topic-size 50 \
        ${GPUS:+--gpu-ids $GPUS}
    cp -f "$EXE_DIR/docs_with_topics_and_sentiment.parquet" \
          "$EXE_DIR/docs_with_topics_and_sentiment.parquet.bak_2model"

    log "Этап 3б: zero-shot GeRaCl (отдельный env, по обоим корпусам)"
    "$GERACL_PY" run_geracl_zeroshot.py \
        --in-parquet "$V2_DIR/docs_with_topics.parquet" \
        --out-npy "$V2_DIR/zeroshot_geracl_probs.npy" \
        --model "$GERACL_MODEL" --batch-size 128
    "$GERACL_PY" run_geracl_zeroshot.py \
        --in-parquet "$EXE_DIR/docs_with_topics.parquet" \
        --out-npy "$EXE_DIR/zeroshot_geracl_probs.npy" \
        --model "$GERACL_MODEL" --batch-size 128

    log "Этап 3в: финальный 3-модельный ансамбль (2 классификатора + готовый zero-shot)"
    "$PY" dynamic_topics_sentiments.py \
        --in-jsonl "$PWD/$OUT/president_putin_selection_v2_for_model.jsonl" \
        --out-dir "$PWD/$V2_DIR" \
        --recompute-sentiment-only \
        --zero-shot-precomputed "$V2_DIR/zeroshot_geracl_probs.npy" \
        --extra-stop-terms "${V2_STOP[@]}" \
        ${GPUS:+--gpu-ids $GPUS}
    "$PY" dynamic_topics_sentiments.py \
        --in-jsonl "$PWD/$OUT/executors_selection_for_model.jsonl" \
        --out-dir "$PWD/$EXE_DIR" \
        --recompute-sentiment-only \
        --zero-shot-precomputed "$EXE_DIR/zeroshot_geracl_probs.npy" \
        --min-topic-size 50 \
        ${GPUS:+--gpu-ids $GPUS}

    log "Этап 3г: финальный рендер графиков (президент — без mixed-постов; эволюция слов тем)"
    "$PY" dynamic_topics_sentiments.py \
        --in-jsonl "$PWD/$OUT/president_putin_selection_v2_for_model.jsonl" \
        --out-dir "$PWD/$V2_DIR" \
        --render-only \
        --sentiment-exclude-mixed \
        --extra-stop-terms "${V2_STOP[@]}"
    "$PY" dynamic_topics_sentiments.py \
        --in-jsonl "$PWD/$OUT/executors_selection_for_model.jsonl" \
        --out-dir "$PWD/$EXE_DIR" \
        --render-only \
        --min-topic-size 50 \
        --extra-stop-terms "${EXE_STOP[@]}"
}

# ----------------------------------------------------------------------------
stage_stage3() {
    log "Этап 4а: агрегаты до/после отсечки 2026-01-16 (3-модельный ансамбль)"
    "$PY" analyze_executors_before_after.py --exclude-footer-urls "$FOOTER_URLS"

    log "Этап 4б: sensitivity-анализ (2-модельный бэкап, без линии президента)"
    "$PY" analyze_executors_before_after.py \
        --input-parquet "$EXE_DIR/docs_with_topics_and_sentiment.parquet.bak_2model" \
        --president-parquet /nonexistent \
        --exclude-footer-urls "$FOOTER_URLS" \
        --out-dir "$EXE_DIR/sensitivity_2model"
}

# ----------------------------------------------------------------------------
stage_package() {
    log "Этап 5: сборка итогового пакета results/ (рисунки 01-06 + данные + README)"
    "$PY" build_results_package.py
}

# ----------------------------------------------------------------------------
stage_verify() {
    log "Этап 6: автопроверка против эталона reproduce_expected.json"
    "$PY" verify_reproduction.py
}

# ----------------------------------------------------------------------------
case "${1:-all}" in
    check)     stage_check ;;
    import)    stage_import ;;
    filters)   stage_filters ;;
    topics)    stage_topics ;;
    sentiment) stage_sentiment ;;
    stage3)    stage_stage3 ;;
    package)   stage_package ;;
    verify)    stage_verify ;;
    all)
        stage_check
        stage_filters
        stage_topics
        stage_sentiment
        stage_stage3
        stage_package
        stage_verify
        ;;
    *)
        echo "Неизвестный этап: $1"
        echo "Допустимо: check | import | filters | topics | sentiment | stage3 | package | verify | all"
        exit 2
        ;;
esac

log "Готово: ${1:-all}"
