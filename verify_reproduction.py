#!/usr/bin/env python3
"""Автопроверка воспроизводимости пайплайна.

Два режима:
  --write-expected  снять эталонные значения с ТЕКУЩИХ артефактов
                    и сохранить их в reproduce_expected.json;
  (по умолчанию)    сверить текущие артефакты с эталоном и напечатать
                    таблицу PASS/WARN/FAIL. Код возврата 1, если есть FAIL.

Типы проверок:
  exact      — значение должно совпасть бит-в-бит (детерминированные CPU-шаги:
               фильтры корпусов, счётчики документов, агрегаты stage3);
  approx     — |actual - expected| <= tol (доли тональности и метрики,
               зависящие от GPU-инференса: допуск на float-шум);
  approx_rel — относительный допуск (число тем BERTopic: при пересчёте
               эмбеддингов с нуля кластеризация может слегка сдвинуться);
  exists     — файл должен существовать;
  filelist   — все файлы из эталонного списка должны существовать.

Проверки уровня warn не валят репродус (печатается WARN, а не FAIL).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
V2 = ROOT / "topic_model_outputs_dynamic_putin_v2_user_bge"
EXE = ROOT / "topic_model_outputs_dynamic_executors_user_bge"
RES = ROOT / "results"
EXPECTED_PATH = ROOT / "reproduce_expected.json"


# ---------------------------------------------------------------------------
# Извлечение значений из артефактов
# ---------------------------------------------------------------------------

def _json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _parquet_label_shares(path: Path) -> dict:
    import pyarrow.parquet as pq

    col = pq.read_table(path, columns=["ensemble_label"]).column(0).to_pylist()
    n = len(col)
    shares = {}
    for lab in ("negative", "neutral", "positive"):
        shares[lab] = round(sum(1 for x in col if x == lab) / n, 6)
    return shares


def _parquet_mixed_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    col = pq.read_table(path, columns=["_mixed"]).column(0).to_pylist()
    return int(sum(1 for x in col if bool(x)))


def _n_topics(topic_info_csv: Path) -> int:
    import pandas as pd

    ti = pd.read_csv(topic_info_csv)
    col = "topic_id" if "topic_id" in ti.columns else "Topic"
    return int((ti[col] != -1).sum())


def _aggregates_counts(csv_path: Path) -> dict:
    import pandas as pd

    df = pd.read_csv(csv_path)
    return {
        f"{r.scenario}|{r.aggregate}": [int(r.before_n), int(r.after_n)]
        for r in df.itertuples()
    }


def _aggregates_negative(csv_path: Path) -> dict:
    import pandas as pd

    df = pd.read_csv(csv_path)
    return {
        f"{r.scenario}|{r.aggregate}": [
            round(float(r.before_negative), 6),
            round(float(r.after_negative), 6),
            round(float(r.delta_negative), 6),
        ]
        for r in df.itertuples()
    }


def _line_count(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _csv_rows(path: Path) -> int:
    return _line_count(path) - 1


def _results_filelist() -> list:
    return sorted(
        str(p.relative_to(RES)) for p in RES.rglob("*") if p.is_file()
    )


# ---------------------------------------------------------------------------
# Реестр проверок: (id, kind, tol, level, getter)
#   kind: exact | approx | approx_rel | exists | filelist
#   level: fail | warn
# ---------------------------------------------------------------------------

CHECKS = [
    # --- Этап 1: фильтры корпусов (CPU, полностью детерминированы) ---
    ("filters/president_v2.written", "exact", None, "fail",
     lambda: _json(OUT / "president_v2_summary.json")["written"]),
    ("filters/president_v2.mixed", "exact", None, "fail",
     lambda: _json(OUT / "president_v2_summary.json")["mixed"]),
    ("filters/president_v2.trigger_counts", "exact", None, "fail",
     lambda: _json(OUT / "president_v2_summary.json")["trigger_counts"]),
    ("filters/executors.written", "exact", None, "fail",
     lambda: _json(OUT / "executors_selection_summary.json")["written"]),
    ("filters/executors.corpus_counts", "exact", None, "fail",
     lambda: _json(OUT / "executors_selection_summary.json")["corpus_counts"]),
    ("filters/president_v2_for_model.jsonl", "exists", None, "fail",
     lambda: OUT / "president_putin_selection_v2_for_model.jsonl"),
    ("filters/executors_for_model.jsonl", "exists", None, "fail",
     lambda: OUT / "executors_selection_for_model.jsonl"),

    # --- Этап 2-3: корпус президента v2 (темы + тональность) ---
    ("president_v2/docs.rows", "exact", None, "fail",
     lambda: _parquet_rows(V2 / "docs_with_topics_and_sentiment.parquet")),
    ("president_v2/docs.mixed_rows", "exact", None, "fail",
     lambda: _parquet_mixed_rows(V2 / "docs_with_topics_and_sentiment.parquet")),
    ("president_v2/sentiment_shares", "approx", 0.01, "fail",
     lambda: _parquet_label_shares(V2 / "docs_with_topics_and_sentiment.parquet")),
    ("president_v2/n_topics", "approx_rel", 0.30, "warn",
     lambda: _n_topics(V2 / "topic_info.csv")),
    ("president_v2/zeroshot_probs.rows", "exact", None, "fail",
     lambda: __import__("numpy").load(V2 / "zeroshot_geracl_probs.npy").shape[0]),
    ("president_v2/key_files", "filelist", None, "fail",
     lambda: [str(V2.name + "/" + n) for n in (
         "docs_with_topics_and_sentiment.parquet", "topic_info.csv",
         "topics_over_time_full.csv", "topics_over_time_evolution.csv",
         "fig_sentiment_dynamics_full.png", "fig_topics_over_time_full_top.png",
         "fig_delta_share_top_full.png",
         "fig_topics_over_time_full_top_interactive.html", "config.json",
     ) if (V2 / n).exists()]),

    # --- Этап 2-3: корпус исполнителей (темы + тональность) ---
    ("executors/docs.rows", "exact", None, "fail",
     lambda: _parquet_rows(EXE / "docs_with_topics_and_sentiment.parquet")),
    ("executors/sentiment_shares", "approx", 0.01, "fail",
     lambda: _parquet_label_shares(EXE / "docs_with_topics_and_sentiment.parquet")),
    ("executors/n_topics", "approx_rel", 0.30, "warn",
     lambda: _n_topics(EXE / "topic_info.csv")),
    ("executors/zeroshot_probs.rows", "exact", None, "fail",
     lambda: __import__("numpy").load(EXE / "zeroshot_geracl_probs.npy").shape[0]),
    ("executors/bak_2model.rows", "exact", None, "fail",
     lambda: _parquet_rows(EXE / "docs_with_topics_and_sentiment.parquet.bak_2model")),

    # --- Этап 4: stage3 (агрегаты до/после; детерминирован при готовом parquet) ---
    ("stage3/docs_total", "exact", None, "fail",
     lambda: _json(EXE / "stage3_summary.json")["docs_total"]),
    ("stage3/cutoff", "exact", None, "fail",
     lambda: _json(EXE / "stage3_summary.json")["cutoff"]),
    ("stage3/aggregate_sizes", "exact", None, "fail",
     lambda: _json(EXE / "stage3_summary.json")["aggregate_sizes"]),
    ("stage3/aggregates.doc_counts", "exact", None, "fail",
     lambda: _aggregates_counts(EXE / "aggregates_sentiment_before_after.csv")),
    ("stage3/aggregates.negative_shares", "approx", 0.015, "fail",
     lambda: _aggregates_negative(EXE / "aggregates_sentiment_before_after.csv")),
    ("stage3/sensitivity_2model.rows", "exact", None, "fail",
     lambda: _csv_rows(EXE / "sensitivity_2model" / "aggregates_sentiment_before_after.csv")),

    # --- Этап 5: итоговый пакет results/ ---
    ("results/filelist", "filelist", None, "fail", _results_filelist),

    # --- Замороженные входы (создавались НЕ скриптами: LLM-аудит и LLM-кодирование) ---
    ("frozen/footer_crosspromo_urls.lines", "exact", None, "fail",
     lambda: _line_count(OUT / "rkn_register_audit" / "footer_crosspromo_rkn_urls.txt")),
    ("frozen/blame_coding.rows", "exact", None, "fail",
     lambda: _csv_rows(OUT / "blame_coding" / "blame_coding_results.csv")),

    # --- Необязательное (legacy v1, нужен только для справочной линии президента) ---
    ("legacy/putin_v1_before_after.csv", "exists", None, "warn",
     lambda: ROOT / "topic_model_outputs_dynamic_putin_user_bge" / "full_sentiment_before_after.csv"),
]


# ---------------------------------------------------------------------------
# Сравнение
# ---------------------------------------------------------------------------

def _compare(kind: str, tol, expected, actual) -> tuple[bool, str]:
    if kind == "exact":
        ok = expected == actual
        return ok, "" if ok else f"expected={expected!r} actual={actual!r}"
    if kind == "approx":
        if isinstance(expected, dict):
            if set(expected) != set(actual):
                return False, f"keys differ: {sorted(expected)} vs {sorted(actual)}"
            diffs = {
                k: (max(abs(a - b) for a, b in zip(_aslist(expected[k]), _aslist(actual[k]))))
                for k in expected
            }
            worst_key = max(diffs, key=diffs.get)
            ok = diffs[worst_key] <= tol
            return ok, f"max|diff|={diffs[worst_key]:.4f} at {worst_key!r} (tol={tol})"
        diff = abs(float(expected) - float(actual))
        return diff <= tol, f"|{expected}-{actual}|={diff:.4f} (tol={tol})"
    if kind == "approx_rel":
        rel = abs(float(actual) - float(expected)) / max(abs(float(expected)), 1e-9)
        return rel <= tol, f"expected={expected} actual={actual} rel.diff={rel:.2%} (tol={tol:.0%})"
    if kind == "exists":
        p = Path(actual)
        return p.exists(), f"missing: {p}"
    if kind == "filelist":
        base = RES if actual == "results/filelist" else ROOT
        missing = [f for f in expected if not (base / f).exists()]
        return not missing, f"missing {len(missing)} files: {missing[:5]}"
    raise ValueError(kind)


def _aslist(v):
    return v if isinstance(v, (list, tuple)) else [v]


# ---------------------------------------------------------------------------
# Режимы
# ---------------------------------------------------------------------------

def write_expected() -> None:
    expected: dict = {}
    for check_id, kind, _tol, _level, getter in CHECKS:
        if kind == "exists":
            continue  # для exists эталон не нужен — только наличие файла
        try:
            value = getter()
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {check_id}: {exc}")
            continue
        expected[check_id] = value
    with open(EXPECTED_PATH, "w", encoding="utf-8") as f:
        json.dump(expected, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"Эталон сохранён: {EXPECTED_PATH} ({len(expected)} значений)")


def verify() -> int:
    if not EXPECTED_PATH.exists():
        print(f"Нет эталона {EXPECTED_PATH}. Сначала: python verify_reproduction.py --write-expected")
        return 2
    with open(EXPECTED_PATH, encoding="utf-8") as f:
        expected_all = json.load(f)

    n_pass = n_warn = n_fail = 0
    width = max(len(c[0]) for c in CHECKS) + 2
    for check_id, kind, tol, level, getter in CHECKS:
        try:
            if kind == "exists":
                ok, detail = _compare(kind, tol, None, getter())
            else:
                if check_id not in expected_all:
                    print(f"[WARN] {check_id:<{width}} нет в эталоне — пропуск")
                    n_warn += 1
                    continue
                if kind == "filelist":
                    ok, detail = _compare(kind, tol, expected_all[check_id], check_id)
                else:
                    ok, detail = _compare(kind, tol, expected_all[check_id], getter())
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"ошибка чтения артефакта: {exc}"

        if ok:
            print(f"[PASS] {check_id:<{width}}" + (f" {detail}" if kind in ("approx", "approx_rel") else ""))
            n_pass += 1
        elif level == "warn":
            print(f"[WARN] {check_id:<{width}} {detail}")
            n_warn += 1
        else:
            print(f"[FAIL] {check_id:<{width}} {detail}")
            n_fail += 1

    print("-" * (width + 30))
    print(f"Итог: PASS={n_pass} WARN={n_warn} FAIL={n_fail}")
    if n_fail == 0:
        print("Репродукция подтверждена: все обязательные проверки сошлись с эталоном.")
    return 1 if n_fail else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-expected", action="store_true",
                    help="Снять эталон с текущих артефактов вместо проверки.")
    args = ap.parse_args()
    if args.write_expected:
        write_expected()
    else:
        sys.exit(verify())


if __name__ == "__main__":
    main()
