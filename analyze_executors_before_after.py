#!/usr/bin/env python3
"""Этап 3 части «исполнители»: до/после-анализ по агрегатам.

Вход: docs_with_topics_and_sentiment.parquet этапа 2 (единый кейс-корпус с
флагами _obj_A/_obj_B2/_obj_C/_obj_GOV/_obj_PARL/_obj_AGENCIES и _terms).

Считает по каждому агрегату и сценарию (v1 — всё окно до отсечки; v2 —
симметричные окна) доли тональностей до/после, Δ, χ²-тест, доверительные
интервалы (бутстреп) для Δ доли негатива; ключевой контраст гипотезы Δneg(A) − Δneg(B2); сравнение с
президентским корпусом (его full_sentiment_before_after.csv); вклад отдельных
сущностей (_terms) — дополнительным разрезом («вместе, а не вместо»);
динамика негатива по агрегатам — PNG.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AGGREGATES = ["A", "B2", "C", "GOV", "PARL", "AGENCIES"]
SENTIMENTS = ["negative", "neutral", "positive"]

# Человекочитаемые русские подписи агрегатов для графиков.
AGG_LABELS_RU = {
    "A": "Исполнители (РКН, Минцифры, Шадаев, депутаты)",
    "B2": "Руководство без Путина (Кремль, АП, Песков)",
    "C": "Безличный дискурс о блокировках",
    "GOV": "Правительство (Мишустин, кабмин)",
    "PARL": "Парламент (Володин, Матвиенко)",
    "AGENCIES": "Прочие ведомства (ФАС, Генпрокуратура)",
    "PRES": "Президент (весь корпус v2)",
    "PRES_PURE": "Президент (без mixed-постов)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="До/после-анализ по агрегатам (этап 3).")
    p.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("topic_model_outputs_dynamic_executors_user_bge/docs_with_topics_and_sentiment.parquet"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("topic_model_outputs_dynamic_executors_user_bge"),
    )
    p.add_argument("--cutoff-date", default="2026-01-16")
    p.add_argument("--analysis-start", default="2022-02-24")
    p.add_argument(
        "--president-before-after-csv",
        type=Path,
        default=Path("topic_model_outputs_dynamic_putin_user_bge/full_sentiment_before_after.csv"),
        help="Готовый файл президентской части для сопоставления.",
    )
    p.add_argument(
        "--president-parquet",
        type=Path,
        default=Path("topic_model_outputs_dynamic_putin_v2_user_bge/docs_with_topics_and_sentiment.parquet"),
        help=(
            "Парокет президентского корпуса v2 (3-модельный сентимент). Если есть — "
            "добавляются псевдоагрегаты PRES (весь) и PRES_PURE (без _mixed) и контрасты к ним."
        ),
    )
    p.add_argument(
        "--exclude-footer-urls",
        type=Path,
        default=None,
        help=(
            "Файл со списком post_url, где 'роскомнадзор' встречается только в "
            "кросс-промо футере одноимённого канала: у этих постов терм "
            "'roskomnadzor' убирается из _terms, флаг _obj_A пересчитывается."
        ),
    )
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--top-n-topics", type=int, default=15)
    p.add_argument("--rolling-days", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_frame(path: Path, analysis_start: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = {"date_day", "ensemble_label", "topic_id", "topic_name"}
    missing = sorted(need.difference(df.columns))
    if missing:
        raise SystemExit(f"В parquet не хватает колонок: {missing}")
    df = df.copy()
    df["date_day"] = pd.to_datetime(df["date_day"], errors="coerce", utc=True)
    df = df[df["date_day"].notna()].copy()
    df = df[df["date_day"] >= analysis_start].copy()
    for agg in AGGREGATES:
        col = f"_obj_{agg}"
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)
    if "_terms" not in df.columns:
        df["_terms"] = ""
    df["_terms"] = df["_terms"].fillna("").astype(str)
    df["ensemble_label"] = df["ensemble_label"].astype(str)
    return df


def label_shares(sub: pd.DataFrame) -> dict:
    n = len(sub)
    out = {"n": n}
    for s in SENTIMENTS:
        out[s] = float((sub["ensemble_label"] == s).mean()) if n else float("nan")
    return out


def chi2_before_after(before: pd.DataFrame, after: pd.DataFrame):
    from scipy.stats import chi2_contingency
    table = []
    for sub in (before, after):
        table.append([int((sub["ensemble_label"] == s).sum()) for s in SENTIMENTS])
    table = np.array(table)
    if table.sum() == 0 or (table.sum(axis=1) == 0).any():
        return float("nan"), float("nan")
    # убираем нулевые столбцы, иначе chi2 падает
    table = table[:, table.sum(axis=0) > 0]
    stat, pval, _, _ = chi2_contingency(table)
    return float(stat), float(pval)


def boot_delta_neg(before: pd.DataFrame, after: pd.DataFrame, n_boot: int, rng) -> tuple[float, float, float]:
    """Δ доли негатива (после − до) и 95% доверительный интервал (бутстреп)."""
    b = (before["ensemble_label"] == "negative").to_numpy()
    a = (after["ensemble_label"] == "negative").to_numpy()
    if len(b) == 0 or len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    delta = a.mean() - b.mean()
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = (rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(delta), float(lo), float(hi)


def boot_contrast(b1, a1, b2, a2, n_boot: int, rng) -> tuple[float, float, float]:
    """(Δneg корпуса 1) − (Δneg корпуса 2) с 95% доверительным интервалом (бутстреп)."""
    arrs = [
        (s["ensemble_label"] == "negative").to_numpy()
        for s in (b1, a1, b2, a2)
    ]
    if any(len(x) == 0 for x in arrs):
        return float("nan"), float("nan"), float("nan")
    vb1, va1, vb2, va2 = arrs
    point = (va1.mean() - vb1.mean()) - (va2.mean() - vb2.mean())
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = (
            (rng.choice(va1, len(va1)).mean() - rng.choice(vb1, len(vb1)).mean())
            - (rng.choice(va2, len(va2)).mean() - rng.choice(vb2, len(vb2)).mean())
        )
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def topic_deltas(sub_b: pd.DataFrame, sub_a: pd.DataFrame, top_n: int) -> pd.DataFrame:
    before = sub_b.groupby(["topic_id", "topic_name"]).size().rename("before_n")
    after = sub_a.groupby(["topic_id", "topic_name"]).size().rename("after_n")
    m = pd.concat([before, after], axis=1).fillna(0).reset_index()
    m["before_share"] = m["before_n"] / max(1, m["before_n"].sum())
    m["after_share"] = m["after_n"] / max(1, m["after_n"].sum())
    m["delta_share"] = m["after_share"] - m["before_share"]
    m["abs_delta_share"] = m["delta_share"].abs()
    return m.sort_values("abs_delta_share", ascending=False).head(top_n)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    analysis_start = pd.Timestamp(args.analysis_start, tz="UTC").normalize()
    cutoff = pd.Timestamp(args.cutoff_date, tz="UTC").normalize()
    df = load_frame(args.input_parquet, analysis_start)

    if args.exclude_footer_urls and args.exclude_footer_urls.exists():
        bad_urls = {
            u.strip()
            for u in args.exclude_footer_urls.read_text(encoding="utf-8").splitlines()
            if u.strip()
        }
        a_terms = {
            "roskomnadzor", "rkn_abbr", "rkn_infra", "mincifry", "shadaev",
            "klishas", "gorelkin", "svintsov", "boyarsky", "mizulina",
        }
        hit = df["post_url"].isin(bad_urls)
        df.loc[hit, "_terms"] = df.loc[hit, "_terms"].apply(
            lambda s: ",".join(t for t in s.split(",") if t and t != "roskomnadzor")
        )
        df.loc[hit, "_obj_A"] = df.loc[hit, "_terms"].apply(
            lambda s: any(t in a_terms for t in s.split(","))
        )
        print(
            f"[exclude-footer] кросс-промо футер РКН: затронуто {int(hit.sum())} постов; "
            f"в A осталось {int(df.loc[hit, '_obj_A'].sum())} из них (по другим термам)"
        )

    pres_aggs: dict[str, pd.DataFrame] = {}
    if args.president_parquet and args.president_parquet.exists():
        cols = ["date_day", "ensemble_label"]
        import pyarrow.parquet as pq
        have = set(pq.read_schema(args.president_parquet).names)
        if "_mixed" in have:
            cols.append("_mixed")
        pdf = pd.read_parquet(args.president_parquet, columns=cols)
        pdf["date_day"] = pd.to_datetime(pdf["date_day"], errors="coerce", utc=True)
        pdf = pdf[pdf["date_day"].notna() & (pdf["date_day"] >= analysis_start)].copy()
        pdf["ensemble_label"] = pdf["ensemble_label"].astype(str)
        pres_aggs["PRES"] = pdf
        if "_mixed" in pdf.columns:
            pres_aggs["PRES_PURE"] = pdf[~pdf["_mixed"].fillna(False).astype(bool)]
    analysis_end = df["date_day"].max()
    n_days_after = int((analysis_end - cutoff).days) + 1
    symmetric_start = cutoff - pd.Timedelta(days=n_days_after)

    scenarios = {
        "v1_full_before": (analysis_start, cutoff),
        "v2_symmetric": (symmetric_start, cutoff),
    }

    rows = []
    contrast_rows = []
    sub_cache = {}
    for sc_name, (b_start, b_end) in scenarios.items():
        agg_frames = {agg: df[df[f"_obj_{agg}"]] for agg in AGGREGATES}
        agg_frames.update(pres_aggs)
        for agg, sub in agg_frames.items():
            before = sub[(sub["date_day"] >= b_start) & (sub["date_day"] < b_end)]
            after = sub[sub["date_day"] >= cutoff]
            sub_cache[(sc_name, agg)] = (before, after)
            sb, sa = label_shares(before), label_shares(after)
            chi2, pval = chi2_before_after(before, after)
            dneg, lo, hi = boot_delta_neg(before, after, args.n_boot, rng)
            rows.append({
                "scenario": sc_name, "aggregate": agg,
                "before_n": sb["n"], "after_n": sa["n"],
                **{f"before_{s}": sb[s] for s in SENTIMENTS},
                **{f"after_{s}": sa[s] for s in SENTIMENTS},
                "delta_negative": dneg, "delta_neg_ci_lo": lo, "delta_neg_ci_hi": hi,
                "chi2": chi2, "chi2_pvalue": pval,
            })
        # ключевые контрасты гипотезы: исполнители vs руководство/президент
        pairs = [("A", "B2"), ("A", "GOV"), ("C", "B2")]
        if "PRES_PURE" in agg_frames:
            pairs += [("A", "PRES_PURE"), ("C", "PRES_PURE"), ("B2", "PRES_PURE")]
        elif "PRES" in agg_frames:
            pairs += [("A", "PRES"), ("C", "PRES"), ("B2", "PRES")]
        for a1, a2 in pairs:
            b1, af1 = sub_cache[(sc_name, a1)]
            b2, af2 = sub_cache[(sc_name, a2)]
            point, lo, hi = boot_contrast(b1, af1, b2, af2, args.n_boot, rng)
            contrast_rows.append({
                "scenario": sc_name, "contrast": f"dNeg({a1}) - dNeg({a2})",
                "point": point, "ci_lo": lo, "ci_hi": hi,
                "significant_95": bool(lo > 0 or hi < 0) if not np.isnan(point) else None,
            })

    res = pd.DataFrame(rows)
    res.to_csv(args.out_dir / "aggregates_sentiment_before_after.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(args.out_dir / "aggregates_contrasts.csv", index=False)

    # Сопоставление с президентской частью (готовые доли из её прогона).
    if args.president_before_after_csv.exists():
        pres = pd.read_csv(args.president_before_after_csv)
        pres.to_csv(args.out_dir / "president_reference_before_after.csv", index=False)

    # Вклад отдельных сущностей — «вместе, а не вместо».
    ent_rows = []
    exploded = df[df["_terms"] != ""].copy()
    exploded = exploded.assign(term=exploded["_terms"].str.split(",")).explode("term")
    for term, sub in exploded.groupby("term"):
        before = sub[(sub["date_day"] >= analysis_start) & (sub["date_day"] < cutoff)]
        after = sub[sub["date_day"] >= cutoff]
        sb, sa = label_shares(before), label_shares(after)
        ent_rows.append({
            "term": term, "before_n": sb["n"], "after_n": sa["n"],
            "before_negative": sb["negative"], "after_negative": sa["negative"],
            "delta_negative": (sa["negative"] - sb["negative"]) if sb["n"] and sa["n"] else float("nan"),
        })
    pd.DataFrame(ent_rows).sort_values("after_n", ascending=False).to_csv(
        args.out_dir / "entities_sentiment_before_after.csv", index=False)

    # Топик-дельты по агрегатам (v1).
    td_frames = []
    for agg in AGGREGATES:
        before, after = sub_cache[("v1_full_before", agg)]
        td = topic_deltas(before, after, args.top_n_topics)
        td.insert(0, "aggregate", agg)
        td_frames.append(td)
    pd.concat(td_frames, ignore_index=True).to_csv(
        args.out_dir / "aggregates_topic_deltas.csv", index=False)

    # Динамика доли негатива по агрегатам.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    plot_frames = {agg: df[df[f"_obj_{agg}"]] for agg in ("A", "B2", "C")}
    if "PRES_PURE" in pres_aggs:
        plot_frames["PRES_PURE"] = pres_aggs["PRES_PURE"]
    for agg, sub in plot_frames.items():
        daily = (
            sub.assign(neg=(sub["ensemble_label"] == "negative").astype(float))
            .groupby(sub["date_day"].dt.normalize())["neg"].agg(["mean", "size"])
        )
        roll = daily["mean"].rolling(f"{args.rolling_days}D", min_periods=3).mean()
        label = AGG_LABELS_RU.get(agg, agg)
        plt.plot(roll.index, roll.values, linewidth=1.4, label=f"{label} — n={len(sub):,}".replace(",", " "))
    plt.axvline(cutoff, linestyle="--", color="black", linewidth=1.4,
                label=f"Ограничения Telegram ({cutoff.date().strftime('%d.%m.%Y')})")
    plt.title(
        "Доля негативных постов по объектам поддержки "
        f"(скользящее среднее {args.rolling_days} дн.)"
    )
    plt.ylabel("Доля постов с негативной тональностью")
    plt.xlabel("Дата")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "fig_negative_share_by_aggregate.png", dpi=180)

    summary = {
        "analysis_start": str(analysis_start.date()),
        "analysis_end": str(analysis_end.date()),
        "cutoff": str(cutoff.date()),
        "n_days_after": n_days_after,
        "docs_total": int(len(df)),
        "aggregate_sizes": {a: int(df[f"_obj_{a}"].sum()) for a in AGGREGATES},
    }
    (args.out_dir / "stage3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("written:",
          "aggregates_sentiment_before_after.csv,",
          "aggregates_contrasts.csv,",
          "entities_sentiment_before_after.csv,",
          "aggregates_topic_deltas.csv,",
          "fig_negative_share_by_aggregate.png")


if __name__ == "__main__":
    main()
