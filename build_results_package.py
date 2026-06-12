#!/usr/bin/env python3
"""Сборка единого пакета результатов results/: данные + русскоязычные визуализации.

Все подписи, легенды и заголовки — на русском; агрегаты подписаны содержательно.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXE = ROOT / "topic_model_outputs_dynamic_executors_user_bge"
PRES = ROOT / "topic_model_outputs_dynamic_putin_v2_user_bge"
OUT = ROOT / "results"
DATA = OUT / "data"
CUTOFF = pd.Timestamp("2026-01-16", tz="UTC")
CUTOFF_LABEL = "Ограничения Telegram (16.01.2026)"

AGG_RU = {
    "A": "Исполнители (РКН, Минцифры, Шадаев, депутаты)",
    "B2": "Руководство без Путина (Кремль, АП, Песков)",
    "C": "Безличный дискурс о блокировках",
    "GOV": "Правительство (Мишустин, кабмин)",
    "PARL": "Парламент (Володин, Матвиенко)",
    "AGENCIES": "Прочие ведомства (ФАС, Генпрокуратура)",
    "PRES": "Президент (весь корпус)",
    "PRES_PURE": "Президент (без mixed-постов)",
}
AGG_RU_SHORT = {
    "A": "Исполнители",
    "B2": "Кремль/АП/Песков",
    "C": "Безличный дискурс",
    "GOV": "Правительство",
    "PARL": "Парламент",
    "AGENCIES": "Прочие ведомства",
    "PRES": "Президент (весь)",
    "PRES_PURE": "Президент (чистый)",
}
BLAME_RU = {
    "putin_lichno": "Путин лично",
    "kreml_ap": "Кремль / АП",
    "peskov": "Песков лично",
    "ispolniteli": "Исполнители (РКН и др.)",
    "oba_urovnya": "Оба уровня",
    "sistema": "Система / государство в целом",
    "vneshnij": "Внешний объект (Дуров, Запад)",
    "meta_ironiya": "Мета-ирония над переадресацией",
    "net_atribucii": "Нет атрибуции вины",
}
ENTITY_RU = {
    "shadaev": "Шадаев", "gorelkin": "Горелкин", "matvienko": "Матвиенко",
    "kirienko": "Кириенко", "mishustin": "Мишустин", "mincifry": "Минцифры",
    "boyarsky": "Боярский", "rkn_abbr": "РКН (аббр.)", "svintsov": "Свинцов",
    "volodin": "Володин", "patrushev": "Патрушев", "peskov": "Песков",
    "roskomnadzor": "Роскомнадзор", "kreml": "Кремль", "medvedev": "Медведев",
    "sovbez": "Совбез", "adm_prezidenta": "Администрация президента",
    "kabmin": "Кабмин", "genprok": "Генпрокуратура", "klishas": "Клишас",
    "mizulina": "Мизулина", "shadaev_": "Шадаев", "fas": "ФАС",
    "rkn_infra": "Инфраструктура РКН (ТСПУ и др.)", "vaino": "Вайно",
    "ap_abbr": "АП (аббр.)",
}


def fig_negative_dynamics() -> None:
    frames = {}
    df = pd.read_parquet(EXE / "docs_with_topics_and_sentiment.parquet",
                         columns=["date_day", "ensemble_label", "_obj_A", "_obj_B2", "_obj_C"])
    df["date_day"] = pd.to_datetime(df["date_day"], utc=True)
    for agg in ("A", "B2", "C"):
        frames[agg] = df[df[f"_obj_{agg}"].fillna(False).astype(bool)]
    p = pd.read_parquet(PRES / "docs_with_topics_and_sentiment.parquet",
                        columns=["date_day", "ensemble_label", "_mixed"])
    p["date_day"] = pd.to_datetime(p["date_day"], utc=True)
    p = p[p["date_day"] >= pd.Timestamp("2022-02-24", tz="UTC")]
    frames["PRES_PURE"] = p[~p["_mixed"].fillna(False).astype(bool)]

    plt.figure(figsize=(14, 6.5))
    for agg, sub in frames.items():
        daily = (sub.assign(neg=(sub["ensemble_label"] == "negative").astype(float))
                 .groupby(sub["date_day"].dt.normalize())["neg"].mean())
        roll = daily.rolling("28D", min_periods=5).mean()
        plt.plot(roll.index, roll.values, linewidth=1.6,
                 label=f"{AGG_RU[agg]} — n={len(sub):,}".replace(",", " "))
    plt.axvline(CUTOFF, linestyle="--", color="black", linewidth=1.5, label=CUTOFF_LABEL)
    plt.title("Доля негативных постов по объектам поддержки (скользящее среднее 28 дн.)")
    plt.ylabel("Доля постов с негативной тональностью")
    plt.xlabel("Дата")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "fig_02_dinamika_negativa_po_obektam.png", dpi=200)
    plt.close()


def fig_before_after() -> None:
    df = pd.read_csv(EXE / "aggregates_sentiment_before_after.csv")
    v1 = df[df.scenario == "v1_full_before"].set_index("aggregate")
    order = ["A", "PRES_PURE", "B2", "C", "GOV", "PARL", "AGENCIES"]
    order = [a for a in order if a in v1.index]
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 6))
    before = [v1.loc[a, "before_negative"] for a in order]
    after = [v1.loc[a, "after_negative"] for a in order]
    ax.bar(x - w / 2, before, w, label="До отсечки (24.02.2022–15.01.2026)", color="#7da7d9")
    ax.bar(x + w / 2, after, w, label="После отсечки (16.01–24.04.2026)", color="#d96459")
    for i, a in enumerate(order):
        d = v1.loc[a, "delta_negative"] * 100
        ax.annotate(f"{d:+.1f} п.п.", (x[i] + w / 2, after[i]), ha="center",
                    va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([AGG_RU_SHORT[a] + f"\n(n={int(v1.loc[a,'before_n'])}/{int(v1.loc[a,'after_n'])})"
                        for a in order], fontsize=9)
    ax.set_ylabel("Доля постов с негативной тональностью")
    ax.set_title("Негативная тональность до и после ограничений Telegram по объектам поддержки\n"
                 "(ансамбль из 3 моделей; подписи над столбцами — изменение в процентных пунктах)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_03_negativ_do_posle_po_obektam.png", dpi=200)
    plt.close(fig)


def fig_contrasts() -> None:
    df = pd.read_csv(EXE / "aggregates_contrasts.csv")
    name_ru = {
        "dNeg(A) - dNeg(PRES_PURE)": "Исполнители − Президент",
        "dNeg(A) - dNeg(B2)": "Исполнители − Кремль/АП/Песков",
        "dNeg(A) - dNeg(GOV)": "Исполнители − Правительство",
        "dNeg(C) - dNeg(PRES_PURE)": "Безличный дискурс − Президент",
        "dNeg(C) - dNeg(B2)": "Безличный дискурс − Кремль/АП/Песков",
        "dNeg(B2) - dNeg(PRES_PURE)": "Кремль/АП/Песков − Президент",
    }
    sc_ru = {"v1_full_before": "Сценарий 1: всё окно «до»", "v2_symmetric": "Сценарий 2: симметричные окна (99 дн.)"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for ax, sc in zip(axes, ["v1_full_before", "v2_symmetric"]):
        sub = df[df.scenario == sc].copy()
        sub["label"] = sub["contrast"].map(name_ru)
        sub = sub.dropna(subset=["label"]).reset_index(drop=True)
        y = np.arange(len(sub))[::-1]
        for yi, (_, r) in zip(y, sub.iterrows()):
            sig = bool(r["significant_95"])
            color = "#b03a2e" if sig else "#7f8c8d"
            ax.plot([r["ci_lo"] * 100, r["ci_hi"] * 100], [yi, yi], color=color, linewidth=2.2)
            ax.plot(r["point"] * 100, yi, "o", color=color, markersize=7)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"], fontsize=9)
        ax.set_title(sc_ru[sc], fontsize=11)
        ax.set_xlabel("Разность изменений доли негатива, п.п. (95% доверительный интервал)")
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("Ключевые контрасты гипотезы «царь хороший — бояре плохие»\n"
                 "(положительное значение = негатив вырос сильнее у первого объекта; красное = значимо)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig_04_kontrasty_gipotezy.png", dpi=200)
    plt.close(fig)


def fig_blame() -> None:
    df = pd.read_csv(ROOT / "outputs/blame_coding/blame_coding_results.csv")
    df = df.dropna(subset=["blame"])
    aft = df[df.period == "after"]
    order = [k for k in BLAME_RU if k in set(aft["blame"])]
    shares = (aft["blame"].value_counts(normalize=True) * 100).reindex(order).fillna(0)
    counts = aft["blame"].value_counts().reindex(order).fillna(0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    y = np.arange(len(order))[::-1]
    colors = ["#b03a2e" if k in ("putin_lichno", "kreml_ap", "peskov", "oba_urovnya")
              else ("#d4ac0d" if k in ("sistema", "meta_ironiya") else
                    ("#1f618d" if k == "ispolniteli" else "#7f8c8d")) for k in order]
    ax.barh(y, shares.values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([BLAME_RU[k] for k in order], fontsize=10)
    for yi, v, c in zip(y, shares.values, counts.values):
        ax.annotate(f"{v:.1f}% (n={c})", (v + 0.4, yi), va="center", fontsize=9)
    ax.set_xlabel("Доля постов, %")
    ax.set_title(f"Адресат вины после ограничений (n={len(aft)})\nLLM-кодирование, каппа=0.61")
    ax.set_xlim(0, max(shares.values) * 1.22)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    inter = df[df.group == "intersection"]
    tab = pd.crosstab(inter["blame"], inter["period"], normalize="columns") * 100
    tab = tab.reindex(order).fillna(0)
    yb = np.arange(len(order))[::-1]
    h = 0.38
    ax.barh(yb + h / 2, tab.get("before", pd.Series(0, index=order)).values, h,
            label=f"До (n={(inter.period=='before').sum()})", color="#7da7d9")
    ax.barh(yb - h / 2, tab.get("after", pd.Series(0, index=order)).values, h,
            label=f"После (n={(inter.period=='after').sum()})", color="#d96459")
    ax.set_yticks(yb)
    ax.set_yticklabels([BLAME_RU[k] for k in order], fontsize=10)
    ax.set_xlabel("Доля постов, %")
    ax.set_title("Посты, где исполнители и руководство\nупоминаются вместе: до vs после")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["right"].set_visible(False)
    fig.suptitle("Направление вины за ограничения Telegram (этап 4)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_05_napravlenie_viny.png", dpi=200)
    plt.close(fig)


def fig_entities() -> None:
    df = pd.read_csv(EXE / "entities_sentiment_before_after.csv")
    df = df[df.after_n >= 30].sort_values("delta_negative", ascending=True)
    df["name"] = df["term"].map(lambda t: ENTITY_RU.get(t, t))
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(df))
    blockers = {"shadaev", "gorelkin", "kirienko", "mincifry", "rkn_abbr", "roskomnadzor",
                "svintsov", "boyarsky", "klishas", "mizulina", "rkn_infra"}
    colors = ["#b03a2e" if t in blockers else "#7f8c8d" for t in df["term"]]
    ax.barh(y, df["delta_negative"] * 100, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n} (n={int(a)})" for n, a in zip(df["name"], df["after_n"])], fontsize=9)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Изменение доли негатива после ограничений, п.п.")
    ax.set_title("Изменение негативной тональности по отдельным акторам (≥30 постов «после»)\n"
                 "красным — акторы, непосредственно связанные с ограничениями интернета")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_06_aktory_delta_negativa.png", dpi=200)
    plt.close(fig)


def fig_corpus_timeline() -> None:
    pres = pd.read_csv(ROOT / "outputs/president_v2_by_date.csv", parse_dates=["date"])
    pres = pres[pres["date"] >= "2022-02-24"]
    df = pd.read_parquet(EXE / "docs_with_topics_and_sentiment.parquet", columns=["date_day"])
    df["date_day"] = pd.to_datetime(df["date_day"], utc=True).dt.tz_localize(None)
    exe_daily = df.groupby(df["date_day"].dt.normalize()).size()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(pres["date"], pres["posts_count"].rolling(7, min_periods=1).mean(),
                 linewidth=1.2, color="#1f618d")
    axes[0].set_title("Президентский корпус v2: постов в день (сглаживание 7 дн.)")
    axes[1].plot(exe_daily.index, exe_daily.rolling(7, min_periods=1).mean(),
                 linewidth=1.2, color="#b03a2e")
    axes[1].set_title("Кейс-корпус (исполнители + руководство + безличный): постов в день (сглаживание 7 дн.)")
    for ax in axes:
        ax.axvline(CUTOFF.tz_localize(None), linestyle="--", color="black", linewidth=1.4)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Постов в день")
    axes[1].set_xlabel("Дата (пунктир — ограничения Telegram, 16.01.2026)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_01_obemy_korpusov_po_datam.png", dpi=200)
    plt.close(fig)


def copy_data() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    pairs = [
        (EXE / "aggregates_sentiment_before_after.csv", "agregaty_tonalnost_do_posle.csv"),
        (EXE / "aggregates_contrasts.csv", "kontrasty_gipotezy.csv"),
        (EXE / "entities_sentiment_before_after.csv", "aktory_tonalnost_do_posle.csv"),
        (EXE / "aggregates_topic_deltas.csv", "topik_delty_po_agregatam.csv"),
        (EXE / "sensitivity_2model/aggregates_contrasts.csv", "sensitivity_2model_kontrasty.csv"),
        (EXE / "full_sentiment_before_after.csv", "keis_korpus_tonalnost_do_posle.csv"),
        (PRES / "full_sentiment_before_after.csv", "prezident_v2_tonalnost_do_posle.csv"),
        (ROOT / "outputs/blame_coding/blame_coding_results.csv", "kodirovanie_viny_rezultaty.csv"),
        (ROOT / "outputs/president_v2_summary.json", "prezident_v2_filtr_summary.json"),
        (ROOT / "outputs/president_v2_by_date.csv", "prezident_v2_po_datam.csv"),
        (ROOT / "outputs/executors_selection_summary.json", "ispolniteli_filtr_summary.json"),
        (ROOT / "outputs/executor_terms_totals.csv", "chastoty_terminov.csv"),
        (ROOT / "outputs/president_audit/audit_synthesis.md", "audit_prezidentskogo_zaprosa.md"),
        (ROOT / "outputs/executor_query_proposal.md", "profilirovanie_terminov_svod.md"),
        (ROOT / "outputs/rkn_register_audit/audit_synthesis.md", "audit_rkn_dinamika.md"),
    ]
    for src, dst in pairs:
        if src.exists():
            shutil.copy2(src, DATA / dst)
        else:
            print("WARN missing:", src)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fig_corpus_timeline()
    fig_negative_dynamics()
    fig_before_after()
    fig_contrasts()
    fig_blame()
    fig_entities()
    copy_data()
    print("figures + data done ->", OUT)


if __name__ == "__main__":
    main()
