#!/usr/bin/env python3
import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from babel import Locale
from pymorphy3 import MorphAnalyzer
from razdel import tokenize as razdel_tokenize


MORPH = MorphAnalyzer(lang="ru")


def norm_text(s: str) -> str:
    return s.lower().replace("ё", "е")


def tokenize(s: str) -> list[str]:
    tokens: list[str] = []
    for t in razdel_tokenize(s):
        normalized = norm_text(t.text)
        if any(ch.isalnum() for ch in normalized):
            tokens.append(normalized)
    return tokens


@lru_cache(maxsize=200_000)
def lemma_ru_token(token: str) -> str:
    if not token:
        return token
    parsed = MORPH.parse(token)
    if not parsed:
        return token
    return norm_text(parsed[0].normal_form)


def load_country_index_ru() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """
    Источник названий стран: данные CLDR (open-source), загружаются через Babel.
    Возвращает два индекса:
    - точный токен -> множество названий стран
    - лемма токена -> множество названий стран
    """
    ru = Locale.parse("ru")
    territories = ru.territories
    stop_words = {
        "и",
        "республика",
        "федерация",
        "королевство",
        "штаты",
        "острова",
        "остров",
        "демократическая",
        "народная",
        "соединенные",
        "центральноафриканская",
    }

    exact_to_countries: dict[str, set[str]] = defaultdict(set)
    lemma_to_countries: dict[str, set[str]] = defaultdict(set)

    # Частые неоднозначные формы в русском тексте, ненадёжные как указатели страны.
    ambiguous_tokens = {"того"}
    for code, name in territories.items():
        if not (len(code) == 2 and code.isalpha()):
            continue
        if code.upper() == "RU":
            continue
        country_name = str(name)
        if "неизвестн" in norm_text(country_name):
            # Убираем техническую запись локали вида "неизвестный регион".
            continue
        tokens = [
            tok
            for tok in tokenize(country_name)
            if len(tok) >= 4 and tok not in stop_words
        ]
        if not tokens:
            continue

        # Используем только опорный токен (обычно существительное), без прилагательных из составных названий.
        core_token = tokens[-1]
        if core_token in ambiguous_tokens:
            continue
        exact_to_countries[core_token].add(country_name)
        lemma_to_countries[lemma_ru_token(core_token)].add(country_name)

    # Явно удаляем формы Россия/РФ из списка исключаемых стран.
    for tok in ("россия", "российская", "российской", "рф", "русский", "русская"):
        exact_to_countries.pop(tok, None)
        lemma_to_countries.pop(lemma_ru_token(tok), None)

    # Добавляем распространённые алиасы, часто встречающиеся в постах.
    def add_alias(alias: str, country_name: str) -> None:
        key = norm_text(alias)
        exact_to_countries[key].add(country_name)
        lemma_to_countries[lemma_ru_token(key)].add(country_name)

    # Добавляем только алиасы, которых нет в формах названий стран CLDR.
    add_alias("сша", "США")
    add_alias("америка", "США")
    add_alias("британия", "Великобритания")

    return dict(exact_to_countries), dict(lemma_to_countries)


def has_name_patronymic(tokens: list[str]) -> bool:
    # (Владимир* /1 Владимирович*)
    for i, t in enumerate(tokens):
        if not t.startswith("владимир"):
            continue
        j_end = min(len(tokens), i + 3)  # разрыв не более одного токена
        for j in range(i + 1, j_end):
            if tokens[j].startswith("владимирович"):
                return True
    return False


def has_supreme_non_court(tokens: list[str], window: int = 1) -> bool:
    # Оставляем упоминания "верховный*", но исключаем судебный контекст вида
    # "верховный суд" / "верховного суда".
    for i, tok in enumerate(tokens):
        if not tok.startswith("верховн"):
            continue
        left = max(0, i - window)
        right = min(len(tokens), i + window + 1)
        neighborhood = tokens[left:right]
        if any(t.startswith("суд") for t in neighborhood):
            continue
        return True
    return False


def include_query(tokens: list[str]) -> bool:
    # (Президент | Верховный | Путин*) | (Владимир* /1 Владимирович*)
    if any(t.startswith("путин") for t in tokens):
        return True
    if any(t.startswith("президент") for t in tokens):
        return True
    if has_supreme_non_court(tokens):
        return True
    if has_name_patronymic(tokens):
        return True
    return False


def find_countries_near_president(
    tokens: list[str],
    exact_to_countries: dict[str, set[str]],
    lemma_to_countries: dict[str, set[str]],
    window: int = 2,
) -> set[str]:
    matched: set[str] = set()
    ambiguous_tokens = {"того"}
    for i, t in enumerate(tokens):
        if not t.startswith("президент"):
            continue
        left = max(0, i - window)
        right = min(len(tokens), i + window + 1)
        for w in tokens[left:right]:
            if w.startswith("президент"):
                continue
            if w in ambiguous_tokens:
                continue

            countries = exact_to_countries.get(w)
            if countries and len(countries) == 1:
                matched.add(next(iter(countries)))
                continue

            countries = lemma_to_countries.get(lemma_ru_token(w))
            if countries:
                # Игнорируем неоднозначные леммы, соответствующие нескольким странам.
                if len(countries) == 1:
                    matched.add(next(iter(countries)))
    return matched


def views_to_int(v):
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        s = v.replace(" ", "").strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            return None
    return None


def write_country_report(counter: Counter, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["country", "count"])
        for country, cnt in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            writer.writerow([country, cnt])


def write_country_report_with_examples(
    counter: Counter,
    examples: dict[str, list[str]],
    out_path: Path,
    max_examples: int = 10,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["country", "count", "post_text_examples"]
        writer.writerow(header)
        for country, cnt in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            ex = (examples.get(country) or [])[:max_examples]
            row = [country, cnt, json.dumps(ex, ensure_ascii=False)]
            writer.writerow(row)


def build_date_plot_with_cutoffs(
    date_counts: Counter,
    out_png: Path,
    out_csv: Path,
    cutoff_dates: list[date],
) -> None:
    if not date_counts:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = sorted(date_counts.keys())
    vals = [date_counts[d] for d in dates]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("date,posts_count\n")
        for d in dates:
            f.write(f"{d.isoformat()},{date_counts[d]}\n")

    x = [d.isoformat() for d in dates]
    plt.figure(figsize=(14, 6))
    plt.plot(x, vals, linewidth=1.5, marker=None, label="Посты в день")

    for c in sorted(cutoff_dates):
        plt.axvline(c.isoformat(), linestyle="--", linewidth=1.4, label=c.isoformat())

    plt.title("Выгруженные посты по датам (с отсечками)")
    plt.xlabel("Дата")
    plt.ylabel("Количество постов")
    plt.grid(True, alpha=0.3)
    step = max(1, len(x) // 15)
    plt.xticks(x[::step], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)

    total = sum(vals)
    print(f"plot_total_posts={total}")
    print(f"plot_min_date={dates[0].isoformat()}")
    print(f"plot_max_date={dates[-1].isoformat()}")
    print(f"plot_png={out_png}")
    print(f"plot_csv={out_csv}")

    for c in sorted(cutoff_dates):
        before = sum(cnt for d, cnt in date_counts.items() if d < c)
        on = date_counts.get(c, 0)
        after = sum(cnt for d, cnt in date_counts.items() if d > c)
        print(f"cutoff={c.isoformat()} before={before} on_date={on} after={after}")

    # Статистика по сегментам между отсечками в хронологическом порядке.
    all_cutoffs = sorted(cutoff_dates)
    if all_cutoffs:
        prev = None
        for idx, cur in enumerate(all_cutoffs):
            if prev is None:
                seg = sum(cnt for d, cnt in date_counts.items() if d < cur)
                print(f"segment_before_{cur.isoformat()}={seg}")
            else:
                seg = sum(cnt for d, cnt in date_counts.items() if prev <= d < cur)
                print(f"segment_{prev.isoformat()}_to_{cur.isoformat()}_minus1d={seg}")
            prev = cur
        tail = sum(cnt for d, cnt in date_counts.items() if d >= all_cutoffs[-1])
        print(f"segment_from_{all_cutoffs[-1].isoformat()}={tail}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Фильтрация постов о Президенте РФ из posts_raw."
    )
    parser.add_argument(
        "--db-path",
        default="patriot_channels_posts_20260423_233414.sqlite",
        help="Путь к SQLite БД с таблицей posts_raw.",
    )
    parser.add_argument(
        "--out-jsonl",
        default="outputs/president_putin_selection_20260423_233414.jsonl",
        help="Выходной JSONL с отобранными постами.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200000,
        help="Интервал логирования прогресса.",
    )
    parser.add_argument(
        "--country-window",
        type=int,
        default=1,
        help=(
            "Размер окна слов вокруг 'президент' для поиска упоминаний стран. "
            "Меньшее окно означает более мягкий фильтр."
        ),
    )
    parser.add_argument(
        "--countries-report-csv",
        default="outputs/president_country_matches.csv",
        help="CSV-отчёт с найденными упоминаниями стран рядом со словом 'президент'.",
    )
    parser.add_argument(
        "--plot-png",
        default="outputs/president_putin_selection_by_date_cutoffs.png",
        help="Путь к выходному PNG с графиком по датам и отсечками.",
    )
    parser.add_argument(
        "--plot-csv",
        default="outputs/president_putin_selection_by_date.csv",
        help="Путь к выходному CSV со счётчиками по датам.",
    )
    parser.add_argument(
        "--cutoffs",
        default="2026-01-16",
        help="Даты отсечек через запятую в формате YYYY-MM-DD.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Отключить построение графика/CSV по датам.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Только перегенерировать график/CSV по датам из уже существующего --plot-csv, без чтения SQLite.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    country_report_path = Path(args.countries_report_csv)
    plot_png = Path(args.plot_png)
    plot_csv = Path(args.plot_csv)

    exact_to_countries, lemma_to_countries = load_country_index_ru()
    cutoff_dates = []
    for chunk in args.cutoffs.split(","):
        chunk = chunk.strip()
        if chunk:
            cutoff_dates.append(date.fromisoformat(chunk))

    if args.plot_only:
        if not plot_csv.exists():
            raise FileNotFoundError(
                f"Не найден --plot-csv для режима --plot-only: {plot_csv}"
            )
        date_counts = Counter()
        with plot_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    d = date.fromisoformat(str(row.get("date", "")).strip())
                except ValueError:
                    continue
                raw_cnt = row.get("posts_count", "0")
                try:
                    cnt = int(float(str(raw_cnt).strip()))
                except ValueError:
                    continue
                if cnt > 0:
                    date_counts[d] += cnt

        if not args.no_plot:
            build_date_plot_with_cutoffs(
                date_counts=date_counts,
                out_png=plot_png,
                out_csv=plot_csv,
                cutoff_dates=cutoff_dates,
            )
        print("Готово: перегенерация графика в режиме --plot-only.")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT id, payload_json FROM posts_raw")

    total = 0
    views_null = 0
    views_gt_1 = 0
    selected = 0
    selected_views_ok = 0
    excluded_country = 0
    excluded_country_names = Counter()
    excluded_country_examples: dict[str, list[str]] = defaultdict(list)
    found_country_names = Counter()
    date_counts = Counter()

    with out_path.open("w", encoding="utf-8") as out:
        for row in cur:
            total += 1
            payload = json.loads(row["payload_json"])

            v = views_to_int(payload.get("views"))
            if v is None:
                views_null += 1
            elif v > 1:
                views_gt_1 += 1

            text = payload.get("text") or ""
            tokens = tokenize(text)
            if not include_query(tokens):
                if total % args.log_every == 0:
                    print(f"processed={total:,} selected={selected:,}", flush=True)
                continue

            countries_near = find_countries_near_president(
                tokens=tokens,
                exact_to_countries=exact_to_countries,
                lemma_to_countries=lemma_to_countries,
                window=args.country_window,
            )
            for country_name in countries_near:
                found_country_names[country_name] += 1

            if countries_near:
                excluded_country += 1
                text = payload.get("text") or ""
                snippet = " ".join(text.split())
                if len(snippet) > 180:
                    snippet = snippet[:177] + "..."
                for country_name in countries_near:
                    excluded_country_names[country_name] += 1
                    if len(excluded_country_examples[country_name]) < 10:
                        excluded_country_examples[country_name].append(snippet)
                if total % args.log_every == 0:
                    print(f"processed={total:,} selected={selected:,}", flush=True)
                continue

            selected += 1

            if v is not None and v >= 1:
                selected_views_ok += 1
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                ds = payload.get("date")
                if ds:
                    d = datetime.fromisoformat(ds).date()
                    date_counts[d] += 1

            if total % args.log_every == 0:
                print(
                    f"processed={total:,} selected={selected:,} "
                    f"selected_views_ok={selected_views_ok:,}",
                    flush=True,
                )

    conn.close()

    print("=== RESULT ===")
    print(f"db_path: {db_path}")
    print("countries_source: CLDR (via Babel, open-source locale data)")
    print(f"country_exact_tokens_count: {len(exact_to_countries)}")
    print(f"country_lemmas_count: {len(lemma_to_countries)}")
    print(f"country_window: {args.country_window}")
    print(f"total_posts: {total}")
    print(f"views_null: {views_null}")
    print(f"views_gt_1: {views_gt_1}")
    print(f"selected_by_query_before_views: {selected}")
    print(f"excluded_by_country_near_president: {excluded_country}")
    print(f"selected_with_views_ge_1_written: {selected_views_ok}")
    print(f"output_jsonl: {out_path}")
    print(f"countries_found_near_president_total_unique: {len(found_country_names)}")
    write_country_report(found_country_names, country_report_path)
    print(f"countries_report_all_matches_csv: {country_report_path}")

    excluded_report_path = country_report_path.with_name(
        country_report_path.stem + "_excluded" + country_report_path.suffix
    )
    write_country_report_with_examples(
        counter=excluded_country_names,
        examples=excluded_country_examples,
        out_path=excluded_report_path,
        max_examples=10,
    )
    print(f"countries_report_excluded_csv: {excluded_report_path}")

    if found_country_names:
        print("top_countries_found_near_president:")
        for country, cnt in found_country_names.most_common(20):
            print(f"  {country}: {cnt}")

    if not args.no_plot:
        build_date_plot_with_cutoffs(
            date_counts=date_counts,
            out_png=plot_png,
            out_csv=plot_csv,
            cutoff_dates=cutoff_dates,
        )


if __name__ == "__main__":
    main()
