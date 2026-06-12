#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_KEYWORDS = [
    "telegram россии",
    "telegram",
    "мессенджер",
    "мессенджере",
    "телеграм",
    "tg",
    "max",
    "макс",
    "блокировка",
    "блокировать",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Анализирует top N топиков по дельте доли (до/после отсечки), "
            "считает их сентимент и формирует CSV по топ-авторам."
        )
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("topic_model_outputs_dynamic_putin_user_bge/docs_with_topics_and_sentiment.parquet"),
        help="Путь к docs_with_topics_and_sentiment.parquet.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("topic_model_outputs_dynamic_putin_user_bge"),
        help="Папка для выходных CSV.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="2026-01-16",
        help="Дата отсечки в формате YYYY-MM-DD (день 'после').",
    )
    parser.add_argument(
        "--top-n-topics",
        type=int,
        default=20,
        help="Сколько топиков брать по abs(delta_share).",
    )
    parser.add_argument(
        "--top-n-authors",
        type=int,
        default=50,
        help="Сколько авторов выгружать в авторский CSV.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help=(
            "Ключевые слова для поиска в topic_name. "
            "Если topic_name содержит любое из них, топик считается целевым."
        ),
    )
    return parser.parse_args()


def ensure_required_columns(df: pd.DataFrame) -> None:
    required = {
        "date_day",
        "topic_id",
        "topic_name",
        "ensemble_negative_prob",
        "ensemble_positive_prob",
        "channel_id",
        "channel_title",
        "channel_username",
        "text",
        "post_url",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"В parquet не хватает обязательных колонок: {missing}")


def build_author_name(df: pd.DataFrame) -> pd.Series:
    username = df["channel_username"].fillna("").astype(str).str.strip()
    title = df["channel_title"].fillna("").astype(str).str.strip()
    channel_id = df["channel_id"].astype("Int64").astype(str)
    return (
        username.where(username.ne(""), "")
        .where(username.eq(""), "@" + username)
        .where(username.ne(""), title.where(title.ne(""), "id_" + channel_id))
    )


def prepare_frame(input_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    ensure_required_columns(df)

    df = df.copy()
    df["date_day"] = pd.to_datetime(df["date_day"], errors="coerce", utc=True)
    df = df[df["date_day"].notna()].copy()
    df["topic_id"] = pd.to_numeric(df["topic_id"], errors="coerce").astype("Int64")
    df = df[df["topic_id"].notna()].copy()
    df["topic_id"] = df["topic_id"].astype(int)
    df["topic_name"] = df["topic_name"].fillna("").astype(str)
    df["ensemble_positive_prob"] = pd.to_numeric(df["ensemble_positive_prob"], errors="coerce")
    df["ensemble_negative_prob"] = pd.to_numeric(df["ensemble_negative_prob"], errors="coerce")
    df["author_name"] = build_author_name(df)

    return df


def calc_topic_delta_table(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    valid_topics = df[~df["topic_id"].isin([-1, 0])].copy()
    before_mask = valid_topics["date_day"] < cutoff_date
    after_mask = valid_topics["date_day"] >= cutoff_date

    before = valid_topics[before_mask].groupby("topic_id", dropna=False).size().rename("before_n")
    after = valid_topics[after_mask].groupby("topic_id", dropna=False).size().rename("after_n")

    merged = (
        before.to_frame()
        .merge(after.to_frame(), left_index=True, right_index=True, how="outer")
        .fillna(0)
        .reset_index()
    )
    merged["before_n"] = merged["before_n"].astype(int)
    merged["after_n"] = merged["after_n"].astype(int)

    total_before = max(1, int(merged["before_n"].sum()))
    total_after = max(1, int(merged["after_n"].sum()))
    merged["before_share"] = merged["before_n"] / total_before
    merged["after_share"] = merged["after_n"] / total_after
    merged["delta_share"] = merged["after_share"] - merged["before_share"]
    merged["abs_delta_share"] = merged["delta_share"].abs()

    topic_names = (
        valid_topics[["topic_id", "topic_name"]]
        .drop_duplicates(subset=["topic_id"], keep="first")
        .copy()
    )
    merged = merged.merge(topic_names, on="topic_id", how="left")
    return merged.sort_values("abs_delta_share", ascending=False)


def calc_top_topics_sentiment(df: pd.DataFrame, top_topics: list[int], delta_table: pd.DataFrame) -> pd.DataFrame:
    top_df = df[df["topic_id"].isin(top_topics)].copy()
    if top_df.empty:
        return pd.DataFrame()

    agg = (
        top_df.groupby(["topic_id", "topic_name"], dropna=False)
        .agg(
            posts_n=("topic_id", "size"),
            positive_prob_mean=("ensemble_positive_prob", "mean"),
            negative_prob_mean=("ensemble_negative_prob", "mean"),
            positive_prob_median=("ensemble_positive_prob", "median"),
            negative_prob_median=("ensemble_negative_prob", "median"),
        )
        .reset_index()
    )

    cols = ["topic_id", "before_n", "after_n", "before_share", "after_share", "delta_share", "abs_delta_share"]
    agg = agg.merge(delta_table[cols], on="topic_id", how="left")
    return agg.sort_values("abs_delta_share", ascending=False)


def calc_top_authors_keyword_topic(df: pd.DataFrame, keywords: list[str], top_n_authors: int) -> pd.DataFrame:
    keys = [k.lower().strip() for k in keywords if str(k).strip()]
    if not keys:
        raise SystemExit("Список ключевых слов пуст.")

    topic_name_lc = df["topic_name"].fillna("").astype(str).str.lower()
    pattern = "|".join(keys)
    keyword_mask = topic_name_lc.str.contains(pattern, regex=True, na=False)

    df_kw = df.copy()
    df_kw["is_keyword_topic"] = keyword_mask

    channel_meta = (
        df_kw.assign(
            channel_title_clean=df_kw["channel_title"].fillna("").astype(str).str.strip(),
            channel_username_clean=df_kw["channel_username"].fillna("").astype(str).str.strip(),
        )
        .groupby("channel_id", dropna=False)
        .agg(
            channel_title=("channel_title_clean", lambda s: s[s.ne("")].mode().iloc[0] if (s.ne("")).any() else ""),
            channel_username=("channel_username_clean", lambda s: s[s.ne("")].mode().iloc[0] if (s.ne("")).any() else ""),
        )
        .reset_index()
    )

    author_totals = (
        df_kw.groupby(["channel_id", "author_name"], dropna=False)
        .agg(
            posts_total=("topic_id", "size"),
            posts_keyword_topic=("is_keyword_topic", "sum"),
            positive_prob_mean_all=("ensemble_positive_prob", "mean"),
            negative_prob_mean_all=("ensemble_negative_prob", "mean"),
        )
        .reset_index()
    )

    keyword_only = (
        df_kw[df_kw["is_keyword_topic"]]
        .groupby(["channel_id", "author_name"], dropna=False)
        .agg(
            positive_prob_mean_keyword_topic=("ensemble_positive_prob", "mean"),
            negative_prob_mean_keyword_topic=("ensemble_negative_prob", "mean"),
        )
        .reset_index()
    )

    keyword_posts = df_kw[df_kw["is_keyword_topic"]].copy()
    keyword_posts["text_clean"] = (
        keyword_posts["text"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    keyword_posts["post_url_clean"] = keyword_posts["post_url"].fillna("").astype(str).str.strip()
    keyword_posts = keyword_posts.sort_values(["channel_id", "date_day"], ascending=[True, False])
    keyword_posts = (
        keyword_posts
        .groupby(["channel_id", "author_name"], dropna=False, group_keys=False)
        .head(5)
        .copy()
    )
    keyword_posts["example_text"] = keyword_posts["text_clean"].str.slice(0, 220)
    keyword_posts["example_row"] = (
        keyword_posts["date_day"].dt.strftime("%Y-%m-%d").fillna("")
        + " | "
        + keyword_posts["example_text"]
        + " | "
        + keyword_posts["post_url_clean"]
    )
    keyword_examples = (
        keyword_posts
        .groupby(["channel_id", "author_name"], dropna=False)["example_row"]
        .apply(list)
        .rename("keyword_topic_examples")
        .reset_index()
    )
    if not keyword_examples.empty:
        examples_wide = keyword_examples["keyword_topic_examples"].apply(
            lambda vals: pd.Series({f"keyword_post_example_{i+1}": vals[i] if i < len(vals) else "" for i in range(5)})
        )
        keyword_examples = pd.concat(
            [keyword_examples.drop(columns=["keyword_topic_examples"]), examples_wide],
            axis=1,
        )

    out = (
        author_totals
        .merge(keyword_only, on=["channel_id", "author_name"], how="left")
        .merge(channel_meta, on="channel_id", how="left")
        .merge(keyword_examples, on=["channel_id", "author_name"], how="left")
    )
    out["keyword_topic_popularity"] = out["posts_keyword_topic"] / out["posts_total"].clip(lower=1)
    out["channel_id"] = out["channel_id"].astype("Int64")
    out = out.sort_values(
        ["negative_prob_mean_keyword_topic", "posts_keyword_topic", "posts_total"],
        ascending=[False, False, False],
        na_position="last",
    ).head(max(1, top_n_authors))
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cutoff_date = pd.to_datetime(args.cutoff_date, utc=True).normalize()
    df = prepare_frame(args.input_parquet)

    delta_table = calc_topic_delta_table(df, cutoff_date=cutoff_date)
    top_topics = delta_table.head(max(1, args.top_n_topics))["topic_id"].astype(int).tolist()

    top_topics_sent = calc_top_topics_sentiment(df=df, top_topics=top_topics, delta_table=delta_table)
    top_authors = calc_top_authors_keyword_topic(
        df=df,
        keywords=args.keywords,
        top_n_authors=args.top_n_authors,
    )

    topics_path = args.out_dir / "top_topics_delta_sentiment.csv"
    authors_path = args.out_dir / "top_authors_keyword_topic_sentiment.csv"

    top_topics_sent.to_csv(topics_path, index=False)
    top_authors.to_csv(authors_path, index=False)

    print(f"Сохранен CSV top N топиков по дельте: {topics_path}")
    print(f"Сохранен CSV top авторов: {authors_path}")
    print(f"Топиков по дельте: {len(top_topics_sent)} | Авторов: {len(top_authors)}")


if __name__ == "__main__":
    main()
