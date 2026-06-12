#!/usr/bin/env python3
import argparse
import glob
import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Iterator


def batched(items: Iterable[tuple], size: int) -> Iterator[list[tuple]]:
    batch: list[tuple] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def progress(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = MEMORY;

        CREATE TABLE IF NOT EXISTS posts_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            channel_id INTEGER,
            channel_username TEXT,
            post_id INTEGER,
            post_date TEXT,
            payload_json TEXT NOT NULL,
            UNIQUE(source_file, line_number)
        );

        CREATE TABLE IF NOT EXISTS channels_enriched (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_json_path TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            channel_name TEXT,
            handle TEXT,
            normalized_handle TEXT,
            resolved_id INTEGER,
            parsed_jsonl_path TEXT,
            payload_json TEXT NOT NULL,
            UNIQUE(source_json_path, row_index)
        );

        CREATE INDEX IF NOT EXISTS idx_posts_channel_post
            ON posts_raw(channel_id, post_id);
        CREATE INDEX IF NOT EXISTS idx_channels_handle
            ON channels_enriched(handle);
        """
    )


def load_posts(
    conn: sqlite3.Connection,
    jsonl_pattern: str,
    batch_size: int,
    log_every_lines: int = 100000,
) -> tuple[int, int, int]:
    files = sorted(glob.glob(jsonl_pattern))
    if not files:
        raise FileNotFoundError(f"No files found by pattern: {jsonl_pattern}")

    inserted = 0
    skipped_duplicates = 0
    broken_lines = 0

    insert_sql = """
        INSERT OR IGNORE INTO posts_raw (
            source_file,
            line_number,
            channel_id,
            channel_username,
            post_id,
            post_date,
            payload_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    def rows() -> Iterator[tuple]:
        nonlocal broken_lines
        total_files = len(files)
        total_processed = 0
        for file_idx, file_path in enumerate(files, start=1):
            progress(f"[posts] file {file_idx}/{total_files}: {file_path}")
            file_processed = 0
            with open(file_path, "r", encoding="utf-8") as f:
                for line_number, raw_line in enumerate(f, start=1):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        payload = json.loads(raw_line)
                    except json.JSONDecodeError:
                        broken_lines += 1
                        continue
                    file_processed += 1
                    total_processed += 1
                    if file_processed % log_every_lines == 0:
                        progress(
                            f"[posts] processed {file_processed:,} valid lines in current file; "
                            f"{total_processed:,} total"
                        )
                    yield (
                        str(file_path),
                        line_number,
                        payload.get("channel_id"),
                        payload.get("channel_username"),
                        payload.get("post_id"),
                        payload.get("date"),
                        json.dumps(payload, ensure_ascii=False),
                    )
            progress(
                f"[posts] finished file {file_idx}/{total_files}, "
                f"valid lines: {file_processed:,}, broken so far: {broken_lines:,}"
            )

    cursor = conn.cursor()
    batch_no = 0
    for batch in batched(rows(), batch_size):
        batch_no += 1
        before_changes = conn.total_changes
        cursor.executemany(insert_sql, batch)
        conn.commit()
        delta = conn.total_changes - before_changes
        inserted += delta
        skipped_duplicates += len(batch) - delta
        if batch_no % 50 == 0:
            progress(
                f"[posts] committed batches: {batch_no:,}; inserted: {inserted:,}; "
                f"skipped/duplicates in loaded batches: {skipped_duplicates:,}"
            )

    return len(files), inserted, skipped_duplicates + broken_lines


def load_channels(
    conn: sqlite3.Connection,
    channels_json_path: str,
    batch_size: int,
    log_every_rows: int = 1000,
) -> tuple[int, int]:
    source_path = Path(channels_json_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing channels file: {channels_json_path}")

    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Channels JSON must contain an array of objects.")

    rows: list[tuple] = []
    for idx, payload in enumerate(data, start=1):
        if not isinstance(payload, dict):
            continue
        rows.append(
            (
                str(source_path),
                idx,
                payload.get("Channel Name"),
                payload.get("Handle"),
                payload.get("normalized_handle"),
                payload.get("resolved_id_y", payload.get("resolved_id_x")),
                payload.get("parsed_jsonl_path"),
                json.dumps(payload, ensure_ascii=False),
            )
        )
        if idx % log_every_rows == 0:
            progress(f"[channels] prepared rows: {idx:,}")

    insert_sql = """
        INSERT OR IGNORE INTO channels_enriched (
            source_json_path,
            row_index,
            channel_name,
            handle,
            normalized_handle,
            resolved_id,
            parsed_jsonl_path,
            payload_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    cursor = conn.cursor()
    inserted = 0
    skipped_duplicates = 0

    progress(f"[channels] total rows to load: {len(rows):,}")
    batch_no = 0
    for batch in batched(rows, batch_size):
        batch_no += 1
        before_changes = conn.total_changes
        cursor.executemany(insert_sql, batch)
        conn.commit()
        delta = conn.total_changes - before_changes
        inserted += delta
        skipped_duplicates += len(batch) - delta
        if batch_no % 5 == 0:
            progress(
                f"[channels] committed batches: {batch_no:,}; "
                f"inserted: {inserted:,}; skipped: {skipped_duplicates:,}"
            )

    return inserted, skipped_duplicates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Загрузка patriot JSONL-постов и enriched channels JSON в SQLite."
    )
    parser.add_argument(
        "--jsonl-pattern",
        default="inputs/patriot_channels_posts_20260423_233414_*.jsonl",
        help="Glob-шаблон для JSONL-файлов с постами.",
    )
    parser.add_argument(
        "--channels-json",
        default="inputs/patriot_channels_posts_channels_enriched_20260423_233414.json",
        help="Путь к enriched JSON-файлу с каналами.",
    )
    parser.add_argument(
        "--db-path",
        default="patriot_channels_posts_20260423_233414.sqlite",
        help="Путь к выходной SQLite БД.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Размер батча для вставки.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        progress(f"Opening DB: {db_path}")
        create_schema(conn)
        progress("Schema ready")

        files_count, posts_inserted, posts_skipped = load_posts(
            conn=conn,
            jsonl_pattern=args.jsonl_pattern,
            batch_size=args.batch_size,
        )
        progress("Posts loading finished")
        channels_inserted, channels_skipped = load_channels(
            conn=conn,
            channels_json_path=args.channels_json,
            batch_size=args.batch_size,
        )
        progress("Channels loading finished")

        print(f"DB: {db_path}")
        print(f"JSONL files processed: {files_count}")
        print(f"posts_raw inserted: {posts_inserted}")
        print(f"posts_raw skipped/invalid: {posts_skipped}")
        print(f"channels_enriched inserted: {channels_inserted}")
        print(f"channels_enriched skipped: {channels_skipped}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
