"""SQLite persistence for web analysis records."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parents[1]
DB_DIR = BASE_DIR / "data" / "db"
DB_PATH = DB_DIR / "analysis.sqlite3"


def _connect() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT NOT NULL,
                detections_json TEXT NOT NULL,
                overlay_filename TEXT NOT NULL,
                mask_filename TEXT NOT NULL,
                vision_answer TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_record(result: Dict[str, Any]) -> int:
    created_at = datetime.now().isoformat(timespec="seconds")
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analysis_records
            (image_name, detections_json, overlay_filename, mask_filename,
             vision_answer, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result["image_name"],
                json.dumps(result["detections"], ensure_ascii=False),
                Path(result["segmentation_overlay"]).name,
                Path(result["segmentation_mask"]).name,
                result["vision_answer"],
                created_at,
            ),
        )
        return int(cursor.lastrowid)


def list_records(limit: int = 20) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 100))
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, image_name, detections_json, overlay_filename,
                   mask_filename, vision_answer, created_at
            FROM analysis_records
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    records: List[Dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "id": row["id"],
                "image_name": row["image_name"],
                "detections": json.loads(row["detections_json"]),
                "overlay_url": f"/api/output/{row['overlay_filename']}",
                "mask_url": f"/api/output/{row['mask_filename']}",
                "vision_answer": row["vision_answer"],
                "created_at": row["created_at"],
            }
        )
    return records
