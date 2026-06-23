"""PostgreSQL database client for ML pipeline metadata."""

import json
from contextlib import contextmanager
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor


class DatabaseClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "mlops",
        user: str = "mlops",
        password: str = "mlops123",
    ):
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }

    @contextmanager
    def get_cursor(self):
        conn = psycopg2.connect(**self.conn_params)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                yield cur
            conn.commit()
        finally:
            conn.close()

    def register_dataset(
        self, name: str, version: str, source: str, num_samples: int, split: str, metadata: dict = None
    ) -> int:
        with self.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO datasets (name, version, source, num_samples, split, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (name, version, source, num_samples, split, json.dumps(metadata or {})),
            )
            return cur.fetchone()["id"]

    def register_training_run(
        self, run_id: str, model_name: str, training_type: str, dataset_id: int, config: dict
    ):
        with self.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_runs (run_id, model_name, training_type, dataset_id, config)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (run_id, model_name, training_type, dataset_id, json.dumps(config)),
            )

    def update_training_run(self, run_id: str, status: str, metrics: dict = None):
        with self.get_cursor() as cur:
            cur.execute(
                """
                UPDATE training_runs
                SET status = %s, metrics = %s, completed_at = %s
                WHERE run_id = %s
                """,
                (status, json.dumps(metrics or {}), datetime.now(), run_id),
            )

    def log_evaluation(self, run_id: str, benchmark: str, metrics: dict):
        with self.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO evaluations (run_id, benchmark, metrics)
                VALUES (%s, %s, %s)
                """,
                (run_id, benchmark, json.dumps(metrics)),
            )

    def log_data_quality(self, dataset_id: int, check_name: str, passed: bool, details: dict = None):
        with self.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO data_quality_logs (dataset_id, check_name, passed, details)
                VALUES (%s, %s, %s, %s)
                """,
                (dataset_id, check_name, passed, json.dumps(details or {})),
            )

    def get_latest_run(self, training_type: str) -> dict | None:
        with self.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM training_runs
                WHERE training_type = %s
                ORDER BY started_at DESC LIMIT 1
                """,
                (training_type,),
            )
            return cur.fetchone()

    def get_evaluation_history(self, benchmark: str, limit: int = 10) -> list[dict]:
        with self.get_cursor() as cur:
            cur.execute(
                """
                SELECT e.*, t.model_name, t.training_type
                FROM evaluations e
                JOIN training_runs t ON e.run_id = t.run_id
                WHERE e.benchmark = %s
                ORDER BY e.evaluated_at DESC
                LIMIT %s
                """,
                (benchmark, limit),
            )
            return cur.fetchall()
