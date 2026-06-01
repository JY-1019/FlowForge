"""SQLite persistence for the Harness Ledger.

A single :class:`LedgerStore` owns one SQLite file (``./ledger.db`` by
default) and holds every table the ledger needs: run records, the per-node
trajectory, branch/planner decisions, contract violations, accumulated
failures, and repair suggestions (with their apply/rollback history).

The store is deliberately dependency-free (stdlib ``sqlite3`` only) and
thread-tolerant: ``check_same_thread=False`` plus a coarse write lock means it
is safe to call from whatever thread the async runner happens to be on.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  REAL,
    finished_at REAL,
    input_repr  TEXT,
    output_repr TEXT,
    status      TEXT,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS trajectory (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT,
    node_id         TEXT,
    node_type       TEXT,
    name            TEXT,
    execution_order INTEGER,
    started_at      REAL,
    finished_at     REAL,
    duration_ms     REAL,
    input_repr      TEXT,
    output_repr     TEXT,
    status          TEXT,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT,
    node_id   TEXT,
    kind      TEXT,
    selected  TEXT,
    rationale TEXT,
    ts        REAL
);

CREATE TABLE IF NOT EXISTS contract_violations (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   TEXT,
    node_id  TEXT,
    contract TEXT,
    reason   TEXT,
    ts       REAL
);

CREATE TABLE IF NOT EXISTS failures (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT,
    node_id   TEXT,
    step_name TEXT,
    signature TEXT,
    error     TEXT,
    ts        REAL
);

CREATE TABLE IF NOT EXISTS suggestions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL,
    signature   TEXT UNIQUE,
    step_name   TEXT,
    cause       TEXT,
    suggestion  TEXT,
    source_file TEXT,
    func_name   TEXT,
    old_source  TEXT,
    new_source  TEXT,
    status      TEXT
);

CREATE TABLE IF NOT EXISTS applies (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    suggestion_id INTEGER,
    applied_at    REAL,
    target_file   TEXT,
    backup_source TEXT,
    status        TEXT
);
"""


class LedgerStore:
    """Thin SQLite wrapper for all ledger tables."""

    def __init__(self, db_path: str = "./ledger.db") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def start_run(self, run_id: str, input_repr: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs(run_id, started_at, input_repr, status) "
                "VALUES (?, ?, ?, 'running')",
                (run_id, time.time(), input_repr),
            )
            self._conn.commit()

    def finish_run(self, run_id: str, output_repr: str, error: str | None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET finished_at=?, output_repr=?, status=?, error=? "
                "WHERE run_id=?",
                (
                    time.time(),
                    output_repr,
                    "failed" if error else "ok",
                    error,
                    run_id,
                ),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Trajectory / decisions / violations
    # ------------------------------------------------------------------

    def record_node(self, **row: Any) -> None:
        cols = (
            "run_id", "node_id", "node_type", "name", "execution_order",
            "started_at", "finished_at", "duration_ms", "input_repr",
            "output_repr", "status", "error",
        )
        with self._lock:
            self._conn.execute(
                f"INSERT INTO trajectory({','.join(cols)}) "
                f"VALUES ({','.join('?' for _ in cols)})",
                tuple(row.get(c) for c in cols),
            )
            self._conn.commit()

    def record_decision(
        self,
        run_id: str,
        node_id: str,
        kind: str,
        selected: str,
        rationale: str = "",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO decisions(run_id, node_id, kind, selected, rationale, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, node_id, kind, selected, rationale, time.time()),
            )
            self._conn.commit()

    def record_violation(
        self, run_id: str, node_id: str, contract: str, reason: str
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO contract_violations(run_id, node_id, contract, reason, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, node_id, contract, reason, time.time()),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Failures (cross-run accumulation for the detector)
    # ------------------------------------------------------------------

    def record_failure(
        self, run_id: str, node_id: str, step_name: str, signature: str, error: str
    ) -> int:
        """Insert a failure and return the running count for this signature."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO failures(run_id, node_id, step_name, signature, error, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, node_id, step_name, signature, error, time.time()),
            )
            self._conn.commit()
            cur = self._conn.execute(
                "SELECT COUNT(*) AS c FROM failures WHERE signature=?", (signature,)
            )
            return int(cur.fetchone()["c"])

    # ------------------------------------------------------------------
    # Suggestions + apply/rollback
    # ------------------------------------------------------------------

    def upsert_suggestion(self, **row: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO suggestions(created_at, signature, step_name, cause, "
                "suggestion, source_file, func_name, old_source, new_source, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending') "
                "ON CONFLICT(signature) DO NOTHING",
                (
                    time.time(), row.get("signature"), row.get("step_name"),
                    row.get("cause"), row.get("suggestion"), row.get("source_file"),
                    row.get("func_name"), row.get("old_source"), row.get("new_source"),
                ),
            )
            self._conn.commit()

    def pending_suggestions(self) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM suggestions WHERE status='pending' ORDER BY created_at"
            )
            return cur.fetchall()

    def mark_suggestion(self, suggestion_id: int, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE suggestions SET status=? WHERE id=?", (status, suggestion_id)
            )
            self._conn.commit()

    def record_apply(
        self, suggestion_id: int, target_file: str, backup_source: str
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO applies(suggestion_id, applied_at, target_file, "
                "backup_source, status) VALUES (?, ?, ?, ?, 'applied')",
                (suggestion_id, time.time(), target_file, backup_source),
            )
            self._conn.commit()

    def last_applied(self) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM applies WHERE status='applied' "
                "ORDER BY applied_at DESC LIMIT 1"
            )
            return cur.fetchone()

    def mark_apply(self, apply_id: int, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE applies SET status=? WHERE id=?", (status, apply_id)
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Read queries (CLI)
    # ------------------------------------------------------------------

    def recent_runs(self, limit: int = 10, failed_only: bool = False) -> list[sqlite3.Row]:
        q = "SELECT * FROM runs"
        if failed_only:
            q += " WHERE status='failed'"
        q += " ORDER BY started_at DESC LIMIT ?"
        with self._lock:
            return self._conn.execute(q, (limit,)).fetchall()

    def nodes_for_run(self, run_id: str, failed_only: bool = False) -> list[sqlite3.Row]:
        q = "SELECT * FROM trajectory WHERE run_id=?"
        if failed_only:
            q += " AND status='error'"
        q += " ORDER BY execution_order"
        with self._lock:
            return self._conn.execute(q, (run_id,)).fetchall()

    def decisions_for_run(self, run_id: str) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM decisions WHERE run_id=? ORDER BY ts", (run_id,)
            ).fetchall()

    def violations_for_run(self, run_id: str) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM contract_violations WHERE run_id=? ORDER BY ts",
                (run_id,),
            ).fetchall()
