from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Final

from .exceptions import PersistenceError
from .optuna_api import OptunaStudy


_REQUIRED_COLUMNS: Final = {
    "studies": {"study_id", "study_name"},
    "study_directions": {"direction", "study_id", "objective"},
    "study_user_attributes": {"study_id", "key", "value_json"},
    "study_system_attributes": {"study_id", "key", "value_json"},
    "trials": {
        "trial_id",
        "number",
        "study_id",
        "state",
        "datetime_start",
        "datetime_complete",
    },
    "trial_user_attributes": {"trial_id", "key", "value_json"},
    "trial_system_attributes": {"trial_id", "key", "value_json"},
    "trial_params": {
        "trial_id",
        "param_name",
        "param_value",
        "distribution_json",
    },
    "trial_values": {
        "trial_id",
        "objective",
        "value",
        "value_type",
    },
    "trial_intermediate_values": {
        "trial_id",
        "step",
        "intermediate_value",
        "intermediate_value_type",
    },
    "trial_heartbeats": {"trial_id", "heartbeat"},
}
_DEPENDENT_TABLES: Final = frozenset(_REQUIRED_COLUMNS) - {"studies"}


def _study_sqlite_path(study: OptunaStudy) -> Path | None:
    storage = getattr(study, "_storage", None)
    backend = getattr(storage, "_backend", storage)
    engine = getattr(backend, "engine", None)
    url = getattr(engine, "url", None)
    driver_name = getattr(url, "drivername", None)
    database = getattr(url, "database", None)
    if not isinstance(driver_name, str) or not driver_name.startswith("sqlite"):
        return None
    if not isinstance(database, str) or database in {"", ":memory:"}:
        return None
    return Path(database).expanduser().resolve()


def _supports_bulk_clone(
    connection: sqlite3.Connection,
    schema_name: str,
) -> bool:
    table_names = {
        row[0]
        for row in connection.execute(
            f"SELECT name FROM {schema_name}.sqlite_master WHERE type = 'table'"
        )
        if isinstance(row[0], str)
    }
    if not set(_REQUIRED_COLUMNS).issubset(table_names):
        return False
    for table_name, required in _REQUIRED_COLUMNS.items():
        columns = {
            row[1]
            for row in connection.execute(
                f'PRAGMA {schema_name}.table_info("{table_name}")'
            )
            if isinstance(row[1], str)
        }
        if not required.issubset(columns):
            return False
    dependent_tables = {
        table_name
        for table_name in table_names
        if any(
            row[2] in {"studies", "trials"}
            for row in connection.execute(
                f'PRAGMA {schema_name}.foreign_key_list("{table_name}")'
            )
        )
    }
    return dependent_tables == _DEPENDENT_TABLES


def populate_study_from_sqlite(
    study: OptunaStudy,
    db_path: str | Path,
    destination_name: str,
) -> bool:
    source_path = _study_sqlite_path(study)
    target_path = Path(db_path).expanduser().resolve()
    if source_path is None or not source_path.is_file() or not target_path.is_file():
        return False
    try:
        with sqlite3.connect(target_path, timeout=30) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            source_schema = "main"
            if source_path != target_path:
                connection.execute(
                    "ATTACH DATABASE ? AS mlpipelineholder_source",
                    (str(source_path),),
                )
                source_schema = "mlpipelineholder_source"
            if not _supports_bulk_clone(
                connection,
                "main",
            ) or not _supports_bulk_clone(connection, source_schema):
                return False
            connection.execute("BEGIN IMMEDIATE")
            source = connection.execute(
                f"SELECT study_id FROM {source_schema}.studies WHERE study_name = ?",
                (study.study_name,),
            ).fetchone()
            if source is None:
                return False
            destination = connection.execute(
                "SELECT study_id FROM studies WHERE study_name = ?",
                (destination_name,),
            ).fetchone()
            if destination is None:
                return False
            source_study_id = source[0]
            destination_study_id = destination[0]
            destination_rows = sum(
                connection.execute(
                    f'SELECT COUNT(*) FROM "{table_name}" WHERE study_id = ?',
                    (destination_study_id,),
                ).fetchone()[0]
                for table_name in (
                    "study_user_attributes",
                    "study_system_attributes",
                    "trials",
                )
            )
            if destination_rows:
                return False
            for table_name in (
                "study_user_attributes",
                "study_system_attributes",
            ):
                connection.execute(
                    f'INSERT INTO "{table_name}"(study_id, key, value_json) '
                    f'SELECT ?, key, value_json FROM {source_schema}."{table_name}" '
                    f'WHERE study_id = ?',
                    (destination_study_id, source_study_id),
                )

            source_trial_ids = [
                row[0]
                for row in connection.execute(
                    f"SELECT trial_id FROM {source_schema}.trials "
                    "WHERE study_id = ? ORDER BY trial_id",
                    (source_study_id,),
                )
            ]
            first_trial_id = connection.execute(
                "SELECT COALESCE(MAX(trial_id), 0) + 1 FROM trials"
            ).fetchone()[0]
            connection.execute(
                "CREATE TEMP TABLE mlpipelineholder_trial_map("
                "source_trial_id INTEGER PRIMARY KEY, "
                "destination_trial_id INTEGER NOT NULL)"
            )
            connection.executemany(
                "INSERT INTO mlpipelineholder_trial_map VALUES (?, ?)",
                (
                    (source_trial_id, first_trial_id + offset)
                    for offset, source_trial_id in enumerate(source_trial_ids)
                ),
            )
            connection.execute(
                "INSERT INTO trials("
                "trial_id, number, study_id, state, datetime_start, datetime_complete) "
                "SELECT trial_map.destination_trial_id, source_trials.number, ?, "
                "source_trials.state, source_trials.datetime_start, "
                f"source_trials.datetime_complete FROM {source_schema}.trials "
                "AS source_trials "
                "JOIN mlpipelineholder_trial_map AS trial_map "
                "ON trial_map.source_trial_id = source_trials.trial_id",
                (destination_study_id,),
            )
            for table_name, columns in (
                ("trial_user_attributes", "key, value_json"),
                ("trial_system_attributes", "key, value_json"),
                ("trial_params", "param_name, param_value, distribution_json"),
                ("trial_values", "objective, value, value_type"),
                (
                    "trial_intermediate_values",
                    "step, intermediate_value, intermediate_value_type",
                ),
                ("trial_heartbeats", "heartbeat"),
            ):
                selected_columns = ", ".join(
                    f"source_rows.{column.strip()}"
                    for column in columns.split(",")
                )
                connection.execute(
                    f'INSERT INTO "{table_name}"(trial_id, {columns}) '
                    f'SELECT trial_map.destination_trial_id, {selected_columns} '
                    f'FROM {source_schema}."{table_name}" AS source_rows '
                    f'JOIN mlpipelineholder_trial_map AS trial_map '
                    f'ON trial_map.source_trial_id = source_rows.trial_id'
                )
            connection.execute("DROP TABLE mlpipelineholder_trial_map")
    except sqlite3.DatabaseError as exc:
        raise PersistenceError(
            f"Failed to clone Optuna Study '{study.study_name}' in SQLite: {exc}"
        ) from exc
    return True
