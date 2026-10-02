"""Database access for the chatbot: SQLite file or PostgreSQL (e.g. Cloud SQL).

Select with DB_BACKEND:
  sqlite    -> read-only SQLite file at DB_PATH (default; bundled in the image)
  postgres  -> PostgreSQL. With CLOUDSQL_INSTANCE set, connects through the
               Cloud SQL Python Connector (IAM auth, no password); otherwise
               to PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE directly.

Every query is checked before it runs: one SELECT statement, only tables from
the schema catalog. Postgres queries also run in a READ ONLY transaction with
a statement timeout.
"""

import logging
import os
import sqlite3
from contextlib import closing

import sqlglot
from sqlglot import expressions as exp

import schema_catalog

logger = logging.getLogger("Text2SQL_RAG")

DB_BACKEND = os.getenv("DB_BACKEND", "sqlite").lower()
DB_PATH = os.getenv("DB_PATH", "database.db")
MAX_ROWS = int(os.getenv("MAX_ROWS", "5000"))
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "10000"))

# Postgres settings
CLOUDSQL_INSTANCE = os.getenv("CLOUDSQL_INSTANCE")  # project:region:instance
PG_SCHEMA = os.getenv("PG_SCHEMA", "reporting")
DB_NAME = os.getenv("DB_NAME", os.getenv("PGDATABASE", "pubsreporting"))
DB_USER = os.getenv("DB_USER", os.getenv("PGUSER"))  # IAM user for Cloud SQL
DB_IAM_AUTH = os.getenv("DB_IAM_AUTH", "true").lower() == "true"

if DB_BACKEND == "sqlite":
    DIALECT, DIALECT_NAME, LIKE_OP = "sqlite", "SQLite", "LIKE"
elif DB_BACKEND == "postgres":
    # SQLite's LIKE is case-insensitive; ILIKE gives Postgres the same behavior
    DIALECT, DIALECT_NAME, LIKE_OP = "postgres", "PostgreSQL", "ILIKE"
else:
    raise ValueError(f"Unknown DB_BACKEND={DB_BACKEND!r}; use 'sqlite' or 'postgres'")

_connector = None


class UnsafeQueryError(ValueError):
    """The generated SQL is not a single read-only query over allowed tables."""


def validate_sql(sql: str) -> None:
    """Reject anything but one SELECT over catalog tables. Raises UnsafeQueryError."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeQueryError(f"Could not parse SQL: {e}") from e
    if len(statements) != 1:
        raise UnsafeQueryError("Exactly one SQL statement is allowed.")
    stmt = statements[0]
    if not isinstance(stmt, exp.Query):
        raise UnsafeQueryError(f"Only SELECT queries are allowed, got {stmt.key.upper()}.")
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop,
                 exp.Alter, exp.Command, exp.Merge)
    if any(stmt.find(t) for t in forbidden):
        raise UnsafeQueryError("Data-modifying statements are not allowed.")
    cte_names = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
    tables = {t.name.lower() for t in stmt.find_all(exp.Table)} - cte_names
    unknown = tables - schema_catalog.TABLE_NAMES
    if unknown:
        raise UnsafeQueryError(
            f"Unknown table(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(schema_catalog.TABLE_NAMES))}."
        )


def _pg_connect():
    import pg8000.dbapi

    global _connector
    if CLOUDSQL_INSTANCE:
        from google.cloud.sql.connector import Connector, RefreshStrategy

        if _connector is None:
            # LAZY refresh suits Cloud Run, where CPU is throttled between requests
            _connector = Connector(refresh_strategy=RefreshStrategy.LAZY)
        kwargs = {"user": DB_USER, "db": DB_NAME}
        if DB_IAM_AUTH:
            kwargs["enable_iam_auth"] = True
        else:
            kwargs["password"] = os.environ["PGPASSWORD"]
        return _connector.connect(CLOUDSQL_INSTANCE, "pg8000", **kwargs)
    return pg8000.dbapi.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        user=DB_USER,
        password=os.getenv("PGPASSWORD"),
        database=DB_NAME,
    )


def run_query(sql: str) -> tuple[list[str], list[tuple]]:
    """Validate and run a read-only query. Returns (column names, rows)."""
    validate_sql(sql)
    if DB_BACKEND == "sqlite":
        with closing(sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchmany(MAX_ROWS)
            cols = [d[0] for d in cur.description] if cur.description else []
        return cols, rows

    with closing(_pg_connect()) as conn:
        cur = conn.cursor()
        try:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
            cur.execute(f"SET LOCAL search_path TO {PG_SCHEMA}")
            cur.execute(sql)
            rows = [tuple(r) for r in cur.fetchmany(MAX_ROWS)]
            cols = [d[0] for d in cur.description] if cur.description else []
        finally:
            conn.rollback()
    return cols, rows


def check_connection() -> None:
    """Fail fast at startup if the database is unreachable."""
    if DB_BACKEND == "sqlite":
        if not os.path.exists(DB_PATH):
            raise RuntimeError(f"Database file {DB_PATH} not found")
    first = schema_catalog.TABLES[0].name
    cols, rows = run_query(f"SELECT COUNT(*) AS n FROM {first}")
    if DB_BACKEND == "postgres":
        target = CLOUDSQL_INSTANCE or os.getenv("PGHOST", "localhost")
    else:
        target = DB_PATH
    logger.info(f"[DB] {DIALECT_NAME} OK ({target}): {first} has {rows[0][0]} rows")


def close() -> None:
    global _connector
    if _connector is not None:
        _connector.close()
        _connector = None
