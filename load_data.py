"""Load the reporting CSVs into SQLite or PostgreSQL (e.g. Cloud SQL).

Only tables and columns listed in schema_catalog.py are loaded; any other
column in a CSV is dropped and reported. This is the "no sensitive data"
gate for the reporting database.

Usage:
  # Local SQLite file (what the container bundles)
  python load_data.py sqlite --data-dir ../data --db database.db

  # Cloud SQL (run from your laptop after `gcloud auth application-default login`)
  PGPASSWORD=... python load_data.py postgres --data-dir ../data \
      --instance als-user-office-software:us-central1:pubs-reporting \
      --user postgres --reader "pubschat-api@als-user-office-software.iam"

  # Any other Postgres
  PGPASSWORD=... python load_data.py postgres --host localhost --user postgres
"""

import argparse
import io
import logging
import os
import sqlite3
import sys
from contextlib import closing

import pandas as pd

import schema_catalog
from schema_catalog import Table

logger = logging.getLogger("load_data")


def read_table(table: Table, data_dir: str) -> pd.DataFrame:
    """Read one CSV and keep only the allow-listed columns, typed per catalog."""
    path = os.path.join(data_dir, table.csv_file)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    wanted = [c.name for c in table.columns]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    dropped = sorted(set(df.columns) - set(wanted))
    if dropped:
        logger.warning(f"{table.csv_file}: dropping columns not in catalog: {dropped}")
    df = df[wanted].copy()
    for col in table.columns:
        if col.pg_type == "INTEGER":
            df[col.name] = pd.to_numeric(df[col.name], errors="coerce").astype("Int64")
        elif col.pg_type == "REAL":
            df[col.name] = pd.to_numeric(df[col.name], errors="coerce")
        elif col.pg_type == "BOOLEAN":
            df[col.name] = df[col.name].map(
                lambda v: None if pd.isna(v) else str(v).strip().lower() in ("true", "t", "1", "yes")
            ).astype("boolean")
    logger.info(f"{table.csv_file}: {len(df):,} rows, {len(wanted)} columns")
    return df


def load_sqlite(data_dir: str, db_path: str) -> None:
    frames = {t.name: read_table(t, data_dir) for t in schema_catalog.TABLES}
    tmp = db_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    with closing(sqlite3.connect(tmp)) as conn:
        for t in schema_catalog.TABLES:
            cols = ", ".join(f"{c.name} {c.sqlite_type}" for c in t.columns)
            conn.execute(f"CREATE TABLE {t.name} ({cols})")
            df = frames[t.name]
            for c in t.columns:  # SQLite stores booleans as 0/1
                if c.sqlite_type == "BOOLEAN":
                    df[c.name] = df[c.name].astype("Int64")
            df.to_sql(t.name, conn, if_exists="append", index=False)
        conn.commit()
    os.replace(tmp, db_path)  # atomic swap: readers never see a half-built file
    logger.info(f"SQLite database written to {db_path}")


def _pg_connect(args):
    import pg8000.dbapi

    password = os.environ.get("PGPASSWORD")
    if args.instance:
        from google.cloud.sql.connector import Connector

        connector = Connector()
        kwargs = {"user": args.user, "db": args.dbname}
        if args.iam:
            kwargs["enable_iam_auth"] = True
        else:
            kwargs["password"] = password
        return connector, connector.connect(args.instance, "pg8000", **kwargs)
    conn = pg8000.dbapi.connect(host=args.host, port=args.port, user=args.user,
                                password=password, database=args.dbname)
    return None, conn


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_postgres(args) -> None:
    frames = {t.name: read_table(t, args.data_dir) for t in schema_catalog.TABLES}
    schema = args.schema
    connector, conn = _pg_connect(args)
    try:
        cur = conn.cursor()
        # Build in a staging schema, then swap, so queries never see partial data
        staging = f"{schema}_staging"
        cur.execute(f"DROP SCHEMA IF EXISTS {staging} CASCADE")
        cur.execute(f"CREATE SCHEMA {staging}")
        for t in schema_catalog.TABLES:
            cols = ", ".join(f"{c.name} {c.pg_type}" for c in t.columns)
            cur.execute(f"CREATE TABLE {staging}.{t.name} ({cols})")
            buf = io.StringIO()
            frames[t.name].to_csv(buf, index=False, header=False, na_rep="")
            buf.seek(0)
            col_list = ", ".join(c.name for c in t.columns)
            cur.execute(
                f"COPY {staging}.{t.name} ({col_list}) FROM STDIN WITH (FORMAT csv, NULL '')",
                stream=buf,
            )
            cur.execute(f"SELECT COUNT(*) FROM {staging}.{t.name}")
            logger.info(f"{staging}.{t.name}: {cur.fetchone()[0]:,} rows loaded")
        # Helpful indexes for the join and common filters
        cur.execute(f"CREATE INDEX ON {staging}.authors (dbpubid)")
        cur.execute(f"CREATE INDEX ON {staging}.pubsjournals (pubyear)")
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cur.execute(f"ALTER SCHEMA {staging} RENAME TO {schema}")
        for reader in args.reader or []:
            role = _quote_ident(reader)
            cur.execute(f"GRANT USAGE ON SCHEMA {schema} TO {role}")
            cur.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO {role}")
            # Belt and braces: the reader's sessions default to read-only. The
            # app also enforces this per query, so failure here is not fatal
            # (e.g. Cloud SQL may not let `postgres` alter IAM-created roles).
            cur.execute("SAVEPOINT role_defaults")
            try:
                cur.execute(f"ALTER ROLE {role} SET default_transaction_read_only = on")
                cur.execute(f"ALTER ROLE {role} SET statement_timeout = '10s'")
                cur.execute("RELEASE SAVEPOINT role_defaults")
            except Exception as e:
                cur.execute("ROLLBACK TO SAVEPOINT role_defaults")
                logger.warning(f"Could not set read-only defaults for {reader}: {e}")
            logger.info(f"Granted read-only access on {schema} to {reader}")
        conn.commit()
        logger.info(f"PostgreSQL schema '{schema}' replaced successfully")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
        if connector:
            connector.close()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="target", required=True)
    s = sub.add_parser("sqlite", help="write a SQLite file")
    s.add_argument("--data-dir", default="../data")
    s.add_argument("--db", default="database.db")
    g = sub.add_parser("postgres", help="load into PostgreSQL / Cloud SQL")
    g.add_argument("--data-dir", default="../data")
    g.add_argument("--instance", help="Cloud SQL connection name project:region:instance")
    g.add_argument("--host", default="localhost")
    g.add_argument("--port", type=int, default=5432)
    g.add_argument("--user", default="postgres")
    g.add_argument("--dbname", default="pubsreporting")
    g.add_argument("--schema", default="reporting")
    g.add_argument("--iam", action="store_true", help="IAM database auth instead of PGPASSWORD")
    g.add_argument("--reader", action="append", help="role to grant read-only access (repeatable)")
    args = p.parse_args(argv)
    if args.target == "sqlite":
        load_sqlite(args.data_dir, args.db)
    else:
        load_postgres(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
