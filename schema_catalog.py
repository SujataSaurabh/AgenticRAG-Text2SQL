"""Reporting schema catalog: the single source of truth for what data exists.

Used for three things:
  1. Allow-list: load_data.py loads ONLY the tables/columns listed here, so a
     sensitive column added to a source export can never reach the database.
  2. DDL: column types for both SQLite and PostgreSQL.
  3. Prompts: SCHEMA_TEXT is the schema description given to the LLM.
     db.py also rejects SQL that references tables not listed here.

To add a table: add an entry to TABLES, re-run load_data.py, redeploy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    name: str
    sqlite_type: str
    pg_type: str
    description: str


@dataclass(frozen=True)
class Table:
    name: str          # lower-case; unquoted SQL identifiers match in both databases
    csv_file: str      # source file in the data directory
    description: str
    columns: tuple[Column, ...]


TABLES: tuple[Table, ...] = (
    Table(
        name="pubsjournals",
        csv_file="PUBSJOURNALS.csv",
        description=(
            "One row per ALS journal publication, with its journal and impact metrics. "
            "Data is complete through 2020; later years are only partially entered."
        ),
        columns=(
            Column("dbpubid", "INTEGER", "INTEGER", "Publication ID (joins to authors.dbpubid)"),
            Column("artchapthesttle", "TEXT", "TEXT", "Publication title"),
            Column("pubyear", "INTEGER", "INTEGER", "Publication year"),
            Column("refereed", "BOOLEAN", "BOOLEAN", "Peer-reviewed publication"),
            Column("beamline", "TEXT", "TEXT", "ALS beamline used, e.g. '7.0.1' (may be NULL)"),
            Column("journalcode", "INTEGER", "INTEGER", "Journal ID"),
            Column("journaltitle", "TEXT", "TEXT", "Journal name"),
            Column("journalimpactfactor", "REAL", "REAL", "Journal impact factor"),
            Column("doehighimpact", "BOOLEAN", "BOOLEAN", "Journal is on the DOE high-impact list"),
            Column("builtauthorlist", "TEXT", "TEXT", "Formatted author list"),
            Column("publisting", "TEXT", "TEXT", "Full formatted citation (HTML)"),
        ),
    ),
    Table(
        name="authors",
        csv_file="AUTHORS.csv",
        description=(
            "One row per author per publication. A person (alsid) appears once per paper. "
            "Some dbpubid values refer to publications not in pubsjournals (non-journal types)."
        ),
        columns=(
            Column("alsid", "INTEGER", "INTEGER", "Person ID"),
            Column("lastname", "TEXT", "TEXT", "Author last name"),
            Column("firstname", "TEXT", "TEXT", "Author first name"),
            Column("institution", "TEXT", "TEXT", "Author's institution on that paper (may be NULL)"),
            Column("dbpubid", "INTEGER", "INTEGER", "Publication ID (joins to pubsjournals.dbpubid)"),
        ),
    ),
)

TABLE_NAMES = frozenset(t.name for t in TABLES)


def schema_text() -> str:
    """Compact schema description for LLM prompts."""
    blocks = []
    for t in TABLES:
        cols = ", ".join(f"{c.name} ({c.pg_type})" for c in t.columns)
        notes = "; ".join(f"{c.name} = {c.description}" for c in t.columns)
        blocks.append(
            f"Table: {t.name.upper()}\nColumns: {cols}\n"
            f"Description: {t.description}\nColumn notes: {notes}"
        )
    return "\n\n".join(blocks)
