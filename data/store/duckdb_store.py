"""DuckDB-based local storage for market data, factors, and backtest results."""

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


class DuckDBStore:
    """Simple DuckDB storage manager. Single-file, zero-config.

    Tables are created on first write. Schema is derived from the DataFrame.
    """

    def __init__(self, db_path: str = "data/trading_god.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
        return self._conn

    def write_df(self, table_name: str, df: pd.DataFrame, mode: str = "append") -> None:
        """Write a DataFrame to a table. Creates table if not exists.

        Args:
            table_name: Target table name.
            df: Data to write.
            mode: 'append' or 'replace'.
        """
        if df.empty:
            return

        if mode == "replace":
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df LIMIT 0"
        )

        # Register df as temp view and insert
        self.conn.register("_temp_df", df)
        self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM _temp_df")
        self.conn.unregister("_temp_df")

    def read_df(
        self,
        table_name: str,
        columns: Optional[list[str]] = None,
        where: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Read data from a table with optional filters.

        Args:
            table_name: Source table name.
            columns: Columns to select (None = all).
            where: SQL WHERE clause (without 'WHERE' keyword).
            order_by: SQL ORDER BY clause.
            limit: Row limit.

        Returns:
            DataFrame with query results.
        """
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM {table_name}"

        if where:
            query += f" WHERE {where}"
        if order_by:
            query += f" ORDER BY {order_by}"
        if limit:
            query += f" LIMIT {limit}"

        return self.conn.execute(query).df()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        result = self.conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        return result[0] > 0

    def table_schema(self, table_name: str) -> pd.DataFrame:
        """Return the schema of a table."""
        return self.conn.execute(f"DESCRIBE {table_name}").df()

    def query(self, sql: str) -> pd.DataFrame:
        """Run an arbitrary SQL query and return results as DataFrame."""
        return self.conn.execute(sql).df()

    def close(self):
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
