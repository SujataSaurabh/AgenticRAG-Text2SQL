import sqlite3
# from google.colab import files
import pandas as pd

# 1. Upload the CSV file to your Colab session (if not already uploaded)
# uploaded = files.upload()

# 2. Read the CSV file into a pandas DataFrame
def process_file(filename):
    df = pd.read_csv(filename)

    # 3. Connect to SQLite database (creates 'database.db' if it doesn't exist)
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # Determine the table name based on the filename from the directory and csv pathname
    filename = filename.split('/')[-1]
    table_name = filename.split('.')[0]
    if table_name == "PUBSJOURNALS":
    # 4. Create the PUBSJOURNALS table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS PUBSJOURNALS (
            dbpubid INT,
            artchapthesttle TEXT,
            pubyear INT,
            refereed BOOL,
            beamline VARCHAR,
            journalcode INT,
            journaltitle TEXT,
            journalimpactfactor TEXT,
            doehighimpact BOOL,
            builtauthorlist TEXT,
            publisting TEXT
        );
        """
    elif table_name == "AUTHORS":
        create_table_query = """
        CREATE TABLE IF NOT EXISTS AUTHORS (
            alsid INT,
            lastname TEXT,
            firstname TEXT,
            institution TEXT,
            dbpubid INT
        );
        """
    # Clear tables before re-inserting to avoid duplicate data on re-run

    # Check if table exists before deleting
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    if cursor.fetchone():
        cursor.execute(f"DELETE FROM {table_name}")
        conn.commit()
    else:
        print(
            f"Table '{table_name}' does not exist yet. Skipping delete operation."
        )

    cursor.execute(create_table_query)
    conn.commit()

    # 5. Insert all rows from the DataFrame into the SQLite table
    df.to_sql(table_name, conn, if_exists="append", index=False)

    # 6. Verify the insertion by checking row count and previewing data
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]
    print(f"Successfully inserted {row_count:,} rows into {table_name}.")

    # Preview first 3 records
    preview_df = pd.read_sql_query(
        f"SELECT * FROM {table_name} LIMIT 3;", conn
    )
    print("\nData Preview:")
    print(preview_df)

    # Close connection
    conn.close()

if __name__ == "__main__":
    process_file("../data/PUBSJOURNALS.csv")
    process_file("../data/AUTHORS.csv")