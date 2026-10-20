import argparse
import os
import sys

import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Impor dataset RetailRocket (events.csv) ke database Neo4J."
    )
    parser.add_argument(
        "-u", "--user", default="neo4j", help="Username Neo4j (default: neo4j)"
    )
    parser.add_argument(
        "-p", "--password", default="neo4j", help="Password Neo4J (default: neo4j)"
    )
    parser.add_argument(
        "-b",
        "--bolt",
        default="neo4j://127.0.0.1:7687",
        help="Neo4J URI / Bolt (default: neo4j://127.0.0.1:7687)",
    )
    parser.add_argument(
        "-s",
        "--source",
        default="data",
        help="Direktori/folder sumber data (default: data)",
    )
    parser.add_argument(
        "-d", "--database", default="neo4j", help="Nama database (default: neo4j)"
    )
    parser.add_argument(
        "--start", default=None, help="Filter awal (termasuk), contoh: 2015-06-01"
    )
    parser.add_argument(
        "--end", default=None, help="Filter akhir (tidak termasuk), contoh: 2015-07-01"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Batasi jumlah baris (untuk uji coba)"
    )
    return parser.parse_args()


def create_constraints(session):
    session.run(
        "CREATE CONSTRAINT visitor_id IF NOT EXISTS "
        "FOR (v:Visitor) REQUIRE v.visitorId IS UNIQUE"
    )
    session.run(
        "CREATE CONSTRAINT item_id IF NOT EXISTS "
        "FOR (i:Item) REQUIRE i.itemId IS UNIQUE"
    )


# Tipe relasi tidak bisa berupa parameter, jadi dibuat 3 query terpisah
QUERIES = {
    "view": """
        UNWIND $batch AS row
        MERGE (v:Visitor {visitorId: row.visitorid})
        MERGE (i:Item {itemId: row.itemid})
        CREATE (v)-[:VIEWED {timestamp: row.timestamp}]->(i)
    """,
    "addtocart": """
        UNWIND $batch AS row
        MERGE (v:Visitor {visitorId: row.visitorid})
        MERGE (i:Item {itemId: row.itemid})
        CREATE (v)-[:ADDED_TO_CART {timestamp: row.timestamp}]->(i)
    """,
    "transaction": """
        UNWIND $batch AS row
        MERGE (v:Visitor {visitorId: row.visitorid})
        MERGE (i:Item {itemId: row.itemid})
        CREATE (v)-[:BOUGHT {timestamp: row.timestamp,
                             transactionId: row.transactionid}]->(i)
    """,
}


def import_batch(tx, records):
    for event, query in QUERIES.items():
        rows = [r for r in records if r["event"] == event]
        if rows:
            tx.run(query, batch=rows)


def main():
    args = parse_arguments()

    # 1. Periksa keberadaan events.csv
    csv_file_path = os.path.join(args.source, "events.csv")
    if not os.path.exists(csv_file_path):
        print(f"Error: File '{csv_file_path}' tidak ditemukan!")
        sys.exit(1)

    # 2. Buat koneksi
    try:
        driver = GraphDatabase.driver(args.bolt, auth=(args.user, args.password))
        driver.verify_connectivity()
    except Exception as e:
        print(f"Gagal terhubung ke database: {e}")
        sys.exit(1)

    # 3. Siapkan parameter batch
    batch_size = 5000

    with open(csv_file_path, "rb") as f:
        total_rows = sum(1 for _ in f) - 1  # dikurangi 1 baris header
    if args.limit:
        total_rows = min(total_rows, args.limit)
    total_chunks = (total_rows // batch_size) + 1

    # 4. Impor per batch
    with driver.session(database=args.database) as session:
        create_constraints(session)

        print(f"Membaca file data dari: {csv_file_path}")
        chunks = pd.read_csv(csv_file_path, chunksize=batch_size, nrows=args.limit)

        print("Memulai proses impor data ke Neo4j...")
        for chunk in tqdm(chunks, total=total_chunks, desc="Mengimpor Batch Data"):
            # Filter periode waktu (timestamp berupa milidetik)
            if args.start or args.end:
                waktu = pd.to_datetime(chunk["timestamp"], unit="ms")
                mask = pd.Series(True, index=chunk.index)
                if args.start:
                    mask &= waktu >= pd.Timestamp(args.start)
                if args.end:
                    mask &= waktu < pd.Timestamp(args.end)
                chunk = chunk[mask].copy()
            if chunk.empty:
                continue

            # Bersihkan transactionid: kosong -> "", angka -> string bulat (bukan 123.0)
            chunk["transactionid"] = (
                chunk["transactionid"].astype("Int64").astype(str).replace("<NA>", "")
            )

            records = chunk.to_dict(orient="records")
            session.execute_write(import_batch, records)

    driver.close()
    print("Proses impor berhasil.")


if __name__ == "__main__":
    main()