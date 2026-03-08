from pathlib import Path
import duckdb

DATASET_PATH = 'hf://datasets/allenai/WildChat-4.8M/data/*.parquet'

query = f"""
SELECT *
FROM read_parquet('{DATASET_PATH}')
WHERE language IN ('English')
AND turn >5
AND CAST(timestamp AS TIMESTAMP) >= TIMESTAMP '2025-01-01'
LIMIT 200
"""

repo_root = Path(__file__).resolve().parent
output_dir = repo_root / "data" / "raw" / "wildchat"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "wildchat_latest_200_filtered.parquet"

con = duckdb.connect()
df = con.execute(query).df()
df.to_parquet(output_file, index=False)

print(f"Saved {len(df)} rows to {output_file}")