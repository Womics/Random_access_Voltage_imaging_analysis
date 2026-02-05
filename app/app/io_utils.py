from pathlib import Path
import pandas as pd

def robust_read_table(path: str) -> pd.DataFrame:
    """Read TSV/CSV-like table robustly. Requires at least 2 columns."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        seps = [",", "\t", None, r"\s+"]
    else:
        seps = ["\t", ",", None, r"\s+"]
    last = None
    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to read as 2+ column table: {path}\nLast error: {last}")
