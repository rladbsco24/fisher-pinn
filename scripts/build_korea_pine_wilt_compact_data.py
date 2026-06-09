from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


YEAR_PATTERN = re.compile(r"(20\d{2})")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _year_from_name(path: Path) -> int:
    match = YEAR_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not infer year from {path}")
    return int(match.group(1))


def _transform_xy(x: np.ndarray, y: np.ndarray, *, raw_crs: str, target_crs: str = "EPSG:5179") -> tuple[np.ndarray, np.ndarray]:
    if raw_crs == target_crs:
        return x, y
    from pyproj import Transformer

    transformer = Transformer.from_crs(raw_crs, target_crs, always_xy=True)
    tx, ty = transformer.transform(x, y)
    return np.asarray(tx, dtype=np.float64), np.asarray(ty, dtype=np.float64)


def _read_points(
    raw_csvs: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[dict[str, object]], dict[str, int], dict[str, int]]:
    xs: list[float] = []
    ys: list[float] = []
    years: list[int] = []
    date_ordinals: list[int] = []
    completed: list[int] = []
    month_keys: list[str] = []
    raw_files: list[dict[str, object]] = []
    year_counts: dict[str, int] = {}
    month_counts: dict[str, int] = {}
    infected_label = "\uac10\uc5fc\ubaa9"
    complete_label = "\uc644\ub8cc"

    for path in raw_csvs:
        year = _year_from_name(path)
        count = 0
        with path.open("r", encoding="cp949", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if len(header) < 3:
                raise ValueError(f"Unexpected CSV header in {path}")
            for row in reader:
                if len(row) < 10:
                    continue
                if len(row) > 8 and row[8].strip() != infected_label:
                    continue
                try:
                    x = float(row[1])
                    y = float(row[2])
                except ValueError:
                    continue
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                date_text = row[9].strip().strip('"') if len(row) > 9 else f"{year}-01-01"
                try:
                    year_value, month_value, day_value = (int(part) for part in date_text.split("-"))
                except ValueError:
                    year_value, month_value, day_value = year, 1, 1
                xs.append(x)
                ys.append(y)
                years.append(year_value)
                date_ordinals.append(np.datetime64(f"{year_value:04d}-{month_value:02d}-{day_value:02d}", "D").astype(int))
                completed.append(1 if len(row) > 10 and row[10].strip() == complete_label else 0)
                month_key = f"{year_value:04d}-{month_value:02d}"
                month_keys.append(month_key)
                count += 1
                month_counts[month_key] = month_counts.get(month_key, 0) + 1

        year_counts[str(year)] = count
        raw_files.append(
            {
                "year": year,
                "file_name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(years, dtype=np.int16),
        np.asarray(date_ordinals, dtype=np.int32),
        np.asarray(completed, dtype=np.int8),
        month_keys,
        raw_files,
        year_counts,
        month_counts,
    )


def build(raw_dir: Path, output_dir: Path, *, raw_crs: str = "EPSG:5181") -> dict[str, object]:
    raw_csvs = sorted(raw_dir.rglob("*.csv"))
    if not raw_csvs:
        raise FileNotFoundError(f"No CSV files found below {raw_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, year, date_ordinal, completed, month_keys, raw_files, year_counts, month_counts = _read_points(raw_csvs)
    x, y = _transform_xy(x, y, raw_crs=raw_crs)
    npz_path = output_dir / "infected_points_2016_2023.npz"
    csv_gz_path = output_dir / "infected_points_2016_2023.csv.gz"
    np.savez_compressed(npz_path, x=x, y=y, year=year)
    with gzip.open(csv_gz_path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_5179", "y_5179", "year"])
        writer.writerows(zip(x, y, year))
    monthly_npz_path = output_dir / "infected_points_2016_2023_monthly.npz"
    monthly_csv_gz_path = output_dir / "infected_points_2016_2023_monthly.csv.gz"
    months = sorted(month_counts)
    month_to_idx = {month: idx for idx, month in enumerate(months)}
    month_index = np.asarray([month_to_idx[month] for month in month_keys], dtype=np.int16)
    np.savez_compressed(
        monthly_npz_path,
        x=x,
        y=y,
        year=year,
        date_ordinal=date_ordinal,
        completed=completed,
        month_index=month_index,
        month_labels=np.asarray(months),
    )
    with gzip.open(monthly_csv_gz_path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_5179", "y_5179", "year", "date_ordinal", "completed", "month_index"])
        writer.writerows(zip(x, y, year, date_ordinal, completed, month_index))

    manifest = {
        "dataset": "Korea Forest Service pine-wilt infected-tree observations",
        "source": {
            "description": "Annual Korea Forest Service pest occurrence CSV files.",
            "source_root_hint": str(raw_dir),
            "encoding": "cp949",
            "raw_crs": raw_crs,
            "target_crs": "EPSG:5179",
            "raw_csv_policy": (
                "Several annual source CSV files exceed GitHub's normal 100 MB single-blob limit. "
                "This repository commits compact derived infected-point CSV/NPZ files plus raw-file checksums."
            ),
        },
        "compact_files": {
            "csv_gzip": "processed/infected_points_2016_2023.csv.gz",
            "npz": "processed/infected_points_2016_2023.npz",
            "columns": ["x_5179", "y_5179", "year"],
            "crs": "EPSG:5179",
            "records": int(len(year)),
            "year_counts": year_counts,
            "monthly_csv_gzip": "processed/infected_points_2016_2023_monthly.csv.gz",
            "monthly_npz": "processed/infected_points_2016_2023_monthly.npz",
            "monthly_columns": ["x_5179", "y_5179", "year", "date_ordinal", "completed", "month_index"],
            "month_counts": month_counts,
        },
        "raw_files": raw_files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact Korea pine-wilt point data from annual CSV files.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "korea_pine_wilt" / "processed")
    parser.add_argument("--raw-crs", default="EPSG:5181")
    args = parser.parse_args()

    manifest = build(args.raw_dir, args.output_dir, raw_crs=args.raw_crs)
    print(json.dumps(manifest["compact_files"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
