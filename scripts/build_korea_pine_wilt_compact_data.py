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


def _read_points(raw_csvs: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]], dict[str, int]]:
    xs: list[float] = []
    ys: list[float] = []
    years: list[int] = []
    raw_files: list[dict[str, object]] = []
    year_counts: dict[str, int] = {}

    for path in raw_csvs:
        year = _year_from_name(path)
        count = 0
        with path.open("r", encoding="cp949", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if len(header) < 3:
                raise ValueError(f"Unexpected CSV header in {path}")
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    x = float(row[1])
                    y = float(row[2])
                except ValueError:
                    continue
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                xs.append(x)
                ys.append(y)
                years.append(year)
                count += 1

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
        raw_files,
        year_counts,
    )


def build(raw_dir: Path, output_dir: Path) -> dict[str, object]:
    raw_csvs = sorted(raw_dir.rglob("*.csv"))
    if not raw_csvs:
        raise FileNotFoundError(f"No CSV files found below {raw_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, year, raw_files, year_counts = _read_points(raw_csvs)
    npz_path = output_dir / "infected_points_2016_2023.npz"
    csv_gz_path = output_dir / "infected_points_2016_2023.csv.gz"
    np.savez_compressed(npz_path, x=x, y=y, year=year)
    with gzip.open(csv_gz_path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_5179", "y_5179", "year"])
        writer.writerows(zip(x, y, year))

    manifest = {
        "dataset": "Korea Forest Service pine-wilt infected-tree observations",
        "source": {
            "description": "Annual Korea Forest Service pest occurrence CSV files.",
            "source_root_hint": str(raw_dir),
            "encoding": "cp949",
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
    args = parser.parse_args()

    manifest = build(args.raw_dir, args.output_dir)
    print(json.dumps(manifest["compact_files"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
