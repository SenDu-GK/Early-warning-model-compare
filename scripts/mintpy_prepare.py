#!/usr/bin/env python
"""
Build a MintPy-friendly time-series and geometry/DEM HDF5 from the provided Cadia CSV + ENVI DEM.

Outputs:
- mintpy_outputs/timeseries_cadia_des.h5   (time-series grid with DATE list + metadata)
- mintpy_outputs/geometryGeo.h5            (latitude/longitude/height on the same grid)
- mintpy_outputs/demGeo.h5                 (DEM only, for tools that expect a separate DEM file)

This script grids the point-based CSV onto the DEM resolution (1 arc-second) by snapping to the
nearest DEM pixel. Pixels without data are left as NaN.
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np


@dataclass
class EnviHeader:
    samples: int
    lines: int
    data_type: int
    byte_order: int
    interleave: str
    lon0: float
    lat0: float
    x_step: float
    y_step: float


def parse_envi_header(path: str) -> EnviHeader:
    """Parse minimal ENVI header fields needed for geolocation."""
    samples = lines = data_type = byte_order = None
    interleave = None
    lon0 = lat0 = x_step = y_step = None

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if line.lower().startswith("samples"):
                samples = int(line.split("=")[1])
            elif line.lower().startswith("lines"):
                lines = int(line.split("=")[1])
            elif line.lower().startswith("data type"):
                data_type = int(line.split("=")[1])
            elif line.lower().startswith("byte order"):
                byte_order = int(line.split("=")[1])
            elif line.lower().startswith("interleave"):
                interleave = line.split("=")[1].strip().lower()
            elif line.lower().startswith("map info"):
                content = line.split("{", 1)[1].rsplit("}", 1)[0]
                parts = [p.strip() for p in content.split(",")]
                lon0 = float(parts[3])
                lat0 = float(parts[4])
                x_step = float(parts[5])
                y_step = float(parts[6])

    missing = [name for name, val in [
        ("samples", samples), ("lines", lines), ("data_type", data_type),
        ("byte_order", byte_order), ("interleave", interleave),
        ("lon0", lon0), ("lat0", lat0), ("x_step", x_step), ("y_step", y_step),
    ] if val is None]
    if missing:
        raise ValueError(f"Missing fields in ENVI header: {missing}")

    return EnviHeader(
        samples=samples,
        lines=lines,
        data_type=data_type,
        byte_order=byte_order,
        interleave=interleave,
        lon0=lon0,
        lat0=lat0,
        x_step=x_step,
        y_step=y_step,
    )


def read_dem_subset(dem_path: str, hdr: EnviHeader,
                    row_start: int, row_end: int, col_start: int, col_end: int) -> np.ndarray:
    """Read a subset of the ENVI DEM into float32."""
    dtype_map = {
        2: np.dtype(np.int16),
        3: np.dtype(np.int32),
        4: np.dtype(np.float32),
        5: np.dtype(np.float64),
    }
    if hdr.data_type not in dtype_map:
        raise ValueError(f"Unsupported ENVI data type: {hdr.data_type}")
    dtype = dtype_map[hdr.data_type]
    if hdr.byte_order == 0:
        dtype = dtype.newbyteorder("<")
    else:
        dtype = dtype.newbyteorder(">")

    dem = np.fromfile(dem_path, dtype=dtype, count=hdr.samples * hdr.lines)
    dem = dem.reshape((hdr.lines, hdr.samples))
    subset = dem[row_start:row_end, col_start:col_end].astype(np.float32)
    return subset


def compute_grid_bounds(lon: Sequence[float], lat: Sequence[float], hdr: EnviHeader,
                        pad_pixels: int = 1) -> Tuple[int, int, int, int, float, float, int, int]:
    """
    Snap the point cloud bounding box to the DEM grid.

    Returns (row_start, row_end, col_start, col_end, start_lon, start_lat, height, width).
    row_end/col_end are exclusive.
    """
    lon_min, lon_max = min(lon), max(lon)
    lat_min, lat_max = min(lat), max(lat)
    x_step = hdr.x_step
    y_step = abs(hdr.y_step)
    lon0, lat0 = hdr.lon0, hdr.lat0

    col_start = math.floor((lon_min - lon0) / x_step) - pad_pixels
    col_end = math.ceil((lon_max - lon0) / x_step) + pad_pixels
    row_start = math.floor((lat0 - lat_max) / y_step) - pad_pixels
    row_end = math.ceil((lat0 - lat_min) / y_step) + pad_pixels

    col_start = max(0, col_start)
    row_start = max(0, row_start)
    col_end = min(hdr.samples, col_end)
    row_end = min(hdr.lines, row_end)

    width = col_end - col_start
    height = row_end - row_start
    start_lon = lon0 + col_start * x_step
    start_lat = lat0 - row_start * y_step
    return row_start, row_end, col_start, col_end, start_lon, start_lat, height, width


def load_csv_points(path: str) -> Tuple[List[str], List[float], List[float], List[List[float]]]:
    """Load Cadia CSV and return (dates, lons, lats, values_per_point)."""
    dates: List[str] = []
    lons: List[float] = []
    lats: List[float] = []
    values: List[List[float]] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        date_cols = [c for c in header if c.startswith("D") and c[1:].isdigit()]
        dates = [c[1:] for c in date_cols]

        for row in reader:
            lats.append(float(row["Fi"]))
            lons.append(float(row["Lambda"]))
            row_vals = []
            for col in date_cols:
                val_str = row[col].strip()
                if val_str in ("", "nan", "NaN"):
                    row_vals.append(np.nan)
                else:
                    row_vals.append(float(val_str))
            values.append(row_vals)

    return dates, lons, lats, values


def grid_timeseries(dates: List[str], lons: List[float], lats: List[float],
                    values: List[List[float]], start_lon: float, start_lat: float,
                    x_step: float, y_step: float, width: int, height: int) -> np.ndarray:
    """Grid scattered points to a 3D array [time, y, x] using nearest DEM pixel."""
    num_dates = len(dates)
    sum_data = np.full((num_dates, height, width), 0.0, dtype=np.float32)
    counts = np.zeros((num_dates, height, width), dtype=np.int32)

    for lon, lat, row_vals in zip(lons, lats, values):
        col = int(round((lon - start_lon) / x_step))
        row = int(round((start_lat - lat) / y_step))
        if row < 0 or row >= height or col < 0 or col >= width:
            continue
        for t in range(num_dates):
            val = row_vals[t]
            if math.isnan(val):
                continue
            sum_data[t, row, col] += val
            counts[t, row, col] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        timeseries = np.where(
            counts > 0,
            sum_data / counts,
            np.nan,
        )
    return timeseries


def write_timeseries(path: str, dates: List[str], data: np.ndarray,
                     start_lon: float, start_lat: float, x_step: float, y_step: float):
    """Write MintPy-like time-series file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    date_bytes = np.array(dates, dtype="S8")
    data32 = data.astype(np.float32)

    with h5py.File(path, "w") as f:
        dset = f.create_dataset(
            "timeseries",
            data=data32,
            compression="gzip",
            chunks=(1, data32.shape[1], data32.shape[2]),
        )
        dset.attrs["UNIT"] = "mm"
        dset.attrs["FILE_TYPE"] = "timeseries"
        f.create_dataset("date", data=date_bytes)
        f.attrs["X_FIRST"] = start_lon
        f.attrs["Y_FIRST"] = start_lat
        f.attrs["X_STEP"] = x_step
        f.attrs["Y_STEP"] = -abs(y_step)
        f.attrs["WIDTH"] = data.shape[2]
        f.attrs["LENGTH"] = data.shape[1]
        f.attrs["REF_DATE"] = dates[0]


def write_geometry(path: str, dem: np.ndarray, start_lon: float, start_lat: float,
                   x_step: float, y_step: float):
    """Write geometry file with latitude/longitude/height."""
    height, width = dem.shape
    lons = start_lon + np.arange(width) * x_step
    lats = start_lat - np.arange(height) * y_step
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("height", data=dem, compression="gzip")
        f.create_dataset("latitude", data=lat_grid.astype(np.float32), compression="gzip")
        f.create_dataset("longitude", data=lon_grid.astype(np.float32), compression="gzip")
        f.attrs["FILE_TYPE"] = "geometry"
        f.attrs["X_FIRST"] = start_lon
        f.attrs["Y_FIRST"] = start_lat
        f.attrs["X_STEP"] = x_step
        f.attrs["Y_STEP"] = -abs(y_step)
        f.attrs["WIDTH"] = width
        f.attrs["LENGTH"] = height


def write_dem_only(path: str, dem: np.ndarray, start_lon: float, start_lat: float,
                   x_step: float, y_step: float):
    """Write DEM-only HDF5 for tools expecting demGeo.h5."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("height", data=dem, compression="gzip")
        f.attrs["FILE_TYPE"] = "dem"
        f.attrs["X_FIRST"] = start_lon
        f.attrs["Y_FIRST"] = start_lat
        f.attrs["X_STEP"] = x_step
        f.attrs["Y_STEP"] = -abs(y_step)
        f.attrs["WIDTH"] = dem.shape[1]
        f.attrs["LENGTH"] = dem.shape[0]


def main(args: argparse.Namespace) -> None:
    hdr = parse_envi_header(args.hdr)
    dates, lons, lats, values = load_csv_points(args.csv)

    row_start, row_end, col_start, col_end, start_lon, start_lat, height, width = compute_grid_bounds(
        lons, lats, hdr, pad_pixels=args.pad)

    dem_subset = read_dem_subset(args.dem, hdr, row_start, row_end, col_start, col_end)
    timeseries = grid_timeseries(dates, lons, lats, values, start_lon, start_lat,
                                 hdr.x_step, abs(hdr.y_step), width, height)

    out_dir = args.out_dir
    ts_path = os.path.join(out_dir, "timeseries_cadia_des.h5")
    geom_path = os.path.join(out_dir, "geometryGeo.h5")
    dem_path = os.path.join(out_dir, "demGeo.h5")

    write_timeseries(ts_path, dates, timeseries, start_lon, start_lat, hdr.x_step, abs(hdr.y_step))
    write_geometry(geom_path, dem_subset, start_lon, start_lat, hdr.x_step, abs(hdr.y_step))
    write_dem_only(dem_path, dem_subset, start_lon, start_lat, hdr.x_step, abs(hdr.y_step))

    print(f"Saved time-series: {ts_path}")
    print(f"Saved geometry   : {geom_path}")
    print(f"Saved DEM        : {dem_path}")
    print(f"Grid size (LxW)  : {height} x {width}")
    print(f"Dates            : {len(dates)} acquisitions from {dates[0]} to {dates[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Cadia CSV + ENVI DEM to MintPy-friendly HDF5 files.")
    parser.add_argument("--csv", default="Cadia_PSI.csv", help="Input Cadia CSV with DYYYYMMDD columns.")
    parser.add_argument("--dem", default=os.path.join("DTM", "DEM"), help="ENVI DEM binary file.")
    parser.add_argument("--hdr", default=os.path.join("DTM", "DEM.HDR"), help="ENVI header file for the DEM.")
    parser.add_argument("--out-dir", default=os.path.join("mintpy_outputs"), help="Directory for output HDF5 files.")
    parser.add_argument("--pad", type=int, default=1, help="Extra pixels of padding around the data bounding box.")
    args = parser.parse_args()
    main(args)
