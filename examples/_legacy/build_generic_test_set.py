"""Build a local generic GPS test set from anonymized and altered landed data.

The generated data is a local development artifact. It is derived from the
already anonymized and spatially altered landed tables, then pseudonymized again and shifted to a
generic `test_set` experiment in spring 2026.

The script intentionally writes to `Data/Output/test-set-gps`, which is ignored
by git through the repository data rules.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from shapely import wkb
from shapely.affinity import translate


PHASES = (
    ("Phase1", "2026-03-02", "2026-03-22"),
    ("Phase2", "2026-03-23", "2026-04-19"),
    ("Phase3", "2026-04-20", "2026-05-10"),
)

DATETIME_HINTS = (
    "_at",
    "_at_in_timezone",
)

DROP_USER_STAT_COLUMNS = {
    "registration_code",
    "match_column",
    "invite_code",
    "questionnaire_match_value",
    "email",
}


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root from the current working directory."""

    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "package-xyt-gps").exists() and (candidate / "Data").exists():
            return candidate
    raise FileNotFoundError("Impossible de trouver la racine du dépôt.")


def read_table(input_dir: Path, name: str) -> pd.DataFrame:
    """Read a landed table, preferring Parquet when available."""

    parquet_path = input_dir / f"{name}.parquet"
    csv_path = input_dir / f"{name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    return pd.DataFrame()


def write_table(output_dir: Path, name: str, table: pd.DataFrame, *, csv_only: bool = False) -> None:
    """Write one test-set table."""

    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / f"{name}.csv", index=False)
    if not csv_only:
        table.to_parquet(output_dir / f"{name}.parquet", index=False)


def select_test_users(storyline: pd.DataFrame, n_users: int) -> list[str]:
    """Select users with enough observed records for a compact tutorial set."""

    if storyline.empty or "user_id" not in storyline.columns:
        raise ValueError("storyline must contain user_id.")
    dates = pd.to_datetime(storyline["started_at"], errors="coerce", utc=True).dt.date
    stats = (
        storyline.assign(_date=dates)
        .dropna(subset=["user_id", "_date"])
        .groupby("user_id")
        .agg(rows=("user_id", "size"), active_days=("_date", "nunique"))
        .sort_values(["active_days", "rows"], ascending=False)
    )
    if len(stats) < n_users:
        raise ValueError(f"Seulement {len(stats)} utilisateurs disponibles, {n_users} demandés.")
    return stats.head(n_users).index.astype(str).tolist()


def remap_user_ids(table: pd.DataFrame, user_map: dict[str, str]) -> pd.DataFrame:
    """Filter and pseudonymize a table on `user_id`."""

    if table.empty or "user_id" not in table.columns:
        return table.copy()
    out = table.loc[table["user_id"].astype(str).isin(user_map)].copy()
    out["user_id"] = out["user_id"].astype(str).map(user_map)
    return out


def remap_entry_id(table: pd.DataFrame, id_column: str, prefix: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Replace row identifiers with deterministic generic ids."""

    out = table.copy()
    if id_column not in out.columns:
        return out, {}
    old_ids = out[id_column].dropna().astype(str).drop_duplicates().tolist()
    mapping = {old_id: f"{prefix}_{index:06d}" for index, old_id in enumerate(old_ids, start=1)}
    out[id_column] = out[id_column].astype(str).map(mapping).where(out[id_column].notna(), pd.NA)
    return out, mapping


def datetime_columns(table: pd.DataFrame) -> list[str]:
    """Return columns that look like datetimes in landed GPS exports."""

    columns = []
    for column in table.columns:
        if str(column).endswith("_timezone"):
            continue
        if column in {"started_on", "phase_start", "phase_end"}:
            columns.append(column)
        elif any(str(column).endswith(hint) for hint in DATETIME_HINTS):
            columns.append(column)
    return columns


def user_time_offsets(storyline: pd.DataFrame, phase_start: str) -> dict[str, pd.Timedelta]:
    """Build per-user offsets so each user starts in spring 2026."""

    target_start = pd.Timestamp(phase_start, tz="UTC")
    offsets: dict[str, pd.Timedelta] = {}
    for index, (user_id, group) in enumerate(storyline.groupby("user_id"), start=0):
        first = pd.to_datetime(group["started_at"], errors="coerce", utc=True).min()
        if pd.isna(first):
            continue
        offsets[str(user_id)] = target_start + pd.Timedelta(days=index % 5) - first
    return offsets


def shift_datetimes(table: pd.DataFrame, offsets: dict[str, pd.Timedelta]) -> pd.DataFrame:
    """Shift all datetime-like columns with the user's time offset."""

    if table.empty or "user_id" not in table.columns:
        return table.copy()
    out = table.copy()
    mapped_offsets = out["user_id"].astype(str).map(offsets)
    for column in datetime_columns(out):
        values = pd.to_datetime(out[column], errors="coerce", utc=True)
        shifted = values + mapped_offsets
        if column == "started_on":
            out[column] = shifted.dt.date.astype("string")
        elif column in {"phase_start", "phase_end"}:
            continue
        else:
            out[column] = shifted.dt.strftime("%Y-%m-%dT%H:%M:%SZ").where(shifted.notna(), pd.NA)
    return out


def assign_test_phases(table: pd.DataFrame) -> pd.DataFrame:
    """Assign the generic spring 2026 phases to rows with `started_at`."""

    out = table.copy()
    if "started_at" not in out.columns:
        return out
    started = pd.to_datetime(out["started_at"], errors="coerce", utc=True)
    phase_name = pd.Series(pd.NA, index=out.index, dtype="string")
    phase_number = pd.Series(pd.NA, index=out.index, dtype="Int64")
    phase_start = pd.Series(pd.NA, index=out.index, dtype="string")
    phase_end = pd.Series(pd.NA, index=out.index, dtype="string")
    for index, (name, start, end) in enumerate(PHASES, start=1):
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask = started.ge(start_ts) & started.le(end_ts)
        phase_name.loc[mask] = name
        phase_number.loc[mask] = index
        phase_start.loc[mask] = start
        phase_end.loc[mask] = end
    out["phase"] = phase_name
    out["phase_number"] = phase_number
    out["phase_start"] = phase_start
    out["phase_end"] = phase_end
    return out


def keep_phase_window(table: pd.DataFrame) -> pd.DataFrame:
    """Keep rows that fall inside the generic phase window."""

    if table.empty or "started_at" not in table.columns:
        return table.copy()
    started = pd.to_datetime(table["started_at"], errors="coerce", utc=True)
    start = pd.Timestamp(PHASES[0][1], tz="UTC")
    end = pd.Timestamp(PHASES[-1][2], tz="UTC") + pd.Timedelta(days=1)
    return table.loc[started.ge(start) & started.lt(end)].copy()


def geometry_to_shape(value):
    """Parse WKB/EWKB hex geometry values when possible."""

    if pd.isna(value):
        return None
    if hasattr(value, "geom_type"):
        return value
    try:
        return wkb.loads(str(value), hex=True)
    except Exception:
        return None


def translate_geometry_hex(value, dx_m: float, dy_m: float, latitude: float = 46.5):
    """Translate one geometry by approximate meters and return WKB hex."""

    geom = geometry_to_shape(value)
    if geom is None:
        return value
    meters_per_degree_lon = 111_320 * math.cos(math.radians(latitude))
    dx = dx_m / meters_per_degree_lon
    dy = dy_m / 111_320
    return translate(geom, xoff=dx, yoff=dy).wkb_hex


def spatial_offsets(user_ids: Iterable[str]) -> dict[str, tuple[float, float]]:
    """Build deterministic extra offsets for already anonymized geometries."""

    offsets = {}
    for index, user_id in enumerate(sorted(user_ids), start=1):
        angle = index * 2.399963229728653
        radius = 250 + (index % 5) * 90
        offsets[user_id] = (math.cos(angle) * radius, math.sin(angle) * radius)
    return offsets


def shift_geometries(storyline: pd.DataFrame) -> pd.DataFrame:
    """Apply a second deterministic spatial offset to geometries."""

    if storyline.empty or "geometry" not in storyline.columns:
        return storyline.copy()
    out = storyline.copy()
    offsets = spatial_offsets(out["user_id"].dropna().astype(str).unique())
    out["geometry"] = [
        translate_geometry_hex(value, *offsets.get(str(user_id), (0.0, 0.0)))
        for value, user_id in zip(out["geometry"], out["user_id"], strict=False)
    ]
    return out


def build_user_statistics(storyline: pd.DataFrame, source_user_statistics: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal user-statistics table for the test set."""

    dates = pd.to_datetime(storyline["started_at"], errors="coerce", utc=True)
    stats = []
    for user_id, group in storyline.assign(_started_at=dates).groupby("user_id"):
        active_dates = group["_started_at"].dt.date.dropna().drop_duplicates().sort_values()
        if active_dates.empty:
            first = latest = pd.NaT
            days_in_range = active_days = inactive_days = 0
        else:
            first = pd.Timestamp(active_dates.iloc[0], tz="UTC")
            latest = pd.Timestamp(active_dates.iloc[-1], tz="UTC")
            days_in_range = int((latest - first).days + 1)
            active_days = int(len(active_dates))
            inactive_days = days_in_range - active_days
        stats.append(
            {
                "user_statistics_id": user_id,
                "user_id": user_id,
                "experiment_name": "test_set",
                "motiontag_project_name": "test_set",
                "first_tracked_at": first.isoformat() if pd.notna(first) else pd.NA,
                "latest_tracked_at": latest.isoformat() if pd.notna(latest) else pd.NA,
                "days_in_range_count": days_in_range,
                "active_days_count": active_days,
                "inactive_days_count": inactive_days,
            }
        )
    out = pd.DataFrame(stats)
    optional = source_user_statistics.drop(columns=[column for column in DROP_USER_STAT_COLUMNS if column in source_user_statistics.columns], errors="ignore")
    keep_cols = [column for column in ["user_id", "operating_system", "os_version", "handset"] if column in optional.columns]
    if keep_cols:
        out = out.merge(optional[keep_cols].drop_duplicates("user_id"), on="user_id", how="left")
    return out


def build_user_presence(user_ids: Iterable[str]) -> pd.DataFrame:
    """Build the generic phase table used by tutorials."""

    rows = []
    for user_id in sorted(user_ids):
        row = {
            "experiment_name": "test_set",
            "user_id": user_id,
            "project_raw": "test_set",
            "in_storyline": True,
            "in_trips": True,
            "in_user_statistics": True,
            "in_questionnaire": False,
        }
        for index, (_, start, end) in enumerate(PHASES, start=1):
            row[f"phase_{index}_start"] = start
            row[f"phase_{index}_end"] = end
        rows.append(row)
    return pd.DataFrame(rows)


def write_landing_config(output_dir: Path) -> None:
    """Write the minimal landing config expected by data-loading notebooks."""

    config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "experiments": [
            {
                "experiment_name": "test_set",
                "storyline_name": "test_set",
                "motiontag_project_name": "test_set",
                "dump_date_range": f"{PHASES[0][1]}--{PHASES[-1][2]}",
                "phase_1": [PHASES[0][1], PHASES[0][2]],
                "phase_2": [PHASES[1][1], PHASES[1][2]],
                "phase_3": [PHASES[2][1], PHASES[2][2]],
            }
        ],
    }
    (output_dir / "landing_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def build_test_set(input_dir: Path, output_dir: Path, n_users: int = 15) -> dict:
    """Build the generic local test set."""

    storyline = read_table(input_dir, "storyline")
    trips = read_table(input_dir, "trips")
    journeys = read_table(input_dir, "journeys")
    user_statistics = read_table(input_dir, "user_statistics")
    public_transport_legs = read_table(input_dir, "public_transport_legs")

    selected_users = select_test_users(storyline, n_users)
    user_map = {old_user_id: f"test_user_{index:03d}" for index, old_user_id in enumerate(selected_users, start=1)}

    storyline = remap_user_ids(storyline, user_map)
    trips = remap_user_ids(trips, user_map)
    journeys = remap_user_ids(journeys, user_map)
    user_statistics = remap_user_ids(user_statistics, user_map)
    public_transport_legs = remap_user_ids(public_transport_legs, user_map)

    offsets = user_time_offsets(storyline, PHASES[0][1])
    tables = {
        "storyline": storyline,
        "trips": trips,
        "journeys": journeys,
        "public_transport_legs": public_transport_legs,
    }
    tables = {name: keep_phase_window(assign_test_phases(shift_datetimes(table, offsets))) for name, table in tables.items()}
    tables["storyline"] = shift_geometries(tables["storyline"])

    tables["storyline"], _ = remap_entry_id(tables["storyline"], "storyline_id", "test_storyline")
    tables["trips"], trip_id_map = remap_entry_id(tables["trips"], "trip_id", "test_trip")
    tables["journeys"], journey_id_map = remap_entry_id(tables["journeys"], "journey_id", "test_journey")
    if "trip_id" in tables["storyline"].columns:
        tables["storyline"]["trip_id"] = tables["storyline"]["trip_id"].astype(str).map(trip_id_map).where(tables["storyline"]["trip_id"].notna(), pd.NA)
    if "journey_id" in tables["trips"].columns:
        tables["trips"]["journey_id"] = tables["trips"]["journey_id"].astype(str).map(journey_id_map).where(tables["trips"]["journey_id"].notna(), pd.NA)

    for table in tables.values():
        if table.empty:
            continue
        table["experiment_name"] = "test_set"
        table["motiontag_project_name"] = "test_set"
        table["source_id"] = "test_set"
        table.drop(columns=["match_column", "registration_code", "invite_code", "questionnaire_match_value"], inplace=True, errors="ignore")

    user_stats = build_user_statistics(tables["storyline"], user_statistics)
    user_presence = build_user_presence(tables["storyline"]["user_id"].dropna().astype(str).unique())

    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(output_dir, "storyline", tables["storyline"])
    write_table(output_dir, "trips", tables["trips"])
    write_table(output_dir, "journeys", tables["journeys"])
    if not tables["public_transport_legs"].empty:
        write_table(output_dir, "public_transport_legs", tables["public_transport_legs"])
    write_table(output_dir, "user_presence", user_presence)
    write_table(output_dir, "user_statistics", user_stats, csv_only=True)
    write_landing_config(output_dir)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(input_dir),
        "output_dir": str(output_dir),
        "experiment_name": "test_set",
        "n_users_requested": n_users,
        "n_users_written": int(tables["storyline"]["user_id"].nunique()),
        "phases": [{"name": name, "start": start, "end": end} for name, start, end in PHASES],
        "privacy_transformations": [
            "source must already be an anonymized_altered landed directory",
            "second deterministic user_id pseudonymization",
            "generic row identifiers",
            "dates shifted to spring 2026",
            "second deterministic geometry translation by user",
        ],
        "tables": {
            name: {"rows": int(len(table)), "users": int(table["user_id"].nunique()) if "user_id" in table.columns and not table.empty else None}
            for name, table in {**tables, "user_statistics": user_stats, "user_presence": user_presence}.items()
        },
    }
    (output_dir / "test_set_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Generic GPS test set\n\n"
        f"Jeu de test local généré depuis `{input_dir}`.\n\n"
        "Il contient 15 utilisateurs pseudonymisés, une expérimentation générique `test_set` "
        "et trois phases au printemps 2026. Les identifiants sont remappés, les dates sont "
        "recalées et les géométries reçoivent une translation déterministe supplémentaire. "
        "Les traces produites servent aux tests techniques et ne doivent pas être "
        "interprétées scientifiquement comme trajectoires réelles.\n\n"
        "Ce dossier est un artefact local ignoré par git.\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    repo = find_repo_root()
    default_anonymized_altered = (
        repo / "Data" / "Output" / "0-landed-data" / "declic-mobility-ziplo" / "anonymized_altered"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_anonymized_altered,
        help="Anonymized and spatially altered landed data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "Data" / "Output" / "test-set-gps",
        help="Output directory for the local generic test set.",
    )
    parser.add_argument("--users", type=int, default=15, help="Number of users to keep.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_test_set(args.input_dir.resolve(), args.output_dir.resolve(), n_users=args.users)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
