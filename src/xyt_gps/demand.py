"""Daily demand profiles derived from mobility legs."""

from __future__ import annotations

import pandas as pd

from .columns import phase_column

DEMAND_PROFILE_BIN_MINUTES = 5
DEMAND_HEATMAP_BIN_MINUTES = 60
DEMAND_PROFILE_ALL_MODES_LABEL = "Tous modes"
WEEKDAY_LABELS = {
    0: "Lundi",
    1: "Mardi",
    2: "Mercredi",
    3: "Jeudi",
    4: "Vendredi",
    5: "Samedi",
    6: "Dimanche",
}
DEFAULT_MOTIF_COLUMNS = (
    "mobility_motif",
    "motif",
    "purpose_mrmt",
    "purpose_niv1",
    "purpose",
    "main_journey_purpose_mrmt",
    "detected_purpose",
)


def _phase_series(df: pd.DataFrame, *, phase_col: str, default_phase_name: str) -> pd.Series:
    if phase_col in df.columns:
        return df[phase_col].fillna(default_phase_name).astype(str)
    column = phase_column(df)
    if column is not None:
        return df[column].fillna(default_phase_name).astype(str)
    return pd.Series(default_phase_name, index=df.index, dtype="object")


def _local_timestamp(value) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT


def _local_datetime_column(df: pd.DataFrame, preferred_col: str, fallback_col: str) -> pd.Series:
    column = preferred_col if preferred_col in df.columns else fallback_col
    if column not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    return df[column].apply(_local_timestamp)


def _leg_demand_bins(start: pd.Timestamp, finish: pd.Timestamp, *, bin_minutes: int) -> list[tuple[object, int]]:
    if pd.isna(start) or pd.isna(finish):
        return []

    start_minute = int(start.hour) * 60 + int(start.minute)
    finish_minute = int(finish.hour) * 60 + int(finish.minute)
    start_date = start.date()
    finish_date = finish.date()
    if finish_date < start_date:
        return []

    ranges: list[tuple[object, int, int]] = []
    if finish_date == start_date:
        if finish_minute < start_minute:
            return []
        ranges.append((start_date, start_minute, finish_minute))
    else:
        ranges.append((start_date, start_minute, 1439))
        ranges.append((finish_date, 0, finish_minute))

    bins: list[tuple[object, int]] = []
    for service_date, first_minute, last_minute in ranges:
        first_bin = (first_minute // bin_minutes) * bin_minutes
        last_active_minute = max(first_minute, last_minute - 1)
        last_bin = min(1435, (last_active_minute // bin_minutes) * bin_minutes)
        bins.extend((service_date, minute) for minute in range(first_bin, last_bin + 1, bin_minutes))
    return bins


def _phase_day_counts(person_day: pd.DataFrame) -> dict[str, int]:
    if person_day.empty or not {"phase", "date"}.issubset(person_day.columns):
        return {}
    phase_dates = person_day[["phase", "date"]].dropna().drop_duplicates().copy()
    phase_dates["phase"] = phase_dates["phase"].astype(str)
    return phase_dates.groupby("phase")["date"].nunique().astype(int).to_dict()


def _infer_motif_column(df: pd.DataFrame, motif_col: str | None) -> str | None:
    if motif_col:
        return motif_col if motif_col in df.columns else None
    for column in DEFAULT_MOTIF_COLUMNS:
        if column in df.columns:
            return column
    return None


def _build_heatmap_modes(
    demand: pd.DataFrame,
    *,
    row_col: str,
    row_order: list,
    row_labels: dict,
    minute_values: list[int],
    heatmap_bin_minutes: int,
    max_rows: int | None = None,
) -> list[dict]:
    if demand.empty or row_col not in demand.columns:
        return []

    work = demand.dropna(subset=[row_col]).copy()
    if work.empty:
        return []
    work["_heatmap_minute"] = (work["minute"] // heatmap_bin_minutes) * heatmap_bin_minutes

    if max_rows is not None:
        top_rows = (
            work.drop_duplicates([row_col, "date", "user_id"])
            .groupby(row_col)
            .size()
            .sort_values(ascending=False)
            .head(max_rows)
            .index
        )
        work = work.loc[work[row_col].isin(top_rows)].copy()
        row_order = [value for value in row_order if value in set(top_rows)]

    modes = [DEMAND_PROFILE_ALL_MODES_LABEL] + sorted(work["mode"].dropna().astype(str).unique().tolist())
    mode_payloads = []
    for mode in modes:
        if mode == DEMAND_PROFILE_ALL_MODES_LABEL:
            mode_work = work.drop_duplicates(["phase", row_col, "_heatmap_minute", "user_id", "date"])
        else:
            mode_work = work.loc[work["mode"].astype(str).eq(mode)].drop_duplicates(
                ["phase", row_col, "_heatmap_minute", "user_id", "date"]
            )
        if mode_work.empty:
            continue

        denominators = (
            mode_work.drop_duplicates(["phase", row_col, "date"])
            .groupby(["phase", row_col])["date"]
            .nunique()
            .to_dict()
        )
        counts = (
            mode_work.groupby(["phase", row_col, "_heatmap_minute"])
            .size()
            .rename("active_user_days")
            .reset_index()
        )

        phase_payloads = []
        for phase in sorted(mode_work["phase"].dropna().astype(str).unique().tolist()):
            phase_counts = counts.loc[counts["phase"].astype(str).eq(str(phase))]
            rows = []
            for row_value in row_order:
                denominator = max(1, int(denominators.get((phase, row_value), 1)))
                row_counts = phase_counts.loc[phase_counts[row_col].eq(row_value)]
                by_minute = row_counts.set_index("_heatmap_minute")["active_user_days"].to_dict()
                values = [
                    round(float(by_minute.get(minute, 0)) / denominator, 2)
                    for minute in minute_values
                ]
                if any(value > 0 for value in values):
                    rows.append(
                        {
                            "label": str(row_labels.get(row_value, row_value)),
                            "values": values,
                        }
                    )
            if rows:
                phase_payloads.append({"phase": str(phase), "rows": rows})
        if phase_payloads:
            mode_payloads.append({"mode": mode, "phases": phase_payloads})
    return mode_payloads


def _build_demand_heatmaps(
    demand: pd.DataFrame,
    *,
    heatmap_bin_minutes: int,
    max_motif_rows: int,
    motif_col: str | None,
) -> list[dict]:
    if demand.empty:
        return []
    minute_values = list(range(0, 1440, heatmap_bin_minutes))
    time_labels = [f"{minute // 60:02d}:00" for minute in minute_values]

    heatmaps = []
    weekday_order = list(WEEKDAY_LABELS)
    weekday_modes = _build_heatmap_modes(
        demand,
        row_col="weekday",
        row_order=weekday_order,
        row_labels=WEEKDAY_LABELS,
        minute_values=minute_values,
        heatmap_bin_minutes=heatmap_bin_minutes,
    )
    if weekday_modes:
        heatmaps.append(
            {
                "key": "weekday",
                "label": "Jours de semaine",
                "time_labels": time_labels,
                "modes": weekday_modes,
            }
        )

    if motif_col and "motif" in demand.columns:
        motif_values = sorted(demand["motif"].dropna().astype(str).unique().tolist())
        motif_modes = _build_heatmap_modes(
            demand,
            row_col="motif",
            row_order=motif_values,
            row_labels={value: value for value in motif_values},
            minute_values=minute_values,
            heatmap_bin_minutes=heatmap_bin_minutes,
            max_rows=max_motif_rows,
        )
        if motif_modes:
            heatmaps.append(
                {
                    "key": "motif",
                    "label": "Motifs agrégés",
                    "source_col": motif_col,
                    "time_labels": time_labels,
                    "modes": motif_modes,
                }
            )
    return heatmaps


def build_daily_demand_profile(
    legs: pd.DataFrame,
    person_day: pd.DataFrame,
    *,
    mode_col: str = "mode_niv1",
    motif_col: str | None = None,
    phase_col: str = "phase",
    default_phase_name: str = "All",
    bin_minutes: int = DEMAND_PROFILE_BIN_MINUTES,
    heatmap_bin_minutes: int = DEMAND_HEATMAP_BIN_MINUTES,
    max_motif_rows: int = 12,
) -> dict:
    """Build an average daily demand profile by phase, mode and time bin.

    The output is intentionally compact and JSON-like so it can be stored in
    `IndicatorResult.metadata` and rendered in HTML without requiring a charting
    dependency. Values are average active users per day in each phase. A user is
    counted once per date, phase, mode and time bin.

    Args:
        legs: Mobility legs with `user_id`, start/end datetime columns and a
            mode column.
        person_day: Person-day indicator table used to count phase days. When
            empty, observed leg dates are used as denominator.
        mode_col: Mode column to use for mode-specific curves.
        motif_col: Optional column used to build the motif heatmap. When
            omitted, common motif/purpose columns are inferred.
        phase_col: phase or period column in `legs`.
        default_phase_name: phase name used when `phase_col` is absent.
        bin_minutes: Time-bin size in minutes.
        heatmap_bin_minutes: Time-bin size used for the heatmap. One hour is
            the default to keep the HTML report readable.
        max_motif_rows: Maximum number of motif rows displayed in the heatmap.

    Returns:
        Dictionary containing time labels and one list of series per phase.
        Returns an empty dictionary when the required columns are missing.
    """

    required = {"user_id", mode_col}
    if legs.empty or not required.issubset(legs.columns):
        return {}

    work = legs.copy()
    work["_xyt_started_local"] = _local_datetime_column(work, "started_at_in_timezone", "started_at")
    work["_xyt_finished_local"] = _local_datetime_column(work, "finished_at_in_timezone", "finished_at")
    work["_xyt_phase"] = _phase_series(work, phase_col=phase_col, default_phase_name=default_phase_name)
    work["_xyt_mode"] = work[mode_col].fillna("Autres").astype(str)
    inferred_motif_col = _infer_motif_column(work, motif_col)
    work["_xyt_motif"] = (
        work[inferred_motif_col].fillna("Non renseigné").astype(str)
        if inferred_motif_col
        else pd.Series(pd.NA, index=work.index, dtype="object")
    )

    records = []
    for user_id, phase, mode, motif, started, finished in work[
        ["user_id", "_xyt_phase", "_xyt_mode", "_xyt_motif", "_xyt_started_local", "_xyt_finished_local"]
    ].itertuples(index=False, name=None):
        for service_date, minute in _leg_demand_bins(
            started,
            finished,
            bin_minutes=bin_minutes,
        ):
            records.append(
                {
                    "phase": str(phase),
                    "mode": str(mode),
                    "minute": int(minute),
                    "user_id": user_id,
                    "date": service_date,
                    "weekday": pd.Timestamp(service_date).weekday(),
                    "motif": motif if pd.notna(motif) else None,
                }
            )
    if not records:
        return {}

    demand = pd.DataFrame(records)
    minute_values = list(range(0, 1440, bin_minutes))
    time_labels = [f"{minute // 60:02d}:{minute % 60:02d}" for minute in minute_values]

    days_by_phase = _phase_day_counts(person_day)
    if not days_by_phase:
        days_by_phase = demand.groupby("phase")["date"].nunique().astype(int).to_dict()

    mode_counts = (
        demand.drop_duplicates(["phase", "mode", "minute", "user_id", "date"])
        .groupby(["phase", "mode", "minute"])
        .size()
        .rename("active_user_days")
        .reset_index()
    )
    all_counts = (
        demand.drop_duplicates(["phase", "minute", "user_id", "date"])
        .groupby(["phase", "minute"])
        .size()
        .rename("active_user_days")
        .reset_index()
    )

    phases = sorted(set(days_by_phase) | set(demand["phase"].astype(str).unique().tolist()))
    phase_profiles = []
    for phase in phases:
        denominator = max(1, int(days_by_phase.get(phase, 1)))
        phase_series = []
        all_phase = all_counts.loc[all_counts["phase"].astype(str).eq(str(phase))]
        all_by_minute = all_phase.set_index("minute")["active_user_days"].to_dict()
        phase_series.append(
            {
                "mode": DEMAND_PROFILE_ALL_MODES_LABEL,
                "values": [
                    round(float(all_by_minute.get(minute, 0)) / denominator, 2)
                    for minute in minute_values
                ],
            }
        )
        phase_modes = mode_counts.loc[mode_counts["phase"].astype(str).eq(str(phase))]
        for mode in sorted(phase_modes["mode"].dropna().astype(str).unique().tolist()):
            mode_phase = phase_modes.loc[phase_modes["mode"].astype(str).eq(mode)]
            mode_by_minute = mode_phase.set_index("minute")["active_user_days"].to_dict()
            phase_series.append(
                {
                    "mode": mode,
                    "values": [
                        round(float(mode_by_minute.get(minute, 0)) / denominator, 2)
                        for minute in minute_values
                    ],
                }
            )
        phase_profiles.append(
            {
                "phase": str(phase),
                "n_days": denominator,
                "series": phase_series,
            }
        )

    return {
        "bin_minutes": bin_minutes,
        "heatmap_bin_minutes": heatmap_bin_minutes,
        "unit": "personnes moyennes en déplacement",
        "time_labels": time_labels,
        "phases": phase_profiles,
        "heatmaps": _build_demand_heatmaps(
            demand,
            heatmap_bin_minutes=heatmap_bin_minutes,
            max_motif_rows=max_motif_rows,
            motif_col=inferred_motif_col,
        ),
    }
