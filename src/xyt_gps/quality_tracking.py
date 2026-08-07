"""Daily presence and weekly participation helpers."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .config import ProjectConfig
from .parsing import assign_phase


TRACKING_STATS_COLUMNS = [
    "user_id",
    "first_tracked_at",
    "latest_tracked_at",
    "days_in_range_count",
    "tracked_days_count",
    "active_days_count",
    "zero_movement_tracked_days_count",
    "untracked_days_count",
    "max_gap",
    "inactive_days_count",
    "tracking_completeness",
    "active_days_share",
    "resampled_days_count",
]


USER_DAY_COVERAGE_COLUMNS = [
    "experiment_name",
    "user_id",
    "date",
    "phase",
    "phase_week_number",
    "analysis_phase",
    "analysis_phase_number",
    "analysis_phase_start",
    "analysis_phase_end",
    "n_week_sequence",
    "day_in_range",
    "tracked_day",
    "active_day",
    "movement_day",
    "zero_movement_tracked_day",
    "untracked_day",
    "resampled_day",
]

WEEKDAY_BALANCE_COLUMNS = [
    "lundi",
    "mardi",
    "mercredi",
    "jeudi",
    "vendredi",
    "samedi",
    "dimanche",
]

WEEKDAY_LABELS = {
    0: "lundi",
    1: "mardi",
    2: "mercredi",
    3: "jeudi",
    4: "vendredi",
    5: "samedi",
    6: "dimanche",
}

WEEKDAY_SHORT_COLUMNS = {
    "lundi": "n_lun",
    "mardi": "n_mar",
    "mercredi": "n_mer",
    "jeudi": "n_jeu",
    "vendredi": "n_ven",
    "samedi": "n_sam",
    "dimanche": "n_dim",
}


def summarize_phase_window_calendar_balance(
    phase_windows: pd.DataFrame,
    *,
    experiment_col: str = "experiment_name",
    phase_col: str = "phase",
    start_col: str = "phase_start",
    end_col: str = "phase_end",
    window_type_col: str = "window_type",
    max_allowed_weekday_spread: int = 0,
) -> pd.DataFrame:
    """Check weekday balance from configured phase windows only.

    This diagnostic is independent from observed tracking. It is meant to be
    run early from configuration files, before person-day coverage exists, to
    verify that analytical phase windows contain the same number of Mondays,
    Tuesdays, ..., Sundays.
    """

    required = {experiment_col, phase_col, start_col, end_col}
    missing = required - set(phase_windows.columns)
    if missing:
        raise KeyError(f"phase_windows missing required column(s): {sorted(missing)}")

    output_columns = [
        window_type_col,
        experiment_col,
        phase_col,
        start_col,
        end_col,
        "n_calendar_days",
        "n_weeks",
        *WEEKDAY_SHORT_COLUMNS.values(),
        "weekday_min",
        "weekday_max",
        "weekday_spread",
        "all_weekdays_present",
        "weekday_balanced",
        "starts_on_monday",
        "ends_on_sunday",
        "full_monday_sunday_weeks",
    ]
    if phase_windows.empty:
        return pd.DataFrame(columns=output_columns)

    rows = []
    for row in phase_windows.to_dict("records"):
        start = pd.to_datetime(row.get(start_col), errors="coerce")
        end = pd.to_datetime(row.get(end_col), errors="coerce")
        if pd.isna(start) or pd.isna(end) or start > end:
            continue

        dates = pd.date_range(start.normalize(), end.normalize(), freq="D")
        weekday_counts = (
            pd.Series(dates.weekday, dtype="Int64")
            .map(WEEKDAY_LABELS)
            .value_counts()
            .reindex(WEEKDAY_BALANCE_COLUMNS, fill_value=0)
        )
        n_days = int(len(dates))
        weekday_min = int(weekday_counts.min())
        weekday_max = int(weekday_counts.max())
        weekday_spread = weekday_max - weekday_min
        all_weekdays_present = weekday_min > 0
        weekday_balanced = all_weekdays_present and weekday_spread <= max_allowed_weekday_spread
        starts_on_monday = start.weekday() == 0
        ends_on_sunday = end.weekday() == 6
        rows.append(
            {
                window_type_col: row.get(window_type_col, "analysis"),
                experiment_col: row.get(experiment_col),
                phase_col: row.get(phase_col),
                start_col: start.date().isoformat(),
                end_col: end.date().isoformat(),
                "n_calendar_days": n_days,
                "n_weeks": n_days / 7,
                **{
                    short_column: int(weekday_counts[weekday])
                    for weekday, short_column in WEEKDAY_SHORT_COLUMNS.items()
                },
                "weekday_min": weekday_min,
                "weekday_max": weekday_max,
                "weekday_spread": weekday_spread,
                "all_weekdays_present": all_weekdays_present,
                "weekday_balanced": weekday_balanced,
                "starts_on_monday": starts_on_monday,
                "ends_on_sunday": ends_on_sunday,
                "full_monday_sunday_weeks": weekday_balanced
                and starts_on_monday
                and ends_on_sunday,
            }
        )

    return pd.DataFrame(rows, columns=output_columns).reset_index(drop=True)


def _movement_mask(storyline: pd.DataFrame) -> pd.Series:
    """Identifier les événements correspondant à un déplacement réel."""

    if "type" not in storyline.columns:
        return pd.Series(True, index=storyline.index)
    event_type = storyline["type"].fillna("").astype(str).str.strip().str.lower()
    return event_type.str.endswith("track")


def _phase_start_lookup(config: ProjectConfig | None) -> dict[str, pd.Timestamp]:
    lookup: dict[str, pd.Timestamp] = {}
    if config is None:
        return lookup
    for phase in config.phases:
        lookup[phase.name] = pd.Timestamp(phase.start_date)
    return lookup


def _phase_number(name: object) -> int | pd.NA:
    value = str(name)
    digits = "".join(character for character in value if character.isdigit())
    if not digits:
        return pd.NA
    return int(digits)


def _assign_phase_columns(
    table: pd.DataFrame,
    phases,
    *,
    date_col: str = "date",
    phase_col: str = "phase",
    default: object = "Other",
    number_col: str | None = None,
    start_col: str | None = None,
    end_col: str | None = None,
    week_col: str | None = None,
) -> pd.DataFrame:
    out = table.copy()
    out[phase_col] = out[date_col].apply(lambda value: assign_phase(value, phases, default=default))
    if number_col is not None:
        out[number_col] = out[phase_col].map(_phase_number).astype("Int64")
    if start_col is not None or end_col is not None or week_col is not None:
        start_lookup = {phase.name: pd.Timestamp(phase.start_date) for phase in phases}
        end_lookup = {phase.name: pd.Timestamp(phase.end_date) for phase in phases}
        phase_start = pd.to_datetime(out[phase_col].map(start_lookup), errors="coerce")
        if start_col is not None:
            out[start_col] = phase_start.dt.date.astype("string")
        if end_col is not None:
            phase_end = pd.to_datetime(out[phase_col].map(end_lookup), errors="coerce")
            out[end_col] = phase_end.dt.date.astype("string")
        if week_col is not None:
            phase_week = ((out[date_col] - phase_start).dt.days // 7 + 1).astype("Int64")
            out[week_col] = phase_week.where(out[phase_col].notna())
    return out


def build_user_day_coverage(
    storyline: pd.DataFrame,
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Construire le calendrier canonique personne-jour du suivi GPS.

    La plage individuelle va du premier au dernier jour réellement observé.
    Les lignes artificielles ``Resampled_stay`` n'étendent jamais cette plage.
    ``tracked_day`` signale une observation réelle et ``active_day`` au moins
    un déplacement réel de type Track ou AnnotatedTrack.
    """

    missing = {"user_id", "started_at"} - set(storyline.columns)
    if missing:
        raise KeyError(f"storyline missing required column(s): {sorted(missing)}")

    events = storyline.copy()
    events["date"] = pd.to_datetime(events["started_at"], errors="coerce", utc=True).dt.normalize().dt.tz_localize(None)
    events = events.dropna(subset=["user_id", "date"])
    if events.empty:
        return pd.DataFrame(columns=USER_DAY_COVERAGE_COLUMNS)

    resampled = (
        events["type"].eq("Resampled_stay")
        if "type" in events.columns
        else pd.Series(False, index=events.index)
    )
    events["tracked_day"] = (~resampled).astype(int)
    events["active_day"] = ((~resampled) & _movement_mask(events)).astype(int)
    events["resampled_day"] = resampled.astype(int)

    daily = (
        events.groupby(["user_id", "date"], as_index=False)
        .agg(
            tracked_day=("tracked_day", "max"),
            active_day=("active_day", "max"),
            resampled_day=("resampled_day", "max"),
        )
    )
    observed = daily.loc[daily["tracked_day"].eq(1)]
    bounds = observed.groupby("user_id")["date"].agg(["min", "max"])

    calendars = []
    for user_id, row in bounds.iterrows():
        calendars.append(
            pd.DataFrame(
                {
                    "user_id": user_id,
                    "date": pd.date_range(row["min"], row["max"], freq="D"),
                }
            )
        )
    if not calendars:
        return pd.DataFrame(columns=USER_DAY_COVERAGE_COLUMNS)

    coverage = pd.concat(calendars, ignore_index=True).merge(
        daily,
        on=["user_id", "date"],
        how="left",
    )
    for column in ("tracked_day", "active_day", "resampled_day"):
        coverage[column] = coverage[column].fillna(0).astype(int)
    coverage["movement_day"] = coverage["active_day"]
    coverage["day_in_range"] = 1
    coverage["zero_movement_tracked_day"] = (
        coverage["tracked_day"].eq(1) & coverage["active_day"].eq(0)
    ).astype(int)
    coverage["untracked_day"] = coverage["tracked_day"].eq(0).astype(int)

    experiment_name = config.experiment_name if config is not None else None
    if "experiment_name" in events.columns:
        experiments = (
            events.loc[events["tracked_day"].eq(1), ["user_id", "experiment_name"]]
            .dropna(subset=["experiment_name"])
            .drop_duplicates("user_id")
        )
        coverage = coverage.merge(experiments, on="user_id", how="left")
        if experiment_name is not None:
            coverage["experiment_name"] = coverage["experiment_name"].fillna(experiment_name)
    else:
        coverage.insert(0, "experiment_name", experiment_name)

    if config is not None and config.phases:
        coverage = _assign_phase_columns(
            coverage,
            config.phases,
            phase_col="phase",
            week_col="phase_week_number",
        )
        analysis_phases = config.analysis_phases or config.phases
        coverage = _assign_phase_columns(
            coverage,
            analysis_phases,
            phase_col="analysis_phase",
            default=pd.NA,
            number_col="analysis_phase_number",
            start_col="analysis_phase_start",
            end_col="analysis_phase_end",
        )
        experiment_start = pd.Timestamp(config.experiment_start_date)
        sequence_week = ((coverage["date"] - experiment_start).dt.days // 7 + 1).astype("Int64")
        coverage["n_week_sequence"] = sequence_week.where(
            coverage["date"].ge(experiment_start)
        )
    else:
        coverage["phase"] = "All"
        first_date = coverage.groupby("user_id")["date"].transform("min")
        coverage["phase_week_number"] = ((coverage["date"] - first_date).dt.days // 7 + 1).astype("Int64")
        coverage["analysis_phase"] = coverage["phase"]
        coverage["analysis_phase_number"] = pd.NA
        coverage["analysis_phase_start"] = pd.NA
        coverage["analysis_phase_end"] = pd.NA
        coverage["n_week_sequence"] = coverage["phase_week_number"]

    return coverage[USER_DAY_COVERAGE_COLUMNS].sort_values(
        ["experiment_name", "user_id", "date"], na_position="last"
    ).reset_index(drop=True)


def summarize_weekday_balance(
    user_day_coverage: pd.DataFrame,
    *,
    group_cols: Iterable[str] = ("experiment_name", "phase"),
    date_col: str = "date",
    day_basis: str = "day_in_range",
    max_allowed_calendar_spread: int = 1,
) -> pd.DataFrame:
    """Summarize weekday balance by analytical group.

    The diagnostic is meant for multi-day mobility studies where averages are
    sensitive to the weekday composition of each phase. It reports both unique
    calendar days and person-days based on the requested daily denominator
    (`day_in_range`, `tracked_day` or `active_day`).

    Args:
        user_day_coverage: Canonical person-day calendar.
        group_cols: Columns used to define analytical groups, typically
            `experiment_name` and `phase`.
        date_col: Date column in `user_day_coverage`.
        day_basis: Boolean or numeric person-day denominator to sum.
        max_allowed_calendar_spread: Maximum accepted gap between the least and
            most represented weekday in unique calendar days. The default `1`
            accepts periods that cannot contain exactly the same number of each
            weekday, while still requiring every weekday to be represented.

    Returns:
        One row per group with weekday counts and min/max/spread diagnostics.

    Raises:
        KeyError: If required columns are missing.
    """

    group_cols = tuple(group_cols)
    required = set(group_cols) | {date_col, day_basis}
    missing = required - set(user_day_coverage.columns)
    if missing:
        raise KeyError(f"user_day_coverage missing required column(s): {sorted(missing)}")

    if user_day_coverage.empty:
        return pd.DataFrame(
            columns=[
                *group_cols,
                "n_users",
                "calendar_days_count",
                "calendar_weekday_min",
                "calendar_weekday_max",
                "calendar_weekday_spread",
                "calendar_all_weekdays_present",
                "calendar_weekday_balanced",
                "person_days_count",
                "person_weekday_min",
                "person_weekday_max",
                "person_weekday_spread",
                "person_all_weekdays_present",
                "person_weekday_balanced",
                *[f"calendar_{weekday}" for weekday in WEEKDAY_BALANCE_COLUMNS],
                *[f"person_days_{weekday}" for weekday in WEEKDAY_BALANCE_COLUMNS],
            ]
        )

    work = user_day_coverage.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work = work.dropna(subset=[date_col])
    if work.empty:
        return summarize_weekday_balance(
            work,
            group_cols=group_cols,
            date_col=date_col,
            day_basis=day_basis,
            max_allowed_calendar_spread=max_allowed_calendar_spread,
        )

    work["_weekday"] = work[date_col].dt.weekday.map(WEEKDAY_LABELS)
    work["_day_weight"] = pd.to_numeric(work[day_basis], errors="coerce").fillna(0)

    group_index = list(group_cols)
    users = (
        work.groupby(group_index, dropna=False)["user_id"].nunique().rename("n_users")
        if "user_id" in work.columns
        else work.groupby(group_index, dropna=False).size().rename("n_users")
    )

    calendar_days = work.drop_duplicates([*group_index, date_col])
    calendar_counts = (
        calendar_days.groupby([*group_index, "_weekday"], dropna=False)
        .size()
        .unstack("_weekday", fill_value=0)
        .reindex(columns=WEEKDAY_BALANCE_COLUMNS, fill_value=0)
        .add_prefix("calendar_")
    )
    person_counts = (
        work.groupby([*group_index, "_weekday"], dropna=False)["_day_weight"]
        .sum()
        .unstack("_weekday", fill_value=0)
        .reindex(columns=WEEKDAY_BALANCE_COLUMNS, fill_value=0)
        .add_prefix("person_days_")
    )

    summary = pd.concat([users, calendar_counts, person_counts], axis=1).reset_index()
    calendar_columns = [f"calendar_{weekday}" for weekday in WEEKDAY_BALANCE_COLUMNS]
    person_columns = [f"person_days_{weekday}" for weekday in WEEKDAY_BALANCE_COLUMNS]

    summary["calendar_days_count"] = summary[calendar_columns].sum(axis=1).astype(int)
    summary["calendar_weekday_min"] = summary[calendar_columns].min(axis=1).astype(int)
    summary["calendar_weekday_max"] = summary[calendar_columns].max(axis=1).astype(int)
    summary["calendar_weekday_spread"] = summary["calendar_weekday_max"] - summary["calendar_weekday_min"]
    summary["calendar_all_weekdays_present"] = summary["calendar_weekday_min"].gt(0)
    summary["calendar_weekday_balanced"] = (
        summary["calendar_all_weekdays_present"]
        & summary["calendar_weekday_spread"].le(max_allowed_calendar_spread)
    )
    summary["person_days_count"] = summary[person_columns].sum(axis=1)
    summary["person_weekday_min"] = summary[person_columns].min(axis=1)
    summary["person_weekday_max"] = summary[person_columns].max(axis=1)
    summary["person_weekday_spread"] = summary["person_weekday_max"] - summary["person_weekday_min"]
    summary["person_all_weekdays_present"] = summary["person_weekday_min"].gt(0)
    summary["person_weekday_balanced"] = (
        summary["person_all_weekdays_present"]
        & summary["person_weekday_spread"].le(max_allowed_calendar_spread)
    )

    ordered_columns = [
        *group_cols,
        "n_users",
        "calendar_days_count",
        "calendar_weekday_min",
        "calendar_weekday_max",
        "calendar_weekday_spread",
        "calendar_all_weekdays_present",
        "calendar_weekday_balanced",
        "person_days_count",
        "person_weekday_min",
        "person_weekday_max",
        "person_weekday_spread",
        "person_all_weekdays_present",
        "person_weekday_balanced",
        *calendar_columns,
        *person_columns,
    ]
    return summary[ordered_columns].sort_values(list(group_cols)).reset_index(drop=True)


def summarize_phase_window_weekday_balance(
    user_day_coverage: pd.DataFrame,
    phase_windows: pd.DataFrame,
    *,
    day_basis: str = "day_in_range",
    experiment_col: str = "experiment_name",
    phase_col: str = "phase",
    start_col: str = "phase_start",
    end_col: str = "phase_end",
    window_type_col: str = "window_type",
    date_col: str = "date",
    max_allowed_calendar_spread: int = 1,
) -> pd.DataFrame:
    """Check weekday balance for explicit phase windows.

    `phase_windows` can contain both official and adjusted analytical windows.
    The output keeps one row per experiment, window type and phase, and exposes
    compact weekday columns (`n_lun`, `n_mar`, ..., `n_dim`) based on the
    requested person-day denominator.
    """

    coverage_required = {experiment_col, date_col, day_basis}
    coverage_missing = coverage_required - set(user_day_coverage.columns)
    if coverage_missing:
        raise KeyError(f"user_day_coverage missing required column(s): {sorted(coverage_missing)}")

    window_required = {experiment_col, phase_col, start_col, end_col}
    window_missing = window_required - set(phase_windows.columns)
    if window_missing:
        raise KeyError(f"phase_windows missing required column(s): {sorted(window_missing)}")

    output_columns = [
        window_type_col,
        experiment_col,
        phase_col,
        start_col,
        end_col,
        "n_users",
        "calendar_days_count",
        "person_days_count",
        *WEEKDAY_SHORT_COLUMNS.values(),
        "weekday_min",
        "weekday_max",
        "weekday_spread",
        "all_weekdays_present",
        "weekday_balanced",
    ]
    if user_day_coverage.empty or phase_windows.empty:
        return pd.DataFrame(columns=output_columns)

    coverage = user_day_coverage.copy()
    coverage[date_col] = pd.to_datetime(coverage[date_col], errors="coerce").dt.normalize()
    coverage = coverage.dropna(subset=[date_col])
    if coverage.empty:
        return pd.DataFrame(columns=output_columns)

    frames = []
    for row in phase_windows.to_dict("records"):
        start = pd.to_datetime(row.get(start_col), errors="coerce")
        end = pd.to_datetime(row.get(end_col), errors="coerce")
        if pd.isna(start) or pd.isna(end) or start > end:
            continue
        experiment_name = row.get(experiment_col)
        selected = coverage.loc[
            coverage[experiment_col].eq(experiment_name)
            & coverage[date_col].ge(start.normalize())
            & coverage[date_col].le(end.normalize())
        ].copy()
        if selected.empty:
            continue
        selected[window_type_col] = row.get(window_type_col, "analysis")
        selected[phase_col] = row.get(phase_col)
        selected[start_col] = start.date().isoformat()
        selected[end_col] = end.date().isoformat()
        frames.append(selected)

    if not frames:
        return pd.DataFrame(columns=output_columns)

    labelled = pd.concat(frames, ignore_index=True, sort=False)
    balance = summarize_weekday_balance(
        labelled,
        group_cols=(window_type_col, experiment_col, phase_col, start_col, end_col),
        date_col=date_col,
        day_basis=day_basis,
        max_allowed_calendar_spread=max_allowed_calendar_spread,
    )
    rename_columns = {
        f"person_days_{weekday}": short_column
        for weekday, short_column in WEEKDAY_SHORT_COLUMNS.items()
    }
    rename_columns.update(
        {
            "person_weekday_min": "weekday_min",
            "person_weekday_max": "weekday_max",
            "person_weekday_spread": "weekday_spread",
            "person_all_weekdays_present": "all_weekdays_present",
            "person_weekday_balanced": "weekday_balanced",
        }
    )
    compact = balance.rename(columns=rename_columns)
    return compact[output_columns].reset_index(drop=True)



def calculate_user_tracking_stats(storyline: pd.DataFrame) -> pd.DataFrame:
    """Calculate observed tracking windows and missing days per user.

    Args:
        storyline: Parsed storyline table. It must contain
            `user_id` and `started_at`. The function expects one row per
            stay or track segment.

    Returns:
        A user-level table with first and latest tracked dates, the numbers
        of calendar, tracked and movement days, the maximum gap between
        tracked days and tracking completeness.

    Raises:
        KeyError: If `user_id` or `started_at` is missing.
    """

    for column in ("user_id", "started_at"):
        if column not in storyline.columns:
            raise KeyError(f"storyline must contain {column}")

    coverage = build_user_day_coverage(storyline)
    rows = []
    for user_id, group in coverage.groupby("user_id", dropna=False):
        tracked_dates = group.loc[group["tracked_day"].eq(1), "date"].sort_values()
        min_date = tracked_dates.min()
        max_date = tracked_dates.max()
        total_days = int(group["day_in_range"].sum())
        tracked_days = int(group["tracked_day"].sum())
        active_days = int(group["active_day"].sum())
        gaps = tracked_dates.diff().dt.days.dropna()
        max_gap_days = int(gaps.max() - 1) if not gaps.empty and gaps.max() > 1 else 0
        rows.append(
            {
                "user_id": user_id,
                "first_tracked_at": min_date,
                "latest_tracked_at": max_date,
                "days_in_range_count": total_days,
                "tracked_days_count": tracked_days,
                "active_days_count": active_days,
                "zero_movement_tracked_days_count": tracked_days - active_days,
                "untracked_days_count": total_days - tracked_days,
                "max_gap": max_gap_days,
                "inactive_days_count": total_days - tracked_days,
                "tracking_completeness": tracked_days / total_days,
                "active_days_share": active_days / total_days,
                "resampled_days_count": int(group["resampled_day"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=TRACKING_STATS_COLUMNS)


def build_daily_tracking_presence(
    storyline: pd.DataFrame,
    config: ProjectConfig | None = None,
    *,
    user_id_column: str = "user_id",
    date_column: str = "started_at",
    exclude_resampled: bool = True,
) -> pd.DataFrame:
    """Build one observed tracking row per user and date.

    This table is intentionally based on observed rows before temporal
    resampling. Otherwise, artificial `Resampled_stay` rows would make missing
    days look like tracked days in participation analyses.

    Args:
        storyline: Parsed storyline table.
        config: Optional project configuration used to assign phases.
        user_id_column: User identifier column.
        date_column: Datetime column used to define active tracking days.
        exclude_resampled: When true, rows where `type == "Resampled_stay"`
            are ignored.

    Returns:
        A table with `user_id`, `tracking_date`, `tracked_day`, `active_day`
        and, when a config with phases is provided, `phase`.

    Raises:
        KeyError: If the requested user or date column is missing.
    """

    missing = {user_id_column, date_column} - set(storyline.columns)
    if missing:
        raise KeyError(f"storyline missing required column(s): {sorted(missing)}")

    out = storyline.copy()
    if exclude_resampled and "type" in out.columns:
        out = out.loc[~out["type"].eq("Resampled_stay")].copy()

    parsed_dates = pd.to_datetime(out[date_column], errors="coerce", utc=True)
    out["tracking_date"] = pd.to_datetime(parsed_dates.dt.date)
    daily = (
        out[[user_id_column, "tracking_date"]]
        .dropna(subset=[user_id_column, "tracking_date"])
        .drop_duplicates()
        .rename(columns={user_id_column: "user_id"})
        .sort_values(["user_id", "tracking_date"])
        .reset_index(drop=True)
    )
    daily["tracked_day"] = 1
    movement_by_day = (
        out.assign(_movement_day=_movement_mask(out).astype(int))
        .groupby([user_id_column, "tracking_date"])["_movement_day"]
        .max()
        .rename("movement_day")
        .reset_index()
        .rename(columns={user_id_column: "user_id"})
    )
    daily = daily.merge(movement_by_day, on=["user_id", "tracking_date"], how="left")
    daily["movement_day"] = daily["movement_day"].fillna(0).astype(int)
    daily["active_day"] = daily["movement_day"]
    if config is not None and config.phases:
        daily["phase"] = daily["tracking_date"].apply(lambda value: assign_phase(value, config.phases))
    return daily


def _phase_week_calendar(config: ProjectConfig) -> pd.DataFrame:
    rows = []
    protocol_week_number = 1
    for phase_index, phase in enumerate(config.phases, start=1):
        phase_start = pd.Timestamp(phase.start_date)
        phase_end = pd.Timestamp(phase.end_date)
        if phase_start > phase_end:
            continue
        phase_week_number = 1
        week_start = phase_start
        while week_start <= phase_end:
            week_end = min(week_start + pd.Timedelta(days=6), phase_end)
            rows.append(
                {
                    "phase": phase.name,
                    "phase_index": phase_index,
                    "phase_week_number": phase_week_number,
                    "protocol_week_number": protocol_week_number,
                    "week_start": week_start,
                    "week_end": week_end,
                    "week_days_count": int((week_end - week_start).days + 1),
                }
            )
            phase_week_number += 1
            protocol_week_number += 1
            week_start = week_start + pd.Timedelta(days=7)
    return pd.DataFrame(
        rows,
        columns=[
            "phase",
            "phase_index",
            "phase_week_number",
            "protocol_week_number",
            "week_start",
            "week_end",
            "week_days_count",
        ],
    )


def _single_period_week_calendar(name: str, start, end) -> pd.DataFrame:
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
        return pd.DataFrame(
            columns=[
                "phase",
                "phase_index",
                "phase_week_number",
                "protocol_week_number",
                "week_start",
                "week_end",
                "week_days_count",
            ]
        )

    rows = []
    week_number = 1
    week_start = start_date
    while week_start <= end_date:
        week_end = min(week_start + pd.Timedelta(days=6), end_date)
        rows.append(
            {
                "phase": name,
                "phase_index": 1,
                "phase_week_number": week_number,
                "protocol_week_number": week_number,
                "week_start": week_start,
                "week_end": week_end,
                "week_days_count": int((week_end - week_start).days + 1),
            }
        )
        week_number += 1
        week_start = week_start + pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def build_weekly_participation_grid(
    storyline: pd.DataFrame,
    config: ProjectConfig,
    *,
    user_ids: Iterable | None = None,
    user_id_column: str = "user_id",
    date_column: str = "started_at",
    exclude_resampled: bool = True,
    default_period_name: str = "All",
) -> pd.DataFrame:
    """Count observed active tracking days by user and protocol week.

    When phases are configured, weeks are relative to these phases rather than
    ISO calendar weeks. Without configured phases, the grid uses one analytical
    period named `default_period_name` and spans the observed tracking dates.

    Args:
        storyline: Parsed storyline table, preferably before resampling.
        config: Project configuration containing phase dates.
        user_ids: Optional full list of expected users. If omitted, users are
            taken from `storyline`.
        user_id_column: User identifier column in `storyline`.
        date_column: Datetime column used to define active tracking days.
        exclude_resampled: When true, artificial `Resampled_stay` rows are
            ignored.
        default_period_name: Name used in the output when no phase is
            configured.

    Returns:
        A complete user x phase-week table. `tracked_days_count` measures GPS
        presence, `active_days_count` measures movement days and
        `participation_score` is based on tracked days.

    Raises:
        ValueError: If no phase is configured and no tracking date is
            available to infer a default analysis period.
    """

    if user_ids is None:
        if user_id_column not in storyline.columns:
            raise KeyError(f"storyline must contain {user_id_column}")
        users = pd.Series(storyline[user_id_column].dropna().unique(), name="user_id")
    else:
        users = pd.Series(list(user_ids), name="user_id").dropna().drop_duplicates()
    users = users.astype(object).sort_values(key=lambda series: series.astype(str)).reset_index(drop=True)

    daily = build_daily_tracking_presence(
        storyline,
        config if config.phases else None,
        user_id_column=user_id_column,
        date_column=date_column,
        exclude_resampled=exclude_resampled,
    )

    calendar = _phase_week_calendar(config)
    if calendar.empty:
        if daily.empty:
            raise ValueError(
                "Cannot build a weekly participation grid without phases and without observed tracking dates."
            )
        calendar = _single_period_week_calendar(
            default_period_name,
            daily["tracking_date"].min(),
            daily["tracking_date"].max(),
        )
        daily["phase"] = default_period_name

    grid = pd.DataFrame({"user_id": users})
    grid["_key"] = 1
    calendar = calendar.copy()
    calendar["_key"] = 1
    grid = pd.merge(grid, calendar, on="_key", how="inner").drop(columns="_key")

    daily = daily.loc[daily["phase"].isin(calendar["phase"])].copy()
    phase_starts = (
        calendar.sort_values(["phase", "phase_week_number"])
        .drop_duplicates("phase")[["phase", "week_start"]]
    )
    daily = daily.merge(phase_starts, on="phase", how="left")
    if daily.empty:
        active = pd.DataFrame(
            {
                "user_id": pd.Series(dtype="object"),
                "phase": pd.Series(dtype="object"),
                "phase_week_number": pd.Series(dtype="int64"),
                "tracked_days_count": pd.Series(dtype="int64"),
                "active_days_count": pd.Series(dtype="int64"),
            }
        )
    else:
        daily["phase_week_number"] = (
            (daily["tracking_date"] - daily["week_start"]).dt.days // 7 + 1
        ).astype(int)
        active = (
            daily.groupby(["user_id", "phase", "phase_week_number"], dropna=False)
            .agg(
                tracked_days_count=("tracked_day", "sum"),
                active_days_count=("active_day", "sum"),
            )
            .clip(upper=7)
            .astype({"tracked_days_count": int, "active_days_count": int})
            .reset_index()
        )

    grid = pd.merge(
        grid,
        active,
        on=["user_id", "phase", "phase_week_number"],
        how="left",
    )
    grid["tracked_days_count"] = grid["tracked_days_count"].fillna(0).astype(int)
    grid["active_days_count"] = grid["active_days_count"].fillna(0).astype(int)
    grid["movement_days_count"] = grid["active_days_count"]
    grid["participation_score"] = grid["tracked_days_count"]
    grid["tracked_days_share_7"] = grid["tracked_days_count"].div(7)
    grid["active_days_share_7"] = grid["active_days_count"].div(7)
    grid["phase_week_completeness"] = (
        grid["tracked_days_count"].div(grid["week_days_count"]).clip(upper=1)
    )
    grid["participation_level"] = pd.cut(
        grid["tracked_days_count"],
        bins=[-1, 0, 2, 4, 6, 7],
        labels=["none", "low", "partial", "good", "complete"],
    )
    if config.experiment_name:
        grid.insert(0, "experiment_name", config.experiment_name)
        sort_columns = ["experiment_name", "user_id", "protocol_week_number"]
    else:
        sort_columns = ["user_id", "protocol_week_number"]
    return grid.sort_values(sort_columns).reset_index(drop=True)


def summarize_participation_grid(
    participation_grid: pd.DataFrame,
    *,
    good_week_min_days: int = 5,
) -> pd.DataFrame:
    """Summarize weekly participation coverage.

    Args:
        participation_grid: Table produced by
            `build_weekly_participation_grid`.
        good_week_min_days: Minimum tracked days for a week to be considered
            good or complete.

    Returns:
        One summary row for all phases and one row per phase.
    """

    if participation_grid.empty:
        return pd.DataFrame(
            columns=[
                "scope",
                "phase",
                "users_count",
                "user_weeks_count",
                "mean_tracked_days_count",
                "mean_active_days_count",
                "good_or_complete_weeks_count",
                "good_or_complete_weeks_share",
                "complete_weeks_count",
                "complete_weeks_share",
            ]
        )

    def summarize(group: pd.DataFrame, *, scope: str, phase: str | None) -> dict:
        good = group["tracked_days_count"].ge(good_week_min_days)
        complete = group["tracked_days_count"].eq(7)
        return {
            "scope": scope,
            "phase": phase,
            "users_count": int(group["user_id"].nunique()),
            "user_weeks_count": int(len(group)),
            "mean_tracked_days_count": float(group["tracked_days_count"].mean()),
            "mean_active_days_count": float(group["active_days_count"].mean()),
            "good_or_complete_weeks_count": int(good.sum()),
            "good_or_complete_weeks_share": float(good.mean()),
            "complete_weeks_count": int(complete.sum()),
            "complete_weeks_share": float(complete.mean()),
        }

    rows = [summarize(participation_grid, scope="all", phase=None)]
    if "phase" in participation_grid.columns:
        for phase, group in participation_grid.groupby("phase", sort=False):
            rows.append(summarize(group, scope="phase", phase=phase))
    return pd.DataFrame(rows)
