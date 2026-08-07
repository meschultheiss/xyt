"""Tracking quality reports and user-level tracking flags."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from .config import ProjectConfig
from .quality_tracking import build_daily_tracking_presence


def build_tracking_gap_report(
    storyline: pd.DataFrame,
    config: ProjectConfig | None = None,
    *,
    user_ids: Iterable | None = None,
    user_id_column: str = "user_id",
    date_column: str = "started_at",
    exclude_resampled: bool = True,
    default_period_name: str = "All",
) -> pd.DataFrame:
    """Summarize observed, missing and consecutive tracking days.

    This is the package version of the useful generic part of
    the historical tracking-quality notebook: one row per user and analytical period, with active
    days, missing days, maximum gap and maximum consecutive observed days.

    Args:
        storyline: Parsed storyline table.
        config: Optional project configuration. When phases are configured,
            the report is computed by phase; otherwise it uses the observed
            period named `default_period_name`.
        user_ids: Optional full list of expected users.
        user_id_column: User identifier column in `storyline`.
        date_column: Datetime column used to define active tracking days.
        exclude_resampled: Whether to ignore artificial `Resampled_stay` rows.
        default_period_name: Period name used when no phase is configured.

    Returns:
        A user-period table with tracking coverage and gap metrics.
    """

    daily = build_daily_tracking_presence(
        storyline,
        config if config is not None and config.phases else None,
        user_id_column=user_id_column,
        date_column=date_column,
        exclude_resampled=exclude_resampled,
    )
    if user_ids is None:
        users = pd.Series(daily["user_id"].dropna().unique(), name="user_id")
    else:
        users = pd.Series(list(user_ids), name="user_id").dropna().drop_duplicates()
    users = users.astype(object).sort_values(key=lambda series: series.astype(str)).reset_index(drop=True)

    periods = []
    if config is not None and config.phases:
        for phase in config.phases:
            periods.append(
                {
                    "phase": phase.name,
                    "period_start": pd.Timestamp(phase.start_date),
                    "period_end": pd.Timestamp(phase.end_date),
                }
            )
    elif not daily.empty:
        periods.append(
            {
                "phase": default_period_name,
                "period_start": daily["tracking_date"].min(),
                "period_end": daily["tracking_date"].max(),
            }
        )
        daily["phase"] = default_period_name

    rows = []
    for period in periods:
        period_start = pd.Timestamp(period["period_start"])
        period_end = pd.Timestamp(period["period_end"])
        period_days = int((period_end - period_start).days + 1)
        period_daily = daily.loc[
            daily["tracking_date"].between(period_start, period_end)
            & daily.get("phase", period["phase"]).eq(period["phase"])
        ].copy()

        for user_id in users:
            user_dates = (
                period_daily.loc[period_daily["user_id"].eq(user_id), "tracking_date"]
                .dropna()
                .drop_duplicates()
                .sort_values()
            )
            tracked_days = int(len(user_dates))
            active_days = int(
                period_daily.loc[period_daily["user_id"].eq(user_id), "active_day"].sum()
            )
            if tracked_days:
                diffs = user_dates.diff().dt.days.dropna()
                max_gap = int(diffs.max() - 1) if not diffs.empty and diffs.max() > 1 else 0
                streak = max_streak = 1
                for diff in diffs:
                    if diff == 1:
                        streak += 1
                    else:
                        streak = 1
                    max_streak = max(max_streak, streak)
            else:
                max_gap = period_days
                max_streak = 0

            rows.append(
                {
                    "user_id": user_id,
                    "phase": period["phase"],
                    "period_start": period_start,
                    "period_end": period_end,
                    "period_days_count": period_days,
                    "tracked_days_count": tracked_days,
                    "active_days_count": active_days,
                    "zero_movement_tracked_days_count": tracked_days - active_days,
                    "missing_days_count": period_days - tracked_days,
                    "max_gap_days": max_gap,
                    "max_consecutive_days": max_streak,
                    "tracking_completeness": tracked_days / period_days if period_days else 0,
                }
            )
    return pd.DataFrame(rows)


def summarize_phase_tracking(
    user_stats: pd.DataFrame,
    config: ProjectConfig,
    *,
    day_thresholds: Iterable[int] = (7, 14, 21),
) -> pd.DataFrame:
    """Summarize user tracking coverage by configured phase.

    Args:
        user_stats: User-level table containing `n_days_phase_1`,
            `n_days_phase_2`, etc.
        config: Project configuration containing phase dates.
        day_thresholds: Day-count thresholds to report, for example
            7, 14 and 21 days.

    Returns:
        One row per phase with user counts above each threshold and mean
        coverage over the theoretical phase duration.
    """

    rows = []
    thresholds = tuple(int(value) for value in day_thresholds)
    for index, phase in enumerate(config.phases, start=1):
        column = f"n_days_phase_{index}"
        if column not in user_stats.columns:
            continue
        phase_start = pd.Timestamp(phase.start_date)
        phase_end = pd.Timestamp(phase.end_date)
        duration = int((phase_end - phase_start).days + 1)
        values = pd.to_numeric(user_stats[column], errors="coerce").fillna(0)
        row = {
            "phase": phase.name,
            "phase_start": phase_start,
            "phase_end": phase_end,
            "phase_days_count": duration,
            "users_count": int(len(values)),
            "mean_tracked_days_count": float(values.mean()) if len(values) else 0.0,
            "mean_coverage": float(values.div(duration).mean()) if duration and len(values) else 0.0,
        }
        for threshold in thresholds:
            row[f"users_ge_{threshold}_days"] = int(values.ge(threshold).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_tracking_periods(
    user_stats: pd.DataFrame,
    config: ProjectConfig,
    *,
    min_days_by_phase: Mapping[str, int] | None = None,
    user_day_coverage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate effective tracked days per configured phase.

    Args:
        user_stats: User-level tracking table, usually produced by
            `calculate_user_tracking_stats`.
        config: Project configuration containing phase definitions and
            tracking thresholds.
        min_days_by_phase: Optional override for minimum tracked days by
            phase name.

    Returns:
        A copy of `user_stats` with `n_days_phase_*`, `phase*_start` and
        `phase*_end` columns. If no phase is configured, the input schema is
        preserved.
    """

    thresholds = min_days_by_phase or config.tracking_thresholds.min_days_by_phase
    out = user_stats.copy()
    if not config.phases or out.empty:
        return out

    for index, phase in enumerate(config.phases, start=1):
        minimum = thresholds.get(phase.name, 1)
        phase_start = pd.to_datetime(phase.start)
        phase_end = pd.to_datetime(phase.end)
        if isinstance(user_day_coverage, pd.DataFrame) and not user_day_coverage.empty:
            phase_days = user_day_coverage.loc[
                user_day_coverage["phase"].astype(str).eq(str(phase.name))
            ].copy()
            summary = (
                phase_days.groupby("user_id", as_index=False)
                .agg(
                    days_in_range=("day_in_range", "sum"),
                    tracked_days=("tracked_day", "sum"),
                    active_days=("active_day", "sum"),
                    untracked_days=("untracked_day", "sum"),
                    first_phase_day=("date", "min"),
                    last_phase_day=("date", "max"),
                )
            )
            out = out.merge(summary, on="user_id", how="left")
            for column in ("days_in_range", "tracked_days", "active_days", "untracked_days"):
                out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
            out[f"days_in_range_phase_{index}"] = out.pop("days_in_range")
            out[f"tracked_days_phase_{index}"] = out.pop("tracked_days")
            out[f"active_days_phase_{index}"] = out.pop("active_days")
            out[f"untracked_days_phase_{index}"] = out.pop("untracked_days")
            out[f"n_days_phase_{index}"] = out[f"tracked_days_phase_{index}"]
            out[f"phase{index}_start"] = out.pop("first_phase_day")
            out[f"phase{index}_end"] = out.pop("last_phase_day")
            continue

        def round_days(days: int) -> int:
            if pd.isna(days) or days < minimum:
                return 0
            if config.tracking_thresholds.round_to_full_weeks:
                days = days - (days % 7)
            return int(days) if days >= minimum else 0

        has_tracking = (
            pd.to_datetime(out["first_tracked_at"], errors="coerce").notna()
            & pd.to_datetime(out["latest_tracked_at"], errors="coerce").notna()
            & pd.to_numeric(out.get("tracked_days_count", out.get("active_days_count", 0)), errors="coerce").fillna(0).gt(0)
        )
        effective_starts = pd.concat(
            [pd.to_datetime(out["first_tracked_at"]), pd.Series(phase_start, index=out.index)],
            axis=1,
        ).max(axis=1)
        effective_ends = pd.concat(
            [pd.to_datetime(out["latest_tracked_at"]), pd.Series(phase_end, index=out.index)],
            axis=1,
        ).min(axis=1)
        raw_days = (effective_ends - effective_starts).dt.days + 1
        raw_days = raw_days.where(effective_starts.le(effective_ends) & has_tracking, 0).clip(lower=0)
        rounded = raw_days.apply(round_days)
        out[f"n_days_phase_{index}"] = rounded
        out[f"phase{index}_start"] = effective_starts.where(rounded.gt(0), pd.NaT)
        out[f"phase{index}_end"] = effective_ends.where(rounded.gt(0), pd.NaT)
    return out


def flag_tracking_quality(user_stats: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Add transparent tracking-quality flags to a user table.

    Args:
        user_stats: User-level table with at least `tracked_days_count`.
        config: Project configuration containing total and phase-level
            tracking thresholds.

    Returns:
        A copy of `user_stats` with boolean flags such as
        `tracked_days_ok`, `phase*_tracked_days_ok`, `tracking_quality_ok`
        and the categorical reason `tracking_quality_reason`.

    Raises:
        KeyError: If `tracked_days_count` is missing.
    """

    if "tracked_days_count" not in user_stats.columns:
        raise KeyError("user_stats must contain tracked_days_count")

    out = user_stats.copy()
    out["tracked_days_ok"] = out["tracked_days_count"].fillna(0).ge(
        config.tracking_thresholds.min_total_tracked_days
    )
    phase_flags = []
    for index, phase in enumerate(config.phases, start=1):
        phase_column = f"n_days_phase_{index}"
        if phase_column not in out.columns:
            continue
        flag_column = f"phase{index}_tracked_days_ok"
        minimum = config.tracking_thresholds.min_days_by_phase.get(phase.name, 1)
        out[flag_column] = out[phase_column].fillna(0).ge(minimum)
        phase_flags.append(flag_column)

    quality_columns = ["tracked_days_ok", *phase_flags]
    out["tracking_quality_ok"] = out[quality_columns].all(axis=1)
    out["tracking_quality_reason"] = "ok"
    out.loc[~out["tracked_days_ok"], "tracking_quality_reason"] = "insufficient_total_tracked_days"
    for flag_column in phase_flags:
        out.loc[
            out["tracking_quality_reason"].eq("ok") & ~out[flag_column],
            "tracking_quality_reason",
        ] = f"insufficient_{flag_column.removesuffix('_ok')}"
    return out


def build_tracking_quality_report(user_stats: pd.DataFrame) -> pd.DataFrame:
    """Build a compact one-row report for user tracking quality.

    Args:
        user_stats: User-level table with tracking-quality columns.

    Returns:
        A one-row table with user counts, valid-user share and median
        tracking-duration indicators. The output is meant for quick notebook
        checks before applying filters.
    """

    if user_stats.empty:
        return pd.DataFrame(
            [
                {
                    "users_count": 0,
                    "tracking_quality_ok_count": 0,
                    "tracking_quality_ok_share": 0.0,
                    "median_tracked_days_count": 0.0,
                    "median_active_days_count": 0.0,
                    "median_inactive_days_count": 0.0,
                    "median_tracking_completeness": 0.0,
                }
            ]
        )

    quality_ok = (
        user_stats["tracking_quality_ok"].fillna(False)
        if "tracking_quality_ok" in user_stats.columns
        else pd.Series(False, index=user_stats.index)
    )
    return pd.DataFrame(
        [
            {
                "users_count": int(len(user_stats)),
                "tracking_quality_ok_count": int(quality_ok.sum()),
                "tracking_quality_ok_share": float(quality_ok.mean()),
                "median_tracked_days_count": float(user_stats["tracked_days_count"].median()),
                "median_active_days_count": float(user_stats["active_days_count"].median()),
                "median_inactive_days_count": float(user_stats["inactive_days_count"].median()),
                "median_tracking_completeness": float(user_stats["tracking_completeness"].median()),
            }
        ]
    )
