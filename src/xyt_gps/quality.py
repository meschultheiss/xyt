"""Quality public facade for the data preparation workflow."""

from .quality_tracking import (
    TRACKING_STATS_COLUMNS,
    USER_DAY_COVERAGE_COLUMNS,
    build_daily_tracking_presence,
    build_user_day_coverage,
    build_weekly_participation_grid,
    calculate_user_tracking_stats,
    summarize_phase_window_calendar_balance,
    summarize_phase_window_weekday_balance,
    summarize_weekday_balance,
    summarize_participation_grid,
)
from .quality_reports import (
    build_tracking_gap_report,
    build_tracking_quality_report,
    calculate_tracking_periods,
    flag_tracking_quality,
    summarize_phase_tracking,
)
from .quality_diagnostics import (
    build_mode_detection_precision,
    build_user_confirmation_rates,
    get_extreme_legs_by_mode,
    summarize_leg_lengths_by_mode,
)
from .quality_resampling import resample_missing_stays
from .quality_selection import (
    build_user_selection_table,
    filter_mobility_dataset_by_users,
    filter_table_by_users,
    select_analysis_users,
    select_valid_tracking_users,
)

__all__ = [
    "TRACKING_STATS_COLUMNS",
    "USER_DAY_COVERAGE_COLUMNS",
    "build_daily_tracking_presence",
    "build_mode_detection_precision",
    "build_tracking_gap_report",
    "build_tracking_quality_report",
    "build_user_day_coverage",
    "build_user_confirmation_rates",
    "build_user_selection_table",
    "build_weekly_participation_grid",
    "calculate_tracking_periods",
    "calculate_user_tracking_stats",
    "filter_mobility_dataset_by_users",
    "filter_table_by_users",
    "flag_tracking_quality",
    "get_extreme_legs_by_mode",
    "resample_missing_stays",
    "select_analysis_users",
    "select_valid_tracking_users",
    "summarize_leg_lengths_by_mode",
    "summarize_participation_grid",
    "summarize_phase_window_calendar_balance",
    "summarize_phase_tracking",
    "summarize_phase_window_weekday_balance",
    "summarize_weekday_balance",
]
