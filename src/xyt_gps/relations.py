"""Relation tables linking legs, staypoints, trips and journeys."""

from __future__ import annotations

import pandas as pd


def _append_mapping_rows(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Append nullable mapping rows without triggering pandas concat warnings."""

    if extra.empty:
        return base
    columns = list(dict.fromkeys([*base.columns, *extra.columns]))
    if base.empty:
        return extra.reindex(columns=columns).reset_index(drop=True)
    records = base.reindex(columns=columns).to_dict("records")
    records.extend(extra.reindex(columns=columns).to_dict("records"))
    return pd.DataFrame.from_records(records, columns=columns)


def _map_trips_to_journeys(
    trips: pd.DataFrame,
    journeys: pd.DataFrame,
    *,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    """Return a vectorized trip-to-journey mapping with nullable matches."""

    if trips.empty:
        return pd.DataFrame(columns=["trip_id", "journey_id"])
    if journeys.empty:
        return trips[["trip_id"]].copy().assign(journey_id=pd.NA)

    pairs = pd.merge(
        trips,
        journeys,
        on="user_id",
        how="left",
        suffixes=("_trip", "_journey"),
    )
    matched = pairs[
        pairs["journey_id"].notna()
        & pairs["started_at_journey"].le(pairs["trip_mid_time"] + tolerance)
        & pairs["finished_at_journey"].ge(pairs["trip_mid_time"] - tolerance)
    ][["trip_id", "journey_id"]].copy()

    missing_trip_ids = trips.loc[~trips["trip_id"].isin(matched["trip_id"]), "trip_id"]
    if not missing_trip_ids.empty:
        missing = pd.DataFrame({"trip_id": missing_trip_ids.to_numpy(), "journey_id": pd.NA})
        matched = _append_mapping_rows(matched, missing)
    return matched


def build_track_trip_journey_map(
    legs: pd.DataFrame,
    trips: pd.DataFrame,
    journeys: pd.DataFrame,
    *,
    tolerance: str | pd.Timedelta = "5s",
) -> pd.DataFrame:
    """Map legs to trips and journeys.

    If legs already contain a valid `trip_id`, the function uses it directly.
    Otherwise it matches each leg midpoint to trips from the same user, then
    matches each trip midpoint to journeys from the same user. Temporal joins
    are vectorized by user to avoid per-leg nested loops.
    """

    tolerance_delta = pd.Timedelta(tolerance)
    t_temp = trips[["trip_id", "user_id", "started_at", "finished_at"]].copy()
    t_temp["trip_mid_time"] = t_temp["started_at"] + (t_temp["finished_at"] - t_temp["started_at"]) / 2
    j_temp = journeys[["journey_id", "user_id", "started_at", "finished_at"]].copy()
    trip_journey = _map_trips_to_journeys(t_temp, j_temp, tolerance=tolerance_delta)

    if "trip_id" in legs.columns and legs["trip_id"].notna().any():
        leg_trip_source = legs[["leg_id", "trip_id"]].copy()
        leg_trip_source["_xyt_order"] = range(len(leg_trip_source))
        valid_trip_ids = set(t_temp["trip_id"].dropna())
        leg_trip_source["trip_id"] = leg_trip_source["trip_id"].where(
            leg_trip_source["trip_id"].isin(valid_trip_ids),
            pd.NA,
        )
        leg_trip_mapping = leg_trip_source.dropna(subset=["trip_id"]).copy()
        out = pd.merge(leg_trip_mapping, trip_journey, on="trip_id", how="left")
        missing = leg_trip_source.loc[~leg_trip_source["leg_id"].isin(out["leg_id"]), ["_xyt_order", "leg_id"]]
        if not missing.empty:
            out = _append_mapping_rows(out, missing.assign(trip_id=pd.NA, journey_id=pd.NA))
        return out.sort_values("_xyt_order").drop(columns="_xyt_order").reset_index(drop=True)

    legs_temp = legs[["leg_id", "user_id", "started_at", "finished_at"]].copy()
    legs_temp["_xyt_order"] = range(len(legs_temp))
    legs_temp["leg_mid_time"] = legs_temp["started_at"] + (legs_temp["finished_at"] - legs_temp["started_at"]) / 2

    if t_temp.empty:
        return legs_temp[["_xyt_order", "leg_id"]].assign(trip_id=pd.NA, journey_id=pd.NA).drop(
            columns="_xyt_order"
        )

    pairs = pd.merge(
        legs_temp,
        t_temp,
        on="user_id",
        how="left",
        suffixes=("_leg", "_trip"),
    )
    leg_trip_matches = pairs[
        pairs["trip_id"].notna()
        & pairs["started_at_trip"].le(pairs["leg_mid_time"] + tolerance_delta)
        & pairs["finished_at_trip"].ge(pairs["leg_mid_time"] - tolerance_delta)
    ][["_xyt_order", "leg_id", "trip_id"]].copy()

    if leg_trip_matches.empty:
        return legs_temp[["_xyt_order", "leg_id"]].assign(trip_id=pd.NA, journey_id=pd.NA).drop(
            columns="_xyt_order"
        )

    out = pd.merge(leg_trip_matches, trip_journey, on="trip_id", how="left")
    missing = legs_temp.loc[~legs_temp["leg_id"].isin(out["leg_id"]), ["_xyt_order", "leg_id"]]
    if not missing.empty:
        out = _append_mapping_rows(out, missing.assign(trip_id=pd.NA, journey_id=pd.NA))
    return out.sort_values("_xyt_order").drop(columns="_xyt_order").reset_index(drop=True)


def build_legs_staypoints_map(
    legs: pd.DataFrame,
    staypoints: pd.DataFrame,
    *,
    tolerance: str | pd.Timedelta = "5s",
) -> pd.DataFrame:
    """Map each staypoint to previous and next legs using near-identical timestamps."""

    stays = staypoints[["activity_id", "user_id", "started_at", "finished_at"]].copy()
    previous_leg = legs[["user_id", "finished_at", "leg_id"]].rename(
        columns={"finished_at": "started_at", "leg_id": "previous_leg_id"}
    )
    next_leg = legs[["user_id", "started_at", "leg_id"]].rename(
        columns={"started_at": "finished_at", "leg_id": "next_leg_id"}
    )

    stays = stays.sort_values(["started_at", "user_id"]).reset_index(drop=True)
    previous_leg = previous_leg.sort_values(["started_at", "user_id"]).reset_index(drop=True)
    mapped = pd.merge_asof(
        stays,
        previous_leg,
        on="started_at",
        by="user_id",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )

    mapped = mapped.sort_values(["finished_at", "user_id"]).reset_index(drop=True)
    next_leg = next_leg.sort_values(["finished_at", "user_id"]).reset_index(drop=True)
    mapped = pd.merge_asof(
        mapped,
        next_leg,
        on="finished_at",
        by="user_id",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    return mapped[["previous_leg_id", "activity_id", "next_leg_id"]].copy()


def add_journey_to_trips(trips: pd.DataFrame, journeys: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Add journey_id and journey purpose to trips."""

    out = pd.merge(
        trips,
        mapping[["trip_id", "journey_id"]].dropna(subset=["trip_id"]).drop_duplicates(subset=["trip_id"]),
        on="trip_id",
        how="left",
    )
    if "main_purpose_mrmt" in journeys.columns:
        out = pd.merge(
            out,
            journeys[["journey_id", "main_purpose_mrmt"]].drop_duplicates(subset=["journey_id"]),
            on="journey_id",
            how="left",
        ).rename(columns={"main_purpose_mrmt": "journey_main_purpose_mrmt"})
    return out


def add_trip_destination_activity(
    trips: pd.DataFrame,
    map_track_trip_journey: pd.DataFrame,
    map_legs_staypoints: pd.DataFrame,
) -> pd.DataFrame:
    """Add leading activity id to trips using the last leg of each trip."""

    if map_track_trip_journey.empty or map_legs_staypoints.empty:
        return trips.copy()
    trip_legs_info = map_track_trip_journey.dropna(subset=["trip_id"]).groupby("trip_id")["leg_id"].agg(["last"]).reset_index()
    trip_legs_info = trip_legs_info.rename(columns={"last": "last_leg_id"})
    leg_to_activity = map_legs_staypoints[["previous_leg_id", "activity_id"]].rename(
        columns={"previous_leg_id": "last_leg_id", "activity_id": "leading_activity_id"}
    )
    trip_activities = pd.merge(trip_legs_info, leg_to_activity, on="last_leg_id", how="left")
    return pd.merge(trips, trip_activities[["trip_id", "leading_activity_id"]], on="trip_id", how="left")


def add_excursion_flags_to_trips_journeys(
    trips: pd.DataFrame,
    journeys: pd.DataFrame,
    legs: pd.DataFrame,
    map_track_trip_journey: pd.DataFrame,
    *,
    excursion_col: str = "excursion",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Propagate leg-level excursion flags to trips and journeys."""

    trips_out = trips.copy()
    journeys_out = journeys.copy()
    if (
        excursion_col not in legs.columns
        or "leg_id" not in legs.columns
        or map_track_trip_journey.empty
        or "leg_id" not in map_track_trip_journey.columns
    ):
        return trips_out, journeys_out

    leg_flags = legs[["leg_id", excursion_col]].copy()
    leg_flags[excursion_col] = leg_flags[excursion_col].fillna(False).astype(bool)
    mapping = map_track_trip_journey.copy()
    for column in ("trip_id", "journey_id"):
        if column not in mapping.columns:
            mapping[column] = pd.NA
    mapped_flags = pd.merge(
        mapping[["leg_id", "trip_id", "journey_id"]],
        leg_flags,
        on="leg_id",
        how="left",
    )
    mapped_flags[excursion_col] = mapped_flags[excursion_col].fillna(False).astype(bool)

    if "trip_id" in trips_out.columns and "trip_id" in mapped_flags.columns:
        trip_flags = (
            mapped_flags.dropna(subset=["trip_id"])
            .groupby("trip_id", dropna=False)[excursion_col]
            .any()
            .astype(int)
            .rename(excursion_col)
            .reset_index()
        )
        trips_out = trips_out.drop(columns=[excursion_col], errors="ignore")
        trips_out = pd.merge(trips_out, trip_flags, on="trip_id", how="left")
        trips_out[excursion_col] = trips_out[excursion_col].fillna(0).astype(int)

    if "journey_id" in journeys_out.columns and "journey_id" in mapped_flags.columns:
        journey_flags = (
            mapped_flags.dropna(subset=["journey_id"])
            .groupby("journey_id", dropna=False)[excursion_col]
            .any()
            .astype(int)
            .rename(excursion_col)
            .reset_index()
        )
        journeys_out = journeys_out.drop(columns=[excursion_col], errors="ignore")
        journeys_out = pd.merge(journeys_out, journey_flags, on="journey_id", how="left")
        journeys_out[excursion_col] = journeys_out[excursion_col].fillna(0).astype(int)

    return trips_out, journeys_out
