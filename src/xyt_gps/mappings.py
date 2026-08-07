"""Mode and purpose mapping configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping

import pandas as pd


DEFAULT_STORYLINE_PURPOSE_NIV1 = {
    "Domicile": ["home"],
    "Travail": ["work", "study"],
    "Loisir": ["leisure", "eat", "family_friends", "sport"],
    "Achat": ["shopping", "errand"],
    "Attente": ["wait"],
    "Santé": ["medical_visit", "assistance"],
    "Autre": ["other", None, "unknown"],
}

DEFAULT_STORYLINE_MODE_NIV1 = {
    "Voiture": ["Mode::Car", "Mode::Carsharing", "Mode::Ridepooling", "Mode::Ecar", "Mode::TaxiUber"],
    "TP": [
        "Mode::Boat",
        "Mode::Bus",
        "Mode::LightRail",
        "Mode::RegionalTrain",
        "Mode::Subway",
        "Mode::Train",
        "Mode::Tram",
        "Mode::Ferry",
        "Mode::Coach",
        "Mode::Cablecar",
    ],
    "Marche": ["Mode::Walk", "Mode::Hiking", "Mode::Scooter"],
    "Vélo": ["Mode::Bicycle", "Mode::Bikesharing", "Mode::Ebicycle"],
    "Autres": ["Mode::Other", "Mode::Airplane"],
}

DEFAULT_STORYLINE_MODE_NIV2 = {
    "Voiture": ["Mode::Car", "Mode::Ridepooling", "Mode::Ecar", "Mode::TaxiUber"],
    "Carsharing": ["Mode::Carsharing"],
    "TPU": [
        "Mode::Boat",
        "Mode::Bus",
        "Mode::LightRail",
        "Mode::Subway",
        "Mode::Tram",
        "Mode::Ferry",
        "Mode::Coach",
        "Mode::Cablecar",
    ],
    "Train": ["Mode::RegionalTrain", "Mode::Train"],
    "Marche": ["Mode::Walk", "Mode::Hiking", "Mode::Scooter"],
    "Vélo": ["Mode::Bicycle", "Mode::Bikesharing", "Mode::Ebicycle"],
    "Autres": ["Mode::Other", "Mode::Airplane"],
}

DEFAULT_STORYLINE_MODE_MRMT = {
    "Voiture": ["Mode::Car", "Mode::Ridepooling", "Mode::Carsharing", "Mode::Ecar"],
    "Taxi": ["Mode::TaxiUber"],
    "2RM": ["Mode::Motorbike"],
    "Train": ["Mode::RegionalTrain", "Mode::Train", "Mode::LightRail", "Mode::Coach"],
    "Bus": ["Mode::Bus", "Mode::Cablecar"],
    "Tram/Métro": ["Mode::Subway", "Mode::Tram"],
    "Bateau": ["Mode::Ferry", "Mode::Boat"],
    "Marche": ["Mode::Walk", "Mode::Hiking", "Mode::Scooter"],
    "Vélo conventionnel": ["Mode::Bicycle", "Mode::Bikesharing"],
    "Vélo électrique": ["Mode::Ebicycle"],
    "Autres": ["Mode::Other"],
    "Airplane": ["Mode::Airplane"],
}

DEFAULT_TRIP_PURPOSE_MRMT = {
    "Domicile": ["At home"],
    "Travail": ["Working"],
    "Formation": ["Education"],
    "Loisir": ["Leisure", "Eat out", "Family & Friends", "Sports"],
    "Achat et services": ["Shopping", "Errand", "Medical visit"],
    "Accompagnement": ["Drop off / pick up someone"],
    "Tout motifs": ["Other", "Unknown", "unknown", None, "Waiting"],
}

DEFAULT_TRIP_MODE_NIV1 = {
    "Voiture": ["Car", "Electric car", "Taxi/Uber", "Motorbike", "Ridepooling", "Carsharing"],
    "TP": [
        "Boat",
        "Bus",
        "LightRail",
        "Regional train",
        "Regional Train",
        "Train",
        "Subway",
        "Tramway",
        "Tram",
        "Rapid transit railway",
        "Cable car",
        "Coach",
        "Ferry",
    ],
    "Marche": ["Walking", "Hiking", "Scooter"],
    "Vélo": ["Bicycle", "Bikesharing", "E-Bike", "Ebicycle"],
    "Autres": ["Other", "Airplane"],
}

DEFAULT_TRIP_MODE_NIV2 = {
    "Voiture": ["Car", "Electric car", "Taxi/Uber", "Motorbike", "Ridepooling"],
    "Carsharing": ["Carsharing"],
    "TPU": ["Boat", "Bus", "LightRail", "Subway", "Tram", "Tramway", "Cable car", "Coach", "Ferry"],
    "Train": ["Regional train", "Regional Train", "Train", "Rapid transit railway"],
    "Marche": ["Walking", "Hiking", "Scooter"],
    "Vélo": ["Bicycle", "Bikesharing", "E-Bike", "Ebicycle"],
    "Autres": ["Other", "Airplane"],
}

DEFAULT_TRIP_MODE_MRMT = {
    "Voiture": ["Car", "Electric car", "Ridepooling", "Carsharing"],
    "Taxi": ["Taxi/Uber"],
    "2RM": ["Motorbike"],
    "Train": ["Regional train", "Regional Train", "Train", "Rapid transit railway"],
    "Bus": ["Bus", "Coach", "Cable car"],
    "Tram/Métro": ["Subway", "Tramway", "Tram", "LightRail"],
    "Bateau": ["Ferry", "Boat"],
    "Marche": ["Walking", "Hiking", "Scooter"],
    "Vélo conventionnel": ["Bicycle", "Bikesharing"],
    "Vélo électrique": ["E-Bike", "Ebicycle"],
    "Autres": ["Other"],
    "Airplane": ["Airplane"],
}


def _freeze_mapping(mapping: Mapping[str, list]) -> dict[str, tuple]:
    return {key: tuple(values) for key, values in mapping.items()}


@dataclass(frozen=True)
class MobilityMappings:
    """Project-level mode and purpose mappings used by transformations."""

    storyline_purpose_niv1: Mapping[str, tuple] = field(
        default_factory=lambda: _freeze_mapping(DEFAULT_STORYLINE_PURPOSE_NIV1)
    )
    storyline_mode_niv1: Mapping[str, tuple] = field(
        default_factory=lambda: _freeze_mapping(DEFAULT_STORYLINE_MODE_NIV1)
    )
    storyline_mode_niv2: Mapping[str, tuple] = field(
        default_factory=lambda: _freeze_mapping(DEFAULT_STORYLINE_MODE_NIV2)
    )
    storyline_mode_mrmt: Mapping[str, tuple] = field(
        default_factory=lambda: _freeze_mapping(DEFAULT_STORYLINE_MODE_MRMT)
    )
    trip_purpose_mrmt: Mapping[str, tuple] = field(
        default_factory=lambda: _freeze_mapping(DEFAULT_TRIP_PURPOSE_MRMT)
    )
    trip_mode_niv1: Mapping[str, tuple] = field(
        default_factory=lambda: _freeze_mapping(DEFAULT_TRIP_MODE_NIV1)
    )
    trip_mode_niv2: Mapping[str, tuple] = field(
        default_factory=lambda: _freeze_mapping(DEFAULT_TRIP_MODE_NIV2)
    )
    trip_mode_mrmt: Mapping[str, tuple] = field(
        default_factory=lambda: _freeze_mapping(DEFAULT_TRIP_MODE_MRMT)
    )
    default_mode: str = "Autres"
    default_purpose: str = "Autre"

    def to_dict(self) -> dict[str, Mapping[str, tuple] | str]:
        return {
            "storyline_purpose_niv1": self.storyline_purpose_niv1,
            "storyline_mode_niv1": self.storyline_mode_niv1,
            "storyline_mode_niv2": self.storyline_mode_niv2,
            "storyline_mode_mrmt": self.storyline_mode_mrmt,
            "trip_purpose_mrmt": self.trip_purpose_mrmt,
            "trip_mode_niv1": self.trip_mode_niv1,
            "trip_mode_niv2": self.trip_mode_niv2,
            "trip_mode_mrmt": self.trip_mode_mrmt,
            "default_mode": self.default_mode,
            "default_purpose": self.default_purpose,
        }


def mode_purpose_mapping(**kwargs) -> MobilityMappings:
    """Build mode and purpose mappings for a project.

    This is a readable factory around `MobilityMappings`. Pass the same keyword
    arguments as the dataclass when a project needs custom groupings.

    Example:
        `mode_purpose_mapping(storyline_mode_niv1={"Marche": ("Mode::Walk",)})`
    """

    return MobilityMappings(**kwargs)


def map_value(value, mapping: Mapping[str, Iterable], *, default: str = "Autres") -> str:
    """Map one provider value to a project category."""

    for category, values in mapping.items():
        if pd.isna(value) and any(item is None for item in values):
            return category
        if value in values:
            return category
    return default


def map_sequence(value, mapping: Mapping[str, Iterable], *, default: str = "Autres") -> str:
    """Map a provider sequence such as `Mode::Walk + Mode::Bus` category by category."""

    if pd.isna(value) or value == "":
        return ""
    modes = [mode.strip() for mode in str(value).split(" + ") if mode.strip()]
    result = " + ".join(map_value(mode, mapping, default=default) for mode in modes)
    if str(value).endswith(" + "):
        result += " + "
    return result
