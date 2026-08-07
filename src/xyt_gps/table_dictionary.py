"""Descriptions of exported mobility tables and columns."""

from __future__ import annotations

from collections.abc import Mapping

import geopandas as gpd
import pandas as pd


WORKING_TABLE_ORDER = (
    "storyline",
    "staypoints",
    "legs",
    "trips",
    "journeys",
    "user_stats",
    "user_day_coverage",
    "map_track_trip_journey",
    "map_legs_staypoints",
)

TABLE_DESCRIPTIONS = {
    "storyline": (
        "Chronologie harmonisée des événements MotionTag. Elle conserve les "
        "événements Track et Stay avec leurs dates, géométries, modes, motifs, "
        "expérience, phase et semaine relative."
    ),
    "staypoints": (
        "Sous-ensemble des événements Stay. Une ligne représente une activité "
        "ou un arrêt, avec son identifiant, sa période, son motif et son "
        "rattachement spatial lorsqu'il est disponible."
    ),
    "legs": (
        "Sous-ensemble des événements Track. Une ligne représente une étape de "
        "déplacement, unité principale pour les distances, modes, géométries, "
        "qualité GPS, excursions, CO2, occupation et santé."
    ),
    "trips": (
        "Déplacement MotionTag regroupant un ou plusieurs legs. La table porte "
        "les modes et motifs agrégés, la période, la distance, la durée et les "
        "zones origine-destination lorsque disponibles."
    ),
    "journeys": (
        "Chaîne de déplacements MotionTag regroupant un ou plusieurs trips. "
        "Elle sert aux agrégations de niveau supérieur lorsque cette unité est "
        "disponible dans la source."
    ),
    "user_stats": (
        "Synthèse par utilisateur : période de suivi, jours observés, jours "
        "actifs, complétude, confirmations, qualité GPS, excursions et flags "
        "d'inclusion analytique."
    ),
    "user_day_coverage": (
        "Table personne-jour. Elle distingue les jours dans la période, les "
        "jours trackés, les jours actifs, les jours sans mouvement observé, les "
        "jours non suivis et les jours recréés."
    ),
    "map_track_trip_journey": (
        "Table de correspondance entre legs, trips et journeys. Elle rend les "
        "jointures vérifiables sans inventer de clé parente lorsqu'un lien est "
        "absent."
    ),
    "map_legs_staypoints": (
        "Table de correspondance entre chaque staypoint et les legs adjacents. "
        "Elle sert à reconstruire les enchaînements activité-déplacement."
    ),
    "od_zone_correspondence": (
        "Table de correspondance origine-destination par découpage territorial. "
        "Elle contient une ligne par leg et par trip, sans géométrie, avec les "
        "identifiants de zone de départ et d'arrivée."
    ),
    "public_transport_legs": (
        "Informations complémentaires MotionTag sur certains legs de transport "
        "public, lorsqu'elles sont fournies dans le dump."
    ),
    "user_expe": (
        "Rattachement utilisateur-expérimentation : présence dans les sources, "
        "code d'invitation, phases, période d'expérience et appariement "
        "questionnaire."
    ),
    "relative_time_index": (
        "Index temporel relatif utilisé pour comparer des expériences décalées "
        "dans le calendrier à semaine de protocole équivalente."
    ),
    "occupancy_co2": (
        "Table d'enrichissement des legs pour les hypothèses d'occupation "
        "véhicule et les émissions CO2."
    ),
    "health": (
        "Table d'enrichissement des legs pour les métriques d'activité physique "
        "et de sédentarité."
    ),
}

TABLE_DESCRIPTION_PATTERNS = {
    "questionnaire_": (
        "Questionnaire apparié aux utilisateurs de l'instrument correspondant. "
        "Les schémas restent séparés lorsque les questionnaires diffèrent."
    ),
    "indicators_": "Table d'indicateurs agrégés produite pour les analyses ou le dashboard.",
}

COLUMN_DESCRIPTIONS = {
    "storyline_id": "Identifiant stable de l'événement MotionTag dans la storyline.",
    "leg_id": "Identifiant stable du leg, c'est-à-dire de l'étape de déplacement.",
    "activity_id": "Identifiant stable du staypoint, activité ou arrêt.",
    "trip_id": "Identifiant stable du déplacement MotionTag.",
    "journey_id": "Identifiant stable de la chaîne de déplacements MotionTag.",
    "user_id": "Identifiant utilisateur utilisé dans les tables du pipeline.",
    "user_statistics_id": "Identifiant de ligne de la table MotionTag UserStatistics.",
    "public_transport_leg_id": "Identifiant de ligne de la table MotionTag PublicTransportLegs.",
    "previous_leg_id": "Identifiant du leg immédiatement avant le staypoint dans la séquence.",
    "next_leg_id": "Identifiant du leg immédiatement après le staypoint dans la séquence.",
    "experiment_name": "Nom canonique de l'expérimentation ou de la composante analytique.",
    "motiontag_project_name": "Nom du projet source MotionTag après harmonisation.",
    "project_raw": "Nom de projet tel que présent dans la table d'affectation source.",
    "invite_code": "Code d'invitation utilisé pour apparier utilisateur et questionnaire.",
    "registration_code": "Code d'inscription issu de la source questionnaire ou utilisateur.",
    "questionnaire_match_value": "Valeur normalisée utilisée pour l'appariement questionnaire.",
    "in_questionnaire": "Indique si l'utilisateur a été retrouvé dans le questionnaire associé.",
    "phase": "Phase analytique attribuée à partir du calendrier de l'expérimentation.",
    "phase_number": "Numéro de phase sous forme numérique lorsque disponible.",
    "phase_start": "Date de début de la phase attribuée à la ligne.",
    "phase_end": "Date de fin de la phase attribuée à la ligne.",
    "analysis_phase": "Phase analytique ajustée pour les agrégations, notamment après retrait des semaines de transition.",
    "analysis_phase_number": "Numéro de phase analytique sous forme numérique lorsque disponible.",
    "analysis_phase_start": "Date de début de la fenêtre analytique attribuée à la ligne.",
    "analysis_phase_end": "Date de fin de la fenêtre analytique attribuée à la ligne.",
    "n_week_sequence": "Semaine relative calculée depuis le début de l'expérimentation.",
    "phase_week_number": "Rang de la semaine à l'intérieur de la phase.",
    "start_expe": "Date de début de la période analytique de l'expérimentation.",
    "end_expe": "Date de fin de la période analytique de l'expérimentation.",
    "type": "Type d'événement MotionTag, généralement Track ou Stay.",
    "started_at": "Timestamp de début en UTC ou timezone source normalisée.",
    "finished_at": "Timestamp de fin en UTC ou timezone source normalisée.",
    "started_at_timezone": "Fuseau horaire associé au début de l'événement.",
    "finished_at_timezone": "Fuseau horaire associé à la fin de l'événement.",
    "started_at_in_timezone": "Début de l'événement dans le fuseau horaire local fourni.",
    "finished_at_in_timezone": "Fin de l'événement dans le fuseau horaire local fourni.",
    "confirmed_at": "Timestamp de confirmation ou correction utilisateur lorsque disponible.",
    "created_at": "Timestamp de création de la ligne dans la source.",
    "updated_at": "Timestamp de dernière mise à jour de la ligne dans la source.",
    "started_on": "Date civile de début utilisée pour les agrégations journalières.",
    "date": "Date civile de la ligne personne-jour.",
    "geometry": "Géométrie spatiale retenue pour la table.",
    "lon": "Longitude du point représentatif, en coordonnées géographiques.",
    "lat": "Latitude du point représentatif, en coordonnées géographiques.",
    "length": "Longueur fournie par la source ou conservée après préparation, en mètres.",
    "length_leg": "Longueur métrique calculée sur la géométrie du leg, en mètres.",
    "duration": "Durée de l'objet de mobilité, généralement en secondes.",
    "distance": "Distance agrégée du trip ou journey, généralement en mètres.",
    "mode": "Mode de déplacement déclaré ou retenu au niveau storyline/leg.",
    "detected_mode": "Mode détecté automatiquement par MotionTag.",
    "main_mode": "Mode principal du trip ou journey fourni par MotionTag.",
    "modes": "Séquence ou combinaison des modes présents dans le trip ou journey.",
    "mode_niv1": "Mode agrégé de niveau 1 selon le mapping du projet.",
    "mode_niv2": "Mode agrégé de niveau 2 selon le mapping du projet.",
    "mode_mrmt": "Mode agrégé selon la nomenclature MRMT utilisée dans le projet.",
    "main_mode_niv1": "Mode principal agrégé de niveau 1.",
    "main_mode_niv2": "Mode principal agrégé de niveau 2.",
    "main_mode_mrmt": "Mode principal agrégé selon la nomenclature MRMT.",
    "modes_niv1": "Séquence de modes agrégés au niveau 1.",
    "modes_niv2": "Séquence de modes agrégés au niveau 2.",
    "modes_mrmt": "Séquence de modes agrégés selon la nomenclature MRMT.",
    "purpose": "Motif source ou motif retenu pour l'activité, le trip ou le journey.",
    "detected_purpose": "Motif détecté automatiquement par MotionTag.",
    "purpose_niv1": "Motif agrégé de niveau 1 selon le mapping du projet.",
    "purpose_mrmt": "Motif agrégé selon la nomenclature MRMT.",
    "main_purpose": "Motif principal du journey fourni par MotionTag.",
    "main_purpose_mrmt": "Motif principal agrégé du journey selon la nomenclature MRMT.",
    "main_journey_purpose_mrmt": "Motif principal du journey propagé au leg pour les enrichissements.",
    "journey_main_purpose_mrmt": "Motif principal du journey propagé au trip.",
    "trip_count": "Compteur technique ou poids unitaire de trip.",
    "journey_count": "Compteur technique ou poids unitaire de journey.",
    "track_count": "Compteur technique ou poids unitaire de track.",
    "tracks_count": "Nombre de tracks composant le trip ou journey.",
    "tracks_count_without_walks": "Nombre de tracks hors marche dans le trip ou journey.",
    "trips_count": "Nombre de trips composant le journey.",
    "long_distance": "Flag fournisseur indiquant un déplacement de longue distance.",
    "misdetected_completely": "Flag source indiquant une détection entièrement erronée.",
    "merged": "Flag source indiquant une ligne fusionnée.",
    "comment_feedback": "Commentaire utilisateur ou retour associé à la correction.",
    "occupancy": "Occupation déclarée dans la source lorsque disponible.",
    "occupancy_rate": "Taux d'occupation retenu pour les calculs CO2.",
    "occupancy_source": "Origine de l'hypothèse d'occupation retenue.",
    "occupancy_declared": "Occupation déclarée par l'utilisateur lorsque disponible.",
    "occupancy_declared_available": "Indique si une occupation déclarée exploitable existe.",
    "occupancy_observed_available": "Indique si une valeur observée d'occupation est disponible.",
    "occupancy_rate_computed": "Indique si le taux d'occupation a été calculé par le pipeline.",
    "occupancy_distance_class": "Classe de distance utilisée pour choisir le taux d'occupation.",
    "co2_factor_g_per_km": "Facteur CO2 total retenu, en gCO2e/km.",
    "co2_direct_factor_g_per_km": "Facteur CO2 direct retenu, en gCO2e/km.",
    "co2_g": "Émissions CO2 totales estimées pour le leg, en grammes.",
    "co2_direct_g": "Émissions CO2 directes estimées pour le leg, en grammes.",
    "co2_assumption": "Hypothèse ou source utilisée pour le calcul CO2.",
    "user_id_day": "Identifiant composite utilisateur-date.",
    "days_in_range_count": "Nombre de jours entre le premier et le dernier jour de suivi.",
    "tracked_days_count": "Nombre de jours disposant d'une observation de suivi.",
    "active_days_count": "Nombre de jours avec au moins un déplacement observé.",
    "zero_movement_tracked_days_count": "Nombre de jours trackés sans déplacement observé.",
    "untracked_days_count": "Nombre de jours dans la plage sans observation de suivi.",
    "inactive_days_count": "Nombre de jours sans déplacement, observés ou non selon la convention.",
    "tracking_completeness": "Part des jours dans la plage disposant d'un suivi.",
    "active_days_share": "Part des jours actifs dans la période de suivi.",
    "resampled_days_count": "Nombre de jours recréés par resampling.",
    "observed_tracking_days_count": "Nombre de jours réellement observés, hors jours recréés.",
    "observed_active_days_count": "Nombre de jours actifs réellement observés, hors jours recréés.",
    "observed_tracking_completeness": "Complétude calculée sur les seuls jours réellement observés.",
    "day_in_range": "Indique si la date est comprise dans la période de suivi retenue.",
    "tracked_day": "Indique si une observation de suivi existe ce jour-là.",
    "active_day": "Indique si au moins un déplacement est observé ce jour-là.",
    "movement_day": "Synonyme analytique d'un jour avec mouvement observé.",
    "zero_movement_tracked_day": "Jour suivi sans déplacement observé.",
    "untracked_day": "Jour compris dans la plage mais sans suivi observé.",
    "resampled_day": "Jour recréé par resampling plutôt qu'observé directement.",
    "max_gap": "Plus longue interruption de suivi détectée pour l'utilisateur.",
    "tracking_quality_ok": "Indique si l'utilisateur satisfait les critères de qualité de suivi.",
    "tracking_quality_reason": "Raison principale du statut de qualité de suivi.",
    "bad_user": "Flag d'exclusion analytique construit à partir de la qualité de suivi ou du signal.",
    "bad_signal_user": "Flag utilisateur indiquant un niveau élevé de pertes de signal.",
    "signal_quality_computed": "Indique si les métriques de qualité du signal ont été calculées.",
    "max_signal_loss_m": "Plus grande distance entre deux points consécutifs du leg, en mètres.",
    "relative_signal_loss": "Rapport entre la plus grande perte de signal et la longueur du leg.",
    "low_quality_legs_1": "Flag de perte de signal selon le premier seuil configuré.",
    "low_quality_legs_2": "Flag de perte de signal selon le second seuil configuré.",
    "low_quality_signal": "Flag agrégé de faible qualité du signal utilisé pour les contrôles.",
    "low_quality_legs_count": "Nombre de legs de faible qualité associés à l'utilisateur.",
    "low_quality_legs_share": "Part des legs de l'utilisateur portant un flag de faible qualité.",
    "mean_relative_signal_loss": "Moyenne utilisateur du ratio de perte de signal.",
    "mean_max_signal_loss_m": "Moyenne utilisateur de la perte de signal maximale, en mètres.",
    "point_per_linestring": "Nombre de points composant la géométrie du leg.",
    "extreme98_length_mode": "Flag indiquant une longueur supérieure au quantile 98 par mode.",
    "extreme99_length_mode": "Flag indiquant une longueur supérieure au quantile 99 par mode.",
    "signal_quality_reason": "Raison du flag de qualité du signal au niveau leg.",
    "excursion": "Indique si l'objet est marqué comme excursion hors zone configurée.",
    "excursion_legs_count": "Nombre de legs en excursion pour l'utilisateur.",
    "total_legs_count": "Nombre total de legs utilisés comme dénominateur.",
    "excursion_leg_ratio": "Part des legs de l'utilisateur marqués comme excursion.",
    "origin_id_zone": "Zone d'origine du leg dans le référentiel spatial configuré.",
    "destination_id_zone": "Zone de destination du leg dans le référentiel spatial configuré.",
    "girec_zone_o": "Zone d'origine selon le découpage GIREC.",
    "girec_zone_d": "Zone de destination selon le découpage GIREC.",
    "modus_zone_o": "Zone d'origine selon le découpage MODUS.",
    "modus_zone_d": "Zone de destination selon le découpage MODUS.",
    "commune_zone_o": "Commune d'origine selon la couche communale configurée.",
    "commune_zone_d": "Commune de destination selon la couche communale configurée.",
    "activity_id_zone": "Zone du staypoint dans le référentiel spatial configuré.",
    "trip_origin_id_zone": "Zone d'origine du trip dans le référentiel spatial configuré.",
    "trip_destination_id_zone": "Zone de destination du trip dans le référentiel spatial configuré.",
    "journey_origin_id_zone": "Zone d'origine du journey dans le référentiel spatial configuré.",
    "journey_destination_id_zone": "Zone de destination du journey dans le référentiel spatial configuré.",
    "grand_geneve_relation": "Relation du leg au périmètre Grand Genève.",
    "grand_geneve_relation_code": "Code numérique ou court de la relation au Grand Genève.",
    "geometry_origin": "Origine de la géométrie retenue, par exemple motiontag ou repaired_google.",
    "geometry_repair_applied": "Indique si une géométrie réparée a remplacé la géométrie MotionTag.",
    "map_match_status": "Statut du traitement réseau pour le leg.",
    "map_match_source": "Fournisseur ou méthode ayant produit la géométrie réseau.",
    "map_match_request_count": "Nombre de requêtes nécessaires pour traiter le leg.",
    "map_match_original_length_m": "Longueur de la géométrie originale envoyée au traitement réseau.",
    "map_match_clean_length_m": "Longueur de la géométrie retenue après traitement réseau.",
    "map_match_provider_distance_m": "Distance retournée par le fournisseur de routing ou snapping.",
    "google_routes_distance_m": "Distance retournée par Google Routes API.",
    "google_routes_duration_s": "Durée retournée par Google Routes API, en secondes.",
    "leg_date": "Date civile du leg.",
    "legs_date": "Date civile du leg, nom historique conservé dans certains exports.",
    "trip_date": "Date civile du trip.",
    "journey_date": "Date civile du journey.",
    "leading_activity_id": "Staypoint amont utilisé pour qualifier le trip.",
    "first_tracked_at": "Premier timestamp de suivi disponible pour l'utilisateur.",
    "latest_tracked_at": "Dernier timestamp de suivi disponible pour l'utilisateur.",
    "latest_confirmed_finished_at": "Dernier événement confirmé par l'utilisateur.",
    "latest_data_transmission_at": "Dernière transmission de données connue.",
    "latest_storyline_finished_at": "Fin du dernier événement storyline connu.",
    "confirmed_tracks_count": "Nombre de tracks confirmés par l'utilisateur.",
    "unconfirmed_tracks_count": "Nombre de tracks non confirmés.",
    "confirmed_days_count": "Nombre de jours contenant une confirmation utilisateur.",
    "unconfirmed_days_count": "Nombre de jours sans confirmation utilisateur.",
    "tracks_count": "Nombre total de tracks.",
    "weight": "Poids utilisateur utilisé dans certaines agrégations, si disponible.",
}


def table_description(table_name: str) -> str:
    """Return a short description for an exported table."""

    if table_name in TABLE_DESCRIPTIONS:
        return TABLE_DESCRIPTIONS[table_name]
    for prefix, description in TABLE_DESCRIPTION_PATTERNS.items():
        if table_name.startswith(prefix):
            return description
    return "[À COMPLÉTER] Table produite par le pipeline ou la source projet."


def column_description(column_name: str, *, table_name: str | None = None) -> str:
    """Return a short description for an exported column."""

    if column_name in COLUMN_DESCRIPTIONS:
        return COLUMN_DESCRIPTIONS[column_name]
    if column_name.endswith("_start"):
        return "Date ou timestamp de début associé à la variable préfixée."
    if column_name.endswith("_end"):
        return "Date ou timestamp de fin associé à la variable préfixée."
    if column_name.endswith("_at"):
        return "Timestamp associé à l'événement ou au statut nommé par la colonne."
    if column_name.endswith("_at_in_timezone"):
        return "Timestamp localisé associé à l'événement ou au statut nommé par la colonne."
    if column_name.endswith("_timezone"):
        return "Fuseau horaire associé au timestamp correspondant."
    if column_name.endswith("_count") or column_name.endswith("_counts"):
        return "Nombre d'objets correspondant au préfixe de la colonne."
    if column_name.endswith("_share") or column_name.endswith("_ratio"):
        return "Part ou ratio calculé pour le préfixe de la colonne."
    if column_name.endswith("_ok"):
        return "Flag booléen indiquant si le critère nommé par la colonne est satisfait."
    if column_name.endswith("_reason"):
        return "Motif textuel associé au flag ou au statut correspondant."
    if column_name.endswith("_zone_o"):
        return "Identifiant de zone d'origine dans le découpage indiqué par le préfixe."
    if column_name.endswith("_zone_d"):
        return "Identifiant de zone de destination dans le découpage indiqué par le préfixe."
    if column_name.endswith("_zone"):
        return "Identifiant de zone spatiale dans le référentiel configuré."
    if column_name.startswith("phase_") and column_name.endswith(("_start", "_end")):
        return "Borne calendaire de la phase expérimentale correspondante."
    if column_name.startswith(("days_in_range_phase_", "tracked_days_phase_", "active_days_phase_", "untracked_days_phase_")):
        return "Nombre de jours calculé pour la phase indiquée par le suffixe."
    if table_name and table_name.startswith("questionnaire_"):
        return "[À COMPLÉTER] Variable issue du questionnaire source."
    return "[À COMPLÉTER] Colonne issue de la source ou d'un enrichissement spécifique."


def _dtype_name(table: pd.DataFrame, column: str) -> str:
    if isinstance(table, gpd.GeoDataFrame) and column == table.geometry.name:
        return "geometry"
    return str(table[column].dtype)


def build_table_descriptions(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build one documentation row per exported table."""

    rows = []
    for table_name, table in tables.items():
        rows.append(
            {
                "table": table_name,
                "description": table_description(table_name),
                "row_count": int(len(table)),
                "column_count": int(len(table.columns)),
            }
        )
    return pd.DataFrame(rows, columns=["table", "description", "row_count", "column_count"])


def build_attribute_dictionary(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a column-level dictionary with dtype and missing-value counts."""

    rows = []
    for table_name, table in tables.items():
        table_description_value = table_description(table_name)
        row_count = int(len(table))
        geometry_crs = str(table.crs) if isinstance(table, gpd.GeoDataFrame) and table.crs else None
        for position, column in enumerate(table.columns, start=1):
            na_count = int(table[column].isna().sum())
            rows.append(
                {
                    "table": table_name,
                    "table_description": table_description_value,
                    "column_order": position,
                    "column": column,
                    "dtype": _dtype_name(table, column),
                    "description": column_description(column, table_name=table_name),
                    "na_count": na_count,
                    "na_share": float(na_count / row_count) if row_count else 0.0,
                    "non_na_count": int(row_count - na_count),
                    "row_count": row_count,
                    "geometry_crs": geometry_crs if isinstance(table, gpd.GeoDataFrame) and column == table.geometry.name else None,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "table",
            "table_description",
            "column_order",
            "column",
            "dtype",
            "description",
            "na_count",
            "na_share",
            "non_na_count",
            "row_count",
            "geometry_crs",
        ],
    )
