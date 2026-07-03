from __future__ import annotations

from datetime import date, timedelta
from difflib import SequenceMatcher
from itertools import combinations
import math
import re
import unicodedata
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
from pybaseball import cache, playerid_reverse_lookup, statcast


st.set_page_config(page_title="Advanced MLB Hit Dashboard", layout="wide")
cache.enable()

RECENT_DAYS = 14
HIT_EVENTS = {"single", "double", "triple", "home_run"}
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
WALK_EVENTS = {"walk", "intent_walk"}
SWING_DESCRIPTIONS = {
    "hit_into_play",
    "foul",
    "foul_tip",
    "foul_bunt",
    "missed_bunt",
    "swinging_strike",
    "swinging_strike_blocked",
}
CONTACT_DESCRIPTIONS = {"hit_into_play", "foul", "foul_tip", "foul_bunt"}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}


def safe_divide(numerator, denominator, default=np.nan):
    """Divide while avoiding inf and zero-denominator errors."""
    result = np.divide(
        pd.to_numeric(numerator, errors="coerce"),
        pd.to_numeric(denominator, errors="coerce"),
    )
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan).fillna(default)
    return default if not np.isfinite(result) else result


def percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1 or numeric.nunique(dropna=True) <= 1:
        return pd.Series(50.0, index=series.index)
    ranked = numeric.rank(pct=True, method="average") * 100
    if not higher_is_better:
        ranked = 100 - ranked
    return ranked.fillna(50.0)


def shrink_rate(
    numerator: pd.Series,
    denominator: pd.Series,
    league_rate: float,
    prior_sample: float,
) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    den = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    return (num + league_rate * prior_sample) / (den + prior_sample)


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(character.lower() for character in str(value) if character.isalnum())


@st.cache_data(ttl=21600, show_spinner=False)
def load_statcast(start_date: str, end_date: str) -> pd.DataFrame:
    return statcast(
        start_dt=start_date,
        end_dt=end_date,
        verbose=False,
        parallel=False,
    )


@st.cache_data(ttl=86400, show_spinner=False)
def lookup_names(player_ids: tuple[int, ...]) -> pd.DataFrame:
    if not player_ids:
        return pd.DataFrame(columns=["player_id", "Player"])
    try:
        lookup = playerid_reverse_lookup(list(player_ids), key_type="mlbam")
    except Exception:
        return pd.DataFrame(columns=["player_id", "Player"])
    if lookup.empty:
        return pd.DataFrame(columns=["player_id", "Player"])
    lookup["Player"] = (
        lookup["name_first"].fillna("").str.title()
        + " "
        + lookup["name_last"].fillna("").str.title()
    ).str.strip()
    return lookup.rename(columns={"key_mlbam": "player_id"})[["player_id", "Player"]]


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    expected_columns = [
        "batter",
        "pitcher",
        "game_pk",
        "at_bat_number",
        "game_date",
        "player_name",
        "home_team",
        "away_team",
        "inning_topbot",
        "description",
        "events",
        "bb_type",
        "launch_speed",
        "launch_angle",
        "launch_speed_angle",
        "estimated_ba_using_speedangle",
        "estimated_slg_using_speedangle",
        "zone",
        "plate_x",
        "plate_z",
        "pitch_name",
        "release_speed",
        "pfx_x",
        "pfx_z",
        "release_extension",
        "stand",
        "p_throws",
        "hc_x",
        "hc_y",
    ]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = np.nan

    numeric_columns = [
        "batter",
        "pitcher",
        "game_pk",
        "at_bat_number",
        "launch_speed",
        "launch_angle",
        "launch_speed_angle",
        "estimated_ba_using_speedangle",
        "estimated_slg_using_speedangle",
        "zone",
        "plate_x",
        "plate_z",
        "release_speed",
        "pfx_x",
        "pfx_z",
        "release_extension",
        "hc_x",
        "hc_y",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["batter_team"] = np.where(
        df["inning_topbot"].eq("Top").fillna(False),
        df["away_team"],
        df["home_team"],
    )
    df["pitcher_team"] = np.where(
        df["inning_topbot"].eq("Top").fillna(False),
        df["home_team"],
        df["away_team"],
    )

    description = df["description"].fillna("").astype(str)
    events = df["events"].fillna("").astype(str)
    zone_numeric = pd.to_numeric(df["zone"], errors="coerce")

    df["is_swing"] = description.isin(SWING_DESCRIPTIONS).fillna(False)
    df["is_contact"] = description.isin(CONTACT_DESCRIPTIONS).fillna(False)
    df["is_whiff"] = description.isin(WHIFF_DESCRIPTIONS).fillna(False)
    df["is_zone"] = zone_numeric.between(1, 9, inclusive="both").fillna(False)
    df["is_zone_swing"] = df["is_zone"] & df["is_swing"]
    df["is_zone_contact"] = df["is_zone"] & df["is_contact"]
    df["is_pa_end"] = events.ne("")
    df["is_hit"] = events.isin(HIT_EVENTS)
    df["is_hr"] = events.eq("home_run")
    df["is_k"] = events.isin(STRIKEOUT_EVENTS)
    df["is_walk"] = events.isin(WALK_EVENTS)
    df["is_bbe"] = df["launch_speed"].notna() & df["launch_angle"].notna()
    df["is_barrel"] = df["launch_speed_angle"].eq(6).fillna(False)
    df["is_hard_hit"] = df["launch_speed"].ge(95).fillna(False)
    df["is_sweet_spot"] = df["launch_angle"].between(8, 32, inclusive="both").fillna(False)
    df["is_line_drive"] = df["bb_type"].eq("line_drive").fillna(False)
    df["is_fly_ball"] = df["bb_type"].eq("fly_ball").fillna(False)
    df["is_ground_ball"] = df["bb_type"].eq("ground_ball").fillna(False)

    df["pa_key"] = (
        df["game_pk"].astype("Int64").astype(str)
        + "-"
        + df["at_bat_number"].astype("Int64").astype(str)
    )

    df["xba_value"] = df["estimated_ba_using_speedangle"].where(
        df["is_bbe"], 0.0
    )
    missing_bbe_xba = df["is_bbe"] & df["xba_value"].isna()
    df.loc[missing_bbe_xba, "xba_value"] = df.loc[
        missing_bbe_xba, "is_hit"
    ].astype(float)
    df["xba_value"] = df["xba_value"].fillna(0.0)

    zone_text = zone_numeric.round().astype("Int64").astype(str)
    zone_mask = zone_numeric.between(1, 9, inclusive="both").fillna(False).to_numpy(bool)
    df["zone_group"] = np.where(zone_mask, zone_text, "Chase")

    df["speed_band"] = (df["release_speed"] / 2.0).round() * 2.0
    df["mov_x_band"] = (df["pfx_x"] / 0.25).round() * 0.25
    df["mov_z_band"] = (df["pfx_z"] / 0.25).round() * 0.25
    return df


def most_common(series: pd.Series, default: str = "") -> str:
    mode = series.dropna().astype(str).mode()
    return default if mode.empty else str(mode.iloc[0])


def aggregate_hitters(df: pd.DataFrame, pitcher_hand: str, min_pa: int) -> pd.DataFrame:
    pa = df[df["is_pa_end"]].copy()
    pitch = df.copy()
    bbe = df[df["is_bbe"]].copy()

    pa_stats = (
        pa.groupby(["batter", "batter_team"], dropna=False)
        .agg(
            PA=("pa_key", "nunique"),
            Hits=("is_hit", "sum"),
            HR=("is_hr", "sum"),
            Strikeouts=("is_k", "sum"),
            Walks=("is_walk", "sum"),
            xHits=("xba_value", "sum"),
            Stand=("stand", lambda x: most_common(x, "?")),
        )
        .reset_index()
        .rename(columns={"batter": "player_id", "batter_team": "Team"})
    )

    pitch_stats = (
        pitch.groupby(["batter", "batter_team"], dropna=False)
        .agg(
            Pitches=("batter", "size"),
            Swings=("is_swing", "sum"),
            Contacts=("is_contact", "sum"),
            Whiffs=("is_whiff", "sum"),
            Zone_Swings=("is_zone_swing", "sum"),
            Zone_Contacts=("is_zone_contact", "sum"),
        )
        .reset_index()
        .rename(columns={"batter": "player_id", "batter_team": "Team"})
    )

    bbe_stats = (
        bbe.groupby(["batter", "batter_team"], dropna=False)
        .agg(
            BBE=("is_bbe", "sum"),
            Avg_xBA_Contact=("estimated_ba_using_speedangle", "mean"),
            Line_Drives=("is_line_drive", "sum"),
            Hard_Hits=("is_hard_hit", "sum"),
            Sweet_Spots=("is_sweet_spot", "sum"),
            Avg_EV=("launch_speed", "mean"),
        )
        .reset_index()
        .rename(columns={"batter": "player_id", "batter_team": "Team"})
    )

    board = pa_stats.merge(pitch_stats, on=["player_id", "Team"], how="left")
    board = board.merge(bbe_stats, on=["player_id", "Team"], how="left")
    board = board[board["PA"].ge(min_pa)].copy()
    if board.empty:
        return board

    board["Hit_PA"] = safe_divide(board["Hits"], board["PA"])
    board["xHit_PA"] = safe_divide(board["xHits"], board["PA"])
    board["K_Pct"] = safe_divide(board["Strikeouts"], board["PA"])
    board["BB_Pct"] = safe_divide(board["Walks"], board["PA"])
    board["Contact_Pct"] = safe_divide(board["Contacts"], board["Swings"])
    board["Whiff_Pct"] = safe_divide(board["Whiffs"], board["Swings"])
    board["Zone_Contact_Pct"] = safe_divide(
        board["Zone_Contacts"], board["Zone_Swings"]
    )
    board["LD_Pct"] = safe_divide(board["Line_Drives"], board["BBE"])
    board["HH_Pct"] = safe_divide(board["Hard_Hits"], board["BBE"])
    board["SweetSpot_Pct"] = safe_divide(board["Sweet_Spots"], board["BBE"])

    split_df = df[df["p_throws"].eq(pitcher_hand)].copy()
    split_pa = split_df[split_df["is_pa_end"]]
    split_stats = (
        split_pa.groupby("batter")
        .agg(
            Platoon_PA=("pa_key", "nunique"),
            Platoon_H=("is_hit", "sum"),
            Platoon_xH=("xba_value", "sum"),
            Platoon_K=("is_k", "sum"),
        )
        .reset_index()
        .rename(columns={"batter": "player_id"})
    )
    board = board.merge(split_stats, on="player_id", how="left")
    for column in ["Platoon_PA", "Platoon_H", "Platoon_xH", "Platoon_K"]:
        board[column] = pd.to_numeric(board[column], errors="coerce").fillna(0)
    return board


def recent_hit_form(df: pd.DataFrame, player_ids: Iterable[int], end_date: pd.Timestamp) -> pd.DataFrame:
    start = end_date - pd.Timedelta(days=RECENT_DAYS - 1)
    recent = df[
        df["batter"].isin(list(player_ids))
        & df["game_date"].between(start, end_date)
    ].copy()
    pa = recent[recent["is_pa_end"]]
    result = (
        pa.groupby("batter")
        .agg(
            Recent_PA=("pa_key", "nunique"),
            Recent_H=("is_hit", "sum"),
            Recent_xH=("xba_value", "sum"),
            Recent_K=("is_k", "sum"),
        )
        .reset_index()
        .rename(columns={"batter": "player_id"})
    )
    return result


def pitcher_hit_splits(df: pd.DataFrame, pitcher_id: int) -> pd.DataFrame:
    rows = df[df["pitcher"].eq(pitcher_id) & df["is_pa_end"]].copy()
    if rows.empty:
        return pd.DataFrame(columns=["Stand"])
    return (
        rows.groupby("stand")
        .agg(
            Pitcher_PA=("pa_key", "nunique"),
            Pitcher_H=("is_hit", "sum"),
            Pitcher_xH=("xba_value", "sum"),
            Pitcher_K=("is_k", "sum"),
        )
        .reset_index()
        .rename(columns={"stand": "Stand"})
    )


def _attack_read(score: float) -> str:
    if score >= 60:
        return "ATTACK"
    if score >= 54:
        return "Lean attack"
    if score > 46:
        return "Neutral"
    if score > 40:
        return "Lean avoid"
    return "AVOID"


def pitcher_attack_profile_hits(df: pd.DataFrame, pitcher_id: int) -> pd.DataFrame:
    """Grade how attackable the selected pitcher has been to LHB and RHB for hits."""
    pa_all = df[df["is_pa_end"]].copy()
    selected_pa = pa_all[pa_all["pitcher"].eq(pitcher_id)].copy()
    bbe_all = df[df["is_bbe"]].copy()
    selected_bbe = bbe_all[bbe_all["pitcher"].eq(pitcher_id)].copy()

    rows = []
    for side, label in [("L", "LHB"), ("R", "RHB")]:
        league_pa = pa_all[pa_all["stand"].eq(side)]
        pitcher_pa = selected_pa[selected_pa["stand"].eq(side)]
        league_bbe = bbe_all[bbe_all["stand"].eq(side)]
        pitcher_bbe = selected_bbe[selected_bbe["stand"].eq(side)]

        lg_pa_n = max(int(league_pa["pa_key"].nunique()), 1)
        p_pa_n = int(pitcher_pa["pa_key"].nunique())
        lg_bbe_n = max(len(league_bbe), 1)
        p_bbe_n = len(pitcher_bbe)

        lg_hit = float(league_pa["is_hit"].sum() / lg_pa_n)
        lg_xhit = float(league_pa["xba_value"].sum() / lg_pa_n)
        lg_k = float(league_pa["is_k"].sum() / lg_pa_n)
        lg_hh = float(league_bbe["is_hard_hit"].sum() / lg_bbe_n)

        hit_rate = (float(pitcher_pa["is_hit"].sum()) + lg_hit * 140) / (p_pa_n + 140)
        xhit_rate = (float(pitcher_pa["xba_value"].sum()) + lg_xhit * 140) / (p_pa_n + 140)
        k_rate = (float(pitcher_pa["is_k"].sum()) + lg_k * 140) / (p_pa_n + 140)
        hh_rate = (float(pitcher_bbe["is_hard_hit"].sum()) + lg_hh * 90) / (p_bbe_n + 90)

        vulnerability_ratio = (
            0.34 * (hit_rate / max(lg_hit, 0.0001))
            + 0.34 * (xhit_rate / max(lg_xhit, 0.0001))
            + 0.17 * (hh_rate / max(lg_hh, 0.0001))
            + 0.15 * (lg_k / max(k_rate, 0.0001))
        )
        attack_score = float(np.clip(50 + 90 * (vulnerability_ratio - 1.0), 0, 100))
        confidence_score = float(100 * (1 - np.exp(-p_pa_n / 150.0)))
        confidence = "High" if confidence_score >= 70 else "Medium" if confidence_score >= 45 else "Low"

        rows.append(
            {
                "Side": side,
                "HitterSide": label,
                "PitcherSideAttackScore": attack_score,
                "PitcherSideRead": _attack_read(attack_score),
                "PitcherSidePA": p_pa_n,
                "Allowed_H_PA": hit_rate,
                "Allowed_xH_PA": xhit_rate,
                "Allowed_K_Pct": k_rate,
                "Allowed_HH_Pct": hh_rate,
                "AttackConfidence": confidence,
            }
        )
    return pd.DataFrame(rows)


def render_pitcher_attack_panel(profile: pd.DataFrame, pitcher_name: str, event_label: str) -> None:
    st.subheader("Pitcher attack / avoid map")
    if profile.empty:
        st.info("Not enough pitcher data was available to build the handedness map.")
        return

    best = profile.sort_values("PitcherSideAttackScore", ascending=False).iloc[0]
    lhb = profile[profile["Side"].eq("L")].iloc[0]
    rhb = profile[profile["Side"].eq("R")].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best side to target", best["HitterSide"], best["PitcherSideRead"])
    c2.metric("LHB read", f"{lhb['PitcherSideRead']} · {lhb['PitcherSideAttackScore']:.0f}")
    c3.metric("RHB read", f"{rhb['PitcherSideRead']} · {rhb['PitcherSideAttackScore']:.0f}")
    st.caption(
        f"The score compares {pitcher_name}'s handedness splits with league rates for {event_label}. "
        "Low-sample splits are pulled strongly toward neutral."
    )
    with st.expander("Handedness split detail", expanded=False):
        detail = profile[
            [
                "HitterSide", "PitcherSideRead", "PitcherSideAttackScore", "PitcherSidePA",
                "Allowed_H_PA", "Allowed_xH_PA", "Allowed_K_Pct", "Allowed_HH_Pct",
                "AttackConfidence",
            ]
        ].rename(
            columns={
                "HitterSide": "Batters", "PitcherSideRead": "Read",
                "PitcherSideAttackScore": "Attack score", "PitcherSidePA": "PA",
                "Allowed_H_PA": "H/PA allowed", "Allowed_xH_PA": "xHit/PA allowed",
                "Allowed_K_Pct": "K%", "Allowed_HH_Pct": "Hard-hit%",
                "AttackConfidence": "Confidence",
            }
        )
        st.dataframe(
            detail.style.background_gradient(cmap="RdYlGn", subset=["Attack score"], vmin=0, vmax=100).format(
                {"Attack score": "{:.1f}", "H/PA allowed": "{:.1%}", "xHit/PA allowed": "{:.1%}", "K%": "{:.1%}", "Hard-hit%": "{:.1%}"}
            ),
            hide_index=True,
            use_container_width=True,
        )


def selected_pitcher_profile(df: pd.DataFrame, pitcher_id: int) -> tuple[pd.DataFrame, str]:
    rows = df[df["pitcher"].eq(pitcher_id) & df["pitch_name"].notna()].copy()
    if rows.empty:
        return pd.DataFrame(), ""
    hand = most_common(rows["p_throws"], "")
    profile = (
        rows.groupby("pitch_name")
        .agg(
            Pitches=("pitch_name", "size"),
            Avg_Speed=("release_speed", "mean"),
            Avg_PFX_X=("pfx_x", "mean"),
            Avg_PFX_Z=("pfx_z", "mean"),
            Avg_Extension=("release_extension", "mean"),
        )
        .reset_index()
    )
    profile["Usage"] = profile["Pitches"] / profile["Pitches"].sum()
    return profile[profile["Usage"].ge(0.03)].copy(), hand


def _similarity_weights(rows: pd.DataFrame, profile_row: pd.Series) -> np.ndarray:
    speed = pd.to_numeric(rows["release_speed"], errors="coerce")
    pfx_x = pd.to_numeric(rows["pfx_x"], errors="coerce")
    pfx_z = pd.to_numeric(rows["pfx_z"], errors="coerce")
    extension = pd.to_numeric(rows["release_extension"], errors="coerce")

    distance = np.zeros(len(rows), dtype=float)
    count = np.zeros(len(rows), dtype=float)
    comparisons = [
        (speed, profile_row.get("Avg_Speed"), 3.0),
        (pfx_x, profile_row.get("Avg_PFX_X"), 0.45),
        (pfx_z, profile_row.get("Avg_PFX_Z"), 0.45),
        (extension, profile_row.get("Avg_Extension"), 0.7),
    ]
    for series, center, scale in comparisons:
        if pd.isna(center):
            continue
        valid = series.notna().to_numpy()
        values = series.fillna(center).to_numpy(dtype=float)
        distance += np.where(valid, ((values - float(center)) / scale) ** 2, 0.0)
        count += valid.astype(float)
    distance = np.divide(distance, np.maximum(count, 1.0))
    return np.exp(-0.5 * distance)


def hit_pitch_shape_match(
    df: pd.DataFrame,
    player_ids: Iterable[int],
    profile: pd.DataFrame,
    pitcher_hand: str,
) -> pd.DataFrame:
    player_ids = [int(value) for value in player_ids]
    if profile.empty or not player_ids:
        return pd.DataFrame(columns=["player_id", "PitchMatchRatio", "MatchSample"])

    hand_rows = df[df["p_throws"].eq(pitcher_hand)].copy()
    league_rows = hand_rows[hand_rows["pitch_name"].isin(profile["pitch_name"])]

    league_baselines = {}
    for _, pitch_row in profile.iterrows():
        pitch_name = pitch_row["pitch_name"]
        sample = league_rows[league_rows["pitch_name"].eq(pitch_name)].copy()
        if sample.empty:
            continue
        weights = _similarity_weights(sample, pitch_row)
        swing_den = np.sum(weights * sample["is_swing"].astype(float).to_numpy())
        contact_num = np.sum(weights * sample["is_contact"].astype(float).to_numpy())
        bbe_mask = sample["is_bbe"].astype(float).to_numpy()
        bbe_den = np.sum(weights * bbe_mask)
        xba_num = np.sum(weights * bbe_mask * sample["xba_value"].to_numpy(float))
        league_baselines[pitch_name] = {
            "contact": contact_num / swing_den if swing_den > 0 else np.nan,
            "xba": xba_num / bbe_den if bbe_den > 0 else np.nan,
        }

    output = []
    for player_id in player_ids:
        player_rows = hand_rows[hand_rows["batter"].eq(player_id)]
        weighted_ratios = []
        usages = []
        effective_sample = 0.0
        for _, pitch_row in profile.iterrows():
            pitch_name = pitch_row["pitch_name"]
            sample = player_rows[player_rows["pitch_name"].eq(pitch_name)].copy()
            baseline = league_baselines.get(pitch_name)
            if sample.empty or not baseline:
                continue
            weights = _similarity_weights(sample, pitch_row)
            swing_den = np.sum(weights * sample["is_swing"].astype(float).to_numpy())
            contact_num = np.sum(weights * sample["is_contact"].astype(float).to_numpy())
            bbe_mask = sample["is_bbe"].astype(float).to_numpy()
            bbe_den = np.sum(weights * bbe_mask)
            xba_num = np.sum(weights * bbe_mask * sample["xba_value"].to_numpy(float))
            contact_rate = contact_num / swing_den if swing_den > 0 else np.nan
            xba_rate = xba_num / bbe_den if bbe_den > 0 else np.nan
            contact_ratio = (
                contact_rate / baseline["contact"]
                if pd.notna(contact_rate)
                and pd.notna(baseline["contact"])
                and baseline["contact"] != 0
                else 1.0
            )
            xba_ratio = (
                xba_rate / baseline["xba"]
                if pd.notna(xba_rate)
                and pd.notna(baseline["xba"])
                and baseline["xba"] != 0
                else 1.0
            )
            ratio = np.clip(0.45 * contact_ratio + 0.55 * xba_ratio, 0.65, 1.40)
            reliability = min(1.0, (swing_den + bbe_den * 2) / 35.0)
            ratio = 1.0 + reliability * (ratio - 1.0)
            weighted_ratios.append(ratio)
            usages.append(float(pitch_row["Usage"]))
            effective_sample += swing_den + bbe_den * 2
        if weighted_ratios:
            matchup = float(np.average(weighted_ratios, weights=usages))
        else:
            matchup = 1.0
        output.append(
            {
                "player_id": player_id,
                "PitchMatchRatio": matchup,
                "MatchSample": effective_sample,
            }
        )
    return pd.DataFrame(output)


def hit_zone_fit(
    df: pd.DataFrame,
    player_ids: Iterable[int],
    pitcher_id: int,
    pitcher_hand: str,
) -> pd.DataFrame:
    pitcher_rows = df[df["pitcher"].eq(pitcher_id)]
    zone_usage = pitcher_rows.groupby("zone_group").size().rename("Pitches").reset_index()
    if zone_usage.empty:
        return pd.DataFrame(columns=["player_id", "ZoneFitRatio"])
    zone_usage["Usage"] = zone_usage["Pitches"] / zone_usage["Pitches"].sum()

    hand_rows = df[df["p_throws"].eq(pitcher_hand)]
    league = (
        hand_rows.groupby("zone_group")
        .agg(
            League_Swings=("is_swing", "sum"),
            League_Contacts=("is_contact", "sum"),
            League_BBE=("is_bbe", "sum"),
            League_xH=("xba_value", "sum"),
        )
        .reset_index()
    )
    league["LeagueContact"] = safe_divide(league["League_Contacts"], league["League_Swings"], 0.75)
    league["League_xBA"] = safe_divide(league["League_xH"], league["League_BBE"], 0.30)

    output = []
    for player_id in player_ids:
        rows = hand_rows[hand_rows["batter"].eq(player_id)]
        grouped = (
            rows.groupby("zone_group")
            .agg(
                Swings=("is_swing", "sum"),
                Contacts=("is_contact", "sum"),
                BBE=("is_bbe", "sum"),
                xH=("xba_value", "sum"),
            )
            .reset_index()
            .merge(league, on="zone_group", how="outer")
            .merge(zone_usage[["zone_group", "Usage"]], on="zone_group", how="inner")
        )
        if grouped.empty:
            output.append({"player_id": player_id, "ZoneFitRatio": 1.0})
            continue
        grouped[["Swings", "Contacts", "BBE", "xH"]] = grouped[
            ["Swings", "Contacts", "BBE", "xH"]
        ].fillna(0)
        contact = (grouped["Contacts"] + grouped["LeagueContact"] * 20) / (
            grouped["Swings"] + 20
        )
        xba = (grouped["xH"] + grouped["League_xBA"] * 10) / (grouped["BBE"] + 10)
        ratio = 0.4 * safe_divide(contact, grouped["LeagueContact"], 1.0) + 0.6 * safe_divide(
            xba, grouped["League_xBA"], 1.0
        )
        ratio = ratio.clip(0.70, 1.35)
        output.append(
            {
                "player_id": player_id,
                "ZoneFitRatio": float(np.average(ratio, weights=grouped["Usage"])),
            }
        )
    return pd.DataFrame(output)


def bvp_hit_stats(df: pd.DataFrame, player_ids: Iterable[int], pitcher_id: int) -> pd.DataFrame:
    ids = list(dict.fromkeys(int(player_id) for player_id in player_ids if pd.notna(player_id)))
    base = pd.DataFrame({"player_id": ids})
    defaults = {
        "BvP_PA": 0,
        "BvP_H": 0,
        "BvP_xH": 0.0,
        "BvP_K": 0,
        "BvP_BBE": 0,
        "BvP_Hard_Hits": 0,
        "BvP_Avg_EV": np.nan,
        "BvP_Hit_PA": 0.0,
        "BvP_xHit_PA": 0.0,
        "BvP_K_Pct": 0.0,
        "BvP_HH_Pct": 0.0,
        "BvPScore": 50.0,
        "BvP_Last_Date": pd.NaT,
    }
    rows = df[df["pitcher"].eq(pitcher_id) & df["batter"].isin(ids)].copy()
    if rows.empty:
        for column, value in defaults.items():
            base[column] = value
        return base

    pa = rows[rows["is_pa_end"]]
    if pa.empty:
        for column, value in defaults.items():
            base[column] = value
        return base

    result = (
        pa.groupby("batter")
        .agg(
            BvP_PA=("pa_key", "nunique"),
            BvP_H=("is_hit", "sum"),
            BvP_xH=("xba_value", "sum"),
            BvP_K=("is_k", "sum"),
            BvP_Last_Date=("game_date", "max"),
        )
        .reset_index()
        .rename(columns={"batter": "player_id"})
    )

    bbe = rows[rows["is_bbe"]]
    if not bbe.empty:
        batted = (
            bbe.groupby("batter")
            .agg(
                BvP_BBE=("is_bbe", "sum"),
                BvP_Hard_Hits=("is_hard_hit", "sum"),
                BvP_Avg_EV=("launch_speed", "mean"),
            )
            .reset_index()
            .rename(columns={"batter": "player_id"})
        )
        result = result.merge(batted, on="player_id", how="left")

    for column in ["BvP_PA", "BvP_H", "BvP_xH", "BvP_K", "BvP_BBE", "BvP_Hard_Hits"]:
        result[column] = pd.to_numeric(result.get(column, 0), errors="coerce").fillna(0)

    result["BvP_Hit_PA"] = safe_divide(result["BvP_H"], result["BvP_PA"], 0.0)
    result["BvP_xHit_PA"] = safe_divide(result["BvP_xH"], result["BvP_PA"], 0.0)
    result["BvP_K_Pct"] = safe_divide(result["BvP_K"], result["BvP_PA"], 0.0)
    result["BvP_HH_Pct"] = safe_divide(result["BvP_Hard_Hits"], result["BvP_BBE"], 0.0)

    raw = 0.5 * result["BvP_Hit_PA"] + 0.5 * result["BvP_xHit_PA"]
    performance = percentile(raw)
    reliability = np.minimum(result["BvP_PA"] / 20.0, 1.0)
    result["BvPScore"] = 50 + reliability * (performance - 50)

    output = base.merge(result, on="player_id", how="left")
    for column, value in defaults.items():
        if column not in output.columns:
            output[column] = value
        elif column != "BvP_Last_Date":
            output[column] = output[column].fillna(value)
    return output


def infer_recent_lineup(df: pd.DataFrame, team: str) -> pd.DataFrame:
    team_rows = df[df["batter_team"].eq(team) & df["game_date"].notna()].copy()
    if team_rows.empty:
        return pd.DataFrame(columns=["player_id", "LineupSpot"])
    latest_date = team_rows["game_date"].max()
    date_rows = team_rows[team_rows["game_date"].eq(latest_date)]
    latest_game = pd.to_numeric(date_rows["game_pk"], errors="coerce").max()
    game_rows = date_rows[date_rows["game_pk"].eq(latest_game)]
    order = (
        game_rows.groupby("batter")["at_bat_number"]
        .min()
        .sort_values()
        .reset_index()
        .rename(columns={"batter": "player_id"})
    )
    order["LineupSpot"] = np.arange(1, len(order) + 1)
    order.loc[order["LineupSpot"].gt(9), "LineupSpot"] = np.nan
    return order[["player_id", "LineupSpot"]]


def add_roster_candidates(
    stats_board: pd.DataFrame,
    team: str,
    active_roster: pd.DataFrame | None,
    recent_lineup: pd.DataFrame,
    min_pa: int,
    include_low_sample: bool,
) -> pd.DataFrame:
    """Union the active roster with the latest observed lineup and Statcast stats."""
    stats = stats_board[stats_board["Team"].eq(team)].copy()
    stats["player_id"] = pd.to_numeric(stats["player_id"], errors="coerce").astype("Int64")
    stats = stats.dropna(subset=["player_id"])
    stats["player_id"] = stats["player_id"].astype(int)

    lineup_ids = set(
        pd.to_numeric(recent_lineup.get("player_id", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )

    roster = active_roster.copy() if active_roster is not None else pd.DataFrame()
    if not roster.empty:
        roster["player_id"] = pd.to_numeric(roster["player_id"], errors="coerce").astype("Int64")
        roster = roster.dropna(subset=["player_id"]).copy()
        roster["player_id"] = roster["player_id"].astype(int)
        roster_ids = set(roster["player_id"].tolist())

        # A same-day transaction can briefly appear in a lineup before the roster feed catches up.
        missing_lineup_ids = sorted(lineup_ids - roster_ids)
        if missing_lineup_ids:
            roster = pd.concat(
                [
                    roster,
                    pd.DataFrame(
                        {
                            "player_id": missing_lineup_ids,
                            "Player": np.nan,
                            "Bats": np.nan,
                            "Position": "",
                            "RosterStatus": "Latest observed lineup",
                        }
                    ),
                ],
                ignore_index=True,
            )

        board = roster.merge(stats, on="player_id", how="left", suffixes=("", "_Stat"))
        board["ActiveRoster"] = board["RosterStatus"].eq("Active") | board["RosterStatus"].str.contains(
            "active", case=False, na=False
        )
        board["Team"] = team
        if "Stand" not in board:
            board["Stand"] = np.nan
        roster_bats = board["Bats"].where(board["Bats"].isin(["L", "R", "S"]))
        board["Stand"] = roster_bats.fillna(board["Stand"])
    else:
        board = stats.copy()
        board["Player"] = np.nan
        board["Bats"] = board.get("Stand", pd.Series(index=board.index, dtype=object))
        board["Position"] = ""
        board["RosterStatus"] = "Statcast sample"
        board["ActiveRoster"] = False

    board["PA"] = pd.to_numeric(board.get("PA", 0), errors="coerce").fillna(0)
    board["SampleStatus"] = np.where(
        board["PA"].ge(min_pa),
        "Qualified",
        "Low sample",
    )
    if not include_low_sample:
        board = board[board["PA"].ge(min_pa)].copy()
    return board


def projected_pa(lineup_spot: pd.Series, team_runs: float, is_away: bool) -> pd.Series:
    mapping = {1: 4.72, 2: 4.60, 3: 4.49, 4: 4.38, 5: 4.26, 6: 4.15, 7: 4.05, 8: 3.94, 9: 3.84}
    base = pd.to_numeric(lineup_spot, errors="coerce").round().map(mapping).fillna(4.15)
    base += 0.10 * (team_runs - 4.5)
    base += 0.06 if is_away else -0.02
    return base.clip(3.2, 5.4)


def parse_sprint_upload(uploaded_file, board: pd.DataFrame) -> pd.DataFrame:
    board = board.copy()
    board["Sprint_Speed"] = np.nan
    if uploaded_file is None:
        return board
    try:
        supplemental = pd.read_csv(uploaded_file)
    except Exception as error:
        st.warning(f"Could not read sprint-speed CSV: {error}")
        return board

    lower = {str(column).lower().strip(): column for column in supplemental.columns}
    speed_column = next(
        (lower[key] for key in ["sprint_speed", "sprint speed", "sprint_speed_ft_s"] if key in lower),
        None,
    )
    if speed_column is None:
        st.warning("Sprint CSV needs a sprint_speed column.")
        return board

    id_column = next((lower[key] for key in ["player_id", "mlbam_id", "id"] if key in lower), None)
    name_column = next((lower[key] for key in ["player", "player_name", "name"] if key in lower), None)
    supplemental["_speed"] = pd.to_numeric(supplemental[speed_column], errors="coerce")

    if id_column is not None:
        supplemental["player_id"] = pd.to_numeric(supplemental[id_column], errors="coerce")
        merged = board.drop(columns=["Sprint_Speed"]).merge(
            supplemental[["player_id", "_speed"]].dropna(subset=["player_id"]).drop_duplicates("player_id"),
            on="player_id",
            how="left",
        )
    elif name_column is not None:
        supplemental["_name_key"] = supplemental[name_column].map(normalize_name)
        board["_name_key"] = board["Player"].map(normalize_name)
        merged = board.drop(columns=["Sprint_Speed"]).merge(
            supplemental[["_name_key", "_speed"]].drop_duplicates("_name_key"),
            on="_name_key",
            how="left",
        ).drop(columns=["_name_key"])
    else:
        st.warning("Sprint CSV needs player_id or Player/player_name.")
        return board
    return merged.rename(columns={"_speed": "Sprint_Speed"})


def build_hit_board(
    df: pd.DataFrame,
    pitcher_id: int,
    team: str,
    min_pa: int,
    end_date: pd.Timestamp,
    park_hit_factor_lhb: float,
    park_hit_factor_rhb: float,
    team_runs: float,
    is_away: bool,
    starter_innings: float,
    bullpen_multiplier: float,
    lineup_override: pd.DataFrame | None,
    lineup_edits: pd.DataFrame | None,
    sprint_upload,
    active_roster: pd.DataFrame | None,
    include_low_sample: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile, pitcher_hand = selected_pitcher_profile(df, pitcher_id)
    if not pitcher_hand:
        return pd.DataFrame(), profile

    recent_lineup = _clean_lineup_seed(lineup_override)
    if recent_lineup.empty:
        recent_lineup = infer_recent_lineup(df, team)
    board = aggregate_hitters(df, pitcher_hand, 0)
    board = add_roster_candidates(
        board, team, active_roster, recent_lineup, min_pa, include_low_sample
    )
    if board.empty:
        return board, profile

    names = lookup_names(tuple(board["player_id"].dropna().astype(int).unique().tolist()))
    board = board.merge(names, on="player_id", how="left", suffixes=("", "_Lookup"))
    board["Player"] = board["Player"].replace("", np.nan).fillna(board.get("Player_Lookup"))
    board["Player"] = board["Player"].fillna("MLB ID " + board["player_id"].astype("Int64").astype(str))
    board = board.drop(columns=["Player_Lookup"], errors="ignore")

    zero_columns = [
        "PA", "Hits", "HR", "Strikeouts", "Walks", "xHits", "Pitches", "Swings",
        "Contacts", "Whiffs", "Zone_Swings", "Zone_Contacts", "BBE", "Line_Drives",
        "Hard_Hits", "Sweet_Spots", "Platoon_PA", "Platoon_H", "Platoon_xH", "Platoon_K",
    ]
    for column in zero_columns:
        board[column] = pd.to_numeric(board.get(column, 0), errors="coerce").fillna(0)

    board["EffectiveStand"] = np.where(
        board["Stand"].eq("S"),
        "L" if pitcher_hand == "R" else "R",
        board["Stand"],
    )
    board["EffectiveStand"] = board["EffectiveStand"].where(
        board["EffectiveStand"].isin(["L", "R"]), "R"
    )

    recent = recent_hit_form(df, board["player_id"], end_date)
    board = board.merge(recent, on="player_id", how="left")
    for column in ["Recent_PA", "Recent_H", "Recent_xH", "Recent_K"]:
        board[column] = pd.to_numeric(board[column], errors="coerce").fillna(0)

    pitcher_splits = pitcher_hit_splits(df, pitcher_id)
    board = board.merge(
        pitcher_splits, left_on="EffectiveStand", right_on="Stand", how="left", suffixes=("", "_Pitcher")
    )
    for column in ["Pitcher_PA", "Pitcher_H", "Pitcher_xH", "Pitcher_K"]:
        board[column] = pd.to_numeric(board[column], errors="coerce").fillna(0)

    side_attack = pitcher_attack_profile_hits(df, pitcher_id)
    board = board.merge(
        side_attack[["Side", "PitcherSideAttackScore", "PitcherSideRead", "AttackConfidence"]],
        left_on="EffectiveStand", right_on="Side", how="left"
    )
    board["PitcherSideAttackScore"] = board["PitcherSideAttackScore"].fillna(50.0)
    board["PitcherSideRead"] = board["PitcherSideRead"].fillna("Neutral")

    board = board.merge(
        hit_pitch_shape_match(df, board["player_id"], profile, pitcher_hand),
        on="player_id",
        how="left",
    )
    board = board.merge(
        hit_zone_fit(df, board["player_id"], pitcher_id, pitcher_hand),
        on="player_id",
        how="left",
    )
    board = board.merge(
        bvp_hit_stats(df, board["player_id"], pitcher_id),
        on="player_id",
        how="left",
    )

    board["PitchMatchRatio"] = board["PitchMatchRatio"].fillna(1.0)
    board["ZoneFitRatio"] = board["ZoneFitRatio"].fillna(1.0)
    board["MatchSample"] = board["MatchSample"].fillna(0)

    inferred = recent_lineup
    board = board.merge(inferred, on="player_id", how="left")
    if lineup_edits is not None and not lineup_edits.empty:
        edits = lineup_edits[["player_id", "Selected", "LineupSpot"]].copy()
        board = board.drop(columns=["LineupSpot"], errors="ignore").merge(edits, on="player_id", how="left")
        board = board[board["Selected"].fillna(True)].copy()
    else:
        board["Selected"] = True

    board = parse_sprint_upload(sprint_upload, board)

    pa_all = df[df["is_pa_end"]]
    league_pa = max(int(pa_all["pa_key"].nunique()), 1)
    league_hit_rate = float(pa_all["is_hit"].sum() / league_pa)
    league_xhit_rate = float(pa_all["xba_value"].sum() / league_pa)

    board["Adj_Hit_PA"] = shrink_rate(board["Hits"], board["PA"], league_hit_rate, 100)
    board["Adj_xHit_PA"] = shrink_rate(board["xHits"], board["PA"], league_xhit_rate, 100)
    board["Adj_Platoon_Hit_PA"] = shrink_rate(
        board["Platoon_H"], board["Platoon_PA"], league_hit_rate, 60
    )
    board["Adj_Platoon_xHit_PA"] = shrink_rate(
        board["Platoon_xH"], board["Platoon_PA"], league_xhit_rate, 60
    )
    board["Pitcher_Allowed_Hit_PA"] = shrink_rate(
        board["Pitcher_H"], board["Pitcher_PA"], league_hit_rate, 150
    )
    board["Pitcher_Allowed_xHit_PA"] = shrink_rate(
        board["Pitcher_xH"], board["Pitcher_PA"], league_xhit_rate, 150
    )
    board["Recent_Hit_PA"] = shrink_rate(
        board["Recent_H"], board["Recent_PA"], league_hit_rate, 35
    )
    board["Recent_xHit_PA"] = shrink_rate(
        board["Recent_xH"], board["Recent_PA"], league_xhit_rate, 35
    )

    hitter_base = (
        board["Adj_Hit_PA"] * 0.24
        + board["Adj_xHit_PA"] * 0.24
        + board["Adj_Platoon_Hit_PA"] * 0.11
        + board["Adj_Platoon_xHit_PA"] * 0.11
        + board["Pitcher_Allowed_Hit_PA"] * 0.10
        + board["Pitcher_Allowed_xHit_PA"] * 0.10
        + league_hit_rate * board["PitchMatchRatio"] * 0.05
        + league_hit_rate * board["ZoneFitRatio"] * 0.05
    )
    recent_blend = 0.5 * board["Recent_Hit_PA"] + 0.5 * board["Recent_xHit_PA"]
    starter_rate = 0.88 * hitter_base + 0.12 * recent_blend

    starter_share = float(np.clip(starter_innings / 9.0, 0.35, 0.78))
    bullpen_rate = league_hit_rate * float(bullpen_multiplier)
    game_per_pa = starter_share * starter_rate + (1 - starter_share) * bullpen_rate
    board["ParkFactor"] = np.where(
        board["EffectiveStand"].eq("L"), float(park_hit_factor_lhb), float(park_hit_factor_rhb)
    )
    board["PrePark_Hit_Per_PA"] = game_per_pa
    game_per_pa = game_per_pa * board["ParkFactor"] / 100.0
    board["Model_Hit_Per_PA"] = game_per_pa.clip(0.04, 0.43)
    board["Projected_PA"] = projected_pa(board["LineupSpot"], team_runs, is_away)
    board["Model_1plus_Hit"] = 1 - (1 - board["Model_Hit_Per_PA"]) ** board["Projected_PA"]
    board["Projected_Hits"] = board["Model_Hit_Per_PA"] * board["Projected_PA"]

    board["PitchMatchScore"] = percentile(board["PitchMatchRatio"])
    board["ZoneFitScore"] = percentile(board["ZoneFitRatio"])
    board["PitcherHitScore"] = percentile(
        0.5 * board["Pitcher_Allowed_Hit_PA"] + 0.5 * board["Pitcher_Allowed_xHit_PA"]
    )
    board["RecentFormScore"] = percentile(recent_blend)
    board["SprintScore"] = percentile(board["Sprint_Speed"]).where(board["Sprint_Speed"].notna(), 50.0)

    board["HitScore"] = (
        percentile(board["Adj_xHit_PA"]) * 0.20
        + percentile(board["Contact_Pct"]) * 0.13
        + percentile(board["K_Pct"], higher_is_better=False) * 0.10
        + board["PitchMatchScore"] * 0.15
        + board["ZoneFitScore"] * 0.10
        + board["PitcherHitScore"] * 0.10
        + percentile(board["LD_Pct"]) * 0.06
        + percentile(board["SweetSpot_Pct"]) * 0.04
        + board["RecentFormScore"] * 0.07
        + board["SprintScore"] * 0.05
    )

    sample_conf = 100 * (1 - np.exp(-board["PA"] / 130.0))
    bbe_conf = 100 * (1 - np.exp(-board["BBE"].fillna(0) / 80.0))
    matchup_conf = 100 * (1 - np.exp(-board["MatchSample"] / 60.0))
    board["Confidence"] = 0.55 * sample_conf + 0.25 * bbe_conf + 0.20 * matchup_conf
    board["Confidence_Level"] = pd.cut(
        board["Confidence"],
        bins=[-np.inf, 48, 72, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    board = board.sort_values(["Model_1plus_Hit", "HitScore"], ascending=False).reset_index(drop=True)
    board.insert(0, "Rank", np.arange(1, len(board) + 1))
    return board, profile




# -----------------------------------------------------------------------------
# Clean slate/matchup interface helpers
# -----------------------------------------------------------------------------
import json
from html import escape
from io import StringIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MLB_TEAM_ABBR = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH",
    134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

TEAM_ALIASES = {
    "OAK": "ATH",
    "ATH": "OAK",
}


# Team presentation metadata used only for the dashboard UI.
TEAM_COLORS = {
    "ARI": ("#A71930", "#E3D4AD"), "ATH": ("#003831", "#EFB21E"),
    "OAK": ("#003831", "#EFB21E"), "ATL": ("#CE1141", "#13274F"),
    "BAL": ("#DF4601", "#000000"), "BOS": ("#BD3039", "#0C2340"),
    "CHC": ("#0E3386", "#CC3433"), "CWS": ("#27251F", "#C4CED4"),
    "CIN": ("#C6011F", "#000000"), "CLE": ("#00385D", "#E50022"),
    "COL": ("#33006F", "#C4CED4"), "DET": ("#0C2340", "#FA4616"),
    "HOU": ("#002D62", "#EB6E1F"), "KC": ("#004687", "#BD9B60"),
    "LAA": ("#BA0021", "#003263"), "LAD": ("#005A9C", "#EF3E42"),
    "MIA": ("#00A3E0", "#EF3340"), "MIL": ("#12284B", "#FFC52F"),
    "MIN": ("#002B5C", "#D31145"), "NYM": ("#002D72", "#FF5910"),
    "NYY": ("#0C2340", "#C4CED4"), "PHI": ("#E81828", "#002D72"),
    "PIT": ("#27251F", "#FDB827"), "SD": ("#2F241D", "#FFC425"),
    "SEA": ("#0C2C56", "#005C5C"), "SF": ("#FD5A1E", "#27251F"),
    "STL": ("#C41E3A", "#0C2340"), "TB": ("#092C5C", "#8FBCE6"),
    "TEX": ("#003278", "#C0111F"), "TOR": ("#134A8E", "#E8291C"),
    "WSH": ("#AB0003", "#14225A"),
}

# Renamed and temporary venues are grouped so a schedule name can still match
# the name used by Baseball Savant's park-factor table.
VENUE_ALIAS_GROUPS = [
    {"daikinpark", "minutemaidpark"},
    {"ratefield", "guaranteedratefield", "uscellularfield", "comiskeypark"},
    {"sutterhealthpark", "oaklandcoliseum", "ringcentralcoliseum"},
    {"loandepotpark", "marlinspark"},
    {"oraclepark", "attpark", "pacbellpark"},
    {"progressivefield", "jacobsfield"},
    {"americanfamilyfield", "millerpark"},
    {"rogerscentre", "skydome"},
    {"georgemsteinbrennerfield", "tropicanafield"},
]


def team_logo_url(team_id: object) -> str:
    numeric = pd.to_numeric(pd.Series([team_id]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return f"https://www.mlbstatic.com/team-logos/{int(numeric)}.svg"


def team_palette(team_code: object) -> tuple[str, str]:
    code = str(team_code or "").upper().strip()
    return TEAM_COLORS.get(code, ("#2563EB", "#7C3AED"))


def normalized_venue_variants(value: object) -> set[str]:
    normalized = normalize_name(value)
    variants = {normalized} if normalized else set()
    for group in VENUE_ALIAS_GROUPS:
        if normalized in group or any(token in normalized or normalized in token for token in group):
            variants.update(group)
    return variants


def venue_names_match(left: object, right: object) -> bool:
    left_variants = normalized_venue_variants(left)
    right_variants = normalized_venue_variants(right)
    for left_name in left_variants:
        for right_name in right_variants:
            if left_name == right_name:
                return True
            if len(left_name) >= 7 and len(right_name) >= 7:
                if left_name in right_name or right_name in left_name:
                    return True
    return False


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy()
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = [
            " ".join(str(part) for part in column if str(part) != "nan").strip()
            for column in table.columns
        ]
    else:
        table.columns = [str(column).strip() for column in table.columns]
    return table


def _numeric_park_factor(value: object) -> float | None:
    text = str(value).replace("%", "").replace(",", "").strip()
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    numeric = float(numeric)
    if 0.5 <= numeric <= 1.8:
        numeric *= 100.0
    if 50.0 <= numeric <= 180.0:
        return numeric
    return None


def _factor_from_html_tables(html_text: str, venue: str, metric: str) -> float | None:
    try:
        tables = pd.read_html(StringIO(html_text))
    except Exception:
        return None

    metric_norm = normalize_name(metric)
    exact_metric_names = {
        "hits": {"hits", "indexhits", "hitfactor", "hitsfactor"},
        "hr": {"hr", "indexhr", "homerun", "homeruns", "hrfactor"},
    }.get(metric_norm, {metric_norm, f"index{metric_norm}"})

    for raw_table in tables:
        table = _flatten_columns(raw_table)
        normalized_columns = {column: normalize_name(column) for column in table.columns}
        venue_columns = [
            column for column, normalized in normalized_columns.items()
            if "venue" in normalized or "stadium" in normalized or normalized == "park"
        ]
        if not venue_columns:
            continue

        metric_columns = [
            column for column, normalized in normalized_columns.items()
            if normalized in exact_metric_names
            or any(name and name in normalized for name in exact_metric_names)
        ]
        if not metric_columns:
            metric_columns = [
                column for column, normalized in normalized_columns.items()
                if "parkfactor" in normalized or normalized in {"factor", "index"}
            ]
        if not metric_columns:
            continue

        for _, row in table.iterrows():
            if not any(venue_names_match(row.get(column), venue) for column in venue_columns):
                continue
            for column in metric_columns:
                factor = _numeric_park_factor(row.get(column))
                if factor is not None:
                    return factor
    return None


def _fetch_savant_side_factor(venue: str, year: int, side: str, metric: str) -> tuple[float | None, str | None]:
    metric_key = "Hits" if normalize_name(metric) == "hits" else "HR"
    attempts = [
        (year, f"index_{metric_key}"),
        (year, metric_key),
        (year - 1, f"index_{metric_key}"),
    ]
    last_error = None
    for query_year, stat_value in attempts:
        params = {
            "year": int(query_year),
            "type": "year",
            "batSide": side,
            "stat": stat_value,
            "condition": "All",
            "rolling": 3,
        }
        url = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?" + urlencode(params)
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MLB-Statcast-Dashboard/2.0",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        try:
            with urlopen(request, timeout=6) as response:
                html_text = response.read().decode("utf-8", errors="replace")
            factor = _factor_from_html_tables(html_text, venue, metric_key)
            if factor is not None:
                return factor, f"Baseball Savant {query_year} · 3-year · All conditions"
            last_error = "Savant loaded, but its park-factor table could not be parsed."
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            last_error = str(exc)
    return None, last_error


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_savant_park_factors(venue: str, year: int, metric: str) -> dict:
    """Read handed park factors from the repository's park_factors.csv file."""
    if not venue or normalize_name(venue) in {"manualmatchup", "venuetbd"}:
        return {
            "L": 100.0,
            "R": 100.0,
            "ok": False,
            "source": "Neutral fallback",
            "error": "A real MLB venue was not available.",
        }

    candidate_paths = [
        Path(__file__).resolve().parent / "park_factors.csv",
        Path.cwd() / "park_factors.csv",
    ]
    csv_path = next((path for path in candidate_paths if path.exists()), None)
    if csv_path is None:
        return {
            "L": 100.0,
            "R": 100.0,
            "ok": False,
            "source": "Neutral fallback",
            "error": "park_factors.csv was not found beside the dashboard file.",
        }

    try:
        factors = pd.read_csv(csv_path)
    except Exception as exc:
        return {
            "L": 100.0,
            "R": 100.0,
            "ok": False,
            "source": "Neutral fallback",
            "error": f"park_factors.csv could not be read: {type(exc).__name__}: {exc}",
        }

    required = {"venue", "hr_l", "hr_r", "hits_l", "hits_r"}
    missing = sorted(required - set(factors.columns))
    if missing:
        return {
            "L": 100.0,
            "R": 100.0,
            "ok": False,
            "source": "Neutral fallback",
            "error": "park_factors.csv is missing columns: " + ", ".join(missing),
        }

    matching_row = None
    for _, row in factors.iterrows():
        if venue_names_match(row.get("venue"), venue):
            matching_row = row
            break

    if matching_row is None:
        return {
            "L": 100.0,
            "R": 100.0,
            "ok": False,
            "source": "Neutral fallback",
            "error": f"No park-factor row matched venue: {venue}",
        }

    metric_prefix = "hits" if normalize_name(metric) == "hits" else "hr"
    left = _numeric_park_factor(matching_row.get(f"{metric_prefix}_l"))
    right = _numeric_park_factor(matching_row.get(f"{metric_prefix}_r"))
    if left is None or right is None:
        return {
            "L": float(left) if left is not None else 100.0,
            "R": float(right) if right is not None else 100.0,
            "ok": False,
            "source": "Partial local park-factor data",
            "error": f"Invalid {metric_prefix.upper()} factor for {venue}.",
        }

    season_value = matching_row.get("season", year)
    rolling_value = matching_row.get("rolling_years", 3)
    try:
        season_text = str(int(float(season_value)))
    except (TypeError, ValueError):
        season_text = str(season_value)
    try:
        rolling_text = str(int(float(rolling_value)))
    except (TypeError, ValueError):
        rolling_text = str(rolling_value)

    return {
        "L": float(left),
        "R": float(right),
        "ok": True,
        "source": f"Local park_factors.csv · {season_text} · {rolling_text}-year",
        "error": None,
    }



def team_id_from_code(team_code: object) -> int | None:
    """Translate a Statcast/scoreboard abbreviation to an MLB team ID."""
    code = str(team_code or "").upper().strip()
    if code == "OAK":
        code = "ATH"
    for team_id, abbreviation in MLB_TEAM_ABBR.items():
        if abbreviation == code:
            return int(team_id)
    return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_active_roster(team_id: int | None, roster_date: str | None) -> tuple[pd.DataFrame, str | None]:
    """Return active non-pitchers, including bat side, from the MLB Stats API."""
    empty = pd.DataFrame(
        columns=["player_id", "Player", "Bats", "Position", "RosterStatus"]
    )
    if team_id is None:
        return empty, "No MLB team ID was available for this matchup."

    base = f"https://statsapi.mlb.com/api/v1/teams/{int(team_id)}/roster?rosterType=active"
    urls = [f"{base}&date={roster_date}"] if roster_date else []
    urls.append(base)

    payload = None
    last_error = None
    for url in urls:
        request = Request(url, headers={"User-Agent": "MLB-Statcast-Dashboard/1.0"})
        try:
            with urlopen(request, timeout=12) as response:
                payload = json.load(response)
            if payload.get("roster"):
                break
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_error = str(exc)
            payload = None

    if not payload or not payload.get("roster"):
        return empty, last_error or "The active roster endpoint returned no players."

    records = []
    for item in payload.get("roster", []):
        person = item.get("person", {}) or {}
        position = item.get("position", {}) or {}
        position_type = str(position.get("type", ""))
        abbreviation = str(position.get("abbreviation", ""))

        # Exclude traditional pitchers, but retain any two-way designation.
        if position_type == "Pitcher" and abbreviation not in {"TWP", "Two-Way"}:
            continue

        player_id = person.get("id")
        if player_id is None:
            continue
        records.append(
            {
                "player_id": int(player_id),
                "Player": person.get("fullName", f"MLB ID {player_id}"),
                "Position": abbreviation or position.get("name", ""),
                "RosterStatus": (item.get("status", {}) or {}).get("description", "Active"),
            }
        )

    roster = pd.DataFrame(records).drop_duplicates("player_id")
    if roster.empty:
        return empty, "The active roster contained no position players."

    ids = ",".join(roster["player_id"].astype(str).tolist())
    people_url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids}"
    request = Request(people_url, headers={"User-Agent": "MLB-Statcast-Dashboard/1.0"})
    bat_map: dict[int, str] = {}
    try:
        with urlopen(request, timeout=12) as response:
            people_payload = json.load(response)
        for person in people_payload.get("people", []):
            person_id = person.get("id")
            bat_side = (person.get("batSide", {}) or {}).get("code")
            if person_id is not None and bat_side:
                bat_map[int(person_id)] = str(bat_side)
    except (HTTPError, URLError, TimeoutError, ValueError):
        pass

    roster["Bats"] = roster["player_id"].map(bat_map)
    return roster[["player_id", "Player", "Bats", "Position", "RosterStatus"]], None



def inject_clean_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #0f172a;
            --muted: #64748b;
            --line: rgba(148, 163, 184, 0.28);
            --glass: rgba(255, 255, 255, 0.88);
        }
        .stApp {
            background:
                radial-gradient(circle at 7% 2%, rgba(59,130,246,.14), transparent 26rem),
                radial-gradient(circle at 93% 5%, rgba(249,115,22,.13), transparent 25rem),
                linear-gradient(180deg, #f8fbff 0%, #f2f6fc 48%, #f8fafc 100%);
            color: var(--ink);
        }
        .block-container {
            max-width: 1780px;
            padding-top: 1.05rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #172554 58%, #312e81 100%);
        }
        [data-testid="stSidebar"] * { color: #f8fafc; }
        [data-testid="stSidebar"] input { color: #0f172a !important; }
        h1 { letter-spacing: -0.045em; font-weight: 900; }
        h2, h3 { letter-spacing: -0.028em; }
        .app-kicker {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            padding: .34rem .7rem;
            border-radius: 999px;
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            color: white;
            font-size: .72rem;
            font-weight: 850;
            letter-spacing: .11em;
            text-transform: uppercase;
            box-shadow: 0 8px 20px rgba(37, 99, 235, .24);
            margin-bottom: .35rem;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(255,255,255,.98), rgba(239,246,255,.92));
            border: 1px solid rgba(96,165,250,.28);
            border-radius: 18px;
            padding: .9rem 1.05rem;
            box-shadow: 0 10px 25px rgba(15,23,42,.07);
        }
        div[role="radiogroup"] { gap: .58rem; flex-wrap: wrap; }
        div[role="radiogroup"] label {
            background: rgba(255,255,255,.9);
            border: 1px solid rgba(148,163,184,.32);
            border-radius: 13px;
            padding: .52rem .82rem;
            min-height: 42px;
            box-shadow: 0 4px 12px rgba(15,23,42,.04);
        }
        div[role="radiogroup"] label:hover {
            border-color: #60a5fa;
            transform: translateY(-1px);
            box-shadow: 0 9px 20px rgba(37,99,235,.11);
        }
        .slate-card {
            min-height: 118px;
            background: linear-gradient(145deg, rgba(255,255,255,.98), rgba(248,250,252,.92));
            border: 1px solid rgba(148,163,184,.28);
            border-radius: 17px;
            padding: .72rem .72rem .62rem;
            text-align: center;
            box-shadow: 0 8px 22px rgba(15,23,42,.06);
            transition: .16s ease;
        }
        .slate-card.selected {
            border: 2px solid #f97316;
            background: linear-gradient(145deg, #fff7ed, #ffffff);
            box-shadow: 0 12px 28px rgba(249,115,22,.20);
        }
        .slate-logos {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: .55rem;
            min-height: 40px;
        }
        .slate-logos img { width: 34px; height: 34px; object-fit: contain; }
        .slate-at { color: #64748b; font-weight: 900; }
        .slate-game { font-weight: 900; color: #0f172a; margin-top: .25rem; }
        .slate-time { color: #475569; font-weight: 750; font-size: .82rem; }
        .slate-venue {
            color: #64748b;
            font-size: .72rem;
            line-height: 1.15;
            margin-top: .18rem;
            min-height: 1.7rem;
        }
        .matchup-hero {
            display: grid;
            grid-template-columns: 130px minmax(0, 1fr) 130px;
            align-items: center;
            gap: 1rem;
            padding: 1.2rem 1.4rem;
            margin: .85rem 0 1rem;
            border-radius: 22px;
            background:
                linear-gradient(105deg, color-mix(in srgb, var(--away) 18%, white), rgba(255,255,255,.96) 40%, rgba(255,255,255,.96) 60%, color-mix(in srgb, var(--home) 18%, white));
            border: 1px solid rgba(148,163,184,.30);
            box-shadow: 0 16px 38px rgba(15,23,42,.10);
            overflow: hidden;
        }
        .hero-team { text-align: center; }
        .hero-team img { width: 76px; height: 76px; object-fit: contain; filter: drop-shadow(0 8px 10px rgba(15,23,42,.16)); }
        .hero-abbr { font-weight: 950; color: #0f172a; font-size: 1.05rem; }
        .hero-center { text-align: center; min-width: 0; }
        .hero-title { font-size: 2rem; font-weight: 950; letter-spacing: -.045em; color: #0f172a; }
        .hero-meta { color: #475569; font-weight: 700; margin-top: .18rem; }
        .hero-probables { color: #64748b; font-size: .88rem; margin-top: .35rem; }
        .hero-badge {
            display: inline-flex;
            padding: .28rem .62rem;
            border-radius: 999px;
            background: linear-gradient(90deg, #f97316, #ef4444);
            color: white;
            font-size: .69rem;
            font-weight: 900;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .36rem;
        }
        .offense-choice {
            text-align: center;
            padding: .55rem;
            border-radius: 15px;
            background: rgba(255,255,255,.78);
            border: 1px solid rgba(148,163,184,.25);
        }
        .offense-choice img { width: 48px; height: 48px; object-fit: contain; }
        .board-heading {
            display: flex;
            align-items: center;
            gap: 1rem;
            background: linear-gradient(110deg, rgba(15,23,42,.98), rgba(30,64,175,.94), rgba(124,58,237,.88));
            color: white;
            border-radius: 19px;
            padding: .95rem 1.2rem;
            margin: 1rem 0 .75rem;
            box-shadow: 0 13px 30px rgba(30,64,175,.20);
        }
        .board-heading img { width: 58px; height: 58px; object-fit: contain; }
        .board-heading h2 { color: white; margin: 0; font-size: 1.55rem; }
        .board-heading p { color: #dbeafe; margin: .12rem 0 0; font-size: .86rem; }
        .leader-card {
            background: linear-gradient(145deg, #ffffff, #eff6ff);
            border: 1px solid rgba(96,165,250,.28);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            min-height: 138px;
            box-shadow: 0 12px 27px rgba(15,23,42,.075);
            position: relative;
            overflow: hidden;
        }
        .leader-card:after {
            content: "";
            position: absolute;
            width: 90px;
            height: 90px;
            border-radius: 50%;
            right: -35px;
            top: -35px;
            background: linear-gradient(135deg, rgba(37,99,235,.18), rgba(249,115,22,.15));
        }
        .leader-rank { color: #64748b; font-size: .75rem; font-weight: 850; letter-spacing: .08em; }
        .leader-name { color: #0f172a; font-size: 1.08rem; font-weight: 900; margin: .18rem 0; }
        .leader-prob { color: #15803d; font-size: 1.55rem; font-weight: 950; }
        .leader-sub { color: #64748b; font-size: .81rem; }
        .context-note {
            background: linear-gradient(145deg, #eff6ff, #ffffff);
            border-left: 5px solid #2563eb;
            border-radius: 12px;
            padding: .78rem .92rem;
            color: #334155;
            margin: .35rem 0 .8rem;
            box-shadow: 0 7px 18px rgba(37,99,235,.08);
        }
        .park-grid { display:grid; grid-template-columns:1fr 1fr; gap:.55rem; margin:.25rem 0 .55rem; }
        .park-chip {
            border-radius: 14px;
            padding: .72rem .78rem;
            background: linear-gradient(145deg, #ecfeff, #ffffff);
            border: 1px solid rgba(6,182,212,.25);
        }
        .park-chip .side { color:#64748b; font-size:.72rem; font-weight:800; }
        .park-chip .factor { color:#0f172a; font-size:1.35rem; font-weight:950; }
        .park-source { color:#64748b; font-size:.76rem; line-height:1.25; }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(148,163,184,.30);
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 22px rgba(15,23,42,.055);
        }
        button[kind="primary"] {
            background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
            border: 0 !important;
            box-shadow: 0 8px 18px rgba(37,99,235,.22) !important;
        }
        [data-baseweb="tab-list"] { gap: .35rem; }
        [data-baseweb="tab"] {
            background: rgba(255,255,255,.78);
            border-radius: 10px 10px 0 0;
            padding-left: .85rem;
            padding-right: .85rem;
        }
        @media (max-width: 900px) {
            .matchup-hero { grid-template-columns: 82px minmax(0,1fr) 82px; padding: .9rem; }
            .hero-team img { width: 54px; height: 54px; }
            .hero-title { font-size: 1.45rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=900, show_spinner=False)
def fetch_mlb_schedule(slate_date: str) -> tuple[list[dict], str | None]:
    """Load an MLB slate. Gracefully falls back to manual selection on failure."""
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={slate_date}&hydrate=probablePitcher,venue"
    )
    request = Request(url, headers={"User-Agent": "MLB-Statcast-Dashboard/1.0"})
    try:
        with urlopen(request, timeout=12) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        return [], str(exc)

    games: list[dict] = []
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            away_info = game.get("teams", {}).get("away", {})
            home_info = game.get("teams", {}).get("home", {})
            away_team = away_info.get("team", {})
            home_team = home_info.get("team", {})
            away_probable = away_info.get("probablePitcher", {}) or {}
            home_probable = home_info.get("probablePitcher", {}) or {}

            game_time = "TBD"
            raw_time = game.get("gameDate")
            if raw_time:
                parsed = pd.to_datetime(raw_time, utc=True, errors="coerce")
                if not pd.isna(parsed):
                    try:
                        eastern = parsed.tz_convert("America/New_York")
                        game_time = eastern.strftime("%I:%M %p ET").lstrip("0")
                    except Exception:
                        game_time = parsed.strftime("%I:%M %p UTC").lstrip("0")

            away_id = away_team.get("id")
            home_id = home_team.get("id")
            games.append(
                {
                    "game_pk": game.get("gamePk"),
                    "game_datetime_utc": raw_time,
                    "away_id": away_id,
                    "home_id": home_id,
                    "slate_date": slate_date,
                    "away_abbr": MLB_TEAM_ABBR.get(away_id, away_team.get("name", "AWAY")[:3].upper()),
                    "home_abbr": MLB_TEAM_ABBR.get(home_id, home_team.get("name", "HOME")[:3].upper()),
                    "away_name": away_team.get("name", "Away"),
                    "home_name": home_team.get("name", "Home"),
                    "away_pitcher_id": away_probable.get("id"),
                    "away_pitcher_name": away_probable.get("fullName", "TBD"),
                    "home_pitcher_id": home_probable.get("id"),
                    "home_pitcher_name": home_probable.get("fullName", "TBD"),
                    "time_et": game_time,
                    "venue": game.get("venue", {}).get("name", "Venue TBD"),
                    "status": game.get("status", {}).get("detailedState", "Scheduled"),
                }
            )
    return games, None


def _parse_mlb_team_lineup(team_box: dict) -> pd.DataFrame:
    """Extract the original 1-9 batting order from an MLB boxscore team block."""
    columns = [
        "player_id", "Player_MLB", "Position_MLB", "LineupSpot",
        "LineupRole", "RawBattingOrder",
    ]
    if not isinstance(team_box, dict):
        return pd.DataFrame(columns=columns)

    players = team_box.get("players", {}) or {}
    candidates: list[dict] = []

    for player_key, record in players.items():
        if not isinstance(record, dict):
            continue
        person = record.get("person", {}) or {}
        player_id = person.get("id")
        if player_id is None:
            digits = "".join(character for character in str(player_key) if character.isdigit())
            player_id = int(digits) if digits else None
        numeric_id = pd.to_numeric(pd.Series([player_id]), errors="coerce").iloc[0]
        if pd.isna(numeric_id):
            continue

        raw_order = record.get("battingOrder")
        numeric_order = pd.to_numeric(pd.Series([raw_order]), errors="coerce").iloc[0]
        if pd.isna(numeric_order):
            continue
        numeric_order = int(numeric_order)
        lineup_spot = numeric_order // 100 if numeric_order >= 100 else numeric_order
        if lineup_spot < 1 or lineup_spot > 9:
            continue

        game_status = record.get("gameStatus", {}) or {}
        is_substitute = bool(game_status.get("isSubstitute", False))
        is_on_bench = bool(game_status.get("isOnBench", False))
        position = record.get("position", {}) or {}
        candidates.append(
            {
                "player_id": int(numeric_id),
                "Player_MLB": person.get("fullName") or person.get("fullNameLastFirst") or f"MLB ID {int(numeric_id)}",
                "Position_MLB": position.get("abbreviation") or position.get("name") or "",
                "LineupSpot": int(lineup_spot),
                "LineupRole": "Substitute" if is_substitute else "Starter",
                "RawBattingOrder": int(numeric_order),
                "_substitute": is_substitute,
                "_bench": is_on_bench,
            }
        )

    # Some versions of the game feed expose the ordered player IDs separately.
    # Use that list only to fill missing lineup spots.
    ordered_ids = team_box.get("battingOrder") or []
    if not isinstance(ordered_ids, list) or len(ordered_ids) < 9:
        ordered_ids = team_box.get("batters") or []
    if isinstance(ordered_ids, list) and len(ordered_ids) >= 9:
        existing_spots = {int(row["LineupSpot"]) for row in candidates}
        player_lookup = {}
        for player_key, record in players.items():
            if not isinstance(record, dict):
                continue
            person = record.get("person", {}) or {}
            pid = person.get("id")
            if pid is None:
                digits = "".join(character for character in str(player_key) if character.isdigit())
                pid = int(digits) if digits else None
            if pid is not None:
                player_lookup[int(pid)] = record

        for spot, player_id in enumerate(ordered_ids[:9], start=1):
            if spot in existing_spots:
                continue
            numeric_id = pd.to_numeric(pd.Series([player_id]), errors="coerce").iloc[0]
            if pd.isna(numeric_id):
                continue
            numeric_id = int(numeric_id)
            record = player_lookup.get(numeric_id, {})
            person = record.get("person", {}) or {}
            position = record.get("position", {}) or {}
            candidates.append(
                {
                    "player_id": numeric_id,
                    "Player_MLB": person.get("fullName") or f"MLB ID {numeric_id}",
                    "Position_MLB": position.get("abbreviation") or position.get("name") or "",
                    "LineupSpot": spot,
                    "LineupRole": "Starter",
                    "RawBattingOrder": spot * 100,
                    "_substitute": False,
                    "_bench": False,
                }
            )

    if not candidates:
        return pd.DataFrame(columns=columns)

    lineup = pd.DataFrame(candidates)
    lineup = lineup.sort_values(
        ["LineupSpot", "_substitute", "_bench", "RawBattingOrder"],
        ascending=[True, True, True, True],
    )
    lineup = lineup.drop_duplicates("LineupSpot", keep="first")
    lineup = lineup[lineup["LineupSpot"].between(1, 9)].copy()
    lineup = lineup.sort_values("LineupSpot").reset_index(drop=True)
    return lineup[columns]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_mlb_confirmed_lineup(game_pk: object, team_side: str) -> dict:
    """Fetch a posted lineup from MLB's game feed with a boxscore fallback."""
    numeric_game = pd.to_numeric(pd.Series([game_pk]), errors="coerce").iloc[0]
    side = str(team_side).lower().strip()
    if pd.isna(numeric_game) or side not in {"away", "home"}:
        return {
            "ok": False,
            "status": "Unavailable",
            "lineup": pd.DataFrame(columns=["player_id", "LineupSpot"]),
            "source": "Manual/recent lineup",
            "game_state": "",
            "updated": "",
            "error": "A valid MLB game and team side were not available.",
        }

    game_pk_int = int(numeric_game)
    endpoints = [
        ("MLB live game feed", f"https://statsapi.mlb.com/api/v1.1/game/{game_pk_int}/feed/live"),
        ("MLB boxscore", f"https://statsapi.mlb.com/api/v1/game/{game_pk_int}/boxscore"),
    ]
    errors: list[str] = []

    for source_name, url in endpoints:
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 MLB-Statcast-Dashboard/3.0",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=12) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            errors.append(f"{source_name}: {type(exc).__name__}: {exc}")
            continue

        if "liveData" in payload:
            team_box = (
                payload.get("liveData", {})
                .get("boxscore", {})
                .get("teams", {})
                .get(side, {})
            )
            game_state = (
                payload.get("gameData", {})
                .get("status", {})
                .get("detailedState", "")
            )
            feed_timestamp = payload.get("metaData", {}).get("timeStamp")
        else:
            team_box = payload.get("teams", {}).get(side, {})
            game_state = ""
            feed_timestamp = None

        lineup = _parse_mlb_team_lineup(team_box)
        unique_spots = int(lineup["LineupSpot"].nunique()) if not lineup.empty else 0
        if unique_spots:
            status = "Confirmed" if unique_spots >= 9 else "Partial"
            updated = str(feed_timestamp or pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %I:%M %p ET"))
            return {
                "ok": unique_spots >= 9,
                "status": status,
                "lineup": lineup,
                "source": source_name,
                "game_state": game_state,
                "updated": updated,
                "error": None,
            }

    return {
        "ok": False,
        "status": "Not posted",
        "lineup": pd.DataFrame(columns=["player_id", "LineupSpot"]),
        "source": "Recent lineup fallback",
        "game_state": "",
        "updated": "",
        "error": " | ".join(errors[-2:]) if errors else "No batting order was present in the MLB game feed yet.",
    }


def _clean_lineup_seed(lineup: pd.DataFrame | None) -> pd.DataFrame:
    if lineup is None or lineup.empty:
        return pd.DataFrame(columns=["player_id", "LineupSpot"])
    clean = lineup[[column for column in ["player_id", "LineupSpot"] if column in lineup.columns]].copy()
    if set(clean.columns) != {"player_id", "LineupSpot"}:
        return pd.DataFrame(columns=["player_id", "LineupSpot"])
    clean["player_id"] = pd.to_numeric(clean["player_id"], errors="coerce")
    clean["LineupSpot"] = pd.to_numeric(clean["LineupSpot"], errors="coerce")
    clean = clean.dropna(subset=["player_id", "LineupSpot"])
    clean = clean[clean["LineupSpot"].between(1, 9)]
    clean["player_id"] = clean["player_id"].astype(int)
    clean["LineupSpot"] = clean["LineupSpot"].astype(int)
    return clean.drop_duplicates("LineupSpot").sort_values("LineupSpot").reset_index(drop=True)


def _lineup_fingerprint(lineup: pd.DataFrame) -> str:
    clean = _clean_lineup_seed(lineup)
    if clean.empty:
        return "fallback"
    return "_".join(
        f"{int(row.player_id)}-{int(row.LineupSpot)}"
        for row in clean.itertuples(index=False)
    )


def match_statcast_team(team_code: str, available_teams: list[str]) -> str | None:
    if team_code in available_teams:
        return team_code
    alias = TEAM_ALIASES.get(team_code)
    if alias in available_teams:
        return alias
    return None


def pitcher_row_from_id(pitcher_summary: pd.DataFrame, pitcher_id: object) -> pd.Series | None:
    if pitcher_id is None:
        return None
    numeric_id = pd.to_numeric(pd.Series([pitcher_id]), errors="coerce").iloc[0]
    if pd.isna(numeric_id):
        return None
    rows = pitcher_summary[pitcher_summary["pitcher"].astype("Int64").eq(int(numeric_id))]
    if rows.empty:
        return None
    return rows.iloc[0]


def manual_matchup_controls(
    pitcher_summary: pd.DataFrame,
    available_teams: list[str],
    key_prefix: str,
) -> dict:
    c1, c2, c3 = st.columns([1.35, 1.0, 0.75])
    with c1:
        selected_display = st.selectbox(
            "Starting pitcher",
            pitcher_summary["Display"].tolist(),
            key=f"{key_prefix}_manual_pitcher",
        )
        row = pitcher_summary.loc[pitcher_summary["Display"].eq(selected_display)].iloc[0]
    with c2:
        opponent_teams = [
            team for team in available_teams if team != str(row["Pitcher_Team"])
        ]
        selected_team = st.selectbox(
            "Batting team",
            opponent_teams,
            key=f"{key_prefix}_manual_team",
        )
    with c3:
        side = st.radio(
            "Side",
            ["Away", "Home"],
            horizontal=True,
            key=f"{key_prefix}_manual_side",
        )
    return {
        "pitcher_id": int(row["pitcher"]),
        "pitcher_display": selected_display,
        "pitcher_name": str(row["Player_Name"]),
        "pitcher_team": str(row["Pitcher_Team"]),
        "batting_team": selected_team,
        "home_away": side,
        "venue": "Manual matchup",
        "time_et": "",
        "status": "Manual",
        "away_abbr": selected_team if side == "Away" else str(row["Pitcher_Team"]),
        "home_abbr": selected_team if side == "Home" else str(row["Pitcher_Team"]),
        "game_pk": None,
        "game_datetime_utc": None,
        "batting_team_id": team_id_from_code(selected_team),
        "opponent_team_id": team_id_from_code(str(row["Pitcher_Team"])),
        "slate_date": date.today().isoformat(),
    }



def render_slate_game_cards(games: list[dict], key_prefix: str) -> object:
    state_key = f"{key_prefix}_selected_game_pk"
    valid_ids = [game["game_pk"] for game in games]
    if st.session_state.get(state_key) not in valid_ids:
        st.session_state[state_key] = valid_ids[0]

    cards_per_row = 6
    for start in range(0, len(games), cards_per_row):
        chunk = games[start : start + cards_per_row]
        columns = st.columns(cards_per_row)
        for index, column in enumerate(columns):
            if index >= len(chunk):
                continue
            game = chunk[index]
            selected = st.session_state[state_key] == game["game_pk"]
            away_logo = team_logo_url(game.get("away_id"))
            home_logo = team_logo_url(game.get("home_id"))
            selected_class = " selected" if selected else ""
            with column:
                st.markdown(
                    f"""
                    <div class="slate-card{selected_class}">
                        <div class="slate-logos">
                            <img src="{away_logo}" alt="{escape(str(game['away_abbr']))}">
                            <span class="slate-at">@</span>
                            <img src="{home_logo}" alt="{escape(str(game['home_abbr']))}">
                        </div>
                        <div class="slate-game">{escape(str(game['away_abbr']))} @ {escape(str(game['home_abbr']))}</div>
                        <div class="slate-time">{escape(str(game['time_et']))}</div>
                        <div class="slate-venue">{escape(str(game['venue']))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Selected" if selected else "Select game",
                    key=f"{key_prefix}_game_card_{game['game_pk']}",
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                ):
                    if not selected:
                        st.session_state[state_key] = game["game_pk"]
                        st.rerun()
    return st.session_state[state_key]


def render_matchup_hero(matchup: dict, model_label: str) -> None:
    away_abbr = str(matchup.get("away_abbr", "AWAY"))
    home_abbr = str(matchup.get("home_abbr", "HOME"))
    away_id = matchup.get("away_id") or team_id_from_code(away_abbr)
    home_id = matchup.get("home_id") or team_id_from_code(home_abbr)
    away_logo = team_logo_url(away_id)
    home_logo = team_logo_url(home_id)
    away_color, _ = team_palette(away_abbr)
    home_color, _ = team_palette(home_abbr)
    probable_text = matchup.get("probables") or ""
    st.markdown(
        f"""
        <div class="matchup-hero" style="--away:{away_color}; --home:{home_color};">
            <div class="hero-team">
                <img src="{away_logo}" alt="{escape(away_abbr)}">
                <div class="hero-abbr">{escape(away_abbr)}</div>
            </div>
            <div class="hero-center">
                <div class="hero-badge">{escape(model_label)}</div>
                <div class="hero-title">{escape(away_abbr)} @ {escape(home_abbr)}</div>
                <div class="hero-meta">{escape(str(matchup.get('time_et', '')))} · {escape(str(matchup.get('venue', '')))} · {escape(str(matchup.get('status', '')))}</div>
                <div class="hero-probables">{escape(str(probable_text))}</div>
            </div>
            <div class="hero-team">
                <img src="{home_logo}" alt="{escape(home_abbr)}">
                <div class="hero-abbr">{escape(home_abbr)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_board_header(matchup: dict, batting_team: str, pitcher_name: str, model_label: str) -> None:
    batting_team_id = matchup.get("batting_team_id") or team_id_from_code(batting_team)
    logo = team_logo_url(batting_team_id)
    st.markdown(
        f"""
        <div class="board-heading">
            <img src="{logo}" alt="{escape(str(batting_team))}">
            <div>
                <div style="font-size:.72rem;font-weight:900;letter-spacing:.11em;text-transform:uppercase;color:#bfdbfe;">{escape(model_label)}</div>
                <h2>{escape(str(batting_team))} hitters vs {escape(str(pitcher_name))}</h2>
                <p>{escape(str(matchup.get('away_abbr', '')))} @ {escape(str(matchup.get('home_abbr', '')))} · {escape(str(matchup.get('time_et', '')))} · {escape(str(matchup.get('venue', '')))}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def matchup_selector(
    pitcher_summary: pd.DataFrame,
    available_teams: list[str],
    key_prefix: str,
) -> dict:
    model_label = "ADVANCED HIT MODEL" if "hit" in key_prefix else "ADVANCED HR MODEL"
    st.subheader("Choose a matchup")
    source = st.radio(
        "Matchup source",
        ["MLB slate", "Manual matchup"],
        horizontal=True,
        key=f"{key_prefix}_source",
    )

    if source == "MLB slate":
        slate_date = st.date_input(
            "Slate date",
            value=date.today(),
            key=f"{key_prefix}_slate_date",
        )
        games, schedule_error = fetch_mlb_schedule(slate_date.isoformat())
        if schedule_error:
            st.warning(f"The MLB slate could not be loaded: {schedule_error}")
        if games:
            st.caption(f"{len(games)} games · select a matchup card")
            selected_game_pk = render_slate_game_cards(games, key_prefix)
            game = next(game for game in games if game["game_pk"] == selected_game_pk)

            offense_key = f"{key_prefix}_offense_side_{selected_game_pk}"
            if st.session_state.get(offense_key) not in {"away", "home"}:
                st.session_state[offense_key] = "away"

            st.markdown("#### Analyze offense")
            offense_columns = st.columns(2)
            for side_name, column, team_id, abbr, full_name in [
                ("away", offense_columns[0], game["away_id"], game["away_abbr"], game["away_name"]),
                ("home", offense_columns[1], game["home_id"], game["home_abbr"], game["home_name"]),
            ]:
                selected_offense = st.session_state[offense_key] == side_name
                with column:
                    st.markdown(
                        f"""
                        <div class="offense-choice">
                            <img src="{team_logo_url(team_id)}" alt="{escape(str(abbr))}"><br>
                            <b>{escape(str(full_name))}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        f"Analyze {abbr} hitters",
                        key=f"{key_prefix}_offense_button_{selected_game_pk}_{side_name}",
                        use_container_width=True,
                        type="primary" if selected_offense else "secondary",
                    ):
                        if not selected_offense:
                            st.session_state[offense_key] = side_name
                            st.rerun()

            away_offense = st.session_state[offense_key] == "away"
            batting_code = game["away_abbr"] if away_offense else game["home_abbr"]
            opponent_code = game["home_abbr"] if away_offense else game["away_abbr"]
            batting_team_id = game["away_id"] if away_offense else game["home_id"]
            opponent_team_id = game["home_id"] if away_offense else game["away_id"]
            probable_id = game["home_pitcher_id"] if away_offense else game["away_pitcher_id"]
            probable_name = game["home_pitcher_name"] if away_offense else game["away_pitcher_name"]
            selected_team = match_statcast_team(batting_code, available_teams)
            opponent_team = match_statcast_team(opponent_code, available_teams) or opponent_code

            if selected_team is None:
                st.warning(
                    f"{batting_code} was not found in the loaded Statcast sample. "
                    "Choose the matching team manually."
                )
                selected_team = st.selectbox(
                    "Batting team in Statcast data",
                    available_teams,
                    key=f"{key_prefix}_team_fallback",
                )
                batting_team_id = team_id_from_code(selected_team) or batting_team_id

            probable_row = pitcher_row_from_id(pitcher_summary, probable_id)
            if probable_row is None:
                st.info(
                    f"{probable_name} is not available in the loaded pitcher sample, "
                    "or the probable pitcher is still TBD. Select the starter below."
                )
                preferred = pitcher_summary[
                    pitcher_summary["Pitcher_Team"].astype(str).eq(str(opponent_team))
                ]
                choices = preferred if not preferred.empty else pitcher_summary
                selected_display = st.selectbox(
                    "Starting pitcher override",
                    choices["Display"].tolist(),
                    key=f"{key_prefix}_pitcher_override_{selected_game_pk}_{batting_code}",
                )
                probable_row = choices.loc[choices["Display"].eq(selected_display)].iloc[0]
            else:
                selected_display = str(probable_row["Display"])
                st.caption(f"Starter selected automatically: {selected_display}")

            matchup = {
                "pitcher_id": int(probable_row["pitcher"]),
                "pitcher_display": selected_display,
                "pitcher_name": str(probable_row["Player_Name"]),
                "pitcher_team": str(probable_row["Pitcher_Team"]),
                "batting_team": selected_team,
                "home_away": "Away" if away_offense else "Home",
                "venue": game["venue"],
                "time_et": game["time_et"],
                "status": game["status"],
                "away_abbr": game["away_abbr"],
                "home_abbr": game["home_abbr"],
                "away_id": game["away_id"],
                "home_id": game["home_id"],
                "game_pk": game["game_pk"],
                "game_datetime_utc": game.get("game_datetime_utc"),
                "batting_team_id": batting_team_id,
                "opponent_team_id": opponent_team_id,
                "slate_date": slate_date.isoformat(),
                "probables": f"Probables: {game['away_pitcher_name']} vs {game['home_pitcher_name']}",
            }
            render_matchup_hero(matchup, model_label)
            return matchup

        st.info("No MLB games were returned for that date. Use manual matchup mode below.")

    matchup = manual_matchup_controls(pitcher_summary, available_teams, key_prefix)
    matchup["away_id"] = team_id_from_code(matchup.get("away_abbr"))
    matchup["home_id"] = team_id_from_code(matchup.get("home_abbr"))
    matchup["probables"] = f"Selected starter: {matchup.get('pitcher_name', '')}"
    render_matchup_hero(matchup, model_label)
    return matchup


def render_leader_cards(
    rankings: pd.DataFrame,
    probability_column: str,
    score_column: str,
    probability_label: str,
) -> None:
    leaders = rankings.head(3)
    columns = st.columns(3)
    for index, (_, row) in enumerate(leaders.iterrows()):
        with columns[index]:
            lineup_text = "—" if pd.isna(row.get("LineupSpot")) else f"#{int(row['LineupSpot'])} lineup"
            confidence_text = row.get("Confidence_Level", "")
            expected_count = pd.to_numeric(pd.Series([row.get(BINARY_EXPECTED_COUNT_COLUMN)]), errors="coerce").iloc[0]
            expected_text = (
                f" · {BINARY_EXPECTED_COUNT_LABEL.lower()} {expected_count:.2f}"
                if pd.notna(expected_count) else ""
            )
            st.markdown(
                f"""
                <div class="leader-card">
                    <div class="leader-rank">RANK {int(row['Rank'])}</div>
                    <div class="leader-name">{row['Player']}</div>
                    <div class="leader-prob">{row[probability_column]:.1%}</div>
                    <div class="leader-sub">{probability_label}{expected_text} · {score_column.replace('Score', ' score')} {row[score_column]:.1f}</div>
                    <div class="leader-sub">{lineup_text} · {confidence_text} confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )



# -----------------------------------------------------------------------------
# Binary outcome backtesting and probability calibration
# -----------------------------------------------------------------------------
BINARY_MODEL_VERSION = "hits-backtest-calibration-v1"
BINARY_TARGET_LABEL = "1+ Hit"
BINARY_TARGET_SLUG = "hits"
BINARY_PROBABILITY_COLUMN = "Model_1plus_Hit"
BINARY_RAW_PROBABILITY_COLUMN = "Raw_Model_1plus_Hit"
BINARY_SCORE_COLUMN = "HitScore"
BINARY_EVENT_KIND = "hit"
BINARY_EXPECTED_COUNT_COLUMN = "Projected_Hits"
BINARY_EXPECTED_COUNT_LABEL = "Projected Hits"
BINARY_PER_PA_COLUMN = "Model_Hit_Per_PA"
BINARY_MIN_CALIBRATION_ROWS = 200


def american_odds_to_probability(value: object) -> float:
    odds = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(odds) or float(odds) == 0:
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def _clip_probability(values: object) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)


def _sigmoid(values: object) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(values: object) -> np.ndarray:
    p = _clip_probability(values)
    return np.log(p / (1.0 - p))


def parse_binary_calibration_upload(uploaded: object) -> tuple[dict | None, str | None]:
    if uploaded is None:
        return None, None
    try:
        content = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8-sig")
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("Calibration file must contain a JSON object.")
        if payload.get("target_slug") not in {None, BINARY_TARGET_SLUG}:
            raise ValueError(
                f"This file targets {payload.get('target_slug')}, not {BINARY_TARGET_SLUG}."
            )
        if not payload.get("ok", True):
            raise ValueError("Calibration file does not contain a usable fitted model.")
        for field in ["intercept", "slope"]:
            if pd.isna(pd.to_numeric(pd.Series([payload.get(field)]), errors="coerce").iloc[0]):
                raise ValueError(f"Calibration JSON is missing a numeric {field}.")
        return payload, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def apply_binary_probability_calibration(
    board: pd.DataFrame,
    calibration: dict | None,
) -> pd.DataFrame:
    output = board.copy()
    if BINARY_RAW_PROBABILITY_COLUMN not in output.columns:
        output[BINARY_RAW_PROBABILITY_COLUMN] = pd.to_numeric(
            output.get(BINARY_PROBABILITY_COLUMN), errors="coerce"
        )
    raw = pd.to_numeric(output[BINARY_RAW_PROBABILITY_COLUMN], errors="coerce")
    output["Calibration_Applied"] = False
    output["Calibration_Version"] = ""
    if calibration:
        intercept = pd.to_numeric(pd.Series([calibration.get("intercept")]), errors="coerce").iloc[0]
        slope = pd.to_numeric(pd.Series([calibration.get("slope")]), errors="coerce").iloc[0]
        if pd.notna(intercept) and pd.notna(slope):
            calibrated = _sigmoid(float(intercept) + float(slope) * _logit(raw.fillna(0.5)))
            output[BINARY_PROBABILITY_COLUMN] = np.clip(calibrated, 0.001, 0.999)
            output["Calibration_Applied"] = True
            output["Calibration_Version"] = str(
                calibration.get("created_at_utc") or calibration.get("version") or "uploaded"
            )
    output = output.drop(columns=["Rank"], errors="ignore")
    output = output.sort_values(
        [BINARY_PROBABILITY_COLUMN, BINARY_SCORE_COLUMN], ascending=False
    ).reset_index(drop=True)
    output.insert(0, "Rank", np.arange(1, len(output) + 1))
    return output


def build_binary_projection_snapshot(
    board: pd.DataFrame,
    matchup: dict,
    lookback_days: int,
    loaded_start: object,
    loaded_end: object,
    lineup_status: str,
    extra_context: dict | None = None,
    generated_at_utc: str | None = None,
) -> pd.DataFrame:
    snapshot = board.copy()
    if generated_at_utc is None:
        generated_at_utc = pd.Timestamp.now(tz="UTC").isoformat()
    snapshot["SlateDate"] = str(pd.Timestamp(matchup.get("slate_date") or date.today()).date())
    snapshot["GameDateTimeUTC"] = matchup.get("game_datetime_utc")
    snapshot["GeneratedAtUTC"] = str(generated_at_utc)
    snapshot["ModelVersion"] = BINARY_MODEL_VERSION
    snapshot["Target"] = BINARY_TARGET_LABEL
    snapshot["LookbackDays"] = int(lookback_days)
    snapshot["StatcastStart"] = str(pd.Timestamp(loaded_start).date())
    snapshot["StatcastEnd"] = str(pd.Timestamp(loaded_end).date())
    def preserve_or_fill(column: str, fallback: object) -> None:
        if column not in snapshot.columns:
            snapshot[column] = fallback
            return
        existing = snapshot[column]
        missing = existing.isna()
        if existing.dtype == object:
            missing = missing | existing.astype(str).str.strip().eq("")
        snapshot[column] = existing.where(~missing, fallback)

    preserve_or_fill("GamePK", matchup.get("game_pk"))
    snapshot["PlayerID"] = pd.to_numeric(snapshot.get("player_id", snapshot.get("PlayerID")), errors="coerce")
    preserve_or_fill("Team", str(matchup.get("batting_team") or ""))
    preserve_or_fill("Opponent", str(matchup.get("pitcher_team") or ""))
    preserve_or_fill("HomeAway", str(matchup.get("home_away") or ""))
    preserve_or_fill("Venue", str(matchup.get("venue") or ""))
    preserve_or_fill("StartingPitcherID", matchup.get("pitcher_id"))
    preserve_or_fill("StartingPitcher", str(matchup.get("pitcher_name") or ""))
    preserve_or_fill("LineupStatus", str(lineup_status or "Unknown"))
    preserve_or_fill("GameDateTimeUTC", matchup.get("game_datetime_utc"))
    snapshot["RawProbability"] = pd.to_numeric(
        snapshot.get(BINARY_RAW_PROBABILITY_COLUMN, snapshot.get(BINARY_PROBABILITY_COLUMN)),
        errors="coerce",
    )
    snapshot["ModelProbability"] = pd.to_numeric(
        snapshot.get(BINARY_PROBABILITY_COLUMN), errors="coerce"
    )
    snapshot["ModelScore"] = pd.to_numeric(snapshot.get(BINARY_SCORE_COLUMN), errors="coerce")
    snapshot["ExpectedCount"] = pd.to_numeric(snapshot.get(BINARY_EXPECTED_COUNT_COLUMN), errors="coerce")
    snapshot["ModelPerPA"] = pd.to_numeric(snapshot.get(BINARY_PER_PA_COLUMN), errors="coerce")
    snapshot["ModelOverProbability"] = pd.to_numeric(
        snapshot["Model_Over_Prob"] if "Model_Over_Prob" in snapshot.columns else snapshot["ModelProbability"],
        errors="coerce",
    )
    snapshot["ProjectionEdge"] = pd.to_numeric(
        snapshot["Projection_Edge"] if "Projection_Edge" in snapshot.columns else pd.Series(np.nan, index=snapshot.index),
        errors="coerce",
    )
    snapshot["SportsbookOdds"] = pd.to_numeric(
        snapshot["Over_Odds"] if "Over_Odds" in snapshot.columns else pd.Series(np.nan, index=snapshot.index),
        errors="coerce",
    )
    snapshot["MarketProbability"] = pd.to_numeric(
        snapshot["Market_Over_Prob"] if "Market_Over_Prob" in snapshot.columns else pd.Series(np.nan, index=snapshot.index),
        errors="coerce",
    )
    snapshot["MarketImpliedProbability"] = snapshot["SportsbookOdds"].map(american_odds_to_probability)
    snapshot["ModelMarketEdge"] = pd.to_numeric(
        snapshot["Model_Market_Edge"] if "Model_Market_Edge" in snapshot.columns else pd.Series(np.nan, index=snapshot.index),
        errors="coerce",
    )
    snapshot["Actual_Event"] = np.nan
    snapshot["Actual_Hits"] = np.nan
    snapshot["Actual_HR"] = np.nan
    snapshot["Actual_PA"] = np.nan
    snapshot["Actual_AB"] = np.nan
    snapshot["ResultStatus"] = ""
    snapshot["Notes"] = ""
    if extra_context:
        for key, value in extra_context.items():
            snapshot[key] = value
    preferred = [
        "SlateDate", "GameDateTimeUTC", "GeneratedAtUTC", "ModelVersion", "Target",
        "LookbackDays", "StatcastStart", "StatcastEnd", "GamePK", "PlayerID", "Player",
        "Team", "Opponent", "HomeAway", "Venue", "StartingPitcherID", "StartingPitcher",
        "LineupStatus", "LineupSpot", "Projected_PA", "ModelPerPA", "ExpectedCount", "EffectiveStand", "SampleStatus",
        "Confidence", "Confidence_Level", "ParkFactor", "WeatherMultiplier",
        "RawProbability", "ModelProbability", "ModelScore", "Calibration_Applied",
        "Calibration_Version", "Market_Line", "Over_Odds", "Under_Odds", "Line_Source", "Line_Updated", "SportsbookOdds", "MarketProbability",
        "ModelOverProbability", "ProjectionEdge",
        "MarketImpliedProbability", "ModelMarketEdge", "Actual_Event", "Actual_Hits",
        "Actual_HR", "Actual_PA", "Actual_AB", "ResultStatus", "Notes",
    ]
    existing = [column for column in preferred if column in snapshot.columns]
    extras = [
        column for column in snapshot.columns
        if column not in existing
        and column not in {"Rank", "player_id", BINARY_PROBABILITY_COLUMN, BINARY_RAW_PROBABILITY_COLUMN, BINARY_SCORE_COLUMN}
    ]
    return snapshot[existing + extras].copy()


def normalize_binary_history(history: pd.DataFrame) -> pd.DataFrame:
    if history is None:
        history = pd.DataFrame()
    result = history.copy()
    aliases = {
        "Date": "SlateDate", "Game_ID": "GamePK", "Player_ID": "PlayerID",
        "Probability": "ModelProbability", "Raw_Probability": "RawProbability",
        "Score": "ModelScore", "Actual": "Actual_Event",
    }
    result = result.rename(
        columns={key: value for key, value in aliases.items() if key in result.columns and value not in result.columns}
    )
    text_defaults = {
        "Player": "", "Team": "", "Opponent": "", "LineupStatus": "Unknown",
        "Confidence_Level": "Unknown", "ModelVersion": "unknown", "Target": BINARY_TARGET_LABEL,
        "ResultStatus": "", "Notes": "",
    }
    for column, default in text_defaults.items():
        if column not in result.columns:
            result[column] = default
    for column in ["SlateDate", "GameDateTimeUTC", "GeneratedAtUTC"]:
        if column not in result.columns:
            result[column] = pd.NaT if column != "SlateDate" else ""
    result["SlateDate"] = pd.to_datetime(result["SlateDate"], errors="coerce").dt.date
    result["GameDateTimeUTC"] = pd.to_datetime(result["GameDateTimeUTC"], utc=True, errors="coerce")
    result["GeneratedAtUTC"] = pd.to_datetime(result["GeneratedAtUTC"], utc=True, errors="coerce")
    numeric_columns = [
        "GamePK", "PlayerID", "StartingPitcherID", "LookbackDays", "LineupSpot",
        "Projected_PA", "ModelPerPA", "ExpectedCount", "Confidence", "ParkFactor", "WeatherMultiplier", "RawProbability",
        "ModelProbability", "ModelScore", "ModelOverProbability", "ProjectionEdge", "SportsbookOdds", "MarketProbability",
        "MarketImpliedProbability", "ModelMarketEdge", "Actual_Event", "Actual_Hits",
        "Actual_HR", "Actual_PA", "Actual_AB",
    ]
    for column in numeric_columns:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    odds_implied = result["SportsbookOdds"].map(american_odds_to_probability)
    result["MarketImpliedProbability"] = result["MarketImpliedProbability"].fillna(odds_implied)
    fair_market = result["MarketProbability"].where(
        result["MarketProbability"].between(0, 1, inclusive="both"),
        result["MarketImpliedProbability"],
    )
    result["ModelMarketEdge"] = result["ModelProbability"] - fair_market
    result["SnapshotAfterStart"] = (
        result["GeneratedAtUTC"].notna()
        & result["GameDateTimeUTC"].notna()
        & result["GeneratedAtUTC"].gt(result["GameDateTimeUTC"])
    )
    return result


def combine_binary_history_uploads(uploaded_files: list[object]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for uploaded in uploaded_files or []:
        try:
            content = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8-sig")
            frame = pd.read_csv(StringIO(content))
            if not frame.empty:
                frame["SourceFile"] = getattr(uploaded, "name", "uploaded.csv")
                frames.append(frame)
        except Exception as exc:
            errors.append(f"{getattr(uploaded, 'name', 'file')}: {type(exc).__name__}: {exc}")
    if not frames:
        return pd.DataFrame(), errors
    return normalize_binary_history(pd.concat(frames, ignore_index=True, sort=False)), errors


def deduplicate_binary_history(
    history: pd.DataFrame,
    latest_only: bool = True,
    exclude_after_start: bool = True,
) -> pd.DataFrame:
    result = normalize_binary_history(history)
    if result.empty:
        return result
    if exclude_after_start:
        result = result[~result["SnapshotAfterStart"].fillna(False)].copy()
    if latest_only:
        result = result.sort_values("GeneratedAtUTC", na_position="first")
        game_key = result["GamePK"].where(
            result["GamePK"].notna(),
            result["SlateDate"].astype(str) + "-" + result["Team"].astype(str),
        )
        result["_GameKey"] = game_key.astype(str)
        result = result.drop_duplicates(["_GameKey", "PlayerID"], keep="last")
        result = result.drop(columns="_GameKey")
    return result.reset_index(drop=True)


@st.cache_data(ttl=1800, max_entries=120, show_spinner=False)
def fetch_completed_batter_results(game_pk: int) -> dict:
    game_pk = int(game_pk)
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    request = Request(url, headers={"User-Agent": "MLB-Hitter-Backtest/1.0", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except Exception as exc:
        return {"ok": False, "final": False, "data": pd.DataFrame(), "error": f"{type(exc).__name__}: {exc}"}
    game_data = payload.get("gameData", {}) or {}
    status = game_data.get("status", {}) or {}
    abstract_state = str(status.get("abstractGameState", ""))
    detailed_state = str(status.get("detailedState", ""))
    final = abstract_state.lower() == "final" or any(
        token in detailed_state.lower() for token in ["final", "game over", "completed early"]
    )
    if not final:
        return {
            "ok": True, "final": False, "data": pd.DataFrame(),
            "status": detailed_state or abstract_state or "Not final", "error": None,
        }
    rows: list[dict] = []
    teams = ((payload.get("liveData", {}) or {}).get("boxscore", {}) or {}).get("teams", {}) or {}
    for side in ["away", "home"]:
        players = ((teams.get(side, {}) or {}).get("players", {}) or {})
        for record in players.values():
            if not isinstance(record, dict):
                continue
            person = record.get("person", {}) or {}
            batting = ((record.get("stats", {}) or {}).get("batting", {}) or {})
            player_id = pd.to_numeric(pd.Series([person.get("id")]), errors="coerce").iloc[0]
            pa = pd.to_numeric(pd.Series([batting.get("plateAppearances")]), errors="coerce").iloc[0]
            if pd.isna(player_id) or pd.isna(pa) or float(pa) <= 0:
                continue
            rows.append({
                "PlayerID": int(player_id),
                "Player_Official": str(person.get("fullName") or ""),
                "Actual_Hits": pd.to_numeric(pd.Series([batting.get("hits")]), errors="coerce").fillna(0).iloc[0],
                "Actual_HR": pd.to_numeric(pd.Series([batting.get("homeRuns")]), errors="coerce").fillna(0).iloc[0],
                "Actual_PA": float(pa),
                "Actual_AB": pd.to_numeric(pd.Series([batting.get("atBats")]), errors="coerce").fillna(0).iloc[0],
            })
    return {
        "ok": True, "final": True, "data": pd.DataFrame(rows),
        "status": detailed_state or "Final", "error": None,
    }


def fill_binary_actual_results(history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    result = normalize_binary_history(history)
    summary = {"matched": 0, "not_final": 0, "unmatched": 0, "errors": []}
    if result.empty:
        return result, summary
    game_ids = pd.to_numeric(result.get("GamePK"), errors="coerce").dropna().astype(int).unique().tolist()
    progress = st.progress(0.0, text="Fetching official hitter results...")
    total = max(len(game_ids), 1)
    for index, game_pk in enumerate(game_ids, start=1):
        game_result = fetch_completed_batter_results(int(game_pk))
        if not game_result.get("ok"):
            summary["errors"].append(f"Game {game_pk}: {game_result.get('error')}")
        elif not game_result.get("final"):
            summary["not_final"] += 1
        else:
            official = game_result.get("data", pd.DataFrame())
            if official is None or official.empty:
                summary["unmatched"] += int(result["GamePK"].eq(game_pk).sum())
            else:
                official = official.set_index("PlayerID")
                game_mask = result["GamePK"].eq(game_pk)
                for row_index in result.index[game_mask]:
                    player_id = result.at[row_index, "PlayerID"]
                    if pd.isna(player_id) or int(player_id) not in official.index:
                        summary["unmatched"] += 1
                        continue
                    actual = official.loc[int(player_id)]
                    if isinstance(actual, pd.DataFrame):
                        actual = actual.iloc[0]
                    for column in ["Actual_Hits", "Actual_HR", "Actual_PA", "Actual_AB"]:
                        result.at[row_index, column] = actual.get(column)
                    result.at[row_index, "Actual_Event"] = float(
                        actual.get("Actual_HR", 0) > 0
                        if BINARY_EVENT_KIND == "hr"
                        else actual.get("Actual_Hits", 0) > 0
                    )
                    result.at[row_index, "ResultStatus"] = str(game_result.get("status") or "Final")
                    summary["matched"] += 1
        progress.progress(index / total, text=f"Checked {index} of {total} games")
    progress.empty()
    return normalize_binary_history(result), summary


def binary_probability_metrics(history: pd.DataFrame, probability_column: str = "ModelProbability") -> dict:
    if history is None or history.empty or probability_column not in history.columns:
        return {"N": 0, "Mean_Probability": np.nan, "Actual_Rate": np.nan, "Brier": np.nan, "Log_Loss": np.nan, "Bias": np.nan}
    frame = history[[probability_column, "Actual_Event"]].copy()
    frame[probability_column] = pd.to_numeric(frame[probability_column], errors="coerce")
    frame["Actual_Event"] = pd.to_numeric(frame["Actual_Event"], errors="coerce")
    frame = frame.dropna()
    frame = frame[frame["Actual_Event"].isin([0, 1])]
    if frame.empty:
        return {"N": 0, "Mean_Probability": np.nan, "Actual_Rate": np.nan, "Brier": np.nan, "Log_Loss": np.nan, "Bias": np.nan}
    p = _clip_probability(frame[probability_column])
    y = frame["Actual_Event"].to_numpy(float)
    return {
        "N": int(len(frame)),
        "Mean_Probability": float(np.mean(p)),
        "Actual_Rate": float(np.mean(y)),
        "Brier": float(np.mean(np.square(p - y))),
        "Log_Loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "Bias": float(np.mean(p - y)),
    }


def binary_probability_calibration_table(history: pd.DataFrame) -> pd.DataFrame:
    frame = history[["ModelProbability", "Actual_Event"]].copy()
    frame["ModelProbability"] = pd.to_numeric(frame["ModelProbability"], errors="coerce")
    frame["Actual_Event"] = pd.to_numeric(frame["Actual_Event"], errors="coerce")
    frame = frame.dropna()
    if frame.empty:
        return pd.DataFrame()
    if BINARY_EVENT_KIND == "hr":
        bins = [0, .05, .08, .11, .14, .17, .20, .25, .35, 1.001]
        labels = ["<5%", "5–7%", "8–10%", "11–13%", "14–16%", "17–19%", "20–24%", "25–34%", "35%+"]
    else:
        bins = [0, .40, .45, .50, .55, .60, .65, .70, .75, .80, 1.001]
        labels = ["<40%", "40–44%", "45–49%", "50–54%", "55–59%", "60–64%", "65–69%", "70–74%", "75–79%", "80%+"]
    frame["Probability_Bucket"] = pd.cut(
        frame["ModelProbability"], bins=bins, labels=labels, right=False, include_lowest=True
    )
    grouped = frame.groupby("Probability_Bucket", observed=False).agg(
        Sample=("Actual_Event", "size"),
        Average_Probability=("ModelProbability", "mean"),
        Actual_Rate=("Actual_Event", "mean"),
    ).reset_index()
    grouped["Calibration_Gap"] = grouped["Actual_Rate"] - grouped["Average_Probability"]
    return grouped


def binary_score_bucket_table(history: pd.DataFrame) -> pd.DataFrame:
    frame = history[["ModelScore", "ModelProbability", "Actual_Event"]].copy()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["ModelScore", "Actual_Event"])
    if frame.empty:
        return pd.DataFrame()
    bins = [-0.001, 30, 45, 55, 70, 85, 100.001]
    labels = ["0–29", "30–44", "45–54", "55–69", "70–84", "85–100"]
    frame["Score_Bucket"] = pd.cut(frame["ModelScore"], bins=bins, labels=labels, right=False, include_lowest=True)
    return frame.groupby("Score_Bucket", observed=False).agg(
        Sample=("Actual_Event", "size"),
        Average_Score=("ModelScore", "mean"),
        Average_Probability=("ModelProbability", "mean"),
        Actual_Rate=("Actual_Event", "mean"),
    ).reset_index()


def grouped_binary_metrics(history: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if group_column not in history.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for group_value, group in history.groupby(group_column, dropna=False):
        metrics = binary_probability_metrics(group)
        if metrics["N"]:
            rows.append({"Group": str(group_value), **metrics})
    return pd.DataFrame(rows)


def _fit_platt_coefficients(probabilities: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
    x = _logit(probabilities)
    y = np.asarray(outcomes, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.array([0.0, 1.0], dtype=float)
    regularization = np.diag([1e-6, 1e-3])
    for _ in range(80):
        predicted = _sigmoid(design @ beta)
        weights = np.clip(predicted * (1 - predicted), 1e-6, None)
        gradient = design.T @ (predicted - y) + regularization @ beta
        hessian = design.T @ (design * weights[:, None]) + regularization
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    intercept = float(np.clip(beta[0], -5.0, 5.0))
    slope = float(np.clip(beta[1], 0.10, 3.0))
    return intercept, slope


def fit_binary_calibration(history: pd.DataFrame, min_rows: int) -> dict:
    required = ["RawProbability", "Actual_Event", "SlateDate"]
    if any(column not in history.columns for column in required):
        return {"ok": False, "n": 0, "message": "Required probability and result columns are missing."}
    frame = history[required].copy()
    frame["RawProbability"] = pd.to_numeric(frame["RawProbability"], errors="coerce")
    frame["Actual_Event"] = pd.to_numeric(frame["Actual_Event"], errors="coerce")
    frame["SlateDate"] = pd.to_datetime(frame["SlateDate"], errors="coerce")
    frame = frame.dropna(subset=["RawProbability", "Actual_Event"]).sort_values("SlateDate")
    frame = frame[frame["Actual_Event"].isin([0, 1])]
    if len(frame) < min_rows or frame["Actual_Event"].nunique() < 2 or frame["RawProbability"].nunique() < 3:
        return {
            "ok": False, "n": int(len(frame)),
            "message": f"Need at least {min_rows} completed hitter-games with both outcomes and varied probabilities.",
        }
    p = _clip_probability(frame["RawProbability"])
    y = frame["Actual_Event"].to_numpy(float)
    intercept, slope = _fit_platt_coefficients(p, y)
    calibrated = _sigmoid(intercept + slope * _logit(p))
    raw_brier = float(np.mean(np.square(p - y)))
    calibrated_brier = float(np.mean(np.square(calibrated - y)))
    raw_logloss = float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p)))
    calibrated_logloss = float(-np.mean(y * np.log(_clip_probability(calibrated)) + (1-y) * np.log(1-_clip_probability(calibrated))))

    split = max(int(len(frame) * 0.70), 1)
    train, test = frame.iloc[:split], frame.iloc[split:]
    holdout = {
        "holdout_n": int(len(test)), "holdout_raw_brier": np.nan,
        "holdout_calibrated_brier": np.nan, "holdout_raw_logloss": np.nan,
        "holdout_calibrated_logloss": np.nan,
    }
    if len(train) >= max(30, min_rows // 2) and len(test) >= 20 and train["Actual_Event"].nunique() >= 2:
        train_p = _clip_probability(train["RawProbability"])
        train_y = train["Actual_Event"].to_numpy(float)
        test_p = _clip_probability(test["RawProbability"])
        test_y = test["Actual_Event"].to_numpy(float)
        h_intercept, h_slope = _fit_platt_coefficients(train_p, train_y)
        test_cal = _sigmoid(h_intercept + h_slope * _logit(test_p))
        holdout = {
            "holdout_n": int(len(test)),
            "holdout_raw_brier": float(np.mean(np.square(test_p - test_y))),
            "holdout_calibrated_brier": float(np.mean(np.square(test_cal - test_y))),
            "holdout_raw_logloss": float(-np.mean(test_y*np.log(test_p) + (1-test_y)*np.log(1-test_p))),
            "holdout_calibrated_logloss": float(-np.mean(test_y*np.log(_clip_probability(test_cal)) + (1-test_y)*np.log(1-_clip_probability(test_cal)))),
        }
    return {
        "ok": True,
        "version": BINARY_MODEL_VERSION,
        "target": BINARY_TARGET_LABEL,
        "target_slug": BINARY_TARGET_SLUG,
        "method": "platt_scaling_on_logit_probability",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "n": int(len(frame)),
        "intercept": intercept,
        "slope": slope,
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "raw_logloss": raw_logloss,
        "calibrated_logloss": calibrated_logloss,
        **holdout,
    }


def render_binary_backtest_tab(
    rankings: pd.DataFrame,
    matchup: dict,
    lookback_days: int,
    loaded_start: object,
    loaded_end: object,
    lineup_status: str,
    calibration_state_key: str,
    widget_prefix: str,
    extra_context: dict | None = None,
    whole_slate_board: pd.DataFrame | None = None,
    odds_api_key: str = "",
    odds_widget_prefix: str = "",
    odds_source_mode: str | None = None,
) -> None:
    st.markdown(f"### {BINARY_TARGET_LABEL} backtesting and calibration")
    st.info(
        "Save the projection snapshot before first pitch. After the game is final, upload the snapshot or your latest master CSV and fetch official results."
    )

    slate_date_text = str(pd.Timestamp(matchup.get("slate_date") or date.today()).date())
    usable_slate = isinstance(whole_slate_board, pd.DataFrame) and not whole_slate_board.empty
    scope_options = ["Current matchup"] + (["Whole slate"] if usable_slate else [])
    snapshot_scope = st.radio(
        "Pregame snapshot scope", scope_options, horizontal=True,
        index=(1 if usable_slate else 0), key=f"{widget_prefix}_snapshot_scope",
    )
    source_board = whole_slate_board.copy() if snapshot_scope == "Whole slate" else rankings.copy()
    if snapshot_scope == "Whole slate":
        source_board = apply_binary_probability_calibration(
            source_board, st.session_state.get(calibration_state_key)
        )
    elif not usable_slate:
        st.caption(
            "To create a pitcher-style full-slate snapshot, first build the whole-slate board in Slate Top 10 & pairings. "
            "The current-matchup snapshot remains available now."
        )

    odds_widget_prefix = odds_widget_prefix or widget_prefix
    quotes_key = f"{odds_widget_prefix}_odds_quotes_{slate_date_text}"
    odds_source_mode = odds_source_mode or st.session_state.get(
        f"{odds_widget_prefix}_odds_source", ODDS_SOURCE_OPTIONS[0]
    )
    auto_cols = st.columns([2, 1])
    with auto_cols[0]:
        st.caption(
            "Automatic lines are event-by-event for hitter props. Fetch them here before downloading the snapshot; "
            "the saved line, prices, source and no-vig probability will then appear just like the pitcher snapshot."
        )
    with auto_cols[1]:
        fetch_snapshot_lines = st.button(
            "Fetch pregame lines", type="primary", width="stretch",
            key=f"{widget_prefix}_fetch_snapshot_lines",
        )
    if fetch_snapshot_lines:
        if not odds_api_key:
            st.error("Add THE_ODDS_API_KEY in this Streamlit app's Secrets or use the temporary sidebar field.")
        else:
            games, schedule_error = fetch_mlb_schedule(slate_date_text)
            if schedule_error:
                st.error(f"MLB schedule unavailable: {schedule_error}")
            elif not games:
                st.error("No MLB games were available for the selected date.")
            else:
                if snapshot_scope == "Whole slate":
                    game_values = pd.to_numeric(source_board.get("GamePK"), errors="coerce").dropna().astype(int).unique().tolist()
                else:
                    current_pk = pd.to_numeric(pd.Series([matchup.get("game_pk")]), errors="coerce").iloc[0]
                    game_values = [int(current_pk)] if pd.notna(current_pk) else []
                with st.spinner(f"Fetching {ODDS_API_MARKET_LABEL.lower()} lines for the pregame snapshot..."):
                    new_quotes, quota, errors = fetch_batter_prop_quotes(odds_api_key, games, game_values)
                existing_quotes = st.session_state.get(quotes_key, pd.DataFrame())
                st.session_state[quotes_key] = combine_batter_quote_frames(existing_quotes, new_quotes, slate_date_text)
                st.session_state[f"{widget_prefix}_snapshot_odds_quota"] = quota
                st.session_state[f"{widget_prefix}_snapshot_odds_errors"] = errors
                st.rerun()

    snapshot_quotes = st.session_state.get(quotes_key, pd.DataFrame())
    source_board = apply_batter_odds_to_board(
        source_board, snapshot_quotes, odds_source_mode, matchup.get("game_pk")
    )
    quota = st.session_state.get(f"{widget_prefix}_snapshot_odds_quota", {})
    line_count = int(pd.to_numeric(source_board.get("Market_Line"), errors="coerce").notna().sum()) if not source_board.empty else 0
    status_cols = st.columns(3)
    status_cols[0].metric("Players in snapshot", f"{len(source_board):,}")
    status_cols[1].metric("Automatic lines matched", f"{line_count:,}")
    status_cols[2].metric("API credits remaining", quota.get("requests_remaining", "—"))
    odds_errors = st.session_state.get(f"{widget_prefix}_snapshot_odds_errors", [])
    if odds_errors:
        with st.expander("Pregame line-fetch details"):
            for error in odds_errors[:50]:
                st.code(error)

    fingerprint = f"{matchup.get('game_pk')}_{matchup.get('slate_date')}_{matchup.get('batting_team')}_{BINARY_TARGET_SLUG}_{snapshot_scope}"
    timestamp_key = f"{widget_prefix}_snapshot_timestamp_{fingerprint}"
    if timestamp_key not in st.session_state:
        st.session_state[timestamp_key] = pd.Timestamp.now(tz="UTC").isoformat()
    snapshot = build_binary_projection_snapshot(
        source_board, matchup, lookback_days, loaded_start, loaded_end, lineup_status,
        extra_context=extra_context,
        generated_at_utc=st.session_state[timestamp_key],
    )

    st.markdown("#### 1. Save today’s pregame predictions")
    st.caption(
        "Automatic values are copied from the selected odds source. You can still overwrite the line or prices manually when a market is missing."
    )
    editor_columns = [
        "PlayerID", "Player", "Team", "Opponent", "LineupSpot", "Projected_PA",
        "ModelPerPA", "ExpectedCount", "ModelProbability", "ModelScore",
        "Market_Line", "Over_Odds", "Under_Odds", "MarketProbability", "Line_Source", "Notes",
    ]
    editable = snapshot[[column for column in editor_columns if column in snapshot.columns]].copy()
    editable = editable.rename(columns={
        "LineupSpot": "Order", "Projected_PA": "Proj PA", "ModelPerPA": "Per PA",
        "ExpectedCount": "Projected Hits", "ModelProbability": "1+ Hit",
        "ModelScore": "Model Score", "Market_Line": "Line", "Over_Odds": "Over Odds",
        "Under_Odds": "Under Odds", "MarketProbability": "No-vig Market",
        "Line_Source": "Line Source",
    })
    editable_columns = ["Line", "Over Odds", "Under Odds", "No-vig Market", "Notes"]
    edited = st.data_editor(
        editable,
        hide_index=True,
        width="stretch",
        disabled=[column for column in editable.columns if column not in editable_columns],
        column_config={
            "Line": st.column_config.NumberColumn("Line", min_value=0.0, step=0.5, format="%.1f"),
            "Over Odds": st.column_config.NumberColumn("Over odds", step=1),
            "Under Odds": st.column_config.NumberColumn("Under odds", step=1),
            "No-vig Market": st.column_config.NumberColumn(
                "No-vig market", min_value=0.0, max_value=1.0, step=0.01, format="%.3f"
            ),
        },
        key=f"{widget_prefix}_pregame_market_editor_{fingerprint}",
    )
    if not edited.empty:
        edited_values = edited.set_index("PlayerID")
        reverse_columns = {
            "Line": "Market_Line", "Over Odds": "Over_Odds", "Under Odds": "Under_Odds",
            "No-vig Market": "MarketProbability", "Notes": "Notes",
        }
        for row_index in snapshot.index:
            player_id = snapshot.at[row_index, "PlayerID"]
            if pd.isna(player_id) or player_id not in edited_values.index:
                continue
            row = edited_values.loc[player_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for display_column, snapshot_column in reverse_columns.items():
                if display_column in row.index:
                    snapshot.at[row_index, snapshot_column] = row[display_column]
    snapshot["SportsbookOdds"] = pd.to_numeric(snapshot.get("Over_Odds"), errors="coerce")
    manual_fair = pd.to_numeric(snapshot.get("MarketProbability"), errors="coerce")
    calculated_fair = pd.Series(
        [no_vig_over_probability_batter(o, u) for o, u in zip(snapshot.get("Over_Odds"), snapshot.get("Under_Odds"))],
        index=snapshot.index, dtype=float,
    )
    snapshot["MarketProbability"] = manual_fair.where(manual_fair.between(0, 1, inclusive="both"), calculated_fair)
    snapshot["ModelMarketEdge"] = pd.to_numeric(snapshot.get("ModelOverProbability"), errors="coerce") - snapshot["MarketProbability"]
    snapshot = normalize_binary_history(snapshot)
    slate_text = str(pd.Timestamp(matchup.get("slate_date") or date.today()).date())
    snapshot_filename = f"{BINARY_TARGET_SLUG}_pregame_{slate_text}.csv"
    st.download_button(
        "Download pregame snapshot",
        snapshot.to_csv(index=False).encode("utf-8"),
        file_name=snapshot_filename,
        mime="text/csv",
        type="primary",
        width="stretch",
    )
    if snapshot["SnapshotAfterStart"].fillna(False).any():
        st.error("This snapshot timestamp is after scheduled first pitch. It will be excluded from a fair backtest.")

    st.markdown("#### 2. Upload history and fill official results")
    uploads = st.file_uploader(
        "Upload pregame snapshots or the latest master-history CSV",
        type=["csv"],
        accept_multiple_files=True,
        key=f"{widget_prefix}_history_upload",
    )
    history_key = f"{widget_prefix}_backtest_history"
    upload_signature_key = f"{widget_prefix}_history_signature"
    signature = tuple((getattr(file, "name", ""), getattr(file, "size", None)) for file in uploads or [])
    if uploads and st.session_state.get(upload_signature_key) != signature:
        uploaded_history, upload_errors = combine_binary_history_uploads(uploads)
        st.session_state[history_key] = uploaded_history
        st.session_state[upload_signature_key] = signature
        st.session_state[f"{widget_prefix}_upload_errors"] = upload_errors
    history = normalize_binary_history(st.session_state.get(history_key, pd.DataFrame()))
    upload_errors = st.session_state.get(f"{widget_prefix}_upload_errors", [])
    if upload_errors:
        with st.expander("Upload details"):
            for error in upload_errors:
                st.code(error)

    action_columns = st.columns(2)
    with action_columns[0]:
        if st.button("Add current snapshot to session master", width="stretch", key=f"{widget_prefix}_add_snapshot"):
            history = normalize_binary_history(pd.concat([history, snapshot], ignore_index=True, sort=False))
            st.session_state[history_key] = history
            st.success("Current snapshot added. Download the master CSV before leaving or rebooting.")
    with action_columns[1]:
        if st.button("Fetch official completed-game results", type="primary", width="stretch", key=f"{widget_prefix}_fetch_results"):
            history, result_summary = fill_binary_actual_results(history)
            st.session_state[history_key] = history
            st.session_state[f"{widget_prefix}_result_summary"] = result_summary

    result_summary = st.session_state.get(f"{widget_prefix}_result_summary", {})
    if result_summary:
        st.caption(
            f"Result update: {result_summary.get('matched', 0)} matched · "
            f"{result_summary.get('not_final', 0)} games not final · "
            f"{result_summary.get('unmatched', 0)} unmatched/scratched hitters."
        )
        if result_summary.get("errors"):
            with st.expander("Result-fetch details"):
                for error in result_summary["errors"][:50]:
                    st.code(error)

    latest_only = st.checkbox(
        "Use latest pregame snapshot per player/game",
        value=True,
        key=f"{widget_prefix}_latest_only",
    )
    exclude_after_start = st.checkbox(
        "Exclude snapshots saved after first pitch",
        value=True,
        key=f"{widget_prefix}_exclude_after_start",
    )
    analysis_history = deduplicate_binary_history(history, latest_only, exclude_after_start)
    completed_history = analysis_history[analysis_history["Actual_Event"].isin([0, 1])].copy()

    counts = st.columns(4)
    counts[0].metric("Master rows", f"{len(history):,}")
    counts[1].metric("Analysis rows", f"{len(analysis_history):,}")
    counts[2].metric("Completed outcomes", f"{len(completed_history):,}")
    counts[3].metric("After-start rows", f"{int(history.get('SnapshotAfterStart', pd.Series(dtype=bool)).fillna(False).sum()):,}")
    st.download_button(
        "Download updated master history",
        normalize_binary_history(history).to_csv(index=False).encode("utf-8"),
        file_name=f"{BINARY_TARGET_SLUG}_backtest_master.csv",
        mime="text/csv",
        width="stretch",
    )

    if completed_history.empty:
        st.warning("No completed outcomes are available yet. Fetch results after games are final.")
        return

    st.markdown("#### 3. Probability accuracy")
    active_metrics = binary_probability_metrics(completed_history, "ModelProbability")
    raw_metrics = binary_probability_metrics(completed_history, "RawProbability")
    metric_table = pd.DataFrame([
        {"Probability": "Active/displayed", **active_metrics},
        {"Probability": "Raw model", **raw_metrics},
    ])
    st.dataframe(
        metric_table.style.format({
            "Mean_Probability": "{:.1%}", "Actual_Rate": "{:.1%}",
            "Brier": "{:.4f}", "Log_Loss": "{:.4f}", "Bias": "{:+.1%}",
        }),
        width="stretch", hide_index=True,
    )
    st.caption("Bias is predicted probability minus actual event rate. Positive means the model is too optimistic.")

    calibration_table = binary_probability_calibration_table(completed_history)
    if not calibration_table.empty:
        st.dataframe(
            calibration_table.style.format({
                "Average_Probability": "{:.1%}", "Actual_Rate": "{:.1%}", "Calibration_Gap": "{:+.1%}",
            }),
            width="stretch", hide_index=True,
        )

    st.markdown("#### 4. Score validation")
    score_table = binary_score_bucket_table(completed_history)
    if not score_table.empty:
        st.dataframe(
            score_table.style.format({
                "Average_Score": "{:.1f}", "Average_Probability": "{:.1%}", "Actual_Rate": "{:.1%}",
            }),
            width="stretch", hide_index=True,
        )

    market_rows = completed_history[
        completed_history["MarketProbability"].notna() | completed_history["MarketImpliedProbability"].notna()
    ].copy()
    if not market_rows.empty:
        st.markdown("#### 5. Model versus market")
        market_rows["FairMarket"] = market_rows["MarketProbability"].fillna(market_rows["MarketImpliedProbability"])
        market_rows["ModelMarketEdge"] = market_rows["ModelProbability"] - market_rows["FairMarket"]
        market_rows["EdgeBucket"] = pd.cut(
            market_rows["ModelMarketEdge"],
            bins=[-1, -.15, -.10, -.05, 0, .05, .10, .15, 1],
            labels=["≤-15%", "-15 to -10%", "-10 to -5%", "-5 to 0%", "0 to +5%", "+5 to +10%", "+10 to +15%", "+15%+"],
            include_lowest=True,
        )
        market_table = market_rows.groupby("EdgeBucket", observed=False).agg(
            Sample=("Actual_Event", "size"),
            Average_Model=("ModelProbability", "mean"),
            Average_Market=("FairMarket", "mean"),
            Actual_Rate=("Actual_Event", "mean"),
        ).reset_index()
        st.dataframe(
            market_table.style.format({
                "Average_Model": "{:.1%}", "Average_Market": "{:.1%}", "Actual_Rate": "{:.1%}",
            }), width="stretch", hide_index=True,
        )

    st.markdown("#### 6. Fit probability calibration")
    minimum_rows = st.number_input(
        "Minimum completed hitter-games",
        min_value=30,
        max_value=5000,
        value=int(BINARY_MIN_CALIBRATION_ROWS),
        step=25 if BINARY_EVENT_KIND == "hit" else 100,
        key=f"{widget_prefix}_minimum_calibration_rows",
    )
    fitted = fit_binary_calibration(completed_history, int(minimum_rows))
    if not fitted.get("ok"):
        st.info(fitted.get("message", "More completed games are required."))
    else:
        fit_table = pd.DataFrame([fitted])
        show_columns = [
            "n", "intercept", "slope", "raw_brier", "calibrated_brier",
            "raw_logloss", "calibrated_logloss", "holdout_n",
            "holdout_raw_brier", "holdout_calibrated_brier",
            "holdout_raw_logloss", "holdout_calibrated_logloss",
        ]
        st.dataframe(
            fit_table[[column for column in show_columns if column in fit_table.columns]].style.format({
                "intercept": "{:+.4f}", "slope": "{:.4f}",
                "raw_brier": "{:.4f}", "calibrated_brier": "{:.4f}",
                "raw_logloss": "{:.4f}", "calibrated_logloss": "{:.4f}",
                "holdout_raw_brier": "{:.4f}", "holdout_calibrated_brier": "{:.4f}",
                "holdout_raw_logloss": "{:.4f}", "holdout_calibrated_logloss": "{:.4f}",
            }), width="stretch", hide_index=True,
        )
        st.caption(
            "The chronological holdout fits on the earlier 70% of games and tests the later 30%. Apply calibration only when holdout Brier score or log loss improves."
        )
        controls = st.columns(2)
        with controls[0]:
            st.download_button(
                "Download calibration JSON",
                json.dumps(fitted, indent=2).encode("utf-8"),
                file_name=f"{BINARY_TARGET_SLUG}_calibration.json",
                mime="application/json",
                width="stretch",
            )
        with controls[1]:
            if st.button("Apply fitted calibration to this dashboard", width="stretch", key=f"{widget_prefix}_apply_fitted_calibration"):
                st.session_state[calibration_state_key] = fitted
                st.rerun()

    st.markdown("#### 7. Segment checks")
    segment_columns = [
        column for column in [
            "LookbackDays", "LineupStatus", "Confidence_Level", "HomeAway",
            "EffectiveStand", "SampleStatus", "ModelVersion",
        ] if column in completed_history.columns
    ]
    if segment_columns:
        segment = st.selectbox(
            "Break results down by", segment_columns, key=f"{widget_prefix}_segment_choice"
        )
        segment_table = grouped_binary_metrics(completed_history, segment)
        if not segment_table.empty:
            st.dataframe(
                segment_table.style.format({
                    "Mean_Probability": "{:.1%}", "Actual_Rate": "{:.1%}",
                    "Brier": "{:.4f}", "Log_Loss": "{:.4f}", "Bias": "{:+.1%}",
                }), width="stretch", hide_index=True,
            )

    with st.expander("Completed history used in this report"):
        st.dataframe(completed_history, width="stretch", hide_index=True, height=500)



# -----------------------------------------------------------------------------
# Automatic batter odds, whole-slate rankings, and three-person pairing tools
# -----------------------------------------------------------------------------
ODDS_API_SPORT = "baseball_mlb"
ODDS_API_MARKET_KEY = "batter_hits"
ODDS_API_MARKET_LABEL = "Batter hits"
ODDS_BOOKMAKERS = {
    "hardrockbet_fl": "Hard Rock Bet FL",
    "prizepicks": "PrizePicks",
    "fanduel": "FanDuel",
    "draftkings": "DraftKings",
    "pinnacle": "Pinnacle",
}
ODDS_SOURCE_OPTIONS = [
    "Hard Rock Bet FL",
    "PrizePicks",
    "FanDuel",
    "DraftKings",
    "Pinnacle",
    "Consensus sportsbooks",
    "Best sportsbook over line",
    "Best sportsbook under line",
]
ODDS_SOURCE_TO_KEY = {title: key for key, title in ODDS_BOOKMAKERS.items()}
SPORTSBOOK_KEYS = {"hardrockbet_fl", "fanduel", "draftkings", "pinnacle"}


def read_streamlit_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or "").strip()


def normalize_person_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    tokens = [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in {"jr", "sr", "ii", "iii", "iv"}]
    return "".join(tokens)


def player_name_match_score(target: object, candidate: object) -> float:
    target_text = normalize_person_name(target)
    candidate_text = normalize_person_name(candidate)
    if not target_text or not candidate_text:
        return 0.0
    if target_text == candidate_text:
        return 1.0
    target_tokens = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", str(target or "")).encode("ascii", "ignore").decode("ascii").lower())
    candidate_tokens = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", str(candidate or "")).encode("ascii", "ignore").decode("ascii").lower())
    suffixes = {"jr", "sr", "ii", "iii", "iv"}
    target_tokens = [token for token in target_tokens if token not in suffixes]
    candidate_tokens = [token for token in candidate_tokens if token not in suffixes]
    ratio = SequenceMatcher(None, target_text, candidate_text).ratio()
    if target_tokens and candidate_tokens and target_tokens[-1] == candidate_tokens[-1]:
        if target_tokens[0][:1] == candidate_tokens[0][:1]:
            return max(0.94, ratio)
        return max(0.78, ratio)
    return ratio


def odds_team_key(value: object) -> str:
    text = normalize_name(value)
    aliases = {
        "oaklandathletics": "athletics",
        "sacramentoathletics": "athletics",
        "athletics": "athletics",
    }
    return aliases.get(text, text)


def _odds_api_error(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return f"HTTP {exc.code}: {(body[:500].strip() or exc.reason)}"
    return f"{type(exc).__name__}: {exc}"


def _odds_quota(response) -> dict:
    headers = getattr(response, "headers", {})
    return {
        "requests_remaining": headers.get("x-requests-remaining"),
        "requests_used": headers.get("x-requests-used"),
        "requests_last": headers.get("x-requests-last"),
    }


@st.cache_data(ttl=600, show_spinner=False, max_entries=8)
def fetch_odds_api_events_batter(api_key: str, commence_from: str, commence_to: str) -> tuple[list[dict], dict, str | None]:
    params = urlencode({
        "apiKey": api_key,
        "dateFormat": "iso",
        "commenceTimeFrom": commence_from,
        "commenceTimeTo": commence_to,
    })
    url = f"https://api.the-odds-api.com/v4/sports/{ODDS_API_SPORT}/events?{params}"
    request = Request(url, headers={"User-Agent": "MLB-Hitter-Dashboard/3.0", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
            quota = _odds_quota(response)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return [], {}, _odds_api_error(exc)
    if not isinstance(payload, list):
        return [], quota, "The events endpoint returned an unexpected response."
    return payload, quota, None


@st.cache_data(ttl=300, show_spinner=False, max_entries=64)
def fetch_odds_api_event_batter_props(api_key: str, event_id: str, bookmaker_keys: str, market_key: str) -> tuple[dict, dict, str | None]:
    params = urlencode({
        "apiKey": api_key,
        "bookmakers": bookmaker_keys,
        "markets": market_key,
        "oddsFormat": "american",
        "dateFormat": "iso",
        "includeMultipliers": "true",
    })
    url = f"https://api.the-odds-api.com/v4/sports/{ODDS_API_SPORT}/events/{event_id}/odds?{params}"
    request = Request(url, headers={"User-Agent": "MLB-Hitter-Dashboard/3.0", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=25) as response:
            payload = json.load(response)
            quota = _odds_quota(response)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return {}, {}, _odds_api_error(exc)
    if not isinstance(payload, dict):
        return {}, quota, "The event odds endpoint returned an unexpected response."
    return payload, quota, None


def slate_utc_bounds_for_odds(slate_date: object) -> tuple[str, str]:
    day = pd.Timestamp(slate_date)
    try:
        start = day.tz_localize("America/New_York").tz_convert("UTC")
    except Exception:
        start = day.tz_localize("UTC")
    return (
        (start - pd.Timedelta(hours=4)).isoformat().replace("+00:00", "Z"),
        (start + pd.Timedelta(days=1, hours=8)).isoformat().replace("+00:00", "Z"),
    )


def match_odds_event_for_game(game: dict, events: list[dict]) -> dict | None:
    away_keys = {odds_team_key(game.get("away_name")), odds_team_key(game.get("away_abbr"))}
    home_keys = {odds_team_key(game.get("home_name")), odds_team_key(game.get("home_abbr"))}
    candidates = []
    for event in events:
        event_away = odds_team_key(event.get("away_team"))
        event_home = odds_team_key(event.get("home_team"))
        if event_away in away_keys and event_home in home_keys:
            candidates.append(event)
    if not candidates:
        return None
    game_time = pd.to_datetime(game.get("game_datetime_utc"), utc=True, errors="coerce")
    if pd.isna(game_time) or len(candidates) == 1:
        return candidates[0]
    return min(
        candidates,
        key=lambda event: abs((pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce") - game_time).total_seconds())
        if not pd.isna(pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")) else float("inf"),
    )


def parse_batter_prop_payload(payload: dict, game: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for bookmaker in payload.get("bookmakers", []) or []:
        bookmaker_key = str(bookmaker.get("key", ""))
        bookmaker_title = str(bookmaker.get("title") or ODDS_BOOKMAKERS.get(bookmaker_key, bookmaker_key))
        for market in bookmaker.get("markets", []) or []:
            if str(market.get("key", "")) != ODDS_API_MARKET_KEY:
                continue
            grouped: dict[tuple[str, float], dict] = {}
            for outcome in market.get("outcomes", []) or []:
                side = str(outcome.get("name", "")).strip().lower()
                if side not in {"over", "under"}:
                    continue
                player = str(outcome.get("description") or outcome.get("participant") or "").strip()
                point = pd.to_numeric(pd.Series([outcome.get("point")]), errors="coerce").iloc[0]
                if not player or pd.isna(point):
                    continue
                key = (normalize_person_name(player), float(point))
                record = grouped.setdefault(key, {
                    "SlateDate": str(game.get("slate_date", "")),
                    "GamePK": game.get("game_pk"),
                    "OddsEventID": payload.get("id"),
                    "BookmakerKey": bookmaker_key,
                    "Bookmaker": bookmaker_title,
                    "MarketKey": ODDS_API_MARKET_KEY,
                    "Market": ODDS_API_MARKET_LABEL,
                    "Player": player,
                    "PlayerNorm": normalize_person_name(player),
                    "Line": float(point),
                    "OverOdds": np.nan,
                    "UnderOdds": np.nan,
                    "OverMultiplier": np.nan,
                    "UnderMultiplier": np.nan,
                    "LastUpdate": market.get("last_update"),
                })
                price = pd.to_numeric(pd.Series([outcome.get("price")]), errors="coerce").iloc[0]
                multiplier = outcome.get("multiplier") if outcome.get("multiplier") is not None else outcome.get("multipliers")
                multiplier = pd.to_numeric(pd.Series([multiplier]), errors="coerce").iloc[0]
                if side == "over":
                    record["OverOdds"] = float(price) if not pd.isna(price) else np.nan
                    record["OverMultiplier"] = float(multiplier) if not pd.isna(multiplier) else np.nan
                else:
                    record["UnderOdds"] = float(price) if not pd.isna(price) else np.nan
                    record["UnderMultiplier"] = float(multiplier) if not pd.isna(multiplier) else np.nan
            rows.extend(grouped.values())
    return pd.DataFrame(rows)


def fetch_batter_prop_quotes(api_key: str, games: list[dict], game_pks: list[int]) -> tuple[pd.DataFrame, dict, list[str]]:
    selected = {int(value) for value in game_pks if pd.notna(value)}
    if not api_key:
        return pd.DataFrame(), {}, ["The Odds API key is missing."]
    if not games or not selected:
        return pd.DataFrame(), {}, ["No MLB games were selected."]
    slate_date_value = next((game.get("slate_date") for game in games if game.get("game_pk") in selected), date.today())
    commence_from, commence_to = slate_utc_bounds_for_odds(slate_date_value)
    events, quota, event_error = fetch_odds_api_events_batter(api_key, commence_from, commence_to)
    if event_error:
        return pd.DataFrame(), quota, [f"Events: {event_error}"]
    bookmaker_keys = ",".join(ODDS_BOOKMAKERS.keys())
    frames, errors = [], []
    total_cost = 0.0
    matched = queried = 0
    for game in games:
        game_pk = pd.to_numeric(pd.Series([game.get("game_pk")]), errors="coerce").iloc[0]
        if pd.isna(game_pk) or int(game_pk) not in selected:
            continue
        event = match_odds_event_for_game(game, events)
        if not event:
            errors.append(f"{game.get('away_abbr')} @ {game.get('home_abbr')}: no matching odds event.")
            continue
        matched += 1
        payload, call_quota, call_error = fetch_odds_api_event_batter_props(
            api_key, str(event.get("id")), bookmaker_keys, ODDS_API_MARKET_KEY
        )
        queried += 1
        if call_quota:
            quota.update({key: value for key, value in call_quota.items() if value is not None})
            try:
                total_cost += float(call_quota.get("requests_last") or 0)
            except (TypeError, ValueError):
                pass
        if call_error:
            errors.append(f"{game.get('away_abbr')} @ {game.get('home_abbr')}: {call_error}")
            continue
        frame = parse_batter_prop_payload(payload, game)
        if frame.empty:
            errors.append(f"{game.get('away_abbr')} @ {game.get('home_abbr')}: no {ODDS_API_MARKET_LABEL.lower()} market returned.")
        else:
            frames.append(frame)
    quota["estimated_credits_used_this_fetch"] = total_cost
    quota["events_matched"] = matched
    quota["events_queried"] = queried
    if not frames:
        return pd.DataFrame(), quota, errors
    quotes = pd.concat(frames, ignore_index=True, sort=False)
    quotes["_Updated"] = pd.to_datetime(quotes.get("LastUpdate"), utc=True, errors="coerce")
    quotes = quotes.sort_values("_Updated", na_position="first").drop_duplicates(
        ["GamePK", "BookmakerKey", "MarketKey", "PlayerNorm", "Line"], keep="last"
    )
    return quotes.drop(columns="_Updated").reset_index(drop=True), quota, errors


def combine_batter_quote_frames(existing: pd.DataFrame, new_quotes: pd.DataFrame, slate_date: object) -> pd.DataFrame:
    frames = []
    slate_text = str(pd.Timestamp(slate_date).date())
    if existing is not None and not existing.empty:
        old = existing.copy()
        if "SlateDate" in old.columns:
            old = old[old["SlateDate"].astype(str).eq(slate_text)]
        if not old.empty:
            frames.append(old)
    if new_quotes is not None and not new_quotes.empty:
        frames.append(new_quotes.copy())
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["_Updated"] = pd.to_datetime(combined.get("LastUpdate"), utc=True, errors="coerce")
    combined = combined.sort_values("_Updated", na_position="first")
    combined = combined.drop_duplicates(["GamePK", "BookmakerKey", "MarketKey", "PlayerNorm", "Line"], keep="last")
    return combined.drop(columns="_Updated").reset_index(drop=True)


def _matched_batter_quotes(quotes: pd.DataFrame, game_pk: object, player_name: str) -> pd.DataFrame:
    if quotes is None or quotes.empty:
        return pd.DataFrame()
    numeric_game = pd.to_numeric(pd.Series([game_pk]), errors="coerce").iloc[0]
    candidates = quotes[pd.to_numeric(quotes.get("GamePK"), errors="coerce").eq(numeric_game)].copy()
    if candidates.empty:
        return candidates
    target = normalize_person_name(player_name)
    exact = candidates[candidates.get("PlayerNorm", pd.Series(index=candidates.index, dtype=object)).eq(target)]
    if not exact.empty:
        return exact
    candidates["_NameScore"] = candidates.get("Player", "").map(lambda value: player_name_match_score(player_name, value))
    best = pd.to_numeric(candidates["_NameScore"], errors="coerce").max()
    if pd.isna(best) or best < 0.84:
        return pd.DataFrame(columns=candidates.columns)
    return candidates[candidates["_NameScore"].ge(best - 0.015)].drop(columns="_NameScore", errors="ignore")


def _latest_batter_quote_per_book(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    work = candidates.copy()
    work["_Updated"] = pd.to_datetime(work.get("LastUpdate"), utc=True, errors="coerce")
    work["_Paired"] = work.get("OverOdds").notna().astype(int) + work.get("UnderOdds").notna().astype(int)
    work = work.sort_values(["BookmakerKey", "_Paired", "_Updated"], na_position="first")
    return work.drop_duplicates("BookmakerKey", keep="last").drop(columns=["_Updated", "_Paired"])


def select_batter_quote(candidates: pd.DataFrame, source_mode: str) -> dict | None:
    if candidates is None or candidates.empty:
        return None
    candidates = _latest_batter_quote_per_book(candidates)
    direct_key = ODDS_SOURCE_TO_KEY.get(source_mode)
    if direct_key:
        direct = candidates[candidates["BookmakerKey"].eq(direct_key)]
        if direct.empty:
            return None
        row = direct.sort_values("LastUpdate", na_position="first").iloc[-1].to_dict()
        row["SelectedSource"] = row.get("Bookmaker") or ODDS_BOOKMAKERS.get(direct_key, direct_key)
        row["BookCount"] = 1
        return row
    books = candidates[candidates["BookmakerKey"].isin(SPORTSBOOK_KEYS)].copy()
    if books.empty:
        return None
    if source_mode == "Consensus sportsbooks":
        return {
            "Line": float(pd.to_numeric(books["Line"], errors="coerce").median()),
            "OverOdds": pd.to_numeric(books["OverOdds"], errors="coerce").median(),
            "UnderOdds": pd.to_numeric(books["UnderOdds"], errors="coerce").median(),
            "OverMultiplier": np.nan,
            "UnderMultiplier": np.nan,
            "SelectedSource": f"Consensus ({books['BookmakerKey'].nunique()} books)",
            "BookmakerKey": "consensus",
            "LastUpdate": pd.to_datetime(books["LastUpdate"], utc=True, errors="coerce").max(),
            "BookCount": int(books["BookmakerKey"].nunique()),
        }
    if source_mode == "Best sportsbook over line":
        best_line = pd.to_numeric(books["Line"], errors="coerce").min()
        subset = books[pd.to_numeric(books["Line"], errors="coerce").eq(best_line)].copy()
        subset["_Price"] = pd.to_numeric(subset["OverOdds"], errors="coerce").fillna(-100000)
        row = subset.sort_values("_Price").iloc[-1].drop(labels="_Price").to_dict()
        row["SelectedSource"] = f"Best over · {row.get('Bookmaker')}"
        return row
    if source_mode == "Best sportsbook under line":
        best_line = pd.to_numeric(books["Line"], errors="coerce").max()
        subset = books[pd.to_numeric(books["Line"], errors="coerce").eq(best_line)].copy()
        subset["_Price"] = pd.to_numeric(subset["UnderOdds"], errors="coerce").fillna(-100000)
        row = subset.sort_values("_Price").iloc[-1].drop(labels="_Price").to_dict()
        row["SelectedSource"] = f"Best under · {row.get('Bookmaker')}"
        return row
    return None


def no_vig_over_probability_batter(over_odds: object, under_odds: object) -> float:
    over = american_odds_to_probability(over_odds)
    under = american_odds_to_probability(under_odds)
    if not np.isfinite(over) or not np.isfinite(under) or over + under <= 0:
        return np.nan
    return float(over / (over + under))


def poisson_over_probability_batter(mean: float, line: float) -> float:
    mean = max(float(mean), 0.0001)
    threshold = max(int(math.floor(float(line))), 0)
    cdf = sum(math.exp(-mean) * mean ** k / math.factorial(k) for k in range(threshold + 1))
    return float(np.clip(1.0 - cdf, 0.0, 1.0))


def apply_batter_odds_to_board(board: pd.DataFrame, quotes: pd.DataFrame, source_mode: str, default_game_pk: object = None) -> pd.DataFrame:
    result = board.copy()
    if "GamePK" not in result.columns:
        result["GamePK"] = default_game_pk
    defaults = {
        "Market_Line": np.nan,
        "Over_Odds": np.nan,
        "Under_Odds": np.nan,
        "Over_Multiplier": np.nan,
        "Under_Multiplier": np.nan,
        "Line_Source": "",
        "Line_Updated": "",
        "Market_Over_Prob": np.nan,
        "Model_Over_Prob": np.nan,
        "Model_Market_Edge": np.nan,
        "Projection_Edge": np.nan,
    }
    for column, default in defaults.items():
        result[column] = default
    if quotes is None or quotes.empty:
        return result
    for index, row in result.iterrows():
        candidates = _matched_batter_quotes(quotes, row.get("GamePK"), str(row.get("Player", "")))
        selected = select_batter_quote(candidates, source_mode)
        if not selected:
            continue
        line = pd.to_numeric(pd.Series([selected.get("Line")]), errors="coerce").iloc[0]
        if pd.isna(line):
            continue
        over_odds = pd.to_numeric(pd.Series([selected.get("OverOdds")]), errors="coerce").iloc[0]
        under_odds = pd.to_numeric(pd.Series([selected.get("UnderOdds")]), errors="coerce").iloc[0]
        displayed_probability = pd.to_numeric(pd.Series([row.get(BINARY_PROBABILITY_COLUMN)]), errors="coerce").iloc[0]
        per_pa = pd.to_numeric(pd.Series([row.get("Model_Hit_Per_PA")]), errors="coerce").iloc[0]
        projected_pa = pd.to_numeric(pd.Series([row.get("Projected_PA")]), errors="coerce").iloc[0]
        expected_count = float(per_pa * projected_pa) if pd.notna(per_pa) and pd.notna(projected_pa) else np.nan
        if float(line) <= 0.5 and pd.notna(displayed_probability):
            model_over = float(displayed_probability)
        elif np.isfinite(expected_count):
            model_over = poisson_over_probability_batter(expected_count, float(line))
        else:
            model_over = np.nan
        market_over = no_vig_over_probability_batter(over_odds, under_odds)
        result.at[index, "Market_Line"] = float(line)
        result.at[index, "Over_Odds"] = float(over_odds) if not pd.isna(over_odds) else np.nan
        result.at[index, "Under_Odds"] = float(under_odds) if not pd.isna(under_odds) else np.nan
        result.at[index, "Over_Multiplier"] = selected.get("OverMultiplier", np.nan)
        result.at[index, "Under_Multiplier"] = selected.get("UnderMultiplier", np.nan)
        result.at[index, "Line_Source"] = str(selected.get("SelectedSource") or selected.get("Bookmaker") or "")
        result.at[index, "Line_Updated"] = str(selected.get("LastUpdate") or "")
        result.at[index, "Market_Over_Prob"] = market_over
        result.at[index, "Model_Over_Prob"] = model_over
        result.at[index, "Model_Market_Edge"] = float(model_over - market_over) if np.isfinite(model_over) and np.isfinite(market_over) else np.nan
        result.at[index, "Projection_Edge"] = float(expected_count - float(line)) if np.isfinite(expected_count) else np.nan
    result["OddsSourceMode"] = source_mode
    return result


def render_batter_odds_section(board: pd.DataFrame, matchup: dict, api_key: str, widget_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    slate_date = str(pd.Timestamp(matchup.get("slate_date") or date.today()).date())
    quotes_key = f"{widget_prefix}_odds_quotes_{slate_date}"
    meta_key = f"{widget_prefix}_odds_meta_{slate_date}"
    errors_key = f"{widget_prefix}_odds_errors_{slate_date}"
    with st.expander(f"Automatic sportsbook / PrizePicks {ODDS_API_MARKET_LABEL.lower()} lines", expanded=False):
        source_mode = st.selectbox("Line source", ODDS_SOURCE_OPTIONS, key=f"{widget_prefix}_odds_source")
        scope = st.radio("Fetch scope", ["Current game", "Entire slate"], horizontal=True, key=f"{widget_prefix}_odds_scope")
        games, schedule_error = fetch_mlb_schedule(slate_date)
        if schedule_error:
            st.warning(f"MLB schedule unavailable: {schedule_error}")
        fetch_clicked = st.button("Fetch / refresh lines", type="primary", width="stretch", key=f"{widget_prefix}_fetch_odds")
        if fetch_clicked:
            if not api_key:
                st.error("Add THE_ODDS_API_KEY in this Streamlit app's Secrets or use the temporary sidebar field.")
            elif not games:
                st.error("No MLB games were available for the selected slate date.")
            else:
                game_pks = [int(game["game_pk"]) for game in games if game.get("game_pk") is not None]
                if scope == "Current game":
                    current_pk = pd.to_numeric(pd.Series([matchup.get("game_pk")]), errors="coerce").iloc[0]
                    game_pks = [int(current_pk)] if pd.notna(current_pk) else []
                with st.spinner(f"Fetching {ODDS_API_MARKET_LABEL.lower()} lines..."):
                    new_quotes, quota, errors = fetch_batter_prop_quotes(api_key, games, game_pks)
                existing = st.session_state.get(quotes_key, pd.DataFrame())
                st.session_state[quotes_key] = combine_batter_quote_frames(existing, new_quotes, slate_date)
                st.session_state[meta_key] = quota
                st.session_state[errors_key] = errors
        quotes = st.session_state.get(quotes_key, pd.DataFrame())
        quota = st.session_state.get(meta_key, {})
        errors = st.session_state.get(errors_key, [])
        metrics = st.columns(4)
        metrics[0].metric("Quotes", f"{len(quotes):,}" if isinstance(quotes, pd.DataFrame) else "0")
        metrics[1].metric("Events queried", quota.get("events_queried", "—"))
        metrics[2].metric("Credits this fetch", quota.get("estimated_credits_used_this_fetch", "—"))
        metrics[3].metric("Credits remaining", quota.get("requests_remaining", "—"))
        if errors:
            with st.expander("Odds-fetch details"):
                for error in errors[:50]:
                    st.code(error)
        if isinstance(quotes, pd.DataFrame) and not quotes.empty:
            st.caption("The selected source is matched by game and player. Manual backtest editing remains available when a market is missing.")
    source_mode = st.session_state.get(f"{widget_prefix}_odds_source", ODDS_SOURCE_OPTIONS[0])
    quotes = st.session_state.get(quotes_key, pd.DataFrame())
    return apply_batter_odds_to_board(board, quotes, source_mode, matchup.get("game_pk")), quotes, source_mode


def lineup_seed_as_roster(lineup_seed: pd.DataFrame) -> pd.DataFrame:
    if lineup_seed is None or lineup_seed.empty:
        return pd.DataFrame()
    roster = lineup_seed[["player_id"]].drop_duplicates().copy()
    roster["Player"] = np.nan
    roster["Bats"] = np.nan
    roster["Position"] = ""
    roster["RosterStatus"] = "Slate lineup"
    return roster


def add_slate_grade(board: pd.DataFrame, probability_weight: float, score_weight: float, confidence_weight: float) -> pd.DataFrame:
    result = board.copy()
    probability = pd.to_numeric(result.get(BINARY_PROBABILITY_COLUMN), errors="coerce")
    score = pd.to_numeric(result.get(BINARY_SCORE_COLUMN), errors="coerce").fillna(50.0).clip(0, 100)
    confidence = pd.to_numeric(result.get("Confidence"), errors="coerce").fillna(50.0).clip(0, 100)
    result["ProbabilityGrade"] = percentile(probability).clip(0, 100)
    total = max(float(probability_weight + score_weight + confidence_weight), 1.0)
    result["SlateGrade"] = (
        result["ProbabilityGrade"] * float(probability_weight)
        + score * float(score_weight)
        + confidence * float(confidence_weight)
    ) / total
    result = result.sort_values(["SlateGrade", BINARY_PROBABILITY_COLUMN, BINARY_SCORE_COLUMN], ascending=False).reset_index(drop=True)
    result = result.drop(columns=["SlateRank"], errors="ignore")
    result.insert(0, "SlateRank", np.arange(1, len(result) + 1))
    return result


def generate_three_man_pairings(
    board: pd.DataFrame,
    number_of_pairings: int,
    candidate_pool: int,
    minimum_confidence: float,
    max_same_team: int,
    require_different_games: bool,
    style: str,
    diversity_strength: float,
) -> pd.DataFrame:
    if board is None or board.empty:
        return pd.DataFrame()
    candidates = board.copy()
    candidates["Confidence"] = pd.to_numeric(candidates.get("Confidence"), errors="coerce").fillna(0.0)
    candidates = candidates[candidates["Confidence"].ge(float(minimum_confidence))]
    candidates = candidates.drop_duplicates(["Player", "GamePK"], keep="first")
    candidates = candidates.sort_values(["SlateGrade", BINARY_PROBABILITY_COLUMN], ascending=False).head(int(candidate_pool)).reset_index(drop=True)
    if len(candidates) < 3:
        return pd.DataFrame()
    rows = []
    for indices in combinations(range(len(candidates)), 3):
        trio = candidates.iloc[list(indices)]
        team_counts = trio.get("Team", pd.Series(["", "", ""])).astype(str).value_counts()
        if not team_counts.empty and int(team_counts.max()) > int(max_same_team):
            continue
        game_values = trio.get("GamePK", pd.Series([np.nan, np.nan, np.nan])).astype(str)
        unique_games = game_values.nunique(dropna=True)
        if require_different_games and unique_games < 3:
            continue
        probabilities = pd.to_numeric(trio[BINARY_PROBABILITY_COLUMN], errors="coerce").clip(0.001, 0.999)
        if probabilities.isna().any():
            continue
        grades = pd.to_numeric(trio["SlateGrade"], errors="coerce").fillna(50.0)
        confidence = pd.to_numeric(trio["Confidence"], errors="coerce").fillna(50.0)
        market_edges = pd.to_numeric(trio.get("Model_Market_Edge"), errors="coerce") if "Model_Market_Edge" in trio.columns else pd.Series(np.nan, index=trio.index)
        mean_edge = float(market_edges.mean()) if market_edges.notna().any() else np.nan
        edge_grade = float(np.clip(50.0 + (mean_edge * 250.0 if np.isfinite(mean_edge) else 0.0), 0, 100))
        geometric_probability = float(np.prod(probabilities.to_numpy()) ** (1.0 / 3.0))
        joint_probability = float(np.prod(probabilities.to_numpy()))
        avg_grade = float(grades.mean())
        min_conf = float(confidence.min())
        if style == "Highest probability":
            raw_score = .35 * avg_grade + .45 * geometric_probability * 100 + .15 * min_conf + .05 * edge_grade
        elif style == "Model strength":
            raw_score = .65 * avg_grade + .20 * geometric_probability * 100 + .10 * min_conf + .05 * edge_grade
        elif style == "Market edge":
            raw_score = .40 * avg_grade + .20 * geometric_probability * 100 + .10 * min_conf + .30 * edge_grade
        else:
            raw_score = .55 * avg_grade + .25 * geometric_probability * 100 + .15 * min_conf + .05 * edge_grade
        if unique_games < 3:
            raw_score -= (3 - unique_games) * 2.5
        if not team_counts.empty and int(team_counts.max()) > 1:
            raw_score -= (int(team_counts.max()) - 1) * 3.0
        player_keys = tuple(f"{row.Player}|{row.GamePK}" for row in trio.itertuples())
        expected_counts = pd.to_numeric(trio.get(BINARY_EXPECTED_COUNT_COLUMN), errors="coerce")
        def leg_text(position: int) -> str:
            count_value = expected_counts.iloc[position] if position < len(expected_counts) else np.nan
            count_part = f" · {count_value:.2f} exp" if pd.notna(count_value) else ""
            return f"{float(probabilities.iloc[position]):.1%}{count_part} · {trio.iloc[position].get('Team', '')}"
        rows.append({
            "PlayerKeys": player_keys,
            "PairingScore": raw_score,
            "EstimatedAll3Probability": joint_probability,
            "AverageLegProbability": float(probabilities.mean()),
            "GeometricLegProbability": geometric_probability,
            "AverageSlateGrade": avg_grade,
            "MinimumConfidence": min_conf,
            "AverageMarketEdge": mean_edge,
            "Player 1": str(trio.iloc[0].get("Player", "")),
            "Leg 1": leg_text(0),
            "Player 2": str(trio.iloc[1].get("Player", "")),
            "Leg 2": leg_text(1),
            "Player 3": str(trio.iloc[2].get("Player", "")),
            "Leg 3": leg_text(2),
            "Games": " | ".join(trio.get("Game", pd.Series(["", "", ""])).astype(str).tolist()),
        })
    if not rows:
        return pd.DataFrame()
    combo_table = pd.DataFrame(rows).sort_values("PairingScore", ascending=False).reset_index(drop=True)
    selected_rows = []
    exposure: dict[str, int] = {}
    remaining = combo_table.copy()
    for _ in range(min(int(number_of_pairings), len(remaining))):
        if remaining.empty:
            break
        remaining["_Adjusted"] = remaining.apply(
            lambda row: float(row["PairingScore"]) - float(diversity_strength) * sum(exposure.get(key, 0) for key in row["PlayerKeys"]),
            axis=1,
        )
        best_index = remaining["_Adjusted"].idxmax()
        chosen = remaining.loc[best_index].copy()
        chosen["SelectionScore"] = chosen["_Adjusted"]
        selected_rows.append(chosen)
        for key in chosen["PlayerKeys"]:
            exposure[key] = exposure.get(key, 0) + 1
        remaining = remaining.drop(index=best_index)
    result = pd.DataFrame(selected_rows).reset_index(drop=True)
    if result.empty:
        return result
    result.insert(0, "Pairing", np.arange(1, len(result) + 1))
    return result.drop(columns=["PlayerKeys", "_Adjusted"], errors="ignore")



def build_whole_slate_model_board(
    df: pd.DataFrame,
    games: list[dict],
    available_teams: list[str],
    min_pa: int,
    loaded_end: object,
    include_low_sample: bool,
    use_confirmed_lineups: bool,
    team_runs: float,
    starter_innings: float,
    bullpen_multiplier: float,
) -> tuple[pd.DataFrame, list[str]]:
    boards, errors = [], []
    total = max(len(games) * 2, 1)
    progress = st.progress(0.0, text="Preparing whole-slate hit board...")
    completed = 0
    for game in games:
        for side in ["away", "home"]:
            completed += 1
            batting_code = game.get(f"{side}_abbr")
            team = match_statcast_team(batting_code, available_teams)
            opponent_side = "home" if side == "away" else "away"
            pitcher_id = game.get(f"{opponent_side}_pitcher_id")
            pitcher_name = game.get(f"{opponent_side}_pitcher_name")
            if team is None or pitcher_id is None:
                errors.append(f"{game.get('away_abbr')} @ {game.get('home_abbr')} · {batting_code}: team or probable starter unavailable.")
                progress.progress(completed / total)
                continue
            lineup_seed = infer_recent_lineup(df, team)
            lineup_status = "Recent lineup fallback"
            if use_confirmed_lineups:
                lineup_result = fetch_mlb_confirmed_lineup(game.get("game_pk"), side)
                posted = _clean_lineup_seed(lineup_result.get("lineup"))
                if lineup_result.get("ok") and len(posted) >= 9:
                    lineup_seed = posted
                    lineup_status = "Confirmed"
                elif lineup_result.get("status") == "Partial":
                    lineup_status = "Partial / recent fallback"
            lineup_seed = _clean_lineup_seed(lineup_seed)
            if lineup_seed.empty:
                errors.append(f"{batting_code} vs {pitcher_name}: no usable lineup seed.")
                progress.progress(completed / total)
                continue
            roster = lineup_seed_as_roster(lineup_seed)
            park = fetch_savant_park_factors(game.get("venue", ""), pd.Timestamp(loaded_end).year, "Hits")
            try:
                board, _ = build_hit_board(
                    df=df, pitcher_id=int(pitcher_id), team=team, min_pa=min_pa,
                    end_date=pd.Timestamp(loaded_end), park_hit_factor_lhb=float(park["L"]),
                    park_hit_factor_rhb=float(park["R"]), team_runs=float(team_runs),
                    is_away=side == "away", starter_innings=float(starter_innings),
                    bullpen_multiplier=float(bullpen_multiplier), lineup_override=lineup_seed,
                    lineup_edits=None, sprint_upload=None, active_roster=roster,
                    include_low_sample=include_low_sample,
                )
            except Exception as exc:
                errors.append(f"{batting_code} vs {pitcher_name}: {type(exc).__name__}: {exc}")
                progress.progress(completed / total)
                continue
            if board.empty:
                errors.append(f"{batting_code} vs {pitcher_name}: model returned no hitters.")
                progress.progress(completed / total)
                continue
            board = board[pd.to_numeric(board.get("LineupSpot"), errors="coerce").between(1, 9)].copy()
            board["GamePK"] = game.get("game_pk")
            board["SlateDate"] = game.get("slate_date")
            board["Team"] = batting_code
            board["Opponent"] = game.get(f"{opponent_side}_abbr")
            board["Game"] = f"{game.get('away_abbr')} @ {game.get('home_abbr')}"
            board["StartingPitcher"] = pitcher_name
            board["HomeAway"] = "Away" if side == "away" else "Home"
            board["LineupStatus"] = lineup_status
            board["Venue"] = game.get("venue")
            board["GameDateTimeUTC"] = game.get("game_datetime_utc")
            boards.append(board)
            progress.progress(completed / total, text=f"Built {completed} of {total} offense matchups")
    progress.empty()
    if not boards:
        return pd.DataFrame(), errors
    combined = pd.concat(boards, ignore_index=True, sort=False)
    combined = combined.drop(columns=["Rank"], errors="ignore")
    return combined, errors



def render_slate_tools(
    df: pd.DataFrame,
    matchup: dict,
    available_teams: list[str],
    min_pa: int,
    loaded_end: object,
    include_low_sample: bool,
    calibration: dict | None,
    odds_quotes: pd.DataFrame,
    odds_source_mode: str,
    widget_prefix: str,
    
) -> None:
    st.markdown("### Whole-slate Top 10 and three-person pairing suggestions")
    st.caption(
        "The Slate Grade combines probability rank, the dashboard's 0–100 model score, and confidence. "
        "Pairings are suggestions—not guarantees—and the displayed all-three probability is an independence approximation."
    )
    slate_date = str(pd.Timestamp(matchup.get("slate_date") or date.today()).date())
    games, schedule_error = fetch_mlb_schedule(slate_date)
    if schedule_error:
        st.warning(f"MLB slate unavailable: {schedule_error}")
    if not games:
        st.info("No games were available for this slate date.")
        return
    controls = st.columns(4)
    with controls[0]:
        slate_team_runs = st.number_input("Default team implied runs", 2.0, 8.0, 4.5, 0.1, key=f"{widget_prefix}_slate_runs")
    with controls[1]:
        slate_starter_innings = st.number_input("Default starter innings", 3.0, 7.5, 5.5, 0.5, key=f"{widget_prefix}_slate_ip")
    with controls[2]:
        slate_bullpen = st.number_input("Default bullpen multiplier", 0.75, 1.30, 1.00, 0.01, key=f"{widget_prefix}_slate_bullpen")
    with controls[3]:
        use_confirmed = st.checkbox("Use confirmed MLB lineups when posted", value=True, key=f"{widget_prefix}_slate_confirmed")
    
    board_key = f"{widget_prefix}_whole_slate_board_{slate_date}"
    error_key = f"{widget_prefix}_whole_slate_errors_{slate_date}"
    if st.button("Build / refresh whole-slate board", type="primary", width="stretch", key=f"{widget_prefix}_build_slate"):
        with st.spinner("Building both offenses for every available game..."):
            built, errors = build_whole_slate_model_board(
                df, games, available_teams, min_pa, loaded_end, include_low_sample,
                use_confirmed, slate_team_runs, slate_starter_innings, slate_bullpen,
            )
        st.session_state[board_key] = built
        st.session_state[error_key] = errors
    slate_board = st.session_state.get(board_key, pd.DataFrame())
    build_errors = st.session_state.get(error_key, [])
    if build_errors:
        with st.expander("Slate-build details"):
            for error in build_errors[:80]:
                st.code(error)
    if slate_board is None or slate_board.empty:
        st.info("Press Build / refresh whole-slate board to create the Top 10 and pairing pool.")
        return
    slate_board = apply_binary_probability_calibration(slate_board, calibration)
    slate_board = apply_batter_odds_to_board(slate_board, odds_quotes, odds_source_mode)
    st.markdown("#### Top 10 settings")
    weight_columns = st.columns(3)
    probability_weight = weight_columns[0].slider("Probability weight", 0, 100, 45, 5, key=f"{widget_prefix}_prob_weight")
    score_weight = weight_columns[1].slider("Model score weight", 0, 100, 35, 5, key=f"{widget_prefix}_score_weight")
    confidence_weight = weight_columns[2].slider("Confidence weight", 0, 100, 20, 5, key=f"{widget_prefix}_confidence_weight")
    ranked_slate = add_slate_grade(slate_board, probability_weight, score_weight, confidence_weight)
    top_columns = [
        "SlateRank", "Player", "Team", "Opponent", "Game", "LineupSpot",
        BINARY_EXPECTED_COUNT_COLUMN, BINARY_PROBABILITY_COLUMN, BINARY_SCORE_COLUMN, "Confidence", "Confidence_Level",
        "SlateGrade", "LineupStatus", "Market_Line", "Market_Over_Prob", "Model_Market_Edge", "Line_Source",
    ]
    top10 = ranked_slate[[column for column in top_columns if column in ranked_slate.columns]].head(10).copy()
    top10 = top10.rename(columns={
        BINARY_EXPECTED_COUNT_COLUMN: BINARY_EXPECTED_COUNT_LABEL,
        BINARY_PROBABILITY_COLUMN: BINARY_TARGET_LABEL,
        BINARY_SCORE_COLUMN: "Model Score",
        "LineupSpot": "Order",
        "Market_Over_Prob": "No-vig Market",
        "Model_Market_Edge": "Model Edge",
        "Market_Line": "Line",
    })
    format_map = {
        BINARY_EXPECTED_COUNT_LABEL: "{:.2f}", BINARY_TARGET_LABEL: "{:.1%}", "Model Score": "{:.1f}", "Confidence": "{:.1f}",
        "SlateGrade": "{:.1f}", "No-vig Market": "{:.1%}", "Model Edge": "{:+.1%}", "Line": "{:.1f}", "Order": "{:.0f}",
    }
    score_subsets = [column for column in ["Model Score", "Confidence", "SlateGrade"] if column in top10.columns]
    top_style = top10.style
    if score_subsets:
        top_style = top_style.background_gradient(cmap="RdYlGn", subset=score_subsets, vmin=0, vmax=100)
    st.dataframe(top_style.format({key: value for key, value in format_map.items() if key in top10.columns}, na_rep="—"), width="stretch", hide_index=True)
    st.download_button(
        "Download whole-slate board",
        ranked_slate.to_csv(index=False).encode("utf-8"),
        file_name=f"{BINARY_TARGET_SLUG}_whole_slate_{slate_date}.csv",
        mime="text/csv",
        width="stretch",
    )
    st.markdown("#### Generate multiple three-person pairings")
    pairing_controls_1 = st.columns(4)
    pair_count = pairing_controls_1[0].slider("Number of pairings", 1, 20, 10, 1, key=f"{widget_prefix}_pair_count")
    pool_size = pairing_controls_1[1].slider("Candidate pool", 6, min(30, max(6, len(ranked_slate))), min(15, max(6, len(ranked_slate))), 1, key=f"{widget_prefix}_pool_size")
    minimum_confidence = pairing_controls_1[2].slider("Minimum confidence", 0, 90, 45, 5, key=f"{widget_prefix}_min_conf")
    max_same_team = pairing_controls_1[3].selectbox("Max hitters from one team", [1, 2, 3], index=0, key=f"{widget_prefix}_max_team")
    pairing_controls_2 = st.columns(3)
    require_different_games = pairing_controls_2[0].checkbox("Require three different games", value=True, key=f"{widget_prefix}_different_games")
    style = pairing_controls_2[1].selectbox("Pairing style", ["Balanced", "Highest probability", "Model strength", "Market edge"], key=f"{widget_prefix}_pair_style")
    diversity = pairing_controls_2[2].slider("Player-exposure diversity", 0.0, 10.0, 3.0, 0.5, key=f"{widget_prefix}_diversity")
    if style == "Market edge" and not pd.to_numeric(ranked_slate.get("Model_Market_Edge"), errors="coerce").notna().any():
        st.warning("Market-edge style needs fetched sportsbook odds. It will behave close to Balanced until lines are available.")
    pairings = generate_three_man_pairings(
        ranked_slate, pair_count, pool_size, minimum_confidence, max_same_team,
        require_different_games, style, diversity,
    )
    if pairings.empty:
        st.warning("No valid three-person combinations met the selected restrictions. Increase the pool or loosen team/game limits.")
        return
    pairing_view = pairings.copy()
    display_columns = [
        "Pairing", "Player 1", "Leg 1", "Player 2", "Leg 2", "Player 3", "Leg 3",
        "PairingScore", "EstimatedAll3Probability", "AverageLegProbability",
        "MinimumConfidence", "AverageMarketEdge", "Games",
    ]
    pairing_view = pairing_view[[column for column in display_columns if column in pairing_view.columns]]
    st.dataframe(
        pairing_view.style.background_gradient(cmap="RdYlGn", subset=[column for column in ["PairingScore", "MinimumConfidence"] if column in pairing_view.columns], axis=0).format({
            "PairingScore": "{:.1f}", "EstimatedAll3Probability": "{:.2%}",
            "AverageLegProbability": "{:.1%}", "MinimumConfidence": "{:.1f}",
            "AverageMarketEdge": "{:+.1%}",
        }, na_rep="—"),
        width="stretch", hide_index=True, height=600,
    )
    st.caption(
        "Estimated all-three probability multiplies the three model probabilities and assumes independence. "
        "Same-game and same-team outcomes can be correlated, so treat it as a comparison tool rather than an exact entry probability."
    )


inject_clean_css()
MODEL_KEY = "clean_hits"
odds_api_key = read_streamlit_secret("THE_ODDS_API_KEY")

st.markdown('<div class="app-kicker">⚾ Statcast matchup lab</div>', unsafe_allow_html=True)
st.title("Advanced MLB Hits Dashboard")
st.caption(
    "A dedicated one-hit model with expected batting average, contact skill, pitch-shape fit, "
    "zone fit, platoon splits, pitcher vulnerability, recent form and projected opportunities."
)

with st.sidebar:
    st.header("Statcast sample")
    yesterday = date.today() - timedelta(days=1)
    end_date_value = st.date_input(
        "Stats through",
        value=yesterday,
        max_value=yesterday,
        key="clean_hits_end_date",
    )
    lookback_days = st.slider(
        "Lookback days", 21, 120, 60, 7, key="clean_hits_lookback"
    )
    min_pa = st.slider(
        "Minimum hitter PA", 10, 100, 30, 5, key="clean_hits_min_pa"
    )
    include_low_sample = st.checkbox(
        "Include active-roster hitters below the PA threshold",
        value=True,
        key="clean_hits_include_low_sample",
        help="Roster players with little or no Statcast history are included with league-average priors and Low confidence.",
    )
    refresh = st.button(
        "Load / refresh Statcast", type="primary", key="clean_hits_refresh"
    )
    st.divider()
    st.subheader("Probability calibration")
    hits_calibration_upload = st.file_uploader(
        "Optional hits calibration JSON", type=["json"], key="clean_hits_calibration_upload"
    )
    if hits_calibration_upload is not None:
        calibration_signature = (
            getattr(hits_calibration_upload, "name", "calibration.json"),
            getattr(hits_calibration_upload, "size", None),
        )
        signature_key = "hits_calibration_upload_signature"
        if st.session_state.get(signature_key) != calibration_signature:
            uploaded_calibration, calibration_error = parse_binary_calibration_upload(hits_calibration_upload)
            if calibration_error:
                st.error(calibration_error)
            elif uploaded_calibration:
                st.session_state["hits_probability_calibration"] = uploaded_calibration
                st.session_state[signature_key] = calibration_signature
    if st.session_state.get("hits_probability_calibration"):
        loaded_calibration = st.session_state["hits_probability_calibration"]
        st.caption(
            f"Calibration active · n={int(loaded_calibration.get('n', 0)):,} · "
            f"slope {float(loaded_calibration.get('slope', 1.0)):.3f}"
        )
        if st.button("Clear hits calibration", key="clean_hits_clear_calibration"):
            st.session_state.pop("hits_probability_calibration", None)
            st.rerun()


    st.divider()
    st.subheader("Automatic odds")
    if odds_api_key:
        st.caption("THE_ODDS_API_KEY loaded from this app's Streamlit Secrets.")
    else:
        temporary_odds_key = st.text_input(
            "Temporary The Odds API key", type="password", key=f"{MODEL_KEY}_temporary_odds_key",
            help="Recommended: store THE_ODDS_API_KEY in this app's Streamlit Secrets.",
        )
        odds_api_key = str(temporary_odds_key or "").strip()
        st.caption("This temporary field resets when the app session ends.")

start_date_value = end_date_value - timedelta(days=lookback_days - 1)
if refresh or "clean_hit_statcast_data" not in st.session_state:
    with st.spinner(
        f"Loading Statcast from {start_date_value} through {end_date_value}..."
    ):
        raw = load_statcast(
            start_date_value.strftime("%Y-%m-%d"),
            end_date_value.strftime("%Y-%m-%d"),
        )
        st.session_state["clean_hit_statcast_data"] = prepare_data(raw)
        st.session_state["clean_hit_loaded_dates"] = (
            start_date_value,
            end_date_value,
        )

df = st.session_state.get("clean_hit_statcast_data", pd.DataFrame())
if df.empty:
    st.warning("No Statcast data was returned. Use completed dates and try again.")
    st.stop()

loaded_start, loaded_end = st.session_state["clean_hit_loaded_dates"]
st.caption(
    f"Using {len(df):,} pitches from {loaded_start} through {loaded_end}. "
    "The slate selector can use a later game date because model statistics stop at the date above."
)

pitcher_summary = (
    df.groupby("pitcher")
    .agg(
        Pitches=("pitcher", "size"),
        Player_Name=("player_name", "first"),
        Pitcher_Team=("pitcher_team", "last"),
        Hand=("p_throws", lambda x: most_common(x, "?")),
    )
    .reset_index()
)
pitcher_summary = pitcher_summary[pitcher_summary["Pitches"].ge(80)].copy()
pitcher_summary["pitcher"] = pd.to_numeric(
    pitcher_summary["pitcher"], errors="coerce"
).astype("Int64")
pitcher_summary = pitcher_summary.dropna(subset=["pitcher"])
pitcher_summary["Display"] = (
    pitcher_summary["Player_Name"].fillna("Unknown pitcher")
    + " — "
    + pitcher_summary["Pitcher_Team"].fillna("?")
    + " — "
    + pitcher_summary["Hand"].fillna("?")
    + " ("
    + pitcher_summary["Pitches"].astype(str)
    + " pitches)"
)
pitcher_summary = pitcher_summary.sort_values("Display")
if pitcher_summary.empty:
    st.warning("No pitcher met the sample threshold. Increase the lookback period.")
    st.stop()

available_teams = sorted(df["batter_team"].dropna().astype(str).unique().tolist())
matchup = matchup_selector(pitcher_summary, available_teams, key_prefix=MODEL_KEY)
selected_pitcher = int(matchup["pitcher_id"])
selected_display = str(matchup["pitcher_display"])
selected_team = str(matchup["batting_team"])
home_away = str(matchup["home_away"])

active_roster, roster_error = fetch_active_roster(
    matchup.get("batting_team_id"), matchup.get("slate_date")
)
if roster_error:
    st.caption(
        f"Active roster could not be fully loaded ({roster_error}). "
        "The app will fall back to hitters found in the Statcast sample."
    )

attack_profile = pitcher_attack_profile_hits(df, selected_pitcher)
render_pitcher_attack_panel(attack_profile, matchup["pitcher_name"], "hits")

park_year = pd.Timestamp(matchup.get("slate_date") or loaded_end).year
auto_park = fetch_savant_park_factors(matchup.get("venue", ""), park_year, "Hits")

with st.expander("Game context and model adjustments", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        team_runs = st.number_input(
            "Team implied runs", 1.0, 9.0, 4.5, 0.1, key="clean_hits_runs"
        )
        starter_innings = st.number_input(
            "Expected starter innings", 2.0, 8.0, 5.5, 0.5, key="clean_hits_ip"
        )
    with c2:
        manual_park_override = st.checkbox(
            "Manual park-factor override",
            value=False,
            key=f"clean_hits_manual_park_{matchup.get('game_pk')}_{normalize_name(matchup.get('venue'))}",
            help="Leave this off to use the automatic Baseball Savant 3-year factors.",
        )
        if manual_park_override:
            park_hit_factor_lhb = st.number_input(
                "Park hit factor — LHB", 70.0, 140.0, float(auto_park["L"]), 1.0,
                key=f"clean_hits_park_l_{matchup.get('game_pk')}_{normalize_name(matchup.get('venue'))}",
            )
            park_hit_factor_rhb = st.number_input(
                "Park hit factor — RHB", 70.0, 140.0, float(auto_park["R"]), 1.0,
                key=f"clean_hits_park_r_{matchup.get('game_pk')}_{normalize_name(matchup.get('venue'))}",
            )
            park_source = "Manual override"
        else:
            park_hit_factor_lhb = float(auto_park["L"])
            park_hit_factor_rhb = float(auto_park["R"])
            park_source = str(auto_park["source"])
            st.markdown(
                f"""
                <div class="park-grid">
                    <div class="park-chip"><div class="side">LHB HITS</div><div class="factor">{park_hit_factor_lhb:.0f}</div></div>
                    <div class="park-chip"><div class="side">RHB HITS</div><div class="factor">{park_hit_factor_rhb:.0f}</div></div>
                </div>
                <div class="park-source">{escape(park_source)}</div>
                """,
                unsafe_allow_html=True,
            )
            if not auto_park["ok"]:
                st.warning("Local park_factors.csv was unavailable or could not be matched, so neutral 100 values are being used. Turn on the manual override to change them.")
        if st.button("Refresh park factors", key=f"clean_hits_refresh_park_{matchup.get('game_pk')}"):
            fetch_savant_park_factors.clear()
            st.rerun()
    with c3:
        bullpen_multiplier = st.number_input(
            "Bullpen hit multiplier",
            0.75,
            1.30,
            1.00,
            0.01,
            key="clean_hits_bullpen",
            help="Above 1.00 means an easier-than-average bullpen for hits.",
        )
        sprint_upload = st.file_uploader(
            "Optional sprint-speed CSV",
            type=["csv"],
            key="clean_hits_sprint_upload",
        )
    with c4:
        st.markdown(
            f"""
            <div class="context-note">
            <b>{matchup['away_abbr']} @ {matchup['home_abbr']}</b><br>
            {matchup['venue']}<br>
            Batting team: {selected_team} ({home_away})<br>
            Park LHB: ×{park_hit_factor_lhb / 100:.2f}<br>
            Park RHB: ×{park_hit_factor_rhb / 100:.2f}<br>
            Source: {escape(park_source)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        template = "player_id,Player,sprint_speed\n660271,Shohei Ohtani,\n"
        st.download_button(
            "Sprint-speed CSV template",
            template,
            "sprint_speed_template.csv",
            "text/csv",
            key="clean_hits_sprint_template",
        )

preview_profile, preview_hand = selected_pitcher_profile(df, selected_pitcher)

preview_recent_lineup = infer_recent_lineup(df, selected_team)
lineup_seed = preview_recent_lineup.copy()
lineup_source_label = "Latest observed lineup from the loaded Statcast sample"
lineup_result = None
lineup_game_pk = matchup.get("game_pk")
lineup_side = "away" if home_away == "Away" else "home"

with st.expander("Automatic MLB lineup", expanded=True):
    use_mlb_lineup = st.checkbox(
        "Use the posted MLB lineup automatically",
        value=bool(lineup_game_pk),
        disabled=not bool(lineup_game_pk),
        key=f"clean_hits_use_mlb_lineup_{lineup_game_pk}_{selected_team}",
        help="When all nine hitters are posted, they replace the recent-lineup fallback and set batting order automatically.",
    )

    if lineup_game_pk and use_mlb_lineup:
        lineup_result = fetch_mlb_confirmed_lineup(lineup_game_pk, lineup_side)
        posted_lineup = _clean_lineup_seed(lineup_result.get("lineup"))
        if lineup_result.get("ok") and len(posted_lineup) >= 9:
            lineup_seed = posted_lineup
            lineup_source_label = f"Confirmed MLB lineup · {lineup_result.get('source', '')}"
            st.success(
                f"Confirmed lineup loaded: {len(posted_lineup)} hitters · "
                f"{lineup_result.get('game_state') or matchup.get('status', '')}"
            )
            lineup_display = lineup_result["lineup"].copy()
            lineup_display = lineup_display.rename(columns={
                "LineupSpot": "Order", "Player_MLB": "Player", "Position_MLB": "Pos",
                "LineupRole": "Role",
            })
            st.dataframe(
                lineup_display[[column for column in ["Order", "Player", "Pos", "Role"] if column in lineup_display.columns]],
                hide_index=True,
                use_container_width=True,
                height=360,
            )
        elif lineup_result.get("status") == "Partial":
            st.warning(
                f"MLB currently shows only {len(posted_lineup)} lineup spots. "
                "The recent observed lineup remains the default until all nine are posted."
            )
        else:
            st.info(
                "The confirmed lineup has not been posted yet. "
                "The most recent observed lineup remains the default."
            )
            if lineup_result.get("error"):
                with st.expander("Lineup connection details"):
                    st.code(str(lineup_result["error"]))

        if st.button(
            "Refresh MLB lineup",
            key=f"clean_hits_refresh_lineup_{lineup_game_pk}_{selected_team}",
        ):
            fetch_mlb_confirmed_lineup.clear()
            st.rerun()
    elif not lineup_game_pk:
        st.caption("Automatic lineups require a game selected from the MLB slate. Manual matchups use the recent-lineup fallback.")

st.caption(f"Lineup source: {lineup_source_label}")
preview_board = aggregate_hitters(df, preview_hand, 0)
preview_board = add_roster_candidates(
    preview_board, selected_team, active_roster, lineup_seed, min_pa, include_low_sample
)
if preview_board.empty:
    st.warning(
        "No roster or Statcast hitters were available. Increase the lookback or disable the PA filter."
    )
    st.stop()
preview_names = lookup_names(
    tuple(preview_board["player_id"].dropna().astype(int).unique().tolist())
)
preview_board = preview_board.merge(preview_names, on="player_id", how="left", suffixes=("", "_Lookup"))
preview_board["Player"] = preview_board["Player"].replace("", np.nan).fillna(preview_board.get("Player_Lookup"))
preview_board["Player"] = preview_board["Player"].fillna(
    "MLB ID " + preview_board["player_id"].astype("Int64").astype(str)
)
preview_board = preview_board.drop(columns=["Player_Lookup"], errors="ignore")
preview_board = preview_board.merge(lineup_seed, on="player_id", how="left")
lineup_input = preview_board[["player_id", "Player", "Position", "Bats", "PA", "SampleStatus", "LineupSpot"]].copy()
default_selected = lineup_input["LineupSpot"].notna()
if not default_selected.any():
    default_selected = pd.Series(True, index=lineup_input.index)
lineup_input.insert(0, "Selected", default_selected)
lineup_input = lineup_input.sort_values(["LineupSpot", "Player"], na_position="last")

with st.expander("Confirm lineup", expanded=False):
    st.caption(
        f"Default order: {lineup_source_label}. "
        "You can still edit the order or remove a late scratch below."
    )
    lineup_edits = st.data_editor(
        lineup_input,
        hide_index=True,
        use_container_width=True,
        disabled=["player_id", "Player", "Position", "Bats", "PA", "SampleStatus"],
        column_config={
            "Selected": st.column_config.CheckboxColumn("Use", default=True),
            "LineupSpot": st.column_config.NumberColumn(
                "Lineup spot", min_value=1, max_value=9, step=1
            ),
        },
        key=f"clean_hits_lineup_{selected_pitcher}_{selected_team}_{_lineup_fingerprint(lineup_seed)}",
    )

rankings, pitch_mix = build_hit_board(
    df=df,
    pitcher_id=selected_pitcher,
    team=selected_team,
    min_pa=min_pa,
    end_date=pd.Timestamp(loaded_end),
    park_hit_factor_lhb=park_hit_factor_lhb,
    park_hit_factor_rhb=park_hit_factor_rhb,
    team_runs=team_runs,
    is_away=home_away == "Away",
    starter_innings=starter_innings,
    bullpen_multiplier=bullpen_multiplier,
    lineup_override=lineup_seed,
    lineup_edits=lineup_edits,
    sprint_upload=sprint_upload,
    active_roster=active_roster,
    include_low_sample=include_low_sample,
)

if rankings.empty:
    st.warning("No selected hitters remain after filtering.")
    st.stop()

rankings = apply_binary_probability_calibration(
    rankings, st.session_state.get("hits_probability_calibration")
)
rankings, loaded_odds_quotes, odds_source_mode = render_batter_odds_section(
    rankings, matchup, odds_api_key, "clean_hits"
)

render_board_header(matchup, selected_team, matchup["pitcher_name"], "ADVANCED HIT BOARD")
render_leader_cards(rankings, "Model_1plus_Hit", "HitScore", "model 1+ hit")

quick_tab, contact_tab, matchup_tab, bvp_tab, pitcher_tab, slate_tab, backtest_tab, notes_tab = st.tabs(
    [
        "Quick board", "Contact profile", "Matchup detail", "Batter vs pitcher",
        "Pitcher profile", "Slate Top 10 & pairings", "Backtest & calibration", "Model notes"
    ]
)

with quick_tab:
    quick_columns = [
        "Rank", "Player", "LineupSpot", "Projected_PA", "Projected_Hits", "Model_1plus_Hit",
        "Raw_Model_1plus_Hit", "HitScore", "Confidence_Level", "Adj_xHit_PA", "Contact_Pct", "K_Pct",
        "EffectiveStand", "PitcherSideRead", "PitcherSideAttackScore", "SampleStatus",
        "PitchMatchScore", "ZoneFitScore", "ParkFactor", "Market_Line",
        "Over_Odds", "Under_Odds", "Market_Over_Prob", "Model_Market_Edge", "Line_Source",
    ]
    quick = rankings[[column for column in quick_columns if column in rankings.columns]].copy()
    if not bool(rankings.get("Calibration_Applied", pd.Series(False, index=rankings.index)).fillna(False).any()):
        quick = quick.drop(columns=["Raw_Model_1plus_Hit"], errors="ignore")
    quick = quick.rename(columns={
        "LineupSpot": "Order", "Projected_PA": "Proj PA", "Projected_Hits": "Proj Hits", "Model_1plus_Hit": "1+ Hit",
        "Raw_Model_1plus_Hit": "Raw 1+ Hit", "HitScore": "Hit Score", "Confidence_Level": "Confidence",
        "Adj_xHit_PA": "Adj xHit/PA", "Contact_Pct": "Contact%", "K_Pct": "K%",
        "EffectiveStand": "Bats vs SP", "PitcherSideRead": "Pitcher Read",
        "PitcherSideAttackScore": "Side Attack", "SampleStatus": "Sample",
        "PitchMatchScore": "Pitch Match", "ZoneFitScore": "Zone Fit",
        "ParkFactor": "Park Factor", "Market_Line": "Line",
        "Over_Odds": "Over Odds", "Under_Odds": "Under Odds",
        "Market_Over_Prob": "No-vig Market", "Model_Market_Edge": "Model Edge",
        "Line_Source": "Line Source",
    })
    styler = quick.style.background_gradient(
        cmap="RdYlGn", subset=["Hit Score", "Side Attack", "Pitch Match", "Zone Fit"], vmin=0, vmax=100
    ).format({
        "Order": "{:.0f}", "Proj PA": "{:.2f}", "Proj Hits": "{:.2f}", "1+ Hit": "{:.1%}", "Raw 1+ Hit": "{:.1%}",
        "Hit Score": "{:.1f}", "Adj xHit/PA": "{:.1%}", "Contact%": "{:.1%}",
        "K%": "{:.1%}", "Side Attack": "{:.1f}", "Pitch Match": "{:.1f}", "Zone Fit": "{:.1f}",
        "Park Factor": "{:.0f}", "Line": "{:.1f}", "Over Odds": "{:+.0f}",
        "Under Odds": "{:+.0f}", "No-vig Market": "{:.1%}", "Model Edge": "{:+.1%}",
    })
    st.dataframe(styler, use_container_width=True, hide_index=True, height=520)

with contact_tab:
    columns = [
        "Player", "PA", "Hits", "Hit_PA", "xHit_PA", "Avg_xBA_Contact",
        "Contact_Pct", "Zone_Contact_Pct", "Whiff_Pct", "K_Pct", "LD_Pct",
        "HH_Pct", "SweetSpot_Pct", "Sprint_Speed",
    ]
    contact = rankings[[column for column in columns if column in rankings.columns]].copy()
    contact = contact.rename(columns={
        "Hit_PA": "H/PA", "xHit_PA": "xHit/PA", "Avg_xBA_Contact": "xBA Contact",
        "Contact_Pct": "Contact%", "Zone_Contact_Pct": "Zone Contact%",
        "Whiff_Pct": "Whiff%", "K_Pct": "K%", "LD_Pct": "LD%",
        "HH_Pct": "Hard Hit%", "SweetSpot_Pct": "Sweet Spot%",
        "Sprint_Speed": "Sprint Speed",
    })
    st.caption(
        "Green highlights stronger hit-probability ingredients within the selected offense. "
        "Whiff% and K% are reversed, so lower swing-and-miss risk appears greener."
    )
    positive_columns = [
        column for column in [
            "H/PA", "xHit/PA", "xBA Contact", "Contact%", "Zone Contact%",
            "LD%", "Hard Hit%", "Sweet Spot%", "Sprint Speed"
        ] if column in contact.columns
    ]
    risk_columns = [column for column in ["Whiff%", "K%"] if column in contact.columns]
    contact_style = contact.style
    if positive_columns:
        contact_style = contact_style.background_gradient(
            cmap="RdYlGn", subset=positive_columns, axis=0
        )
    if risk_columns:
        contact_style = contact_style.background_gradient(
            cmap="RdYlGn_r", subset=risk_columns, axis=0
        )
    contact_style = contact_style.set_properties(
        subset=["Player"], **{"font-weight": "700", "background-color": "#f8fafc"}
    ).format({
        "H/PA": "{:.1%}", "xHit/PA": "{:.1%}", "xBA Contact": "{:.3f}",
        "Contact%": "{:.1%}", "Zone Contact%": "{:.1%}", "Whiff%": "{:.1%}",
        "K%": "{:.1%}", "LD%": "{:.1%}", "Hard Hit%": "{:.1%}",
        "Sweet Spot%": "{:.1%}", "Sprint Speed": "{:.1f}",
    })
    st.dataframe(
        contact_style,
        use_container_width=True,
        hide_index=True,
        height=520,
    )

with matchup_tab:
    columns = [
        "Player", "PitchMatchScore", "ZoneFitScore", "PitcherHitScore",
        "RecentFormScore", "BvP_PA", "BvP_H", "BvPScore", "MatchSample",
        "Platoon_PA", "Pitcher_PA",
    ]
    detail = rankings[[column for column in columns if column in rankings.columns]].copy()
    detail = detail.rename(columns={
        "PitchMatchScore": "Pitch Match", "ZoneFitScore": "Zone Fit",
        "PitcherHitScore": "Pitcher Vulnerability", "RecentFormScore": "Recent Form",
        "BvP_PA": "BvP PA", "BvP_H": "BvP Hits", "BvPScore": "BvP Score",
        "MatchSample": "Matched Pitches", "Platoon_PA": "Platoon PA",
        "Pitcher_PA": "Pitcher Split PA",
    })
    score_cols = ["Pitch Match", "Zone Fit", "Pitcher Vulnerability", "Recent Form", "BvP Score"]
    st.dataframe(
        detail.style.background_gradient(cmap="RdYlGn", subset=score_cols, vmin=0, vmax=100).format(
            {column: "{:.1f}" for column in score_cols}
        ),
        use_container_width=True,
        hide_index=True,
        height=520,
    )


with bvp_tab:
    st.caption(
        "Direct history against the selected starting pitcher. BvP Score is heavily "
        "shrunk toward 50 until the hitter reaches a meaningful sample; treat very small "
        "samples as context rather than a standalone reason to bet."
    )
    bvp_columns = [
        "Player", "BvP_PA", "BvP_H", "BvP_Hit_PA", "BvP_xHit_PA",
        "BvP_K", "BvP_K_Pct", "BvP_BBE", "BvP_Avg_EV", "BvP_HH_Pct",
        "BvPScore", "BvP_Last_Date",
    ]
    bvp_view = rankings[[column for column in bvp_columns if column in rankings.columns]].copy()
    bvp_view = bvp_view.rename(columns={
        "BvP_PA": "PA", "BvP_H": "Hits", "BvP_Hit_PA": "H/PA",
        "BvP_xHit_PA": "xH/PA", "BvP_K": "K", "BvP_K_Pct": "K%",
        "BvP_BBE": "BBE", "BvP_Avg_EV": "Avg EV", "BvP_HH_Pct": "Hard Hit%",
        "BvPScore": "BvP Score", "BvP_Last_Date": "Last Faced",
    })
    if "Last Faced" in bvp_view.columns:
        bvp_view["Last Faced"] = pd.to_datetime(bvp_view["Last Faced"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")
    bvp_view = bvp_view.sort_values(["PA", "BvP Score"], ascending=[False, False])
    positive_bvp = [column for column in ["H/PA", "xH/PA", "Avg EV", "Hard Hit%", "BvP Score"] if column in bvp_view.columns]
    risk_bvp = [column for column in ["K%"] if column in bvp_view.columns]
    bvp_style = bvp_view.style
    if positive_bvp:
        bvp_style = bvp_style.background_gradient(cmap="RdYlGn", subset=positive_bvp, axis=0)
    if risk_bvp:
        bvp_style = bvp_style.background_gradient(cmap="RdYlGn_r", subset=risk_bvp, axis=0)
    bvp_style = bvp_style.set_properties(
        subset=["Player"], **{"font-weight": "700", "background-color": "#f8fafc"}
    ).format({
        "PA": "{:.0f}", "Hits": "{:.0f}", "H/PA": "{:.1%}", "xH/PA": "{:.1%}",
        "K": "{:.0f}", "K%": "{:.1%}", "BBE": "{:.0f}", "Avg EV": "{:.1f}",
        "Hard Hit%": "{:.1%}", "BvP Score": "{:.1f}",
    }, na_rep="—")
    st.dataframe(
        bvp_style,
        use_container_width=True,
        hide_index=True,
        height=520,
    )

with pitcher_tab:
    if pitch_mix.empty:
        st.info("No pitch profile was available.")
    else:
        st.dataframe(
            pitch_mix.sort_values("Usage", ascending=False).style.format({
                "Usage": "{:.1%}", "Avg_Speed": "{:.1f}", "Avg_PFX_X": "{:.2f}",
                "Avg_PFX_Z": "{:.2f}", "Avg_Extension": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )


with slate_tab:
    render_slate_tools(
        df=df, matchup=matchup, available_teams=available_teams, min_pa=min_pa,
        loaded_end=loaded_end, include_low_sample=include_low_sample,
        calibration=st.session_state.get("hits_probability_calibration"),
        odds_quotes=loaded_odds_quotes, odds_source_mode=odds_source_mode,
        widget_prefix="clean_hits_slate",
    )

with backtest_tab:
    lineup_status_for_backtest = (
        str(lineup_result.get("status")) if isinstance(lineup_result, dict)
        else ("Manual" if not lineup_game_pk else "Recent lineup fallback")
    )
    render_binary_backtest_tab(
        rankings=rankings,
        matchup=matchup,
        lookback_days=lookback_days,
        loaded_start=loaded_start,
        loaded_end=loaded_end,
        lineup_status=lineup_status_for_backtest,
        calibration_state_key="hits_probability_calibration",
        widget_prefix="clean_hits_backtest",
        extra_context={
            "ParkSource": park_source,
            "TeamImpliedRuns": team_runs,
            "StarterInnings": starter_innings,
            "BullpenMultiplier": bullpen_multiplier,
        },
        whole_slate_board=st.session_state.get(
            f"clean_hits_slate_whole_slate_board_{str(pd.Timestamp(matchup.get('slate_date') or date.today()).date())}",
            pd.DataFrame(),
        ),
        odds_api_key=odds_api_key,
        odds_widget_prefix="clean_hits",
        odds_source_mode=odds_source_mode,
    )

with notes_tab:
    st.markdown(
        """
        ### Reading the board
        - **1+ Hit** combines the modeled hit rate per plate appearance with projected plate appearances.
        - **Projected Hits** is the expected event count: modeled rate per PA × projected PA. It is not itself a probability.
        - **Hit Score** is a 0–100 comparison score within the selected offense, not a literal probability.
        - **Pitcher Read / Side Attack** grades whether the starter has been more attackable or avoidable for LHB or RHB, with small samples shrunk toward neutral.
        - **Pitch Match** compares the hitter with the starter's pitch types, velocity, movement and extension.
        - **Zone Fit** weights hitter production by the locations the starter uses.
        - **Park Factor** is automatically matched to the selected venue from Baseball Savant when its live table is available; 105 means ×1.05 and 95 means ×0.95. A neutral/manual fallback remains available.
        - **Sample** marks hitters below the selected PA threshold. Active-roster players with no history use league-average priors and remain Low confidence.
        - **Confidence** reflects sample size and matchup-data depth, not certainty that the outcome will occur.

        - **Automatic odds** can populate current-game or whole-slate lines, no-vig probability and model edge from The Odds API.
        - **Slate Top 10 & pairings** builds both offenses across the slate, grades hitters from probability, model score and confidence, and creates multiple diversified three-person suggestions.
        - **Backtest & calibration** exports timestamped pregame snapshots, retrieves official results, checks Brier score/log loss, validates Hit Score buckets and creates a probability-calibration JSON.
        - Download the updated master-history CSV after every result update because Streamlit temporary storage can reset.

        The probabilities remain heuristic until tested and calibrated on held-out historical games.
        """
    )
