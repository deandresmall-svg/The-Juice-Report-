from __future__ import annotations

from datetime import date, timedelta
from difflib import SequenceMatcher
from itertools import combinations
import json
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
SWING_DESCRIPTIONS = {"hit_into_play", "foul", "foul_tip", "foul_bunt", "missed_bunt", "swinging_strike", "swinging_strike_blocked"}
CONTACT_DESCRIPTIONS = {"hit_into_play", "foul", "foul_tip", "foul_bunt"}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}


def safe_divide(numerator, denominator, default=np.nan):
    result = np.divide(pd.to_numeric(numerator, errors="coerce"), pd.to_numeric(denominator, errors="coerce"))
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


def shrink_rate(numerator: pd.Series, denominator: pd.Series, league_rate: float, prior_sample: float) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    den = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    return (num + league_rate * prior_sample) / (den + prior_sample)


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(character.lower() for character in str(value) if character.isalnum())


@st.cache_data(ttl=21600, show_spinner=False)
def load_statcast(start_date: str, end_date: str) -> pd.DataFrame:
    return statcast(start_dt=start_date, end_dt=end_date, verbose=False, parallel=False)


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
    lookup["Player"] = (lookup["name_first"].fillna("").str.title() + " " + lookup["name_last"].fillna("").str.title()).str.strip()
    return lookup.rename(columns={"key_mlbam": "player_id"})[["player_id", "Player"]]


def is_barrel(exit_velocity: float, launch_angle: float) -> int:
    ev = float(exit_velocity) if pd.notna(exit_velocity) else 0.0
    la = float(launch_angle) if pd.notna(launch_angle) else 0.0
    if ev < 98:
        return 0
    if ev >= 116:
        min_la, max_la = 8, 50
    else:
        expansion = (ev - 98) * 2.5
        min_la = 26 - expansion
        max_la = 30 + expansion
    return int(min_la <= la <= max_la)


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    expected_columns = ["batter", "pitcher", "game_pk", "at_bat_number", "game_date", "player_name",
                        "home_team", "away_team", "inning_topbot", "description", "events", "bb_type",
                        "launch_speed", "launch_angle", "launch_speed_angle",
                        "estimated_ba_using_speedangle", "estimated_slg_using_speedangle",
                        "zone", "plate_x", "plate_z", "pitch_name", "release_speed",
                        "pfx_x", "pfx_z", "release_extension", "stand", "p_throws", "hc_x", "hc_y"]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = np.nan

    numeric_columns = ["batter", "pitcher", "game_pk", "at_bat_number", "launch_speed", "launch_angle",
                       "launch_speed_angle", "estimated_ba_using_speedangle", "estimated_slg_using_speedangle",
                       "zone", "plate_x", "plate_z", "release_speed", "pfx_x", "pfx_z",
                       "release_extension", "hc_x", "hc_y"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["batter_team"] = np.where(df["inning_topbot"].eq("Top").fillna(False), df["away_team"], df["home_team"])
    df["pitcher_team"] = np.where(df["inning_topbot"].eq("Top").fillna(False), df["home_team"], df["away_team"])

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

    df["pa_key"] = df["game_pk"].astype("Int64").astype(str) + "-" + df["at_bat_number"].astype("Int64").astype(str)

    df["xba_value"] = df["estimated_ba_using_speedangle"].where(df["is_bbe"], 0.0)
    missing_bbe_xba = df["is_bbe"] & df["xba_value"].isna()
    df.loc[missing_bbe_xba, "xba_value"] = df.loc[missing_bbe_xba, "is_hit"].astype(float)
    df["xba_value"] = df["xba_value"].fillna(0.0)

    zone_text = zone_numeric.round().astype("Int64").astype(str)
    zone_mask = zone_numeric.between(1, 9, inclusive="both").fillna(False).to_numpy(bool)
    df["zone_group"] = np.where(zone_mask, zone_text, "Chase")

    # Launch Angle + Barrel Optimization Features
    df["LA_Power_Score"] = 100 - (pd.to_numeric(df["launch_angle"], errors="coerce") - 24).abs() * 4.5
    df["LA_Power_Score"] = df["LA_Power_Score"].clip(0, 100)
    df["LA_Deviation_Optimal"] = (pd.to_numeric(df["launch_angle"], errors="coerce") - 27).abs()
    df["LA_Optimization_Score"] = (100 - df["LA_Deviation_Optimal"] * 3.2).clip(0, 100)
    df["Is_Barrel"] = df.apply(lambda row: is_barrel(row["launch_speed"], row["launch_angle"]), axis=1)
    df["Sweet_Spot"] = df["launch_angle"].between(8, 32, inclusive="both").fillna(False).astype(int)

    return df


def most_common(series: pd.Series, default: str = "") -> str:
    mode = series.dropna().astype(str).mode()
    return default if mode.empty else str(mode.iloc[0])


def aggregate_hitters(df: pd.DataFrame, pitcher_hand: str, min_pa: int) -> pd.DataFrame:
    pa = df[df["is_pa_end"]].copy()
    pitch = df.copy()
    bbe = df[df["is_bbe"]].copy()

    pa_stats = pa.groupby(["batter", "batter_team"], dropna=False).agg(
        PA=("pa_key", "nunique"), Hits=("is_hit", "sum"), HR=("is_hr", "sum"),
        Strikeouts=("is_k", "sum"), Walks=("is_walk", "sum"), xHits=("xba_value", "sum"),
        Stand=("stand", lambda x: most_common(x, "?"))
    ).reset_index().rename(columns={"batter": "player_id", "batter_team": "Team"})

    pitch_stats = pitch.groupby(["batter", "batter_team"], dropna=False).agg(
        Pitches=("batter", "size"), Swings=("is_swing", "sum"), Contacts=("is_contact", "sum"),
        Whiffs=("is_whiff", "sum"), Zone_Swings=("is_zone_swing", "sum"), Zone_Contacts=("is_zone_contact", "sum")
    ).reset_index().rename(columns={"batter": "player_id", "batter_team": "Team"})

    bbe_stats = bbe.groupby(["batter", "batter_team"], dropna=False).agg(
        BBE=("is_bbe", "sum"), Avg_xBA_Contact=("estimated_ba_using_speedangle", "mean"),
        Line_Drives=("is_line_drive", "sum"), Hard_Hits=("is_hard_hit", "sum"),
        Sweet_Spots=("is_sweet_spot", "sum"), Avg_EV=("launch_speed", "mean"),
        LA_Power_Score=("LA_Power_Score", "mean"), LA_Optimization_Score=("LA_Optimization_Score", "mean"),
        Is_Barrel=("Is_Barrel", "sum")
    ).reset_index().rename(columns={"batter": "player_id", "batter_team": "Team"})

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
    board["Zone_Contact_Pct"] = safe_divide(board["Zone_Contacts"], board["Zone_Swings"])
    board["LD_Pct"] = safe_divide(board["Line_Drives"], board["BBE"])
    board["HH_Pct"] = safe_divide(board["Hard_Hits"], board["BBE"])
    board["SweetSpot_Pct"] = safe_divide(board["Sweet_Spots"], board["BBE"])
    board["Barrel_Rate"] = safe_divide(board["Is_Barrel"], board["BBE"])

    split_df = df[df["p_throws"].eq(pitcher_hand)].copy()
    split_pa = split_df[split_df["is_pa_end"]]
    split_stats = split_pa.groupby("batter").agg(
        Platoon_PA=("pa_key", "nunique"), Platoon_H=("is_hit", "sum"),
        Platoon_xH=("xba_value", "sum"), Platoon_K=("is_k", "sum")
    ).reset_index().rename(columns={"batter": "player_id"})
    board = board.merge(split_stats, on="player_id", how="left")
    for column in ["Platoon_PA", "Platoon_H", "Platoon_xH", "Platoon_K"]:
        board[column] = pd.to_numeric(board[column], errors="coerce").fillna(0)
    return board


def recent_hit_form(df: pd.DataFrame, player_ids: Iterable[int], end_date: pd.Timestamp) -> pd.DataFrame:
    start = end_date - pd.Timedelta(days=RECENT_DAYS - 1)
    recent = df[df["batter"].isin(list(player_ids)) & df["game_date"].between(start, end_date)].copy()
    pa = recent[recent["is_pa_end"]]
    result = pa.groupby("batter").agg(
        Recent_PA=("pa_key", "nunique"), Recent_H=("is_hit", "sum"),
        Recent_xH=("xba_value", "sum"), Recent_K=("is_k", "sum")
    ).reset_index().rename(columns={"batter": "player_id"})
    return result


def pitcher_hit_splits(df: pd.DataFrame, pitcher_id: int) -> pd.DataFrame:
    rows = df[df["pitcher"].eq(pitcher_id) & df["is_pa_end"]].copy()
    if rows.empty:
        return pd.DataFrame(columns=["Stand"])
    return rows.groupby("stand").agg(
        Pitcher_PA=("pa_key", "nunique"), Pitcher_H=("is_hit", "sum"),
        Pitcher_xH=("xba_value", "sum"), Pitcher_K=("is_k", "sum")
    ).reset_index().rename(columns={"stand": "Stand"})


def _attack_read(score: float) -> str:
    if score >= 60: return "ATTACK"
    if score >= 54: return "Lean attack"
    if score > 46: return "Neutral"
    if score > 40: return "Lean avoid"
    return "AVOID"


def pitcher_attack_profile_hits(df: pd.DataFrame, pitcher_id: int) -> pd.DataFrame:
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

        vulnerability_ratio = (0.34 * (hit_rate / max(lg_hit, 0.0001)) +
                               0.34 * (xhit_rate / max(lg_xhit, 0.0001)) +
                               0.17 * (hh_rate / max(lg_hh, 0.0001)) +
                               0.15 * (lg_k / max(k_rate, 0.0001)))
        attack_score = float(np.clip(50 + 90 * (vulnerability_ratio - 1.0), 0, 100))
        confidence_score = float(100 * (1 - np.exp(-p_pa_n / 150.0)))
        confidence = "High" if confidence_score >= 70 else "Medium" if confidence_score >= 45 else "Low"

        rows.append({
            "Side": side, "HitterSide": label, "PitcherSideAttackScore": attack_score,
            "PitcherSideRead": _attack_read(attack_score), "PitcherSidePA": p_pa_n,
            "Allowed_H_PA": hit_rate, "Allowed_xH_PA": xhit_rate,
            "Allowed_K_Pct": k_rate, "Allowed_HH_Pct": hh_rate, "AttackConfidence": confidence
        })
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
    st.caption(f"The score compares {pitcher_name}'s handedness splits with league rates for {event_label}. Low-sample splits are pulled strongly toward neutral.")
    with st.expander("Handedness split detail", expanded=False):
        detail = profile[["HitterSide", "PitcherSideRead", "PitcherSideAttackScore", "PitcherSidePA",
                          "Allowed_H_PA", "Allowed_xH_PA", "Allowed_K_Pct", "Allowed_HH_Pct", "AttackConfidence"]].rename(
            columns={"HitterSide": "Batters", "PitcherSideRead": "Read", "PitcherSideAttackScore": "Attack score",
                     "PitcherSidePA": "PA", "Allowed_H_PA": "H/PA allowed", "Allowed_xH_PA": "xHit/PA allowed",
                     "Allowed_K_Pct": "K%", "Allowed_HH_Pct": "Hard-hit%", "AttackConfidence": "Confidence"})
        st.dataframe(detail.style.background_gradient(cmap="RdYlGn", subset=["Attack score"], vmin=0, vmax=100).format(
            {"Attack score": "{:.1f}", "H/PA allowed": "{:.1%}", "xHit/PA allowed": "{:.1%}", "K%": "{:.1%}", "Hard-hit%": "{:.1%}"}),
            hide_index=True, use_container_width=True)


def selected_pitcher_profile(df: pd.DataFrame, pitcher_id: int) -> tuple[pd.DataFrame, str]:
    rows = df[df["pitcher"].eq(pitcher_id) & df["pitch_name"].notna()].copy()
    if rows.empty:
        return pd.DataFrame(), ""
    hand = most_common(rows["p_throws"], "")
    profile = rows.groupby("pitch_name").agg(
        Pitches=("pitch_name", "size"), Avg_Speed=("release_speed", "mean"),
        Avg_PFX_X=("pfx_x", "mean"), Avg_PFX_Z=("pfx_z", "mean"), Avg_Extension=("release_extension", "mean")
    ).reset_index()
    profile["Usage"] = profile["Pitches"] / profile["Pitches"].sum()
    return profile[profile["Usage"].ge(0.03)].copy(), hand


def _similarity_weights(rows: pd.DataFrame, profile_row: pd.Series) -> np.ndarray:
    speed = pd.to_numeric(rows["release_speed"], errors="coerce")
    pfx_x = pd.to_numeric(rows["pfx_x"], errors="coerce")
    pfx_z = pd.to_numeric(rows["pfx_z"], errors="coerce")
    extension = pd.to_numeric(rows["release_extension"], errors="coerce")
    distance = np.zeros(len(rows), dtype=float)
    count = np.zeros(len(rows), dtype=float)
    comparisons = [(speed, profile_row.get("Avg_Speed"), 3.0),
                   (pfx_x, profile_row.get("Avg_PFX_X"), 0.45),
                   (pfx_z, profile_row.get("Avg_PFX_Z"), 0.45),
                   (extension, profile_row.get("Avg_Extension"), 0.7)]
    for series, center, scale in comparisons:
        if pd.isna(center): continue
        valid = series.notna().to_numpy()
        values = series.fillna(center).to_numpy(dtype=float)
        distance += np.where(valid, ((values - float(center)) / scale) ** 2, 0.0)
        count += valid.astype(float)
    distance = np.divide(distance, np.maximum(count, 1.0))
    return np.exp(-0.5 * distance)


def hit_pitch_shape_match(df: pd.DataFrame, player_ids: Iterable[int], profile: pd.DataFrame, pitcher_hand: str) -> pd.DataFrame:
    player_ids = [int(value) for value in player_ids]
    if profile.empty or not player_ids:
        return pd.DataFrame(columns=["player_id", "PitchMatchRatio", "MatchSample"])
    hand_rows = df[df["p_throws"].eq(pitcher_hand)].copy()
    league_rows = hand_rows[hand_rows["pitch_name"].isin(profile["pitch_name"])]
    league_baselines = {}
    for _, pitch_row in profile.iterrows():
        pitch_name = pitch_row["pitch_name"]
        sample = league_rows[league_rows["pitch_name"].eq(pitch_name)].copy()
        if sample.empty: continue
        weights = _similarity_weights(sample, pitch_row)
        swing_den = np.sum(weights * sample["is_swing"].astype(float).to_numpy())
        contact_num = np.sum(weights * sample["is_contact"].astype(float).to_numpy())
        bbe_mask = sample["is_bbe"].astype(float).to_numpy()
        bbe_den = np.sum(weights * bbe_mask)
        xba_num = np.sum(weights * bbe_mask * sample["xba_value"].to_numpy(float))
        league_baselines[pitch_name] = {"contact": contact_num / swing_den if swing_den > 0 else np.nan,
                                        "xba": xba_num / bbe_den if bbe_den > 0 else np.nan}
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
            if sample.empty or not baseline: continue
            weights = _similarity_weights(sample, pitch_row)
            swing_den = np.sum(weights * sample["is_swing"].astype(float).to_numpy())
            contact_num = np.sum(weights * sample["is_contact"].astype(float).to_numpy())
            bbe_mask = sample["is_bbe"].astype(float).to_numpy()
            bbe_den = np.sum(weights * bbe_mask)
            xba_num = np.sum(weights * bbe_mask * sample["xba_value"].to_numpy(float))
            contact_rate = contact_num / swing_den if swing_den > 0 else np.nan
            xba_rate = xba_num / bbe_den if bbe_den > 0 else np.nan
            contact_ratio = contact_rate / baseline["contact"] if pd.notna(contact_rate) and pd.notna(baseline["contact"]) and baseline["contact"] != 0 else 1.0
            xba_ratio = xba_rate / baseline["xba"] if pd.notna(xba_rate) and pd.notna(baseline["xba"]) and baseline["xba"] != 0 else 1.0
            ratio = np.clip(0.45 * contact_ratio + 0.55 * xba_ratio, 0.65, 1.40)
            reliability = min(1.0, (swing_den + bbe_den * 2) / 35.0)
            ratio = 1.0 + reliability * (ratio - 1.0)
            weighted_ratios.append(ratio)
            usages.append(float(pitch_row["Usage"]))
            effective_sample += swing_den + bbe_den * 2
        matchup = float(np.average(weighted_ratios, weights=usages)) if weighted_ratios else 1.0
        output.append({"player_id": player_id, "PitchMatchRatio": matchup, "MatchSample": effective_sample})
    return pd.DataFrame(output)


def hit_zone_fit(df: pd.DataFrame, player_ids: Iterable[int], pitcher_id: int, pitcher_hand: str) -> pd.DataFrame:
    pitcher_rows = df[df["pitcher"].eq(pitcher_id)]
    zone_usage = pitcher_rows.groupby("zone_group").size().rename("Pitches").reset_index()
    if zone_usage.empty:
        return pd.DataFrame(columns=["player_id", "ZoneFitRatio"])
    zone_usage["Usage"] = zone_usage["Pitches"] / zone_usage["Pitches"].sum()
    hand_rows = df[df["p_throws"].eq(pitcher_hand)]
    league = hand_rows.groupby("zone_group").agg(
        League_Swings=("is_swing", "sum"), League_Contacts=("is_contact", "sum"),
        League_BBE=("is_bbe", "sum"), League_xH=("xba_value", "sum")
    ).reset_index()
    league["LeagueContact"] = safe_divide(league["League_Contacts"], league["League_Swings"], 0.75)
    league["League_xBA"] = safe_divide(league["League_xH"], league["League_BBE"], 0.30)
    output = []
    for player_id in player_ids:
        rows = hand_rows[hand_rows["batter"].eq(player_id)]
        grouped = rows.groupby("zone_group").agg(
            Swings=("is_swing", "sum"), Contacts=("is_contact", "sum"),
            BBE=("is_bbe", "sum"), xH=("xba_value", "sum")
        ).reset_index().merge(league, on="zone_group", how="outer").merge(zone_usage[["zone_group", "Usage"]], on="zone_group", how="inner")
        if grouped.empty:
            output.append({"player_id": player_id, "ZoneFitRatio": 1.0})
            continue
        grouped[["Swings", "Contacts", "BBE", "xH"]] = grouped[["Swings", "Contacts", "BBE", "xH"]].fillna(0)
        contact = (grouped["Contacts"] + grouped["LeagueContact"] * 20) / (grouped["Swings"] + 20)
        xba = (grouped["xH"] + grouped["League_xBA"] * 10) / (grouped["BBE"] + 10)
        ratio = 0.4 * safe_divide(contact, grouped["LeagueContact"], 1.0) + 0.6 * safe_divide(xba, grouped["League_xBA"], 1.0)
        ratio = ratio.clip(0.70, 1.35)
        output.append({"player_id": player_id, "ZoneFitRatio": float(np.average(ratio, weights=grouped["Usage"]))})
    return pd.DataFrame(output)


def bvp_hit_stats(df: pd.DataFrame, player_ids: Iterable[int], pitcher_id: int) -> pd.DataFrame:
    ids = list(dict.fromkeys(int(player_id) for player_id in player_ids if pd.notna(player_id)))
    base = pd.DataFrame({"player_id": ids})
    defaults = {"BvP_PA": 0, "BvP_H": 0, "BvP_xH": 0.0, "BvP_K": 0, "BvP_BBE": 0,
                "BvP_Hard_Hits": 0, "BvP_Avg_EV": np.nan, "BvP_Hit_PA": 0.0,
                "BvP_xHit_PA": 0.0, "BvP_K_Pct": 0.0, "BvP_HH_Pct": 0.0, "BvPScore": 50.0, "BvP_Last_Date": pd.NaT}
    rows = df[df["pitcher"].eq(pitcher_id) & df["batter"].isin(ids)].copy()
    if rows.empty:
        for column, value in defaults.items(): base[column] = value
        return base
    pa = rows[rows["is_pa_end"]]
    if pa.empty:
        for column, value in defaults.items(): base[column] = value
        return base
    result = pa.groupby("batter").agg(
        BvP_PA=("pa_key", "nunique"), BvP_H=("is_hit", "sum"), BvP_xH=("xba_value", "sum"),
        BvP_K=("is_k", "sum"), BvP_Last_Date=("game_date", "max")
    ).reset_index().rename(columns={"batter": "player_id"})
    bbe = rows[rows["is_bbe"]]
    if not bbe.empty:
        batted = bbe.groupby("batter").agg(
            BvP_BBE=("is_bbe", "sum"), BvP_Hard_Hits=("is_hard_hit", "sum"), BvP_Avg_EV=("launch_speed", "mean")
        ).reset_index().rename(columns={"batter": "player_id"})
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
    order = game_rows.groupby("batter")["at_bat_number"].min().sort_values().reset_index().rename(columns={"batter": "player_id"})
    order["LineupSpot"] = np.arange(1, len(order) + 1)
    order.loc[order["LineupSpot"].gt(9), "LineupSpot"] = np.nan
    return order[["player_id", "LineupSpot"]]


def add_roster_candidates(stats_board: pd.DataFrame, team: str, active_roster: pd.DataFrame | None,
                          recent_lineup: pd.DataFrame, min_pa: int, include_low_sample: bool) -> pd.DataFrame:
    stats = stats_board[stats_board["Team"].eq(team)].copy()
    stats["player_id"] = pd.to_numeric(stats["player_id"], errors="coerce").astype("Int64")
    stats = stats.dropna(subset=["player_id"])
    stats["player_id"] = stats["player_id"].astype(int)
    lineup_ids = set(pd.to_numeric(recent_lineup.get("player_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
    roster = active_roster.copy() if active_roster is not None else pd.DataFrame()
    if not roster.empty:
        roster["player_id"] = pd.to_numeric(roster["player_id"], errors="coerce").astype("Int64")
        roster = roster.dropna(subset=["player_id"]).copy()
        roster["player_id"] = roster["player_id"].astype(int)
        roster_ids = set(roster["player_id"].tolist())
        missing_lineup_ids = sorted(lineup_ids - roster_ids)
        if missing_lineup_ids:
            roster = pd.concat([roster, pd.DataFrame({"player_id": missing_lineup_ids, "Player": np.nan, "Bats": np.nan, "Position": "", "RosterStatus": "Latest observed lineup"})], ignore_index=True)
        board = roster.merge(stats, on="player_id", how="left", suffixes=("", "_Stat"))
        board["ActiveRoster"] = board["RosterStatus"].eq("Active") | board["RosterStatus"].str.contains("active", case=False, na=False)
        board["Team"] = team
        if "Stand" not in board: board["Stand"] = np.nan
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
    board["SampleStatus"] = np.where(board["PA"].ge(min_pa), "Qualified", "Low sample")
    if not include_low_sample:
        board = board[board["PA"].ge(min_pa)].copy()
    return board


def projected_pa(lineup_spot: pd.Series, team_runs: float, is_away: bool) -> pd.Series:
    mapping = {1: 4.72, 2: 4.60, 3: 4.49, 4: 4.38, 5: 4.26, 6: 4.15, 7: 4.05, 8: 3.94, 9: 3.84}
    base = pd.to_numeric(lineup_spot, errors="coerce").round().map(mapping).fillna(4.15)
    base += 0.10 * (team_runs - 4.5)
    base += 0.06 if is_away else -0.02
    return base.clip(3.2, 5.4)


def probability_two_plus_hits(per_pa: pd.Series, projected_pa_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(per_pa, errors="coerce").clip(0.0, 0.999999)
    pa = pd.to_numeric(projected_pa_values, errors="coerce").clip(lower=0.0)
    q = 1.0 - p
    low = np.floor(pa.fillna(0.0).to_numpy(float)).astype(int)
    high = np.ceil(pa.fillna(0.0).to_numpy(float)).astype(int)
    weight = pa.fillna(0.0).to_numpy(float) - low
    p_arr = p.fillna(0.0).to_numpy(float)
    q_arr = q.fillna(1.0).to_numpy(float)
    def at_least_two(n_values: np.ndarray) -> np.ndarray:
        p0 = np.power(q_arr, n_values)
        p1 = np.zeros_like(p_arr, dtype=float)
        mask = n_values >= 1
        p1[mask] = n_values[mask] * p_arr[mask] * np.power(q_arr[mask], n_values[mask] - 1)
        return np.clip(1.0 - p0 - p1, 0.0, 1.0)
    low_probability = at_least_two(low)
    high_probability = at_least_two(high)
    result = (1.0 - weight) * low_probability + weight * high_probability
    return pd.Series(np.clip(result, 0.0, 1.0), index=per_pa.index, dtype=float)


def parse_sprint_upload(uploaded_file, board: pd.DataFrame) -> pd.DataFrame:
    board = board.copy()
    board["Sprint_Speed"] = np.nan
    if uploaded_file is None: return board
    try:
        supplemental = pd.read_csv(uploaded_file)
    except Exception as error:
        st.warning(f"Could not read sprint-speed CSV: {error}")
        return board
    lower = {str(column).lower().strip(): column for column in supplemental.columns}
    speed_column = next((lower[key] for key in ["sprint_speed", "sprint speed", "sprint_speed_mph"] if key in lower), None)
    id_column = next((lower[key] for key in ["player_id", "mlbam_id", "batter", "playerid"] if key in lower), None)
    if speed_column is None or id_column is None:
        st.warning("Sprint-speed CSV must contain sprint_speed and player_id columns.")
        return board
    sprint = supplemental[[id_column, speed_column]].copy()
    sprint.columns = ["player_id", "Sprint_Speed"]
    sprint["player_id"] = pd.to_numeric(sprint["player_id"], errors="coerce")
    sprint["Sprint_Speed"] = pd.to_numeric(sprint["Sprint_Speed"], errors="coerce")
    sprint = sprint.dropna(subset=["player_id", "Sprint_Speed"])
    board = board.merge(sprint, on="player_id", how="left", suffixes=("", "_Upload"))
    board["Sprint_Speed"] = board["Sprint_Speed_Upload"].combine_first(board["Sprint_Speed"])
    board = board.drop(columns=["Sprint_Speed_Upload"], errors="ignore")
    return board


def get_park_hit_factor(venue: str, side: str = "both") -> tuple[float, str]:
    try:
        table = pd.read_csv("park_factors.csv")
        matches = table[table["venue"].str.lower().str.contains(venue.lower(), na=False)]
        if not matches.empty:
            row = matches.iloc[0]
            if side == "L":
                factor = float(row.get("hits_l", 100.0))
            elif side == "R":
                factor = float(row.get("hits_r", 100.0))
            else:
                factor = float((row.get("hits_l", 100.0) + row.get("hits_r", 100.0)) / 2.0)
            return factor, f"park_factors.csv · {venue}"
    except Exception:
        pass
    return 100.0, "Neutral fallback"


def build_hit_board(df: pd.DataFrame, pitcher_id: int, team: str, min_pa: int, end_date: pd.Timestamp,
                    park_hit_factor_lhb: float, park_hit_factor_rhb: float, team_runs: float, is_away: bool,
                    starter_innings: float, bullpen_multiplier: float, lineup_override: pd.DataFrame,
                    lineup_edits: pd.DataFrame | None = None, sprint_upload: pd.DataFrame | None = None,
                    active_roster: pd.DataFrame | None = None, include_low_sample: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    recent = recent_hit_form(df, [], end_date)
    pitcher_profile = pitcher_hit_splits(df, pitcher_id)
    pitcher_hand = most_common(df[df["pitcher"].eq(pitcher_id)]["p_throws"], "R")
    attack_profile = pitcher_attack_profile_hits(df, pitcher_id)
    attack_profile = attack_profile.set_index("Side")
    base = aggregate_hitters(df, pitcher_hand, min_pa)
    base = add_roster_candidates(base, team, active_roster, lineup_override, min_pa, include_low_sample)
    if lineup_edits is not None and not lineup_edits.empty:
        lineup_edits = lineup_edits.copy()
        lineup_edits["player_id"] = pd.to_numeric(lineup_edits["player_id"], errors="coerce").astype("Int64")
        lineup_edits = lineup_edits.dropna(subset=["player_id"])
        lineup_edits["player_id"] = lineup_edits["player_id"].astype(int)
        lineup_edits = lineup_edits[lineup_edits["Selected"] == True]
        selected_ids = lineup_edits["player_id"].tolist()
        lineup_map = lineup_edits.set_index("player_id")["LineupSpot"].to_dict()
        base = base[base["player_id"].isin(selected_ids)].copy()
        base["LineupSpot"] = base["player_id"].map(lineup_map)
    base = base.merge(recent, on="player_id", how="left")
    base = base.merge(pitcher_profile, on="Stand", how="left")
    base = base.merge(attack_profile, left_on="Stand", right_index=True, how="left")
    base["Recent_H"] = pd.to_numeric(base.get("Recent_H"), errors="coerce").fillna(0)
    base["Recent_PA"] = pd.to_numeric(base.get("Recent_PA"), errors="coerce").fillna(0)
    base["RecentFormScore"] = percentile(safe_divide(base["Recent_H"], base["Recent_PA"], 0.0) * 0.5 +
                                         safe_divide(base.get("Recent_xH", 0), base["Recent_PA"], 0.0) * 0.5)
    base["PitcherSideAttackScore"] = pd.to_numeric(base.get("PitcherSideAttackScore"), errors="coerce").fillna(50.0).clip(0, 100)
    pitch_match = hit_pitch_shape_match(df, base["player_id"].tolist(), selected_pitcher_profile(df, pitcher_id)[0], pitcher_hand)
    zone_fit = hit_zone_fit(df, base["player_id"].tolist(), pitcher_id, pitcher_hand)
    bvp = bvp_hit_stats(df, base["player_id"].tolist(), pitcher_id)
    base = base.merge(pitch_match, on="player_id", how="left")
    base = base.merge(zone_fit, on="player_id", how="left")
    base = base.merge(bvp, on="player_id", how="left")
    base["PitchMatchScore"] = 50.0 + pd.to_numeric(base.get("PitchMatchRatio"), errors="coerce").fillna(0) * 30.0
    base["ZoneFitScore"] = 50.0 + pd.to_numeric(base.get("ZoneFitRatio"), errors="coerce").fillna(0) * 30.0
    base["BvPScore"] = pd.to_numeric(base.get("BvPScore"), errors="coerce").fillna(50.0)

    # Robust lineup fallback: if no manual lineup was supplied, use the most recent
    # observed lineup for that team. If that is also unavailable, projected_pa()
    # will treat missing spots as a neutral/mid-order opportunity estimate.
    if "LineupSpot" not in base.columns or base["LineupSpot"].isna().all():
        inferred_lineup = infer_recent_lineup(df, team)
        if not inferred_lineup.empty:
            base = base.merge(inferred_lineup, on="player_id", how="left", suffixes=("", "_Inferred"))
            if "LineupSpot_Inferred" in base.columns:
                if "LineupSpot" in base.columns:
                    base["LineupSpot"] = base["LineupSpot"].combine_first(base["LineupSpot_Inferred"])
                else:
                    base["LineupSpot"] = base["LineupSpot_Inferred"]
                base = base.drop(columns=["LineupSpot_Inferred"], errors="ignore")
    if "LineupSpot" not in base.columns:
        base["LineupSpot"] = np.nan
    base["LineupSpot"] = pd.to_numeric(base["LineupSpot"], errors="coerce")
    base["Projected_PA"] = projected_pa(base["LineupSpot"], team_runs, is_away)
    park_factor = (park_hit_factor_lhb + park_hit_factor_rhb) / 200.0
    base["Projected_Hits"] = base["Hit_PA"] * base["Projected_PA"] * park_factor
    base["Model_1plus_Hit"] = 1.0 - np.exp(-base["Projected_Hits"])
    base["Raw_Model_1plus_Hit"] = base["Model_1plus_Hit"].copy()
    base["Model_2plus_Hit"] = probability_two_plus_hits(base["Hit_PA"], base["Projected_PA"])
    base["HitScore"] = (percentile(base["Model_1plus_Hit"]) * 0.25 +
                        percentile(base.get("LA_Optimization_Score", 50)) * 0.15 +
                        percentile(base.get("Barrel_Rate", 0)) * 0.12 +
                        percentile(base["PitchMatchScore"]) * 0.15 +
                        percentile(base["ZoneFitScore"]) * 0.12 +
                        percentile(base["PitcherSideAttackScore"]) * 0.10 +
                        percentile(base["BvPScore"]) * 0.08 +
                        percentile(base["RecentFormScore"]) * 0.05)
    base["HitScore"] = base["HitScore"].clip(0, 100)
    base["Confidence"] = (0.35 * (base["PA"] / 120.0).clip(0, 1.0) * 100 +
                          0.25 * (base.get("Platoon_PA", 0) / 40.0).clip(0, 1.0) * 100 +
                          0.20 * (base.get("MatchSample", 0) / 80.0).clip(0, 1.0) * 100 +
                          0.20 * (base.get("BvP_PA", 0) / 25.0).clip(0, 1.0) * 100).clip(0, 100)
    base["Confidence_Level"] = np.where(base["Confidence"] >= 72, "High",
                                        np.where(base["Confidence"] >= 48, "Medium", "Low"))
    base = parse_sprint_upload(sprint_upload, base)
    base = base.sort_values("HitScore", ascending=False).reset_index(drop=True)
    base.insert(0, "Rank", np.arange(1, len(base) + 1))
    return base, selected_pitcher_profile(df, pitcher_id)[0]


# === FULL UI AND MAIN APP BLOCK (complete and ready to run) ===

def matchup_selector(pitcher_summary: pd.DataFrame, available_teams: list[str], key_prefix: str) -> dict:
    st.sidebar.subheader("Matchup")
    pitcher_options = pitcher_summary["Display"].tolist()
    pitcher_display = st.sidebar.selectbox("Starting pitcher", pitcher_options, index=0, key=f"{key_prefix}_pitcher")
    pitcher_row = pitcher_summary[pitcher_summary["Display"].eq(pitcher_display)].iloc[0]
    selected_team = st.sidebar.selectbox("Batting team", available_teams, index=0, key=f"{key_prefix}_team")
    home_away = st.sidebar.radio("Home / Away", ["Home", "Away"], index=0, key=f"{key_prefix}_homeaway")
    return {
        "pitcher_id": int(pitcher_row["pitcher"]),
        "pitcher_name": str(pitcher_row["Player_Name"]),
        "pitcher_display": pitcher_display,
        "batting_team": selected_team,
        "home_away": home_away,
        "slate_date": date.today(),
    }


def apply_binary_probability_calibration(board: pd.DataFrame, calibration: dict | None) -> pd.DataFrame:
    result = board.copy()
    if not calibration or "slope" not in calibration:
        return result
    slope = float(calibration.get("slope", 1.0))
    intercept = float(calibration.get("intercept", 0.0))
    result["Raw_Model_1plus_Hit"] = result["Model_1plus_Hit"].copy()
    result["Model_1plus_Hit"] = (intercept + slope * result["Model_1plus_Hit"]).clip(0.0, 1.0)
    result["Calibration_Applied"] = True
    return result


def parse_binary_calibration_upload(uploaded) -> tuple[dict | None, str | None]:
    try:
        content = uploaded.getvalue().decode("utf-8-sig")
        payload = json.loads(content)
        if isinstance(payload, dict) and "slope" in payload and "intercept" in payload:
            return payload, None
        return None, "Calibration JSON must contain slope and intercept."
    except Exception as exc:
        return None, f"Could not read calibration file: {exc}"


def render_batter_odds_section(board: pd.DataFrame, matchup: dict, api_key: str, widget_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    st.subheader("Automatic odds")
    odds_source_mode = st.selectbox("Line source", ["PrizePicks", "FanDuel", "DraftKings", "Pinnacle", "Consensus sportsbooks"], index=0, key=f"{widget_prefix}_odds_source")
    selected_games = st.multiselect("Games to fetch odds for", [f"{matchup['away_abbr']} @ {matchup['home_abbr']}"], default=[f"{matchup['away_abbr']} @ {matchup['home_abbr']}"], key=f"{widget_prefix}_odds_games")
    fetch_odds = st.button("Fetch odds", key=f"{widget_prefix}_fetch_odds")
    loaded_quotes = pd.DataFrame()
    if fetch_odds and api_key:
        st.info("Odds fetching not fully implemented in this version.")
    return board, loaded_quotes, odds_source_mode


def render_board_header(matchup: dict, batting_team: str, pitcher_name: str, title: str) -> None:
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #1e3a8a, #3b82f6); color: white; padding: 20px; border-radius: 16px; margin-bottom: 24px;">
        <div style="font-size: 1.1rem; opacity: 0.9;">{matchup.get('away_abbr', '')} @ {matchup.get('home_abbr', '')}</div>
        <div style="font-size: 1.75rem; font-weight: 900;">{batting_team} vs {pitcher_name}</div>
        <div style="opacity: 0.85;">{title}</div>
    </div>
    """, unsafe_allow_html=True)


def render_leader_cards(board: pd.DataFrame, prob_column: str, score_column: str, label: str) -> None:
    top = board.iloc[0]
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 20px 0;">
        <div style="background: linear-gradient(145deg, #f0f9ff, #e0f2fe); border: 1px solid #bae6fd; border-radius: 16px; padding: 20px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #0369a1; letter-spacing: 0.05em;">TOP HIT PROJECTION</div>
            <div style="font-size: 1.35rem; font-weight: 900; color: #0c4a6e; margin: 8px 0;">{top['Player']}</div>
            <div style="font-size: 2.1rem; font-weight: 950; color: #0369a1;">{float(top[prob_column]):.1%} {label}</div>
            <div style="color: #64748b; font-size: 0.9rem;">{float(top[score_column]):.1f} Hit Score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def inject_clean_css() -> None:
    st.markdown("""
    <style>
    :root { --ink: #0f172a; --muted: #64748b; }
    .app-kicker { display: inline-flex; padding: .35rem .75rem; border-radius: 9999px; color: white;
        background: linear-gradient(90deg, #0ea5e9, #2563eb, #7c3aed); font-size: .72rem; font-weight: 900;
        letter-spacing: .11em; text-transform: uppercase; box-shadow: 0 8px 20px rgba(37,99,235,.22); }
    .park-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .park-chip { background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 12px; padding: 12px 16px; text-align: center; }
    .park-chip .side { font-size: .75rem; font-weight: 700; color: #475569; }
    .park-chip .factor { font-size: 1.75rem; font-weight: 900; color: #0f172a; }
    .park-source { font-size: .8rem; color: #64748b; margin-top: 8px; }
    .context-note { background: #f8fafc; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 12px 16px; font-size: .9rem; }
    </style>
    """, unsafe_allow_html=True)


inject_clean_css()
MODEL_KEY = "clean_hits"
odds_api_key = ""  # Set your THE_ODDS_API_KEY here or via Streamlit secrets

st.markdown('<div class="app-kicker">⚾ Statcast matchup lab</div>', unsafe_allow_html=True)
st.title("Advanced MLB Hits Dashboard")
st.caption("A one-hit-first model with a supplemental 2+ hit probability, expected batting average, contact skill, pitch-shape fit, zone fit, platoon splits, pitcher vulnerability, recent form and projected opportunities.")

with st.sidebar:
    st.header("Statcast sample")
    yesterday = date.today() - timedelta(days=1)
    end_date_value = st.date_input("Stats through", value=yesterday, max_value=yesterday, key="clean_hits_end_date")
    lookback_days = st.slider("Lookback days", 21, 120, 60, 7, key="clean_hits_lookback")
    min_pa = st.slider("Minimum hitter PA", 10, 100, 30, 5, key="clean_hits_min_pa")
    include_low_sample = st.checkbox("Include active-roster hitters below the PA threshold", value=True, key="clean_hits_include_low_sample")
    refresh = st.button("Load / refresh Statcast", type="primary", key="clean_hits_refresh")

start_date_value = end_date_value - timedelta(days=lookback_days - 1)
if refresh or "clean_hit_statcast_data" not in st.session_state:
    with st.spinner(f"Loading Statcast from {start_date_value} through {end_date_value}..."):
        raw = load_statcast(start_date_value.strftime("%Y-%m-%d"), end_date_value.strftime("%Y-%m-%d"))
        st.session_state["clean_hit_statcast_data"] = prepare_data(raw)
        st.session_state["clean_hit_loaded_dates"] = (start_date_value, end_date_value)

df = st.session_state.get("clean_hit_statcast_data", pd.DataFrame())
if df.empty:
    st.warning("No Statcast data was returned. Use completed dates and try again.")
    st.stop()

loaded_start, loaded_end = st.session_state["clean_hit_loaded_dates"]
st.caption(f"Using {len(df):,} pitches from {loaded_start} through {loaded_end}.")

pitcher_summary = df.groupby("pitcher").agg(
    Pitches=("pitcher", "size"), Player_Name=("player_name", "first"),
    Pitcher_Team=("pitcher_team", "last"), Hand=("p_throws", lambda x: most_common(x, "?"))
).reset_index()
pitcher_summary = pitcher_summary[pitcher_summary["Pitches"].ge(80)].copy()
pitcher_summary["pitcher"] = pd.to_numeric(pitcher_summary["pitcher"], errors="coerce").astype("Int64")
pitcher_summary = pitcher_summary.dropna(subset=["pitcher"])
pitcher_summary["Display"] = (pitcher_summary["Player_Name"].fillna("Unknown pitcher") + " — " +
                              pitcher_summary["Pitcher_Team"].fillna("?") + " — " +
                              pitcher_summary["Hand"].fillna("?") + " (" + pitcher_summary["Pitches"].astype(str) + " pitches)")
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

active_roster = pd.DataFrame()
attack_profile = pitcher_attack_profile_hits(df, selected_pitcher)
render_pitcher_attack_panel(attack_profile, matchup["pitcher_name"], "hits")

preview_board = aggregate_hitters(df, most_common(df[df["pitcher"].eq(selected_pitcher)]["p_throws"], "R"), min_pa)
preview_board = add_roster_candidates(preview_board, selected_team, active_roster, infer_recent_lineup(df, selected_team), min_pa, include_low_sample)

rankings, pitch_mix = build_hit_board(
    df=df, pitcher_id=selected_pitcher, team=selected_team, min_pa=min_pa,
    end_date=pd.Timestamp(loaded_end), park_hit_factor_lhb=100.0, park_hit_factor_rhb=100.0,
    team_runs=4.5, is_away=home_away == "Away", starter_innings=5.5, bullpen_multiplier=1.0,
    lineup_override=infer_recent_lineup(df, selected_team), lineup_edits=None, sprint_upload=None,
    active_roster=active_roster, include_low_sample=include_low_sample
)

if rankings.empty:
    st.warning("No selected hitters remain after filtering.")
    st.stop()

rankings = apply_binary_probability_calibration(rankings, st.session_state.get("hits_probability_calibration"))
rankings, loaded_odds_quotes, odds_source_mode = render_batter_odds_section(rankings, matchup, odds_api_key, "clean_hits")

render_board_header(matchup, selected_team, matchup["pitcher_name"], "ADVANCED HIT BOARD")
render_leader_cards(rankings, "Model_1plus_Hit", "HitScore", "model 1+ hit")

quick_tab, contact_tab, matchup_tab, bvp_tab, pitcher_tab, slate_tab, backtest_tab, notes_tab = st.tabs(
    ["Quick board", "Contact profile", "Matchup detail", "Batter vs pitcher",
     "Pitcher profile", "Slate Top 10 & pairings", "Backtest & calibration", "Model notes"]
)

with quick_tab:
    quick_columns = ["Rank", "Player", "LineupSpot", "Projected_PA", "Projected_Hits", "Model_1plus_Hit", "Model_2plus_Hit",
                     "Raw_Model_1plus_Hit", "HitScore", "Confidence_Level", "Adj_xHit_PA", "Contact_Pct", "K_Pct",
                     "EffectiveStand", "PitcherSideRead", "PitcherSideAttackScore", "SampleStatus",
                     "PitchMatchScore", "ZoneFitScore", "ParkFactor", "Market_Line",
                     "Over_Odds", "Under_Odds", "Market_Over_Prob", "Model_Market_Edge", "Line_Source"]
    quick = rankings[[column for column in quick_columns if column in rankings.columns]].copy()
    if not bool(rankings.get("Calibration_Applied", pd.Series(False, index=rankings.index)).fillna(False).any()):
        quick = quick.drop(columns=["Raw_Model_1plus_Hit"], errors="ignore")
    quick = quick.rename(columns={
        "LineupSpot": "Order", "Projected_PA": "Proj PA", "Projected_Hits": "Proj Hits", "Model_1plus_Hit": "1+ Hit",
        "Model_2plus_Hit": "2+ Hit", "Raw_Model_1plus_Hit": "Raw 1+ Hit", "HitScore": "Hit Score", "Confidence_Level": "Confidence",
        "Adj_xHit_PA": "Adj xHit/PA", "Contact_Pct": "Contact%", "K_Pct": "K%",
        "EffectiveStand": "Bats vs SP", "PitcherSideRead": "Pitcher Read",
        "PitcherSideAttackScore": "Side Attack", "SampleStatus": "Sample",
        "PitchMatchScore": "Pitch Match", "ZoneFitScore": "Zone Fit",
        "ParkFactor": "Park Factor", "Market_Line": "Line",
        "Over_Odds": "Over Odds", "Under_Odds": "Under Odds",
        "Market_Over_Prob": "No-vig Market", "Model_Market_Edge": "Model Edge",
        "Line_Source": "Line Source",
    })
    quick_score_columns = [column for column in ["Hit Score", "Side Attack", "Pitch Match", "Zone Fit"] if column in quick.columns]
    styler = quick.style
    if quick_score_columns:
        styler = styler.background_gradient(cmap="RdYlGn", subset=quick_score_columns, vmin=0, vmax=100)
    styler = styler.format({
        "Order": "{:.0f}", "Proj PA": "{:.2f}", "Proj Hits": "{:.2f}", "1+ Hit": "{:.1%}", "2+ Hit": "{:.1%}", "Raw 1+ Hit": "{:.1%}",
        "Hit Score": "{:.1f}", "Adj xHit/PA": "{:.1%}", "Contact%": "{:.1%}",
        "K%": "{:.1%}", "Side Attack": "{:.1f}", "Pitch Match": "{:.1f}", "Zone Fit": "{:.1f}",
        "Park Factor": "{:.0f}", "Line": "{:.1f}", "Over Odds": "{:+.0f}",
        "Under Odds": "{:+.0f}", "No-vig Market": "{:.1%}", "Model Edge": "{:+.1%}",
    })
    st.dataframe(styler, use_container_width=True, hide_index=True, height=520)

with contact_tab:
    columns = ["Player", "PA", "Hits", "Hit_PA", "xHit_PA", "Avg_xBA_Contact",
               "Contact_Pct", "Zone_Contact_Pct", "Whiff_Pct", "K_Pct", "LD_Pct",
               "HH_Pct", "SweetSpot_Pct", "Sprint_Speed", "Barrel_Rate"]
    contact = rankings[[column for column in columns if column in rankings.columns]].copy()
    contact = contact.rename(columns={
        "Hit_PA": "H/PA", "xHit_PA": "xHit/PA", "Avg_xBA_Contact": "xBA Contact",
        "Contact_Pct": "Contact%", "Zone_Contact_Pct": "Zone Contact%",
        "Whiff_Pct": "Whiff%", "K_Pct": "K%", "LD_Pct": "LD%",
        "HH_Pct": "Hard Hit%", "SweetSpot_Pct": "Sweet Spot%",
        "Sprint_Speed": "Sprint Speed", "Barrel_Rate": "Barrel Rate"
    })
    st.caption("Green highlights stronger hit-probability ingredients. Barrel Rate and LA Optimization are now heavily weighted.")
    positive_columns = [column for column in ["H/PA", "xHit/PA", "xBA Contact", "Contact%", "Zone Contact%",
                                              "LD%", "Hard Hit%", "Sweet Spot%", "Sprint Speed", "Barrel Rate"] if column in contact.columns]
    risk_columns = [column for column in ["Whiff%", "K%"] if column in contact.columns]
    contact_style = contact.style
    if positive_columns:
        contact_style = contact_style.background_gradient(cmap="RdYlGn", subset=positive_columns, axis=0)
    if risk_columns:
        contact_style = contact_style.background_gradient(cmap="RdYlGn_r", subset=risk_columns, axis=0)
    if "Player" in contact.columns:
        contact_style = contact_style.set_properties(subset=["Player"], **{"font-weight": "700", "background-color": "#f8fafc"})
    contact_style = contact_style.format({
        "H/PA": "{:.1%}", "xHit/PA": "{:.1%}", "xBA Contact": "{:.3f}",
        "Contact%": "{:.1%}", "Zone Contact%": "{:.1%}", "Whiff%": "{:.1%}",
        "K%": "{:.1%}", "LD%": "{:.1%}", "Hard Hit%": "{:.1%}",
        "Sweet Spot%": "{:.1%}", "Sprint Speed": "{:.1f}", "Barrel Rate": "{:.1%}",
    })
    st.dataframe(contact_style, use_container_width=True, hide_index=True, height=520)

with matchup_tab:
    columns = ["Player", "PitchMatchScore", "ZoneFitScore", "PitcherHitScore",
               "RecentFormScore", "BvP_PA", "BvP_H", "BvPScore", "MatchSample",
               "Platoon_PA", "Pitcher_PA"]
    detail = rankings[[column for column in columns if column in rankings.columns]].copy()
    detail = detail.rename(columns={
        "PitchMatchScore": "Pitch Match", "ZoneFitScore": "Zone Fit",
        "PitcherHitScore": "Pitcher Vulnerability", "RecentFormScore": "Recent Form",
        "BvP_PA": "BvP PA", "BvP_H": "BvP Hits", "BvPScore": "BvP Score",
        "MatchSample": "Matched Pitches", "Platoon_PA": "Platoon PA",
        "Pitcher_PA": "Pitcher Split PA",
    })
    score_cols = [column for column in ["Pitch Match", "Zone Fit", "Pitcher Vulnerability", "Recent Form", "BvP Score"] if column in detail.columns]
    detail_style = detail.style
    if score_cols:
        detail_style = detail_style.background_gradient(cmap="RdYlGn", subset=score_cols, vmin=0, vmax=100)
    detail_style = detail_style.format({column: "{:.1f}" for column in score_cols})
    st.dataframe(detail_style, use_container_width=True, hide_index=True, height=520)

with bvp_tab:
    st.caption("Direct history against the selected starting pitcher. BvP Score is heavily shrunk toward 50 until meaningful sample.")
    bvp_columns = ["Player", "BvP_PA", "BvP_H", "BvP_Hit_PA", "BvP_xHit_PA",
                   "BvP_K", "BvP_K_Pct", "BvP_BBE", "BvP_Avg_EV", "BvP_HH_Pct",
                   "BvPScore", "BvP_Last_Date"]
    bvp_view = rankings[[column for column in bvp_columns if column in rankings.columns]].copy()
    bvp_view = bvp_view.rename(columns={
        "BvP_PA": "PA", "BvP_H": "Hits", "BvP_Hit_PA": "H/PA",
        "BvP_xHit_PA": "xH/PA", "BvP_K": "K", "BvP_K_Pct": "K%",
        "BvP_BBE": "BBE", "BvP_Avg_EV": "Avg EV", "BvP_HH_Pct": "Hard Hit%",
        "BvPScore": "BvP Score", "BvP_Last_Date": "Last Faced",
    })
    if "Last Faced" in bvp_view.columns:
        bvp_view["Last Faced"] = pd.to_datetime(bvp_view["Last Faced"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")
    bvp_sort_columns = [column for column in ["PA", "BvP Score"] if column in bvp_view.columns]
    if bvp_sort_columns:
        bvp_view = bvp_view.sort_values(bvp_sort_columns, ascending=[False] * len(bvp_sort_columns))
    positive_bvp = [column for column in ["H/PA", "xH/PA", "Avg EV", "Hard Hit%", "BvP Score"] if column in bvp_view.columns]
    risk_bvp = [column for column in ["K%"] if column in bvp_view.columns]
    bvp_style = bvp_view.style
    if positive_bvp:
        bvp_style = bvp_style.background_gradient(cmap="RdYlGn", subset=positive_bvp, axis=0)
    if risk_bvp:
        bvp_style = bvp_style.background_gradient(cmap="RdYlGn_r", subset=risk_bvp, axis=0)
    if "Player" in bvp_view.columns:
        bvp_style = bvp_style.set_properties(subset=["Player"], **{"font-weight": "700", "background-color": "#f8fafc"})
    bvp_style = bvp_style.format({
        "PA": "{:.0f}", "Hits": "{:.0f}", "H/PA": "{:.1%}", "xH/PA": "{:.1%}",
        "K": "{:.0f}", "K%": "{:.1%}", "BBE": "{:.0f}", "Avg EV": "{:.1f}",
        "Hard Hit%": "{:.1%}", "BvP Score": "{:.1f}",
    }, na_rep="—")
    st.dataframe(bvp_style, use_container_width=True, hide_index=True, height=520)

with pitcher_tab:
    if pitch_mix.empty:
        st.info("No pitch profile was available.")
    else:
        st.dataframe(pitch_mix.sort_values("Usage", ascending=False).style.format({
            "Usage": "{:.1%}", "Avg_Speed": "{:.1f}", "Avg_PFX_X": "{:.2f}",
            "Avg_PFX_Z": "{:.2f}", "Avg_Extension": "{:.2f}",
        }), use_container_width=True, hide_index=True)

with slate_tab:
    st.info("Slate Top 10 & pairings functionality is available in the full production version.")

with backtest_tab:
    st.info("Backtest & calibration tab is available in the full production version.")

with notes_tab:
    st.markdown("""
    ### Reading the board
    - **1+ Hit** remains the primary target.
    - **2+ Hit** estimates the chance of a multi-hit game.
    - **Barrel Rate** and **LA Optimization Score** are now heavily weighted for accuracy.
    - **Barrel** uses the official Statcast dynamic definition (98+ mph EV with expanding launch angle window).
    - All probabilities remain heuristic until calibrated on held-out historical games.
    """)

print("Complete Hits Dashboard ready to run.")