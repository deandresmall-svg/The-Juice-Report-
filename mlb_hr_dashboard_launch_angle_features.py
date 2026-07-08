from __future__ import annotations

from datetime import date, timedelta
from difflib import SequenceMatcher
from itertools import combinations
import math
import re
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pybaseball import cache, playerid_reverse_lookup, statcast


st.set_page_config(page_title="Advanced MLB Home Run Dashboard", layout="wide")
cache.enable()

RECENT_DAYS = 14
HIT_EVENTS = {"single", "double", "triple", "home_run"}
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
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



def add_launch_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add HR-focused launch-angle and barrel-range features safely.

    Uses regular float arrays instead of pandas nullable booleans so missing
    Statcast values do not trigger `boolean value of NA is ambiguous` errors.
    """
    result = df.copy()
    ev = pd.to_numeric(result.get("launch_speed"), errors="coerce").astype("float64")
    la = pd.to_numeric(result.get("launch_angle"), errors="coerce").astype("float64")

    valid_bbe = ev.notna() & la.notna()
    result["Sweet_Spot"] = (la.between(8, 32, inclusive="both") & valid_bbe).fillna(False).astype(int)

    barrel_range = (
        ev.ge(98)
        & (
            la.between(26, 30, inclusive="both")
            | (ev.ge(105) & la.between(20, 35, inclusive="both"))
        )
        & valid_bbe
    )
    result["Barrel_Range"] = barrel_range.fillna(False).astype(int)

    barrel_expansion = ((ev - 98.0) * 2.5).clip(lower=0.0)
    barrel_min_la = pd.Series(26.0 - barrel_expansion, index=result.index, dtype="float64")
    barrel_max_la = pd.Series(30.0 + barrel_expansion, index=result.index, dtype="float64")
    barrel_min_la = barrel_min_la.mask(ev.ge(116).fillna(False), 8.0)
    barrel_max_la = barrel_max_la.mask(ev.ge(116).fillna(False), 50.0)
    dynamic_barrel = ev.ge(98).fillna(False) & la.ge(barrel_min_la).fillna(False) & la.le(barrel_max_la).fillna(False)
    result["Dynamic_Barrel"] = (dynamic_barrel & valid_bbe).fillna(False).astype(int)

    result["LA_Deviation_from_Optimal"] = (la - 27.0).abs()
    result["LA_Optimization_Score"] = (100.0 - result["LA_Deviation_from_Optimal"] * 3.0).clip(0.0, 100.0)
    result.loc[~valid_bbe, "LA_Optimization_Score"] = np.nan
    result["LA_Power_Score"] = result["LA_Optimization_Score"]
    return result


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
    top_mask = df["inning_topbot"].eq("Top").fillna(False)
    df["batter_team"] = np.where(top_mask, df["away_team"], df["home_team"])
    df["pitcher_team"] = np.where(top_mask, df["home_team"], df["away_team"])

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
    df["is_bbe"] = df["launch_speed"].notna() & df["launch_angle"].notna()
    df["is_barrel"] = df["launch_speed_angle"].eq(6).fillna(False)
    df["is_hard_hit"] = df["launch_speed"].ge(95).fillna(False)
    df["is_sweet_spot"] = df["launch_angle"].between(8, 32, inclusive="both").fillna(False)
    df["is_line_drive"] = df["bb_type"].eq("line_drive").fillna(False)
    df["is_fly_ball"] = df["bb_type"].eq("fly_ball").fillna(False)
    df["is_popup"] = df["bb_type"].eq("popup").fillna(False)
    df["is_ground_ball"] = df["bb_type"].eq("ground_ball").fillna(False)
    df["is_air"] = df["is_line_drive"] | df["is_fly_ball"] | df["is_popup"]

    stand = df["stand"].fillna("").astype(str)
    df["is_pull"] = (
        (stand.eq("R") & df["hc_x"].lt(112.0))
        | (stand.eq("L") & df["hc_x"].gt(138.0))
    ).fillna(False)
    df["is_pull_air"] = df["is_pull"] & df["is_air"]

    df["pa_key"] = (
        df["game_pk"].astype("Int64").astype(str)
        + "-"
        + df["at_bat_number"].astype("Int64").astype(str)
    )

    df["xba_value"] = df["estimated_ba_using_speedangle"].where(df["is_bbe"], 0.0)
    missing_bbe_xba = df["is_bbe"] & df["xba_value"].isna()
    df.loc[missing_bbe_xba, "xba_value"] = df.loc[missing_bbe_xba, "is_hit"].astype(float)
    df["xba_value"] = df["xba_value"].fillna(0.0)

    df["xslg_value"] = df["estimated_slg_using_speedangle"].where(df["is_bbe"], 0.0)
    missing_bbe_xslg = df["is_bbe"] & df["xslg_value"].isna()
    actual_bases = events.map({"single": 1.0, "double": 2.0, "triple": 3.0, "home_run": 4.0}).fillna(0.0)
    df.loc[missing_bbe_xslg, "xslg_value"] = actual_bases[missing_bbe_xslg]
    df["xslg_value"] = df["xslg_value"].fillna(0.0)
    df["xiso_value"] = (df["xslg_value"] - df["xba_value"]).clip(lower=0.0)

    zone_text = zone_numeric.round().astype("Int64").astype(str)
    zone_mask = zone_numeric.between(1, 9, inclusive="both").fillna(False).to_numpy(bool)
    df["zone_group"] = np.where(zone_mask, zone_text, "Chase")

    # HR launch-angle features
    df = add_launch_angle_features(df)
    return df


def most_common(series: pd.Series, default: str = "") -> str:
    mode = series.dropna().astype(str).mode()
    return default if mode.empty else str(mode.iloc[0])


def quantile_90(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return np.nan if values.empty else float(values.quantile(0.90))


def aggregate_hr_hitters(df: pd.DataFrame, pitcher_hand: str, min_pa: int) -> pd.DataFrame:
    pa = df[df["is_pa_end"]].copy()
    bbe = df[df["is_bbe"]].copy()
    pitch = df.copy()

    pa_stats = (
        pa.groupby(["batter", "batter_team"], dropna=False)
        .agg(
            PA=("pa_key", "nunique"),
            HR=("is_hr", "sum"),
            Strikeouts=("is_k", "sum"),
            xSLG_Total=("xslg_value", "sum"),
            xISO_Total=("xiso_value", "sum"),
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
        )
        .reset_index()
        .rename(columns={"batter": "player_id", "batter_team": "Team"})
    )

    bbe_stats = (
        bbe.groupby(["batter", "batter_team"], dropna=False)
        .agg(
            BBE=("is_bbe", "sum"),
            Barrels=("is_barrel", "sum"),
            Hard_Hits=("is_hard_hit", "sum"),
            Sweet_Spots=("is_sweet_spot", "sum"),
            Sweet_Spot_BBE=("Sweet_Spot", "sum"),
            Barrel_Range=("Barrel_Range", "sum"),
            Dynamic_Barrels=("Dynamic_Barrel", "sum"),
            Fly_Balls=("is_fly_ball", "sum"),
            Air_Balls=("is_air", "sum"),
            Pull_Air=("is_pull_air", "sum"),
            Avg_EV=("launch_speed", "mean"),
            EV90=("launch_speed", quantile_90),
            Max_EV=("launch_speed", "max"),
            Avg_LA=("launch_angle", "mean"),
            LA_Power_Score=("LA_Power_Score", "mean"),
            LA_Optimization_Score=("LA_Optimization_Score", "mean"),
            xSLG_Contact=("estimated_slg_using_speedangle", "mean"),
            xBA_Contact=("estimated_ba_using_speedangle", "mean"),
        )
        .reset_index()
        .rename(columns={"batter": "player_id", "batter_team": "Team"})
    )

    board = pa_stats.merge(pitch_stats, on=["player_id", "Team"], how="left")
    board = board.merge(bbe_stats, on=["player_id", "Team"], how="left")
    board = board[board["PA"].ge(min_pa)].copy()
    if board.empty:
        return board

    board["HR_PA"] = safe_divide(board["HR"], board["PA"])
    board["Brl_PA"] = safe_divide(board["Barrels"], board["PA"])
    board["Brl_BIP"] = safe_divide(board["Barrels"], board["BBE"])
    board["xSLG_PA"] = safe_divide(board["xSLG_Total"], board["PA"])
    board["xISO_PA"] = safe_divide(board["xISO_Total"], board["PA"])
    board["HH_Pct"] = safe_divide(board["Hard_Hits"], board["BBE"])
    board["SweetSpot_Pct"] = safe_divide(board["Sweet_Spots"], board["BBE"])
    board["SweetSpot_BBE_Pct"] = safe_divide(board.get("Sweet_Spot_BBE", 0), board["BBE"])
    board["Barrel_Range_BIP"] = safe_divide(board.get("Barrel_Range", 0), board["BBE"])
    board["Dynamic_Brl_BIP"] = safe_divide(board.get("Dynamic_Barrels", 0), board["BBE"])
    board["FB_Pct"] = safe_divide(board["Fly_Balls"], board["BBE"])
    board["Air_Pct"] = safe_divide(board["Air_Balls"], board["BBE"])
    board["PullAir_BIP"] = safe_divide(board["Pull_Air"], board["BBE"])
    board["PullAir_Air"] = safe_divide(board["Pull_Air"], board["Air_Balls"])
    board["Contact_Pct"] = safe_divide(board["Contacts"], board["Swings"])
    board["Whiff_Pct"] = safe_divide(board["Whiffs"], board["Swings"])
    board["K_Pct"] = safe_divide(board["Strikeouts"], board["PA"])

    split = df[df["p_throws"].eq(pitcher_hand) & df["is_pa_end"]].copy()
    split_stats = (
        split.groupby("batter")
        .agg(
            Platoon_PA=("pa_key", "nunique"),
            Platoon_HR=("is_hr", "sum"),
            Platoon_xSLG=("xslg_value", "sum"),
            Platoon_xISO=("xiso_value", "sum"),
            Platoon_Barrels=("is_barrel", "sum"),
        )
        .reset_index()
        .rename(columns={"batter": "player_id"})
    )
    board = board.merge(split_stats, on="player_id", how="left")
    for column in ["Platoon_PA", "Platoon_HR", "Platoon_xSLG", "Platoon_xISO", "Platoon_Barrels"]:
        board[column] = pd.to_numeric(board[column], errors="coerce").fillna(0)
    return board


def recent_hr_form(df: pd.DataFrame, player_ids: Iterable[int], end_date: pd.Timestamp) -> pd.DataFrame:
    start = end_date - pd.Timedelta(days=RECENT_DAYS - 1)
    recent = df[
        df["batter"].isin(list(player_ids))
        & df["game_date"].between(start, end_date)
    ]
    pa = recent[recent["is_pa_end"]]
    return (
        pa.groupby("batter")
        .agg(
            Recent_PA=("pa_key", "nunique"),
            Recent_HR=("is_hr", "sum"),
            Recent_Barrels=("is_barrel", "sum"),
            Recent_xSLG=("xslg_value", "sum"),
            Recent_xISO=("xiso_value", "sum"),
        )
        .reset_index()
        .rename(columns={"batter": "player_id"})
    )


def pitcher_hr_splits(df: pd.DataFrame, pitcher_id: int) -> pd.DataFrame:
    pa = df[df["pitcher"].eq(pitcher_id) & df["is_pa_end"]]
    if pa.empty:
        return pd.DataFrame(columns=["Stand"])
    return (
        pa.groupby("stand")
        .agg(
            Pitcher_PA=("pa_key", "nunique"),
            Pitcher_HR=("is_hr", "sum"),
            Pitcher_Barrels=("is_barrel", "sum"),
            Pitcher_xSLG=("xslg_value", "sum"),
            Pitcher_xISO=("xiso_value", "sum"),
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


def pitcher_attack_profile_hr(df: pd.DataFrame, pitcher_id: int) -> pd.DataFrame:
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

        lg_hr = float(league_pa["is_hr"].sum() / lg_pa_n)
        lg_barrel = float(league_pa["is_barrel"].sum() / lg_pa_n)
        lg_xslg = float(league_pa["xslg_value"].sum() / lg_pa_n)
        lg_xiso = float(league_pa["xiso_value"].sum() / lg_pa_n)
        lg_hh = float(league_bbe["is_hard_hit"].sum() / lg_bbe_n)
        lg_fb = float(league_bbe["is_fly_ball"].sum() / lg_bbe_n)

        hr_rate = (float(pitcher_pa["is_hr"].sum()) + lg_hr * 220) / (p_pa_n + 220)
        barrel_rate = (float(pitcher_pa["is_barrel"].sum()) + lg_barrel * 180) / (p_pa_n + 180)
        xslg_rate = (float(pitcher_pa["xslg_value"].sum()) + lg_xslg * 180) / (p_pa_n + 180)
        xiso_rate = (float(pitcher_pa["xiso_value"].sum()) + lg_xiso * 180) / (p_pa_n + 180)
        hh_rate = (float(pitcher_bbe["is_hard_hit"].sum()) + lg_hh * 110) / (p_bbe_n + 110)
        fb_rate = (float(pitcher_bbe["is_fly_ball"].sum()) + lg_fb * 110) / (p_bbe_n + 110)

        vulnerability_ratio = (
            0.27 * (hr_rate / max(lg_hr, 0.0001))
            + 0.25 * (barrel_rate / max(lg_barrel, 0.0001))
            + 0.22 * (xslg_rate / max(lg_xslg, 0.0001))
            + 0.12 * (xiso_rate / max(lg_xiso, 0.0001))
            + 0.08 * (hh_rate / max(lg_hh, 0.0001))
            + 0.06 * (fb_rate / max(lg_fb, 0.0001))
        )
        attack_score = float(np.clip(50 + 90 * (vulnerability_ratio - 1.0), 0, 100))
        confidence_score = float(100 * (1 - np.exp(-p_pa_n / 180.0)))
        confidence = "High" if confidence_score >= 70 else "Medium" if confidence_score >= 45 else "Low"

        rows.append(
            {
                "Side": side,
                "HitterSide": label,
                "PitcherSideAttackScore": attack_score,
                "PitcherSideRead": _attack_read(attack_score),
                "PitcherSidePA": p_pa_n,
                "Allowed_HR_PA": hr_rate,
                "Allowed_Brl_PA": barrel_rate,
                "Allowed_xSLG_PA": xslg_rate,
                "Allowed_xISO_PA": xiso_rate,
                "Allowed_HH_Pct": hh_rate,
                "Allowed_FB_Pct": fb_rate,
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
                "Allowed_HR_PA", "Allowed_Brl_PA", "Allowed_xSLG_PA", "Allowed_xISO_PA",
                "Allowed_HH_Pct", "Allowed_FB_Pct", "AttackConfidence",
            ]
        ].rename(
            columns={
                "HitterSide": "Batters", "PitcherSideRead": "Read",
                "PitcherSideAttackScore": "Attack score", "PitcherSidePA": "PA",
                "Allowed_HR_PA": "HR/PA allowed", "Allowed_Brl_PA": "Brl/PA allowed",
                "Allowed_xSLG_PA": "xSLG/PA allowed", "Allowed_xISO_PA": "xISO/PA allowed",
                "Allowed_HH_Pct": "Hard-hit%", "Allowed_FB_Pct": "Fly-ball%",
                "AttackConfidence": "Confidence",
            }
        )
        st.dataframe(
            detail.style.background_gradient(cmap="RdYlGn", subset=["Attack score"], vmin=0, vmax=100).format(
                {"Attack score": "{:.1f}", "HR/PA allowed": "{:.2%}", "Brl/PA allowed": "{:.2%}", "xSLG/PA allowed": "{:.3f}", "xISO/PA allowed": "{:.3f}", "Hard-hit%": "{:.1%}", "Fly-ball%": "{:.1%}"}
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
            Allowed_xSLG=("estimated_slg_using_speedangle", "mean"),
            Allowed_Barrel=("is_barrel", "mean"),
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
        values = series.fillna(center).to_numpy(float)
        distance += np.where(valid, ((values - float(center)) / scale) ** 2, 0.0)
        count += valid.astype(float)
    distance = np.divide(distance, np.maximum(count, 1.0))
    return np.exp(-0.5 * distance)


def hr_pitch_shape_match(
    df: pd.DataFrame,
    player_ids: Iterable[int],
    profile: pd.DataFrame,
    pitcher_hand: str,
) -> pd.DataFrame:
    player_ids = [int(value) for value in player_ids]
    if profile.empty or not player_ids:
        return pd.DataFrame(columns=["player_id", "PitchMatchRatio", "MatchSample"])

    hand_rows = df[df["p_throws"].eq(pitcher_hand)].copy()
    league_baselines = {}
    for _, pitch_row in profile.iterrows():
        pitch_name = pitch_row["pitch_name"]
        sample = hand_rows[hand_rows["pitch_name"].eq(pitch_name)].copy()
        if sample.empty:
            continue
        weights = _similarity_weights(sample, pitch_row)
        swing_den = np.sum(weights * sample["is_swing"].astype(float).to_numpy())
        contact_num = np.sum(weights * sample["is_contact"].astype(float).to_numpy())
        bbe = sample["is_bbe"].astype(float).to_numpy()
        bbe_den = np.sum(weights * bbe)
        xslg_num = np.sum(weights * bbe * sample["xslg_value"].to_numpy(float))
        barrel_num = np.sum(weights * sample["is_barrel"].astype(float).to_numpy())
        pull_air_num = np.sum(weights * sample["is_pull_air"].astype(float).to_numpy())
        league_baselines[pitch_name] = {
            "contact": contact_num / swing_den if swing_den > 0 else np.nan,
            "xslg": xslg_num / bbe_den if bbe_den > 0 else np.nan,
            "barrel": barrel_num / bbe_den if bbe_den > 0 else np.nan,
            "pullair": pull_air_num / bbe_den if bbe_den > 0 else np.nan,
        }

    output = []
    for player_id in player_ids:
        player_rows = hand_rows[hand_rows["batter"].eq(player_id)]
        ratios = []
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
            bbe = sample["is_bbe"].astype(float).to_numpy()
            bbe_den = np.sum(weights * bbe)
            xslg_num = np.sum(weights * bbe * sample["xslg_value"].to_numpy(float))
            barrel_num = np.sum(weights * sample["is_barrel"].astype(float).to_numpy())
            pull_air_num = np.sum(weights * sample["is_pull_air"].astype(float).to_numpy())
            contact = contact_num / swing_den if swing_den > 0 else np.nan
            xslg = xslg_num / bbe_den if bbe_den > 0 else np.nan
            barrel = barrel_num / bbe_den if bbe_den > 0 else np.nan
            pullair = pull_air_num / bbe_den if bbe_den > 0 else np.nan

            def ratio(value, league_value):
                if pd.notna(value) and pd.notna(league_value) and league_value > 0:
                    return float(value / league_value)
                return 1.0

            pitch_ratio = (
                0.15 * ratio(contact, baseline["contact"])
                + 0.40 * ratio(xslg, baseline["xslg"])
                + 0.30 * ratio(barrel, baseline["barrel"])
                + 0.15 * ratio(pullair, baseline["pullair"])
            )
            pitch_ratio = float(np.clip(pitch_ratio, 0.55, 1.65))
            reliability = min(1.0, (swing_den + bbe_den * 3) / 45.0)
            pitch_ratio = 1.0 + reliability * (pitch_ratio - 1.0)
            ratios.append(pitch_ratio)
            usages.append(float(pitch_row["Usage"]))
            effective_sample += swing_den + bbe_den * 3
        matchup = float(np.average(ratios, weights=usages)) if ratios else 1.0
        output.append(
            {"player_id": player_id, "PitchMatchRatio": matchup, "MatchSample": effective_sample}
        )
    return pd.DataFrame(output)



def build_batter_pitch_type_table(
    df: pd.DataFrame,
    rankings: pd.DataFrame,
    profile: pd.DataFrame,
    pitcher_hand: str,
) -> pd.DataFrame:
    if rankings.empty or profile.empty or not pitcher_hand:
        return pd.DataFrame()

    hand_rows = df[df["p_throws"].eq(pitcher_hand)].copy()
    if hand_rows.empty:
        return pd.DataFrame()

    player_names = rankings.set_index("player_id")["Player"].to_dict()
    player_ids = [int(value) for value in rankings["player_id"].dropna().unique()]
    output: list[dict] = []

    def shrunk_rate(numerator: float, denominator: float, baseline: float, prior: float) -> float:
        baseline = float(baseline) if pd.notna(baseline) else 0.0
        return float((float(numerator) + baseline * prior) / (float(denominator) + prior))

    def safe_ratio(value: float, baseline: float) -> float:
        if pd.isna(value) or pd.isna(baseline) or float(baseline) <= 0:
            return 1.0
        return float(np.clip(float(value) / float(baseline), 0.35, 2.75))

    for _, pitch_row in profile.sort_values("Usage", ascending=False).iterrows():
        pitch_name = str(pitch_row["pitch_name"])
        league = hand_rows[hand_rows["pitch_name"].eq(pitch_name)].copy()
        if league.empty:
            continue

        lg_swings = float(league["is_swing"].sum())
        lg_bbe = float(league["is_bbe"].sum())
        lg_pa = float(league["is_pa_end"].sum())
        lg_contact = float(league["is_contact"].sum()) / max(lg_swings, 1.0)
        lg_whiff = float(league["is_whiff"].sum()) / max(lg_swings, 1.0)
        lg_xslg = float(league.loc[league["is_bbe"], "xslg_value"].sum()) / max(lg_bbe, 1.0)
        lg_xba = float(league.loc[league["is_bbe"], "xba_value"].sum()) / max(lg_bbe, 1.0)
        lg_barrel = float(league["is_barrel"].sum()) / max(lg_bbe, 1.0)
        lg_hard_hit = float(league["is_hard_hit"].sum()) / max(lg_bbe, 1.0)
        lg_pull_air = float(league["is_pull_air"].sum()) / max(lg_bbe, 1.0)
        lg_hr = float(league["is_hr"].sum()) / max(lg_pa, 1.0)

        for player_id in player_ids:
            rows = league[league["batter"].eq(player_id)].copy()
            pitches_seen = int(len(rows))
            swings = int(rows["is_swing"].sum())
            contacts = int(rows["is_contact"].sum())
            whiffs = int(rows["is_whiff"].sum())
            bbe = int(rows["is_bbe"].sum())
            pa_ends = int(rows["is_pa_end"].sum())
            home_runs = int(rows["is_hr"].sum())
            barrels = int(rows["is_barrel"].sum())
            hard_hits = int(rows["is_hard_hit"].sum())
            pull_air = int(rows["is_pull_air"].sum())
            xslg_sum = float(rows.loc[rows["is_bbe"], "xslg_value"].sum())
            xba_sum = float(rows.loc[rows["is_bbe"], "xba_value"].sum())

            contact = shrunk_rate(contacts, swings, lg_contact, 35)
            whiff = shrunk_rate(whiffs, swings, lg_whiff, 35)
            xslg = shrunk_rate(xslg_sum, bbe, lg_xslg, 12)
            xba = shrunk_rate(xba_sum, bbe, lg_xba, 12)
            barrel = shrunk_rate(barrels, bbe, lg_barrel, 15)
            hard_hit = shrunk_rate(hard_hits, bbe, lg_hard_hit, 18)
            pull_air_rate = shrunk_rate(pull_air, bbe, lg_pull_air, 18)
            hr_rate = shrunk_rate(home_runs, pa_ends, lg_hr, 30)

            matchup_ratio = (
                0.30 * safe_ratio(xslg, lg_xslg)
                + 0.22 * safe_ratio(barrel, lg_barrel)
                + 0.16 * safe_ratio(hr_rate, lg_hr)
                + 0.12 * safe_ratio(hard_hit, lg_hard_hit)
                + 0.08 * safe_ratio(pull_air_rate, lg_pull_air)
                + 0.07 * safe_ratio(contact, lg_contact)
                + 0.05 * safe_ratio(lg_whiff, whiff)
            )
            reliability = float(1.0 - np.exp(-(pitches_seen + 3.0 * bbe) / 80.0))
            raw_score = float(np.clip(50.0 + 55.0 * (matchup_ratio - 1.0), 0.0, 100.0))
            pitch_score = float(np.clip(50.0 + reliability * (raw_score - 50.0), 0.0, 100.0))
            sample = "High" if pitches_seen >= 100 or bbe >= 25 else "Medium" if pitches_seen >= 40 or bbe >= 10 else "Low"

            output.append(
                {
                    "player_id": player_id,
                    "Player": player_names.get(player_id, f"MLB ID {player_id}"),
                    "Pitch Type": pitch_name,
                    "Pitcher Usage": float(pitch_row.get("Usage", np.nan)),
                    "Pitcher Velo": float(pitch_row.get("Avg_Speed", np.nan)),
                    "Pitches Seen": pitches_seen,
                    "PA Ends": pa_ends,
                    "BBE": bbe,
                    "HR": home_runs,
                    "HR/PA": hr_rate,
                    "Brl/BIP": barrel,
                    "xSLG Contact": xslg,
                    "xBA Contact": xba,
                    "Hard Hit%": hard_hit,
                    "Pull Air/BIP": pull_air_rate,
                    "Contact%": contact,
                    "Whiff%": whiff,
                    "Pitch Type Score": pitch_score,
                    "Sample": sample,
                }
            )

    return pd.DataFrame(output)

def hr_zone_fit(
    df: pd.DataFrame,
    player_ids: Iterable[int],
    pitcher_id: int,
    pitcher_hand: str,
) -> pd.DataFrame:
    pitcher_rows = df[df["pitcher"].eq(pitcher_id)]
    usage = pitcher_rows.groupby("zone_group").size().rename("Pitches").reset_index()
    if usage.empty:
        return pd.DataFrame(columns=["player_id", "ZoneFitRatio"])
    usage["Usage"] = usage["Pitches"] / usage["Pitches"].sum()

    hand_rows = df[df["p_throws"].eq(pitcher_hand)]
    league = (
        hand_rows.groupby("zone_group")
        .agg(
            League_BBE=("is_bbe", "sum"),
            League_xSLG=("xslg_value", "sum"),
            League_Barrel=("is_barrel", "sum"),
            League_PullAir=("is_pull_air", "sum"),
        )
        .reset_index()
    )
    league["League_xSLG_Rate"] = safe_divide(league["League_xSLG"], league["League_BBE"], 0.50)
    league["League_Barrel_Rate"] = safe_divide(league["League_Barrel"], league["League_BBE"], 0.07)
    league["League_PullAir_Rate"] = safe_divide(league["League_PullAir"], league["League_BBE"], 0.12)

    output = []
    for player_id in player_ids:
        rows = hand_rows[hand_rows["batter"].eq(player_id)]
        grouped = (
            rows.groupby("zone_group")
            .agg(
                BBE=("is_bbe", "sum"),
                xSLG=("xslg_value", "sum"),
                Barrels=("is_barrel", "sum"),
                PullAir=("is_pull_air", "sum"),
            )
            .reset_index()
            .merge(league, on="zone_group", how="outer")
            .merge(usage[["zone_group", "Usage"]], on="zone_group", how="inner")
        )
        if grouped.empty:
            output.append({"player_id": player_id, "ZoneFitRatio": 1.0})
            continue
        grouped[["BBE", "xSLG", "Barrels", "PullAir"]] = grouped[
            ["BBE", "xSLG", "Barrels", "PullAir"]
        ].fillna(0)
        xslg = (grouped["xSLG"] + grouped["League_xSLG_Rate"] * 12) / (grouped["BBE"] + 12)
        barrel = (grouped["Barrels"] + grouped["League_Barrel_Rate"] * 15) / (grouped["BBE"] + 15)
        pullair = (grouped["PullAir"] + grouped["League_PullAir_Rate"] * 15) / (grouped["BBE"] + 15)
        zone_ratio = (
            0.45 * safe_divide(xslg, grouped["League_xSLG_Rate"], 1.0)
            + 0.35 * safe_divide(barrel, grouped["League_Barrel_Rate"], 1.0)
            + 0.20 * safe_divide(pullair, grouped["League_PullAir_Rate"], 1.0)
        ).clip(0.60, 1.50)
        output.append(
            {
                "player_id": player_id,
                "ZoneFitRatio": float(np.average(zone_ratio, weights=grouped["Usage"])),
            }
        )
    return pd.DataFrame(output)


def bvp_hr_stats(df: pd.DataFrame, player_ids: Iterable[int], pitcher_id: int) -> pd.DataFrame:
    ids = list(player_ids)
    pa = df[df["pitcher"].eq(pitcher_id) & df["batter"].isin(ids) & df["is_pa_end"]]
    if pa.empty:
        return pd.DataFrame(
            {"player_id": ids, "BvP_PA": 0, "BvP_HR": 0, "BvP_Barrels": 0, "BvPScore": 50.0}
        )
    result = (
        pa.groupby("batter")
        .agg(
            BvP_PA=("pa_key", "nunique"),
            BvP_HR=("is_hr", "sum"),
            BvP_Barrels=("is_barrel", "sum"),
            BvP_xSLG=("xslg_value", "sum"),
        )
        .reset_index()
        .rename(columns={"batter": "player_id"})
    )
    raw = (
        safe_divide(result["BvP_HR"], result["BvP_PA"], 0.0) * 0.35
        + safe_divide(result["BvP_Barrels"], result["BvP_PA"], 0.0) * 0.35
        + safe_divide(result["BvP_xSLG"], result["BvP_PA"], 0.0) * 0.30
    )
    performance = percentile(raw)
    reliability = np.minimum(result["BvP_PA"] / 20.0, 1.0)
    result["BvPScore"] = 50 + reliability * (performance - 50)
    return pd.DataFrame({"player_id": ids}).merge(result, on="player_id", how="left").fillna(
        {"BvP_PA": 0, "BvP_HR": 0, "BvP_Barrels": 0, "BvP_xSLG": 0.0, "BvPScore": 50.0}
    )


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


def convert_rate_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if not valid.empty and valid.median() > 1.5:
        numeric = numeric / 100.0
    return numeric


BAT_TRACKING_METRICS = [
    "Bat_Speed",
    "Fast_Swing_Rate",
    "Blast_Contact_Rate",
    "Squared_Up_Contact_Rate",
    "Attack_Angle",
    "Attack_Direction",
]


def _normalize_column(value: object) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _bat_name_key(value: object) -> str:
    if pd.isna(value):
        return ""
    raw = str(value).strip()
    if "," in raw:
        pieces = [piece.strip() for piece in raw.split(",", 1)]
        if len(pieces) == 2:
            raw = f"{pieces[1]} {pieces[0]}"
    return normalize_name(raw)


def _first_tracking_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {_normalize_column(column): column for column in frame.columns}
    for alias in aliases:
        key = _normalize_column(alias)
        if key in normalized:
            return normalized[key]
    for alias in aliases:
        key = _normalize_column(alias)
        for normalized_name, original in normalized.items():
            if key and (key in normalized_name or normalized_name in key):
                return original
    return None


def _standardize_bat_tracking_frame(
    raw: pd.DataFrame,
    season: int | None,
    source_label: str,
) -> pd.DataFrame:
    columns = [
        "player_id", "_name_key", *BAT_TRACKING_METRICS,
        "BatTrackingSeason", "BatTrackingSource",
    ]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=columns)

    frame = raw.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    id_column = _first_tracking_column(
        frame,
        ["player_id", "playerid", "mlbam_id", "mlbamid", "batter_id", "batter", "id"],
    )
    name_column = _first_tracking_column(
        frame,
        ["last_name, first_name", "player_name", "player", "name", "last_name_first_name"],
    )

    aliases = {
        "Bat_Speed": [
            "avg_bat_speed", "average_bat_speed", "avg_swing_speed",
            "average_swing_speed", "bat_speed", "competitive_swing_speed",
        ],
        "Fast_Swing_Rate": [
            "fast_swing_rate", "fast_swing_percent", "fast_swing_pct",
            "fast_swing_percentage", "fast_swings_percent", "fast_swings_pct",
        ],
        "Blast_Contact_Rate": [
            "blasts_per_bat_contact", "blast_per_bat_contact",
            "blasts_per_contact", "blast_per_contact", "blasts_contact",
            "blast_contact_rate", "blast_rate", "blast_percent", "blast_pct",
        ],
        "Squared_Up_Contact_Rate": [
            "squared_up_per_bat_contact", "squared_up_per_contact",
            "squared_up_contact", "squared_up_contact_rate",
            "squared_up_rate", "squared_up_percent", "squared_up_pct",
        ],
        "Attack_Angle": [
            "avg_attack_angle", "average_attack_angle", "attack_angle",
        ],
        "Attack_Direction": [
            "avg_attack_direction", "average_attack_direction", "attack_direction",
        ],
    }

    output = pd.DataFrame(index=frame.index)
    if id_column is not None:
        output["player_id"] = pd.to_numeric(frame[id_column], errors="coerce")
    else:
        output["player_id"] = np.nan

    if name_column is not None:
        output["_name_key"] = frame[name_column].map(_bat_name_key)
    else:
        output["_name_key"] = ""

    for target, candidates in aliases.items():
        source = _first_tracking_column(frame, candidates)
        if source is None:
            output[target] = np.nan
        elif target in {
            "Fast_Swing_Rate", "Blast_Contact_Rate", "Squared_Up_Contact_Rate"
        }:
            output[target] = convert_rate_column(frame[source])
        else:
            output[target] = pd.to_numeric(frame[source], errors="coerce")

    contacts_column = _first_tracking_column(
        frame,
        ["bat_contacts", "contacts", "contact", "batted_ball_contacts"],
    )
    blast_count_column = _first_tracking_column(frame, ["blasts", "blast_count"])
    squared_count_column = _first_tracking_column(
        frame, ["squared_ups", "squared_up", "squared_up_count"]
    )
    if contacts_column is not None:
        contacts = pd.to_numeric(frame[contacts_column], errors="coerce")
        if blast_count_column is not None:
            derived = safe_divide(
                pd.to_numeric(frame[blast_count_column], errors="coerce"), contacts
            )
            output["Blast_Contact_Rate"] = output["Blast_Contact_Rate"].combine_first(derived)
        if squared_count_column is not None:
            derived = safe_divide(
                pd.to_numeric(frame[squared_count_column], errors="coerce"), contacts
            )
            output["Squared_Up_Contact_Rate"] = (
                output["Squared_Up_Contact_Rate"].combine_first(derived)
            )

    output["BatTrackingSeason"] = season
    output["BatTrackingSource"] = source_label
    has_identifier = output["player_id"].notna() | output["_name_key"].ne("")
    has_metric = output[BAT_TRACKING_METRICS].notna().any(axis=1)
    return output[has_identifier & has_metric].copy()


def _read_url_text_allowing_error_body(url: str) -> tuple[str | None, str | None]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MLB-Statcast-Dashboard/3.0",
            "Accept": "text/csv,text/plain,*/*",
        },
    )
    try:
        with urlopen(request, timeout=12) as response:
            return response.read().decode("utf-8-sig", errors="replace"), None
    except HTTPError as exc:
        try:
            body = exc.read().decode("utf-8-sig", errors="replace")
        except Exception:
            body = ""
        if body and "," in body.splitlines()[0]:
            return body, None
        return None, str(exc)
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        return None, str(exc)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_savant_bat_tracking_dataset(year: int) -> dict:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    raw_columns: list[str] = []
    seasons = [int(year)]
    if int(year) - 1 >= 2023:
        seasons.append(int(year) - 1)

    for season in seasons:
        urls = [
            "https://baseballsavant.mlb.com/leaderboard/bat-tracking?"
            + urlencode({"type": "batter", "year": season, "minSwings": 1, "csv": "true"}),
            "https://baseballsavant.mlb.com/leaderboard/bat-tracking?"
            + urlencode({"year": season, "type": "batter", "csv": "true"}),
            "https://baseballsavant.mlb.com/leaderboard/bat-tracking?"
            + urlencode({"csv": "true", "year": season}),
        ]

        season_frame = pd.DataFrame()
        for url in urls:
            body, error = _read_url_text_allowing_error_body(url)
            if body is None:
                if error:
                    errors.append(f"{season}: {error}")
                continue
            first_line = body.splitlines()[0] if body.splitlines() else ""
            if "," not in first_line:
                errors.append(f"{season}: Savant returned a page instead of CSV data.")
                continue
            try:
                raw = pd.read_csv(StringIO(body))
            except Exception as exc:
                errors.append(f"{season}: CSV parse failed ({exc})")
                continue
            raw_columns.extend([str(column) for column in raw.columns])
            season_frame = _standardize_bat_tracking_frame(
                raw,
                season,
                f"Baseball Savant {season}",
            )
            if not season_frame.empty:
                break

        if not season_frame.empty:
            frames.append(season_frame)

    empty = pd.DataFrame(
        columns=[
            "player_id", "_name_key", *BAT_TRACKING_METRICS,
            "BatTrackingSeason", "BatTrackingSource",
        ]
    )
    if not frames:
        return {
            "data": empty,
            "ok": False,
            "source": "Neutral fallback",
            "error": " | ".join(dict.fromkeys(errors)) or "No Savant bat-tracking rows were returned.",
            "columns": sorted(set(raw_columns)),
        }

    combined = pd.concat(frames, ignore_index=True)
    combined["_dedupe_key"] = np.where(
        combined["player_id"].notna(),
        "id:" + combined["player_id"].astype("Int64").astype(str),
        "name:" + combined["_name_key"],
    )
    combined = combined.sort_values("BatTrackingSeason", ascending=False)
    combined = combined.drop_duplicates("_dedupe_key", keep="first").drop(columns="_dedupe_key")
    seasons_used = sorted(
        pd.to_numeric(combined["BatTrackingSeason"], errors="coerce").dropna().astype(int).unique(),
        reverse=True,
    )
    source = "Baseball Savant " + "/".join(str(value) for value in seasons_used)
    return {
        "data": combined,
        "ok": True,
        "source": source,
        "error": None,
        "columns": sorted(set(raw_columns)),
    }


def _apply_tracking_source(
    board: pd.DataFrame,
    source: pd.DataFrame,
    source_label: str,
) -> pd.DataFrame:
    if source is None or source.empty:
        return board

    result = board.copy()
    source = source.copy()
    if "_name_key" not in source:
        source["_name_key"] = ""

    id_source = source[source["player_id"].notna()].copy()
    if not id_source.empty:
        keep = ["player_id", *BAT_TRACKING_METRICS, "BatTrackingSeason", "BatTrackingSource"]
        id_source = id_source[keep].drop_duplicates("player_id")
        merged = result.merge(id_source, on="player_id", how="left", suffixes=("", "_Incoming"))
        incoming_metric = pd.Series(False, index=merged.index)
        for column in BAT_TRACKING_METRICS:
            incoming = f"{column}_Incoming"
            incoming_metric |= merged[incoming].notna()
            merged[column] = merged[incoming].combine_first(merged[column])
            merged = merged.drop(columns=incoming)
        for column in ["BatTrackingSeason", "BatTrackingSource"]:
            incoming = f"{column}_Incoming"
            if incoming in merged:
                merged[column] = merged[incoming].combine_first(merged[column])
                merged = merged.drop(columns=incoming)
        merged.loc[incoming_metric, "BatTrackingSource"] = merged.loc[
            incoming_metric, "BatTrackingSource"
        ].fillna(source_label)
        result = merged

    result["_name_key"] = result["Player"].map(_bat_name_key)
    name_source = source[source["_name_key"].ne("")].drop_duplicates("_name_key")
    if not name_source.empty:
        keep = ["_name_key", *BAT_TRACKING_METRICS, "BatTrackingSeason", "BatTrackingSource"]
        merged = result.merge(name_source[keep], on="_name_key", how="left", suffixes=("", "_Incoming"))
        incoming_metric = pd.Series(False, index=merged.index)
        for column in BAT_TRACKING_METRICS:
            incoming = f"{column}_Incoming"
            usable = merged[column].notna()
            incoming_metric |= usable
            merged[column] = merged[incoming].combine_first(merged[column])
            merged = merged.drop(columns=incoming)
        for column in ["BatTrackingSeason", "BatTrackingSource"]:
            incoming = f"{column}_Incoming"
            if incoming in merged:
                merged[column] = merged[incoming].combine_first(merged[column])
                merged = merged.drop(columns=incoming)
        merged.loc[incoming_metric, "BatTrackingSource"] = merged.loc[
            incoming_metric, "BatTrackingSource"
        ].fillna(source_label)
        result = merged

    return result.drop(columns=["_name_key"], errors="ignore")


def load_local_park_factors() -> pd.DataFrame:
    """Load local park factors CSV if available."""
    try:
        path = Path("park_factors.csv")
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame(columns=["venue", "hr_l", "hr_r"])


def get_park_hr_factor(venue: str, side: str = "both") -> tuple[float, str]:
    table = load_local_park_factors()
    if table.empty or "venue" not in table.columns:
        return 100.0, "Neutral fallback"
    matches = table[table["venue"].str.lower().str.contains(venue.lower(), na=False)]
    if matches.empty:
        return 100.0, "Neutral fallback"
    row = matches.iloc[0]
    if side == "L":
        factor = float(row.get("hr_l", 100.0))
    elif side == "R":
        factor = float(row.get("hr_r", 100.0))
    else:
        factor = float((row.get("hr_l", 100.0) + row.get("hr_r", 100.0)) / 2.0)
    return factor, f"park_factors.csv · {venue}"


def build_hr_board(
    df: pd.DataFrame,
    pitcher_id: int,
    team: str,
    min_pa: int,
    end_date: pd.Timestamp,
    park_hr_factor_lhb: float,
    park_hr_factor_rhb: float,
    team_runs: float,
    is_away: bool,
    starter_innings: float,
    bullpen_multiplier: float,
    weather_multiplier: float,
    lineup_override: pd.DataFrame,
    lineup_edits: pd.DataFrame | None = None,
    bat_tracking_upload: pd.DataFrame | None = None,
    bat_tracking_auto: pd.DataFrame | None = None,
    active_roster: pd.DataFrame | None = None,
    include_low_sample: bool = True,
) -> pd.DataFrame:
    pitcher_profile = pitcher_hr_splits(df, pitcher_id)
    pitcher_hand = most_common(df[df["pitcher"].eq(pitcher_id)]["p_throws"], "R")
    attack_profile = pitcher_attack_profile_hr(df, pitcher_id)

    lineup_seed = lineup_override.copy() if lineup_override is not None and not lineup_override.empty else infer_recent_lineup(df, team)
    base = aggregate_hr_hitters(df, pitcher_hand, min_pa)
    base = add_roster_candidates(base, team, active_roster, lineup_seed, min_pa, include_low_sample)
    if not base.empty:
        names = lookup_names(tuple(base["player_id"].dropna().astype(int).unique().tolist()))
        if not names.empty:
            base = base.merge(names, on="player_id", how="left", suffixes=("", "_Lookup"))
            if "Player_Lookup" in base:
                base["Player"] = base.get("Player").replace("", np.nan).fillna(base["Player_Lookup"])
                base = base.drop(columns=["Player_Lookup"], errors="ignore")
        base["Player"] = base.get("Player", pd.Series(index=base.index, dtype=object)).fillna(
            "MLB ID " + base["player_id"].astype("Int64").astype(str)
        )
        if "LineupSpot" not in base.columns:
            base = base.merge(lineup_seed, on="player_id", how="left")
        else:
            seed_spots = lineup_seed.rename(columns={"LineupSpot": "LineupSpot_Seed"})
            base = base.merge(seed_spots, on="player_id", how="left")
            base["LineupSpot"] = pd.to_numeric(base["LineupSpot"], errors="coerce").combine_first(
                pd.to_numeric(base.get("LineupSpot_Seed"), errors="coerce")
            )
            base = base.drop(columns=["LineupSpot_Seed"], errors="ignore")

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

    if "LineupSpot" not in base.columns:
        base["LineupSpot"] = np.nan
    base["LineupSpot"] = pd.to_numeric(base["LineupSpot"], errors="coerce").fillna(6.0)

    recent = recent_hr_form(df, base["player_id"].tolist(), end_date)
    base = base.merge(recent, on="player_id", how="left")
    base = base.merge(pitcher_profile, on="Stand", how="left")
    base = base.merge(attack_profile, left_on="Stand", right_on="Side", how="left")

    base["Recent_HR"] = pd.to_numeric(base.get("Recent_HR"), errors="coerce").fillna(0)
    base["Recent_PA"] = pd.to_numeric(base.get("Recent_PA"), errors="coerce").fillna(0)
    base["RecentFormScore"] = percentile(
        safe_divide(base["Recent_HR"], base["Recent_PA"], 0.0) * 0.4
        + safe_divide(base.get("Recent_Barrels", 0), base["Recent_PA"], 0.0) * 0.3
        + safe_divide(base.get("Recent_xSLG", 0), base["Recent_PA"], 0.0) * 0.3
    )

    base["PitcherSideAttackScore"] = pd.to_numeric(base.get("PitcherSideAttackScore"), errors="coerce").fillna(50.0)
    base["PitcherSideAttackScore"] = base["PitcherSideAttackScore"].clip(0, 100)

    pitch_match = hr_pitch_shape_match(
        df, base["player_id"].tolist(), selected_pitcher_profile(df, pitcher_id)[0], pitcher_hand
    )
    zone_fit = hr_zone_fit(df, base["player_id"].tolist(), pitcher_id, pitcher_hand)
    bvp = bvp_hr_stats(df, base["player_id"].tolist(), pitcher_id)

    base = base.merge(pitch_match, on="player_id", how="left")
    base = base.merge(zone_fit, on="player_id", how="left")
    base = base.merge(bvp, on="player_id", how="left")

    base["PitchMatchScore"] = 50.0 + pd.to_numeric(base.get("PitchMatchRatio"), errors="coerce").fillna(0) * 30.0
    base["ZoneFitScore"] = 50.0 + pd.to_numeric(base.get("ZoneFitRatio"), errors="coerce").fillna(0) * 30.0
    base["BvPScore"] = pd.to_numeric(base.get("BvPScore"), errors="coerce").fillna(50.0)

    base["Projected_PA"] = projected_pa(base["LineupSpot"], team_runs, is_away)
    base["Projected_HR"] = base["HR_PA"] * base["Projected_PA"]
    base["Model_1plus_HR"] = 1.0 - np.exp(-base["Projected_HR"])
    base["Raw_Model_1plus_HR"] = base["Model_1plus_HR"].copy()

    # Park factor integration
    park_factor = (park_hr_factor_lhb + park_hr_factor_rhb) / 2.0 / 100.0
    base["Projected_HR"] = base["Projected_HR"] * park_factor
    base["Model_1plus_HR"] = 1.0 - np.exp(-base["Projected_HR"])

    pa_all = df[df["is_pa_end"]]
    league_pa = max(int(pa_all["pa_key"].nunique()), 1)
    league_brl_pa = float(pd.to_numeric(pa_all["is_barrel"], errors="coerce").fillna(False).sum() / league_pa)
    base["Adj_Brl_PA"] = shrink_rate(base["Barrels"], base["PA"], league_brl_pa, 100)
    base["LA_Power_Score"] = pd.to_numeric(
        base.get("LA_Power_Score", pd.Series(50.0, index=base.index)), errors="coerce"
    ).fillna(50.0).clip(0, 100)
    base["LA_Optimization_Score"] = pd.to_numeric(
        base.get("LA_Optimization_Score", pd.Series(50.0, index=base.index)), errors="coerce"
    ).fillna(base["LA_Power_Score"]).clip(0, 100)
    base["Barrel_Range_BIP"] = pd.to_numeric(
        base.get("Barrel_Range_BIP", pd.Series(0.0, index=base.index)), errors="coerce"
    ).fillna(0.0)

    base["HRScore"] = (
        percentile(base["Model_1plus_HR"]) * 0.25
        + percentile(base["LA_Power_Score"]) * 0.15
        + percentile(base["PitchMatchScore"]) * 0.18
        + percentile(base["ZoneFitScore"]) * 0.14
        + percentile(base["PitcherSideAttackScore"]) * 0.12
        + percentile(base["BvPScore"]) * 0.08
        + percentile(base["RecentFormScore"]) * 0.08
        + percentile(base["Adj_Brl_PA"]) * 0.05
    )
    base["HRScore"] = base["HRScore"].clip(0, 100)

    base["Confidence"] = (
        0.35 * (base["PA"] / 120.0).clip(0, 1.0) * 100
        + 0.25 * (base.get("Platoon_PA", 0) / 40.0).clip(0, 1.0) * 100
        + 0.20 * (base.get("MatchSample", 0) / 80.0).clip(0, 1.0) * 100
        + 0.20 * (base.get("BvP_PA", 0) / 25.0).clip(0, 1.0) * 100
    ).clip(0, 100)

    base["Confidence_Level"] = np.where(
        base["Confidence"] >= 72, "High", np.where(base["Confidence"] >= 48, "Medium", "Low")
    )

    base = base.sort_values("HRScore", ascending=False).reset_index(drop=True)
    base.insert(0, "Rank", np.arange(1, len(base) + 1))
    return base


def render_leader_cards(board: pd.DataFrame, prob_column: str, score_column: str, label: str) -> None:
    top = board.iloc[0]
    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 20px 0;">
            <div style="background: linear-gradient(145deg, #f0f9ff, #e0f2fe); border: 1px solid #bae6fd; border-radius: 16px; padding: 20px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #0369a1; letter-spacing: 0.05em;">TOP HR PROJECTION</div>
                <div style="font-size: 1.35rem; font-weight: 900; color: #0c4a6e; margin: 8px 0;">{top['Player']}</div>
                <div style="font-size: 2.1rem; font-weight: 950; color: #0369a1;">{float(top[prob_column]):.1%} {label}</div>
                <div style="color: #64748b; font-size: 0.9rem;">{float(top[score_column]):.1f} HR Score</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_clean_css() -> None:
    st.markdown(
        """
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
        """,
        unsafe_allow_html=True,
    )


def matchup_selector(pitcher_summary: pd.DataFrame, available_teams: list[str], key_prefix: str) -> dict:
    st.sidebar.subheader("Matchup")
    pitcher_options = pitcher_summary["Display"].tolist()
    pitcher_display = st.sidebar.selectbox(
        "Starting pitcher", pitcher_options, index=0, key=f"{key_prefix}_pitcher"
    )
    pitcher_row = pitcher_summary[pitcher_summary["Display"].eq(pitcher_display)].iloc[0]
    selected_team = st.sidebar.selectbox(
        "Batting team", available_teams, index=0, key=f"{key_prefix}_team"
    )
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
    result["Raw_Model_1plus_HR"] = result["Model_1plus_HR"].copy()
    result["Model_1plus_HR"] = (intercept + slope * result["Model_1plus_HR"]).clip(0.0, 1.0)
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
    odds_source_mode = st.selectbox(
        "Line source", ["PrizePicks", "FanDuel", "DraftKings", "Pinnacle", "Consensus sportsbooks"], index=0,
        key=f"{widget_prefix}_odds_source"
    )
    game_label = f"{matchup.get('away_abbr', 'Selected team')} @ {matchup.get('home_abbr', 'selected pitcher')}"
    selected_games = st.multiselect(
        "Games to fetch odds for", [game_label],
        default=[game_label],
        key=f"{widget_prefix}_odds_games"
    )
    fetch_odds = st.button("Fetch odds", key=f"{widget_prefix}_fetch_odds")
    loaded_quotes = pd.DataFrame()
    if fetch_odds and api_key:
        st.info("Odds fetching not fully implemented in this rewrite.")
    return board, loaded_quotes, odds_source_mode


def render_board_header(matchup: dict, batting_team: str, pitcher_name: str, title: str) -> None:
    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, #1e3a8a, #3b82f6); color: white; padding: 20px; border-radius: 16px; margin-bottom: 24px;">
            <div style="font-size: 1.1rem; opacity: 0.9;">{matchup.get('away_abbr', '')} @ {matchup.get('home_abbr', '')}</div>
            <div style="font-size: 1.75rem; font-weight: 900;">{batting_team} vs {pitcher_name}</div>
            <div style="opacity: 0.85;">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_binary_backtest_tab(*args, **kwargs) -> None:
    st.info("Backtest tab placeholder. Implement as needed.")
    pass


inject_clean_css()

st.markdown('<div class="app-kicker">💥 Statcast power lab</div>', unsafe_allow_html=True)
st.title("Advanced MLB Home Run Dashboard")
st.caption(
    "A dedicated home-run model with barrels per PA, pulled air contact, xSLG/xISO, "
    "upper-end exit velocity, pitch-shape fit, zone fit, pitcher vulnerability and environment."
)

with st.sidebar:
    st.header("Statcast sample")
    yesterday = date.today() - timedelta(days=1)
    end_date_value = st.date_input(
        "Stats through", value=yesterday, max_value=yesterday, key="clean_hr_end_date"
    )
    lookback_days = st.slider(
        "Lookback days", 21, 120, 60, 7, key="clean_hr_lookback"
    )
    min_pa = st.slider(
        "Minimum hitter PA", 10, 100, 30, 5, key="clean_hr_min_pa"
    )
    include_low_sample = st.checkbox(
        "Include active-roster hitters below the PA threshold",
        value=True,
        key="clean_hr_include_low_sample",
    )
    refresh = st.button(
        "Load / refresh Statcast", type="primary", key="clean_hr_refresh"
    )

start_date_value = end_date_value - timedelta(days=lookback_days - 1)
if refresh or "clean_hr_statcast_data" not in st.session_state:
    with st.spinner(
        f"Loading Statcast from {start_date_value} through {end_date_value}..."
    ):
        raw = load_statcast(
            start_date_value.strftime("%Y-%m-%d"),
            end_date_value.strftime("%Y-%m-%d"),
        )
        st.session_state["clean_hr_statcast_data"] = prepare_data(raw)
        st.session_state["clean_hr_loaded_dates"] = (start_date_value, end_date_value)

df = st.session_state.get("clean_hr_statcast_data", pd.DataFrame())
if df.empty:
    st.warning("No Statcast data was returned.")
    st.stop()

loaded_start, loaded_end = st.session_state["clean_hr_loaded_dates"]
st.caption(
    f"Using {len(df):,} pitches from {loaded_start} through {loaded_end}."
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

available_teams = sorted(df["batter_team"].dropna().astype(str).unique().tolist())
matchup = matchup_selector(pitcher_summary, available_teams, key_prefix="clean_hr")

selected_pitcher = int(matchup["pitcher_id"])
selected_display = str(matchup["pitcher_display"])
selected_team = str(matchup["batting_team"])
home_away = str(matchup["home_away"])

active_roster = pd.DataFrame()
attack_profile = pitcher_attack_profile_hr(df, selected_pitcher)
render_pitcher_attack_panel(attack_profile, matchup["pitcher_name"], "home runs")

preview_board = aggregate_hr_hitters(df, most_common(df[df["pitcher"].eq(selected_pitcher)]["p_throws"], "R"), min_pa)
preview_board = add_roster_candidates(preview_board, selected_team, active_roster, infer_recent_lineup(df, selected_team), min_pa, include_low_sample)

rankings = build_hr_board(
    df=df,
    pitcher_id=selected_pitcher,
    team=selected_team,
    min_pa=min_pa,
    end_date=pd.Timestamp(loaded_end),
    park_hr_factor_lhb=100.0,
    park_hr_factor_rhb=100.0,
    team_runs=4.5,
    is_away=home_away == "Away",
    starter_innings=5.5,
    bullpen_multiplier=1.0,
    weather_multiplier=1.0,
    lineup_override=pd.DataFrame(),
    lineup_edits=None,
    bat_tracking_upload=None,
    bat_tracking_auto=None,
    active_roster=active_roster,
    include_low_sample=include_low_sample,
)

if rankings.empty:
    st.warning("No hitters available.")
    st.stop()

render_board_header(matchup, selected_team, matchup["pitcher_name"], "ADVANCED HOME RUN BOARD")
render_leader_cards(rankings, "Model_1plus_HR", "HRScore", "model 1+ HR")

quick_tab, power_tab, matchup_tab, pitch_type_tab, tracking_tab, pitcher_tab, slate_tab, backtest_tab, notes_tab = st.tabs(
    [
        "Quick board", "Power profile", "Matchup detail", "Batter vs pitch type",
        "Bat tracking", "Pitcher profile", "Slate Top 10 & pairings", "Backtest & calibration", "Model notes"
    ]
)

with quick_tab:
    st.dataframe(rankings.head(15), use_container_width=True)

with power_tab:
    st.subheader("Power and launch-angle profile")
    power_columns = [
        "Rank", "Player", "Team", "LineupSpot", "Model_1plus_HR", "HRScore",
        "Projected_HR", "HR_PA", "Brl_PA", "Adj_Brl_PA", "Brl_BIP",
        "Barrel_Range_BIP", "Dynamic_Brl_BIP", "Avg_EV", "EV90", "Max_EV",
        "Avg_LA", "LA_Power_Score", "LA_Optimization_Score", "FB_Pct", "PullAir_Air",
        "PitchMatchScore", "ZoneFitScore", "PitcherSideAttackScore", "RecentFormScore",
    ]
    available_power_columns = [column for column in power_columns if column in rankings.columns]
    power_table = rankings[available_power_columns].copy()
    st.dataframe(power_table, use_container_width=True)

    with st.expander("Launch Angle Distribution", expanded=False):
        bbe_angles = df[df["is_bbe"] & df["launch_angle"].notna()].copy()
        if bbe_angles.empty:
            st.info("No batted-ball launch-angle data is available for this sample.")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            non_hr = pd.to_numeric(bbe_angles.loc[~bbe_angles["is_hr"], "launch_angle"], errors="coerce").dropna()
            hr = pd.to_numeric(bbe_angles.loc[bbe_angles["is_hr"], "launch_angle"], errors="coerce").dropna()
            ax.hist([non_hr, hr], bins=50, label=["Non-HR BBE", "HR"], alpha=0.75)
            ax.axvspan(20, 35, alpha=0.2, label="Optimal HR Range")
            ax.set_title("Launch Angle Distribution")
            ax.set_xlabel("Launch angle")
            ax.set_ylabel("Batted balls")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

with notes_tab:
    st.markdown("Model rewritten with improved weighting, shrinkage and matchup factors. Launch-angle power features are active in the HR score.")

print("Home Run Dashboard updated successfully.")