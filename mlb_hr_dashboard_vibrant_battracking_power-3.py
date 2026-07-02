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

    # Approximate pulled-air classification using Savant hit-coordinate orientation.
    # It is useful as a relative feature but should not be treated as an official Savant pull label.
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
            Fly_Balls=("is_fly_ball", "sum"),
            Air_Balls=("is_air", "sum"),
            Pull_Air=("is_pull_air", "sum"),
            Avg_EV=("launch_speed", "mean"),
            EV90=("launch_speed", quantile_90),
            Max_EV=("launch_speed", "max"),
            Avg_LA=("launch_angle", "mean"),
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
    """Grade how attackable the selected pitcher has been to LHB and RHB for home runs."""
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
    """Show each displayed hitter's results against the starter's pitch types.

    The split is against the same pitch type from pitchers with the selected
    starter's handedness. Small samples are shrunk toward the pitch-type league
    baseline. This surfaces the data already summarized by Pitch Match without
    double-counting it in the model.
    """
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
    # Last-resort token match handles small Savant header changes.
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

    # Derive contact rates from counts when Savant supplies counts instead of rates.
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
        # Savant's CSV link can occasionally return a non-200 code with a usable body.
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
    """Best-effort automatic Baseball Savant bat-tracking download.

    Current-season data are preferred. The prior season is used as a fallback
    for hitters who do not yet have current-season tracking data.
    """
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

    # MLBAM ID is the preferred join key.
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

    # Fill unmatched names as a fallback for CSVs that omit MLBAM IDs.
    result["_name_key"] = result["Player"].map(_bat_name_key)
    name_source = source[source["_name_key"].ne("")].drop_duplicates("_name_key")
    if not name_source.empty:
        keep = ["_name_key", *BAT_TRACKING_METRICS, "BatTrackingSeason", "BatTrackingSource"]
        merged = result.merge(name_source[keep], on="_name_key", how="left", suffixes=("", "_Incoming"))
        incoming_metric = pd.Series(False, index=merged.index)
        for column in BAT_TRACKING_METRICS:
            incoming = f"{column}_Incoming"
            usable = merged[column].isna() & merged[incoming].notna()
            incoming_metric |= usable
            merged.loc[usable, column] = merged.loc[usable, incoming]
            merged = merged.drop(columns=incoming)
        for column in ["BatTrackingSeason", "BatTrackingSource"]:
            incoming = f"{column}_Incoming"
            usable = merged[column].isna() & merged[incoming].notna()
            merged.loc[usable, column] = merged.loc[usable, incoming]
            merged = merged.drop(columns=incoming)
        merged.loc[incoming_metric, "BatTrackingSource"] = merged.loc[
            incoming_metric, "BatTrackingSource"
        ].fillna(source_label)
        result = merged

    return result.drop(columns=["_name_key"], errors="ignore")


def _calculate_bat_tracking_score(board: pd.DataFrame) -> pd.DataFrame:
    result = board.copy()
    result["BatTrackingMetricCount"] = result[BAT_TRACKING_METRICS].notna().sum(axis=1)
    result["BatTrackingAvailable"] = result["BatTrackingMetricCount"].gt(0)

    # Transparent absolute scales keep a one-player or small-team sample from
    # collapsing to the neutral 50th percentile.
    component_scores = {
        "Bat_Speed": ((result["Bat_Speed"] - 65.0) / 15.0 * 100.0).clip(0, 100),
        "Fast_Swing_Rate": (result["Fast_Swing_Rate"] / 0.60 * 100.0).clip(0, 100),
        "Blast_Contact_Rate": (result["Blast_Contact_Rate"] / 0.30 * 100.0).clip(0, 100),
        "Squared_Up_Contact_Rate": (
            (result["Squared_Up_Contact_Rate"] - 0.15) / 0.35 * 100.0
        ).clip(0, 100),
        "Attack_Angle": (100.0 - (result["Attack_Angle"] - 12.5).abs() * 6.0).clip(0, 100),
    }
    weights = {
        "Bat_Speed": 0.30,
        "Fast_Swing_Rate": 0.20,
        "Blast_Contact_Rate": 0.25,
        "Squared_Up_Contact_Rate": 0.15,
        "Attack_Angle": 0.10,
    }
    numerator = pd.Series(0.0, index=result.index)
    denominator = pd.Series(0.0, index=result.index)
    for column, score in component_scores.items():
        available = result[column].notna()
        numerator += score.fillna(0.0) * weights[column]
        denominator += available.astype(float) * weights[column]

    result["BatTrackingScore"] = safe_divide(numerator, denominator, default=50.0).clip(0, 100)
    result.loc[~result["BatTrackingAvailable"], "BatTrackingScore"] = 50.0
    result["BatTrackingSource"] = result["BatTrackingSource"].fillna("No tracking data")
    return result


def parse_bat_tracking_upload(
    uploaded_file,
    board: pd.DataFrame,
    automatic_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply automatic Savant data, then let an uploaded CSV override it."""
    board = board.copy()
    for column in BAT_TRACKING_METRICS:
        board[column] = np.nan
    board["BatTrackingSeason"] = np.nan
    board["BatTrackingSource"] = np.nan

    if automatic_data is not None and not automatic_data.empty:
        board = _apply_tracking_source(board, automatic_data, "Automatic Savant")

    if uploaded_file is not None:
        try:
            supplemental = pd.read_csv(uploaded_file)
        except Exception as error:
            st.warning(f"Could not read bat-tracking CSV: {error}")
            return _calculate_bat_tracking_score(board)

        manual = _standardize_bat_tracking_frame(
            supplemental,
            None,
            "Manual CSV override",
        )
        if manual.empty:
            st.warning(
                "The uploaded bat-tracking CSV did not contain a recognized player identifier "
                "and bat-tracking metric. The automatic data remain in use."
            )
        else:
            board = _apply_tracking_source(board, manual, "Manual CSV override")

    return _calculate_bat_tracking_score(board)


def load_stadium_weather_metadata(venue: str) -> dict:
    """Read fallback coordinates, approximate outfield bearing and roof type."""
    paths = [
        Path(__file__).resolve().parent / "stadium_weather.csv",
        Path.cwd() / "stadium_weather.csv",
    ]
    csv_path = next((path for path in paths if path.exists()), None)
    if csv_path is None:
        return {
            "ok": False,
            "latitude": None,
            "longitude": None,
            "outfield_bearing": None,
            "roof_type": "open",
            "error": "stadium_weather.csv was not found beside the dashboard file.",
        }
    try:
        table = pd.read_csv(csv_path)
    except Exception as exc:
        return {
            "ok": False,
            "latitude": None,
            "longitude": None,
            "outfield_bearing": None,
            "roof_type": "open",
            "error": f"stadium_weather.csv could not be read: {type(exc).__name__}: {exc}",
        }

    required = {"venue", "latitude", "longitude", "outfield_bearing", "roof_type"}
    if not required.issubset(table.columns):
        return {
            "ok": False,
            "latitude": None,
            "longitude": None,
            "outfield_bearing": None,
            "roof_type": "open",
            "error": "stadium_weather.csv is missing required columns.",
        }

    matches = table[table["venue"].apply(lambda value: venue_names_match(value, venue))]
    if matches.empty:
        return {
            "ok": False,
            "latitude": None,
            "longitude": None,
            "outfield_bearing": None,
            "roof_type": "open",
            "error": f"No stadium_weather.csv row matched {venue}.",
        }
    row = matches.iloc[0]
    return {
        "ok": True,
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "outfield_bearing": float(row["outfield_bearing"]),
        "roof_type": str(row["roof_type"]).lower().strip(),
        "source": "stadium_weather.csv",
        "error": None,
    }


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_mlb_venue_coordinates(venue_id: object) -> dict:
    """Use MLB's venue record for coordinates; return an empty result on failure."""
    numeric = pd.to_numeric(pd.Series([venue_id]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return {"ok": False, "latitude": None, "longitude": None, "error": "No venue ID."}
    url = f"https://statsapi.mlb.com/api/v1/venues/{int(numeric)}"
    request = Request(url, headers={"User-Agent": "MLB-Statcast-Dashboard/3.0"})
    try:
        with urlopen(request, timeout=12) as response:
            payload = json.load(response)
        venues = payload.get("venues", [])
        if not venues:
            raise ValueError("MLB venue response was empty")
        coordinates = venues[0].get("location", {}).get("defaultCoordinates", {})
        latitude = pd.to_numeric(pd.Series([coordinates.get("latitude")]), errors="coerce").iloc[0]
        longitude = pd.to_numeric(pd.Series([coordinates.get("longitude")]), errors="coerce").iloc[0]
        if pd.isna(latitude) or pd.isna(longitude):
            raise ValueError("MLB venue response did not include coordinates")
        return {
            "ok": True,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "source": "MLB venue coordinates",
            "error": None,
        }
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return {
            "ok": False,
            "latitude": None,
            "longitude": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def classify_stadium_wind(wind_from_degrees: float, outfield_bearing: float | None) -> tuple[str, float | None]:
    """Convert meteorological wind-from degrees to Out/In/Cross at the stadium."""
    if pd.isna(wind_from_degrees) or outfield_bearing is None or pd.isna(outfield_bearing):
        return "Cross/Calm", None
    wind_toward = (float(wind_from_degrees) + 180.0) % 360.0
    difference = abs((wind_toward - float(outfield_bearing) + 180.0) % 360.0 - 180.0)
    if difference <= 45.0:
        return "Out", difference
    if difference >= 135.0:
        return "In", difference
    return "Cross/Calm", difference


def weather_code_text(code: object) -> str:
    numeric = pd.to_numeric(pd.Series([code]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "Forecast"
    code = int(numeric)
    if code == 0:
        return "Clear"
    if code in {1, 2, 3}:
        return "Partly cloudy" if code < 3 else "Overcast"
    if code in {45, 48}:
        return "Fog"
    if code in {51, 53, 55, 56, 57}:
        return "Drizzle"
    if code in {61, 63, 65, 66, 67, 80, 81, 82}:
        return "Rain"
    if code in {71, 73, 75, 77, 85, 86}:
        return "Snow"
    if code in {95, 96, 99}:
        return "Thunderstorms"
    return "Forecast"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_open_meteo_game_weather(
    latitude: float,
    longitude: float,
    game_datetime_utc: str,
    outfield_bearing: float | None,
) -> dict:
    """Fetch the hourly forecast closest to scheduled first pitch."""
    target = pd.to_datetime(game_datetime_utc, utc=True, errors="coerce")
    if pd.isna(target):
        return {"ok": False, "error": "The selected game did not have a valid first-pitch time."}

    target_date = target.strftime("%Y-%m-%d")
    params = {
        "latitude": round(float(latitude), 5),
        "longitude": round(float(longitude), 5),
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation_probability,"
            "pressure_msl,wind_speed_10m,wind_direction_10m,wind_gusts_10m,weather_code"
        ),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "GMT",
        "start_date": target_date,
        "end_date": target_date,
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
    request = Request(
        url,
        headers={
            "User-Agent": "MLB-Statcast-Dashboard/3.0",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.load(response)
        hourly = pd.DataFrame(payload.get("hourly", {}))
        if hourly.empty or "time" not in hourly.columns:
            raise ValueError("Open-Meteo returned no hourly forecast rows")
        hourly["forecast_time"] = pd.to_datetime(hourly["time"], utc=True, errors="coerce")
        hourly = hourly.dropna(subset=["forecast_time"]).copy()
        if hourly.empty:
            raise ValueError("Open-Meteo returned invalid hourly timestamps")
        nearest_index = (hourly["forecast_time"] - target).abs().idxmin()
        row = hourly.loc[nearest_index]
        wind_from = float(pd.to_numeric(pd.Series([row.get("wind_direction_10m")]), errors="coerce").iloc[0])
        stadium_wind, angle_difference = classify_stadium_wind(wind_from, outfield_bearing)
        return {
            "ok": True,
            "temperature_f": float(row.get("temperature_2m")),
            "humidity_pct": float(row.get("relative_humidity_2m")),
            "precip_probability": float(row.get("precipitation_probability", 0.0)),
            "pressure_hpa": float(row.get("pressure_msl", 1013.25)),
            "wind_mph": float(row.get("wind_speed_10m", 0.0)),
            "wind_gust_mph": float(row.get("wind_gusts_10m", 0.0)),
            "wind_from_degrees": wind_from,
            "wind_direction": stadium_wind,
            "wind_angle_difference": angle_difference,
            "weather_code": row.get("weather_code"),
            "condition": weather_code_text(row.get("weather_code")),
            "forecast_time_utc": row["forecast_time"].strftime("%Y-%m-%d %H:%M UTC"),
            "source": "Open-Meteo hourly forecast",
            "error": None,
        }
    except (HTTPError, URLError, TimeoutError, ValueError, OSError, TypeError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def automatic_game_weather(matchup: dict) -> dict:
    metadata = load_stadium_weather_metadata(str(matchup.get("venue", "")))
    mlb_location = fetch_mlb_venue_coordinates(matchup.get("venue_id"))
    latitude = mlb_location.get("latitude") if mlb_location.get("ok") else metadata.get("latitude")
    longitude = mlb_location.get("longitude") if mlb_location.get("ok") else metadata.get("longitude")
    if latitude is None or longitude is None:
        return {
            "ok": False,
            "metadata": metadata,
            "error": metadata.get("error") or mlb_location.get("error") or "No stadium coordinates were available.",
        }
    result = fetch_open_meteo_game_weather(
        float(latitude),
        float(longitude),
        str(matchup.get("game_datetime_utc") or ""),
        metadata.get("outfield_bearing"),
    )
    result["metadata"] = metadata
    result["coordinate_source"] = mlb_location.get("source") if mlb_location.get("ok") else metadata.get("source")
    return result

def weather_carry_multiplier(
    temperature_f: float,
    humidity_pct: float,
    wind_mph: float,
    wind_direction: str,
    pressure_hpa: float,
    manual_multiplier: float,
    enclosed: bool = False,
) -> float:
    if enclosed:
        return float(np.clip(manual_multiplier, 0.80, 1.20))
    temperature_factor = 1.0 + (temperature_f - 70.0) * 0.0025
    humidity_factor = 1.0 + (humidity_pct - 50.0) * 0.0003
    pressure_factor = 1.0 + (1013.25 - pressure_hpa) * 0.00025
    if wind_direction == "Out":
        wind_factor = 1.0 + wind_mph * 0.008
    elif wind_direction == "In":
        wind_factor = 1.0 - wind_mph * 0.008
    else:
        wind_factor = 1.0
    return float(
        np.clip(
            temperature_factor
            * humidity_factor
            * pressure_factor
            * wind_factor
            * manual_multiplier,
            0.72,
            1.32,
        )
    )


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
    lineup_override: pd.DataFrame | None,
    lineup_edits: pd.DataFrame | None,
    bat_tracking_upload,
    bat_tracking_auto: pd.DataFrame | None,
    active_roster: pd.DataFrame | None,
    include_low_sample: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile, pitcher_hand = selected_pitcher_profile(df, pitcher_id)
    if not pitcher_hand:
        return pd.DataFrame(), profile

    recent_lineup = _clean_lineup_seed(lineup_override)
    if recent_lineup.empty:
        recent_lineup = infer_recent_lineup(df, team)
    board = aggregate_hr_hitters(df, pitcher_hand, 0)
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
        "PA", "HR", "Strikeouts", "xSLG_Total", "xISO_Total", "Pitches", "Swings",
        "Contacts", "Whiffs", "BBE", "Barrels", "Hard_Hits", "Sweet_Spots",
        "Fly_Balls", "Air_Balls", "Pull_Air", "Platoon_PA", "Platoon_HR",
        "Platoon_xSLG", "Platoon_xISO", "Platoon_Barrels",
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

    recent = recent_hr_form(df, board["player_id"], end_date)
    board = board.merge(recent, on="player_id", how="left")
    for column in ["Recent_PA", "Recent_HR", "Recent_Barrels", "Recent_xSLG", "Recent_xISO"]:
        board[column] = pd.to_numeric(board[column], errors="coerce").fillna(0)

    pitcher_splits = pitcher_hr_splits(df, pitcher_id)
    board = board.merge(
        pitcher_splits, left_on="EffectiveStand", right_on="Stand", how="left", suffixes=("", "_Pitcher")
    )
    for column in ["Pitcher_PA", "Pitcher_HR", "Pitcher_Barrels", "Pitcher_xSLG", "Pitcher_xISO"]:
        board[column] = pd.to_numeric(board[column], errors="coerce").fillna(0)

    side_attack = pitcher_attack_profile_hr(df, pitcher_id)
    board = board.merge(
        side_attack[["Side", "PitcherSideAttackScore", "PitcherSideRead", "AttackConfidence"]],
        left_on="EffectiveStand", right_on="Side", how="left"
    )
    board["PitcherSideAttackScore"] = board["PitcherSideAttackScore"].fillna(50.0)
    board["PitcherSideRead"] = board["PitcherSideRead"].fillna("Neutral")

    board = board.merge(
        hr_pitch_shape_match(df, board["player_id"], profile, pitcher_hand),
        on="player_id",
        how="left",
    )
    board = board.merge(
        hr_zone_fit(df, board["player_id"], pitcher_id, pitcher_hand),
        on="player_id",
        how="left",
    )
    board = board.merge(
        bvp_hr_stats(df, board["player_id"], pitcher_id),
        on="player_id",
        how="left",
    )
    board["PitchMatchRatio"] = board["PitchMatchRatio"].fillna(1.0)
    board["ZoneFitRatio"] = board["ZoneFitRatio"].fillna(1.0)
    board["MatchSample"] = board["MatchSample"].fillna(0)

    board = board.merge(recent_lineup, on="player_id", how="left")
    if lineup_edits is not None and not lineup_edits.empty:
        edits = lineup_edits[["player_id", "Selected", "LineupSpot"]].copy()
        board = board.drop(columns=["LineupSpot"], errors="ignore").merge(edits, on="player_id", how="left")
        board = board[board["Selected"].fillna(True)].copy()
    else:
        board["Selected"] = True

    board = parse_bat_tracking_upload(bat_tracking_upload, board, bat_tracking_auto)

    pa_all = df[df["is_pa_end"]]
    league_pa = max(int(pa_all["pa_key"].nunique()), 1)
    league_hr_rate = float(pa_all["is_hr"].sum() / league_pa)
    league_barrel_pa = float(pa_all["is_barrel"].sum() / league_pa)
    league_xslg_pa = float(pa_all["xslg_value"].sum() / league_pa)
    league_xiso_pa = float(pa_all["xiso_value"].sum() / league_pa)

    board["Adj_HR_PA"] = shrink_rate(board["HR"], board["PA"], league_hr_rate, 180)
    board["Adj_Brl_PA"] = shrink_rate(board["Barrels"], board["PA"], league_barrel_pa, 140)
    board["Adj_xSLG_PA"] = shrink_rate(board["xSLG_Total"], board["PA"], league_xslg_pa, 140)
    board["Adj_xISO_PA"] = shrink_rate(board["xISO_Total"], board["PA"], league_xiso_pa, 140)
    board["Adj_Platoon_HR_PA"] = shrink_rate(
        board["Platoon_HR"], board["Platoon_PA"], league_hr_rate, 90
    )
    board["Adj_Platoon_Brl_PA"] = shrink_rate(
        board["Platoon_Barrels"], board["Platoon_PA"], league_barrel_pa, 90
    )
    board["Pitcher_HR_PA"] = shrink_rate(
        board["Pitcher_HR"], board["Pitcher_PA"], league_hr_rate, 220
    )
    board["Pitcher_Brl_PA"] = shrink_rate(
        board["Pitcher_Barrels"], board["Pitcher_PA"], league_barrel_pa, 180
    )
    board["Pitcher_xSLG_PA"] = shrink_rate(
        board["Pitcher_xSLG"], board["Pitcher_PA"], league_xslg_pa, 180
    )
    board["Recent_HR_PA"] = shrink_rate(
        board["Recent_HR"], board["Recent_PA"], league_hr_rate, 45
    )
    board["Recent_Brl_PA"] = shrink_rate(
        board["Recent_Barrels"], board["Recent_PA"], league_barrel_pa, 40
    )
    board["Recent_xSLG_PA"] = shrink_rate(
        board["Recent_xSLG"], board["Recent_PA"], league_xslg_pa, 40
    )

    barrel_component = league_hr_rate * np.power(
        (board["Adj_Brl_PA"] / max(league_barrel_pa, 0.0001)).clip(0.35, 2.8), 0.70
    )
    xslg_component = league_hr_rate * np.power(
        (board["Adj_xSLG_PA"] / max(league_xslg_pa, 0.0001)).clip(0.45, 2.3), 0.55
    )
    xiso_component = league_hr_rate * np.power(
        (board["Adj_xISO_PA"] / max(league_xiso_pa, 0.0001)).clip(0.45, 2.5), 0.60
    )
    pitcher_component = (
        board["Pitcher_HR_PA"] * 0.45
        + league_hr_rate * np.power(
            (board["Pitcher_Brl_PA"] / max(league_barrel_pa, 0.0001)).clip(0.45, 2.5), 0.55
        ) * 0.35
        + league_hr_rate * np.power(
            (board["Pitcher_xSLG_PA"] / max(league_xslg_pa, 0.0001)).clip(0.50, 2.2), 0.45
        ) * 0.20
    )

    hitter_base = (
        board["Adj_HR_PA"] * 0.20
        + barrel_component * 0.22
        + xslg_component * 0.12
        + xiso_component * 0.10
        + board["Adj_Platoon_HR_PA"] * 0.08
        + league_hr_rate
        * np.power((board["Adj_Platoon_Brl_PA"] / max(league_barrel_pa, 0.0001)).clip(0.45, 2.5), 0.65)
        * 0.08
        + pitcher_component * 0.10
        + league_hr_rate * board["PitchMatchRatio"] * 0.06
        + league_hr_rate * board["ZoneFitRatio"] * 0.04
    )
    recent_component = (
        board["Recent_HR_PA"] * 0.30
        + league_hr_rate
        * np.power((board["Recent_Brl_PA"] / max(league_barrel_pa, 0.0001)).clip(0.45, 2.8), 0.70)
        * 0.40
        + league_hr_rate
        * np.power((board["Recent_xSLG_PA"] / max(league_xslg_pa, 0.0001)).clip(0.45, 2.4), 0.55)
        * 0.30
    )
    starter_rate = 0.88 * hitter_base + 0.12 * recent_component
    bat_multiplier = (1.0 + (board["BatTrackingScore"] - 50.0) / 500.0).clip(0.90, 1.10)
    starter_rate *= bat_multiplier

    starter_share = float(np.clip(starter_innings / 9.0, 0.35, 0.78))
    bullpen_rate = league_hr_rate * float(bullpen_multiplier)
    game_per_pa = starter_share * starter_rate + (1 - starter_share) * bullpen_rate
    board["ParkFactor"] = np.where(
        board["EffectiveStand"].eq("L"), float(park_hr_factor_lhb), float(park_hr_factor_rhb)
    )
    board["PrePark_HR_Per_PA"] = game_per_pa
    game_per_pa = game_per_pa * board["ParkFactor"] / 100.0
    game_per_pa *= float(weather_multiplier)
    board["Model_HR_Per_PA"] = game_per_pa.clip(0.001, 0.18)
    board["Projected_PA"] = projected_pa(board["LineupSpot"], team_runs, is_away)
    board["Model_1plus_HR"] = 1 - (1 - board["Model_HR_Per_PA"]) ** board["Projected_PA"]

    board["PitchMatchScore"] = percentile(board["PitchMatchRatio"])
    board["ZoneFitScore"] = percentile(board["ZoneFitRatio"])
    board["PitcherPowerScore"] = percentile(pitcher_component)
    board["RecentFormScore"] = percentile(recent_component)
    board["EVPowerScore"] = 0.45 * percentile(board["EV90"]) + 0.55 * percentile(board["Max_EV"])

    board["HRScore"] = (
        percentile(board["Adj_Brl_PA"]) * 0.20
        + percentile(board["PullAir_BIP"]) * 0.12
        + percentile(board["Adj_xSLG_PA"]) * 0.10
        + percentile(board["Adj_xISO_PA"]) * 0.08
        + board["EVPowerScore"] * 0.10
        + board["PitchMatchScore"] * 0.15
        + board["PitcherPowerScore"] * 0.10
        + board["ZoneFitScore"] * 0.06
        + board["RecentFormScore"] * 0.05
        + board["BatTrackingScore"] * 0.03
        + board["BvPScore"] * 0.01
    )

    sample_conf = 100 * (1 - np.exp(-board["PA"] / 180.0))
    bbe_conf = 100 * (1 - np.exp(-board["BBE"].fillna(0) / 110.0))
    matchup_conf = 100 * (1 - np.exp(-board["MatchSample"] / 75.0))
    board["Confidence"] = 0.50 * sample_conf + 0.30 * bbe_conf + 0.20 * matchup_conf
    board["Confidence_Level"] = pd.cut(
        board["Confidence"], bins=[-np.inf, 45, 70, np.inf], labels=["Low", "Medium", "High"]
    ).astype(str)

    board = board.sort_values(["Model_1plus_HR", "HRScore"], ascending=False).reset_index(drop=True)
    board.insert(0, "Rank", np.arange(1, len(board) + 1))
    return board, profile




# -----------------------------------------------------------------------------
# Clean slate/matchup interface helpers
# -----------------------------------------------------------------------------
import json
from html import escape
from io import StringIO
from pathlib import Path
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
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .stApp {
            background: #ffffff !important;
            color: var(--ink) !important;
        }
        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.96) !important;
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
                    "game_datetime_utc": raw_time,
                    "venue": game.get("venue", {}).get("name", "Venue TBD"),
                    "venue_id": game.get("venue", {}).get("id"),
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
        "venue_id": None,
        "time_et": "",
        "game_datetime_utc": None,
        "status": "Manual",
        "away_abbr": selected_team if side == "Away" else str(row["Pitcher_Team"]),
        "home_abbr": selected_team if side == "Home" else str(row["Pitcher_Team"]),
        "game_pk": None,
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
                "venue_id": game.get("venue_id"),
                "time_et": game["time_et"],
                "game_datetime_utc": game.get("game_datetime_utc"),
                "status": game["status"],
                "away_abbr": game["away_abbr"],
                "home_abbr": game["home_abbr"],
                "away_id": game["away_id"],
                "home_id": game["home_id"],
                "game_pk": game["game_pk"],
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
            st.markdown(
                f"""
                <div class="leader-card">
                    <div class="leader-rank">RANK {int(row['Rank'])}</div>
                    <div class="leader-name">{row['Player']}</div>
                    <div class="leader-prob">{row[probability_column]:.1%}</div>
                    <div class="leader-sub">{probability_label} · {score_column.replace('Score', ' score')} {row[score_column]:.1f}</div>
                    <div class="leader-sub">{lineup_text} · {confidence_text} confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )



# -----------------------------------------------------------------------------
# Binary outcome backtesting and probability calibration
# -----------------------------------------------------------------------------
BINARY_MODEL_VERSION = "hr-backtest-calibration-v1"
BINARY_TARGET_LABEL = "1+ HR"
BINARY_TARGET_SLUG = "hr"
BINARY_PROBABILITY_COLUMN = "Model_1plus_HR"
BINARY_RAW_PROBABILITY_COLUMN = "Raw_Model_1plus_HR"
BINARY_SCORE_COLUMN = "HRScore"
BINARY_EVENT_KIND = "hr"
BINARY_MIN_CALIBRATION_ROWS = 500


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
    snapshot["GamePK"] = matchup.get("game_pk")
    snapshot["PlayerID"] = pd.to_numeric(snapshot.get("player_id"), errors="coerce")
    snapshot["Team"] = str(matchup.get("batting_team") or "")
    snapshot["Opponent"] = str(matchup.get("pitcher_team") or "")
    snapshot["HomeAway"] = str(matchup.get("home_away") or "")
    snapshot["Venue"] = str(matchup.get("venue") or "")
    snapshot["StartingPitcherID"] = matchup.get("pitcher_id")
    snapshot["StartingPitcher"] = str(matchup.get("pitcher_name") or "")
    snapshot["LineupStatus"] = str(lineup_status or "Unknown")
    snapshot["RawProbability"] = pd.to_numeric(
        snapshot.get(BINARY_RAW_PROBABILITY_COLUMN, snapshot.get(BINARY_PROBABILITY_COLUMN)),
        errors="coerce",
    )
    snapshot["ModelProbability"] = pd.to_numeric(
        snapshot.get(BINARY_PROBABILITY_COLUMN), errors="coerce"
    )
    snapshot["ModelScore"] = pd.to_numeric(snapshot.get(BINARY_SCORE_COLUMN), errors="coerce")
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
        "LineupStatus", "LineupSpot", "Projected_PA", "EffectiveStand", "SampleStatus",
        "Confidence", "Confidence_Level", "ParkFactor", "WeatherMultiplier",
        "RawProbability", "ModelProbability", "ModelScore", "Calibration_Applied",
        "Calibration_Version", "Market_Line", "Over_Odds", "Under_Odds", "Line_Source", "Line_Updated", "SportsbookOdds", "MarketProbability",
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
    if history is None or history.empty:
        return pd.DataFrame()
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
        "Projected_PA", "Confidence", "ParkFactor", "WeatherMultiplier", "RawProbability",
        "ModelProbability", "ModelScore", "SportsbookOdds", "MarketProbability",
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
) -> None:
    st.markdown(f"### {BINARY_TARGET_LABEL} backtesting and calibration")
    st.info(
        "Save the projection snapshot before first pitch. After the game is final, upload the snapshot or your latest master CSV and fetch official results."
    )
    fingerprint = f"{matchup.get('game_pk')}_{matchup.get('slate_date')}_{matchup.get('batting_team')}_{BINARY_TARGET_SLUG}"
    timestamp_key = f"{widget_prefix}_snapshot_timestamp_{fingerprint}"
    if timestamp_key not in st.session_state:
        st.session_state[timestamp_key] = pd.Timestamp.now(tz="UTC").isoformat()
    snapshot = build_binary_projection_snapshot(
        rankings, matchup, lookback_days, loaded_start, loaded_end, lineup_status,
        extra_context=extra_context,
        generated_at_utc=st.session_state[timestamp_key],
    )

    st.markdown("#### 1. Save today’s pregame predictions")
    st.caption(
        "Sportsbook odds and market probability are optional. American odds create a raw implied probability; enter a fair/no-vig probability manually when you have one."
    )
    editor_columns = [
        "PlayerID", "Player", "ModelProbability", "ModelScore", "SportsbookOdds",
        "MarketProbability", "Notes",
    ]
    editable = snapshot[[column for column in editor_columns if column in snapshot.columns]].copy()
    edited = st.data_editor(
        editable,
        hide_index=True,
        width="stretch",
        disabled=[column for column in ["PlayerID", "Player", "ModelProbability", "ModelScore"] if column in editable.columns],
        column_config={
            "SportsbookOdds": st.column_config.NumberColumn("American odds", step=1),
            "MarketProbability": st.column_config.NumberColumn(
                "Fair market probability", min_value=0.0, max_value=1.0, step=0.01, format="%.3f"
            ),
        },
        key=f"{widget_prefix}_pregame_market_editor_{fingerprint}",
    )
    if not edited.empty:
        edited_values = edited.set_index("PlayerID")
        for row_index in snapshot.index:
            player_id = snapshot.at[row_index, "PlayerID"]
            if pd.isna(player_id) or player_id not in edited_values.index:
                continue
            row = edited_values.loc[player_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for column in ["SportsbookOdds", "MarketProbability", "Notes"]:
                if column in row.index:
                    snapshot.at[row_index, column] = row[column]
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
ODDS_API_MARKET_KEY = "batter_home_runs"
ODDS_API_MARKET_LABEL = "Batter home runs"
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
        per_pa = pd.to_numeric(pd.Series([row.get("Model_HR_Per_PA")]), errors="coerce").iloc[0]
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
            "Leg 1": f"{float(probabilities.iloc[0]):.1%} · {trio.iloc[0].get('Team', '')}",
            "Player 2": str(trio.iloc[1].get("Player", "")),
            "Leg 2": f"{float(probabilities.iloc[1]):.1%} · {trio.iloc[1].get('Team', '')}",
            "Player 3": str(trio.iloc[2].get("Player", "")),
            "Leg 3": f"{float(probabilities.iloc[2]):.1%} · {trio.iloc[2].get('Team', '')}",
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
    use_weather: bool,
    bat_tracking_auto: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str]]:
    boards, errors = [], []
    total = max(len(games) * 2, 1)
    progress = st.progress(0.0, text="Preparing whole-slate HR board...")
    completed = 0
    for game in games:
        weather_multiplier = 1.0
        weather_label = "Neutral"
        if use_weather:
            weather = automatic_game_weather(game)
            metadata = weather.get("metadata") or load_stadium_weather_metadata(str(game.get("venue", "")))
            enclosed = str(metadata.get("roof_type", "open")).lower() == "fixed"
            if weather.get("ok"):
                weather_multiplier = weather_carry_multiplier(
                    float(weather.get("temperature_f", 75.0)), float(weather.get("humidity_pct", 50.0)),
                    float(weather.get("wind_mph", 5.0)), str(weather.get("wind_direction", "Cross/Calm")),
                    float(weather.get("pressure_hpa", 1013.25)), 1.0, enclosed=enclosed,
                )
                weather_label = str(weather.get("condition", "Forecast"))
            elif enclosed:
                weather_label = "Fixed roof"
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
            park = fetch_savant_park_factors(game.get("venue", ""), pd.Timestamp(loaded_end).year, "HR")
            try:
                board, _ = build_hr_board(
                    df=df, pitcher_id=int(pitcher_id), team=team, min_pa=min_pa,
                    end_date=pd.Timestamp(loaded_end), park_hr_factor_lhb=float(park["L"]),
                    park_hr_factor_rhb=float(park["R"]), team_runs=float(team_runs),
                    is_away=side == "away", starter_innings=float(starter_innings),
                    bullpen_multiplier=float(bullpen_multiplier), weather_multiplier=float(weather_multiplier),
                    lineup_override=lineup_seed, lineup_edits=None, bat_tracking_upload=None,
                    bat_tracking_auto=bat_tracking_auto, active_roster=roster,
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
            board["SlateWeather"] = weather_label
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
    bat_tracking_auto: pd.DataFrame | None,
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
    use_slate_weather = st.checkbox("Use automatic weather in slate model", value=True, key=f"{widget_prefix}_slate_weather")
    board_key = f"{widget_prefix}_whole_slate_board_{slate_date}"
    error_key = f"{widget_prefix}_whole_slate_errors_{slate_date}"
    if st.button("Build / refresh whole-slate board", type="primary", width="stretch", key=f"{widget_prefix}_build_slate"):
        with st.spinner("Building both offenses for every available game..."):
            built, errors = build_whole_slate_model_board(
                df, games, available_teams, min_pa, loaded_end, include_low_sample,
                use_confirmed, slate_team_runs, slate_starter_innings, slate_bullpen,
                use_slate_weather, bat_tracking_auto,
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
        BINARY_PROBABILITY_COLUMN, BINARY_SCORE_COLUMN, "Confidence", "Confidence_Level",
        "SlateGrade", "LineupStatus", "Market_Line", "Market_Over_Prob", "Model_Market_Edge", "Line_Source",
    ]
    top10 = ranked_slate[[column for column in top_columns if column in ranked_slate.columns]].head(10).copy()
    top10 = top10.rename(columns={
        BINARY_PROBABILITY_COLUMN: BINARY_TARGET_LABEL,
        BINARY_SCORE_COLUMN: "Model Score",
        "LineupSpot": "Order",
        "Market_Over_Prob": "No-vig Market",
        "Model_Market_Edge": "Model Edge",
        "Market_Line": "Line",
    })
    format_map = {
        BINARY_TARGET_LABEL: "{:.1%}", "Model Score": "{:.1f}", "Confidence": "{:.1f}",
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
MODEL_KEY = "clean_hr"
odds_api_key = read_streamlit_secret("THE_ODDS_API_KEY")

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
        help="Roster players with little or no Statcast history are included with league-average priors and Low confidence.",
    )
    refresh = st.button(
        "Load / refresh Statcast", type="primary", key="clean_hr_refresh"
    )
    st.divider()
    st.subheader("Probability calibration")
    hr_calibration_upload = st.file_uploader(
        "Optional HR calibration JSON", type=["json"], key="clean_hr_calibration_upload"
    )
    if hr_calibration_upload is not None:
        calibration_signature = (
            getattr(hr_calibration_upload, "name", "calibration.json"),
            getattr(hr_calibration_upload, "size", None),
        )
        signature_key = "hr_calibration_upload_signature"
        if st.session_state.get(signature_key) != calibration_signature:
            uploaded_calibration, calibration_error = parse_binary_calibration_upload(hr_calibration_upload)
            if calibration_error:
                st.error(calibration_error)
            elif uploaded_calibration:
                st.session_state["hr_probability_calibration"] = uploaded_calibration
                st.session_state[signature_key] = calibration_signature
    if st.session_state.get("hr_probability_calibration"):
        loaded_calibration = st.session_state["hr_probability_calibration"]
        st.caption(
            f"Calibration active · n={int(loaded_calibration.get('n', 0)):,} · "
            f"slope {float(loaded_calibration.get('slope', 1.0)):.3f}"
        )
        if st.button("Clear HR calibration", key="clean_hr_clear_calibration"):
            st.session_state.pop("hr_probability_calibration", None)
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
    st.warning("No Statcast data was returned. Use completed dates and try again.")
    st.stop()

loaded_start, loaded_end = st.session_state["clean_hr_loaded_dates"]
st.caption(
    f"Using {len(df):,} pitches from {loaded_start} through {loaded_end}. "
    "The slate selector can use today's games while model statistics stop at the date above."
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

attack_profile = pitcher_attack_profile_hr(df, selected_pitcher)
render_pitcher_attack_panel(attack_profile, matchup["pitcher_name"], "home runs")

park_year = pd.Timestamp(matchup.get("slate_date") or loaded_end).year
auto_park = fetch_savant_park_factors(matchup.get("venue", ""), park_year, "HR")

with st.expander("Game, park and weather adjustments", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        team_runs = st.number_input(
            "Team implied runs", 1.0, 9.0, 4.5, 0.1, key="clean_hr_runs"
        )
        starter_innings = st.number_input(
            "Expected starter innings", 2.0, 8.0, 5.5, 0.5, key="clean_hr_ip"
        )
        bullpen_multiplier = st.number_input(
            "Bullpen HR multiplier",
            0.65,
            1.45,
            1.00,
            0.01,
            key="clean_hr_bullpen",
            help="Above 1.00 means a more home-run-prone bullpen.",
        )
    with c2:
        manual_park_override = st.checkbox(
            "Manual park-factor override",
            value=False,
            key=f"clean_hr_manual_park_{matchup.get('game_pk')}_{normalize_name(matchup.get('venue'))}",
            help="Leave this off to use the local three-year park_factors.csv values.",
        )
        if manual_park_override:
            park_hr_factor_lhb = st.number_input(
                "Park HR factor — LHB", 60.0, 160.0, float(auto_park["L"]), 1.0,
                key=f"clean_hr_park_l_{matchup.get('game_pk')}_{normalize_name(matchup.get('venue'))}",
            )
            park_hr_factor_rhb = st.number_input(
                "Park HR factor — RHB", 60.0, 160.0, float(auto_park["R"]), 1.0,
                key=f"clean_hr_park_r_{matchup.get('game_pk')}_{normalize_name(matchup.get('venue'))}",
            )
            park_source = "Manual override"
        else:
            park_hr_factor_lhb = float(auto_park["L"])
            park_hr_factor_rhb = float(auto_park["R"])
            park_source = str(auto_park["source"])
            st.markdown(
                f"""
                <div class="park-grid">
                    <div class="park-chip"><div class="side">LHB HOME RUNS</div><div class="factor">{park_hr_factor_lhb:.0f}</div></div>
                    <div class="park-chip"><div class="side">RHB HOME RUNS</div><div class="factor">{park_hr_factor_rhb:.0f}</div></div>
                </div>
                <div class="park-source">{escape(park_source)}</div>
                """,
                unsafe_allow_html=True,
            )
            if not auto_park["ok"]:
                st.warning("Local park_factors.csv was unavailable or could not be matched, so neutral 100 values are being used. Turn on the manual override to change them.")
        if st.button("Refresh park factors", key=f"clean_hr_refresh_park_{matchup.get('game_pk')}"):
            fetch_savant_park_factors.clear()
            st.rerun()

    with c3:
        use_auto_weather = st.checkbox(
            "Automatic game-time weather",
            value=True,
            key=f"clean_hr_auto_weather_{matchup.get('game_pk')}_{normalize_name(matchup.get('venue'))}",
            help="Uses the Open-Meteo hourly forecast nearest scheduled first pitch.",
        )
        weather_result = automatic_game_weather(matchup) if use_auto_weather else {"ok": False, "error": "Automatic weather disabled."}
        metadata = weather_result.get("metadata") or load_stadium_weather_metadata(str(matchup.get("venue", "")))
        roof_type = str(metadata.get("roof_type", "open")).lower()
        if roof_type == "fixed":
            roof_status = "Closed (fixed roof)"
            roof_closed = True
            st.info("Fixed roof: outdoor weather is neutralized.")
        elif roof_type == "retractable":
            roof_status = st.selectbox(
                "Roof status",
                ["Open / use forecast", "Closed / neutral weather"],
                key=f"clean_hr_roof_{matchup.get('game_pk')}_{normalize_name(matchup.get('venue'))}",
            )
            roof_closed = roof_status.startswith("Closed")
        else:
            roof_status = "Open air"
            roof_closed = False

        if use_auto_weather and weather_result.get("ok"):
            st.success(
                f"{weather_result.get('condition', 'Forecast')} · "
                f"{weather_result.get('forecast_time_utc', '')}"
            )
            default_temperature = float(np.clip(weather_result.get("temperature_f", 75.0), 20.0, 120.0))
            default_humidity = float(np.clip(weather_result.get("humidity_pct", 50.0), 1.0, 100.0))
            default_wind = float(np.clip(weather_result.get("wind_mph", 5.0), 0.0, 50.0))
            default_pressure = float(np.clip(weather_result.get("pressure_hpa", 1013.25), 930.0, 1060.0))
            default_wind_direction = str(weather_result.get("wind_direction", "Cross/Calm"))
            precip_probability = float(weather_result.get("precip_probability", 0.0))
            weather_source = str(weather_result.get("source", "Open-Meteo"))
            st.caption(
                f"Rain chance {precip_probability:.0f}% · gusts {weather_result.get('wind_gust_mph', 0.0):.0f} mph · "
                f"wind from {weather_result.get('wind_from_degrees', 0.0):.0f}° classified {default_wind_direction}."
            )
        else:
            default_temperature = 75.0
            default_humidity = 50.0
            default_wind = 5.0
            default_pressure = 1013.25
            default_wind_direction = "Cross/Calm"
            precip_probability = np.nan
            weather_source = "Manual fallback"
            if use_auto_weather:
                st.warning("Automatic weather was unavailable; the editable manual defaults are being used.")
                if weather_result.get("error"):
                    st.caption(str(weather_result["error"])[:300])

        weather_key = f"{matchup.get('game_pk')}_{normalize_name(matchup.get('venue'))}"
        temperature_f = st.number_input(
            "Temperature °F", 20.0, 120.0, default_temperature, 1.0,
            key=f"clean_hr_temp_{weather_key}",
        )
        humidity_pct = st.number_input(
            "Humidity %", 1.0, 100.0, default_humidity, 1.0,
            key=f"clean_hr_humidity_{weather_key}",
        )
        wind_mph = st.number_input(
            "Wind mph", 0.0, 50.0, default_wind, 1.0,
            key=f"clean_hr_wind_{weather_key}",
        )
        wind_options = ["Cross/Calm", "Out", "In"]
        wind_index = wind_options.index(default_wind_direction) if default_wind_direction in wind_options else 0
        wind_direction = st.selectbox(
            "Stadium wind effect", wind_options, index=wind_index,
            key=f"clean_hr_wind_dir_{weather_key}",
        )
        pressure_hpa = st.number_input(
            "Sea-level pressure (hPa)", 930.0, 1060.0, default_pressure, 0.5,
            key=f"clean_hr_pressure_{weather_key}",
        )
        manual_weather = st.number_input(
            "Extra manual weather multiplier",
            0.80,
            1.20,
            1.00,
            0.01,
            key=f"clean_hr_weather_manual_{weather_key}",
        )
        if st.button("Refresh weather", key=f"clean_hr_refresh_weather_{weather_key}"):
            fetch_open_meteo_game_weather.clear()
            fetch_mlb_venue_coordinates.clear()
            st.rerun()

    with c4:
        use_auto_bat_tracking = st.checkbox(
            "Automatic Savant bat tracking",
            value=True,
            key="clean_hr_auto_bat_tracking",
            help="Downloads current-season Baseball Savant bat-tracking data and uses the previous season as a player-level fallback.",
        )
        bat_tracking_result = {
            "data": pd.DataFrame(), "ok": False, "source": "Disabled", "error": None, "columns": []
        }
        if use_auto_bat_tracking:
            bat_tracking_result = fetch_savant_bat_tracking_dataset(int(pd.Timestamp(loaded_end).year))
            if bat_tracking_result["ok"]:
                st.success(
                    f"Bat tracking: {bat_tracking_result['source']} · "
                    f"{len(bat_tracking_result['data']):,} players"
                )
            else:
                st.warning(
                    "Automatic bat tracking was unavailable. Players without a manual CSV "
                    "will stay neutral internally and display as No data."
                )
                if bat_tracking_result.get("error"):
                    st.caption(str(bat_tracking_result["error"])[:280])
        bat_tracking_auto = bat_tracking_result["data"] if use_auto_bat_tracking else pd.DataFrame()
        if st.button("Refresh bat tracking", key="clean_hr_refresh_bat_tracking"):
            fetch_savant_bat_tracking_dataset.clear()
            st.rerun()

        bat_tracking_upload = st.file_uploader(
            "Optional manual bat-tracking CSV override",
            type=["csv"],
            key="clean_hr_bat_upload",
            help="Any nonblank uploaded values override the automatic Savant values for the matching player.",
        )
        st.markdown(
            f"""
            <div class="context-note">
            <b>{matchup['away_abbr']} @ {matchup['home_abbr']}</b><br>
            {matchup['venue']}<br>
            Batting team: {selected_team} ({home_away})<br>
            Park LHB: ×{park_hr_factor_lhb / 100:.2f}<br>
            Park RHB: ×{park_hr_factor_rhb / 100:.2f}<br>
            Park source: {escape(park_source)}<br>
            Weather: {escape(weather_source)} · {escape(roof_status)}<br>
            Bat source: {escape(str(bat_tracking_result.get('source', 'Disabled')))}
            </div>
            """,
            unsafe_allow_html=True,
        )
        template = (
            "player_id,Player,avg_bat_speed,fast_swing_rate,blasts_per_bat_contact,"
            "squared_up_per_bat_contact,avg_attack_angle,avg_attack_direction\n"
            "660271,Shohei Ohtani,,,,,,\n"
        )
        st.download_button(
            "Bat-tracking CSV template",
            template,
            "bat_tracking_template.csv",
            "text/csv",
            key="clean_hr_bat_template",
        )

weather_multiplier = weather_carry_multiplier(
    temperature_f,
    humidity_pct,
    wind_mph,
    wind_direction,
    pressure_hpa,
    manual_weather,
    enclosed=roof_closed,
)
st.caption(
    f"Weather carry adjustment: ×{weather_multiplier:.3f}. "
    "Automatic values are editable. This is a simple transparent adjustment, not an official ball-flight model."
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
        key=f"clean_hr_use_mlb_lineup_{lineup_game_pk}_{selected_team}",
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
            key=f"clean_hr_refresh_lineup_{lineup_game_pk}_{selected_team}",
        ):
            fetch_mlb_confirmed_lineup.clear()
            st.rerun()
    elif not lineup_game_pk:
        st.caption("Automatic lineups require a game selected from the MLB slate. Manual matchups use the recent-lineup fallback.")

st.caption(f"Lineup source: {lineup_source_label}")
preview_board = aggregate_hr_hitters(df, preview_hand, 0)
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
        key=f"clean_hr_lineup_{selected_pitcher}_{selected_team}_{_lineup_fingerprint(lineup_seed)}",
    )

rankings, pitch_mix = build_hr_board(
    df=df,
    pitcher_id=selected_pitcher,
    team=selected_team,
    min_pa=min_pa,
    end_date=pd.Timestamp(loaded_end),
    park_hr_factor_lhb=park_hr_factor_lhb,
    park_hr_factor_rhb=park_hr_factor_rhb,
    team_runs=team_runs,
    is_away=home_away == "Away",
    starter_innings=starter_innings,
    bullpen_multiplier=bullpen_multiplier,
    weather_multiplier=weather_multiplier,
    lineup_override=lineup_seed,
    lineup_edits=lineup_edits,
    bat_tracking_upload=bat_tracking_upload,
    bat_tracking_auto=bat_tracking_auto,
    active_roster=active_roster,
    include_low_sample=include_low_sample,
)

if rankings.empty:
    st.warning("No selected hitters remain after filtering.")
    st.stop()

rankings = apply_binary_probability_calibration(
    rankings, st.session_state.get("hr_probability_calibration")
)
rankings, loaded_odds_quotes, odds_source_mode = render_batter_odds_section(
    rankings, matchup, odds_api_key, "clean_hr"
)

render_board_header(matchup, selected_team, matchup["pitcher_name"], "ADVANCED HOME RUN BOARD")
render_leader_cards(rankings, "Model_1plus_HR", "HRScore", "model 1+ HR")

quick_tab, power_tab, matchup_tab, pitch_type_tab, tracking_tab, pitcher_tab, slate_tab, backtest_tab, notes_tab = st.tabs(
    [
        "Quick board", "Power profile", "Matchup detail", "Batter vs pitch type",
        "Bat tracking", "Pitcher profile", "Slate Top 10 & pairings", "Backtest & calibration", "Model notes"
    ]
)

pitch_type_table = build_batter_pitch_type_table(
    df,
    rankings,
    pitch_mix,
    preview_hand,
)

with quick_tab:
    quick_columns = [
        "Rank", "Player", "LineupSpot", "Projected_PA", "Model_1plus_HR",
        "Raw_Model_1plus_HR", "HRScore", "Confidence_Level", "Adj_Brl_PA", "Adj_xISO_PA",
        "EffectiveStand", "PitcherSideRead", "PitcherSideAttackScore", "SampleStatus",
        "PitchMatchScore", "ZoneFitScore", "PitcherPowerScore", "ParkFactor",
        "Market_Line", "Over_Odds", "Under_Odds", "Market_Over_Prob",
        "Model_Market_Edge", "Line_Source",
    ]
    quick = rankings[[column for column in quick_columns if column in rankings.columns]].copy()
    if not bool(rankings.get("Calibration_Applied", pd.Series(False, index=rankings.index)).fillna(False).any()):
        quick = quick.drop(columns=["Raw_Model_1plus_HR"], errors="ignore")
    quick = quick.rename(columns={
        "LineupSpot": "Order", "Projected_PA": "Proj PA", "Model_1plus_HR": "1+ HR",
        "Raw_Model_1plus_HR": "Raw 1+ HR", "HRScore": "HR Score", "Confidence_Level": "Confidence",
        "Adj_Brl_PA": "Adj Brl/PA", "Adj_xISO_PA": "Adj xISO/PA",
        "EffectiveStand": "Bats vs SP", "PitcherSideRead": "Pitcher Read",
        "PitcherSideAttackScore": "Side Attack", "SampleStatus": "Sample",
        "PitchMatchScore": "Pitch Match", "ZoneFitScore": "Zone Fit",
        "PitcherPowerScore": "Pitcher Power", "ParkFactor": "Park Factor",
        "Market_Line": "Line", "Over_Odds": "Over Odds", "Under_Odds": "Under Odds",
        "Market_Over_Prob": "No-vig Market", "Model_Market_Edge": "Model Edge",
        "Line_Source": "Line Source",
    })
    score_cols = ["HR Score", "Side Attack", "Pitch Match", "Zone Fit", "Pitcher Power"]
    styler = quick.style.background_gradient(
        cmap="RdYlGn", subset=score_cols, vmin=0, vmax=100
    ).format({
        "Order": "{:.0f}", "Proj PA": "{:.2f}", "1+ HR": "{:.1%}", "Raw 1+ HR": "{:.1%}",
        "HR Score": "{:.1f}", "Adj Brl/PA": "{:.2%}", "Adj xISO/PA": "{:.3f}",
        "Side Attack": "{:.1f}", "Pitch Match": "{:.1f}", "Zone Fit": "{:.1f}", "Pitcher Power": "{:.1f}",
        "Park Factor": "{:.0f}", "Line": "{:.1f}", "Over Odds": "{:+.0f}",
        "Under Odds": "{:+.0f}", "No-vig Market": "{:.1%}", "Model Edge": "{:+.1%}",
    })
    st.dataframe(styler, use_container_width=True, hide_index=True, height=520)

with power_tab:
    st.caption(
        "Green cells are stronger home-run ingredients. Red cells in K% and Whiff% "
        "show swing-and-miss risk. LA Power Fit rewards launch angles closest to the "
        "typical power window rather than simply rewarding the highest angle."
    )
    columns = [
        "Player", "PA", "HR", "HR_PA", "Brl_PA", "Brl_BIP", "PullAir_BIP",
        "PullAir_Air", "xSLG_PA", "xISO_PA", "xSLG_Contact", "HH_Pct",
        "SweetSpot_Pct", "FB_Pct", "Avg_EV", "EV90", "Max_EV", "Avg_LA",
        "K_Pct", "Whiff_Pct",
    ]
    power = rankings[[column for column in columns if column in rankings.columns]].copy()
    power["LA_Power_Fit"] = (
        100.0 - (pd.to_numeric(power.get("Avg_LA"), errors="coerce") - 22.0).abs() * 6.0
    ).clip(0, 100)
    power = power.rename(columns={
        "HR_PA": "HR/PA", "Brl_PA": "Brl/PA", "Brl_BIP": "Brl/BIP",
        "PullAir_BIP": "Pull Air/BIP", "PullAir_Air": "Pull Air/Air",
        "xSLG_PA": "xSLG/PA", "xISO_PA": "xISO/PA", "xSLG_Contact": "xSLG Contact",
        "HH_Pct": "Hard Hit%", "SweetSpot_Pct": "Sweet Spot%", "FB_Pct": "FB%",
        "Avg_EV": "Avg EV", "Max_EV": "Max EV", "Avg_LA": "Avg LA",
        "LA_Power_Fit": "LA Power Fit", "K_Pct": "K%", "Whiff_Pct": "Whiff%",
    })
    positive_columns = [
        "HR/PA", "Brl/PA", "Brl/BIP", "Pull Air/BIP", "Pull Air/Air",
        "xSLG/PA", "xISO/PA", "xSLG Contact", "Hard Hit%", "Sweet Spot%",
        "FB%", "Avg EV", "EV90", "Max EV", "LA Power Fit",
    ]
    positive_columns = [column for column in positive_columns if column in power.columns]
    risk_columns = [column for column in ["K%", "Whiff%"] if column in power.columns]

    power_style = power.style
    if positive_columns:
        power_style = power_style.background_gradient(
            cmap="RdYlGn", subset=positive_columns, axis=0
        )
    if risk_columns:
        power_style = power_style.background_gradient(
            cmap="RdYlGn_r", subset=risk_columns, axis=0
        )
    power_style = power_style.set_properties(
        subset=["Player"], **{"font-weight": "700", "background-color": "#f8fafc"}
    ).format({
        "HR/PA": "{:.2%}", "Brl/PA": "{:.2%}", "Brl/BIP": "{:.1%}",
        "Pull Air/BIP": "{:.1%}", "Pull Air/Air": "{:.1%}",
        "xSLG/PA": "{:.3f}", "xISO/PA": "{:.3f}", "xSLG Contact": "{:.3f}",
        "Hard Hit%": "{:.1%}", "Sweet Spot%": "{:.1%}", "FB%": "{:.1%}",
        "Avg EV": "{:.1f}", "EV90": "{:.1f}", "Max EV": "{:.1f}",
        "Avg LA": "{:.1f}", "LA Power Fit": "{:.0f}",
        "K%": "{:.1%}", "Whiff%": "{:.1%}",
    })
    st.dataframe(
        power_style,
        use_container_width=True,
        hide_index=True,
        height=560,
    )

with matchup_tab:
    columns = [
        "Player", "PitchMatchScore", "ZoneFitScore", "PitcherPowerScore",
        "RecentFormScore", "BvP_PA", "BvP_HR", "BvPScore", "MatchSample",
        "Platoon_PA", "Pitcher_PA",
    ]
    detail = rankings[[column for column in columns if column in rankings.columns]].copy()
    detail = detail.rename(columns={
        "PitchMatchScore": "Pitch Match", "ZoneFitScore": "Zone Fit",
        "PitcherPowerScore": "Pitcher Power", "RecentFormScore": "Recent Form",
        "BvP_PA": "BvP PA", "BvP_HR": "BvP HR", "BvPScore": "BvP Score",
        "MatchSample": "Matched Pitches", "Platoon_PA": "Platoon PA",
        "Pitcher_PA": "Pitcher Split PA",
    })
    score_cols = ["Pitch Match", "Zone Fit", "Pitcher Power", "Recent Form", "BvP Score"]
    st.dataframe(
        detail.style.background_gradient(cmap="RdYlGn", subset=score_cols, vmin=0, vmax=100).format(
            {column: "{:.1f}" for column in score_cols}
        ),
        use_container_width=True,
        hide_index=True,
        height=520,
    )


with pitch_type_tab:
    st.caption(
        "Historical results against each pitch type used by the selected starter, filtered to "
        "the starter's throwing hand. Rates are shrunk toward league pitch-type averages, so "
        "small samples stay near neutral. Pitch Type Score is 0–100; 50 is neutral."
    )
    if pitch_type_table.empty:
        st.info("No batter-versus-pitch-type data were available for this matchup.")
    else:
        hitter_options = rankings.sort_values("Rank")["Player"].tolist()
        selected_pitch_hitter = st.selectbox(
            "Hitter",
            hitter_options,
            key=f"clean_hr_pitch_type_hitter_{selected_pitcher}_{selected_team}",
        )
        pitch_view = pitch_type_table[pitch_type_table["Player"].eq(selected_pitch_hitter)].copy()
        pitch_view = pitch_view.sort_values("Pitcher Usage", ascending=False)
        display_columns = [
            "Pitch Type", "Pitcher Usage", "Pitcher Velo", "Pitches Seen", "PA Ends", "BBE", "HR",
            "HR/PA", "Brl/BIP", "xSLG Contact", "xBA Contact", "Hard Hit%",
            "Pull Air/BIP", "Contact%", "Whiff%", "Pitch Type Score", "Sample",
        ]
        pitch_view = pitch_view[[column for column in display_columns if column in pitch_view.columns]]
        positive = [
            column for column in [
                "HR/PA", "Brl/BIP", "xSLG Contact", "xBA Contact", "Hard Hit%",
                "Pull Air/BIP", "Contact%", "Pitch Type Score"
            ] if column in pitch_view.columns
        ]
        risk = [column for column in ["Whiff%"] if column in pitch_view.columns]
        pitch_style = pitch_view.style
        if positive:
            pitch_style = pitch_style.background_gradient(cmap="RdYlGn", subset=positive, axis=0)
        if risk:
            pitch_style = pitch_style.background_gradient(cmap="RdYlGn_r", subset=risk, axis=0)
        pitch_style = pitch_style.format({
            "Pitcher Usage": "{:.1%}", "Pitcher Velo": "{:.1f}", "HR/PA": "{:.2%}",
            "Brl/BIP": "{:.1%}", "xSLG Contact": "{:.3f}", "xBA Contact": "{:.3f}",
            "Hard Hit%": "{:.1%}", "Pull Air/BIP": "{:.1%}", "Contact%": "{:.1%}",
            "Whiff%": "{:.1%}", "Pitch Type Score": "{:.1f}",
        })
        st.dataframe(
            pitch_style,
            use_container_width=True,
            hide_index=True,
            height=470,
        )

with tracking_tab:
    available_count = int(rankings.get("BatTrackingAvailable", pd.Series(False, index=rankings.index)).sum())
    st.caption(
        f"Actual bat-tracking data matched {available_count} of {len(rankings)} displayed hitters. "
        "A missing player is shown as No data; the model uses a neutral 50 internally so missing "
        "tracking does not punish or boost that hitter."
    )
    columns = [
        "Player", "BatTrackingAvailable", "BatTrackingScore", "BatTrackingMetricCount",
        "BatTrackingSeason", "BatTrackingSource", "Bat_Speed", "Fast_Swing_Rate",
        "Blast_Contact_Rate", "Squared_Up_Contact_Rate", "Attack_Angle",
        "Attack_Direction",
    ]
    tracking = rankings[[column for column in columns if column in rankings.columns]].copy()
    tracking["Data Status"] = np.where(
        tracking.get("BatTrackingAvailable", False), "Available", "No data"
    )
    tracking["Bat Tracking Display"] = pd.to_numeric(
        tracking.get("BatTrackingScore"), errors="coerce"
    ).where(tracking["Data Status"].eq("Available"))
    tracking = tracking.rename(columns={
        "Bat Tracking Display": "Bat Tracking", "BatTrackingMetricCount": "Metrics",
        "BatTrackingSeason": "Season", "BatTrackingSource": "Source",
        "Bat_Speed": "Bat Speed", "Fast_Swing_Rate": "Fast Swing%",
        "Blast_Contact_Rate": "Blast/Contact",
        "Squared_Up_Contact_Rate": "Squared Up/Contact",
        "Attack_Angle": "Attack Angle", "Attack_Direction": "Attack Direction",
    })
    display_columns = [
        "Player", "Data Status", "Bat Tracking", "Metrics", "Season", "Source",
        "Bat Speed", "Fast Swing%", "Blast/Contact", "Squared Up/Contact",
        "Attack Angle", "Attack Direction",
    ]
    tracking = tracking[[column for column in display_columns if column in tracking.columns]]
    tracking_style = tracking.style
    tracking_positive = [
        column for column in [
            "Bat Tracking", "Bat Speed", "Fast Swing%", "Blast/Contact",
            "Squared Up/Contact"
        ] if column in tracking.columns
    ]
    if tracking_positive:
        tracking_style = tracking_style.background_gradient(
            cmap="RdYlGn", subset=tracking_positive, axis=0
        )
    tracking_style = tracking_style.set_properties(
        subset=["Player"], **{"font-weight": "700", "background-color": "#f8fafc"}
    ).format({
        "Bat Tracking": "{:.1f}", "Metrics": "{:.0f}", "Season": "{:.0f}",
        "Bat Speed": "{:.1f}", "Fast Swing%": "{:.1%}",
        "Blast/Contact": "{:.1%}", "Squared Up/Contact": "{:.1%}",
        "Attack Angle": "{:.1f}", "Attack Direction": "{:.1f}",
    }, na_rep="—")
    st.dataframe(
        tracking_style,
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
                "Allowed_xSLG": "{:.3f}", "Allowed_Barrel": "{:.1%}",
            }),
            use_container_width=True,
            hide_index=True,
        )


with slate_tab:
    render_slate_tools(
        df=df, matchup=matchup, available_teams=available_teams, min_pa=min_pa,
        loaded_end=loaded_end, include_low_sample=include_low_sample,
        calibration=st.session_state.get("hr_probability_calibration"),
        odds_quotes=loaded_odds_quotes, odds_source_mode=odds_source_mode,
        widget_prefix="clean_hr_slate", bat_tracking_auto=bat_tracking_auto,
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
        calibration_state_key="hr_probability_calibration",
        widget_prefix="clean_hr_backtest",
        extra_context={
            "ParkSource": park_source,
            "WeatherMultiplier": weather_multiplier,
            "WeatherSource": weather_source,
            "RoofStatus": roof_status,
            "TemperatureF": temperature_f,
            "HumidityPct": humidity_pct,
            "WindMPH": wind_mph,
            "WindDirection": wind_direction,
            "PressureHPA": pressure_hpa,
            "TeamImpliedRuns": team_runs,
            "StarterInnings": starter_innings,
            "BullpenMultiplier": bullpen_multiplier,
        },
    )

with notes_tab:
    st.markdown(
        """
        ### Reading the board
        - **1+ HR** combines the modeled home-run rate per plate appearance with projected plate appearances.
        - **HR Score** is a 0–100 comparison score within the selected offense, not a literal probability.
        - **Pitcher Read / Side Attack** grades whether the starter has been more attackable or avoidable for LHB or RHB, with small samples shrunk toward neutral.
        - **Pitch Match** compares the hitter with the starter's pitch types, velocity, movement and extension.
        - **Batter vs pitch type** surfaces the underlying historical split against the starter's pitch mix and throwing hand; it is already summarized by Pitch Match and is not added a second time.
        - **Zone Fit** weights hitter power by the locations the starter uses.
        - **Park Factor** is automatically matched to the selected venue from Baseball Savant when its live table is available; 110 means ×1.10 and 90 means ×0.90. A neutral/manual fallback remains available.
        - The weather multiplier is applied after the park adjustment.
        - **Sample** marks hitters below the selected PA threshold. Active-roster players with no history use league-average priors and remain Low confidence.
        - **Confidence** reflects sample size and matchup-data depth, not certainty that the outcome will occur.

        - **Automatic odds** can populate current-game or whole-slate lines, no-vig probability and model edge from The Odds API.
        - **Slate Top 10 & pairings** builds both offenses across the slate, grades hitters from probability, model score and confidence, and creates multiple diversified three-person suggestions.
        - **Backtest & calibration** exports timestamped pregame snapshots, retrieves official results, checks Brier score/log loss, validates HR Score buckets and creates a probability-calibration JSON.
        - Download the updated master-history CSV after every result update because Streamlit temporary storage can reset.

        The probabilities remain heuristic until tested and calibrated on held-out historical games.
        """
    )
