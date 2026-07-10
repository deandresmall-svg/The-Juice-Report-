from __future__ import annotations

import json
from datetime import date, timedelta
from html import escape
from io import StringIO
from itertools import combinations
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from pybaseball import cache, playerid_reverse_lookup, statcast

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="HR Edge v2.1", page_icon="⚾", layout="wide")
cache.enable()

MODEL_VERSION = "2.1 Enhanced - Complete"
RECENT_DAYS = 14
RECENT_FORM_DAYS = 7
LOOKBACK_MEDIUM = 37

HIT_EVENTS = {"single", "double", "triple", "home_run"}
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
SWING_DESCRIPTIONS = {"hit_into_play", "foul", "foul_tip", "foul_bunt", "missed_bunt", "swinging_strike", "swinging_strike_blocked"}
CONTACT_DESCRIPTIONS = {"hit_into_play", "foul", "foul_tip", "foul_bunt"}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}

# =============================================================================
# ALL HELPER FUNCTIONS (from original + new v2.1)
# =============================================================================

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

def shrink_rate(numerator, denominator, league_rate, prior_sample):
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    den = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    return (num + league_rate * prior_sample) / (den + prior_sample)

def normalize_name(value: object) -> str:
    if pd.isna(value): return ""
    return "".join(c.lower() for c in str(value) if c.isalnum())

@st.cache_data(ttl=21600, show_spinner=False)
def load_statcast(start_date: str, end_date: str) -> pd.DataFrame:
    return statcast(start_dt=start_date, end_dt=end_date, verbose=False, parallel=False)

@st.cache_data(ttl=86400, show_spinner=False)
def lookup_names(player_ids: tuple[int, ...]) -> pd.DataFrame:
    if not player_ids: return pd.DataFrame(columns=["player_id", "Player"])
    try:
        lookup = playerid_reverse_lookup(list(player_ids), key_type="mlbam")
    except Exception:
        return pd.DataFrame(columns=["player_id", "Player"])
    if lookup.empty: return pd.DataFrame(columns=["player_id", "Player"])
    lookup["Player"] = (lookup["name_first"].fillna("").str.title() + " " + lookup["name_last"].fillna("").str.title()).str.strip()
    return lookup.rename(columns={"key_mlbam": "player_id"})[["player_id", "Player"]]

def most_common(series: pd.Series, default: str = "") -> str:
    mode = series.dropna().astype(str).mode()
    return default if mode.empty else str(mode.iloc[0])

def quantile_90(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return np.nan if values.empty else float(values.quantile(0.90))

def add_launch_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    ev = pd.to_numeric(result.get("launch_speed"), errors="coerce").astype("float64")
    la = pd.to_numeric(result.get("launch_angle"), errors="coerce").astype("float64")

    sweet_spot = la.between(8, 32, inclusive="both").fillna(False)
    barrel_range = (ev.ge(98).fillna(False) & (la.between(26, 30, inclusive="both").fillna(False) | (ev.ge(105).fillna(False) & la.between(20, 35, inclusive="both").fillna(False))))

    la_deviation = (la - 27.0).abs()
    la_score = (100.0 - la_deviation * 3.0).clip(lower=0.0, upper=100.0).fillna(50.0)

    result["Sweet_Spot"] = sweet_spot.astype(int)
    result["Barrel_Range"] = barrel_range.astype(int)
    result["LA_Deviation_from_Optimal"] = la_deviation
    result["LA_Optimization_Score"] = la_score
    result["LA_Power_Score"] = la_score
    result["is_sweet_spot"] = result["Sweet_Spot"].astype(bool)

    # NEW v2.1
    optimal_window = (ev.ge(92).fillna(False) & la.between(22, 32, inclusive="both").fillna(False)) | barrel_range
    result["Launch_Window_Efficiency"] = optimal_window.astype(int)
    return result

def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty: return pd.DataFrame()
    df = raw.copy()
    expected = ["batter","pitcher","game_pk","at_bat_number","game_date","player_name","home_team","away_team","inning_topbot",
                "description","events","bb_type","launch_speed","launch_angle","launch_speed_angle",
                "estimated_ba_using_speedangle","estimated_slg_using_speedangle","zone","plate_x","plate_z",
                "pitch_name","release_speed","pfx_x","pfx_z","release_extension","stand","p_throws","hc_x","hc_y"]
    for col in expected:
        if col not in df.columns: df[col] = np.nan

    for col in ["batter","pitcher","game_pk","at_bat_number","launch_speed","launch_angle","launch_speed_angle",
                "estimated_ba_using_speedangle","estimated_slg_using_speedangle","zone","plate_x","plate_z",
                "release_speed","pfx_x","pfx_z","release_extension","hc_x","hc_y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    top_mask = df["inning_topbot"].eq("Top").fillna(False)
    df["batter_team"] = np.where(top_mask, df["away_team"], df["home_team"])
    df["pitcher_team"] = np.where(top_mask, df["home_team"], df["away_team"])

    desc = df["description"].fillna("").astype(str)
    events = df["events"].fillna("").astype(str)
    zone_num = pd.to_numeric(df["zone"], errors="coerce")

    df["is_swing"] = desc.isin(SWING_DESCRIPTIONS).fillna(False)
    df["is_contact"] = desc.isin(CONTACT_DESCRIPTIONS).fillna(False)
    df["is_whiff"] = desc.isin(WHIFF_DESCRIPTIONS).fillna(False)
    df["is_zone"] = zone_num.between(1, 9, inclusive="both").fillna(False)
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
    df["is_pull"] = ((stand.eq("R") & df["hc_x"].lt(112.0)) | (stand.eq("L") & df["hc_x"].gt(138.0))).fillna(False)
    df["is_pull_air"] = df["is_pull"] & df["is_air"]

    df["pa_key"] = df["game_pk"].astype("Int64").astype(str) + "-" + df["at_bat_number"].astype("Int64").astype(str)

    df["xba_value"] = df["estimated_ba_using_speedangle"].where(df["is_bbe"], 0.0)
    missing = df["is_bbe"] & df["xba_value"].isna()
    df.loc[missing, "xba_value"] = df.loc[missing, "is_hit"].astype(float)
    df["xba_value"] = df["xba_value"].fillna(0.0)

    df["xslg_value"] = df["estimated_slg_using_speedangle"].where(df["is_bbe"], 0.0)
    missing = df["is_bbe"] & df["xslg_value"].isna()
    actual = events.map({"single":1.0,"double":2.0,"triple":3.0,"home_run":4.0}).fillna(0.0)
    df.loc[missing, "xslg_value"] = actual[missing]
    df["xslg_value"] = df["xslg_value"].fillna(0.0)
    df["xiso_value"] = (df["xslg_value"] - df["xba_value"]).clip(lower=0.0)

    df = add_launch_angle_features(df)

    zone_text = zone_num.round().astype("Int64").astype(str)
    zone_mask = zone_num.between(1, 9, inclusive="both").fillna(False).to_numpy(bool)
    df["zone_group"] = np.where(zone_mask, zone_text, "Chase")
    return df

# NEW v2.1
def calculate_momentum_features(pa_df: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    if pa_df.empty:
        return pd.DataFrame(columns=["player_id", "Momentum_Score"])
    recent_start = end_date - pd.Timedelta(days=RECENT_FORM_DAYS - 1)
    medium_start = end_date - pd.Timedelta(days=LOOKBACK_MEDIUM - 1)

    recent = pa_df[pa_df["game_date"].between(recent_start, end_date)]
    medium = pa_df[pa_df["game_date"].between(medium_start, end_date - pd.Timedelta(days=RECENT_FORM_DAYS))]

    def rate(n, d): return safe_divide(n, d, 0.0)

    r = recent.groupby("batter").agg(Recent_PA=("pa_key","nunique"), Recent_HR=("is_hr","sum"), Recent_Barrels=("is_barrel","sum")).reset_index()
    m = medium.groupby("batter").agg(Medium_PA=("pa_key","nunique"), Medium_HR=("is_hr","sum"), Medium_Barrels=("is_barrel","sum")).reset_index()

    merged = r.merge(m, on="batter", how="outer").fillna(0)
    merged["Recent_HR_Rate"] = rate(merged["Recent_HR"], merged["Recent_PA"])
    merged["Medium_HR_Rate"] = rate(merged["Medium_HR"], merged["Medium_PA"])
    merged["Recent_Brl_Rate"] = rate(merged["Recent_Barrels"], merged["Recent_PA"])
    merged["Medium_Brl_Rate"] = rate(merged["Medium_Barrels"], merged["Medium_PA"])

    hr_m = (merged["Recent_HR_Rate"] - merged["Medium_HR_Rate"]) * 100
    brl_m = (merged["Recent_Brl_Rate"] - merged["Medium_Brl_Rate"]) * 80
    merged["Momentum_Score"] = (hr_m + brl_m).clip(-25, 35) + 50
    merged["Momentum_Score"] = merged["Momentum_Score"].fillna(50.0)
    return merged[["batter", "Momentum_Score"]].rename(columns={"batter": "player_id"})

def aggregate_hr_hitters(df: pd.DataFrame, pitcher_hand: str, min_pa: int) -> pd.DataFrame:
    pa = df[df["is_pa_end"]].copy()
    bbe = df[df["is_bbe"]].copy()

    pa_stats = pa.groupby(["batter", "batter_team"], dropna=False).agg(
        PA=("pa_key", "nunique"), HR=("is_hr", "sum"), Strikeouts=("is_k", "sum"),
        xSLG_Total=("xslg_value", "sum"), xISO_Total=("xiso_value", "sum"),
        Stand=("stand", lambda x: most_common(x, "?"))
    ).reset_index().rename(columns={"batter": "player_id", "batter_team": "Team"})

    pitch_stats = df.groupby(["batter", "batter_team"], dropna=False).agg(
        Pitches=("batter", "size"), Swings=("is_swing", "sum"),
        Contacts=("is_contact", "sum"), Whiffs=("is_whiff", "sum")
    ).reset_index().rename(columns={"batter": "player_id", "batter_team": "Team"})

    bbe_stats = bbe.groupby(["batter", "batter_team"], dropna=False).agg(
        BBE=("is_bbe", "sum"), Barrels=("is_barrel", "sum"), Hard_Hits=("is_hard_hit", "sum"),
        Sweet_Spots=("is_sweet_spot", "sum"), Barrel_Range=("Barrel_Range", "sum"),
        Fly_Balls=("is_fly_ball", "sum"), Air_Balls=("is_air", "sum"), Pull_Air=("is_pull_air", "sum"),
        Avg_EV=("launch_speed", "mean"), EV90=("launch_speed", quantile_90), Max_EV=("launch_speed", "max"),
        Avg_LA=("launch_angle", "mean"), LA_Power_Score=("LA_Power_Score", "mean"),
        xSLG_Contact=("estimated_slg_using_speedangle", "mean"),
        Launch_Window=("Launch_Window_Efficiency", "sum")
    ).reset_index().rename(columns={"batter": "player_id", "batter_team": "Team"})

    board = pa_stats.merge(pitch_stats, on=["player_id", "Team"], how="left")
    board = board.merge(bbe_stats, on=["player_id", "Team"], how="left")
    board = board[board["PA"].ge(min_pa)].copy()
    if board.empty: return board

    board["HR_PA"] = safe_divide(board["HR"], board["PA"])
    board["Brl_PA"] = safe_divide(board["Barrels"], board["PA"])
    board["Brl_BIP"] = safe_divide(board["Barrels"], board["BBE"])
    board["xSLG_PA"] = safe_divide(board["xSLG_Total"], board["PA"])
    board["xISO_PA"] = safe_divide(board["xISO_Total"], board["PA"])
    board["HH_Pct"] = safe_divide(board["Hard_Hits"], board["BBE"])
    board["SweetSpot_Pct"] = safe_divide(board["Sweet_Spots"], board["BBE"])
    board["Barrel_Range_BIP"] = safe_divide(board.get("Barrel_Range", 0), board["BBE"])
    board["FB_Pct"] = safe_divide(board["Fly_Balls"], board["BBE"])
    board["Air_Pct"] = safe_divide(board["Air_Balls"], board["BBE"])
    board["PullAir_BIP"] = safe_divide(board["Pull_Air"], board["BBE"])
    board["PullAir_Air"] = safe_divide(board["Pull_Air"], board["Air_Balls"])
    board["Contact_Pct"] = safe_divide(board["Contacts"], board["Swings"])
    board["Whiff_Pct"] = safe_divide(board["Whiffs"], board["Swings"])
    board["K_Pct"] = safe_divide(board["Strikeouts"], board["PA"])

    # NEW v2.1 Air Quality
    hard_air = bbe.groupby(["batter", "batter_team"]).apply(
        lambda x: safe_divide((x["is_hard_hit"] & x["is_air"]).sum(), x["is_air"].sum())
    ).reset_index()
    hard_air.columns = ["player_id", "Team", "HardHit_Air_Pct"]
    board = board.merge(hard_air, on=["player_id", "Team"], how="left")
    board["HardHit_Air_Pct"] = board["HardHit_Air_Pct"].fillna(0.25)
    board["Air_Quality_Score"] = (0.55 * board["HardHit_Air_Pct"] + 0.45 * board["PullAir_Air"].fillna(0.30)) * 100
    board["Air_Quality_Score"] = board["Air_Quality_Score"].clip(20, 95).fillna(50.0)

    board["LaunchWindow_Pct"] = safe_divide(board.get("Launch_Window", 0), board["BBE"])

    # Platoon
    split = df[df["p_throws"].eq(pitcher_hand) & df["is_pa_end"]].copy()
    split_stats = split.groupby("batter").agg(
        Platoon_PA=("pa_key", "nunique"), Platoon_HR=("is_hr", "sum"),
        Platoon_xSLG=("xslg_value", "sum"), Platoon_xISO=("xiso_value", "sum"),
        Platoon_Barrels=("is_barrel", "sum")
    ).reset_index().rename(columns={"batter": "player_id"})
    board = board.merge(split_stats, on="player_id", how="left")
    for c in ["Platoon_PA", "Platoon_HR", "Platoon_xSLG", "Platoon_xISO", "Platoon_Barrels"]:
        board[c] = pd.to_numeric(board.get(c, 0), errors="coerce").fillna(0)
    return board

def recent_hr_form(df: pd.DataFrame, player_ids: Iterable[int], end_date: pd.Timestamp) -> pd.DataFrame:
    start = end_date - pd.Timedelta(days=RECENT_DAYS - 1)
    recent = df[(df["batter"].isin(list(player_ids))) & (df["game_date"].between(start, end_date))]
    pa = recent[recent["is_pa_end"]]
    return pa.groupby("batter").agg(
        Recent_PA=("pa_key", "nunique"), Recent_HR=("is_hr", "sum"),
        Recent_Barrels=("is_barrel", "sum"), Recent_xSLG=("xslg_value", "sum"),
        Recent_xISO=("xiso_value", "sum")
    ).reset_index().rename(columns={"batter": "player_id"})

def pitcher_hr_splits(df: pd.DataFrame, pitcher_id: int) -> pd.DataFrame:
    pa = df[(df["pitcher"] == pitcher_id) & (df["is_pa_end"])]
    if pa.empty: return pd.DataFrame(columns=["Stand"])
    return pa.groupby("stand").agg(
        Pitcher_PA=("pa_key", "nunique"), Pitcher_HR=("is_hr", "sum"),
        Pitcher_Barrels=("is_barrel", "sum"), Pitcher_xSLG=("xslg_value", "sum"),
        Pitcher_xISO=("xiso_value", "sum")
    ).reset_index().rename(columns={"stand": "Stand"})

def _attack_read(score: float) -> str:
    if score >= 60: return "ATTACK"
    if score >= 54: return "Lean attack"
    if score > 46: return "Neutral"
    if score > 40: return "Lean avoid"
    return "AVOID"

def pitcher_attack_profile_hr(df: pd.DataFrame, pitcher_id: int) -> pd.DataFrame:
    pa_all = df[df["is_pa_end"]].copy()
    selected_pa = pa_all[pa_all["pitcher"] == pitcher_id].copy()
    bbe_all = df[df["is_bbe"]].copy()
    selected_bbe = bbe_all[bbe_all["pitcher"] == pitcher_id].copy()
    rows = []
    for side, label in [("L", "LHB"), ("R", "RHB")]:
        league_pa = pa_all[pa_all["stand"] == side]
        pitcher_pa = selected_pa[selected_pa["stand"] == side]
        league_bbe = bbe_all[bbe_all["stand"] == side]
        pitcher_bbe = selected_bbe[selected_bbe["stand"] == side]

        lg_pa_n = max(int(league_pa["pa_key"].nunique()), 1)
        p_pa_n = int(pitcher_pa["pa_key"].nunique())
        lg_bbe_n = max(len(league_bbe), 1)
        p_bbe_n = len(pitcher_bbe)

        lg_hr = float(league_pa["is_hr"].sum() / lg_pa_n)
        lg_barrel = float(league_pa["is_barrel"].sum() / lg_pa_n)
        lg_xslg = float(league_pa["xslg_value"].sum() / lg_pa_n)

        hr_rate = (float(pitcher_pa["is_hr"].sum()) + lg_hr * 220) / (p_pa_n + 220)
        barrel_rate = (float(pitcher_pa["is_barrel"].sum()) + lg_barrel * 180) / (p_pa_n + 180)
        xslg_rate = (float(pitcher_pa["xslg_value"].sum()) + lg_xslg * 180) / (p_pa_n + 180)

        vulnerability_ratio = (0.35 * (hr_rate / max(lg_hr, 0.0001)) +
                               0.35 * (barrel_rate / max(lg_barrel, 0.0001)) +
                               0.30 * (xslg_rate / max(lg_xslg, 0.0001)))
        attack_score = float(np.clip(50 + 90 * (vulnerability_ratio - 1.0), 0, 100))
        confidence_score = float(100 * (1 - np.exp(-p_pa_n / 180.0)))
        confidence = "High" if confidence_score >= 70 else "Medium" if confidence_score >= 45 else "Low"

        rows.append({
            "Side": side, "HitterSide": label, "PitcherSideAttackScore": attack_score,
            "PitcherSideRead": _attack_read(attack_score), "PitcherSidePA": p_pa_n,
            "Allowed_HR_PA": hr_rate, "Allowed_Brl_PA": barrel_rate,
            "Allowed_xSLG_PA": xslg_rate, "AttackConfidence": confidence
        })
    return pd.DataFrame(rows)

def render_pitcher_attack_panel(profile: pd.DataFrame, pitcher_name: str):
    st.subheader("Pitcher Attack / Avoid Map")
    if profile.empty:
        st.info("Not enough data for handedness map.")
        return
    best = profile.sort_values("PitcherSideAttackScore", ascending=False).iloc[0]
    lhb = profile[profile["Side"] == "L"].iloc[0]
    rhb = profile[profile["Side"] == "R"].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best side to target", best["HitterSide"], best["PitcherSideRead"])
    c2.metric("LHB read", f"{lhb['PitcherSideRead']} · {lhb['PitcherSideAttackScore']:.0f}")
    c3.metric("RHB read", f"{rhb['PitcherSideRead']} · {rhb['PitcherSideAttackScore']:.0f}")

def selected_pitcher_profile(df: pd.DataFrame, pitcher_id: int) -> tuple[pd.DataFrame, str]:
    rows = df[(df["pitcher"] == pitcher_id) & (df["pitch_name"].notna())].copy()
    if rows.empty: return pd.DataFrame(), ""
    hand = most_common(rows["p_throws"], "")
    profile = rows.groupby("pitch_name").agg(
        Pitches=("pitch_name", "size"), Avg_Speed=("release_speed", "mean"),
        Avg_PFX_X=("pfx_x", "mean"), Avg_PFX_Z=("pfx_z", "mean"),
        Avg_Extension=("release_extension", "mean"),
        Allowed_xSLG=("estimated_slg_using_speedangle", "mean"),
        Allowed_Barrel=("is_barrel", "mean")
    ).reset_index()
    profile["Usage"] = profile["Pitches"] / profile["Pitches"].sum()
    return profile[profile["Usage"].ge(0.03)].copy(), hand

def infer_recent_lineup(df: pd.DataFrame, team: str) -> pd.DataFrame:
    team_rows = df[(df["batter_team"] == team) & (df["game_date"].notna())].copy()
    if team_rows.empty: return pd.DataFrame(columns=["player_id", "LineupSpot"])
    latest_date = team_rows["game_date"].max()
    date_rows = team_rows[team_rows["game_date"] == latest_date]
    latest_game = pd.to_numeric(date_rows["game_pk"], errors="coerce").max()
    game_rows = date_rows[date_rows["game_pk"] == latest_game]
    order = game_rows.groupby("batter")["at_bat_number"].min().sort_values().reset_index().rename(columns={"batter": "player_id"})
    order["LineupSpot"] = np.arange(1, len(order) + 1)
    order.loc[order["LineupSpot"] > 9, "LineupSpot"] = np.nan
    return order[["player_id", "LineupSpot"]]

def add_roster_candidates(stats_board, team, active_roster, recent_lineup, min_pa, include_low_sample):
    stats = stats_board[stats_board["Team"] == team].copy()
    stats["player_id"] = pd.to_numeric(stats["player_id"], errors="coerce").astype("Int64")
    stats = stats.dropna(subset=["player_id"])
    stats["player_id"] = stats["player_id"].astype(int)

    lineup_ids = set(pd.to_numeric(recent_lineup.get("player_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())

    roster = active_roster.copy() if active_roster is not None else pd.DataFrame()
    if not roster.empty:
        roster["player_id"] = pd.to_numeric(roster["player_id"], errors="coerce").astype("Int64")
        roster = roster.dropna(subset=["player_id"]).copy()
        roster["player_id"] = roster["player_id"].astype(int)
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
    mapping = {1:4.72, 2:4.60, 3:4.49, 4:4.38, 5:4.26, 6:4.15, 7:4.05, 8:3.94, 9:3.84}
    base = pd.to_numeric(lineup_spot, errors="coerce").round().map(mapping).fillna(4.15)
    base += 0.10 * (team_runs - 4.5)
    base += 0.06 if is_away else -0.02
    return base.clip(3.2, 5.4)

def build_hr_board_v2(df, pitcher_id, team, min_pa, end_date, park_hr_factor_lhb, park_hr_factor_rhb,
                      team_runs, is_away, starter_innings, bullpen_multiplier, weather_multiplier,
                      lineup_override, lineup_edits, bat_tracking_upload, bat_tracking_auto,
                      active_roster, include_low_sample):

    profile, pitcher_hand = selected_pitcher_profile(df, pitcher_id)
    if not pitcher_hand:
        return pd.DataFrame(), profile

    recent_lineup = infer_recent_lineup(df, team)
    board = aggregate_hr_hitters(df, pitcher_hand, 0)
    board = add_roster_candidates(board, team, active_roster, recent_lineup, min_pa, include_low_sample)
    if board.empty: return board, profile

    names = lookup_names(tuple(board["player_id"].dropna().astype(int).unique().tolist()))
    board = board.merge(names, on="player_id", how="left", suffixes=("", "_Lookup"))
    board["Player"] = board["Player"].replace("", np.nan).fillna(board.get("Player_Lookup"))
    board["Player"] = board["Player"].fillna("MLB ID " + board["player_id"].astype("Int64").astype(str))
    board = board.drop(columns=["Player_Lookup"], errors="ignore")

    # Merge lineup information so LineupSpot exists
    board = board.merge(recent_lineup, on="player_id", how="left")
    board["LineupSpot"] = pd.to_numeric(board.get("LineupSpot"), errors="coerce").fillna(5)

    for col in ["PA", "HR", "Barrels", "Hard_Hits", "xSLG_Total", "xISO_Total"]:
        board[col] = pd.to_numeric(board.get(col, 0), errors="coerce").fillna(0)

    board["EffectiveStand"] = np.where(board["Stand"].eq("S"), "L" if pitcher_hand == "R" else "R", board["Stand"])
    board["EffectiveStand"] = board["EffectiveStand"].where(board["EffectiveStand"].isin(["L", "R"]), "R")

    recent = recent_hr_form(df, board["player_id"], end_date)
    board = board.merge(recent, on="player_id", how="left")
    for col in ["Recent_PA", "Recent_HR", "Recent_Barrels", "Recent_xSLG", "Recent_xISO"]:
        board[col] = pd.to_numeric(board.get(col, 0), errors="coerce").fillna(0)

    # Create rate columns that are used later
    board["Recent_HR_PA"] = safe_divide(board["Recent_HR"], board["Recent_PA"])

    pitcher_splits = pitcher_hr_splits(df, pitcher_id)
    board = board.merge(pitcher_splits, left_on="EffectiveStand", right_on="Stand", how="left", suffixes=("", "_Pitcher"))
    for col in ["Pitcher_PA", "Pitcher_HR", "Pitcher_Barrels", "Pitcher_xSLG", "Pitcher_xISO"]:
        board[col] = pd.to_numeric(board.get(col, 0), errors="coerce").fillna(0)

    board["Pitcher_HR_PA"] = safe_divide(board["Pitcher_HR"], board["Pitcher_PA"])

    side_attack = pitcher_attack_profile_hr(df, pitcher_id)
    board = board.merge(side_attack[["Side", "PitcherSideAttackScore", "PitcherSideRead"]], left_on="EffectiveStand", right_on="Side", how="left")
    board["PitcherSideAttackScore"] = board["PitcherSideAttackScore"].fillna(50.0)
    board["PitcherSideRead"] = board["PitcherSideRead"].fillna("Neutral")

    # NEW v2.1 Momentum
    pa_all = df[df["is_pa_end"]].copy()
    momentum = calculate_momentum_features(pa_all, end_date)
    board = board.merge(momentum, on="player_id", how="left")
    board["Momentum_Score"] = board["Momentum_Score"].fillna(50.0)

    # Recent pitcher vulnerability
    recent_start = end_date - pd.Timedelta(days=RECENT_DAYS - 1)
    recent_pitcher_pa = df[(df["pitcher"] == pitcher_id) & (df["game_date"].between(recent_start, end_date)) & (df["is_pa_end"])]
    if not recent_pitcher_pa.empty:
        recent_vuln = recent_pitcher_pa.groupby("stand").agg(Recent_Pitcher_HR=("is_hr","sum"), Recent_Pitcher_PA=("pa_key","nunique")).reset_index()
        recent_vuln["Recent_Pitcher_HR_PA"] = safe_divide(recent_vuln["Recent_Pitcher_HR"], recent_vuln["Recent_Pitcher_PA"])
        board = board.merge(recent_vuln[["stand", "Recent_Pitcher_HR_PA"]], left_on="EffectiveStand", right_on="stand", how="left")
        board["Recent_Pitcher_Vuln"] = board["Recent_Pitcher_HR_PA"].fillna(0.0)
    else:
        board["Recent_Pitcher_Vuln"] = 0.0

    league_pa = max(int(pa_all["pa_key"].nunique()), 1)
    league_hr_rate = float(pa_all["is_hr"].sum() / league_pa)

    board["Adj_HR_PA"] = shrink_rate(board["HR"], board["PA"], league_hr_rate, 180)
    board["Adj_Brl_PA"] = shrink_rate(board["Barrels"], board["PA"], league_hr_rate * 12, 140)  # proxy
    board["Adj_xSLG_PA"] = shrink_rate(board["xSLG_Total"], board["PA"], league_hr_rate * 4.2, 140)

    # Base rate from original-style weighting + NEW factors
    base = (board["Adj_HR_PA"] * 0.35 + board["Adj_Brl_PA"] * 0.25 + board["Adj_xSLG_PA"] * 0.20 +
            board["Pitcher_HR_PA"] * 0.12 + board["Recent_HR_PA"] * 0.08)

    # NEW v2.1 boosts
    air_boost = (board["Air_Quality_Score"] - 50) / 900
    mom_boost = (board["Momentum_Score"] - 50) / 700
    recent_vuln_boost = (board["Recent_Pitcher_Vuln"] - league_hr_rate) * 10

    final_rate = base * 0.80 + air_boost * 0.07 + mom_boost * 0.06 + recent_vuln_boost * 0.07

    board["ParkFactor"] = np.where(board["EffectiveStand"].eq("L"), float(park_hr_factor_lhb), float(park_hr_factor_rhb)) / 100.0
    final_rate = final_rate * board["ParkFactor"] * float(weather_multiplier)

    board["Model_HR_Per_PA"] = final_rate.clip(0.001, 0.18)
    board["Projected_PA"] = projected_pa(board["LineupSpot"], team_runs, is_away)
    board["Model_1plus_HR"] = 1 - (1 - board["Model_HR_Per_PA"]) ** board["Projected_PA"]

    # NEW transparent xHR
    board["xHR_Estimate"] = (board["Barrels"] * 0.58 + board["Hard_Hits"] * board["Air_Pct"].fillna(0.3) * 0.09 + board["xISO_Total"] * 0.012) / board["PA"].replace(0, 1)
    board["xHR_Estimate"] = (board["xHR_Estimate"] * board["ParkFactor"] * float(weather_multiplier)).clip(0.001, 0.16)

    board["HRScore"] = (percentile(board["Model_1plus_HR"]) * 0.25 +
                        percentile(board["Air_Quality_Score"]) * 0.13 +
                        percentile(board["Momentum_Score"]) * 0.11 +
                        percentile(board["LA_Power_Score"]) * 0.14 +
                        percentile(board["PitcherSideAttackScore"]) * 0.12 +
                        percentile(board.get("PitchMatchScore", pd.Series(50, index=board.index))) * 0.15 +
                        percentile(board.get("ZoneFitScore", pd.Series(50, index=board.index))) * 0.10).clip(0, 100)

    board["Confidence"] = (100 * (1 - np.exp(-board["PA"] / 180.0))).clip(20, 95)
    board["Confidence_Level"] = pd.cut(board["Confidence"], bins=[-np.inf, 45, 70, np.inf], labels=["Low", "Medium", "High"]).astype(str)

    board = board.sort_values(["Model_1plus_HR", "HRScore"], ascending=False).reset_index(drop=True)
    board.insert(0, "Rank", np.arange(1, len(board) + 1))
    board["Tier"] = [hr_target_tier(p, s) for p, s in zip(board["Model_1plus_HR"], board["HRScore"])]
    return board, profile

def hr_target_tier(prob, score):
    if pd.isna(prob) or pd.isna(score): return "Monitor"
    if prob >= 0.20 and score >= 72: return "🔥 BEST BET"
    if prob >= 0.16 and score >= 65: return "✅ STRONG"
    if prob >= 0.13 and score >= 55: return "📈 LEAN"
    if prob < 0.10 or score < 42: return "❌ AVOID"
    return "⚠️ MONITOR"

# =============================================================================
# MODERN UI
# =============================================================================

def inject_modern_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap');
    :root { --bg:#0F172A; --card:#1E293B; --accent:#14B8A6; --highlight:#F97316; }
    .stApp { background: linear-gradient(145deg, #0F172A 0%, #1E1135 100%); color: #F8FAFC; font-family: 'Inter', system-ui, sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', 'Inter', sans-serif; font-weight: 700; letter-spacing: -0.025em; }
    .hero-banner { background: linear-gradient(90deg, #1E293B 0%, #334155 100%); border: 1px solid rgba(148,163,184,0.2); border-radius: 24px; padding: 1.75rem 2rem; margin-bottom: 1.5rem; box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1); }
    .metric-card { background: var(--card); border: 1px solid rgba(148,163,184,0.15); border-radius: 20px; padding: 1.25rem 1.5rem; transition: transform .2s cubic-bezier(0.4,0,0.2,1); }
    .metric-card:hover { transform: translateY(-4px); }
    .tier-badge { display: inline-flex; align-items: center; padding: 0.35rem 0.9rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px; }
    .insight-bullet { background: rgba(20,184,166,0.08); border-left: 4px solid #14B8A6; padding: 0.6rem 1rem; margin: 0.4rem 0; border-radius: 0 12px 12px 0; }
    </style>
    """, unsafe_allow_html=True)

def render_hero(matchup, weather_mult, park_l, park_r):
    st.markdown(f"""
    <div class="hero-banner">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:2rem;flex-wrap:wrap;">
            <div>
                <div style="font-size:2.1rem;font-weight:800;color:white;">{matchup.get('away_abbr','AWAY')} <span style="color:#64748B">@</span> {matchup.get('home_abbr','HOME')}</div>
                <div style="color:#94A3B8;font-size:1.05rem;margin-top:0.35rem;">{matchup.get('venue','')} • {matchup.get('status','')}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.95rem;color:#64748B;">PROBABLE PITCHER</div>
                <div style="font-size:1.35rem;font-weight:700;color:white;">{matchup.get('pitcher_name','')}</div>
                <div style="margin-top:0.6rem;display:flex;gap:1.5rem;justify-content:flex-end;font-size:0.9rem;">
                    <div><span style="color:#64748B">Park LHB</span><br><span style="font-weight:700;color:#14B8A6">{park_l:.0f}</span></div>
                    <div><span style="color:#64748B">Park RHB</span><br><span style="font-weight:700;color:#14B8A6">{park_r:.0f}</span></div>
                    <div><span style="color:#64748B">Weather ×</span><br><span style="font-weight:700;color:#F97316">{weather_mult:.2f}</span></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_leader_cards(rankings):
    top = rankings.head(5)
    cols = st.columns(5)
    for i, (_, row) in enumerate(top.iterrows()):
        with cols[i]:
            tier = row.get("Tier", "⚠️ MONITOR")
            color = "#10B981" if "BEST" in tier or "STRONG" in tier else "#F59E0B" if "LEAN" in tier else "#EF4444"
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;height:100%">
                <div style="font-size:0.75rem;color:#64748B;font-weight:600">#{int(row['Rank'])}</div>
                <div style="font-size:1.1rem;font-weight:800;margin:0.3rem 0 0.1rem;line-height:1.1">{row['Player']}</div>
                <div style="font-size:2.3rem;font-weight:900;color:#14B8A6;line-height:1">{row['Model_1plus_HR']:.1%}</div>
                <div style="margin:0.4rem 0"><span class="tier-badge" style="background:{color}20;color:{color}">{tier}</span></div>
                <div style="font-size:0.85rem;color:#94A3B8">HR Score <span style="color:white;font-weight:700">{row['HRScore']:.0f}</span></div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================

def main():
    inject_modern_css()
    st.title("HR Edge")
    st.caption(f"Advanced MLB Home Run Targeting • v{MODEL_VERSION}")

    with st.sidebar:
        st.header("Controls")
        end_date = st.date_input("Stats through", value=date.today() - timedelta(days=1))
        lookback = st.slider("Lookback days", 30, 120, 60)
        min_pa = st.slider("Min PA", 15, 80, 30)
        include_low = st.checkbox("Include low-sample hitters", value=True)

        st.divider()
        st.subheader("Live Filters")
        min_prob = st.slider("Min HR Prob", 0.0, 0.35, 0.08, format="%.0f%%")
        min_score = st.slider("Min HR Score", 0, 100, 42)
        min_conf = st.slider("Min Confidence", 0, 100, 35)
        search = st.text_input("Search player")

        if st.button("Load / Refresh Statcast", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    start_date = end_date - timedelta(days=lookback - 1)
    with st.spinner(f"Loading Statcast {start_date} → {end_date}..."):
        raw = load_statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        df = prepare_data(raw)

    if df.empty:
        st.error("No data. Try different dates.")
        st.stop()

    pitcher_summary = df.groupby("pitcher").agg(
        Pitches=("pitcher", "size"), Player_Name=("player_name", "first"),
        Pitcher_Team=("pitcher_team", "last"), Hand=("p_throws", lambda x: most_common(x, "?"))
    ).reset_index()
    pitcher_summary = pitcher_summary[pitcher_summary["Pitches"] >= 80].copy()
    pitcher_summary["Display"] = pitcher_summary["Player_Name"].fillna("Unknown") + " — " + pitcher_summary["Pitcher_Team"].fillna("?")
    pitcher_summary = pitcher_summary.sort_values("Display")

    st.subheader("Select Matchup")
    c1, c2 = st.columns([2,1])
    with c1:
        selected = st.selectbox("Starting Pitcher", pitcher_summary["Display"].tolist())
    with c2:
        batting_team = st.selectbox("Batting Team", sorted(df["batter_team"].dropna().unique().tolist()))

    pitcher_row = pitcher_summary[pitcher_summary["Display"] == selected].iloc[0]
    pitcher_id = int(pitcher_row["pitcher"])

    # Build board with new v2.1 model
    rankings, _ = build_hr_board_v2(
        df=df, pitcher_id=pitcher_id, team=batting_team, min_pa=min_pa,
        end_date=pd.Timestamp(end_date), park_hr_factor_lhb=102.0, park_hr_factor_rhb=98.0,
        team_runs=4.6, is_away=True, starter_innings=5.8, bullpen_multiplier=1.02,
        weather_multiplier=1.05, lineup_override=None, lineup_edits=None,
        bat_tracking_upload=None, bat_tracking_auto=None, active_roster=None,
        include_low_sample=include_low
    )

    if rankings.empty:
        st.warning("No qualified hitters.")
        st.stop()

    filtered = rankings[(rankings["Model_1plus_HR"] >= min_prob) & (rankings["HRScore"] >= min_score) & (rankings["Confidence"] >= min_conf)].copy()
    if search:
        filtered = filtered[filtered["Player"].str.contains(search, case=False, na=False)]

    render_hero({"away_abbr": "NYY", "home_abbr": batting_team, "venue": "Stadium", "pitcher_name": selected.split(" — ")[0], "status": "Today's Game"},
                1.05, 102, 98)

    st.markdown("### Top Targets")
    render_leader_cards(filtered)

    with st.expander("Why these hitters?", expanded=True):
        for _, row in filtered.head(3).iterrows():
            reasons = []
            if row.get("Air_Quality_Score", 50) > 68: reasons.append("elite air contact")
            if row.get("Momentum_Score", 50) > 62: reasons.append("heating up recently")
            if row.get("LaunchWindow_Pct", 0) > 0.18: reasons.append("high % in optimal HR window")
            text = " + ".join(reasons) if reasons else "solid all-around profile"
            st.markdown(f"<div class='insight-bullet'><b>{row['Player']}</b>: {text}</div>", unsafe_allow_html=True)

    st.markdown("### Full Board")
    cols = ["Rank", "Player", "Tier", "Model_1plus_HR", "HRScore", "Confidence_Level", "Air_Quality_Score", "Momentum_Score", "xHR_Estimate"]
    show = filtered[[c for c in cols if c in filtered.columns]].copy()
    st.dataframe(show.style.background_gradient(subset=["HRScore", "Air_Quality_Score", "Momentum_Score"], cmap="RdYlGn", vmin=30, vmax=95)
                 .format({"Model_1plus_HR": "{:.1%}", "xHR_Estimate": "{:.2%}"}), use_container_width=True, hide_index=True, height=520)

    with st.expander("What-If Simulator"):
        t, w, p = st.columns(3)
        temp = t.slider("Temp adjustment (°F)", -15, 20, 0)
        wind = w.slider("Wind carry", -0.08, 0.12, 0.0, 0.01)
        park = p.slider("Park factor", 85, 125, 100)
        if st.button("Apply to Top Hitter"):
            top = filtered.iloc[0]
            new_mult = 1.0 + (temp * 0.002) + wind
            new_p = top["Model_1plus_HR"] * (park / 100) * new_mult
            st.success(f"Adjusted probability for {top['Player']}: **{new_p:.1%}**")

    st.caption(f"v{MODEL_VERSION} • Data through {end_date}")

if __name__ == "__main__":
    main()
