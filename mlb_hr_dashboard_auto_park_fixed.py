from __future__ import annotations

from datetime import date, timedelta
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

def weather_carry_multiplier(
    temperature_f: float,
    humidity_pct: float,
    wind_mph: float,
    wind_direction: str,
    manual_multiplier: float,
) -> float:
    temperature_factor = 1.0 + (temperature_f - 70.0) * 0.0025
    humidity_factor = 1.0 + (humidity_pct - 50.0) * 0.0003
    if wind_direction == "Out":
        wind_factor = 1.0 + wind_mph * 0.008
    elif wind_direction == "In":
        wind_factor = 1.0 - wind_mph * 0.008
    else:
        wind_factor = 1.0
    return float(np.clip(temperature_factor * humidity_factor * wind_factor * manual_multiplier, 0.72, 1.32))


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
    lineup_edits: pd.DataFrame | None,
    bat_tracking_upload,
    bat_tracking_auto: pd.DataFrame | None,
    active_roster: pd.DataFrame | None,
    include_low_sample: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile, pitcher_hand = selected_pitcher_profile(df, pitcher_id)
    if not pitcher_hand:
        return pd.DataFrame(), profile

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



def _factor_from_frame(frame: pd.DataFrame, venue: str, metric: str) -> float | None:
    if frame is None or frame.empty:
        return None
    table = _flatten_columns(frame)
    cols = {c: normalize_name(c) for c in table.columns}
    venue_cols = [c for c,n in cols.items() if 'venue' in n or 'stadium' in n or n in {'park','name','parkname'}]
    wanted = {'hits': {'hits','indexhits','hitfactor','hitsfactor'}, 'hr': {'hr','indexhr','homerun','homeruns','hrfactor'}}.get(normalize_name(metric), {normalize_name(metric)})
    metric_cols = [c for c,n in cols.items() if n in wanted or any(w in n for w in wanted)]
    if venue_cols and not metric_cols:
        metric_cols = [c for c,n in cols.items() if n in {'value','index','factor','parkfactor'}]
    for _, row in table.iterrows():
        if not any(venue_names_match(row.get(c), venue) for c in venue_cols):
            continue
        for c in metric_cols:
            value = _numeric_park_factor(row.get(c))
            if value is not None:
                return value
    return None


def _parse_savant_payload(text: str, content_type: str, venue: str, metric: str) -> float | None:
    frames = []
    s = text.lstrip()
    if 'json' in content_type.lower() or s.startswith(('{','[')):
        try:
            obj = json.loads(text)
            candidates = [obj]
            if isinstance(obj, dict):
                candidates += [obj.get(k) for k in ('data','results','rows','leaderboard','items')]
            for value in candidates:
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    frames.append(pd.DataFrame(value))
        except Exception:
            pass
    if 'csv' in content_type.lower() or (',' in text[:500] and '\n' in text[:2000]):
        try:
            frames.append(pd.read_csv(StringIO(text)))
        except Exception:
            pass
    if '<table' in text.lower() or '<html' in text.lower():
        try:
            frames.extend(pd.read_html(StringIO(text)))
        except Exception:
            pass
    for frame in frames:
        value = _factor_from_frame(frame, venue, metric)
        if value is not None:
            return value
    return None


def _get_savant(url: str) -> tuple[str, str]:
    req = Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36',
        'Accept': 'application/json,text/csv,text/html,*/*',
        'Referer': 'https://baseballsavant.mlb.com/leaderboard/statcast-park-factors',
    })
    with urlopen(req, timeout=15) as response:
        return response.read().decode('utf-8-sig', errors='replace'), response.headers.get('Content-Type','')


def _fetch_savant_side_factor(venue: str, year: int, side: str, metric: str) -> tuple[float | None, str | None]:
    metric_key = 'Hits' if normalize_name(metric) == 'hits' else 'HR'
    paths = [
        'https://baseballsavant.mlb.com/leaderboard/services/statcast-park-factors',
        'https://baseballsavant.mlb.com/leaderboard/services/park-factors',
        'https://baseballsavant.mlb.com/leaderboard/statcast-park-factors',
    ]
    errors = []
    for query_year in (int(year), int(year)-1):
        for stat_value in (f'index_{metric_key}', metric_key, f'index_{metric_key.lower()}', metric_key.lower()):
            params = {'year': query_year, 'type': 'year', 'batSide': side, 'stat': stat_value, 'condition': 'All', 'rolling': 3, 'csv': 'true'}
            for path in paths:
                try:
                    text, ctype = _get_savant(path + '?' + urlencode(params))
                    value = _parse_savant_payload(text, ctype, venue, metric_key)
                    if value is not None:
                        return value, f'Baseball Savant {query_year} · 3-year · {side}HB'
                except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
                    errors.append(f'{path.rsplit("/",1)[-1]}: {type(exc).__name__}: {exc}')
    return None, ' | '.join(list(dict.fromkeys(errors))[-5:]) or 'No matching Savant venue row was returned.'


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_savant_park_factors(venue: str, year: int, metric: str) -> dict:
    if not venue or normalize_name(venue) in {'manualmatchup','venuetbd'}:
        return {'L':100.0,'R':100.0,'ok':False,'source':'Neutral fallback','error':'A real MLB venue was not available.'}
    results = {side: _fetch_savant_side_factor(venue, int(year), side, metric) for side in ('L','R')}
    left, lmsg = results['L']; right, rmsg = results['R']
    good = [m for v,m in results.values() if v is not None and m]
    bad = [f'{s}: {m}' for s,(v,m) in results.items() if v is None and m]
    return {
        'L': float(left) if left is not None else 100.0,
        'R': float(right) if right is not None else 100.0,
        'ok': left is not None and right is not None,
        'source': good[0] if good else 'Neutral fallback',
        'error': ' || '.join(bad) if bad else None,
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

inject_clean_css()
MODEL_KEY = "clean_hr"

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
            help="Leave this off to use the automatic Baseball Savant 3-year factors.",
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
                st.warning("Automatic Savant factors were unavailable, so neutral 100 values are being used. The app now tries Savant's data service and CSV output first.")
                if auto_park.get("error"):
                    with st.expander("Park-factor connection details"):
                        st.code(str(auto_park["error"]))
        if st.button("Refresh park factors", key=f"clean_hr_refresh_park_{matchup.get('game_pk')}"):
            fetch_savant_park_factors.clear()
            st.rerun()
        temperature_f = st.number_input(
            "Temperature °F", 35.0, 110.0, 75.0, 1.0, key="clean_hr_temp"
        )
    with c3:
        humidity_pct = st.number_input(
            "Humidity %", 5.0, 100.0, 50.0, 1.0, key="clean_hr_humidity"
        )
        wind_mph = st.number_input(
            "Wind mph", 0.0, 30.0, 5.0, 1.0, key="clean_hr_wind"
        )
        wind_direction = st.selectbox(
            "Wind direction", ["Cross/Calm", "Out", "In"], key="clean_hr_wind_dir"
        )
        manual_weather = st.number_input(
            "Manual weather multiplier",
            0.80,
            1.20,
            1.00,
            0.01,
            key="clean_hr_weather_manual",
        )
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
    temperature_f, humidity_pct, wind_mph, wind_direction, manual_weather
)
st.caption(
    f"Weather carry adjustment: ×{weather_multiplier:.3f}. "
    "This is a simple transparent adjustment, not an official ball-flight model."
)

preview_profile, preview_hand = selected_pitcher_profile(df, selected_pitcher)
preview_recent_lineup = infer_recent_lineup(df, selected_team)
preview_board = aggregate_hr_hitters(df, preview_hand, 0)
preview_board = add_roster_candidates(
    preview_board, selected_team, active_roster, preview_recent_lineup, min_pa, include_low_sample
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
preview_board = preview_board.merge(preview_recent_lineup, on="player_id", how="left")
lineup_input = preview_board[["player_id", "Player", "Position", "Bats", "PA", "SampleStatus", "LineupSpot"]].copy()
default_selected = lineup_input["LineupSpot"].notna()
if not default_selected.any():
    default_selected = pd.Series(True, index=lineup_input.index)
lineup_input.insert(0, "Selected", default_selected)
lineup_input = lineup_input.sort_values(["LineupSpot", "Player"], na_position="last")

with st.expander("Confirm lineup", expanded=False):
    st.caption(
        "The default order is inferred from the team's latest game in the loaded sample. "
        "Update it when the confirmed lineup is available."
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
        key=f"clean_hr_lineup_{selected_pitcher}_{selected_team}",
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
    lineup_edits=lineup_edits,
    bat_tracking_upload=bat_tracking_upload,
    bat_tracking_auto=bat_tracking_auto,
    active_roster=active_roster,
    include_low_sample=include_low_sample,
)

if rankings.empty:
    st.warning("No selected hitters remain after filtering.")
    st.stop()

render_board_header(matchup, selected_team, matchup["pitcher_name"], "ADVANCED HOME RUN BOARD")
render_leader_cards(rankings, "Model_1plus_HR", "HRScore", "model 1+ HR")

quick_tab, power_tab, matchup_tab, tracking_tab, pitcher_tab, notes_tab = st.tabs(
    ["Quick board", "Power profile", "Matchup detail", "Bat tracking", "Pitcher profile", "Model notes"]
)

with quick_tab:
    quick_columns = [
        "Rank", "Player", "LineupSpot", "Projected_PA", "Model_1plus_HR",
        "HRScore", "Confidence_Level", "Adj_Brl_PA", "Adj_xISO_PA",
        "EffectiveStand", "PitcherSideRead", "PitcherSideAttackScore", "SampleStatus",
        "PitchMatchScore", "ZoneFitScore", "PitcherPowerScore", "ParkFactor",
    ]
    quick = rankings[[column for column in quick_columns if column in rankings.columns]].copy()
    quick = quick.rename(columns={
        "LineupSpot": "Order", "Projected_PA": "Proj PA", "Model_1plus_HR": "1+ HR",
        "HRScore": "HR Score", "Confidence_Level": "Confidence",
        "Adj_Brl_PA": "Adj Brl/PA", "Adj_xISO_PA": "Adj xISO/PA",
        "EffectiveStand": "Bats vs SP", "PitcherSideRead": "Pitcher Read",
        "PitcherSideAttackScore": "Side Attack", "SampleStatus": "Sample",
        "PitchMatchScore": "Pitch Match", "ZoneFitScore": "Zone Fit",
        "PitcherPowerScore": "Pitcher Power", "ParkFactor": "Park Factor",
    })
    score_cols = ["HR Score", "Side Attack", "Pitch Match", "Zone Fit", "Pitcher Power"]
    styler = quick.style.background_gradient(
        cmap="RdYlGn", subset=score_cols, vmin=0, vmax=100
    ).format({
        "Order": "{:.0f}", "Proj PA": "{:.2f}", "1+ HR": "{:.1%}",
        "HR Score": "{:.1f}", "Adj Brl/PA": "{:.2%}", "Adj xISO/PA": "{:.3f}",
        "Side Attack": "{:.1f}", "Pitch Match": "{:.1f}", "Zone Fit": "{:.1f}", "Pitcher Power": "{:.1f}",
        "Park Factor": "{:.0f}",
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

with notes_tab:
    st.markdown(
        """
        ### Reading the board
        - **1+ HR** combines the modeled home-run rate per plate appearance with projected plate appearances.
        - **HR Score** is a 0–100 comparison score within the selected offense, not a literal probability.
        - **Pitcher Read / Side Attack** grades whether the starter has been more attackable or avoidable for LHB or RHB, with small samples shrunk toward neutral.
        - **Pitch Match** compares the hitter with the starter's pitch types, velocity, movement and extension.
        - **Zone Fit** weights hitter power by the locations the starter uses.
        - **Park Factor** is automatically matched to the selected venue from Baseball Savant when its live table is available; 110 means ×1.10 and 90 means ×0.90. A neutral/manual fallback remains available.
        - The weather multiplier is applied after the park adjustment.
        - **Sample** marks hitters below the selected PA threshold. Active-roster players with no history use league-average priors and remain Low confidence.
        - **Confidence** reflects sample size and matchup-data depth, not certainty that the outcome will occur.

        The probabilities remain heuristic until tested and calibrated on held-out historical games.
        """
    )
