from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
from pybaseball import cache, playerid_reverse_lookup, statcast


st.set_page_config(page_title="MLB Hitter Fantasy Score Lab", layout="wide")
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
# Hitter Fantasy Score model
# -----------------------------------------------------------------------------
import hashlib
import math

FANTASY_EVENT_NAMES = ["Single", "Double", "Triple", "HomeRun", "Walk", "HBP", "Strikeout"]
DEFAULT_PP_SCORING = {
    "Single": 3.0,
    "Double": 5.0,
    "Triple": 8.0,
    "HomeRun": 10.0,
    "Run": 2.0,
    "RBI": 2.0,
    "Walk": 2.0,
    "HBP": 2.0,
    "StolenBase": 5.0,
}


def add_fantasy_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    events = out.get("events", pd.Series("", index=out.index)).fillna("").astype(str)
    out["fs_single"] = events.eq("single")
    out["fs_double"] = events.eq("double")
    out["fs_triple"] = events.eq("triple")
    out["fs_hr"] = events.eq("home_run")
    out["fs_walk"] = events.isin({"walk", "intent_walk"})
    out["fs_hbp"] = events.eq("hit_by_pitch")
    out["fs_k"] = events.isin(STRIKEOUT_EVENTS)
    out["fs_reach"] = (
        out["fs_single"] | out["fs_double"] | out["fs_triple"]
        | out["fs_hr"] | out["fs_walk"] | out["fs_hbp"]
    )
    return out


def _aggregate_event_counts(frame: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    columns = [
        group_col, f"{prefix}_PA", f"{prefix}_Single", f"{prefix}_Double",
        f"{prefix}_Triple", f"{prefix}_HR", f"{prefix}_Walk", f"{prefix}_HBP",
        f"{prefix}_K", f"{prefix}_xH", f"{prefix}_xSLG",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    pa = frame[frame["is_pa_end"]].copy()
    if pa.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        pa.groupby(group_col)
        .agg(
            **{
                f"{prefix}_PA": ("pa_key", "nunique"),
                f"{prefix}_Single": ("fs_single", "sum"),
                f"{prefix}_Double": ("fs_double", "sum"),
                f"{prefix}_Triple": ("fs_triple", "sum"),
                f"{prefix}_HR": ("fs_hr", "sum"),
                f"{prefix}_Walk": ("fs_walk", "sum"),
                f"{prefix}_HBP": ("fs_hbp", "sum"),
                f"{prefix}_K": ("fs_k", "sum"),
                f"{prefix}_xH": ("xba_value", "sum"),
                f"{prefix}_xSLG": ("xslg_value", "sum"),
            }
        )
        .reset_index()
    )
    return grouped


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_mlb_hitting_season_stats(player_ids: tuple[int, ...], season: int) -> pd.DataFrame:
    """Fetch season counting stats used for run, RBI and stolen-base baselines."""
    columns = [
        "player_id", "Season_PA", "Season_R", "Season_RBI", "Season_SB",
        "Season_CS", "Season_BB", "Season_HBP", "Season_H", "Season_2B",
        "Season_3B", "Season_HR",
    ]
    ids = sorted({int(value) for value in player_ids if pd.notna(value)})
    if not ids:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for start in range(0, len(ids), 50):
        batch = ids[start:start + 50]
        params = {
            "stats": "season",
            "group": "hitting",
            "season": int(season),
            "playerIds": ",".join(str(value) for value in batch),
            "hydrate": "person",
        }
        url = "https://statsapi.mlb.com/api/v1/stats?" + urlencode(params)
        request = Request(
            url,
            headers={"User-Agent": "MLB-Hitter-Fantasy-Lab/1.0", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=15) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError):
            continue

        for block in payload.get("stats", []):
            for split in block.get("splits", []):
                player = split.get("player", {}) or split.get("person", {}) or {}
                stat = split.get("stat", {}) or {}
                player_id = player.get("id")
                if player_id is None:
                    continue
                rows.append(
                    {
                        "player_id": int(player_id),
                        "Season_PA": stat.get("plateAppearances", 0),
                        "Season_R": stat.get("runs", 0),
                        "Season_RBI": stat.get("rbi", 0),
                        "Season_SB": stat.get("stolenBases", 0),
                        "Season_CS": stat.get("caughtStealing", 0),
                        "Season_BB": stat.get("baseOnBalls", 0),
                        "Season_HBP": stat.get("hitByPitch", 0),
                        "Season_H": stat.get("hits", 0),
                        "Season_2B": stat.get("doubles", 0),
                        "Season_3B": stat.get("triples", 0),
                        "Season_HR": stat.get("homeRuns", 0),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=columns)
    output = pd.DataFrame(rows)
    for column in columns:
        if column not in output.columns:
            output[column] = 0
    output = output.groupby("player_id", as_index=False).last()
    for column in columns[1:]:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    return output[columns]


def build_fantasy_bvp_table(
    df: pd.DataFrame,
    rankings: pd.DataFrame,
    pitcher_id: int,
    pitcher_name: str,
    team: str,
    opponent: str,
) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()
    player_ids = rankings["player_id"].dropna().astype(int).tolist()
    rows = df[df["batter"].isin(player_ids) & df["pitcher"].eq(int(pitcher_id))].copy()
    if rows.empty:
        return pd.DataFrame()
    pa = rows[rows["is_pa_end"]].copy()
    if pa.empty:
        return pd.DataFrame()
    stats = (
        pa.groupby("batter")
        .agg(
            PA=("pa_key", "nunique"),
            H=("is_hit", "sum"),
            HR=("is_hr", "sum"),
            BB=("fs_walk", "sum"),
            HBP=("fs_hbp", "sum"),
            K=("fs_k", "sum"),
            xH=("xba_value", "sum"),
            xSLG=("xslg_value", "sum"),
            Last_Date=("game_date", "max"),
        )
        .reset_index()
        .rename(columns={"batter": "player_id"})
    )
    bbe = rows[rows["is_bbe"]].groupby("batter").agg(
        BBE=("is_bbe", "sum"), Avg_EV=("launch_speed", "mean"),
        Hard_Hit=("is_hard_hit", "sum"), Barrels=("is_barrel", "sum"),
    ).reset_index().rename(columns={"batter": "player_id"})
    stats = stats.merge(bbe, on="player_id", how="left")
    stats = stats.merge(rankings[["player_id", "Player"]], on="player_id", how="left")
    stats["H/PA"] = safe_divide(stats["H"], stats["PA"], 0.0)
    stats["xH/PA"] = safe_divide(stats["xH"], stats["PA"], 0.0)
    stats["HR/PA"] = safe_divide(stats["HR"], stats["PA"], 0.0)
    stats["K%"] = safe_divide(stats["K"], stats["PA"], 0.0)
    stats["Hard Hit%"] = safe_divide(stats["Hard_Hit"], stats["BBE"], 0.0)
    stats["Brl/BIP"] = safe_divide(stats["Barrels"], stats["BBE"], 0.0)
    reliability = 1.0 - np.exp(-pd.to_numeric(stats["PA"], errors="coerce").fillna(0) / 18.0)
    raw = (
        percentile(stats["H/PA"]) * 0.30
        + percentile(stats["xH/PA"]) * 0.20
        + percentile(stats["HR/PA"]) * 0.20
        + percentile(stats["Hard Hit%"]) * 0.15
        + percentile(stats["K%"], higher_is_better=False) * 0.15
    )
    stats["BvP Fantasy Score"] = 50.0 + reliability * (raw - 50.0)
    stats["Pitcher"] = pitcher_name
    stats["Team"] = team
    stats["Opponent"] = opponent
    return stats.sort_values(["BvP Fantasy Score", "PA"], ascending=False).reset_index(drop=True)


def build_fantasy_components(
    df: pd.DataFrame,
    hr_board: pd.DataFrame,
    pitcher_id: int,
    pitcher_hand: str,
    end_date: pd.Timestamp,
    hit_park_l: float,
    hit_park_r: float,
    team_runs: float,
    season: int,
    scoring: dict[str, float],
) -> pd.DataFrame:
    if hr_board.empty:
        return pd.DataFrame()

    board = hr_board.copy()
    event_df = add_fantasy_event_flags(df)
    player_ids = board["player_id"].dropna().astype(int).tolist()

    overall = _aggregate_event_counts(event_df[event_df["batter"].isin(player_ids)], "batter", "FSAll")
    overall = overall.rename(columns={"batter": "player_id"})
    platoon = _aggregate_event_counts(
        event_df[event_df["batter"].isin(player_ids) & event_df["p_throws"].eq(pitcher_hand)],
        "batter", "FSPlat",
    ).rename(columns={"batter": "player_id"})
    recent_start = pd.Timestamp(end_date) - pd.Timedelta(days=RECENT_DAYS - 1)
    recent = _aggregate_event_counts(
        event_df[
            event_df["batter"].isin(player_ids)
            & event_df["game_date"].between(recent_start, pd.Timestamp(end_date))
        ],
        "batter", "FSRecent",
    ).rename(columns={"batter": "player_id"})

    pitcher_rows = event_df[event_df["pitcher"].eq(int(pitcher_id))].copy()
    pitcher_split = _aggregate_event_counts(pitcher_rows, "stand", "FSOpp")
    pitcher_split = pitcher_split.rename(columns={"stand": "EffectiveStand"})

    board = board.merge(overall, on="player_id", how="left")
    board = board.merge(platoon, on="player_id", how="left")
    board = board.merge(recent, on="player_id", how="left")
    board = board.merge(pitcher_split, on="EffectiveStand", how="left")

    season_stats = fetch_mlb_hitting_season_stats(tuple(player_ids), int(season))
    board = board.merge(season_stats, on="player_id", how="left")

    numeric_prefixes = ["FSAll", "FSPlat", "FSRecent", "FSOpp"]
    event_cols = ["PA", "Single", "Double", "Triple", "HR", "Walk", "HBP", "K", "xH", "xSLG"]
    for prefix in numeric_prefixes:
        for suffix in event_cols:
            column = f"{prefix}_{suffix}"
            board[column] = pd.to_numeric(board.get(column, 0), errors="coerce").fillna(0.0)
    for column in [
        "Season_PA", "Season_R", "Season_RBI", "Season_SB", "Season_CS", "Season_BB",
        "Season_HBP", "Season_H", "Season_2B", "Season_3B", "Season_HR",
    ]:
        board[column] = pd.to_numeric(board.get(column, 0), errors="coerce").fillna(0.0)

    league_pa_rows = event_df[event_df["is_pa_end"]]
    league_pa = max(float(league_pa_rows["pa_key"].nunique()), 1.0)
    league_rates = {
        "Single": float(league_pa_rows["fs_single"].sum()) / league_pa,
        "Double": float(league_pa_rows["fs_double"].sum()) / league_pa,
        "Triple": float(league_pa_rows["fs_triple"].sum()) / league_pa,
        "HR": float(league_pa_rows["fs_hr"].sum()) / league_pa,
        "Walk": float(league_pa_rows["fs_walk"].sum()) / league_pa,
        "HBP": float(league_pa_rows["fs_hbp"].sum()) / league_pa,
        "K": float(league_pa_rows["fs_k"].sum()) / league_pa,
    }

    match_ratio = (
        0.55 * pd.to_numeric(board.get("PitchMatchRatio", 1.0), errors="coerce").fillna(1.0)
        + 0.30 * pd.to_numeric(board.get("ZoneFitRatio", 1.0), errors="coerce").fillna(1.0)
        + 0.15 * (pd.to_numeric(board.get("PitcherSideAttackScore", 50.0), errors="coerce").fillna(50.0) / 50.0)
    ).clip(0.70, 1.35)

    rate_columns: dict[str, str] = {}
    for outcome in ["Single", "Double", "Triple", "Walk", "HBP", "K"]:
        league = max(league_rates[outcome], 0.0001)
        all_rate = shrink_rate(board[f"FSAll_{outcome}"], board["FSAll_PA"], league, 120)
        plat_rate = shrink_rate(board[f"FSPlat_{outcome}"], board["FSPlat_PA"], league, 70)
        pitcher_rate = shrink_rate(board[f"FSOpp_{outcome}"], board["FSOpp_PA"], league, 180)
        recent_rate = shrink_rate(board[f"FSRecent_{outcome}"], board["FSRecent_PA"], league, 40)
        if outcome == "K":
            matchup_piece = league / match_ratio.clip(0.75, 1.25)
        else:
            matchup_piece = league * match_ratio
        blended = (
            0.45 * all_rate + 0.20 * plat_rate + 0.15 * pitcher_rate
            + 0.10 * recent_rate + 0.10 * matchup_piece
        )
        if outcome in {"Single", "Double", "Triple"}:
            park = np.where(
                board["EffectiveStand"].eq("L"), float(hit_park_l), float(hit_park_r)
            )
            elasticity = {"Single": 0.55, "Double": 0.75, "Triple": 0.60}[outcome]
            blended = blended * (1.0 + elasticity * (park / 100.0 - 1.0))
        output_column = f"Model_{outcome}_Per_PA"
        board[output_column] = blended.clip(0.0002, 0.42)
        rate_columns[outcome] = output_column

    board["Model_HR_Per_PA"] = pd.to_numeric(board["Model_HR_Per_PA"], errors="coerce").fillna(league_rates["HR"]).clip(0.0002, 0.20)

    total_event_rate = (
        board["Model_Single_Per_PA"] + board["Model_Double_Per_PA"]
        + board["Model_Triple_Per_PA"] + board["Model_HR_Per_PA"]
        + board["Model_Walk_Per_PA"] + board["Model_HBP_Per_PA"]
    )
    scale = np.minimum(1.0, 0.68 / total_event_rate.clip(lower=0.0001))
    for column in [
        "Model_Single_Per_PA", "Model_Double_Per_PA", "Model_Triple_Per_PA",
        "Model_HR_Per_PA", "Model_Walk_Per_PA", "Model_HBP_Per_PA",
    ]:
        board[column] = board[column] * scale

    board["Model_Hit_Per_PA"] = (
        board["Model_Single_Per_PA"] + board["Model_Double_Per_PA"]
        + board["Model_Triple_Per_PA"] + board["Model_HR_Per_PA"]
    )
    board["Model_OnBase_Per_PA"] = (
        board["Model_Hit_Per_PA"] + board["Model_Walk_Per_PA"] + board["Model_HBP_Per_PA"]
    )
    board["Model_TB_Per_PA"] = (
        board["Model_Single_Per_PA"] + 2 * board["Model_Double_Per_PA"]
        + 3 * board["Model_Triple_Per_PA"] + 4 * board["Model_HR_Per_PA"]
    )

    league_run_pa = 0.120
    league_rbi_pa = 0.112
    league_sb_pa = 0.015
    hist_run = shrink_rate(board["Season_R"], board["Season_PA"], league_run_pa, 85)
    hist_rbi = shrink_rate(board["Season_RBI"], board["Season_PA"], league_rbi_pa, 85)
    hist_sb = shrink_rate(board["Season_SB"], board["Season_PA"], league_sb_pa, 70)

    lineup_run_factor = board["LineupSpot"].map({
        1: 1.18, 2: 1.14, 3: 1.09, 4: 1.03, 5: 0.98,
        6: 0.92, 7: 0.87, 8: 0.82, 9: 0.78,
    }).fillna(0.92)
    lineup_rbi_factor = board["LineupSpot"].map({
        1: 0.84, 2: 0.96, 3: 1.13, 4: 1.26, 5: 1.18,
        6: 1.02, 7: 0.91, 8: 0.83, 9: 0.76,
    }).fillna(0.95)
    team_factor = float(np.clip(team_runs / 4.5, 0.55, 1.65))
    onbase_factor = np.sqrt((board["Model_OnBase_Per_PA"] / 0.325).clip(0.55, 1.65))
    power_factor = np.sqrt((board["Model_TB_Per_PA"] / 0.400).clip(0.55, 1.85))

    context_run = league_run_pa * team_factor * lineup_run_factor * onbase_factor
    context_rbi = league_rbi_pa * team_factor * lineup_rbi_factor * power_factor
    board["Model_Run_Per_PA"] = (0.52 * hist_run + 0.48 * context_run).clip(0.025, 0.30)
    board["Model_RBI_Per_PA"] = (0.52 * hist_rbi + 0.48 * context_rbi).clip(0.020, 0.32)
    board["Model_SB_Per_PA"] = (
        hist_sb * np.sqrt((board["Model_OnBase_Per_PA"] / 0.325).clip(0.55, 1.75))
    ).clip(0.0, 0.14)

    pa = pd.to_numeric(board["Projected_PA"], errors="coerce").fillna(4.1).clip(3.0, 5.5)
    board["Proj_Singles"] = pa * board["Model_Single_Per_PA"]
    board["Proj_Doubles"] = pa * board["Model_Double_Per_PA"]
    board["Proj_Triples"] = pa * board["Model_Triple_Per_PA"]
    board["Proj_HR"] = pa * board["Model_HR_Per_PA"]
    board["Proj_Walks"] = pa * board["Model_Walk_Per_PA"]
    board["Proj_HBP"] = pa * board["Model_HBP_Per_PA"]
    board["Proj_Runs"] = pa * board["Model_Run_Per_PA"]
    board["Proj_RBI"] = pa * board["Model_RBI_Per_PA"]
    board["Proj_SB"] = pa * board["Model_SB_Per_PA"]
    board["Proj_Total_Bases"] = pa * board["Model_TB_Per_PA"]
    board["Model_1plus_Hit"] = 1.0 - np.power(1.0 - board["Model_Hit_Per_PA"], pa)
    board["Model_1plus_OnBase"] = 1.0 - np.power(1.0 - board["Model_OnBase_Per_PA"], pa)
    board["Model_1plus_SB"] = 1.0 - np.exp(-board["Proj_SB"])

    board["Projected_FS"] = (
        scoring["Single"] * board["Proj_Singles"]
        + scoring["Double"] * board["Proj_Doubles"]
        + scoring["Triple"] * board["Proj_Triples"]
        + scoring["HomeRun"] * board["Proj_HR"]
        + scoring["Run"] * board["Proj_Runs"]
        + scoring["RBI"] * board["Proj_RBI"]
        + scoring["Walk"] * board["Proj_Walks"]
        + scoring["HBP"] * board["Proj_HBP"]
        + scoring["StolenBase"] * board["Proj_SB"]
    )
    board["Projected_FS_NoSpeed"] = board["Projected_FS"] - scoring["StolenBase"] * board["Proj_SB"]
    board["ContactScore"] = (
        percentile(board["Model_Hit_Per_PA"]) * 0.38
        + percentile(board.get("Contact_Pct", pd.Series(0.7, index=board.index))) * 0.24
        + percentile(board.get("K_Pct", pd.Series(0.22, index=board.index)), higher_is_better=False) * 0.20
        + percentile(board.get("Adj_xSLG_PA", pd.Series(0.4, index=board.index))) * 0.18
    )
    return board


def _seed_for_player(player_id: object, extra: str = "") -> int:
    value = f"{player_id}-{extra}".encode("utf-8")
    return int(hashlib.sha256(value).hexdigest()[:8], 16)


def simulate_fantasy_scores(
    row: pd.Series,
    scoring: dict[str, float],
    simulations: int = 7000,
    seed_extra: str = "",
) -> np.ndarray:
    rng = np.random.default_rng(_seed_for_player(row.get("player_id"), seed_extra))
    projected_pa = float(np.clip(row.get("Projected_PA", 4.1), 3.0, 5.5))
    floor_pa = int(math.floor(projected_pa))
    n_pa = floor_pa + (rng.random(simulations) < (projected_pa - floor_pa)).astype(int)
    max_pa = max(int(n_pa.max()), 1)

    probabilities = np.array([
        row.get("Model_Single_Per_PA", 0.15), row.get("Model_Double_Per_PA", 0.045),
        row.get("Model_Triple_Per_PA", 0.005), row.get("Model_HR_Per_PA", 0.035),
        row.get("Model_Walk_Per_PA", 0.085), row.get("Model_HBP_Per_PA", 0.012),
    ], dtype=float)
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = np.clip(probabilities, 0.0, 0.95)
    if probabilities.sum() > 0.80:
        probabilities *= 0.80 / probabilities.sum()
    cumulative = np.cumsum(probabilities)
    scores = np.zeros(simulations, dtype=float)
    event_points = np.array([
        scoring["Single"], scoring["Double"], scoring["Triple"],
        scoring["HomeRun"], scoring["Walk"], scoring["HBP"],
    ], dtype=float)

    for pa_index in range(max_pa):
        active = n_pa > pa_index
        if not active.any():
            continue
        draws = rng.random(simulations)
        event_index = np.searchsorted(cumulative, draws, side="right")
        valid = active & (event_index < len(event_points))
        scores[valid] += event_points[event_index[valid]]

    run_rate = float(np.clip(row.get("Model_Run_Per_PA", 0.12), 0.0, 0.45))
    rbi_rate = float(np.clip(row.get("Model_RBI_Per_PA", 0.11), 0.0, 0.45))
    sb_rate = float(np.clip(row.get("Model_SB_Per_PA", 0.015), 0.0, 0.20))
    scores += scoring["Run"] * rng.binomial(n_pa, run_rate)
    scores += scoring["RBI"] * rng.binomial(n_pa, rbi_rate)
    scores += scoring["StolenBase"] * rng.binomial(n_pa, sb_rate)
    return scores


def add_distribution_columns(board: pd.DataFrame, scoring: dict[str, float]) -> pd.DataFrame:
    output = board.copy()
    p25, p50, p75, p90, stds = [], [], [], [], []
    for _, row in output.iterrows():
        scores = simulate_fantasy_scores(row, scoring, simulations=6000, seed_extra="distribution")
        p25.append(float(np.quantile(scores, 0.25)))
        p50.append(float(np.quantile(scores, 0.50)))
        p75.append(float(np.quantile(scores, 0.75)))
        p90.append(float(np.quantile(scores, 0.90)))
        stds.append(float(np.std(scores)))
    output["FS_P25"] = p25
    output["FS_Median"] = p50
    output["FS_P75"] = p75
    output["FS_P90"] = p90
    output["FS_Std"] = stds
    return output


def finalize_fantasy_rankings(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return board
    output = board.copy()
    lineup_score = (112.5 - 12.5 * pd.to_numeric(output["LineupSpot"], errors="coerce").fillna(6)).clip(0, 100)
    output["FantasyScoreRating"] = (
        percentile(output["Projected_FS"]) * 0.27
        + percentile(output["Proj_Total_Bases"]) * 0.16
        + percentile(output["Model_1plus_Hit"]) * 0.13
        + percentile(output["Proj_Runs"]) * 0.10
        + percentile(output["Proj_RBI"]) * 0.10
        + percentile(output["Model_1plus_HR"]) * 0.09
        + percentile(output["Proj_Walks"] + output["Proj_HBP"]) * 0.05
        + percentile(output["Proj_SB"]) * 0.05
        + lineup_score * 0.03
        + pd.to_numeric(output.get("Confidence", 50), errors="coerce").fillna(50) * 0.02
    ).clip(0, 100)
    output["MatchupRating"] = (
        pd.to_numeric(output.get("PitchMatchScore", 50), errors="coerce").fillna(50) * 0.35
        + pd.to_numeric(output.get("ZoneFitScore", 50), errors="coerce").fillna(50) * 0.20
        + pd.to_numeric(output.get("PitcherPowerScore", 50), errors="coerce").fillna(50) * 0.20
        + pd.to_numeric(output.get("RecentFormScore", 50), errors="coerce").fillna(50) * 0.15
        + pd.to_numeric(output.get("BvPScore", 50), errors="coerce").fillna(50) * 0.10
    ).clip(0, 100)
    output = output.sort_values(["Projected_FS", "FantasyScoreRating"], ascending=False).reset_index(drop=True)
    output.insert(0, "Rank", np.arange(1, len(output) + 1))
    return output


def _game_label(game: dict) -> str:
    return f"{game.get('away_abbr')} @ {game.get('home_abbr')} · {game.get('time_et')} · {game.get('venue')}"


def resolve_probable_pitcher_id(game: dict, batting_side: str, df: pd.DataFrame) -> int | None:
    key = "home_pitcher_id" if batting_side == "away" else "away_pitcher_id"
    value = pd.to_numeric(pd.Series([game.get(key)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    pitcher_id = int(value)
    if not df["pitcher"].eq(pitcher_id).any():
        return None
    return pitcher_id


def build_selected_slate(
    df: pd.DataFrame,
    games: list[dict],
    team_run_map: dict[str, float],
    min_pa: int,
    end_date: pd.Timestamp,
    starter_innings: float,
    bullpen_hr_multiplier: float,
    use_weather: bool,
    assume_retractable_open: bool,
    scoring: dict[str, float],
    include_low_sample: bool,
) -> dict:
    boards: list[pd.DataFrame] = []
    pitch_tables: list[pd.DataFrame] = []
    bvp_tables: list[pd.DataFrame] = []
    environments: list[dict] = []
    lineup_rows: list[dict] = []
    errors: list[str] = []
    event_df = add_fantasy_event_flags(df)
    available_teams = sorted(df["batter_team"].dropna().astype(str).unique().tolist())

    for game in games:
        weather_result = automatic_game_weather(game) if use_weather else {"ok": False, "error": "Disabled"}
        metadata = weather_result.get("metadata") or load_stadium_weather_metadata(str(game.get("venue", "")))
        roof_type = str(metadata.get("roof_type", "open")).lower()
        enclosed = roof_type == "fixed" or (roof_type == "retractable" and not assume_retractable_open)
        if use_weather and weather_result.get("ok") and not enclosed:
            weather_multiplier = weather_carry_multiplier(
                float(weather_result.get("temperature_f", 75.0)),
                float(weather_result.get("humidity_pct", 50.0)),
                float(weather_result.get("wind_mph", 5.0)),
                str(weather_result.get("wind_direction", "Cross/Calm")),
                float(weather_result.get("pressure_hpa", 1013.25)),
                1.0,
                enclosed=False,
            )
        else:
            weather_multiplier = 1.0

        park_year = pd.Timestamp(game.get("slate_date") or end_date).year
        park_hr = fetch_savant_park_factors(str(game.get("venue", "")), park_year, "HR")
        park_hits = fetch_savant_park_factors(str(game.get("venue", "")), park_year, "Hits")

        environments.append({
            "Game": f"{game.get('away_abbr')} @ {game.get('home_abbr')}",
            "Venue": game.get("venue"),
            "Roof": roof_type,
            "Weather Used": bool(use_weather and weather_result.get("ok") and not enclosed),
            "Condition": weather_result.get("condition", "Neutral") if weather_result.get("ok") else "Neutral",
            "Temp F": weather_result.get("temperature_f", np.nan),
            "Wind mph": weather_result.get("wind_mph", np.nan),
            "Wind": weather_result.get("wind_direction", "Neutral"),
            "Weather Mult": weather_multiplier,
            "HR L": park_hr.get("L", 100), "HR R": park_hr.get("R", 100),
            "Hits L": park_hits.get("L", 100), "Hits R": park_hits.get("R", 100),
        })

        for side in ("away", "home"):
            team_code = str(game.get(f"{side}_abbr", ""))
            opponent_side = "home" if side == "away" else "away"
            opponent_code = str(game.get(f"{opponent_side}_abbr", ""))
            statcast_team = match_statcast_team(team_code, available_teams)
            if not statcast_team:
                errors.append(f"{team_code}: could not match the team to the Statcast sample.")
                continue
            pitcher_id = resolve_probable_pitcher_id(game, side, df)
            if pitcher_id is None:
                errors.append(f"{team_code}: probable opposing pitcher was unavailable in the loaded sample.")
                continue
            pitcher_name = str(game.get(f"{opponent_side}_pitcher_name", "TBD"))
            profile, pitcher_hand = selected_pitcher_profile(df, pitcher_id)
            if not pitcher_hand:
                errors.append(f"{team_code}: no pitch profile was available for {pitcher_name}.")
                continue

            lineup_result = fetch_mlb_confirmed_lineup(game.get("game_pk"), side)
            lineup = _clean_lineup_seed(lineup_result.get("lineup"))
            lineup_status = str(lineup_result.get("status", "Not posted"))
            lineup_source = str(lineup_result.get("source", "Recent lineup fallback"))
            if lineup.empty:
                lineup = infer_recent_lineup(df, statcast_team)
                lineup_status = "Recent fallback"
                lineup_source = "Latest observed Statcast lineup"

            active_roster, _ = fetch_active_roster(game.get(f"{side}_id"), game.get("slate_date"))
            team_runs = float(team_run_map.get(team_code, 4.5))
            hr_board, _ = build_hr_board(
                df=df,
                pitcher_id=pitcher_id,
                team=statcast_team,
                min_pa=min_pa,
                end_date=end_date,
                park_hr_factor_lhb=float(park_hr.get("L", 100.0)),
                park_hr_factor_rhb=float(park_hr.get("R", 100.0)),
                team_runs=team_runs,
                is_away=side == "away",
                starter_innings=starter_innings,
                bullpen_multiplier=bullpen_hr_multiplier,
                weather_multiplier=weather_multiplier,
                lineup_override=lineup,
                lineup_edits=None,
                bat_tracking_upload=None,
                bat_tracking_auto=None,
                active_roster=active_roster,
                include_low_sample=include_low_sample,
            )
            if hr_board.empty:
                errors.append(f"{team_code}: no hitter board could be built.")
                continue

            if not lineup.empty:
                lineup_ids = set(lineup["player_id"].astype(int).tolist())
                hr_board = hr_board[hr_board["player_id"].astype(int).isin(lineup_ids)].copy()
            else:
                hr_board = hr_board.sort_values("PA", ascending=False).head(9).copy()
            if hr_board.empty:
                errors.append(f"{team_code}: the posted lineup did not match the loaded Statcast players.")
                continue

            fantasy = build_fantasy_components(
                event_df, hr_board, pitcher_id, pitcher_hand, end_date,
                float(park_hits.get("L", 100.0)), float(park_hits.get("R", 100.0)),
                team_runs, park_year, scoring,
            )
            fantasy["Team"] = team_code
            fantasy["Opponent"] = opponent_code
            fantasy["Game"] = f"{game.get('away_abbr')} @ {game.get('home_abbr')}"
            fantasy["Pitcher"] = pitcher_name
            fantasy["PitcherHand"] = pitcher_hand
            fantasy["Venue"] = game.get("venue")
            fantasy["Game Time"] = game.get("time_et")
            fantasy["Team Implied Runs"] = team_runs
            fantasy["WeatherMultiplier"] = weather_multiplier
            fantasy["LineupStatus"] = lineup_status
            fantasy["LineupSource"] = lineup_source
            boards.append(fantasy)

            pitch_table = build_batter_pitch_type_table(event_df, fantasy, profile, pitcher_hand)
            if not pitch_table.empty:
                pitch_table["Team"] = team_code
                pitch_table["Opponent"] = opponent_code
                pitch_table["Pitcher"] = pitcher_name
                pitch_tables.append(pitch_table)

            bvp = build_fantasy_bvp_table(event_df, fantasy, pitcher_id, pitcher_name, team_code, opponent_code)
            if not bvp.empty:
                bvp_tables.append(bvp)

            for row in lineup.itertuples(index=False):
                lineup_rows.append({
                    "Game": f"{game.get('away_abbr')} @ {game.get('home_abbr')}",
                    "Team": team_code,
                    "Order": int(row.LineupSpot),
                    "player_id": int(row.player_id),
                    "Status": lineup_status,
                    "Source": lineup_source,
                })

    if not boards:
        return {
            "board": pd.DataFrame(), "pitch_types": pd.DataFrame(), "bvp": pd.DataFrame(),
            "environment": pd.DataFrame(environments), "lineups": pd.DataFrame(lineup_rows),
            "errors": errors,
        }
    board = pd.concat(boards, ignore_index=True)
    board = finalize_fantasy_rankings(board)
    board = add_distribution_columns(board, scoring)
    return {
        "board": board,
        "pitch_types": pd.concat(pitch_tables, ignore_index=True) if pitch_tables else pd.DataFrame(),
        "bvp": pd.concat(bvp_tables, ignore_index=True) if bvp_tables else pd.DataFrame(),
        "environment": pd.DataFrame(environments),
        "lineups": pd.DataFrame(lineup_rows),
        "errors": errors,
    }


def render_fantasy_leaders(board: pd.DataFrame) -> None:
    leaders = board.head(3)
    columns = st.columns(3)
    for index, (_, row) in enumerate(leaders.iterrows()):
        with columns[index]:
            st.markdown(
                f"""
                <div class="leader-card">
                    <div class="leader-rank">RANK {int(row['Rank'])}</div>
                    <div class="leader-name">{escape(str(row['Player']))}</div>
                    <div class="leader-prob">{row['Projected_FS']:.2f}</div>
                    <div class="leader-sub">Projected fantasy score · rating {row['FantasyScoreRating']:.1f}</div>
                    <div class="leader-sub">{escape(str(row['Team']))} vs {escape(str(row['Opponent']))} · #{int(row['LineupSpot'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


inject_clean_css()
MODEL_KEY = "hitter_fs"

st.markdown('<div class="app-kicker">⚡ PrizePicks hitter lab</div>', unsafe_allow_html=True)
st.title("MLB Hitter Fantasy Score Dashboard")
st.caption(
    "Projects hitter fantasy score from expected singles, extra-base hits, home runs, runs, RBI, "
    "walks, hit-by-pitches and stolen bases, then layers in lineup opportunity, pitcher matchup, "
    "park, weather, platoon splits, pitch types and confirmed MLB batting orders."
)

with st.sidebar:
    st.header("Model sample")
    yesterday = date.today() - timedelta(days=1)
    end_date_value = st.date_input("Stats through", value=yesterday, max_value=yesterday, key="fs_end")
    lookback_days = st.slider("Lookback days", 28, 120, 60, 7, key="fs_lookback")
    min_pa = st.slider("Minimum hitter PA", 10, 100, 25, 5, key="fs_min_pa")
    include_low_sample = st.checkbox(
        "Include low-sample active hitters", value=True, key="fs_low_sample"
    )
    refresh_data = st.button("Load / refresh Statcast", type="primary", key="fs_refresh_data")
    st.divider()
    st.header("PrizePicks scoring")
    st.caption("Defaults are editable so the app can follow any scoring change shown in PrizePicks.")
    scoring = {
        "Single": st.number_input("Single", 0.0, 20.0, 3.0, 0.5, key="fs_pts_single"),
        "Double": st.number_input("Double", 0.0, 20.0, 5.0, 0.5, key="fs_pts_double"),
        "Triple": st.number_input("Triple", 0.0, 25.0, 8.0, 0.5, key="fs_pts_triple"),
        "HomeRun": st.number_input("Home run", 0.0, 30.0, 10.0, 0.5, key="fs_pts_hr"),
        "Run": st.number_input("Run", 0.0, 10.0, 2.0, 0.5, key="fs_pts_run"),
        "RBI": st.number_input("RBI", 0.0, 10.0, 2.0, 0.5, key="fs_pts_rbi"),
        "Walk": st.number_input("Walk", 0.0, 10.0, 2.0, 0.5, key="fs_pts_walk"),
        "HBP": st.number_input("Hit by pitch", 0.0, 10.0, 2.0, 0.5, key="fs_pts_hbp"),
        "StolenBase": st.number_input("Stolen base", 0.0, 15.0, 5.0, 0.5, key="fs_pts_sb"),
    }

start_date_value = end_date_value - timedelta(days=lookback_days - 1)
if refresh_data or "fs_statcast_data" not in st.session_state:
    with st.spinner(f"Loading Statcast from {start_date_value} through {end_date_value}..."):
        raw = load_statcast(start_date_value.strftime("%Y-%m-%d"), end_date_value.strftime("%Y-%m-%d"))
        st.session_state["fs_statcast_data"] = add_fantasy_event_flags(prepare_data(raw))
        st.session_state["fs_loaded_dates"] = (start_date_value, end_date_value)

df = st.session_state.get("fs_statcast_data", pd.DataFrame())
if df.empty:
    st.warning("No Statcast data was returned. Use completed dates and try again.")
    st.stop()

loaded_start, loaded_end = st.session_state["fs_loaded_dates"]
st.caption(f"Using {len(df):,} pitches from {loaded_start} through {loaded_end}.")

st.subheader("Choose the PrizePicks slate")
slate_date = st.date_input("Slate date", value=date.today(), key="fs_slate_date")
games, schedule_error = fetch_mlb_schedule(slate_date.strftime("%Y-%m-%d"))
if schedule_error:
    st.error(f"MLB schedule could not be loaded: {schedule_error}")
    st.stop()
if not games:
    st.info("No MLB games were found for that date.")
    st.stop()

labels = [_game_label(game) for game in games]
label_map = {label: game for label, game in zip(labels, games)}
scope_mode = st.radio(
    "Board scope", ["Selected games", "Full slate"], horizontal=True, key="fs_scope"
)
if scope_mode == "Full slate":
    selected_labels = labels
    st.caption(f"Full-slate mode will evaluate {len(games)} games and can take longer.")
else:
    default_count = min(3, len(labels))
    selected_labels = st.multiselect(
        "Games to analyze", labels, default=labels[:default_count], key="fs_games"
    )
selected_games = [label_map[label] for label in selected_labels if label in label_map]
if not selected_games:
    st.info("Select at least one game.")
    st.stop()

team_rows = []
for game in selected_games:
    team_rows.extend([
        {"Team": game.get("away_abbr"), "Opponent": game.get("home_abbr"), "Team Implied Runs": 4.5},
        {"Team": game.get("home_abbr"), "Opponent": game.get("away_abbr"), "Team Implied Runs": 4.5},
    ])
st.markdown("#### Team scoring environment")
st.caption("Enter sportsbook team totals when available. Higher totals raise run and RBI projections.")
team_total_editor = st.data_editor(
    pd.DataFrame(team_rows),
    hide_index=True,
    use_container_width=True,
    disabled=["Team", "Opponent"],
    column_config={
        "Team Implied Runs": st.column_config.NumberColumn(min_value=1.0, max_value=9.0, step=0.1, format="%.1f")
    },
    key="fs_team_totals",
)
team_run_map = {
    str(row["Team"]): float(row["Team Implied Runs"])
    for _, row in team_total_editor.iterrows()
}

with st.expander("Shared matchup assumptions", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        starter_innings = st.number_input("Expected starter innings", 2.0, 8.0, 5.5, 0.5, key="fs_starter_ip")
    with c2:
        bullpen_hr_multiplier = st.number_input("Bullpen HR multiplier", 0.65, 1.45, 1.00, 0.01, key="fs_bp_hr")
    with c3:
        use_weather = st.checkbox("Automatic first-pitch weather", value=True, key="fs_weather")
    with c4:
        assume_retractable_open = st.checkbox(
            "Assume retractable roofs open", value=False, key="fs_roof_open",
            help="When off, retractable-roof games use neutral outdoor weather until roof status is known.",
        )

build_board = st.button("Build fantasy score board", type="primary", key="fs_build")
settings_fingerprint = str((
    tuple(game.get("game_pk") for game in selected_games), tuple(sorted(team_run_map.items())),
    min_pa, str(end_date_value), starter_innings, bullpen_hr_multiplier,
    use_weather, assume_retractable_open, tuple(sorted(scoring.items())), include_low_sample,
))

if build_board:
    with st.spinner("Building lineup, matchup, weather and fantasy-score projections..."):
        bundle = build_selected_slate(
            df=df,
            games=selected_games,
            team_run_map=team_run_map,
            min_pa=min_pa,
            end_date=pd.Timestamp(end_date_value),
            starter_innings=starter_innings,
            bullpen_hr_multiplier=bullpen_hr_multiplier,
            use_weather=use_weather,
            assume_retractable_open=assume_retractable_open,
            scoring=scoring,
            include_low_sample=include_low_sample,
        )
        st.session_state["fs_bundle"] = bundle
        st.session_state["fs_bundle_fingerprint"] = settings_fingerprint

bundle = st.session_state.get("fs_bundle")
if not bundle:
    st.info("Set the games and team totals, then click **Build fantasy score board**.")
    st.stop()
if st.session_state.get("fs_bundle_fingerprint") != settings_fingerprint:
    st.warning("Settings changed after the current board was built. Click **Build fantasy score board** to apply them.")

board = bundle.get("board", pd.DataFrame())
if board.empty:
    st.error("No fantasy board was produced.")
    for message in bundle.get("errors", [])[:12]:
        st.caption(message)
    st.stop()

if bundle.get("errors"):
    with st.expander(f"Skipped items / connection details ({len(bundle['errors'])})"):
        for message in bundle["errors"]:
            st.write(f"- {message}")

render_fantasy_leaders(board)

(
    board_tab, pp_tab, breakdown_tab, contact_tab, power_tab,
    run_tab, pitch_tab, bvp_tab, environment_tab, notes_tab,
) = st.tabs([
    "Fantasy board", "PrizePicks comparison", "Scoring breakdown", "Contact & total bases",
    "Power & matchup", "Runs, RBI & speed", "Batter vs pitch type", "Batter vs pitcher",
    "Lineups & environment", "Model notes",
])

with board_tab:
    quick_columns = [
        "Rank", "Player", "Team", "Opponent", "LineupSpot", "Projected_PA", "Projected_FS",
        "FantasyScoreRating", "FS_Median", "FS_P75", "Model_1plus_Hit", "Model_1plus_HR",
        "Proj_Total_Bases", "MatchupRating", "Confidence_Level", "LineupStatus",
    ]
    quick = board[[column for column in quick_columns if column in board.columns]].copy()
    quick = quick.rename(columns={
        "LineupSpot": "Order", "Projected_PA": "Proj PA", "Projected_FS": "Projected FS",
        "FantasyScoreRating": "FS Rating", "FS_Median": "Median", "FS_P75": "75th %ile",
        "Model_1plus_Hit": "1+ Hit", "Model_1plus_HR": "1+ HR",
        "Proj_Total_Bases": "Proj TB", "MatchupRating": "Matchup", "Confidence_Level": "Confidence",
        "LineupStatus": "Lineup",
    })
    score_cols = [column for column in ["Projected FS", "FS Rating", "75th %ile", "1+ Hit", "1+ HR", "Proj TB", "Matchup"] if column in quick.columns]
    style = quick.style.background_gradient(cmap="RdYlGn", subset=score_cols, axis=0).format({
        "Order": "{:.0f}", "Proj PA": "{:.2f}", "Projected FS": "{:.2f}",
        "FS Rating": "{:.1f}", "Median": "{:.1f}", "75th %ile": "{:.1f}",
        "1+ Hit": "{:.1%}", "1+ HR": "{:.1%}", "Proj TB": "{:.2f}", "Matchup": "{:.1f}",
    })
    st.dataframe(style, use_container_width=True, hide_index=True, height=620)

with pp_tab:
    st.caption("Enter the current PrizePicks Hitter Fantasy Score line. Probabilities come from a discrete event simulation, not a normal distribution.")
    comparison_seed = board[["player_id", "Player", "Team", "Opponent", "Projected_FS", "FS_Median", "FS_P75"]].copy()
    comparison_seed["PP Line"] = comparison_seed["Projected_FS"].round() - 0.5
    comparison = st.data_editor(
        comparison_seed,
        hide_index=True,
        use_container_width=True,
        disabled=["player_id", "Player", "Team", "Opponent", "Projected_FS", "FS_Median", "FS_P75"],
        column_config={
            "player_id": None,
            "Projected_FS": st.column_config.NumberColumn("Projection", format="%.2f"),
            "FS_Median": st.column_config.NumberColumn("Median", format="%.1f"),
            "FS_P75": st.column_config.NumberColumn("75th %ile", format="%.1f"),
            "PP Line": st.column_config.NumberColumn(min_value=0.0, max_value=40.0, step=0.5, format="%.1f"),
        },
        key="fs_pp_lines",
    )
    model_lookup = board.set_index("player_id")
    over_probs, under_probs, push_probs = [], [], []
    for _, line_row in comparison.iterrows():
        model_row = model_lookup.loc[int(line_row["player_id"])]
        scores = simulate_fantasy_scores(model_row, scoring, simulations=9000, seed_extra=f"line-{line_row['PP Line']}")
        line = float(line_row["PP Line"])
        over_probs.append(float(np.mean(scores > line)))
        under_probs.append(float(np.mean(scores < line)))
        push_probs.append(float(np.mean(np.isclose(scores, line))))
    comparison["Edge"] = comparison["Projected_FS"] - comparison["PP Line"]
    comparison["Over Prob"] = over_probs
    comparison["Under Prob"] = under_probs
    comparison["Push Prob"] = push_probs
    comparison = comparison.sort_values(["Over Prob", "Edge"], ascending=False)
    style = comparison.drop(columns=["player_id"]).style.background_gradient(
        cmap="RdYlGn", subset=["Edge", "Over Prob"], axis=0
    ).format({
        "Projected_FS": "{:.2f}", "FS_Median": "{:.1f}", "FS_P75": "{:.1f}",
        "PP Line": "{:.1f}", "Edge": "{:+.2f}", "Over Prob": "{:.1%}",
        "Under Prob": "{:.1%}", "Push Prob": "{:.1%}",
    })
    st.dataframe(style, use_container_width=True, hide_index=True, height=620)

with breakdown_tab:
    columns = [
        "Player", "Team", "Projected_FS", "Proj_Singles", "Proj_Doubles", "Proj_Triples",
        "Proj_HR", "Proj_Runs", "Proj_RBI", "Proj_Walks", "Proj_HBP", "Proj_SB",
        "Proj_Total_Bases", "FS_P25", "FS_Median", "FS_P75", "FS_P90",
    ]
    detail = board[[column for column in columns if column in board.columns]].copy()
    detail = detail.rename(columns={
        "Projected_FS": "Projected FS", "Proj_Singles": "Singles", "Proj_Doubles": "Doubles",
        "Proj_Triples": "Triples", "Proj_HR": "HR", "Proj_Runs": "Runs", "Proj_RBI": "RBI",
        "Proj_Walks": "Walks", "Proj_HBP": "HBP", "Proj_SB": "SB", "Proj_Total_Bases": "Total Bases",
        "FS_P25": "25th %ile", "FS_Median": "Median", "FS_P75": "75th %ile", "FS_P90": "90th %ile",
    })
    positive = [column for column in detail.columns if column not in {"Player", "Team"}]
    st.dataframe(
        detail.style.background_gradient(cmap="RdYlGn", subset=positive, axis=0).format(
            {column: "{:.2f}" for column in positive}
        ), use_container_width=True, hide_index=True, height=620,
    )

with contact_tab:
    columns = [
        "Player", "Team", "Model_1plus_Hit", "Model_Hit_Per_PA", "Proj_Total_Bases",
        "Contact_Pct", "Whiff_Pct", "K_Pct", "Adj_xSLG_PA", "HH_Pct", "SweetSpot_Pct",
        "ContactScore",
    ]
    contact = board[[column for column in columns if column in board.columns]].copy()
    contact = contact.rename(columns={
        "Model_1plus_Hit": "1+ Hit", "Model_Hit_Per_PA": "Hit/PA", "Proj_Total_Bases": "Proj TB",
        "Contact_Pct": "Contact%", "Whiff_Pct": "Whiff%", "K_Pct": "K%",
        "Adj_xSLG_PA": "Adj xSLG/PA", "HH_Pct": "Hard Hit%", "SweetSpot_Pct": "Sweet Spot%",
        "ContactScore": "Contact Score",
    })
    positive = [column for column in ["1+ Hit", "Hit/PA", "Proj TB", "Contact%", "Adj xSLG/PA", "Hard Hit%", "Sweet Spot%", "Contact Score"] if column in contact.columns]
    risks = [column for column in ["Whiff%", "K%"] if column in contact.columns]
    style = contact.style
    if positive:
        style = style.background_gradient(cmap="RdYlGn", subset=positive, axis=0)
    if risks:
        style = style.background_gradient(cmap="RdYlGn_r", subset=risks, axis=0)
    st.dataframe(style.format({
        "1+ Hit": "{:.1%}", "Hit/PA": "{:.1%}", "Proj TB": "{:.2f}", "Contact%": "{:.1%}",
        "Whiff%": "{:.1%}", "K%": "{:.1%}", "Adj xSLG/PA": "{:.3f}", "Hard Hit%": "{:.1%}",
        "Sweet Spot%": "{:.1%}", "Contact Score": "{:.1f}",
    }), use_container_width=True, hide_index=True, height=620)

with power_tab:
    columns = [
        "Player", "Team", "Model_1plus_HR", "HRScore", "Adj_Brl_PA", "PullAir_BIP",
        "Adj_xISO_PA", "EVPowerScore", "PitchMatchScore", "ZoneFitScore", "PitcherPowerScore",
        "RecentFormScore", "MatchupRating", "ParkFactor", "WeatherMultiplier",
    ]
    power = board[[column for column in columns if column in board.columns]].copy()
    power = power.rename(columns={
        "Model_1plus_HR": "1+ HR", "HRScore": "HR Score", "Adj_Brl_PA": "Adj Brl/PA",
        "PullAir_BIP": "Pull Air/BIP", "Adj_xISO_PA": "Adj xISO/PA", "EVPowerScore": "EV Power",
        "PitchMatchScore": "Pitch Match", "ZoneFitScore": "Zone Fit", "PitcherPowerScore": "Pitcher Power",
        "RecentFormScore": "Recent Form", "MatchupRating": "Matchup", "ParkFactor": "HR Park",
        "WeatherMultiplier": "Weather Mult",
    })
    positive = [column for column in power.columns if column not in {"Player", "Team"}]
    st.dataframe(
        power.style.background_gradient(cmap="RdYlGn", subset=positive, axis=0).format({
            "1+ HR": "{:.1%}", "HR Score": "{:.1f}", "Adj Brl/PA": "{:.2%}",
            "Pull Air/BIP": "{:.1%}", "Adj xISO/PA": "{:.3f}", "EV Power": "{:.1f}",
            "Pitch Match": "{:.1f}", "Zone Fit": "{:.1f}", "Pitcher Power": "{:.1f}",
            "Recent Form": "{:.1f}", "Matchup": "{:.1f}", "HR Park": "{:.0f}", "Weather Mult": "{:.3f}",
        }), use_container_width=True, hide_index=True, height=620,
    )

with run_tab:
    columns = [
        "Player", "Team", "LineupSpot", "Team Implied Runs", "Proj_Runs", "Proj_RBI", "Proj_SB",
        "Model_1plus_OnBase", "Model_1plus_SB", "Season_PA", "Season_R", "Season_RBI", "Season_SB",
    ]
    run_view = board[[column for column in columns if column in board.columns]].copy()
    run_view = run_view.rename(columns={
        "LineupSpot": "Order", "Proj_Runs": "Proj Runs", "Proj_RBI": "Proj RBI", "Proj_SB": "Proj SB",
        "Model_1plus_OnBase": "1+ On Base", "Model_1plus_SB": "1+ SB", "Season_PA": "Season PA",
        "Season_R": "Season Runs", "Season_RBI": "Season RBI", "Season_SB": "Season SB",
    })
    positive = [column for column in ["Team Implied Runs", "Proj Runs", "Proj RBI", "Proj SB", "1+ On Base", "1+ SB", "Season Runs", "Season RBI", "Season SB"] if column in run_view.columns]
    st.dataframe(
        run_view.style.background_gradient(cmap="RdYlGn", subset=positive, axis=0).format({
            "Order": "{:.0f}", "Team Implied Runs": "{:.1f}", "Proj Runs": "{:.2f}",
            "Proj RBI": "{:.2f}", "Proj SB": "{:.3f}", "1+ On Base": "{:.1%}", "1+ SB": "{:.1%}",
            "Season PA": "{:.0f}", "Season Runs": "{:.0f}", "Season RBI": "{:.0f}", "Season SB": "{:.0f}",
        }), use_container_width=True, hide_index=True, height=620,
    )

with pitch_tab:
    pitch_types = bundle.get("pitch_types", pd.DataFrame())
    if pitch_types.empty:
        st.info("No batter-versus-pitch-type rows were available.")
    else:
        player_options = board.sort_values("Rank")["Player"].tolist()
        selected_player = st.selectbox("Hitter", player_options, key="fs_pitch_player")
        view = pitch_types[pitch_types["Player"].eq(selected_player)].sort_values("Pitcher Usage", ascending=False).copy()
        columns = [
            "Player", "Team", "Pitcher", "Pitch Type", "Pitcher Usage", "Pitcher Velo", "Pitches Seen",
            "PA Ends", "BBE", "HR/PA", "Brl/BIP", "xSLG Contact", "xBA Contact", "Hard Hit%",
            "Contact%", "Whiff%", "Pitch Type Score", "Sample",
        ]
        view = view[[column for column in columns if column in view.columns]]
        positive = [column for column in ["HR/PA", "Brl/BIP", "xSLG Contact", "xBA Contact", "Hard Hit%", "Contact%", "Pitch Type Score"] if column in view.columns]
        risk = [column for column in ["Whiff%"] if column in view.columns]
        style = view.style
        if positive:
            style = style.background_gradient(cmap="RdYlGn", subset=positive, axis=0)
        if risk:
            style = style.background_gradient(cmap="RdYlGn_r", subset=risk, axis=0)
        st.dataframe(style.format({
            "Pitcher Usage": "{:.1%}", "Pitcher Velo": "{:.1f}", "HR/PA": "{:.2%}", "Brl/BIP": "{:.1%}",
            "xSLG Contact": "{:.3f}", "xBA Contact": "{:.3f}", "Hard Hit%": "{:.1%}",
            "Contact%": "{:.1%}", "Whiff%": "{:.1%}", "Pitch Type Score": "{:.1f}",
        }), use_container_width=True, hide_index=True, height=560)

with bvp_tab:
    bvp = bundle.get("bvp", pd.DataFrame())
    if bvp.empty:
        st.info("No direct batter-versus-pitcher history was available for the selected games.")
    else:
        display_columns = [
            "Player", "Team", "Pitcher", "PA", "H", "HR", "BB", "HBP", "K", "H/PA", "xH/PA",
            "HR/PA", "Avg_EV", "Hard Hit%", "Brl/BIP", "BvP Fantasy Score", "Last_Date",
        ]
        view = bvp[[column for column in display_columns if column in bvp.columns]].copy()
        positive = [column for column in ["H/PA", "xH/PA", "HR/PA", "Avg_EV", "Hard Hit%", "Brl/BIP", "BvP Fantasy Score"] if column in view.columns]
        risk = [column for column in ["K"] if column in view.columns]
        style = view.style.background_gradient(cmap="RdYlGn", subset=positive, axis=0)
        if risk:
            style = style.background_gradient(cmap="RdYlGn_r", subset=risk, axis=0)
        st.dataframe(style.format({
            "H/PA": "{:.1%}", "xH/PA": "{:.1%}", "HR/PA": "{:.2%}", "Avg_EV": "{:.1f}",
            "Hard Hit%": "{:.1%}", "Brl/BIP": "{:.1%}", "BvP Fantasy Score": "{:.1f}",
        }), use_container_width=True, hide_index=True, height=580)
        st.caption("Direct BvP is descriptive and strongly reduced toward neutral when the sample is small.")

with environment_tab:
    lineups = bundle.get("lineups", pd.DataFrame()).copy()
    if not lineups.empty:
        name_lookup = board[["player_id", "Player"]].drop_duplicates("player_id")
        lineups = lineups.merge(name_lookup, on="player_id", how="left")
        st.markdown("#### Lineup status")
        st.dataframe(
            lineups[[column for column in ["Game", "Team", "Order", "Player", "Status", "Source"] if column in lineups.columns]],
            use_container_width=True, hide_index=True, height=420,
        )
    environment = bundle.get("environment", pd.DataFrame())
    if not environment.empty:
        st.markdown("#### Park and first-pitch weather")
        st.dataframe(
            environment.style.format({
                "Temp F": "{:.1f}", "Wind mph": "{:.1f}", "Weather Mult": "{:.3f}",
                "HR L": "{:.0f}", "HR R": "{:.0f}", "Hits L": "{:.0f}", "Hits R": "{:.0f}",
            }), use_container_width=True, hide_index=True,
        )

with notes_tab:
    st.markdown(
        """
        ### Reading the fantasy board

        - **Projected FS** is the expected score using the editable scoring values in the sidebar.
        - **FS Rating** is a 0–100 slate-relative setup score. Rank is driven by the projected fantasy score, not the rating.
        - **Median / percentiles** come from a discrete simulation of plate appearances and scoring events. Hitter fantasy scores are highly skewed, so the mean projection can be higher than the median.
        - **PrizePicks Over probability** is the simulated chance of finishing strictly above the entered line. Integer lines can produce a push probability.
        - The hit-event model blends overall, platoon, recent, opposing-pitcher and pitch-match information. Home-run probability comes from the dedicated HR framework.
        - Runs and RBI blend MLB season rates with batting order, on-base/power profile and the entered team implied total.
        - Stolen bases use MLB season rates and projected on-base opportunity. Players with no season data are shrunk heavily toward the league baseline.
        - Confirmed MLB lineups are preferred. Before posting, the model uses the most recent observed lineup and labels it as a fallback.
        - Retractable-roof weather is neutral unless **Assume retractable roofs open** is enabled.
        - This is a heuristic projection model until calibrated against a large held-out game sample. Treat probabilities as estimates rather than guarantees.
        """
    )
