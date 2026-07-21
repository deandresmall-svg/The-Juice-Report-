from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pybaseball import cache, statcast


K_BACKFILL_VERSION = "opponent-offense-k-v2"
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
CALLED_STRIKE_DESCRIPTIONS = {"called_strike"}
OUTS_BY_EVENT = {
    "strikeout": 1,
    "strikeout_double_play": 2,
    "field_out": 1,
    "force_out": 1,
    "fielders_choice_out": 1,
    "grounded_into_double_play": 2,
    "double_play": 2,
    "triple_play": 3,
    "sac_fly": 1,
    "sac_bunt": 1,
    "sac_fly_double_play": 2,
}
SUMMARY_COLUMNS = [
    "Date", "GamePK", "PitcherID", "Team", "Opponent", "Hand",
    "Actual_BF", "Actual_K", "Actual_Outs", "Actual_Pitches",
    "Swings", "Whiffs", "CalledStrikes", "StarterLengthOK",
]


def mode_text(series: pd.Series, default: str = "") -> str:
    values = series.dropna().astype(str)
    return str(values.mode().iloc[0]) if not values.empty else default


def prepare_minimal_statcast(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep only fields required by the K backfill and derive pitch flags."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    needed = [
        "pitcher", "game_pk", "at_bat_number", "game_date", "home_team",
        "away_team", "inning_topbot", "description", "events", "p_throws",
        "outs_on_play",
    ]
    work = raw.reindex(columns=needed).copy()
    for column in ["pitcher", "game_pk", "at_bat_number", "outs_on_play"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    top = work["inning_topbot"].eq("Top").fillna(False)
    work["batter_team"] = np.where(top, work["away_team"], work["home_team"])
    work["pitcher_team"] = np.where(top, work["home_team"], work["away_team"])
    description = work["description"].fillna("").astype(str)
    events = work["events"].fillna("").astype(str)
    work["is_pa_end"] = events.ne("")
    work["is_k"] = events.isin(STRIKEOUT_EVENTS)
    work["is_swing"] = description.isin({
        "hit_into_play", "foul", "foul_tip", "foul_bunt", "missed_bunt",
        "swinging_strike", "swinging_strike_blocked",
    })
    work["is_whiff"] = description.isin(WHIFF_DESCRIPTIONS)
    work["is_called_strike"] = description.isin(CALLED_STRIKE_DESCRIPTIONS)
    work["pa_key"] = (
        work["game_pk"].astype("Int64").astype(str)
        + "-"
        + work["at_bat_number"].astype("Int64").astype(str)
    )
    event_outs = events.map(OUTS_BY_EVENT).fillna(0.0)
    work["outs_recorded"] = (
        work["outs_on_play"].where(work["outs_on_play"].notna(), event_outs)
        .fillna(0.0).clip(0, 3)
    )
    return work


def summarize_starts(prepared: pd.DataFrame) -> pd.DataFrame:
    """Reduce pitch-level rows to one row per starting pitcher/game."""
    if prepared is None or prepared.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    work = prepared[
        prepared["game_date"].notna()
        & prepared["game_pk"].notna()
        & prepared["pitcher"].notna()
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    first_ab = (
        work.groupby(["game_pk", "pitcher_team", "pitcher"], dropna=False)["at_bat_number"]
        .min().reset_index(name="FirstAB")
    )
    starters = (
        first_ab.sort_values(["game_pk", "pitcher_team", "FirstAB"])
        .drop_duplicates(["game_pk", "pitcher_team"], keep="first")
        [["game_pk", "pitcher_team", "pitcher"]]
    )
    work = work.merge(
        starters.assign(IsStarter=True),
        on=["game_pk", "pitcher_team", "pitcher"],
        how="left",
    )
    work = work[work["IsStarter"].fillna(False)].copy()
    summaries = (
        work.groupby(["game_date", "game_pk", "pitcher", "pitcher_team"], dropna=False)
        .agg(
            Opponent=("batter_team", lambda s: mode_text(s, "")),
            Hand=("p_throws", lambda s: mode_text(s, "R")),
            Actual_BF=("pa_key", lambda s: int(s[work.loc[s.index, "is_pa_end"]].nunique())),
            Actual_K=("is_k", "sum"),
            Actual_Outs=("outs_recorded", "sum"),
            Actual_Pitches=("pitcher", "size"),
            Swings=("is_swing", "sum"),
            Whiffs=("is_whiff", "sum"),
            CalledStrikes=("is_called_strike", "sum"),
        )
        .reset_index()
        .rename(columns={
            "game_date": "Date", "game_pk": "GamePK", "pitcher": "PitcherID",
            "pitcher_team": "Team",
        })
    )
    numeric = [
        "GamePK", "PitcherID", "Actual_BF", "Actual_K", "Actual_Outs",
        "Actual_Pitches", "Swings", "Whiffs", "CalledStrikes",
    ]
    for column in numeric:
        summaries[column] = pd.to_numeric(summaries[column], errors="coerce").fillna(0)
    summaries["StarterLengthOK"] = summaries["Actual_Pitches"].ge(40) | summaries["Actual_Outs"].ge(9)
    summaries["Date"] = pd.to_datetime(summaries["Date"]).dt.strftime("%Y-%m-%d")
    return summaries[SUMMARY_COLUMNS].sort_values(["Date", "GamePK", "Team"])


def date_chunks(start: pd.Timestamp, end: pd.Timestamp, days: int):
    current = start
    while current <= end:
        chunk_end = min(current + pd.Timedelta(days=days - 1), end)
        yield current, chunk_end
        current = chunk_end + pd.Timedelta(days=1)


def download_start_summaries(
    history_start: pd.Timestamp,
    target_end: pd.Timestamp,
    checkpoint: Path,
    state_path: Path,
    chunk_days: int,
) -> pd.DataFrame:
    """Download/resume small Statcast chunks and checkpoint starter summaries."""
    existing = pd.read_csv(checkpoint) if checkpoint.exists() else pd.DataFrame(columns=SUMMARY_COLUMNS)
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
    else:
        state = {}
    completed = set(state.get("completed_ranges", []))
    chunks = list(date_chunks(history_start, target_end, chunk_days))
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        range_key = f"{chunk_start.date()}:{chunk_end.date()}"
        if range_key in completed:
            print(f"[{index}/{len(chunks)}] Resume skip {range_key}", flush=True)
            continue
        print(f"[{index}/{len(chunks)}] Downloading {range_key}", flush=True)
        raw = statcast(
            start_dt=str(chunk_start.date()),
            end_dt=str(chunk_end.date()),
            verbose=False,
            parallel=False,
        )
        prepared = prepare_minimal_statcast(raw)
        chunk_summaries = summarize_starts(prepared)
        if not chunk_summaries.empty:
            existing = pd.concat([existing, chunk_summaries], ignore_index=True)
            existing = (
                existing.drop_duplicates(["Date", "GamePK", "PitcherID"], keep="last")
                .sort_values(["Date", "GamePK", "PitcherID"])
                .reset_index(drop=True)
            )
        existing.to_csv(checkpoint, index=False)
        completed.add(range_key)
        state_path.write_text(
            json.dumps({"completed_ranges": sorted(completed)}, indent=2),
            encoding="utf-8",
        )
        del raw, prepared, chunk_summaries
        gc.collect()
        print(f"    checkpoint: {len(existing):,} starter rows", flush=True)
    return existing


def build_backfill_from_summaries(
    starts: pd.DataFrame,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> pd.DataFrame:
    starts = starts.copy()
    starts["Date"] = pd.to_datetime(starts["Date"], errors="coerce")
    for column in ["StarterLengthOK"]:
        if starts[column].dtype != bool:
            starts[column] = starts[column].astype(str).str.lower().eq("true")
    starts = starts.sort_values(["Date", "GamePK", "PitcherID"]).reset_index(drop=True)
    eligible = starts[starts["StarterLengthOK"]].copy()
    rows: list[dict] = []
    targets = eligible[eligible["Date"].between(target_start, target_end)]
    for number, current in enumerate(targets.itertuples(index=False), start=1):
        current_date = pd.Timestamp(current.Date).normalize()
        prior = eligible[eligible["Date"].lt(current_date)]
        pitcher_prior = prior[prior["PitcherID"].eq(current.PitcherID)].sort_values("Date")
        if len(pitcher_prior) < 2:
            continue
        recent = pitcher_prior.tail(5)
        league_bf = max(float(prior["Actual_BF"].sum()), 1.0)
        league_k_rate = float(prior["Actual_K"].sum() / league_bf)
        league_whiff = float(prior["Whiffs"].sum() / max(prior["Swings"].sum(), 1.0))
        league_csw = float(
            (prior["Whiffs"].sum() + prior["CalledStrikes"].sum())
            / max(prior["Actual_Pitches"].sum(), 1.0)
        )
        pitcher_bf = float(pitcher_prior["Actual_BF"].sum())
        recent_bf = float(recent["Actual_BF"].sum())
        pitcher_k = (float(pitcher_prior["Actual_K"].sum()) + league_k_rate * 120.0) / (pitcher_bf + 120.0)
        recent_k = (float(recent["Actual_K"].sum()) + pitcher_k * 55.0) / (recent_bf + 55.0)
        whiff = (
            float(pitcher_prior["Whiffs"].sum()) + league_whiff * 180.0
        ) / (float(pitcher_prior["Swings"].sum()) + 180.0)
        csw = (
            float(pitcher_prior["Whiffs"].sum() + pitcher_prior["CalledStrikes"].sum())
            + league_csw * 300.0
        ) / (float(pitcher_prior["Actual_Pitches"].sum()) + 300.0)
        opponent_prior = prior[
            prior["Opponent"].eq(current.Opponent) & prior["Hand"].eq(current.Hand)
        ]
        opponent_bf = float(opponent_prior["Actual_BF"].sum())
        opponent_k = (
            float(opponent_prior["Actual_K"].sum()) + league_k_rate * 220.0
        ) / (opponent_bf + 220.0)
        outs = pd.to_numeric(pitcher_prior["Actual_Outs"], errors="coerce").dropna()
        rows.append({
            "BackfillVersion": K_BACKFILL_VERSION,
            "Date": str(current_date.date()),
            "GamePK": int(current.GamePK),
            "PitcherID": int(current.PitcherID),
            "Team": str(current.Team),
            "Opponent": str(current.Opponent),
            "Hand": str(current.Hand),
            "Season_BF": float(pitcher_prior["Actual_BF"].mean()),
            "Recent_BF": float(recent["Actual_BF"].mean()),
            "Last_Start_Pitches": float(pitcher_prior.iloc[-1]["Actual_Pitches"]),
            "Last3_Pitches": float(pitcher_prior.tail(3)["Actual_Pitches"].mean()),
            "Recent_Outs": float(recent["Actual_Outs"].mean()),
            "Outs_SD": float(outs.tail(8).std(ddof=0)) if len(outs) > 1 else 3.0,
            "Short_Hook_Rate": float(pitcher_prior["Actual_Outs"].lt(15).mean()),
            "Prior_Starts": int(len(pitcher_prior)),
            "Pitcher_K_Rate": pitcher_k,
            "Recent_K_Rate": recent_k,
            "Whiff_Pct": whiff,
            "CSW_Pct": csw,
            "Opponent_K_Rate": opponent_k,
            "Hand_L": 1.0 if str(current.Hand) == "L" else 0.0,
            "Actual_BF": float(current.Actual_BF),
            "Actual_K": float(current.Actual_K),
            "Actual_K_Rate": float(current.Actual_K / max(current.Actual_BF, 1.0)),
            "Actual_Outs": float(current.Actual_Outs),
            "Actual_Pitches": float(current.Actual_Pitches),
        })
        if number % 250 == 0:
            print(f"Built features for {number:,}/{len(targets):,} targets", flush=True)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Date", "GamePK", "PitcherID"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a memory-safe chronological pitcher K backfill.")
    parser.add_argument("--start", required=True, help="First target date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Last target date, YYYY-MM-DD")
    parser.add_argument("--output", default="pitcher_k_chronological_backfill_v2.csv")
    parser.add_argument("--chunk-days", type=int, default=7, help="Statcast download days held in memory at once")
    parser.add_argument("--warmup-days", type=int, default=90, help="Pregame history before the target start")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Checkpoint folder; defaults to pitcher_k_backfill_work_<target year>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_start = pd.Timestamp(args.start).normalize()
    target_end = pd.Timestamp(args.end).normalize()
    if target_end < target_start:
        raise SystemExit("--end must be on or after --start")
    if not 1 <= args.chunk_days <= 14:
        raise SystemExit("--chunk-days must be between 1 and 14")
    work_dir = Path(args.work_dir or f"pitcher_k_backfill_work_{target_start.year}")
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = work_dir / "starter_summaries.csv"
    state_path = work_dir / "download_state.json"
    history_start = target_start - pd.Timedelta(days=max(args.warmup_days, 1))
    cache.enable()
    print(f"Target dates: {target_start.date()} through {target_end.date()}")
    print(f"Statcast history: {history_start.date()} through {target_end.date()}")
    print(f"Download chunk: {args.chunk_days} days")
    starts = download_start_summaries(
        history_start, target_end, checkpoint, state_path, args.chunk_days
    )
    print(f"Building leakage-safe features from {len(starts):,} summarized starts...")
    backfill = build_backfill_from_summaries(starts, target_start, target_end)
    output = Path(args.output)
    backfill.to_csv(output, index=False)
    if backfill.empty:
        print(f"Finished, but no eligible target starts were produced: {output.resolve()}")
    else:
        print(f"Finished: {output.resolve()}")
        print(f"Rows: {len(backfill):,}")
        print(f"Dates: {backfill['Date'].min()} through {backfill['Date'].max()}")
        print(f"Schema: {K_BACKFILL_VERSION}")


if __name__ == "__main__":
    main()
