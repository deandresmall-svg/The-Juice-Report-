from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import date, datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PrizePicks Prop Tracker", page_icon="📊", layout="wide")
APP_VERSION = "1.0.0"

COLUMNS = [
    "leg_id", "entry_id", "placed_at", "slate_date", "sport", "league", "player",
    "team", "opponent", "prop_type", "line", "direction", "actual", "result",
    "tier", "entry_type", "entry_fee", "entry_return", "payout_multiplier", "notes",
    "source", "model_probability", "model_edge", "model_score", "confidence",
    "snapshot_timestamp",
]

ALIASES = {
    "leg_id": ["leg_id", "pick_id", "projection_id", "selection_id"],
    "entry_id": ["entry_id", "lineup_id", "contest_id", "slip_id"],
    "placed_at": ["placed_at", "created_at", "submitted_at", "timestamp"],
    "slate_date": ["slate_date", "date", "game_date", "event_date", "start_time"],
    "sport": ["sport", "sport_name"],
    "league": ["league", "league_name", "competition"],
    "player": ["player", "player_name", "name", "description", "athlete", "display_name"],
    "team": ["team", "team_abbr", "team_name", "player_team"],
    "opponent": ["opponent", "opp", "opponent_team"],
    "prop_type": ["prop_type", "market", "stat_type", "stat", "category", "prop"],
    "line": ["line", "projection", "value", "threshold", "line_score"],
    "direction": ["direction", "pick", "selection", "choice", "side", "over_under"],
    "actual": ["actual", "result_value", "final_value", "score", "actual_value"],
    "result": ["result", "status", "outcome", "pick_result", "settlement"],
    "tier": ["tier", "odds_type", "pick_type", "demon_goblin"],
    "entry_type": ["entry_type", "contest_type", "lineup_type", "game_type"],
    "entry_fee": ["entry_fee", "amount", "stake", "entry_amount", "fee"],
    "entry_return": ["entry_return", "payout", "return", "won_amount", "prize"],
    "payout_multiplier": ["payout_multiplier", "multiplier", "payout_mult"],
    "notes": ["notes", "note", "comment"],
    "source": ["source", "import_source"],
    "model_probability": ["model_probability", "model_prob", "probability", "projected_probability"],
    "model_edge": ["model_edge", "edge", "probability_edge", "market_edge"],
    "model_score": ["model_score", "hit_score", "hr_score", "k_score", "overall_score"],
    "confidence": ["confidence", "confidence_score", "model_confidence"],
    "snapshot_timestamp": ["snapshot_timestamp", "generated_at", "prediction_timestamp"],
}

RESULT_MAP = {
    "win": "Win", "won": "Win", "w": "Win", "green": "Win", "correct": "Win",
    "loss": "Loss", "lost": "Loss", "l": "Loss", "red": "Loss", "incorrect": "Loss",
    "push": "Push", "tie": "Push", "tied": "Push",
    "void": "Void", "dnp": "Void", "reboot": "Void", "cancelled": "Void", "canceled": "Void",
    "pending": "Pending", "open": "Pending", "live": "Pending",
}
DIRECTION_MAP = {
    "more": "More", "over": "More", "higher": "More", "yes": "More",
    "less": "Less", "under": "Less", "lower": "Less", "no": "Less",
}
PROP_MAP = {
    "hits": "Hits", "hit": "Hits", "1+ hit": "Hits", "batter hits": "Hits",
    "home runs": "Home Runs", "home run": "Home Runs", "hr": "Home Runs", "hrs": "Home Runs",
    "strikeouts": "Pitcher Strikeouts", "pitcher strikeouts": "Pitcher Strikeouts", "k": "Pitcher Strikeouts",
    "earned runs": "Pitcher Earned Runs", "pitcher earned runs": "Pitcher Earned Runs",
    "outs": "Pitcher Outs", "pitcher outs": "Pitcher Outs", "pitching outs": "Pitcher Outs",
    "fantasy score": "Fantasy Score", "hitter fantasy score": "Hitter Fantasy Score",
    "pitcher fantasy score": "Pitcher Fantasy Score", "total bases": "Total Bases",
    "runs+rbi": "Runs + RBI", "runs + rbi": "Runs + RBI", "runs": "Runs", "rbi": "RBI",
    "walks": "Walks", "stolen bases": "Stolen Bases",
}


def key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def norm_name(value: Any) -> str:
    text = re.sub(r"[^\w\s-]", "", str(value or "").strip().lower())
    return re.sub(r"\s+", " ", text)


def number(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace("$", "").replace(",", "").replace("%", "")
    if not text or text.lower() in {"none", "nan", "null", "n/a", "-"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+", text)
        return float(match.group()) if match else np.nan


def probability(value: Any) -> float:
    val = number(value)
    if pd.isna(val):
        return np.nan
    if val > 1:
        val /= 100
    return float(np.clip(val, 0, 1))


def dt(value: Any) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, (int, float, np.number)):
        val = float(value)
        if val > 10_000_000_000:
            return pd.to_datetime(val, unit="ms", utc=True, errors="coerce")
        if val > 1_000_000_000:
            return pd.to_datetime(val, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def norm_result(value: Any) -> str:
    text = str(value or "").strip().lower()
    return RESULT_MAP.get(text, str(value or "").strip().title()) if text else ""


def norm_direction(value: Any) -> str:
    text = str(value or "").strip().lower()
    return DIRECTION_MAP.get(text, str(value or "").strip().title()) if text else ""


def norm_prop(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return PROP_MAP.get(text.lower(), text.title())


def grade(direction: str, line: float, actual: float, existing: str = "") -> str:
    current = norm_result(existing)
    if current in {"Win", "Loss", "Push", "Void"}:
        return current
    if pd.isna(line) or pd.isna(actual):
        return current or "Pending"
    if actual == line:
        return "Push"
    side = norm_direction(direction)
    if side == "More":
        return "Win" if actual > line else "Loss"
    if side == "Less":
        return "Win" if actual < line else "Loss"
    return current or "Pending"


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def make_id(row: pd.Series) -> str:
    existing = str(row.get("leg_id", "") or "").strip()
    if existing and existing.lower() not in {"nan", "none"}:
        return existing
    payload = "|".join([
        str(row.get("entry_id", "") or ""), str(row.get("slate_date", "") or ""),
        norm_name(row.get("player", "")), key(row.get("prop_type", "")),
        str(row.get("line", "") or ""), str(row.get("direction", "") or ""),
    ])
    return "leg_" + hashlib.sha1(payload.encode()).hexdigest()[:16]


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty()
    out = df.copy()
    for col in COLUMNS:
        if col not in out:
            out[col] = np.nan
    for col in ["line", "actual", "entry_fee", "entry_return", "payout_multiplier", "model_score", "confidence"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["model_probability"] = out["model_probability"].map(probability)
    out["model_edge"] = out["model_edge"].map(probability)
    out["placed_at"] = out["placed_at"].map(dt)
    out["snapshot_timestamp"] = out["snapshot_timestamp"].map(dt)
    dates = pd.to_datetime(out["slate_date"], errors="coerce")
    placed = pd.to_datetime(out["placed_at"], errors="coerce", utc=True)
    dates = dates.fillna(placed.dt.tz_convert(None))
    out["slate_date"] = dates.dt.date
    out["player"] = out["player"].fillna("").astype(str).str.strip()
    out["sport"] = out["sport"].fillna("").astype(str).str.strip().str.upper()
    out["league"] = out["league"].fillna("").astype(str).str.strip().str.upper()
    out["prop_type"] = out["prop_type"].map(norm_prop)
    out["direction"] = out["direction"].map(norm_direction)
    out["result"] = out["result"].map(norm_result)
    out["tier"] = out["tier"].fillna("").astype(str).str.strip().str.title()
    out["entry_type"] = out["entry_type"].fillna("").astype(str).str.strip().str.title()
    out["source"] = out["source"].fillna("Unknown").astype(str).str.strip()
    out["result"] = [grade(d, l, a, r) for d, l, a, r in zip(out.direction, out.line, out.actual, out.result)]
    out["leg_id"] = out.apply(make_id, axis=1)
    out = out[COLUMNS].drop_duplicates("leg_id", keep="last")
    return out.sort_values(["slate_date", "placed_at"], ascending=[False, False], na_position="last").reset_index(drop=True)


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [x for x in frames if x is not None and not x.empty]
    return finalize(pd.concat(valid, ignore_index=True, sort=False)) if valid else empty()


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    def walk(prefix: str, value: Any, depth: int = 0) -> None:
        if depth > 4:
            return
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}_{k}" if prefix else str(k), v, depth + 1)
        elif not isinstance(value, list):
            out[key(prefix)] = value
    walk("", record)
    return out


def first(mapping: dict[str, Any], options: Iterable[str]) -> Any:
    normalized = {key(k): v for k, v in mapping.items()}
    for option in options:
        if key(option) in normalized:
            return normalized[key(option)]
    return None


def map_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    combined = {**record, **flatten(record)}
    mapped = {col: first(combined, [col, *ALIASES[col]]) for col in COLUMNS}
    mapped["direction"] = norm_direction(mapped["direction"])
    mapped["result"] = norm_result(mapped["result"])
    mapped["prop_type"] = norm_prop(mapped["prop_type"])
    mapped["source"] = mapped["source"] or source
    return mapped


def looks_like_leg(record: dict[str, Any]) -> bool:
    keys = set(flatten(record)) | {key(k) for k in record}
    has_player = any(any(token in k for token in ["player", "athlete", "description"]) for k in keys)
    has_prop = any(any(token in k for token in ["stat_type", "prop_type", "market", "category", "stat"]) for k in keys)
    has_line_or_pick = any(any(token in k for token in ["line", "projection", "threshold", "direction", "selection", "choice"]) for k in keys)
    return has_player and has_prop and has_line_or_pick


def extract_json(payload: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    def walk(value: Any, context: dict[str, Any] | None = None, depth: int = 0) -> None:
        if depth > 12:
            return
        if isinstance(value, dict):
            if id(value) in seen:
                return
            seen.add(id(value))
            if looks_like_leg(value):
                merged = dict(context or {})
                merged.update(value)
                records.append(merged)
            next_context = dict(context or {})
            for field in ["entry_id", "lineup_id", "created_at", "placed_at", "entry_fee", "amount", "payout", "entry_type", "status"]:
                candidate = first(value, [field])
                if candidate is not None and not isinstance(candidate, (dict, list)):
                    next_context[field] = candidate
            for child in value.values():
                walk(child, next_context, depth + 1)
        elif isinstance(value, list):
            for child in value:
                walk(child, context, depth + 1)
    walk(payload)
    return records


def normalize_csv(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    source_cols = {key(c): c for c in raw.columns}
    out = pd.DataFrame(index=raw.index)
    for col, options in ALIASES.items():
        chosen = next((source_cols[key(o)] for o in [col, *options] if key(o) in source_cols), None)
        out[col] = raw[chosen] if chosen else np.nan
    out["source"] = out["source"].fillna(source)
    return finalize(out)


def parse_upload(file: Any) -> tuple[pd.DataFrame, str]:
    data = file.getvalue()
    if file.name.lower().endswith(".csv"):
        raw = pd.read_csv(io.BytesIO(data))
        return normalize_csv(raw, f"CSV: {file.name}"), f"Imported {len(raw):,} CSV rows."
    payload = json.loads(data.decode("utf-8"))
    records = extract_json(payload)
    if not records and isinstance(payload, list):
        records = [x for x in payload if isinstance(x, dict)]
    frame = finalize(pd.DataFrame([map_record(r, f"JSON: {file.name}") for r in records]))
    return frame, f"Detected {len(frame):,} potential prop legs."


def csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    for col in ["placed_at", "snapshot_timestamp"]:
        out[col] = pd.to_datetime(out[col], errors="coerce", utc=True).astype("string")
    return out.to_csv(index=False).encode()


def binary(series: pd.Series) -> pd.Series:
    return series.map({"Win": 1.0, "Loss": 0.0})


def pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


def performance(df: pd.DataFrame, group: str, minimum: int = 1) -> pd.DataFrame:
    work = df[df.result.isin(["Win", "Loss"])].copy()
    if work.empty or group not in work:
        return pd.DataFrame()
    work["_win"] = binary(work.result)
    result = work.groupby(group, dropna=False).agg(
        Legs=("_win", "size"), Wins=("_win", "sum"), Win_Rate=("_win", "mean"),
        Avg_Line=("line", "mean"), Avg_Model_Prob=("model_probability", "mean"),
        Avg_Model_Edge=("model_edge", "mean"),
    ).reset_index()
    return result[result.Legs >= minimum].sort_values(["Win_Rate", "Legs"], ascending=[False, False])


def entry_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df.entry_id.fillna("").astype(str).str.strip().ne("")].copy()
    rows = []
    for entry_id, group in work.groupby("entry_id"):
        graded = group[group.result.isin(["Win", "Loss", "Push", "Void"])]
        fee = pd.to_numeric(group.entry_fee, errors="coerce").dropna()
        ret = pd.to_numeric(group.entry_return, errors="coerce").dropna()
        dates = pd.to_datetime(group.slate_date, errors="coerce").dropna()
        rows.append({
            "entry_id": entry_id, "date": dates.min().date() if not dates.empty else pd.NaT,
            "legs": len(group), "wins": int((graded.result == "Win").sum()),
            "losses": int((graded.result == "Loss").sum()), "pushes": int((graded.result == "Push").sum()),
            "voids": int((graded.result == "Void").sum()),
            "entry_type": next((x for x in group.entry_type.astype(str) if x and x != "nan"), ""),
            "entry_fee": float(fee.iloc[0]) if not fee.empty else np.nan,
            "entry_return": float(ret.iloc[0]) if not ret.empty else np.nan,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["profit"] = result.entry_return - result.entry_fee
        result["roi"] = result.profit / result.entry_fee.replace(0, np.nan)
    return result


def calibration(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    work = df[df.result.isin(["Win", "Loss"])].copy()
    work.model_probability = pd.to_numeric(work.model_probability, errors="coerce")
    work = work.dropna(subset=["model_probability"])
    if work.empty:
        return pd.DataFrame(), np.nan
    work["_win"] = binary(work.result)
    work["Bucket"] = pd.cut(work.model_probability, [0, .5, .55, .6, .65, .7, .75, 1.0001],
                            labels=["<50%", "50–54%", "55–59%", "60–64%", "65–69%", "70–74%", "75%+"],
                            include_lowest=True, right=False)
    table = work.groupby("Bucket", observed=False).agg(
        Sample=("_win", "size"), Average_Probability=("model_probability", "mean"),
        Actual_Win_Rate=("_win", "mean")).reset_index()
    table["Calibration_Gap"] = table.Actual_Win_Rate - table.Average_Probability
    return table, float(np.mean((work.model_probability - work._win) ** 2))


def edge_buckets(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df.result.isin(["Win", "Loss"])].copy()
    work.model_edge = pd.to_numeric(work.model_edge, errors="coerce")
    work = work.dropna(subset=["model_edge"])
    if work.empty:
        return pd.DataFrame()
    work["_win"] = binary(work.result)
    work["Bucket"] = pd.cut(work.model_edge, [-1, 0, .025, .05, .075, .1, 1.0001],
                            labels=["Negative", "0–2.4%", "2.5–4.9%", "5–7.4%", "7.5–9.9%", "10%+"],
                            include_lowest=True, right=False)
    return work.groupby("Bucket", observed=False).agg(
        Sample=("_win", "size"), Average_Edge=("model_edge", "mean"), Actual_Win_Rate=("_win", "mean")).reset_index()


def normalize_snapshot(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [key(c) for c in frame.columns]
    choices = {
        "slate_date": ["slate_date", "date", "gamedate"], "player": ["player", "pitcher"],
        "prop_type": ["prop_type", "target", "market"],
        "model_probability": ["model_probability", "model_1plus_hit", "model_1plus_hr", "over_probability"],
        "model_edge": ["model_edge", "model_market_edge"],
        "model_score": ["model_score", "hit_score", "hr_score", "k_score", "overall_score"],
        "confidence": ["confidence", "confidence_score"],
        "snapshot_timestamp": ["snapshot_timestamp", "generated_at", "prediction_timestamp"],
    }
    out = pd.DataFrame(index=frame.index)
    for target, options in choices.items():
        chosen = next((c for c in options if c in frame.columns), None)
        out[target] = frame[chosen] if chosen else np.nan
    if out.prop_type.isna().all():
        if "model_1plus_hit" in frame: out.prop_type = "Hits"
        elif "model_1plus_hr" in frame: out.prop_type = "Home Runs"
        elif "proj_k" in frame: out.prop_type = "Pitcher Strikeouts"
    out.slate_date = pd.to_datetime(out.slate_date, errors="coerce").dt.date
    out["player_key"] = out.player.map(norm_name)
    out["prop_key"] = out.prop_type.map(lambda x: key(norm_prop(x)))
    out.model_probability = out.model_probability.map(probability)
    out.model_edge = out.model_edge.map(probability)
    out.model_score = pd.to_numeric(out.model_score, errors="coerce")
    out.confidence = pd.to_numeric(out.confidence, errors="coerce")
    out.snapshot_timestamp = out.snapshot_timestamp.map(dt)
    return out


def merge_snapshot(ledger: pd.DataFrame, snapshot: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if ledger.empty or snapshot.empty:
        return ledger, 0
    left = ledger.copy()
    left["player_key"] = left.player.map(norm_name)
    left["prop_key"] = left.prop_type.map(lambda x: key(norm_prop(x)))
    left.slate_date = pd.to_datetime(left.slate_date, errors="coerce").dt.date
    snap = snapshot.sort_values("snapshot_timestamp").drop_duplicates(["slate_date", "player_key", "prop_key"], keep="last")
    merged = left.merge(snap, on=["slate_date", "player_key", "prop_key"], how="left", suffixes=("", "_snap"))
    matched = int(merged.get("model_probability_snap", pd.Series(dtype=float)).notna().sum())
    for col in ["model_probability", "model_edge", "model_score", "confidence", "snapshot_timestamp"]:
        other = f"{col}_snap"
        if other in merged:
            merged[col] = merged[other].combine_first(merged[col])
    return finalize(merged[[c for c in COLUMNS if c in merged]]), matched


def show_ledger(df: pd.DataFrame, height: int = 480) -> None:
    if df.empty:
        st.info("No prop legs match the current filters.")
        return
    cols = ["slate_date", "player", "sport", "prop_type", "line", "direction", "actual", "result", "tier", "entry_id", "model_probability", "model_edge", "confidence"]
    display = df[[c for c in cols if c in df]].rename(columns={
        "slate_date":"Date", "player":"Player", "sport":"Sport", "prop_type":"Prop", "line":"Line",
        "direction":"Pick", "actual":"Actual", "result":"Result", "tier":"Tier", "entry_id":"Entry",
        "model_probability":"Model Prob", "model_edge":"Model Edge", "confidence":"Confidence"})
    st.dataframe(display.style.format({"Line":"{:.2f}", "Actual":"{:.2f}", "Model Prob":"{:.1%}", "Model Edge":"{:+.1%}", "Confidence":"{:.1f}"}),
                 width="stretch", hide_index=True, height=height)


if "ledger" not in st.session_state:
    st.session_state.ledger = empty()

st.title("📊 PrizePicks Individual Prop Tracker")
st.caption("Track each leg separately, measure win rate by prop type, and validate your model probabilities.")

with st.sidebar:
    st.markdown("## Data")
    st.caption("Cloud storage is temporary. Download the master CSV after every session.")
    st.download_button("Download current master CSV", csv_bytes(st.session_state.ledger),
                       file_name=f"prizepicks_master_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                       mime="text/csv", width="stretch")
    if st.button("Clear session ledger", width="stretch"):
        st.session_state.ledger = empty(); st.rerun()

    if not st.session_state.ledger.empty:
        st.markdown("### Filters")
        frame = st.session_state.ledger
        dates = pd.to_datetime(frame.slate_date, errors="coerce").dropna()
        min_date = dates.min().date() if not dates.empty else date.today()
        max_date = dates.max().date() if not dates.empty else date.today()
        date_range = st.date_input("Slate date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        sports = sorted(x for x in frame.sport.astype(str).unique() if x)
        props = sorted(x for x in frame.prop_type.astype(str).unique() if x)
        directions = sorted(x for x in frame.direction.astype(str).unique() if x)
        results = sorted(x for x in frame.result.astype(str).unique() if x)
        chosen_sports = st.multiselect("Sport", sports, sports)
        chosen_props = st.multiselect("Prop type", props, props)
        chosen_directions = st.multiselect("Direction", directions, directions)
        chosen_results = st.multiselect("Result", results, results)
    else:
        date_range = None; chosen_sports = chosen_props = chosen_directions = chosen_results = []

filtered = st.session_state.ledger.copy()
if not filtered.empty:
    if isinstance(date_range, tuple) and len(date_range) == 2:
        values = pd.to_datetime(filtered.slate_date, errors="coerce").dt.date
        filtered = filtered[values.between(*date_range)]
    if chosen_sports: filtered = filtered[filtered.sport.isin(chosen_sports)]
    if chosen_props: filtered = filtered[filtered.prop_type.isin(chosen_props)]
    if chosen_directions: filtered = filtered[filtered.direction.isin(chosen_directions)]
    if chosen_results: filtered = filtered[filtered.result.isin(chosen_results)]

overview, breakdown, players, entries_tab, model, import_tab = st.tabs(["Overview", "Prop breakdown", "Players", "Entries", "Model validation", "Import & manage"])

with overview:
    graded = filtered[filtered.result.isin(["Win", "Loss"])]
    wins = int((graded.result == "Win").sum()); losses = int((graded.result == "Loss").sum())
    win_rate = wins / (wins + losses) if wins + losses else np.nan
    entry_df = entry_summary(filtered)
    staked = entry_df.entry_fee.sum(min_count=1) if not entry_df.empty else np.nan
    returned = entry_df.entry_return.sum(min_count=1) if not entry_df.empty else np.nan
    profit = returned - staked if pd.notna(staked) and pd.notna(returned) else np.nan
    roi = profit / staked if pd.notna(profit) and staked else np.nan
    row1 = st.columns(5)
    row1[0].metric("Tracked legs", f"{len(filtered):,}"); row1[1].metric("Graded legs", f"{wins+losses:,}")
    row1[2].metric("Wins", f"{wins:,}"); row1[3].metric("Losses", f"{losses:,}"); row1[4].metric("Leg win rate", pct(win_rate))
    row2 = st.columns(5)
    row2[0].metric("Pushes", f"{(filtered.result=='Push').sum():,}"); row2[1].metric("Voids", f"{(filtered.result=='Void').sum():,}")
    row2[2].metric("Pending", f"{(filtered.result=='Pending').sum():,}"); row2[3].metric("Entry profit", "—" if pd.isna(profit) else f"${profit:,.2f}")
    row2[4].metric("Entry ROI", pct(roi))
    st.markdown("### Recent prop legs"); show_ledger(filtered.head(100))

with breakdown:
    group_map = {"Prop type":"prop_type", "Sport":"sport", "Direction":"direction", "Tier":"tier", "Entry type":"entry_type", "Team":"team", "Opponent":"opponent"}
    c1, c2 = st.columns([2,1]); label = c1.selectbox("Break results down by", list(group_map)); minimum = c2.number_input("Minimum sample", 1, 500, 3)
    table = performance(filtered, group_map[label], int(minimum))
    if table.empty: st.info("Not enough graded legs for this breakdown.")
    else:
        st.dataframe(table.style.format({"Wins":"{:.0f}", "Win_Rate":"{:.1%}", "Avg_Line":"{:.2f}", "Avg_Model_Prob":"{:.1%}", "Avg_Model_Edge":"{:+.1%}"}), width="stretch", hide_index=True)
        st.bar_chart(table.set_index(group_map[label]).Win_Rate.sort_values(ascending=False))

with players:
    table = performance(filtered, "player", 1)
    if table.empty: st.info("No graded player results yet.")
    else: st.dataframe(table.style.format({"Wins":"{:.0f}", "Win_Rate":"{:.1%}", "Avg_Line":"{:.2f}", "Avg_Model_Prob":"{:.1%}", "Avg_Model_Edge":"{:+.1%}"}), width="stretch", hide_index=True, height=560)

with entries_tab:
    st.caption("Leg win rate and entry profit are different. Individual parlay legs do not have standalone dollar profit.")
    table = entry_summary(filtered)
    if table.empty: st.info("Entry totals appear after imported rows contain entry IDs and fee/return values.")
    else:
        settled = table.dropna(subset=["entry_fee", "entry_return"])
        staked = settled.entry_fee.sum(); returned = settled.entry_return.sum(); profit = returned-staked; roi = profit/staked if staked else np.nan
        cols = st.columns(4); cols[0].metric("Entries", len(table)); cols[1].metric("Staked", f"${staked:,.2f}"); cols[2].metric("Returned", f"${returned:,.2f}"); cols[3].metric("ROI", pct(roi))
        st.dataframe(table.style.format({"entry_fee":"${:,.2f}", "entry_return":"${:,.2f}", "profit":"${:+,.2f}", "roi":"{:+.1%}"}), width="stretch", hide_index=True)

with model:
    table, brier = calibration(filtered)
    c1, c2 = st.columns(2); c1.metric("Legs with model probability", int(filtered.model_probability.notna().sum())); c2.metric("Brier score", "—" if pd.isna(brier) else f"{brier:.3f}")
    if table.empty: st.info("Merge dashboard pregame snapshots to evaluate calibration.")
    else: st.dataframe(table.style.format({"Average_Probability":"{:.1%}", "Actual_Win_Rate":"{:.1%}", "Calibration_Gap":"{:+.1%}"}), width="stretch", hide_index=True)
    st.markdown("### Results by model edge")
    edge = edge_buckets(filtered)
    if edge.empty: st.info("No model-edge data yet.")
    else: st.dataframe(edge.style.format({"Average_Edge":"{:+.1%}", "Actual_Win_Rate":"{:.1%}"}), width="stretch", hide_index=True)
    st.markdown("### Merge a dashboard pregame snapshot")
    upload = st.file_uploader("Upload Hits, HR, pitcher, or fantasy snapshot CSV", type=["csv"], key="snapshot")
    if upload:
        snap = normalize_snapshot(pd.read_csv(upload)); st.caption(f"Snapshot rows detected: {len(snap):,}")
        if st.button("Merge snapshot into tracked legs", width="stretch"):
            st.session_state.ledger, matched = merge_snapshot(st.session_state.ledger, snap)
            st.success(f"Matched model data to {matched:,} legs."); st.rerun()

with import_tab:
    st.markdown("### Upload CSV or JSON history")
    uploads = st.file_uploader("Upload one or more files", type=["csv", "json"], accept_multiple_files=True, key="uploads")
    if uploads:
        frames = []
        for file in uploads:
            try:
                frame, message = parse_upload(file); frames.append(frame); st.caption(f"{file.name}: {message}")
            except Exception as exc: st.error(f"{file.name}: {exc}")
        preview = combine(frames); show_ledger(preview.head(200), 380)
        if not preview.empty and st.button("Add uploaded rows to master ledger", width="stretch"):
            st.session_state.ledger = combine([st.session_state.ledger, preview]); st.rerun()

    st.divider(); st.markdown("### Paste copied JSON response")
    st.caption("Copy only the response body. Do not upload HAR files or share cookies, authorization headers, or tokens.")
    pasted = st.text_area("Paste JSON", height=180, placeholder='{"data": [...]}')
    if st.button("Parse pasted JSON", disabled=not pasted.strip(), width="stretch"):
        try:
            payload = json.loads(pasted); records = extract_json(payload)
            if not records and isinstance(payload, list): records = [x for x in payload if isinstance(x, dict)]
            st.session_state.json_preview = finalize(pd.DataFrame([map_record(r, "Pasted JSON") for r in records])); st.rerun()
        except Exception as exc: st.error(f"Could not parse JSON: {exc}")
    preview = st.session_state.get("json_preview", empty())
    if not preview.empty:
        show_ledger(preview.head(200), 360)
        if st.button("Add parsed JSON rows", width="stretch"):
            st.session_state.ledger = combine([st.session_state.ledger, preview]); st.session_state.json_preview = empty(); st.rerun()

    st.divider(); st.markdown("### Add a leg manually")
    with st.form("manual", clear_on_submit=True):
        a,b,c = st.columns(3); slate_date = a.date_input("Slate date", date.today()); player = b.text_input("Player"); sport = c.text_input("Sport", "MLB")
        d,e,f = st.columns(3); prop = d.text_input("Prop type", placeholder="Hits"); line = e.number_input("Line", value=.5, step=.5); direction = f.selectbox("Pick", ["More","Less"])
        g,h,i = st.columns(3); actual_text = g.text_input("Actual", placeholder="Blank if pending"); tier = h.selectbox("Tier", ["","Standard","Goblin","Demon"]); entry_id = i.text_input("Entry ID")
        j,k,l = st.columns(3); fee = j.number_input("Entry fee", min_value=0.0); ret = k.number_input("Entry return", min_value=0.0); entry_type = l.text_input("Entry type")
        notes = st.text_input("Notes"); submitted = st.form_submit_button("Add manual leg", width="stretch")
    if submitted:
        row = pd.DataFrame([{"slate_date":slate_date,"player":player,"sport":sport,"prop_type":prop,"line":line,"direction":direction,"actual":number(actual_text),"tier":tier,"entry_id":entry_id,"entry_fee":fee or np.nan,"entry_return":ret or np.nan,"entry_type":entry_type,"notes":notes,"source":"Manual"}])
        st.session_state.ledger = combine([st.session_state.ledger, finalize(row)]); st.rerun()

    st.divider(); st.markdown("### Edit or remove rows")
    current = st.session_state.ledger
    if current.empty: st.info("No rows loaded.")
    else:
        editable = ["leg_id","entry_id","slate_date","player","sport","prop_type","line","direction","actual","result","tier","entry_type","entry_fee","entry_return","notes"]
        editor = st.data_editor(current[editable], width="stretch", hide_index=True, num_rows="dynamic", key="editor",
            column_config={"result":st.column_config.SelectboxColumn(options=["Pending","Win","Loss","Push","Void"]), "direction":st.column_config.SelectboxColumn(options=["More","Less"]), "slate_date":st.column_config.DateColumn()})
        if st.button("Save edited ledger", width="stretch"):
            preserved = current.drop(columns=editable, errors="ignore")
            updated = pd.concat([editor.reset_index(drop=True), preserved.reset_index(drop=True)], axis=1) if len(editor)==len(preserved) else editor
            st.session_state.ledger = finalize(updated); st.rerun()

    st.divider(); st.markdown("### Import template")
    st.download_button("Download blank leg template", csv_bytes(empty()), file_name="prizepicks_leg_template.csv", mime="text/csv", width="stretch")

st.caption("Confirm imported data against your official account history. Never share login credentials or authentication tokens.")
