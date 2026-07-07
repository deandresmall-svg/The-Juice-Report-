from __future__ import annotations

import hashlib
import html as html_lib
import io
import json
import re
from datetime import date, datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PrizePicks Prop Tracker", page_icon="📊", layout="wide")
APP_VERSION = "1.1.0"

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
    "earned runs": "Pitcher Earned Runs", "earned runs allowed": "Pitcher Earned Runs", "pitcher earned runs": "Pitcher Earned Runs",
    "outs": "Pitcher Outs", "pitcher outs": "Pitcher Outs", "pitching outs": "Pitcher Outs",
    "fantasy score": "Fantasy Score", "hitter fs": "Hitter Fantasy Score", "hitter fantasy score": "Hitter Fantasy Score",
    "pitcher fs": "Pitcher Fantasy Score", "pitcher fantasy score": "Pitcher Fantasy Score", "total bases": "Total Bases",
    "runs+rbi": "Runs + RBI", "runs + rbi": "Runs + RBI", "runs": "Runs", "rbi": "RBI",
    "walks": "Walks", "stolen bases": "Stolen Bases",
}


def _as_text(value: Any) -> str:
    """Convert a scalar to text without evaluating pd.NA as a boolean."""
    if value is None:
        return ""
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return ""
    except Exception:
        pass
    return str(value).strip()


def key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _as_text(value).lower()).strip("_")


def norm_name(value: Any) -> str:
    text = re.sub(r"[^\w\s-]", "", _as_text(value).lower())
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


def signed_rate(value: Any) -> float:
    """Normalize a signed probability edge without clipping negatives to zero."""
    val = number(value)
    if pd.isna(val):
        return np.nan
    if abs(val) > 1:
        val /= 100
    return float(np.clip(val, -1, 1))


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
    raw = _as_text(value)
    text = raw.lower()
    return RESULT_MAP.get(text, raw.title()) if text else ""


def norm_direction(value: Any) -> str:
    raw = _as_text(value)
    text = raw.lower()
    return DIRECTION_MAP.get(text, raw.title()) if text else ""


def norm_prop(value: Any) -> str:
    text = re.sub(r"\s+", " ", _as_text(value))
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
    out["model_edge"] = out["model_edge"].map(signed_rate)
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
    name = file.name.lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(io.BytesIO(data))
        return normalize_csv(raw, f"CSV: {file.name}"), f"Imported {len(raw):,} CSV rows."
    if name.endswith(".txt"):
        text = data.decode("utf-8", errors="replace")
        frame = parse_prizepicks_page_text(text, f"Page text: {file.name}")
        return frame, f"Detected {len(frame):,} expanded PrizePicks detail legs from page text."
    payload = json.loads(data.decode("utf-8", errors="replace"))
    if isinstance(payload, dict) and {"detail_text", "detail_html", "elements"}.issubset(payload.keys()):
        frame = parse_prizepicks_detail_inspector(payload, f"Detail inspector: {file.name}")
        return frame, f"Detected {len(frame):,} expanded PrizePicks detail legs from inspector JSON."
    records = extract_json(payload)
    if not records and isinstance(payload, list):
        records = [x for x in payload if isinstance(x, dict)]
    frame = finalize(pd.DataFrame([map_record(r, f"JSON: {file.name}") for r in records]))
    return frame, f"Detected {len(frame):,} potential prop legs."



# ---------- PrizePicks page/detail importers ----------

def money_to_float(value: Any) -> float:
    if value is None:
        return np.nan
    text = str(value).replace(',', '').replace('$', '').strip()
    return number(text)


def clean_visible_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or '').splitlines() if line.strip()]


def make_entry_id(header: str, placed_at: Any, players: list[str]) -> str:
    base = '|'.join([str(header or ''), str(placed_at or ''), ','.join(players)])
    return 'pp_' + hashlib.sha1(base.encode('utf-8')).hexdigest()[:14]


def parse_pp_datetime(lines: list[str]) -> pd.Timestamp:
    for line in lines:
        if '@' in line and re.search(r'\b\d{4}\b', line):
            parsed = pd.to_datetime(line.replace('@', ''), errors='coerce')
            if pd.notna(parsed):
                return parsed
    return pd.NaT


def parse_detail_meta(detail_text: str) -> dict[str, Any]:
    lines = clean_visible_lines(detail_text)
    meta: dict[str, Any] = {}
    header = next((line for line in lines if re.match(r'^\d+\-Pick\s+(Flex|Power)\s+Play$', line, re.I)), '')
    meta['header'] = header
    if header:
        m = re.match(r'^(\d+)\-Pick\s+(Flex|Power)\s+Play$', header, re.I)
        if m:
            meta['entry_type'] = f"{m.group(2).title()} Play"
            meta['pick_count'] = int(m.group(1))
    result = next((line for line in lines if line.strip().lower() in {'win','won','lost','loss','self refund','partial_win','partial win'}), '')
    meta['entry_result'] = norm_result(result.replace('Self Refund', 'Void')) if result else ''
    pay_line = next((line for line in lines if re.search(r'\$[\d,.]+\s+to\s+win\s+\$[\d,.]+', line, re.I)), '')
    if pay_line:
        m = re.search(r'\$([\d,.]+)\s+to\s+win\s+\$([\d,.]+)', pay_line, re.I)
        if m:
            meta['entry_fee'] = money_to_float(m.group(1))
            meta['entry_return'] = money_to_float(m.group(2))
    meta['placed_at'] = parse_pp_datetime(lines)
    meta['slate_date'] = meta['placed_at'].date() if pd.notna(meta.get('placed_at')) else pd.NaT
    return meta


def detect_direction_from_html(row_html: str) -> str:
    snippet = str(row_html or '')
    if re.search(r'rotate\-180|M8\.707\s+13\.293|M7\.293\s+13\.293|V2\.5', snippet):
        return 'Less'
    if re.search(r'M8\.707\s+2\.707|V13\.5', snippet):
        return 'More'
    return ''


def tier_from_html(row_html: str) -> str:
    snippet = str(row_html or '')
    if re.search(r'alt=["\']Demon["\']|>\s*Demon\s*<', snippet, re.I):
        return 'Demon'
    if re.search(r'alt=["\']Goblin["\']|>\s*Goblin\s*<', snippet, re.I):
        return 'Goblin'
    return 'Standard'


def html_for_projection(detail_html: str, projection_id: str) -> str:
    if not detail_html or not projection_id:
        return ''
    needle = f'data-projectionid="{projection_id}"'
    start = detail_html.find(needle)
    if start < 0:
        return ''
    next_start = detail_html.find('data-projectionid="', start + len(needle))
    if next_start < 0:
        next_start = min(len(detail_html), start + 8000)
    return detail_html[start:next_start]


def parse_detail_leg_lines(lines: list[str], meta: dict[str, Any], direction: str = '', projection_id: str = '', tier: str = 'Standard', source: str = 'PrizePicks detail') -> dict[str, Any] | None:
    if 'Final' not in lines:
        return None
    final_idx = lines.index('Final')
    sport_idx = None
    known_sports = {'MLB','NBA','WNBA','NFL','NHL','NCAAF','NCAAB','PGA','TENNIS','SOCCER','CS2','LOL','MMA'}
    for idx in range(final_idx - 1, -1, -1):
        if lines[idx].upper() in known_sports:
            sport_idx = idx
            break
    if sport_idx is None or sport_idx < 1 or final_idx + 3 >= len(lines):
        return None
    player = lines[sport_idx - 2] if sport_idx >= 2 else lines[0]
    position = lines[sport_idx - 1] if sport_idx >= 1 else ''
    sport = lines[sport_idx].upper()
    team = lines[sport_idx + 1] if sport_idx + 1 < final_idx else ''
    if 'vs' in lines[sport_idx + 1:final_idx]:
        vs_idx = lines.index('vs', sport_idx + 1, final_idx)
        opponent = lines[vs_idx + 1] if vs_idx + 1 < final_idx else ''
    else:
        opponent = lines[sport_idx + 3] if sport_idx + 3 < final_idx else ''
    line_value = number(lines[final_idx + 1])
    actual_value = number(lines[-1])
    prop_type = ' '.join(lines[final_idx + 2:-1])
    row = {
        'leg_id': f"{make_entry_id(meta.get('header',''), meta.get('placed_at'), [player])}_{projection_id}" if projection_id else np.nan,
        'entry_id': '',
        'placed_at': meta.get('placed_at'),
        'slate_date': meta.get('slate_date'),
        'sport': sport,
        'league': sport,
        'player': player,
        'team': team,
        'opponent': opponent,
        'prop_type': prop_type,
        'line': line_value,
        'direction': direction,
        'actual': actual_value,
        'result': '',
        'tier': tier,
        'entry_type': meta.get('entry_type',''),
        'entry_fee': meta.get('entry_fee', np.nan),
        'entry_return': meta.get('entry_return', np.nan),
        'payout_multiplier': np.nan,
        'notes': f"Position: {position}; projection_id: {projection_id}".strip('; '),
        'source': source,
    }
    if not row['direction'] and meta.get('entry_result') == 'Win' and pd.notna(line_value) and pd.notna(actual_value) and actual_value != line_value:
        row['direction'] = 'More' if actual_value > line_value else 'Less'
    row['result'] = grade(row['direction'], row['line'], row['actual'], '')
    return row


def parse_prizepicks_detail_inspector(payload: dict[str, Any], source: str = 'PrizePicks detail inspector') -> pd.DataFrame:
    detail_text = payload.get('detail_text', '')
    detail_html = payload.get('detail_html', '')
    meta = parse_detail_meta(detail_text)
    rows: list[dict[str, Any]] = []
    for element in payload.get('elements', []):
        attrs = element.get('attributes', {}) or {}
        projection_id = str(attrs.get('data-projectionid', '') or '')
        if not projection_id:
            continue
        lines = clean_visible_lines(element.get('visible_text', ''))
        row_html = html_for_projection(detail_html, projection_id)
        direction = detect_direction_from_html(row_html)
        tier = tier_from_html(row_html)
        row = parse_detail_leg_lines(lines, meta, direction, projection_id, tier, source)
        if row:
            rows.append(row)
    if rows:
        players = [r.get('player','') for r in rows]
        entry_id = make_entry_id(meta.get('header',''), meta.get('placed_at'), players)
        for r in rows:
            r['entry_id'] = entry_id
            if str(r.get('leg_id','')).startswith('pp_'):
                r['leg_id'] = f"{entry_id}_{r.get('notes','').split('projection_id: ')[-1]}"
    return finalize(pd.DataFrame(rows))


def split_detail_blocks_from_text(text: str) -> list[str]:
    lines = clean_visible_lines(text)
    starts = [i for i, line in enumerate(lines) if re.match(r'^\d+\-Pick\s+(Flex|Power)\s+Play$', line, re.I)]
    blocks = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        block_lines = lines[start:end]
        if 'Final' in block_lines:
            blocks.append('\n'.join(block_lines))
    return blocks


def parse_prizepicks_page_text(text: str, source: str = 'PrizePicks page text') -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    known_sports = {'MLB','NBA','WNBA','NFL','NHL','NCAAF','NCAAB','PGA','TENNIS','SOCCER','CS2','LOL','MMA'}
    for block in split_detail_blocks_from_text(text):
        meta = parse_detail_meta(block)
        lines = clean_visible_lines(block)
        final_indices = [i for i, line in enumerate(lines) if line == 'Final']
        block_rows: list[dict[str, Any]] = []
        for leg_num, final_idx in enumerate(final_indices, 1):
            sport_idx = None
            for idx in range(final_idx - 1, -1, -1):
                if lines[idx].upper() in known_sports:
                    sport_idx = idx
                    break
            if sport_idx is None or sport_idx < 2:
                continue
            # In the page text export, each leg is: Player, Position, Sport, teams/scores, Final, line, prop, actual.
            start = sport_idx - 2
            end = min(len(lines), final_idx + 4)
            leg_lines = lines[start:end]
            row = parse_detail_leg_lines(leg_lines, meta, '', f'text_{leg_num}', 'Standard', source)
            if row:
                block_rows.append(row)
        if block_rows:
            players = [r.get('player','') for r in block_rows]
            entry_id = make_entry_id(meta.get('header',''), meta.get('placed_at'), players)
            for r in block_rows:
                r['entry_id'] = entry_id
                if str(r.get('leg_id','')).startswith('pp_'):
                    r['leg_id'] = f"{entry_id}_{r.get('notes','').split('projection_id: ')[-1]}"
            rows.extend(block_rows)
    return finalize(pd.DataFrame(rows))

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


def _first_snapshot_column(frame: pd.DataFrame, options: list[str]) -> pd.Series:
    chosen = next((column for column in options if column in frame.columns), None)
    if chosen is None:
        return pd.Series(np.nan, index=frame.index, dtype="object")
    return frame[chosen]


def normalize_snapshot(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize dashboard snapshots into one row per player and prop.

    Hitter snapshots generally contain one modeled target per row. Pitcher
    snapshots contain strikeout, earned-run, and outs probabilities in the
    same pitcher row, so those rows must be expanded before they can match
    individual PrizePicks legs. All imported probabilities are stored as the
    probability of the MORE/OVER side; merge_snapshot flips them for LESS
    selections.
    """
    frame = raw.copy()
    frame.columns = [key(c) for c in frame.columns]

    date_series = _first_snapshot_column(
        frame, ["slate_date", "slatedate", "date", "game_date", "gamedate"]
    )
    player_series = _first_snapshot_column(frame, ["player", "pitcher", "player_name", "pitcher_name"])
    timestamp_series = _first_snapshot_column(
        frame, [
            "snapshot_timestamp", "generated_at", "generatedatutc",
            "generated_at_utc", "prediction_timestamp",
        ]
    )
    confidence_series = _first_snapshot_column(
        frame, ["confidence", "confidence_score", "overall_confidence"]
    )

    # Pitcher snapshots store all three prop targets in a single row.
    pitcher_markers = {
        "proj_k", "k_model_over_prob", "er_model_over_prob",
        "outs_model_over_prob", "k_score", "run_prevention_score", "outs_score",
    }
    if pitcher_markers.intersection(frame.columns):
        configs = [
            {
                "prop_type": "Pitcher Strikeouts",
                "probability": ["k_model_over_prob", "k_over_probability", "strikeouts_over_probability"],
                "edge": ["k_probability_edge", "k_model_edge", "strikeouts_model_edge"],
                "score": ["k_score", "strikeout_score"],
            },
            {
                "prop_type": "Pitcher Earned Runs",
                "probability": ["er_model_over_prob", "er_over_probability", "earned_runs_over_probability"],
                "edge": ["er_probability_edge", "er_model_edge", "earned_runs_model_edge"],
                "score": ["run_prevention_score", "er_score", "earned_runs_score"],
            },
            {
                "prop_type": "Pitcher Outs",
                "probability": ["outs_model_over_prob", "outs_over_probability"],
                "edge": ["outs_probability_edge", "outs_model_edge"],
                "score": ["outs_score"],
            },
        ]
        expanded: list[pd.DataFrame] = []
        for config in configs:
            part = pd.DataFrame(index=frame.index)
            part["slate_date"] = date_series
            part["player"] = player_series
            part["prop_type"] = config["prop_type"]
            part["model_probability"] = _first_snapshot_column(frame, config["probability"])
            part["model_edge"] = _first_snapshot_column(frame, config["edge"])
            part["model_score"] = _first_snapshot_column(frame, config["score"])
            part["confidence"] = confidence_series
            part["snapshot_timestamp"] = timestamp_series
            # Keep rows even if probability is blank so score-only matches can
            # still be diagnosed, but remove rows with no usable player/date.
            expanded.append(part)
        out = pd.concat(expanded, ignore_index=True, sort=False)
    else:
        choices = {
            "slate_date": ["slate_date", "slatedate", "date", "game_date", "gamedate"],
            "player": ["player", "pitcher", "player_name", "pitcher_name"],
            "prop_type": ["prop_type", "target", "market", "stat_type", "category"],
            "model_probability": [
                "model_probability", "modelprobability", "model_1plus_hit",
                "model_1plus_hr", "over_probability", "model_over_probability",
                "pp_over_probability",
            ],
            "model_edge": ["model_edge", "model_market_edge", "probability_edge"],
            "model_score": [
                "model_score", "hit_score", "hr_score", "k_score",
                "overall_score", "fantasy_score_rating", "fs_rating",
            ],
            "confidence": ["confidence", "confidence_score", "overall_confidence"],
            "snapshot_timestamp": [
                "snapshot_timestamp", "generated_at", "generatedatutc",
                "generated_at_utc", "prediction_timestamp",
            ],
        }
        out = pd.DataFrame(index=frame.index)
        for target, options in choices.items():
            out[target] = _first_snapshot_column(frame, options)

        if out.prop_type.isna().all():
            if "model_1plus_hit" in frame.columns:
                out.prop_type = "Hits"
            elif "model_1plus_hr" in frame.columns:
                out.prop_type = "Home Runs"
            elif any(column in frame.columns for column in ["projected_fs", "fs_rating", "fantasy_score_rating"]):
                out.prop_type = "Hitter Fantasy Score"

    out["slate_date"] = pd.to_datetime(out["slate_date"], errors="coerce").dt.date
    out["player"] = out["player"].astype("string").fillna("").str.strip()
    out["prop_type"] = out["prop_type"].astype("string").fillna("").str.strip()
    out["player_key"] = out["player"].map(norm_name)
    out["prop_key"] = out["prop_type"].map(lambda x: key(norm_prop(x)))
    out["model_probability"] = out["model_probability"].map(probability)
    out["model_edge"] = out["model_edge"].map(number)
    # Edges can be stored either as decimal probability points or percentages.
    out.loc[out["model_edge"].abs() > 1, "model_edge"] = out.loc[out["model_edge"].abs() > 1, "model_edge"] / 100.0
    out["model_score"] = pd.to_numeric(out["model_score"], errors="coerce")
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out["snapshot_timestamp"] = out["snapshot_timestamp"].map(dt)
    return out.reset_index(drop=True)


def _snapshot_date_key(series: pd.Series) -> pd.Series:
    """Return a stable YYYY-MM-DD string key for snapshot joins.

    PrizePicks imports and dashboard snapshots can represent dates as Python
    ``date`` objects, pandas timestamps, or strings.  Pandas refuses to merge
    object and datetime64 columns, so both sides are normalized to the same
    nullable string dtype before joining.
    """
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.strftime("%Y-%m-%d").astype("string").fillna("")


def _snapshot_text_key(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def merge_snapshot(ledger: pd.DataFrame, snapshot: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if ledger.empty or snapshot.empty:
        return ledger, 0

    left = ledger.copy()
    snap = snapshot.copy()

    # Build dedicated join keys instead of merging on the display columns.
    # This prevents object/datetime64 and object/float dtype mismatches when a
    # source file contains blank dates, players, or prop names.
    left["slate_date_key"] = _snapshot_date_key(left.get("slate_date", pd.Series(index=left.index, dtype="object")))
    snap["slate_date_key"] = _snapshot_date_key(snap.get("slate_date", pd.Series(index=snap.index, dtype="object")))

    left["player_key"] = _snapshot_text_key(left.get("player", pd.Series(index=left.index, dtype="object")).map(norm_name))
    left["prop_key"] = _snapshot_text_key(left.get("prop_type", pd.Series(index=left.index, dtype="object")).map(lambda x: key(norm_prop(x))))
    snap["player_key"] = _snapshot_text_key(snap.get("player_key", snap.get("player", pd.Series(index=snap.index, dtype="object")).map(norm_name)))
    snap["prop_key"] = _snapshot_text_key(snap.get("prop_key", snap.get("prop_type", pd.Series(index=snap.index, dtype="object")).map(lambda x: key(norm_prop(x)))))

    join_cols = ["slate_date_key", "player_key", "prop_key"]
    snap = snap[(snap[join_cols] != "").all(axis=1)].copy()
    if snap.empty:
        return finalize(left[[c for c in COLUMNS if c in left]]), 0

    if "snapshot_timestamp" in snap.columns:
        snap["snapshot_timestamp"] = pd.to_datetime(snap["snapshot_timestamp"], errors="coerce", utc=True)
        snap = snap.sort_values("snapshot_timestamp", na_position="first")

    snap = snap.drop_duplicates(join_cols, keep="last")
    merged = left.merge(snap, on=join_cols, how="left", suffixes=("", "_snap"), validate="many_to_one")

    # Dashboard snapshots store the probability/edge for the MORE side. For a
    # PrizePicks LESS selection, convert it to the selected-side probability
    # and selected-side edge before saving it to the ledger.
    less_mask = merged.get("direction", pd.Series("", index=merged.index)).map(norm_direction).eq("Less")
    if "model_probability_snap" in merged.columns:
        snap_probability = pd.to_numeric(merged["model_probability_snap"], errors="coerce")
        merged.loc[less_mask & snap_probability.notna(), "model_probability_snap"] = 1.0 - snap_probability[less_mask & snap_probability.notna()]
    if "model_edge_snap" in merged.columns:
        snap_edge = pd.to_numeric(merged["model_edge_snap"], errors="coerce")
        merged.loc[less_mask & snap_edge.notna(), "model_edge_snap"] = -snap_edge[less_mask & snap_edge.notna()]

    matched_mask = pd.Series(False, index=merged.index)
    for candidate in ["model_probability_snap", "model_edge_snap", "model_score_snap", "snapshot_timestamp_snap"]:
        if candidate in merged.columns:
            matched_mask = matched_mask | merged[candidate].notna()
    matched = int(matched_mask.sum())

    for col in ["model_probability", "model_edge", "model_score", "confidence", "snapshot_timestamp"]:
        other = f"{col}_snap"
        if other in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[other]
            else:
                merged[col] = merged[other].combine_first(merged[col])

    return finalize(merged[[c for c in COLUMNS if c in merged.columns]]), matched


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
        snap = normalize_snapshot(pd.read_csv(upload))
        probability_rows = int(snap.get("model_probability", pd.Series(dtype=float)).notna().sum())
        st.caption(f"Snapshot rows detected: {len(snap):,} · rows with model probability: {probability_rows:,}")
        if probability_rows == 0:
            st.warning("No recognized model-probability columns were found in this snapshot. Use a pregame snapshot downloaded from the dashboard, not only a results/master-history export.")
        if st.button("Merge snapshot into tracked legs", width="stretch"):
            try:
                st.session_state.ledger, matched = merge_snapshot(st.session_state.ledger, snap)
            except Exception as exc:
                st.error(f"Snapshot merge failed: {exc}")
            else:
                st.success(f"Matched model data to {matched:,} legs.")
                st.rerun()

with import_tab:
    st.markdown("### Upload CSV, JSON, or PrizePicks page text")
    uploads = st.file_uploader("Upload one or more files", type=["csv", "json", "txt"], accept_multiple_files=True, key="uploads")
    st.caption("Supports tracker CSVs, copied JSON responses, prizepicks_completed_page.txt, and prizepicks_detail_inspector.json files.")
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
