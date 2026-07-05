from __future__ import annotations

import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from datetime import date, timedelta
from html import escape
from io import StringIO
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st
from pybaseball import cache, playerid_reverse_lookup, statcast


st.set_page_config(page_title="MLB Pitcher Lab", page_icon="⚾", layout="wide")
cache.enable()

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

MODEL_VERSION = "pitcher-lab-backtest-odds-v2"
ODDS_API_SPORT = "baseball_mlb"
ODDS_API_MARKETS = {
    "pitcher_strikeouts": {"prefix": "K", "label": "Strikeouts", "projection": "Proj_K"},
    "pitcher_earned_runs": {"prefix": "ER", "label": "Earned runs", "projection": "Proj_ER"},
    "pitcher_outs": {"prefix": "Outs", "label": "Outs", "projection": "Proj_Outs"},
}
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

TEAM_NAME_TO_ABBR = {
    "losangelesangels": "LAA", "arizonadiamondbacks": "ARI", "athletics": "ATH",
    "oaklandathletics": "ATH", "sacramentoathletics": "ATH", "atlantabraves": "ATL",
    "baltimoreorioles": "BAL", "bostonredsox": "BOS", "chicagocubs": "CHC",
    "chicagowhitesox": "CWS", "cincinnatireds": "CIN", "clevelandguardians": "CLE",
    "coloradorockies": "COL", "detroittigers": "DET", "houstonastros": "HOU",
    "kansascityroyals": "KC", "losangelesdodgers": "LAD", "miamimarlins": "MIA",
    "milwaukeebrewers": "MIL", "minnesotatwins": "MIN", "newyorkmets": "NYM",
    "newyorkyankees": "NYY", "philadelphiaphillies": "PHI", "pittsburghpirates": "PIT",
    "sandiegopadres": "SD", "seattlemariners": "SEA", "sanfranciscogiants": "SF",
    "stlouiscardinals": "STL", "tampabayrays": "TB", "texasrangers": "TEX",
    "torontobluejays": "TOR", "washingtonnationals": "WSH",
}

BACKTEST_TARGETS = {
    "Strikeouts": {
        "projection": "Proj_K", "actual": "Actual_K", "baseline": "Baseline_K",
        "score": "K_Score", "line": "K_Line", "lower_is_better": False,
    },
    "Earned runs": {
        "projection": "Proj_ER", "actual": "Actual_ER", "baseline": "Baseline_ER",
        "score": "Run_Prevention_Score", "line": "ER_Line", "lower_is_better": True,
    },
    "Outs": {
        "projection": "Proj_Outs", "actual": "Actual_Outs", "baseline": "Baseline_Outs",
        "score": "Outs_Score", "line": "Outs_Line", "lower_is_better": False,
    },
}

MLB_TEAM_ABBR = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH",
    134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

TEAM_ALIASES = {"OAK": "ATH", "ATH": "OAK"}

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
    {"uniqlofieldatdodgerstadium", "dodgerstadium"},
]


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def safe_divide(numerator, denominator, default=np.nan):
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    result = np.divide(num, den)
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan).fillna(default)
    return default if not np.isfinite(result) else result


def percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1 or numeric.nunique(dropna=True) <= 1:
        return pd.Series(50.0, index=series.index)
    ranked = numeric.rank(pct=True, method="average") * 100.0
    if not higher_is_better:
        ranked = 100.0 - ranked
    return ranked.fillna(50.0)


def shrink_rate(numerator, denominator, league_rate: float, prior_sample: float):
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    den = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    return (num + float(league_rate) * float(prior_sample)) / (den + float(prior_sample))


def normalize_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return "".join(character.lower() for character in str(value) if character.isalnum())


def weighted_recent_mean(series: pd.Series, count: int = 5, default: float = np.nan) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().tail(count)
    if values.empty:
        return float(default)
    weights = np.arange(1, len(values) + 1, dtype=float)
    return float(np.average(values.to_numpy(dtype=float), weights=weights))


def innings_to_outs(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    try:
        if "." in text:
            whole_text, partial_text = text.split(".", 1)
            whole = int(whole_text or 0)
            partial = int((partial_text or "0")[0])
            partial = partial if partial in {0, 1, 2} else 0
            return float(whole * 3 + partial)
        return float(int(float(text)) * 3)
    except (TypeError, ValueError):
        return np.nan


def poisson_cdf(k: int, mean: float) -> float:
    mean = max(float(mean), 0.0001)
    k = max(int(k), 0)
    return float(sum(math.exp(-mean) * mean ** i / math.factorial(i) for i in range(k + 1)))


def probability_over_line(mean: float, line: float) -> float:
    threshold = math.floor(float(line))
    return float(np.clip(1.0 - poisson_cdf(threshold, mean), 0.0, 1.0))


def normal_cdf(value: float, mean: float, sd: float) -> float:
    sd = max(float(sd), 0.35)
    z = (float(value) - float(mean)) / (sd * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def team_logo_url(team_id: object) -> str:
    numeric = pd.to_numeric(pd.Series([team_id]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return f"https://www.mlbstatic.com/team-logos/{int(numeric)}.svg"


def team_palette(team_code: object) -> tuple[str, str]:
    return TEAM_COLORS.get(str(team_code or "").upper().strip(), ("#2563EB", "#7C3AED"))


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



# -----------------------------------------------------------------------------
# The Odds API: automatic pitcher prop lines
# -----------------------------------------------------------------------------

def read_streamlit_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or "").strip()


def normalize_lookup_text(value: object) -> str:
    raw = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = raw.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "", ascii_text)


def normalize_person_name(value: object) -> str:
    raw = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = raw.encode("ascii", "ignore").decode("ascii").lower()
    ascii_text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", ascii_text)
    return re.sub(r"[^a-z0-9]+", "", ascii_text)


def canonical_team_abbr(value: object) -> str:
    normalized = normalize_lookup_text(value)
    if normalized in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[normalized]
    upper = str(value or "").strip().upper()
    if upper in set(MLB_TEAM_ABBR.values()) | {"OAK"}:
        return "ATH" if upper == "OAK" else upper
    return upper


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
    if target_tokens and candidate_tokens and target_tokens[-1] == candidate_tokens[-1]:
        if target_tokens[0][:1] == candidate_tokens[0][:1]:
            return max(0.94, SequenceMatcher(None, target_text, candidate_text).ratio())
        return max(0.78, SequenceMatcher(None, target_text, candidate_text).ratio())
    return SequenceMatcher(None, target_text, candidate_text).ratio()


def _api_error_message(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        detail = body[:500].strip()
        return f"HTTP {exc.code}: {detail or exc.reason}"
    return f"{type(exc).__name__}: {exc}"


def _response_quota(response) -> dict:
    headers = getattr(response, "headers", {})
    return {
        "requests_remaining": headers.get("x-requests-remaining"),
        "requests_used": headers.get("x-requests-used"),
        "requests_last": headers.get("x-requests-last"),
    }


@st.cache_data(ttl=600, show_spinner=False, max_entries=8)
def fetch_odds_api_events(
    api_key: str,
    commence_time_from: str,
    commence_time_to: str,
) -> tuple[list[dict], dict, str | None]:
    params = urlencode(
        {
            "apiKey": api_key,
            "dateFormat": "iso",
            "commenceTimeFrom": commence_time_from,
            "commenceTimeTo": commence_time_to,
        }
    )
    url = f"https://api.the-odds-api.com/v4/sports/{ODDS_API_SPORT}/events?{params}"
    request = Request(url, headers={"User-Agent": "MLB-Pitcher-Lab/2.0", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
            quota = _response_quota(response)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return [], {}, _api_error_message(exc)
    if not isinstance(payload, list):
        return [], quota, "The events endpoint returned an unexpected response."
    return payload, quota, None


@st.cache_data(ttl=300, show_spinner=False, max_entries=64)
def fetch_odds_api_event_props(
    api_key: str,
    event_id: str,
    bookmaker_keys: str,
    market_keys: str,
) -> tuple[dict, dict, str | None]:
    params = urlencode(
        {
            "apiKey": api_key,
            "bookmakers": bookmaker_keys,
            "markets": market_keys,
            "oddsFormat": "american",
            "dateFormat": "iso",
            "includeMultipliers": "true",
        }
    )
    url = f"https://api.the-odds-api.com/v4/sports/{ODDS_API_SPORT}/events/{event_id}/odds?{params}"
    request = Request(url, headers={"User-Agent": "MLB-Pitcher-Lab/2.0", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=25) as response:
            payload = json.load(response)
            quota = _response_quota(response)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return {}, {}, _api_error_message(exc)
    if not isinstance(payload, dict):
        return {}, quota, "The event odds endpoint returned an unexpected response."
    return payload, quota, None


def slate_utc_bounds(slate_date: object) -> tuple[str, str]:
    day = pd.Timestamp(slate_date)
    try:
        start = day.tz_localize("America/New_York").tz_convert("UTC")
    except Exception:
        start = day.tz_localize("UTC")
    end = start + pd.Timedelta(days=1, hours=8)
    start = start - pd.Timedelta(hours=4)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def match_odds_event(game: dict, events: list[dict]) -> dict | None:
    away = canonical_team_abbr(game.get("away_abbr") or game.get("away_name"))
    home = canonical_team_abbr(game.get("home_abbr") or game.get("home_name"))
    candidates = [
        event for event in events
        if canonical_team_abbr(event.get("away_team")) == away
        and canonical_team_abbr(event.get("home_team")) == home
    ]
    if not candidates:
        return None
    game_time = pd.to_datetime(game.get("game_datetime_utc"), utc=True, errors="coerce")
    if pd.isna(game_time) or len(candidates) == 1:
        return candidates[0]
    return min(
        candidates,
        key=lambda event: abs(
            (pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce") - game_time).total_seconds()
        ) if not pd.isna(pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")) else float("inf"),
    )


def parse_pitcher_prop_payload(payload: dict, game: dict) -> pd.DataFrame:
    rows: list[dict] = []
    event_id = payload.get("id")
    for bookmaker in payload.get("bookmakers", []) or []:
        bookmaker_key = str(bookmaker.get("key", ""))
        bookmaker_title = str(bookmaker.get("title") or ODDS_BOOKMAKERS.get(bookmaker_key, bookmaker_key))
        for market in bookmaker.get("markets", []) or []:
            market_key = str(market.get("key", ""))
            if market_key not in ODDS_API_MARKETS:
                continue
            last_update = market.get("last_update")
            grouped: dict[tuple[str, float], dict] = {}
            for outcome in market.get("outcomes", []) or []:
                side = str(outcome.get("name", "")).strip().lower()
                if side not in {"over", "under"}:
                    continue
                player = str(outcome.get("description") or outcome.get("participant") or "").strip()
                if not player:
                    continue
                point = pd.to_numeric(pd.Series([outcome.get("point")]), errors="coerce").iloc[0]
                if pd.isna(point):
                    continue
                key = (normalize_person_name(player), float(point))
                record = grouped.setdefault(
                    key,
                    {
                        "SlateDate": str(game.get("slate_date", "")),
                        "GamePK": game.get("game_pk"),
                        "OddsEventID": event_id,
                        "AwayTeam": payload.get("away_team"),
                        "HomeTeam": payload.get("home_team"),
                        "CommenceTimeUTC": payload.get("commence_time"),
                        "BookmakerKey": bookmaker_key,
                        "Bookmaker": bookmaker_title,
                        "MarketKey": market_key,
                        "Market": ODDS_API_MARKETS[market_key]["label"],
                        "Player": player,
                        "PlayerNorm": normalize_person_name(player),
                        "Line": float(point),
                        "OverOdds": np.nan,
                        "UnderOdds": np.nan,
                        "OverMultiplier": np.nan,
                        "UnderMultiplier": np.nan,
                        "LastUpdate": last_update,
                    },
                )
                price = pd.to_numeric(pd.Series([outcome.get("price")]), errors="coerce").iloc[0]
                multiplier = outcome.get("multiplier")
                if multiplier is None:
                    multiplier = outcome.get("multipliers")
                multiplier_numeric = pd.to_numeric(pd.Series([multiplier]), errors="coerce").iloc[0]
                if side == "over":
                    record["OverOdds"] = float(price) if not pd.isna(price) else np.nan
                    record["OverMultiplier"] = float(multiplier_numeric) if not pd.isna(multiplier_numeric) else np.nan
                else:
                    record["UnderOdds"] = float(price) if not pd.isna(price) else np.nan
                    record["UnderMultiplier"] = float(multiplier_numeric) if not pd.isna(multiplier_numeric) else np.nan
            rows.extend(grouped.values())
    return pd.DataFrame(rows)


def fetch_pitcher_prop_quotes(
    api_key: str,
    schedule: list[dict],
    game_pks: list[int],
) -> tuple[pd.DataFrame, dict, list[str]]:
    selected = {int(value) for value in game_pks if pd.notna(value)}
    if not api_key:
        return pd.DataFrame(), {}, ["The Odds API key is missing."]
    if not schedule or not selected:
        return pd.DataFrame(), {}, ["No games were selected."]
    slate_date_value = next((game.get("slate_date") for game in schedule if game.get("game_pk") in selected), date.today())
    commence_from, commence_to = slate_utc_bounds(slate_date_value)
    events, events_quota, events_error = fetch_odds_api_events(api_key, commence_from, commence_to)
    if events_error:
        return pd.DataFrame(), events_quota, [f"Events: {events_error}"]

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    quota_meta = dict(events_quota)
    quota_cost_total = 0.0
    matched_events = 0
    queried_events = 0
    bookmaker_keys = ",".join(ODDS_BOOKMAKERS.keys())
    market_keys = ",".join(ODDS_API_MARKETS.keys())

    for game in schedule:
        numeric_game_pk = pd.to_numeric(pd.Series([game.get("game_pk")]), errors="coerce").iloc[0]
        if pd.isna(numeric_game_pk) or int(numeric_game_pk) not in selected:
            continue
        event = match_odds_event(game, events)
        if not event:
            errors.append(f"{game.get('away_abbr')} @ {game.get('home_abbr')}: no matching event was found.")
            continue
        matched_events += 1
        payload, call_quota, call_error = fetch_odds_api_event_props(
            api_key,
            str(event.get("id")),
            bookmaker_keys,
            market_keys,
        )
        queried_events += 1
        if call_quota:
            quota_meta.update({key: value for key, value in call_quota.items() if value is not None})
            try:
                quota_cost_total += float(call_quota.get("requests_last") or 0)
            except (TypeError, ValueError):
                pass
        if call_error:
            errors.append(f"{game.get('away_abbr')} @ {game.get('home_abbr')}: {call_error}")
            continue
        frame = parse_pitcher_prop_payload(payload, game)
        if frame.empty:
            errors.append(f"{game.get('away_abbr')} @ {game.get('home_abbr')}: no requested pitcher props were returned.")
        else:
            frames.append(frame)

    quota_meta["estimated_credits_used_this_fetch"] = quota_cost_total
    quota_meta["events_matched"] = matched_events
    quota_meta["events_queried"] = queried_events
    if not frames:
        return pd.DataFrame(), quota_meta, errors
    quotes = pd.concat(frames, ignore_index=True, sort=False)
    quotes["LastUpdateParsed"] = pd.to_datetime(quotes.get("LastUpdate"), utc=True, errors="coerce")
    quotes = quotes.sort_values("LastUpdateParsed", na_position="first")
    quotes = quotes.drop_duplicates(
        ["GamePK", "BookmakerKey", "MarketKey", "PlayerNorm", "Line"], keep="last"
    ).drop(columns="LastUpdateParsed")
    return quotes.reset_index(drop=True), quota_meta, errors


def _matched_pitcher_quote_rows(quotes: pd.DataFrame, game_pk: object, pitcher_name: str, market_key: str) -> pd.DataFrame:
    if quotes is None or quotes.empty:
        return pd.DataFrame()
    game_numeric = pd.to_numeric(pd.Series([game_pk]), errors="coerce").iloc[0]
    quote_game = pd.to_numeric(quotes.get("GamePK"), errors="coerce")
    candidates = quotes[(quote_game.eq(game_numeric)) & quotes.get("MarketKey", pd.Series(index=quotes.index, dtype=object)).eq(market_key)].copy()
    if candidates.empty:
        return candidates
    target_norm = normalize_person_name(pitcher_name)
    exact = candidates[candidates.get("PlayerNorm", pd.Series(index=candidates.index, dtype=object)).eq(target_norm)]
    if not exact.empty:
        return exact
    candidates["_NameScore"] = candidates.get("Player", "").map(lambda value: player_name_match_score(pitcher_name, value))
    best = pd.to_numeric(candidates["_NameScore"], errors="coerce").max()
    if pd.isna(best) or best < 0.84:
        return pd.DataFrame(columns=candidates.columns)
    return candidates[candidates["_NameScore"].ge(best - 0.015)].drop(columns="_NameScore", errors="ignore")


def _latest_per_book(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    work = candidates.copy()
    work["_Updated"] = pd.to_datetime(work.get("LastUpdate"), utc=True, errors="coerce")
    work["_Paired"] = work.get("OverOdds").notna().astype(int) + work.get("UnderOdds").notna().astype(int)
    work = work.sort_values(["BookmakerKey", "_Paired", "_Updated"], na_position="first")
    return work.drop_duplicates("BookmakerKey", keep="last").drop(columns=["_Updated", "_Paired"])


def select_prop_quote(candidates: pd.DataFrame, source_mode: str) -> dict | None:
    if candidates is None or candidates.empty:
        return None
    candidates = _latest_per_book(candidates)
    direct_key = ODDS_SOURCE_TO_KEY.get(source_mode)
    if direct_key:
        direct = candidates[candidates["BookmakerKey"].eq(direct_key)].copy()
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
        line = float(pd.to_numeric(books["Line"], errors="coerce").median())
        over_odds = pd.to_numeric(books["OverOdds"], errors="coerce").median()
        under_odds = pd.to_numeric(books["UnderOdds"], errors="coerce").median()
        return {
            "Line": line,
            "OverOdds": float(over_odds) if not pd.isna(over_odds) else np.nan,
            "UnderOdds": float(under_odds) if not pd.isna(under_odds) else np.nan,
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
        row["BookCount"] = int(books["BookmakerKey"].nunique())
        return row
    if source_mode == "Best sportsbook under line":
        best_line = pd.to_numeric(books["Line"], errors="coerce").max()
        subset = books[pd.to_numeric(books["Line"], errors="coerce").eq(best_line)].copy()
        subset["_Price"] = pd.to_numeric(subset["UnderOdds"], errors="coerce").fillna(-100000)
        row = subset.sort_values("_Price").iloc[-1].drop(labels="_Price").to_dict()
        row["SelectedSource"] = f"Best under · {row.get('Bookmaker')}"
        row["BookCount"] = int(books["BookmakerKey"].nunique())
        return row
    return None


def american_implied_probability(odds: object) -> float:
    numeric = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(numeric) or numeric == 0:
        return np.nan
    numeric = float(numeric)
    if numeric < 0:
        return -numeric / (-numeric + 100.0)
    return 100.0 / (numeric + 100.0)


def no_vig_over_probability(over_odds: object, under_odds: object) -> float:
    over = american_implied_probability(over_odds)
    under = american_implied_probability(under_odds)
    if not np.isfinite(over) or not np.isfinite(under) or over + under <= 0:
        return np.nan
    return float(over / (over + under))


def format_american_odds(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "—"
    rounded = int(round(float(numeric)))
    return f"+{rounded}" if rounded > 0 else str(rounded)


def apply_odds_quotes_to_board(board: pd.DataFrame, quotes: pd.DataFrame, source_mode: str) -> pd.DataFrame:
    result = board.copy()
    for market_key, config in ODDS_API_MARKETS.items():
        prefix = config["prefix"]
        for column, default in {
            f"{prefix}_Line": np.nan,
            f"{prefix}_Over_Odds": np.nan,
            f"{prefix}_Under_Odds": np.nan,
            f"{prefix}_Over_Multiplier": np.nan,
            f"{prefix}_Under_Multiplier": np.nan,
            f"{prefix}_Line_Source": "",
            f"{prefix}_Line_Updated": "",
            f"{prefix}_Market_Over_Prob": np.nan,
            f"{prefix}_Model_Over_Prob": np.nan,
            f"{prefix}_Probability_Edge": np.nan,
            f"{prefix}_Projection_Edge": np.nan,
        }.items():
            result[column] = default

        for index, row in result.iterrows():
            candidates = _matched_pitcher_quote_rows(quotes, row.get("GamePK"), str(row.get("Pitcher", "")), market_key)
            selected = select_prop_quote(candidates, source_mode)
            if not selected:
                continue
            line = pd.to_numeric(pd.Series([selected.get("Line")]), errors="coerce").iloc[0]
            if pd.isna(line):
                continue
            over_odds = pd.to_numeric(pd.Series([selected.get("OverOdds")]), errors="coerce").iloc[0]
            under_odds = pd.to_numeric(pd.Series([selected.get("UnderOdds")]), errors="coerce").iloc[0]
            projection = float(pd.to_numeric(pd.Series([row.get(config["projection"])]), errors="coerce").iloc[0])
            if prefix == "Outs":
                sd = float(pd.to_numeric(pd.Series([row.get("Outs_SD")]), errors="coerce").fillna(2.5).iloc[0])
                model_over = 1.0 - normal_cdf(float(line), projection, max(sd, 0.75))
            else:
                model_over = probability_over_line(projection, float(line))
            market_over = no_vig_over_probability(over_odds, under_odds)
            result.at[index, f"{prefix}_Line"] = float(line)
            result.at[index, f"{prefix}_Over_Odds"] = float(over_odds) if not pd.isna(over_odds) else np.nan
            result.at[index, f"{prefix}_Under_Odds"] = float(under_odds) if not pd.isna(under_odds) else np.nan
            result.at[index, f"{prefix}_Over_Multiplier"] = selected.get("OverMultiplier", np.nan)
            result.at[index, f"{prefix}_Under_Multiplier"] = selected.get("UnderMultiplier", np.nan)
            result.at[index, f"{prefix}_Line_Source"] = str(selected.get("SelectedSource") or selected.get("Bookmaker") or "")
            updated = selected.get("LastUpdate")
            result.at[index, f"{prefix}_Line_Updated"] = str(updated) if updated is not None and not pd.isna(updated) else ""
            result.at[index, f"{prefix}_Market_Over_Prob"] = market_over
            result.at[index, f"{prefix}_Model_Over_Prob"] = float(model_over)
            result.at[index, f"{prefix}_Probability_Edge"] = float(model_over - market_over) if np.isfinite(market_over) else np.nan
            result.at[index, f"{prefix}_Projection_Edge"] = float(projection - float(line))
    result["OddsSourceMode"] = source_mode
    return result


def combine_odds_quote_frames(existing: pd.DataFrame, new_quotes: pd.DataFrame, slate_date: object) -> pd.DataFrame:
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
    keys = ["GamePK", "BookmakerKey", "MarketKey", "PlayerNorm", "Line"]
    combined = combined.drop_duplicates([key for key in keys if key in combined.columns], keep="last")
    return combined.drop(columns="_Updated", errors="ignore").reset_index(drop=True)

# -----------------------------------------------------------------------------
# Data loading and preparation
# -----------------------------------------------------------------------------

@st.cache_data(ttl=21600, show_spinner=False, max_entries=2)
def load_statcast(start_date: str, end_date: str) -> pd.DataFrame:
    return statcast(start_dt=start_date, end_dt=end_date, verbose=False, parallel=False)


@st.cache_data(ttl=86400, show_spinner=False, max_entries=8)
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


def prepare_statcast(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    expected = [
        "batter", "pitcher", "game_pk", "at_bat_number", "game_date", "player_name",
        "home_team", "away_team", "inning_topbot", "description", "events", "bb_type",
        "launch_speed", "launch_angle", "launch_speed_angle", "estimated_woba_using_speedangle",
        "woba_value", "estimated_ba_using_speedangle", "estimated_slg_using_speedangle", "zone",
        "pitch_name", "release_speed", "pfx_x", "pfx_z", "release_extension", "stand", "p_throws",
        "outs_on_play", "type",
    ]
    for column in expected:
        if column not in df.columns:
            df[column] = np.nan

    numeric_columns = [
        "batter", "pitcher", "game_pk", "at_bat_number", "launch_speed", "launch_angle",
        "launch_speed_angle", "estimated_woba_using_speedangle", "woba_value",
        "estimated_ba_using_speedangle", "estimated_slg_using_speedangle", "zone",
        "release_speed", "pfx_x", "pfx_z", "release_extension", "outs_on_play",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    top = df["inning_topbot"].eq("Top").fillna(False)
    df["batter_team"] = np.where(top, df["away_team"], df["home_team"])
    df["pitcher_team"] = np.where(top, df["home_team"], df["away_team"])

    description = df["description"].fillna("").astype(str)
    events = df["events"].fillna("").astype(str)
    df["is_pa_end"] = events.ne("")
    df["is_k"] = events.isin(STRIKEOUT_EVENTS)
    df["is_bb"] = events.isin(WALK_EVENTS)
    df["is_hbp"] = events.eq("hit_by_pitch")
    df["is_hit"] = events.isin(HIT_EVENTS)
    df["is_hr"] = events.eq("home_run")
    df["is_swing"] = description.isin(SWING_DESCRIPTIONS)
    df["is_contact"] = description.isin(CONTACT_DESCRIPTIONS)
    df["is_whiff"] = description.isin(WHIFF_DESCRIPTIONS)
    df["is_called_strike"] = description.isin(CALLED_STRIKE_DESCRIPTIONS)
    df["is_bbe"] = df["launch_speed"].notna() & df["launch_angle"].notna()
    df["is_barrel"] = df["launch_speed_angle"].eq(6).fillna(False)
    df["is_hard_hit"] = df["launch_speed"].ge(95).fillna(False)
    df["is_ground_ball"] = df["bb_type"].eq("ground_ball").fillna(False)
    df["is_fly_ball"] = df["bb_type"].eq("fly_ball").fillna(False)

    df["pa_key"] = (
        df["game_pk"].astype("Int64").astype(str)
        + "-"
        + df["at_bat_number"].astype("Int64").astype(str)
    )

    event_outs = events.map(OUTS_BY_EVENT).fillna(0.0)
    df["outs_recorded"] = df["outs_on_play"].where(df["outs_on_play"].notna(), event_outs)
    df["outs_recorded"] = pd.to_numeric(df["outs_recorded"], errors="coerce").fillna(0.0).clip(0, 3)

    actual_woba = df["woba_value"].copy()
    fallback_woba = events.map(
        {
            "walk": 0.69, "intent_walk": 0.69, "hit_by_pitch": 0.72,
            "single": 0.88, "double": 1.25, "triple": 1.58, "home_run": 2.03,
        }
    ).fillna(0.0)
    df["xwoba_value"] = df["estimated_woba_using_speedangle"].where(df["is_bbe"])
    df["xwoba_value"] = df["xwoba_value"].where(df["xwoba_value"].notna(), actual_woba)
    df["xwoba_value"] = df["xwoba_value"].where(df["xwoba_value"].notna(), fallback_woba)
    df["xwoba_value"] = pd.to_numeric(df["xwoba_value"], errors="coerce").fillna(0.0)
    return df


def league_context(df: pd.DataFrame) -> dict:
    pa = df[df["is_pa_end"]].copy()
    pitches = max(len(df), 1)
    swings = max(int(df["is_swing"].sum()), 1)
    bbe = max(int(df["is_bbe"].sum()), 1)
    plate_appearances = max(int(pa["pa_key"].nunique()), 1)
    return {
        "K_PA": float(pa["is_k"].sum() / plate_appearances),
        "BB_PA": float(pa["is_bb"].sum() / plate_appearances),
        "HR_PA": float(pa["is_hr"].sum() / plate_appearances),
        "xwOBA": float(pa["xwoba_value"].sum() / plate_appearances),
        "Brl_BBE": float(df["is_barrel"].sum() / bbe),
        "HH_BBE": float(df["is_hard_hit"].sum() / bbe),
        "Whiff": float(df["is_whiff"].sum() / swings),
        "CSW": float((df["is_whiff"].sum() + df["is_called_strike"].sum()) / pitches),
        "ER9": 4.30,
    }


# -----------------------------------------------------------------------------
# MLB schedule, game feeds and pitcher game logs
# -----------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False, max_entries=32)
def fetch_mlb_schedule(slate_date: str) -> tuple[list[dict], str | None]:
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={slate_date}&hydrate=probablePitcher,venue"
    )
    request = Request(url, headers={"User-Agent": "MLB-Pitcher-Lab/1.0", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return [], f"{type(exc).__name__}: {exc}"

    games: list[dict] = []
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            teams = game.get("teams", {})
            away_info = teams.get("away", {})
            home_info = teams.get("home", {})
            away_team = away_info.get("team", {}) or {}
            home_team = home_info.get("team", {}) or {}
            away_probable = away_info.get("probablePitcher", {}) or {}
            home_probable = home_info.get("probablePitcher", {}) or {}
            raw_time = game.get("gameDate")
            game_time = "TBD"
            if raw_time:
                parsed = pd.to_datetime(raw_time, utc=True, errors="coerce")
                if not pd.isna(parsed):
                    try:
                        game_time = parsed.tz_convert("America/New_York").strftime("%I:%M %p ET").lstrip("0")
                    except Exception:
                        game_time = parsed.strftime("%I:%M %p UTC").lstrip("0")
            away_id = away_team.get("id")
            home_id = home_team.get("id")
            games.append(
                {
                    "game_pk": game.get("gamePk"),
                    "slate_date": slate_date,
                    "game_datetime_utc": raw_time,
                    "time_et": game_time,
                    "status": game.get("status", {}).get("detailedState", "Scheduled"),
                    "venue": game.get("venue", {}).get("name", "Venue TBD"),
                    "venue_id": game.get("venue", {}).get("id"),
                    "away_id": away_id,
                    "home_id": home_id,
                    "away_abbr": MLB_TEAM_ABBR.get(away_id, away_team.get("name", "AWAY")[:3].upper()),
                    "home_abbr": MLB_TEAM_ABBR.get(home_id, home_team.get("name", "HOME")[:3].upper()),
                    "away_name": away_team.get("name", "Away"),
                    "home_name": home_team.get("name", "Home"),
                    "away_pitcher_id": away_probable.get("id"),
                    "away_pitcher_name": away_probable.get("fullName", "TBD"),
                    "home_pitcher_id": home_probable.get("id"),
                    "home_pitcher_name": home_probable.get("fullName", "TBD"),
                }
            )
    return games, None


def _parse_team_lineup(team_box: dict) -> pd.DataFrame:
    columns = ["player_id", "Player", "Position", "LineupSpot", "RawBattingOrder"]
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
        raw_order = pd.to_numeric(pd.Series([record.get("battingOrder")]), errors="coerce").iloc[0]
        numeric_id = pd.to_numeric(pd.Series([player_id]), errors="coerce").iloc[0]
        if pd.isna(raw_order) or pd.isna(numeric_id):
            continue
        raw_order = int(raw_order)
        lineup_spot = raw_order // 100 if raw_order >= 100 else raw_order
        if not 1 <= lineup_spot <= 9:
            continue
        game_status = record.get("gameStatus", {}) or {}
        if bool(game_status.get("isSubstitute", False)):
            continue
        position = record.get("position", {}) or {}
        candidates.append(
            {
                "player_id": int(numeric_id),
                "Player": person.get("fullName") or f"MLB ID {int(numeric_id)}",
                "Position": position.get("abbreviation") or position.get("name") or "",
                "LineupSpot": int(lineup_spot),
                "RawBattingOrder": raw_order,
            }
        )

    ordered_ids = team_box.get("battingOrder") or team_box.get("batters") or []
    existing_spots = {row["LineupSpot"] for row in candidates}
    lookup: dict[int, dict] = {}
    for player_key, record in players.items():
        person = record.get("person", {}) if isinstance(record, dict) else {}
        player_id = person.get("id")
        if player_id is None:
            digits = "".join(character for character in str(player_key) if character.isdigit())
            player_id = int(digits) if digits else None
        if player_id is not None:
            lookup[int(player_id)] = record
    if isinstance(ordered_ids, list) and len(ordered_ids) >= 9:
        for spot, player_id in enumerate(ordered_ids[:9], start=1):
            if spot in existing_spots:
                continue
            numeric_id = pd.to_numeric(pd.Series([player_id]), errors="coerce").iloc[0]
            if pd.isna(numeric_id):
                continue
            record = lookup.get(int(numeric_id), {})
            person = record.get("person", {}) or {}
            position = record.get("position", {}) or {}
            candidates.append(
                {
                    "player_id": int(numeric_id),
                    "Player": person.get("fullName") or f"MLB ID {int(numeric_id)}",
                    "Position": position.get("abbreviation") or position.get("name") or "",
                    "LineupSpot": spot,
                    "RawBattingOrder": spot * 100,
                }
            )
    if not candidates:
        return pd.DataFrame(columns=columns)
    lineup = pd.DataFrame(candidates).sort_values(["LineupSpot", "RawBattingOrder"])
    return lineup.drop_duplicates("LineupSpot").head(9).reset_index(drop=True)[columns]


@st.cache_data(ttl=300, show_spinner=False, max_entries=64)
def fetch_game_lineups(game_pk: object) -> dict:
    numeric_game = pd.to_numeric(pd.Series([game_pk]), errors="coerce").iloc[0]
    if pd.isna(numeric_game):
        return {"ok": False, "away": pd.DataFrame(), "home": pd.DataFrame(), "error": "No game ID."}
    game_pk_int = int(numeric_game)
    endpoints = [
        ("MLB live game feed", f"https://statsapi.mlb.com/api/v1.1/game/{game_pk_int}/feed/live"),
        ("MLB boxscore", f"https://statsapi.mlb.com/api/v1/game/{game_pk_int}/boxscore"),
    ]
    errors: list[str] = []
    for source, url in endpoints:
        request = Request(url, headers={"User-Agent": "MLB-Pitcher-Lab/1.0", "Accept": "application/json"})
        try:
            with urlopen(request, timeout=15) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
            continue
        if "liveData" in payload:
            teams = payload.get("liveData", {}).get("boxscore", {}).get("teams", {})
            game_state = payload.get("gameData", {}).get("status", {}).get("detailedState", "")
            updated = payload.get("metaData", {}).get("timeStamp", "")
        else:
            teams = payload.get("teams", {})
            game_state = ""
            updated = ""
        away = _parse_team_lineup(teams.get("away", {}))
        home = _parse_team_lineup(teams.get("home", {}))
        away_count = int(away["LineupSpot"].nunique()) if not away.empty else 0
        home_count = int(home["LineupSpot"].nunique()) if not home.empty else 0
        if away_count or home_count:
            return {
                "ok": away_count >= 9 and home_count >= 9,
                "away": away,
                "home": home,
                "away_status": "Confirmed" if away_count >= 9 else ("Partial" if away_count else "Not posted"),
                "home_status": "Confirmed" if home_count >= 9 else ("Partial" if home_count else "Not posted"),
                "source": source,
                "game_state": game_state,
                "updated": updated,
                "error": None,
            }
    return {
        "ok": False,
        "away": pd.DataFrame(),
        "home": pd.DataFrame(),
        "away_status": "Not posted",
        "home_status": "Not posted",
        "source": "Recent lineup fallback",
        "game_state": "",
        "updated": "",
        "error": " | ".join(errors[-2:]) if errors else "No posted batting order yet.",
    }


@st.cache_data(ttl=21600, show_spinner=False, max_entries=256)
def fetch_pitcher_game_log(pitcher_id: int, season: int) -> tuple[pd.DataFrame, str | None]:
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{int(pitcher_id)}/stats"
        f"?stats=gameLog&group=pitching&season={int(season)}"
    )
    request = Request(url, headers={"User-Agent": "MLB-Pitcher-Lab/1.0", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"

    splits: list[dict] = []
    for stats_block in payload.get("stats", []):
        for split in stats_block.get("splits", []):
            stat = split.get("stat", {}) or {}
            opponent = split.get("opponent", {}) or {}
            team = split.get("team", {}) or {}
            row = {
                "Date": pd.to_datetime(split.get("date"), errors="coerce"),
                "Opponent": opponent.get("name", ""),
                "Team": team.get("name", ""),
                "GamesStarted": pd.to_numeric(pd.Series([stat.get("gamesStarted")]), errors="coerce").iloc[0],
                "IP": stat.get("inningsPitched"),
                "K": pd.to_numeric(pd.Series([stat.get("strikeOuts")]), errors="coerce").iloc[0],
                "ER": pd.to_numeric(pd.Series([stat.get("earnedRuns")]), errors="coerce").iloc[0],
                "BF": pd.to_numeric(pd.Series([stat.get("battersFaced")]), errors="coerce").iloc[0],
                "Pitches": pd.to_numeric(pd.Series([stat.get("numberOfPitches")]), errors="coerce").iloc[0],
                "Hits": pd.to_numeric(pd.Series([stat.get("hits")]), errors="coerce").iloc[0],
                "BB": pd.to_numeric(pd.Series([stat.get("baseOnBalls")]), errors="coerce").iloc[0],
                "HR": pd.to_numeric(pd.Series([stat.get("homeRuns")]), errors="coerce").iloc[0],
            }
            row["Outs"] = innings_to_outs(row["IP"])
            splits.append(row)
    if not splits:
        return pd.DataFrame(), "The MLB game-log response contained no pitching splits."
    frame = pd.DataFrame(splits).sort_values("Date").reset_index(drop=True)
    for column in ["GamesStarted", "K", "ER", "BF", "Pitches", "Hits", "BB", "HR", "Outs"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    starters = frame[frame["GamesStarted"].fillna(0).ge(1)].copy()
    if starters.empty:
        starters = frame[(frame["Pitches"].fillna(0).ge(40)) | (frame["Outs"].fillna(0).ge(9))].copy()
    return starters.reset_index(drop=True), None


# -----------------------------------------------------------------------------
# Lineup, pitcher and pitch-type profiles
# -----------------------------------------------------------------------------

def infer_recent_lineup(df: pd.DataFrame, team: str) -> pd.DataFrame:
    team_code = str(team or "").upper()
    team_rows = df[df["batter_team"].astype(str).str.upper().isin({team_code, TEAM_ALIASES.get(team_code, "")})].copy()
    team_rows = team_rows[team_rows["game_date"].notna()]
    if team_rows.empty:
        return pd.DataFrame(columns=["player_id", "Player", "Position", "LineupSpot"])
    latest_date = team_rows["game_date"].max()
    date_rows = team_rows[team_rows["game_date"].eq(latest_date)]
    latest_game = pd.to_numeric(date_rows["game_pk"], errors="coerce").max()
    game_rows = date_rows[date_rows["game_pk"].eq(latest_game)]
    order = (
        game_rows.groupby("batter")["at_bat_number"]
        .min()
        .sort_values()
        .head(9)
        .reset_index()
        .rename(columns={"batter": "player_id"})
    )
    order["LineupSpot"] = np.arange(1, len(order) + 1)
    names = lookup_names(tuple(order["player_id"].dropna().astype(int).tolist()))
    order = order.merge(names, on="player_id", how="left")
    order["Player"] = order["Player"].fillna(order["player_id"].apply(lambda value: f"MLB ID {int(value)}"))
    order["Position"] = ""
    return order[["player_id", "Player", "Position", "LineupSpot"]]


def pitcher_statcast_profile(df: pd.DataFrame, pitcher_id: int, league: dict) -> dict:
    pitches = df[df["pitcher"].eq(int(pitcher_id))].copy()
    pa = pitches[pitches["is_pa_end"]].copy()
    plate_appearances = int(pa["pa_key"].nunique()) if not pa.empty else 0
    swings = int(pitches["is_swing"].sum())
    bbe = int(pitches["is_bbe"].sum())
    pitch_count = len(pitches)
    hand = "R"
    if not pitches.empty and not pitches["p_throws"].dropna().empty:
        hand = str(pitches["p_throws"].dropna().mode().iloc[0])
    k_rate = (int(pa["is_k"].sum()) + league["K_PA"] * 150) / (plate_appearances + 150)
    bb_rate = (int(pa["is_bb"].sum()) + league["BB_PA"] * 150) / (plate_appearances + 150)
    hr_rate = (int(pa["is_hr"].sum()) + league["HR_PA"] * 180) / (plate_appearances + 180)
    xwoba = (float(pa["xwoba_value"].sum()) + league["xwOBA"] * 180) / (plate_appearances + 180)
    barrel = (int(pitches["is_barrel"].sum()) + league["Brl_BBE"] * 100) / (bbe + 100)
    hard_hit = (int(pitches["is_hard_hit"].sum()) + league["HH_BBE"] * 100) / (bbe + 100)
    whiff = (int(pitches["is_whiff"].sum()) + league["Whiff"] * 200) / (swings + 200)
    csw = (
        int(pitches["is_whiff"].sum())
        + int(pitches["is_called_strike"].sum())
        + league["CSW"] * 350
    ) / (pitch_count + 350)
    return {
        "Hand": hand if hand in {"L", "R"} else "R",
        "Statcast_PA": plate_appearances,
        "PitchCountSample": pitch_count,
        "K_PA": float(k_rate),
        "BB_PA": float(bb_rate),
        "HR_PA": float(hr_rate),
        "xwOBA_Allowed": float(xwoba),
        "Brl_BBE_Allowed": float(barrel),
        "HH_BBE_Allowed": float(hard_hit),
        "Whiff_Pct": float(whiff),
        "CSW_Pct": float(csw),
    }


def build_lineup_profile(
    df: pd.DataFrame,
    lineup: pd.DataFrame,
    opponent_team: str,
    pitcher_hand: str,
    league: dict,
) -> tuple[dict, pd.DataFrame]:
    lineup = lineup.copy() if lineup is not None else pd.DataFrame()
    lineup_ids = (
        pd.to_numeric(lineup.get("player_id", pd.Series(dtype=float)), errors="coerce")
        .dropna().astype(int).tolist()
    )
    team_code = str(opponent_team or "").upper()
    team_values = {team_code}
    if TEAM_ALIASES.get(team_code):
        team_values.add(TEAM_ALIASES[team_code])

    if lineup_ids:
        sample = df[df["batter"].isin(lineup_ids) & df["p_throws"].eq(pitcher_hand)].copy()
    else:
        sample = df[df["batter_team"].isin(team_values) & df["p_throws"].eq(pitcher_hand)].copy()

    pa = sample[sample["is_pa_end"]].copy()
    if lineup.empty and not sample.empty:
        lineup = infer_recent_lineup(df, opponent_team)
        lineup_ids = lineup["player_id"].dropna().astype(int).tolist()

    player_ids = lineup_ids or sorted(pa["batter"].dropna().astype(int).unique().tolist())[:9]
    base = pd.DataFrame({"player_id": player_ids})
    if not lineup.empty:
        base = base.merge(lineup[["player_id", "Player", "Position", "LineupSpot"]], on="player_id", how="left")
    names = lookup_names(tuple(player_ids))
    base = base.merge(names, on="player_id", how="left", suffixes=("", "_Lookup"))
    if "Player" not in base.columns:
        base["Player"] = np.nan
    if "Player_Lookup" in base.columns:
        base["Player"] = base["Player"].fillna(base["Player_Lookup"])
    base["Player"] = base["Player"].fillna(base["player_id"].apply(lambda value: f"MLB ID {int(value)}"))
    base["LineupSpot"] = pd.to_numeric(base.get("LineupSpot"), errors="coerce")
    base["LineupSpot"] = base["LineupSpot"].fillna(pd.Series(np.arange(1, len(base) + 1), index=base.index)).clip(1, 9)

    if pa.empty:
        for column, value in {
            "PA": 0, "K": 0, "BB": 0, "HR": 0, "xwOBA": league["xwOBA"],
            "K_Pct": league["K_PA"], "BB_Pct": league["BB_PA"], "HR_Pct": league["HR_PA"],
            "Brl_BBE": league["Brl_BBE"], "HH_BBE": league["HH_BBE"],
            "Whiff_Pct": league["Whiff"], "EffectiveStand": "R" if pitcher_hand == "L" else "L",
        }.items():
            base[column] = value
    else:
        pa_stats = (
            pa.groupby("batter")
            .agg(PA=("pa_key", "nunique"), K=("is_k", "sum"), BB=("is_bb", "sum"), HR=("is_hr", "sum"), xwOBA_Total=("xwoba_value", "sum"))
            .reset_index().rename(columns={"batter": "player_id"})
        )
        pitch_stats = (
            sample.groupby("batter")
            .agg(Swings=("is_swing", "sum"), Whiffs=("is_whiff", "sum"), BBE=("is_bbe", "sum"), Barrels=("is_barrel", "sum"), HardHits=("is_hard_hit", "sum"))
            .reset_index().rename(columns={"batter": "player_id"})
        )
        stands = sample.groupby("batter")["stand"].agg(lambda values: values.dropna().mode().iloc[0] if not values.dropna().empty else np.nan).reset_index().rename(columns={"batter": "player_id", "stand": "EffectiveStand"})
        base = base.merge(pa_stats, on="player_id", how="left").merge(pitch_stats, on="player_id", how="left").merge(stands, on="player_id", how="left")
        for column in ["PA", "K", "BB", "HR", "xwOBA_Total", "Swings", "Whiffs", "BBE", "Barrels", "HardHits"]:
            base[column] = pd.to_numeric(base.get(column, 0), errors="coerce").fillna(0.0)
        base["K_Pct"] = (base["K"] + league["K_PA"] * 30) / (base["PA"] + 30)
        base["BB_Pct"] = (base["BB"] + league["BB_PA"] * 30) / (base["PA"] + 30)
        base["HR_Pct"] = (base["HR"] + league["HR_PA"] * 40) / (base["PA"] + 40)
        base["xwOBA"] = (base["xwOBA_Total"] + league["xwOBA"] * 35) / (base["PA"] + 35)
        base["Brl_BBE"] = (base["Barrels"] + league["Brl_BBE"] * 25) / (base["BBE"] + 25)
        base["HH_BBE"] = (base["HardHits"] + league["HH_BBE"] * 25) / (base["BBE"] + 25)
        base["Whiff_Pct"] = (base["Whiffs"] + league["Whiff"] * 50) / (base["Swings"] + 50)
        base["EffectiveStand"] = base["EffectiveStand"].where(base["EffectiveStand"].isin(["L", "R"]), "R" if pitcher_hand == "L" else "L")

    lineup_weight_map = {1: 1.08, 2: 1.07, 3: 1.05, 4: 1.04, 5: 1.01, 6: 0.99, 7: 0.96, 8: 0.93, 9: 0.90}
    base["Weight"] = base["LineupSpot"].round().map(lineup_weight_map).fillna(1.0)
    weights = base["Weight"].to_numpy(dtype=float)
    profile = {
        "Lineup_PA_Sample": int(pd.to_numeric(base.get("PA", 0), errors="coerce").fillna(0).sum()),
        "Opp_K_PA": float(np.average(base["K_Pct"], weights=weights)) if len(base) else league["K_PA"],
        "Opp_BB_PA": float(np.average(base["BB_Pct"], weights=weights)) if len(base) else league["BB_PA"],
        "Opp_HR_PA": float(np.average(base["HR_Pct"], weights=weights)) if len(base) else league["HR_PA"],
        "Opp_xwOBA": float(np.average(base["xwOBA"], weights=weights)) if len(base) else league["xwOBA"],
        "Opp_Brl_BBE": float(np.average(base["Brl_BBE"], weights=weights)) if len(base) else league["Brl_BBE"],
        "Opp_HH_BBE": float(np.average(base["HH_BBE"], weights=weights)) if len(base) else league["HH_BBE"],
        "Opp_Whiff": float(np.average(base["Whiff_Pct"], weights=weights)) if len(base) else league["Whiff"],
    }
    keep = ["LineupSpot", "Player", "Position", "EffectiveStand", "PA", "K_Pct", "BB_Pct", "HR_Pct", "xwOBA", "Brl_BBE", "HH_BBE", "Whiff_Pct"]
    return profile, base[[column for column in keep if column in base.columns]].sort_values("LineupSpot")


def build_pitch_type_matchup(
    df: pd.DataFrame,
    pitcher_id: int,
    lineup: pd.DataFrame,
    opponent_team: str,
    pitcher_hand: str,
    league: dict,
) -> tuple[float, pd.DataFrame]:
    pitcher_pitches = df[df["pitcher"].eq(int(pitcher_id)) & df["pitch_name"].notna()].copy()
    if pitcher_pitches.empty:
        return 50.0, pd.DataFrame()
    total_pitches = max(len(pitcher_pitches), 1)
    pitcher_rows = []
    for pitch_name, group in pitcher_pitches.groupby("pitch_name"):
        usage = len(group) / total_pitches
        if usage < 0.025:
            continue
        swings = max(int(group["is_swing"].sum()), 1)
        pa = group[group["is_pa_end"]]
        pa_count = max(int(pa["pa_key"].nunique()), 1)
        pitcher_rows.append(
            {
                "Pitch Type": str(pitch_name),
                "Usage": usage,
                "Velocity": float(pd.to_numeric(group["release_speed"], errors="coerce").mean()),
                "Pitcher Whiff%": float(group["is_whiff"].sum() / swings),
                "Pitcher CSW%": float((group["is_whiff"].sum() + group["is_called_strike"].sum()) / max(len(group), 1)),
                "Pitcher xwOBA": float(pa["xwoba_value"].sum() / pa_count),
            }
        )
    pitch_table = pd.DataFrame(pitcher_rows)
    if pitch_table.empty:
        return 50.0, pd.DataFrame()

    lineup_ids = pd.to_numeric(lineup.get("player_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist() if lineup is not None else []
    team_code = str(opponent_team or "").upper()
    team_values = {team_code, TEAM_ALIASES.get(team_code, team_code)}
    if lineup_ids:
        hitter_sample = df[df["batter"].isin(lineup_ids) & df["p_throws"].eq(pitcher_hand)].copy()
    else:
        hitter_sample = df[df["batter_team"].isin(team_values) & df["p_throws"].eq(pitcher_hand)].copy()

    matchup_rows: list[dict] = []
    for row in pitch_table.itertuples(index=False):
        pitch_name = getattr(row, "_0", None) if not hasattr(row, "Pitch_Type") else row.Pitch_Type
        # itertuples sanitizes spaces; locate by position to stay robust.
        pitch_name = row[0]
        sample = hitter_sample[hitter_sample["pitch_name"].eq(pitch_name)]
        swings = int(sample["is_swing"].sum())
        pa = sample[sample["is_pa_end"]]
        pa_count = int(pa["pa_key"].nunique())
        league_pitch = df[df["pitch_name"].eq(pitch_name) & df["p_throws"].eq(pitcher_hand)]
        league_pitch_pa = league_pitch[league_pitch["is_pa_end"]]
        league_pa_count = max(int(league_pitch_pa["pa_key"].nunique()), 1)
        league_swings = max(int(league_pitch["is_swing"].sum()), 1)
        league_pitch_k = float(league_pitch_pa["is_k"].sum() / league_pa_count)
        league_pitch_xwoba = float(league_pitch_pa["xwoba_value"].sum() / league_pa_count)
        league_pitch_whiff = float(league_pitch["is_whiff"].sum() / league_swings)
        hitter_k = (int(pa["is_k"].sum()) + league_pitch_k * 35) / (pa_count + 35)
        hitter_xwoba = (float(pa["xwoba_value"].sum()) + league_pitch_xwoba * 35) / (pa_count + 35)
        hitter_whiff = (int(sample["is_whiff"].sum()) + league_pitch_whiff * 60) / (swings + 60)
        k_ratio = hitter_k / max(league_pitch_k, 0.001)
        whiff_ratio = hitter_whiff / max(league_pitch_whiff, 0.001)
        xwoba_ratio = max(league_pitch_xwoba, 0.001) / max(hitter_xwoba, 0.001)
        raw_ratio = k_ratio ** 0.42 * whiff_ratio ** 0.28 * xwoba_ratio ** 0.30
        score = float(np.clip(50.0 + 65.0 * math.log(max(raw_ratio, 0.25)), 0.0, 100.0))
        matchup_rows.append(
            {
                "Pitch Type": pitch_name,
                "Usage": float(row[1]),
                "Velocity": float(row[2]),
                "Pitcher Whiff%": float(row[3]),
                "Pitcher CSW%": float(row[4]),
                "Pitcher xwOBA": float(row[5]),
                "Opponent Pitches": int(len(sample)),
                "Opponent PA": int(pa_count),
                "Opponent K%": float(hitter_k),
                "Opponent Whiff%": float(hitter_whiff),
                "Opponent xwOBA": float(hitter_xwoba),
                "Match Score": score,
            }
        )
    result = pd.DataFrame(matchup_rows)
    if result.empty:
        return 50.0, result
    overall = float(np.average(result["Match Score"], weights=result["Usage"]))
    return overall, result.sort_values("Usage", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Park and weather
# -----------------------------------------------------------------------------

def load_local_csv(filename: str) -> pd.DataFrame:
    paths = [Path(__file__).resolve().parent / filename, Path.cwd() / filename]
    path = next((candidate for candidate in paths if candidate.exists()), None)
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def park_run_factor(venue: str, lineup_table: pd.DataFrame) -> tuple[float, str]:
    table = load_local_csv("park_factors.csv")
    if table.empty or "venue" not in table.columns:
        return 100.0, "Neutral fallback"
    matches = table[table["venue"].apply(lambda value: venue_names_match(value, venue))]
    if matches.empty:
        return 100.0, "Neutral fallback"
    row = matches.iloc[0]
    if lineup_table is None or lineup_table.empty or "EffectiveStand" not in lineup_table.columns:
        hit_factor = np.mean([float(row.get("hits_l", 100)), float(row.get("hits_r", 100))])
        hr_factor = np.mean([float(row.get("hr_l", 100)), float(row.get("hr_r", 100))])
    else:
        weights = np.array([1.08, 1.07, 1.05, 1.04, 1.01, 0.99, 0.96, 0.93, 0.90])[: len(lineup_table)]
        hit_values = []
        hr_values = []
        for stand in lineup_table["EffectiveStand"].fillna("R"):
            suffix = "l" if str(stand).upper() == "L" else "r"
            hit_values.append(float(row.get(f"hits_{suffix}", 100)))
            hr_values.append(float(row.get(f"hr_{suffix}", 100)))
        hit_factor = float(np.average(hit_values, weights=weights))
        hr_factor = float(np.average(hr_values, weights=weights))
    factor = 0.68 * hit_factor + 0.32 * hr_factor
    return float(factor), f"park_factors.csv · {int(row.get('rolling_years', 3))}-year"


def load_stadium_weather_metadata(venue: str) -> dict:
    table = load_local_csv("stadium_weather.csv")
    if table.empty or "venue" not in table.columns:
        return {"ok": False, "roof_type": "open", "error": "stadium_weather.csv was unavailable."}
    matches = table[table["venue"].apply(lambda value: venue_names_match(value, venue))]
    if matches.empty:
        return {"ok": False, "roof_type": "open", "error": f"No weather metadata matched {venue}."}
    row = matches.iloc[0]
    return {
        "ok": True,
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "outfield_bearing": float(row["outfield_bearing"]),
        "roof_type": str(row["roof_type"]).lower().strip(),
        "error": None,
    }


def classify_stadium_wind(wind_from_degrees: float, outfield_bearing: float | None) -> str:
    if outfield_bearing is None or pd.isna(wind_from_degrees):
        return "Cross/Calm"
    wind_toward = (float(wind_from_degrees) + 180.0) % 360.0
    difference = abs((wind_toward - float(outfield_bearing) + 180.0) % 360.0 - 180.0)
    if difference <= 45.0:
        return "Out"
    if difference >= 135.0:
        return "In"
    return "Cross/Calm"


@st.cache_data(ttl=1800, show_spinner=False, max_entries=64)
def fetch_game_weather(latitude: float, longitude: float, game_datetime_utc: str, outfield_bearing: float | None) -> dict:
    target = pd.to_datetime(game_datetime_utc, utc=True, errors="coerce")
    if pd.isna(target):
        return {"ok": False, "error": "No valid first-pitch time."}
    target_date = target.strftime("%Y-%m-%d")
    params = {
        "latitude": round(float(latitude), 5),
        "longitude": round(float(longitude), 5),
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,precipitation_probability",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "GMT",
        "start_date": target_date,
        "end_date": target_date,
    }
    request = Request(
        "https://api.open-meteo.com/v1/forecast?" + urlencode(params),
        headers={"User-Agent": "MLB-Pitcher-Lab/1.0", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.load(response)
        hourly = pd.DataFrame(payload.get("hourly", {}))
        hourly["forecast_time"] = pd.to_datetime(hourly["time"], utc=True, errors="coerce")
        hourly = hourly.dropna(subset=["forecast_time"])
        if hourly.empty:
            raise ValueError("No hourly weather rows")
        row = hourly.loc[(hourly["forecast_time"] - target).abs().idxmin()]
        wind_from = float(row.get("wind_direction_10m", 0.0))
        return {
            "ok": True,
            "temperature_f": float(row.get("temperature_2m", 70.0)),
            "humidity_pct": float(row.get("relative_humidity_2m", 50.0)),
            "pressure_hpa": float(row.get("pressure_msl", 1013.25)),
            "wind_mph": float(row.get("wind_speed_10m", 0.0)),
            "wind_direction": classify_stadium_wind(wind_from, outfield_bearing),
            "precip_probability": float(row.get("precipitation_probability", 0.0)),
            "forecast_time": row["forecast_time"].strftime("%Y-%m-%d %H:%M UTC"),
            "error": None,
        }
    except (HTTPError, URLError, TimeoutError, ValueError, OSError, TypeError, KeyError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def automatic_weather(game: dict) -> dict:
    metadata = load_stadium_weather_metadata(str(game.get("venue", "")))
    if not metadata.get("ok"):
        return {"ok": False, "metadata": metadata, "error": metadata.get("error")}
    result = fetch_game_weather(
        metadata["latitude"], metadata["longitude"], str(game.get("game_datetime_utc") or ""), metadata.get("outfield_bearing")
    )
    result["metadata"] = metadata
    return result


def weather_run_multiplier(weather: dict) -> float:
    metadata = weather.get("metadata", {}) if isinstance(weather, dict) else {}
    roof_type = str(metadata.get("roof_type", "open"))
    if roof_type == "fixed":
        return 1.0
    if not weather.get("ok"):
        return 1.0
    temperature = float(weather.get("temperature_f", 70.0))
    humidity = float(weather.get("humidity_pct", 50.0))
    pressure = float(weather.get("pressure_hpa", 1013.25))
    wind = float(weather.get("wind_mph", 0.0))
    wind_direction = str(weather.get("wind_direction", "Cross/Calm"))
    carry = 1.0 + (temperature - 70.0) * 0.0025 + (humidity - 50.0) * 0.0003 + (1013.25 - pressure) * 0.00025
    if wind_direction == "Out":
        carry += wind * 0.007
    elif wind_direction == "In":
        carry -= wind * 0.007
    return float(np.clip(1.0 + (carry - 1.0) * 0.45, 0.88, 1.14))


# -----------------------------------------------------------------------------
# Projection model
# -----------------------------------------------------------------------------

def game_log_profile(logs: pd.DataFrame, league: dict) -> dict:
    if logs is None or logs.empty:
        return {
            "Starts": 0, "Season_Outs": 16.5, "Recent_Outs": 16.5, "Outs_SD": 2.7,
            "Season_BF": 22.5, "Recent_BF": 22.5, "Season_Pitches": 88.0, "Recent_Pitches": 88.0,
            "Season_K_BF": league["K_PA"], "Recent_K_BF": league["K_PA"],
            "Season_ER9": league["ER9"], "Recent_ER9": league["ER9"],
        }
    clean = logs.dropna(subset=["Outs"]).copy()
    if clean.empty:
        return game_log_profile(pd.DataFrame(), league)
    clean["K_BF"] = safe_divide(clean["K"], clean["BF"], league["K_PA"])
    clean["ER9"] = safe_divide(clean["ER"] * 27.0, clean["Outs"], league["ER9"])
    return {
        "Starts": int(len(clean)),
        "Season_Outs": float(clean["Outs"].mean()),
        "Recent_Outs": weighted_recent_mean(clean["Outs"], 5, clean["Outs"].mean()),
        "Outs_SD": float(max(clean["Outs"].tail(10).std(ddof=1) if len(clean.tail(10)) > 1 else 2.5, 1.5)),
        "Season_BF": float(clean["BF"].mean()),
        "Recent_BF": weighted_recent_mean(clean["BF"], 5, clean["BF"].mean()),
        "Season_Pitches": float(clean["Pitches"].mean()),
        "Recent_Pitches": weighted_recent_mean(clean["Pitches"], 5, clean["Pitches"].mean()),
        "Season_K_BF": float(clean["K"].sum() / max(clean["BF"].sum(), 1.0)),
        "Recent_K_BF": weighted_recent_mean(clean["K_BF"], 5, clean["K_BF"].mean()),
        "Season_ER9": float(clean["ER"].sum() * 27.0 / max(clean["Outs"].sum(), 1.0)),
        "Recent_ER9": weighted_recent_mean(clean["ER9"], 5, clean["ER9"].mean()),
    }


def build_pitcher_projection(
    df: pd.DataFrame,
    league: dict,
    game: dict,
    pitcher_id: int,
    pitcher_name: str,
    pitcher_team: str,
    opponent_team: str,
    opponent_side: str,
    lineup: pd.DataFrame,
    lineup_status: str,
    weather: dict,
    season: int,
) -> tuple[dict, dict]:
    logs, log_error = fetch_pitcher_game_log(int(pitcher_id), int(season))
    log = game_log_profile(logs, league)
    pitcher = pitcher_statcast_profile(df, int(pitcher_id), league)
    opponent, hitter_table = build_lineup_profile(df, lineup, opponent_team, pitcher["Hand"], league)
    pitch_match_score, pitch_match_table = build_pitch_type_matchup(df, int(pitcher_id), lineup, opponent_team, pitcher["Hand"], league)
    park_factor, park_source = park_run_factor(str(game.get("venue", "")), hitter_table)
    weather_factor = weather_run_multiplier(weather)

    base_outs = 0.55 * log["Recent_Outs"] + 0.30 * log["Season_Outs"] + 0.15 * 16.5
    lineup_difficulty = (
        (opponent["Opp_xwOBA"] / max(league["xwOBA"], 0.001)) ** 0.18
        * (opponent["Opp_BB_PA"] / max(league["BB_PA"], 0.001)) ** 0.06
        * (pitcher["BB_PA"] / max(league["BB_PA"], 0.001)) ** 0.05
    )
    projected_outs = float(np.clip(base_outs / max(lineup_difficulty, 0.75), 9.0, 22.0))
    bf_per_out = (
        0.55 * (log["Recent_BF"] / max(log["Recent_Outs"], 1.0))
        + 0.45 * (log["Season_BF"] / max(log["Season_Outs"], 1.0))
    )
    projected_bf = float(np.clip(projected_outs * bf_per_out, 13.0, 31.0))
    projected_pitches = float(np.clip(0.60 * log["Recent_Pitches"] + 0.40 * log["Season_Pitches"], 65.0, 112.0))

    base_k_rate = 0.35 * pitcher["K_PA"] + 0.35 * log["Recent_K_BF"] + 0.30 * log["Season_K_BF"]
    matchup_factor = (
        (opponent["Opp_K_PA"] / max(league["K_PA"], 0.001)) ** 0.36
        * (opponent["Opp_Whiff"] / max(league["Whiff"], 0.001)) ** 0.16
        * (pitcher["Whiff_Pct"] / max(league["Whiff"], 0.001)) ** 0.16
        * (1.0 + (pitch_match_score - 50.0) / 320.0)
    )
    adjusted_k_rate = float(np.clip(base_k_rate * matchup_factor, 0.075, 0.46))
    projected_k = float(np.clip(projected_bf * adjusted_k_rate, 1.0, 13.5))

    quality_er9 = (
        league["ER9"]
        * (pitcher["xwOBA_Allowed"] / max(league["xwOBA"], 0.001)) ** 1.25
        * (pitcher["BB_PA"] / max(league["BB_PA"], 0.001)) ** 0.18
        * (pitcher["Brl_BBE_Allowed"] / max(league["Brl_BBE"], 0.001)) ** 0.24
        * (pitcher["HR_PA"] / max(league["HR_PA"], 0.001)) ** 0.15
    )
    base_er9 = 0.45 * log["Recent_ER9"] + 0.35 * log["Season_ER9"] + 0.20 * quality_er9
    offense_factor = (
        (opponent["Opp_xwOBA"] / max(league["xwOBA"], 0.001)) ** 0.62
        * (opponent["Opp_BB_PA"] / max(league["BB_PA"], 0.001)) ** 0.13
        * (opponent["Opp_Brl_BBE"] / max(league["Brl_BBE"], 0.001)) ** 0.18
        * (opponent["Opp_HR_PA"] / max(league["HR_PA"], 0.001)) ** 0.07
    )
    adjusted_er9 = float(np.clip(base_er9 * offense_factor * (park_factor / 100.0) ** 0.58 * weather_factor, 1.35, 9.0))
    projected_er = float(np.clip(adjusted_er9 * projected_outs / 27.0, 0.35, 6.5))

    p_0_1 = poisson_cdf(1, projected_er)
    p_0_3 = poisson_cdf(3, projected_er)
    p_2_3 = max(p_0_3 - p_0_1, 0.0)
    p_4_plus = max(1.0 - p_0_3, 0.0)

    sample_conf = 100.0 * (1.0 - math.exp(-max(pitcher["Statcast_PA"], 0) / 220.0))
    starts_conf = 100.0 * (1.0 - math.exp(-max(log["Starts"], 0) / 8.0))
    lineup_conf = {"Confirmed": 100.0, "Partial": 72.0, "Recent fallback": 55.0, "Not posted": 48.0}.get(lineup_status, 50.0)
    confidence = float(np.clip(0.40 * sample_conf + 0.40 * starts_conf + 0.20 * lineup_conf, 0.0, 100.0))
    confidence_level = "High" if confidence >= 70 else ("Medium" if confidence >= 45 else "Low")

    row = {
        "PitcherID": int(pitcher_id),
        "Pitcher": str(pitcher_name),
        "Team": pitcher_team,
        "Opponent": opponent_team,
        "Matchup": f"{game.get('away_abbr')} @ {game.get('home_abbr')}",
        "Game": f"{game.get('away_abbr')} @ {game.get('home_abbr')} · {game.get('time_et')}",
        "GamePK": game.get("game_pk"),
        "SlateDate": game.get("slate_date"),
        "GameDateTimeUTC": game.get("game_datetime_utc"),
        "PitcherLocation": "Home" if str(pitcher_team) == str(game.get("home_abbr")) else "Away",
        "Venue": game.get("venue"),
        "Hand": pitcher["Hand"],
        "Proj_K": projected_k,
        "Proj_ER": projected_er,
        "Proj_Outs": projected_outs,
        "Proj_Innings": projected_outs / 3.0,
        "Proj_BF": projected_bf,
        "Proj_Pitches": projected_pitches,
        "Adj_K_Rate": adjusted_k_rate,
        "Pitcher_K_Rate": pitcher["K_PA"],
        "Opponent_K_Rate": opponent["Opp_K_PA"],
        "Whiff_Pct": pitcher["Whiff_Pct"],
        "CSW_Pct": pitcher["CSW_Pct"],
        "PitchTypeScore": pitch_match_score,
        "xwOBA_Allowed": pitcher["xwOBA_Allowed"],
        "BB_Rate": pitcher["BB_PA"],
        "K_minus_BB": pitcher["K_PA"] - pitcher["BB_PA"],
        "Barrel_Allowed": pitcher["Brl_BBE_Allowed"],
        "HardHit_Allowed": pitcher["HH_BBE_Allowed"],
        "Opp_xwOBA": opponent["Opp_xwOBA"],
        "Opp_Barrel": opponent["Opp_Brl_BBE"],
        "Recent_Outs": log["Recent_Outs"],
        "Recent_K_Rate": log["Recent_K_BF"],
        "Recent_ER9": log["Recent_ER9"],
        "Season_ER9": log["Season_ER9"],
        "Outs_SD": log["Outs_SD"],
        "Starts": log["Starts"],
        "Park_Run_Factor": park_factor,
        "Park_Source": park_source,
        "Weather_Factor": weather_factor,
        "Lineup_Status": lineup_status,
        "Lineup_Sample_PA": opponent["Lineup_PA_Sample"],
        "Confidence": confidence,
        "Confidence_Level": confidence_level,
        "P_0_1_ER": p_0_1,
        "P_2_3_ER": p_2_3,
        "P_4plus_ER": p_4_plus,
    }
    detail = {
        "lineup": hitter_table,
        "pitch_types": pitch_match_table,
        "game_log": logs.tail(10).copy() if logs is not None and not logs.empty else pd.DataFrame(),
        "weather": weather,
        "log_error": log_error,
        "park_source": park_source,
        "opponent_side": opponent_side,
    }
    return row, detail


def parse_calibration_payload(payload: object) -> dict:
    """Normalize a calibration JSON payload into the app's expected structure."""
    if not isinstance(payload, dict):
        return {}
    targets = payload.get("targets", payload)
    if not isinstance(targets, dict):
        return {}
    normalized: dict[str, dict] = {}
    aliases = {
        "K": "Strikeouts", "Strikeouts": "Strikeouts",
        "ER": "Earned runs", "Earned runs": "Earned runs",
        "Outs": "Outs",
    }
    for key, value in targets.items():
        target_name = aliases.get(str(key))
        if target_name is None or not isinstance(value, dict):
            continue
        slope = pd.to_numeric(pd.Series([value.get("slope")]), errors="coerce").iloc[0]
        intercept = pd.to_numeric(pd.Series([value.get("intercept")]), errors="coerce").iloc[0]
        if pd.isna(slope) or pd.isna(intercept):
            continue
        normalized[target_name] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "n": int(pd.to_numeric(pd.Series([value.get("n", 0)]), errors="coerce").fillna(0).iloc[0]),
        }
    if not normalized:
        return {}
    return {
        "version": str(payload.get("version", "pitcher-calibration")),
        "created_at_utc": str(payload.get("created_at_utc", "")),
        "targets": normalized,
    }


def apply_projection_calibration(board: pd.DataFrame, calibration: dict | None) -> pd.DataFrame:
    """Apply saved linear calibration while retaining the uncalibrated projections."""
    result = board.copy()
    payload = parse_calibration_payload(calibration or {})
    target_payload = payload.get("targets", {}) if payload else {}
    limits = {"Strikeouts": (0.0, 18.0), "Earned runs": (0.0, 10.0), "Outs": (3.0, 27.0)}
    for target_name, config in BACKTEST_TARGETS.items():
        projection_col = config["projection"]
        raw_col = f"Raw_{projection_col}"
        if projection_col not in result.columns:
            continue
        if raw_col not in result.columns:
            result[raw_col] = pd.to_numeric(result[projection_col], errors="coerce")
        else:
            result[raw_col] = pd.to_numeric(result[raw_col], errors="coerce")
        coefficients = target_payload.get(target_name)
        if not coefficients:
            result[projection_col] = result[raw_col]
            continue
        calibrated = float(coefficients["intercept"]) + float(coefficients["slope"]) * result[raw_col]
        lower, upper = limits[target_name]
        result[projection_col] = calibrated.clip(lower, upper)
    if "Proj_Outs" in result.columns:
        result["Proj_Innings"] = pd.to_numeric(result["Proj_Outs"], errors="coerce") / 3.0
    if "Proj_ER" in result.columns:
        er_means = pd.to_numeric(result["Proj_ER"], errors="coerce").fillna(0.0)
        result["P_0_1_ER"] = er_means.map(lambda mean: poisson_cdf(1, float(mean)))
        result["P_2_3_ER"] = er_means.map(lambda mean: max(poisson_cdf(3, float(mean)) - poisson_cdf(1, float(mean)), 0.0))
        result["P_4plus_ER"] = er_means.map(lambda mean: max(1.0 - poisson_cdf(3, float(mean)), 0.0))
    result["Calibration_Applied"] = bool(target_payload)
    result["Calibration_Version"] = payload.get("version", "") if payload else ""
    return result


def assign_scores(board: pd.DataFrame) -> pd.DataFrame:
    result = board.copy()
    derived_columns = [
        "Rank", "K_Rank", "ER_Rank", "Outs_Rank", "K_Score",
        "Run_Prevention_Score", "Outs_Score", "Overall_Score",
    ]
    result = result.drop(columns=[column for column in derived_columns if column in result.columns], errors="ignore")
    result["K_Score"] = (
        percentile(result["Proj_K"]) * 0.30
        + percentile(result["Adj_K_Rate"]) * 0.15
        + percentile(result["Whiff_Pct"]) * 0.10
        + percentile(result["CSW_Pct"]) * 0.10
        + percentile(result["Opponent_K_Rate"]) * 0.12
        + result["PitchTypeScore"].clip(0, 100) * 0.08
        + percentile(result["Proj_BF"]) * 0.10
        + percentile(result["Recent_K_Rate"]) * 0.05
    )
    result["Run_Prevention_Score"] = (
        percentile(result["Proj_ER"], higher_is_better=False) * 0.30
        + percentile(result["xwOBA_Allowed"], higher_is_better=False) * 0.15
        + percentile(result["K_minus_BB"]) * 0.10
        + percentile(result["Barrel_Allowed"], higher_is_better=False) * 0.10
        + percentile(result["HardHit_Allowed"], higher_is_better=False) * 0.08
        + percentile(result["Opp_xwOBA"], higher_is_better=False) * 0.12
        + percentile(result["Park_Run_Factor"], higher_is_better=False) * 0.08
        + percentile(result["Recent_ER9"], higher_is_better=False) * 0.07
    )
    result["Outs_Score"] = (
        percentile(result["Proj_Outs"]) * 0.45
        + percentile(result["Recent_Outs"]) * 0.20
        + percentile(result["Proj_Pitches"]) * 0.10
        + percentile(result["BB_Rate"], higher_is_better=False) * 0.10
        + percentile(result["Opp_xwOBA"], higher_is_better=False) * 0.08
        + percentile(result["Starts"]) * 0.07
    )
    result["Overall_Score"] = 0.40 * result["K_Score"] + 0.35 * result["Run_Prevention_Score"] + 0.25 * result["Outs_Score"]
    result["K_Rank"] = result["Proj_K"].rank(method="min", ascending=False).astype(int)
    result["ER_Rank"] = result["Proj_ER"].rank(method="min", ascending=True).astype(int)
    result["Outs_Rank"] = result["Proj_Outs"].rank(method="min", ascending=False).astype(int)
    result = result.sort_values(["Overall_Score", "Proj_K"], ascending=False).reset_index(drop=True)
    result.insert(0, "Rank", np.arange(1, len(result) + 1))
    return result


# -----------------------------------------------------------------------------
# Backtesting, result collection and calibration
# -----------------------------------------------------------------------------

def build_projection_snapshot(
    board: pd.DataFrame,
    slate_date: object,
    lookback_days: int,
    generated_at_utc: str,
) -> pd.DataFrame:
    snapshot = board.copy()
    snapshot["SlateDate"] = str(pd.Timestamp(slate_date).date())
    snapshot["GeneratedAtUTC"] = str(generated_at_utc)
    snapshot["ModelVersion"] = MODEL_VERSION
    snapshot["LookbackDays"] = int(lookback_days)
    snapshot["Baseline_K"] = (
        pd.to_numeric(snapshot.get("Recent_K_Rate"), errors="coerce")
        * pd.to_numeric(snapshot.get("Proj_BF"), errors="coerce")
    )
    snapshot["Baseline_ER"] = (
        pd.to_numeric(snapshot.get("Recent_ER9"), errors="coerce")
        * pd.to_numeric(snapshot.get("Proj_Outs"), errors="coerce") / 27.0
    )
    snapshot["Baseline_Outs"] = pd.to_numeric(snapshot.get("Recent_Outs"), errors="coerce")
    blank_numeric = [
        "K_Line", "ER_Line", "Outs_Line", "K_Closing_Line", "ER_Closing_Line", "Outs_Closing_Line",
        "Actual_K", "Actual_ER", "Actual_Outs", "Actual_BF", "Actual_Pitches",
    ]
    for column in blank_numeric:
        if column not in snapshot.columns:
            snapshot[column] = np.nan
    if "Notes" not in snapshot.columns:
        snapshot["Notes"] = ""
    preferred_columns = [
        "SlateDate", "GameDateTimeUTC", "GeneratedAtUTC", "ModelVersion", "LookbackDays",
        "GamePK", "PitcherID", "Pitcher", "Team", "Opponent", "PitcherLocation", "Game", "Venue", "Hand",
        "Lineup_Status", "Confidence", "Confidence_Level", "Calibration_Applied", "Calibration_Version",
        "Proj_K", "Proj_ER", "Proj_Outs", "Proj_BF", "Proj_Pitches", "Outs_SD",
        "Raw_Proj_K", "Raw_Proj_ER", "Raw_Proj_Outs",
        "Baseline_K", "Baseline_ER", "Baseline_Outs",
        "K_Score", "Run_Prevention_Score", "Outs_Score", "Overall_Score",
        "Adj_K_Rate", "Pitcher_K_Rate", "Opponent_K_Rate", "Whiff_Pct", "CSW_Pct", "PitchTypeScore",
        "xwOBA_Allowed", "BB_Rate", "K_minus_BB", "Barrel_Allowed", "HardHit_Allowed", "Opp_xwOBA",
        "Recent_Outs", "Recent_K_Rate", "Recent_ER9", "Season_ER9", "Starts",
        "Park_Run_Factor", "Weather_Factor",
        "K_Line", "ER_Line", "Outs_Line", "K_Closing_Line", "ER_Closing_Line", "Outs_Closing_Line",
        "OddsSourceMode",
        "K_Line_Source", "K_Over_Odds", "K_Under_Odds", "K_Line_Updated", "K_Market_Over_Prob", "K_Model_Over_Prob", "K_Probability_Edge", "K_Projection_Edge",
        "ER_Line_Source", "ER_Over_Odds", "ER_Under_Odds", "ER_Line_Updated", "ER_Market_Over_Prob", "ER_Model_Over_Prob", "ER_Probability_Edge", "ER_Projection_Edge",
        "Outs_Line_Source", "Outs_Over_Odds", "Outs_Under_Odds", "Outs_Line_Updated", "Outs_Market_Over_Prob", "Outs_Model_Over_Prob", "Outs_Probability_Edge", "Outs_Projection_Edge",
        "Actual_K", "Actual_ER", "Actual_Outs", "Actual_BF", "Actual_Pitches", "Notes",
    ]
    existing = [column for column in preferred_columns if column in snapshot.columns]
    extras = [column for column in snapshot.columns if column not in existing and column not in {"DetailKey", "Rank", "K_Rank", "ER_Rank", "Outs_Rank"}]
    return snapshot[existing + extras].copy()


def normalize_backtest_history(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    result = history.copy()
    aliases = {
        "Date": "SlateDate", "Pitcher_ID": "PitcherID", "Game_ID": "GamePK",
        "Projected_K": "Proj_K", "Projected_ER": "Proj_ER", "Projected_Outs": "Proj_Outs",
        "ActualK": "Actual_K", "ActualER": "Actual_ER", "ActualOuts": "Actual_Outs",
    }
    result = result.rename(columns={key: value for key, value in aliases.items() if key in result.columns and value not in result.columns})
    for column in ["SlateDate", "GameDateTimeUTC", "GeneratedAtUTC"]:
        if column not in result.columns:
            result[column] = pd.NaT if column != "SlateDate" else ""
    result["SlateDate"] = pd.to_datetime(result["SlateDate"], errors="coerce").dt.date
    result["GameDateTimeUTC"] = pd.to_datetime(result["GameDateTimeUTC"], utc=True, errors="coerce")
    result["GeneratedAtUTC"] = pd.to_datetime(result["GeneratedAtUTC"], utc=True, errors="coerce")
    numeric_columns = {
        "GamePK", "PitcherID", "LookbackDays", "Proj_K", "Proj_ER", "Proj_Outs", "Proj_BF", "Proj_Pitches",
        "Outs_SD", "Baseline_K", "Baseline_ER", "Baseline_Outs", "K_Score", "Run_Prevention_Score",
        "Outs_Score", "Overall_Score", "K_Line", "ER_Line", "Outs_Line", "K_Closing_Line", "ER_Closing_Line",
        "Outs_Closing_Line", "K_Over_Odds", "K_Under_Odds", "ER_Over_Odds", "ER_Under_Odds",
        "Outs_Over_Odds", "Outs_Under_Odds", "K_Market_Over_Prob", "K_Model_Over_Prob",
        "K_Probability_Edge", "K_Projection_Edge", "ER_Market_Over_Prob", "ER_Model_Over_Prob",
        "ER_Probability_Edge", "ER_Projection_Edge", "Outs_Market_Over_Prob", "Outs_Model_Over_Prob",
        "Outs_Probability_Edge", "Outs_Projection_Edge", "Actual_K", "Actual_ER", "Actual_Outs", "Actual_BF", "Actual_Pitches",
    }
    for column in numeric_columns:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if "ModelVersion" not in result.columns:
        result["ModelVersion"] = "unknown"
    if "Lineup_Status" not in result.columns:
        result["Lineup_Status"] = "Unknown"
    if "Confidence_Level" not in result.columns:
        result["Confidence_Level"] = "Unknown"
    if "Pitcher" not in result.columns:
        result["Pitcher"] = ""
    result["SnapshotAfterStart"] = (
        result["GeneratedAtUTC"].notna()
        & result["GameDateTimeUTC"].notna()
        & result["GeneratedAtUTC"].gt(result["GameDateTimeUTC"])
    )
    return result


def combine_history_uploads(uploaded_files: list[object]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for uploaded in uploaded_files or []:
        try:
            content = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
            frame = pd.read_csv(StringIO(content.decode("utf-8-sig")))
            if not frame.empty:
                frame["SourceFile"] = getattr(uploaded, "name", "uploaded.csv")
                frames.append(frame)
        except Exception as exc:
            errors.append(f"{getattr(uploaded, 'name', 'file')}: {type(exc).__name__}: {exc}")
    if not frames:
        return pd.DataFrame(), errors
    return normalize_backtest_history(pd.concat(frames, ignore_index=True, sort=False)), errors


def append_snapshot_to_history(history: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """Add or replace one generated slate snapshot in the in-session master history."""
    current = normalize_backtest_history(history)
    addition = snapshot.copy() if snapshot is not None else pd.DataFrame()
    if addition.empty:
        return current
    if "SourceFile" not in addition.columns:
        addition["SourceFile"] = "session_current_snapshot"
    addition = normalize_backtest_history(addition)
    combined = normalize_backtest_history(
        pd.concat([current, addition], ignore_index=True, sort=False)
    )

    # Re-clicking the button should replace the same generated snapshot rather
    # than inflating the master with duplicate pitcher rows.
    game_key = combined["GamePK"].where(
        combined["GamePK"].notna(), combined["SlateDate"].astype(str)
    ).astype(str)
    pitcher_key = combined["PitcherID"].where(
        combined["PitcherID"].notna(),
        combined["Pitcher"].map(normalize_lookup_text),
    ).astype(str)
    generated_key = combined["GeneratedAtUTC"].astype(str)
    combined["_SessionSnapshotKey"] = game_key + "|" + pitcher_key + "|" + generated_key
    combined = combined.drop_duplicates("_SessionSnapshotKey", keep="last")
    return combined.drop(columns="_SessionSnapshotKey").reset_index(drop=True)


def deduplicate_history(history: pd.DataFrame, latest_only: bool = True, exclude_after_start: bool = True) -> pd.DataFrame:
    result = normalize_backtest_history(history)
    if result.empty:
        return result
    if exclude_after_start and "SnapshotAfterStart" in result.columns:
        result = result[~result["SnapshotAfterStart"].fillna(False)].copy()
    if latest_only:
        result = result.sort_values("GeneratedAtUTC", na_position="first")
        game_key = result["GamePK"].where(result["GamePK"].notna(), result["SlateDate"].astype(str))
        result["_GameKey"] = game_key.astype(str)
        result = result.drop_duplicates(["_GameKey", "PitcherID"], keep="last").drop(columns="_GameKey")
    return result.reset_index(drop=True)



@st.cache_data(ttl=300, max_entries=160, show_spinner=False)
def fetch_completed_pitcher_results(game_pk: int) -> dict:
    """Fetch official pitcher box-score results for one MLB game.

    This intentionally uses the game-specific live feed instead of the season
    game-log cache. The projection board can cache a pitcher's season game log
    for several hours; reusing that cache after a game ends can make the result
    grader miss the newly completed start.
    """
    game_pk = int(game_pk)
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    request = Request(
        url,
        headers={"User-Agent": "MLB-Pitcher-Backtest/1.1", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return {
            "ok": False,
            "final": False,
            "data": pd.DataFrame(),
            "status": "Fetch error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    game_data = payload.get("gameData", {}) or {}
    status = game_data.get("status", {}) or {}
    abstract_state = str(status.get("abstractGameState", "") or "")
    detailed_state = str(status.get("detailedState", "") or "")
    coded_state = str(status.get("codedGameState", "") or "")
    status_text = detailed_state or abstract_state or coded_state or "Unknown"
    final = (
        abstract_state.lower() == "final"
        or coded_state.upper() == "F"
        or any(
            token in detailed_state.lower()
            for token in ["final", "game over", "completed early"]
        )
    )
    if not final:
        return {
            "ok": True,
            "final": False,
            "data": pd.DataFrame(),
            "status": status_text,
            "error": None,
        }

    rows: list[dict] = []
    teams = (((payload.get("liveData", {}) or {}).get("boxscore", {}) or {}).get("teams", {}) or {})
    for side in ["away", "home"]:
        team_box = teams.get(side, {}) or {}
        players = team_box.get("players", {}) or {}
        ordered_pitchers = team_box.get("pitchers", []) or []
        ordered_lookup: dict[int, int] = {}
        for order_index, raw_pitcher_id in enumerate(ordered_pitchers):
            numeric_id = pd.to_numeric(pd.Series([raw_pitcher_id]), errors="coerce").iloc[0]
            if not pd.isna(numeric_id):
                ordered_lookup[int(numeric_id)] = int(order_index)

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
            player_id_int = int(numeric_id)

            pitching = ((record.get("stats", {}) or {}).get("pitching", {}) or {})
            if not pitching:
                continue
            ip = pitching.get("inningsPitched")
            outs = innings_to_outs(ip)
            bf = pd.to_numeric(pd.Series([pitching.get("battersFaced")]), errors="coerce").iloc[0]
            pitches = pd.to_numeric(pd.Series([pitching.get("numberOfPitches")]), errors="coerce").iloc[0]
            # Ignore rostered pitchers who did not appear.
            if (pd.isna(outs) or float(outs) <= 0) and (pd.isna(bf) or float(bf) <= 0) and (pd.isna(pitches) or float(pitches) <= 0):
                continue

            games_started = pd.to_numeric(
                pd.Series([pitching.get("gamesStarted")]), errors="coerce"
            ).iloc[0]
            rows.append({
                "PitcherID": player_id_int,
                "Pitcher_Official": str(person.get("fullName") or ""),
                "Actual_K": pd.to_numeric(pd.Series([pitching.get("strikeOuts")]), errors="coerce").fillna(0).iloc[0],
                "Actual_ER": pd.to_numeric(pd.Series([pitching.get("earnedRuns")]), errors="coerce").fillna(0).iloc[0],
                "Actual_Outs": float(outs) if not pd.isna(outs) else np.nan,
                "Actual_BF": float(bf) if not pd.isna(bf) else np.nan,
                "Actual_Pitches": float(pitches) if not pd.isna(pitches) else np.nan,
                "GamesStarted": float(games_started) if not pd.isna(games_started) else np.nan,
                "PitchingOrder": ordered_lookup.get(player_id_int, 999),
                "Side": side,
            })

    data = pd.DataFrame(rows)
    if not data.empty:
        # A player should appear only once, but keep the fullest pitching line if
        # MLB's response ever contains a duplicate record.
        data["_Completeness"] = data[
            ["Actual_K", "Actual_ER", "Actual_Outs", "Actual_BF", "Actual_Pitches"]
        ].notna().sum(axis=1)
        data = (
            data.sort_values(["PitcherID", "_Completeness", "Actual_Pitches"], ascending=[True, False, False])
            .drop_duplicates("PitcherID", keep="first")
            .drop(columns="_Completeness")
            .reset_index(drop=True)
        )
    return {
        "ok": True,
        "final": True,
        "data": data,
        "status": status_text or "Final",
        "error": None,
    }


def fill_actual_results(history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fill official K, ER, outs, BF and pitch counts for completed games.

    GamePK is the primary key because it handles doubleheaders and avoids stale
    season game-log caches. A date-based game-log fallback remains for older
    imported rows that do not contain a GamePK.
    """
    result = normalize_backtest_history(history)
    summary = {
        "matched": 0,
        "not_final": 0,
        "unmatched": 0,
        "fallback_matched": 0,
        "missing_game_pk": 0,
        "errors": [],
    }
    if result.empty or "PitcherID" not in result.columns:
        return result, summary

    # Primary path: fetch one official game feed per GamePK and match by MLB pitcher ID.
    game_ids = (
        pd.to_numeric(result.get("GamePK"), errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    rows_without_game_pk = int(pd.to_numeric(result.get("GamePK"), errors="coerce").isna().sum())
    summary["missing_game_pk"] = rows_without_game_pk

    total_work = max(len(game_ids) + (1 if rows_without_game_pk else 0), 1)
    completed_work = 0
    progress = st.progress(0.0, text="Fetching official pitcher results...")

    for game_pk in game_ids:
        try:
            game_result = fetch_completed_pitcher_results(int(game_pk))
            game_mask = pd.to_numeric(result["GamePK"], errors="coerce").eq(int(game_pk))
            if not game_result.get("ok"):
                summary["errors"].append(f"Game {game_pk}: {game_result.get('error')}")
            elif not game_result.get("final"):
                summary["not_final"] += 1
            else:
                official = game_result.get("data", pd.DataFrame())
                if official is None or official.empty:
                    unmatched_rows = int(game_mask.sum())
                    summary["unmatched"] += unmatched_rows
                    summary["errors"].append(
                        f"Game {game_pk} is final, but MLB returned no pitcher box-score rows."
                    )
                else:
                    official = official.set_index("PitcherID")
                    for row_index in result.index[game_mask]:
                        pitcher_id = result.at[row_index, "PitcherID"]
                        numeric_pitcher = pd.to_numeric(pd.Series([pitcher_id]), errors="coerce").iloc[0]
                        if pd.isna(numeric_pitcher) or int(numeric_pitcher) not in official.index:
                            summary["unmatched"] += 1
                            continue
                        actual = official.loc[int(numeric_pitcher)]
                        if isinstance(actual, pd.DataFrame):
                            actual = actual.iloc[0]
                        for column in ["Actual_K", "Actual_ER", "Actual_Outs", "Actual_BF", "Actual_Pitches"]:
                            result.at[row_index, column] = pd.to_numeric(
                                pd.Series([actual.get(column)]), errors="coerce"
                            ).iloc[0]
                        summary["matched"] += 1
        except Exception as exc:
            summary["errors"].append(f"Game {game_pk}: {type(exc).__name__}: {exc}")
        completed_work += 1
        progress.progress(
            min(completed_work / total_work, 1.0),
            text=f"Checked {completed_work} of {total_work} result groups",
        )

    # Compatibility fallback for old CSV rows that lack GamePK.
    missing_pk_mask = pd.to_numeric(result.get("GamePK"), errors="coerce").isna()
    fallback_candidates = result[
        missing_pk_mask & result["PitcherID"].notna() & result["SlateDate"].notna()
    ].copy()
    if not fallback_candidates.empty:
        pairs = fallback_candidates[["PitcherID", "SlateDate"]].drop_duplicates()
        for pair in pairs.itertuples(index=False):
            pitcher_id = int(pair.PitcherID)
            slate_day = pair.SlateDate
            try:
                # Clear only the season game-log cache before grading old rows so
                # a pregame cached log cannot hide a newly completed appearance.
                fetch_pitcher_game_log.clear()
                logs, error = fetch_pitcher_game_log(pitcher_id, int(pd.Timestamp(slate_day).year))
                if error and (logs is None or logs.empty):
                    summary["errors"].append(f"{pitcher_id} {slate_day}: {error}")
                    continue
                game_rows = logs[
                    pd.to_datetime(logs["Date"], errors="coerce").dt.date.eq(slate_day)
                ].copy()
                mask = missing_pk_mask & result["PitcherID"].eq(pitcher_id) & result["SlateDate"].eq(slate_day)
                if game_rows.empty:
                    summary["unmatched"] += int(mask.sum())
                    continue
                actual = game_rows.sort_values(["Pitches", "Outs"], ascending=False).iloc[0]
                for column, source_column in {
                    "Actual_K": "K",
                    "Actual_ER": "ER",
                    "Actual_Outs": "Outs",
                    "Actual_BF": "BF",
                    "Actual_Pitches": "Pitches",
                }.items():
                    result.loc[mask, column] = pd.to_numeric(
                        pd.Series([actual.get(source_column)]), errors="coerce"
                    ).iloc[0]
                matched_rows = int(mask.sum())
                summary["matched"] += matched_rows
                summary["fallback_matched"] += matched_rows
            except Exception as exc:
                summary["errors"].append(
                    f"Fallback {pitcher_id} {slate_day}: {type(exc).__name__}: {exc}"
                )
        completed_work += 1
        progress.progress(
            min(completed_work / total_work, 1.0),
            text=f"Checked {completed_work} of {total_work} result groups",
        )

    progress.empty()
    return normalize_backtest_history(result), summary


def projection_metrics(history: pd.DataFrame, projection_col: str, actual_col: str, baseline_col: str) -> dict:
    columns = [column for column in [projection_col, actual_col, baseline_col] if column in history.columns]
    clean = history[columns].copy()
    clean[projection_col] = pd.to_numeric(clean[projection_col], errors="coerce")
    clean[actual_col] = pd.to_numeric(clean[actual_col], errors="coerce")
    clean = clean.dropna(subset=[projection_col, actual_col])
    if clean.empty:
        return {"N": 0, "MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "Correlation": np.nan, "Baseline_MAE": np.nan}
    errors = clean[projection_col] - clean[actual_col]
    baseline_mae = np.nan
    if baseline_col in clean.columns:
        baseline = pd.to_numeric(clean[baseline_col], errors="coerce")
        valid = baseline.notna()
        if valid.any():
            baseline_mae = float((baseline[valid] - clean.loc[valid, actual_col]).abs().mean())
    correlation = clean[[projection_col, actual_col]].corr().iloc[0, 1] if len(clean) >= 3 else np.nan
    return {
        "N": int(len(clean)),
        "MAE": float(errors.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "Bias": float(errors.mean()),
        "Correlation": float(correlation) if not pd.isna(correlation) else np.nan,
        "Baseline_MAE": baseline_mae,
    }


def fit_linear_calibration(history: pd.DataFrame, projection_col: str, actual_col: str, min_rows: int = 30) -> dict:
    clean = history[[projection_col, actual_col, "SlateDate"]].copy()
    clean[projection_col] = pd.to_numeric(clean[projection_col], errors="coerce")
    clean[actual_col] = pd.to_numeric(clean[actual_col], errors="coerce")
    clean["SlateDate"] = pd.to_datetime(clean["SlateDate"], errors="coerce")
    clean = clean.dropna(subset=[projection_col, actual_col]).sort_values("SlateDate")
    if len(clean) < min_rows or clean[projection_col].nunique() < 2:
        return {
            "ok": False, "n": int(len(clean)),
            "message": f"Need at least {min_rows} completed starts with varied projections.",
            "slope": np.nan, "intercept": np.nan, "raw_mae": np.nan,
            "calibrated_mae": np.nan, "holdout_n": 0,
            "holdout_raw_mae": np.nan, "holdout_calibrated_mae": np.nan,
        }
    slope, intercept = np.polyfit(clean[projection_col].to_numpy(float), clean[actual_col].to_numpy(float), 1)
    slope = float(np.clip(slope, 0.25, 1.75))
    intercept = float(intercept)
    calibrated = intercept + slope * clean[projection_col]
    raw_mae = float((clean[projection_col] - clean[actual_col]).abs().mean())
    calibrated_mae = float((calibrated - clean[actual_col]).abs().mean())

    split_index = max(int(len(clean) * 0.70), min_rows // 2)
    train = clean.iloc[:split_index]
    test = clean.iloc[split_index:]
    holdout_raw_mae = np.nan
    holdout_calibrated_mae = np.nan
    if len(train) >= 20 and len(test) >= 10 and train[projection_col].nunique() >= 2:
        holdout_slope, holdout_intercept = np.polyfit(train[projection_col].to_numpy(float), train[actual_col].to_numpy(float), 1)
        holdout_slope = float(np.clip(holdout_slope, 0.25, 1.75))
        holdout_prediction = float(holdout_intercept) + holdout_slope * test[projection_col]
        holdout_raw_mae = float((test[projection_col] - test[actual_col]).abs().mean())
        holdout_calibrated_mae = float((holdout_prediction - test[actual_col]).abs().mean())
    return {
        "ok": True, "n": int(len(clean)), "slope": slope, "intercept": intercept,
        "raw_mae": raw_mae, "calibrated_mae": calibrated_mae,
        "holdout_n": int(len(test)), "holdout_raw_mae": holdout_raw_mae,
        "holdout_calibrated_mae": holdout_calibrated_mae,
    }


def make_calibration_bundle(history: pd.DataFrame, min_rows: int = 30) -> tuple[dict, pd.DataFrame]:
    payload = {
        "version": MODEL_VERSION,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "method": "linear_actual_equals_intercept_plus_slope_times_projection",
        "targets": {},
    }
    rows: list[dict] = []
    for target_name, config in BACKTEST_TARGETS.items():
        fit = fit_linear_calibration(history, config["projection"], config["actual"], min_rows=min_rows)
        row = {"Target": target_name, **fit}
        rows.append(row)
        if fit.get("ok"):
            payload["targets"][target_name] = {
                "projection_column": config["projection"],
                "actual_column": config["actual"],
                "intercept": fit["intercept"],
                "slope": fit["slope"],
                "n": fit["n"],
            }
    return payload, pd.DataFrame(rows)


def poisson_prop_probabilities(mean: float, line: float) -> tuple[float, float, float]:
    mean = max(float(mean), 0.0001)
    line = float(line)
    floor_line = math.floor(line)
    is_integer = abs(line - round(line)) < 1e-9
    over = 1.0 - poisson_cdf(floor_line, mean)
    if is_integer:
        under = poisson_cdf(floor_line - 1, mean) if floor_line >= 1 else 0.0
        push = max(1.0 - over - under, 0.0)
    else:
        under = poisson_cdf(floor_line, mean)
        push = 0.0
    return float(np.clip(over, 0, 1)), float(np.clip(under, 0, 1)), float(np.clip(push, 0, 1))


def prop_calibration_records(history: pd.DataFrame, target_name: str) -> pd.DataFrame:
    config = BACKTEST_TARGETS[target_name]
    projection_col, actual_col, line_col = config["projection"], config["actual"], config["line"]
    required = [projection_col, actual_col, line_col]
    if any(column not in history.columns for column in required):
        return pd.DataFrame()
    clean = history.copy()
    for column in required:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=required)
    if clean.empty:
        return clean
    records: list[dict] = []
    for row in clean.itertuples(index=False):
        projection = float(getattr(row, projection_col))
        actual = float(getattr(row, actual_col))
        line = float(getattr(row, line_col))
        if target_name in {"Strikeouts", "Earned runs"}:
            over, under, push_probability = poisson_prop_probabilities(projection, line)
        else:
            sd = pd.to_numeric(pd.Series([getattr(row, "Outs_SD", 2.5)]), errors="coerce").fillna(2.5).iloc[0]
            over = float(np.clip(1.0 - normal_cdf(line, projection, float(sd)), 0, 1))
            under = 1.0 - over
            push_probability = 0.0
        if actual == line:
            outcome = np.nan
        elif over >= under:
            outcome = float(actual > line)
        else:
            outcome = float(actual < line)
        no_push_total = max(over + under, 1e-9)
        preferred_probability = max(over, under) / no_push_total
        records.append({
            "Target": target_name,
            "Preferred_Side": "Over" if over >= under else "Under",
            "Preferred_Probability": preferred_probability,
            "Outcome": outcome,
            "Projection": projection,
            "Line": line,
            "Actual": actual,
            "Projection_Edge": projection - line,
            "Push_Probability": push_probability,
        })
    return pd.DataFrame(records).dropna(subset=["Outcome"])


def probability_calibration_table(records: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if records is None or records.empty:
        return pd.DataFrame(), np.nan
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.001]
    labels = ["50–54%", "55–59%", "60–64%", "65–69%", "70–74%", "75%+"]
    frame = records.copy()
    frame["Probability_Bucket"] = pd.cut(frame["Preferred_Probability"], bins=bins, labels=labels, right=False, include_lowest=True)
    grouped = frame.groupby("Probability_Bucket", observed=False).agg(
        Sample=("Outcome", "size"),
        Average_Probability=("Preferred_Probability", "mean"),
        Actual_Win_Rate=("Outcome", "mean"),
    ).reset_index()
    grouped["Calibration_Gap"] = grouped["Actual_Win_Rate"] - grouped["Average_Probability"]
    brier = float(np.mean(np.square(frame["Preferred_Probability"] - frame["Outcome"])))
    return grouped, brier


def score_bucket_table(history: pd.DataFrame, target_name: str) -> pd.DataFrame:
    config = BACKTEST_TARGETS[target_name]
    score_col, projection_col, actual_col, line_col = config["score"], config["projection"], config["actual"], config["line"]
    required = [score_col, projection_col, actual_col]
    if any(column not in history.columns for column in required):
        return pd.DataFrame()
    frame = history.copy()
    for column in required + [line_col]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required)
    if frame.empty:
        return frame
    bins = [-0.001, 30, 45, 55, 70, 85, 100.001]
    labels = ["0–29", "30–44", "45–54", "55–69", "70–84", "85–100"]
    frame["Score_Bucket"] = pd.cut(frame[score_col], bins=bins, labels=labels, include_lowest=True, right=False)
    if line_col in frame.columns:
        valid_line = frame[line_col].notna()
        frame["Over_Result"] = np.where(valid_line & frame[actual_col].ne(frame[line_col]), frame[actual_col].gt(frame[line_col]).astype(float), np.nan)
    else:
        frame["Over_Result"] = np.nan
    grouped = frame.groupby("Score_Bucket", observed=False).agg(
        Sample=(actual_col, "size"),
        Average_Projection=(projection_col, "mean"),
        Average_Actual=(actual_col, "mean"),
        Average_Score=(score_col, "mean"),
        Over_Rate=("Over_Result", "mean"),
    ).reset_index()
    return grouped


def grouped_accuracy(history: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if group_column not in history.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for group_value, group in history.groupby(group_column, dropna=False):
        for target_name, config in BACKTEST_TARGETS.items():
            metrics = projection_metrics(group, config["projection"], config["actual"], config["baseline"])
            if metrics["N"]:
                rows.append({"Group": str(group_value), "Target": target_name, **metrics})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#0f172a; --muted:#64748b; }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {
            background:#ffffff !important; color:var(--ink) !important;
        }
        [data-testid="stHeader"] { background:rgba(255,255,255,.96) !important; }
        .block-container { max-width:1800px; padding-top:1rem; padding-bottom:2.5rem; }
        [data-testid="stSidebar"] { background:linear-gradient(180deg,#071b33,#0f3b66 58%,#164e63); }
        [data-testid="stSidebar"] * { color:#f8fafc; }
        [data-testid="stSidebar"] input { color:#0f172a !important; }
        .app-kicker { display:inline-flex; padding:.35rem .75rem; border-radius:999px; color:white;
            background:linear-gradient(90deg,#0ea5e9,#2563eb,#7c3aed); font-size:.72rem; font-weight:900;
            letter-spacing:.11em; text-transform:uppercase; box-shadow:0 8px 20px rgba(37,99,235,.22); }
        .hero { border-radius:24px; padding:1.1rem 1.25rem; margin:.7rem 0 1rem;
            background:linear-gradient(115deg,#082f49,#1d4ed8,#6d28d9); color:white;
            box-shadow:0 18px 38px rgba(15,23,42,.18); }
        .hero h2 { color:white; margin:0; }
        .hero p { color:#dbeafe; margin:.25rem 0 0; }
        div[data-testid="stMetric"] { background:linear-gradient(145deg,#ffffff,#eff6ff); border:1px solid #bfdbfe;
            border-radius:17px; padding:.8rem 1rem; box-shadow:0 9px 24px rgba(15,23,42,.07); }
        [data-testid="stDataFrame"] { border:1px solid #cbd5e1; border-radius:14px; overflow:hidden; }
        [data-baseweb="tab"] { background:#f8fafc; border-radius:10px 10px 0 0; }
        .leader { background:linear-gradient(145deg,#ffffff,#f0f9ff); border:1px solid #bae6fd; border-radius:17px;
            padding:.95rem 1rem; min-height:128px; box-shadow:0 10px 25px rgba(15,23,42,.07); }
        .leader .label { font-size:.72rem; font-weight:900; color:#64748b; letter-spacing:.08em; }
        .leader .name { font-size:1.1rem; font-weight:900; color:#0f172a; margin:.2rem 0; }
        .leader .value { font-size:1.55rem; font-weight:950; color:#0369a1; }
        .note { background:#eff6ff; border-left:5px solid #2563eb; border-radius:12px; padding:.75rem .9rem; color:#334155; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def score_label(value: float) -> str:
    value = float(value)
    if value >= 85:
        return "Elite"
    if value >= 70:
        return "Strong"
    if value >= 55:
        return "Above average"
    if value >= 45:
        return "Neutral"
    if value >= 30:
        return "Weak"
    return "Poor"


def render_leader(row: pd.Series, label: str, value_text: str, sub_text: str) -> None:
    st.markdown(
        f"""
        <div class="leader">
            <div class="label">{escape(label.upper())}</div>
            <div class="name">{escape(str(row['Pitcher']))}</div>
            <div class="value">{escape(value_text)}</div>
            <div style="color:#64748b;font-size:.82rem;">{escape(sub_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_board(frame: pd.DataFrame, score_columns: list[str], lower_better: list[str] | None = None):
    lower_better = lower_better or []
    styler = frame.style
    positive = [column for column in score_columns if column in frame.columns and column not in lower_better]
    negative = [column for column in lower_better if column in frame.columns]
    if positive:
        styler = styler.background_gradient(cmap="RdYlGn", subset=positive, axis=0)
    if negative:
        styler = styler.background_gradient(cmap="RdYlGn_r", subset=negative, axis=0)
    return styler


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

inject_css()
st.markdown('<div class="app-kicker">⚾ The Juice Report · Pitcher Lab</div>', unsafe_allow_html=True)
st.title("Advanced MLB Pitcher Dashboard")
st.caption("Projected strikeouts, earned runs, outs, opponent-lineup fit, pitch-type matchups and 0–100 category scores.")

with st.sidebar:
    st.header("Pitcher slate")
    slate_date = st.date_input("Slate date", value=date.today(), key="pitcher_slate_date")
    lookback_days = st.slider("Statcast lookback days", 21, 90, 45, 3)
    min_starts = st.slider("Minimum starts for full confidence", 1, 8, 3)
    include_tbd = st.checkbox("Show games with one probable pitcher TBD", value=False)
    calibration_upload = st.file_uploader("Optional calibration JSON", type=["json"], key="pitcher_calibration_upload")
    uploaded_calibration: dict = {}
    if calibration_upload is not None:
        try:
            uploaded_calibration = parse_calibration_payload(json.loads(calibration_upload.getvalue().decode("utf-8-sig")))
            if uploaded_calibration:
                st.success("Calibration file loaded. Rebuild the board to apply it.")
            else:
                st.warning("The calibration JSON did not contain valid K, ER or outs coefficients.")
        except Exception as exc:
            st.warning(f"Calibration file could not be read: {type(exc).__name__}: {exc}")
    session_calibration = parse_calibration_payload(st.session_state.get("pitcher_calibration", {}))
    active_calibration = uploaded_calibration or session_calibration
    st.caption("The model automatically uses posted MLB lineups and falls back to the most recent observed batting order.")
    st.divider()
    st.subheader("Automatic prop lines")
    secret_odds_key = read_streamlit_secret("THE_ODDS_API_KEY")
    if secret_odds_key:
        odds_api_key = secret_odds_key
        st.success("The Odds API key loaded from Streamlit Secrets.")
    else:
        odds_api_key = st.text_input(
            "The Odds API key (session only)",
            type="password",
            help="Recommended: save THE_ODDS_API_KEY in Streamlit Secrets. This field is only a temporary fallback.",
            key="pitcher_odds_api_key_input",
        ).strip()
        st.caption("Add THE_ODDS_API_KEY to Streamlit Secrets so the key survives a reboot.")
    odds_source_mode = st.selectbox(
        "Default line source",
        ODDS_SOURCE_OPTIONS,
        index=0,
        key="pitcher_odds_source_mode",
    )
    st.caption("Props are fetched only when you press the fetch button, preventing accidental API-credit use on every Streamlit rerun.")
    build_button = st.button("Build / refresh pitcher board", type="primary", width="stretch")
    if st.button("Clear odds cache and loaded lines", width="stretch"):
        fetch_odds_api_events.clear()
        fetch_odds_api_event_props.clear()
        st.session_state.pop("pitcher_odds_quotes", None)
        st.session_state.pop("pitcher_odds_meta", None)
        st.session_state.pop("pitcher_odds_errors", None)
        st.rerun()
    if st.button("Clear cached MLB data", width="stretch"):
        fetch_mlb_schedule.clear()
        fetch_game_lineups.clear()
        fetch_pitcher_game_log.clear()
        fetch_game_weather.clear()
        st.cache_data.clear()
        st.rerun()

schedule, schedule_error = fetch_mlb_schedule(str(slate_date))
if schedule_error:
    st.error(f"Could not load the MLB slate: {schedule_error}")
elif not schedule:
    st.info("No MLB games were returned for the selected date.")
else:
    probable_count = sum(int(game.get("away_pitcher_id") is not None) + int(game.get("home_pitcher_id") is not None) for game in schedule)
    st.markdown(
        f"""
        <div class="hero">
            <h2>{len(schedule)} games · {probable_count} probable starters</h2>
            <p>{escape(str(slate_date))} · MLB schedule, posted lineups, park context and first-pitch weather.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if build_button and schedule:
    end_date = pd.Timestamp(slate_date) - pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=int(lookback_days) - 1)
    with st.spinner(f"Loading Statcast from {start_date.date()} through {end_date.date()}..."):
        try:
            raw = load_statcast(str(start_date.date()), str(end_date.date()))
            prepared = prepare_statcast(raw)
        except Exception as exc:
            st.error(f"Statcast could not be loaded: {type(exc).__name__}: {exc}")
            prepared = pd.DataFrame()
    if not prepared.empty:
        league = league_context(prepared)
        rows: list[dict] = []
        details: dict[str, dict] = {}
        progress = st.progress(0.0, text="Building pitcher projections...")
        total_games = max(len(schedule), 1)
        for game_index, game in enumerate(schedule):
            lineups = fetch_game_lineups(game.get("game_pk"))
            weather = automatic_weather(game)
            configurations = [
                {
                    "pitcher_id": game.get("away_pitcher_id"), "pitcher_name": game.get("away_pitcher_name"),
                    "pitcher_team": game.get("away_abbr"), "opponent_team": game.get("home_abbr"),
                    "opponent_side": "home", "lineup": lineups.get("home", pd.DataFrame()),
                    "lineup_status": lineups.get("home_status", "Not posted"),
                },
                {
                    "pitcher_id": game.get("home_pitcher_id"), "pitcher_name": game.get("home_pitcher_name"),
                    "pitcher_team": game.get("home_abbr"), "opponent_team": game.get("away_abbr"),
                    "opponent_side": "away", "lineup": lineups.get("away", pd.DataFrame()),
                    "lineup_status": lineups.get("away_status", "Not posted"),
                },
            ]
            for config in configurations:
                if config["pitcher_id"] is None:
                    if include_tbd:
                        continue
                    continue
                lineup = config["lineup"]
                lineup_status = config["lineup_status"]
                if lineup is None or lineup.empty:
                    lineup = infer_recent_lineup(prepared, str(config["opponent_team"]))
                    lineup_status = "Recent fallback" if not lineup.empty else "Not posted"
                try:
                    row, detail = build_pitcher_projection(
                        prepared, league, game, int(config["pitcher_id"]), str(config["pitcher_name"]),
                        str(config["pitcher_team"]), str(config["opponent_team"]), str(config["opponent_side"]),
                        lineup, lineup_status, weather, int(pd.Timestamp(slate_date).year),
                    )
                    if row["Starts"] < min_starts:
                        row["Confidence"] *= 0.82
                        row["Confidence_Level"] = "Low" if row["Confidence"] < 45 else "Medium"
                    key = f"{row['GamePK']}:{row['PitcherID']}"
                    row["DetailKey"] = key
                    details[key] = detail
                    rows.append(row)
                except Exception as exc:
                    st.warning(f"Skipped {config['pitcher_name']}: {type(exc).__name__}: {exc}")
            progress.progress((game_index + 1) / total_games, text=f"Processed {game_index + 1} of {total_games} games")
        progress.empty()
        if rows:
            raw_board = pd.DataFrame(rows)
            calibrated_board = apply_projection_calibration(raw_board, active_calibration)
            board = assign_scores(calibrated_board)
            st.session_state["pitcher_board"] = board
            st.session_state["pitcher_details"] = details
            st.session_state["pitcher_window"] = f"{start_date.date()} to {end_date.date()}"
            st.session_state["pitcher_lookback_days"] = int(lookback_days)
            st.session_state["pitcher_generated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        else:
            st.warning("No probable pitchers could be projected from the selected slate.")

base_board = st.session_state.get("pitcher_board", pd.DataFrame())
details = st.session_state.get("pitcher_details", {})

if base_board.empty:
    st.info("Choose a slate date and press **Build / refresh pitcher board**.")
    st.stop()

loaded_quotes = st.session_state.get("pitcher_odds_quotes", pd.DataFrame())
board = apply_odds_quotes_to_board(base_board, loaded_quotes, odds_source_mode) if loaded_quotes is not None and not loaded_quotes.empty else base_board.copy()

with st.expander("Automatic sportsbook / PrizePicks pitcher lines", expanded=loaded_quotes is None or loaded_quotes.empty):
    unique_games = board[["GamePK", "Game"]].drop_duplicates().copy()
    unique_games["GamePK"] = pd.to_numeric(unique_games["GamePK"], errors="coerce")
    unique_games = unique_games.dropna(subset=["GamePK"])
    game_label_to_pk = {
        str(row.Game): int(row.GamePK)
        for row in unique_games.itertuples(index=False)
    }
    game_labels = list(game_label_to_pk.keys())
    selected_odds_games = st.multiselect(
        "Games to fetch",
        game_labels,
        default=game_labels,
        key=f"pitcher_odds_games_{slate_date}",
        help="Each game request asks for strikeouts, earned runs and outs. Selecting fewer games preserves API credits.",
    )
    estimated_credits = 3 * len(selected_odds_games)
    st.caption(
        f"Maximum expected cost: about {estimated_credits} credits ({len(selected_odds_games)} games × 3 markets). "
        "The actual cost can be lower when a requested market is unavailable. Responses are cached for five minutes."
    )
    fetch_col, clear_col = st.columns([2, 1])
    with fetch_col:
        fetch_odds_button = st.button("Fetch / refresh selected prop lines", type="primary", width="stretch")
    with clear_col:
        clear_loaded_button = st.button("Clear loaded lines", width="stretch", key="clear_loaded_pitcher_lines")
    if clear_loaded_button:
        st.session_state.pop("pitcher_odds_quotes", None)
        st.session_state.pop("pitcher_odds_meta", None)
        st.session_state.pop("pitcher_odds_errors", None)
        st.rerun()
    if fetch_odds_button:
        if not odds_api_key:
            st.error("Add THE_ODDS_API_KEY in Streamlit Secrets or enter the key in the sidebar.")
        elif not selected_odds_games:
            st.warning("Select at least one game.")
        else:
            selected_game_pks = [game_label_to_pk[label] for label in selected_odds_games]
            with st.spinner("Matching MLB events and loading pitcher props..."):
                new_quotes, odds_meta, odds_errors = fetch_pitcher_prop_quotes(odds_api_key, schedule, selected_game_pks)
            combined_quotes = combine_odds_quote_frames(loaded_quotes, new_quotes, slate_date)
            st.session_state["pitcher_odds_quotes"] = combined_quotes
            st.session_state["pitcher_odds_meta"] = odds_meta
            st.session_state["pitcher_odds_errors"] = odds_errors
            st.session_state["pitcher_odds_last_fetch"] = pd.Timestamp.now(tz="UTC").isoformat()
            st.rerun()

    odds_meta = st.session_state.get("pitcher_odds_meta", {})
    odds_errors = st.session_state.get("pitcher_odds_errors", [])
    if odds_meta:
        meta_cols = st.columns(4)
        meta_cols[0].metric("Events queried", odds_meta.get("events_queried", "—"))
        meta_cols[1].metric("Credits this fetch", odds_meta.get("estimated_credits_used_this_fetch", "—"))
        meta_cols[2].metric("Credits remaining", odds_meta.get("requests_remaining", "—"))
        meta_cols[3].metric("Current source", odds_source_mode)
    if odds_errors:
        with st.expander(f"Unavailable or unmatched lines ({len(odds_errors)})"):
            for error in odds_errors:
                st.write(f"• {error}")
    if loaded_quotes is not None and not loaded_quotes.empty:
        line_columns = [
            "Pitcher", "Game", "K_Line", "K_Over_Odds", "K_Under_Odds", "K_Line_Source",
            "ER_Line", "ER_Over_Odds", "ER_Under_Odds", "ER_Line_Source",
            "Outs_Line", "Outs_Over_Odds", "Outs_Under_Odds", "Outs_Line_Source",
        ]
        line_columns = [column for column in line_columns if column in board.columns]
        line_preview = board[line_columns].copy()
        for column in ["K_Over_Odds", "K_Under_Odds", "ER_Over_Odds", "ER_Under_Odds", "Outs_Over_Odds", "Outs_Under_Odds"]:
            if column in line_preview.columns:
                line_preview[column] = line_preview[column].map(format_american_odds)
        st.dataframe(line_preview, width="stretch", hide_index=True, height=min(480, 38 * (len(line_preview) + 1)))
        if odds_source_mode == "PrizePicks":
            st.caption("PrizePicks odds/multipliers returned by the provider are indicative. The projection line is the main field used by this model.")

window_text = st.session_state.get("pitcher_window", "")
calibration_note = " · Saved calibration applied" if bool(board.get("Calibration_Applied", pd.Series(False, index=board.index)).any()) else ""
st.caption(f"Statcast sample: {window_text}{calibration_note}. Scores are relative comparisons within the current slate; projections are estimates, not guarantees.")

leaders = st.columns(4)
with leaders[0]:
    render_leader(board.sort_values("Proj_K", ascending=False).iloc[0], "Top K projection", f"{board['Proj_K'].max():.1f} K", "Ranked by projected strikeouts")
with leaders[1]:
    er_row = board.sort_values("Proj_ER").iloc[0]
    render_leader(er_row, "Best run prevention", f"{er_row['Proj_ER']:.2f} ER", f"Score {er_row['Run_Prevention_Score']:.0f}")
with leaders[2]:
    outs_row = board.sort_values("Proj_Outs", ascending=False).iloc[0]
    render_leader(outs_row, "Deepest outing", f"{outs_row['Proj_Outs']:.1f} outs", f"{outs_row['Proj_Innings']:.1f} projected innings")
with leaders[3]:
    overall_row = board.iloc[0]
    render_leader(overall_row, "Best overall", f"{overall_row['Overall_Score']:.0f}", score_label(overall_row["Overall_Score"]))

(
    board_tab, k_tab, er_tab, outs_tab, lineup_tab, pitch_tab, logs_tab, lines_tab, backtest_tab, notes_tab
) = st.tabs([
    "Pitcher board", "Strikeouts", "Earned runs", "Outs / innings", "Opponent lineup",
    "Pitch-type matchup", "Recent starts", "Line comparison", "Backtest & calibration", "Model notes",
])

with board_tab:
    columns = [
        "Rank", "Pitcher", "Team", "Opponent", "Game", "Proj_K", "Proj_ER", "Proj_Outs",
        "K_Score", "Run_Prevention_Score", "Outs_Score", "Overall_Score", "Confidence_Level", "Lineup_Status",
    ]
    automatic_columns = [
        "K_Line", "K_Projection_Edge", "ER_Line", "ER_Projection_Edge",
        "Outs_Line", "Outs_Projection_Edge", "K_Line_Source",
    ]
    for automatic_column in automatic_columns:
        if automatic_column in board.columns:
            values = board[automatic_column]
            if values.astype(str).str.strip().ne("").any() and not (pd.api.types.is_numeric_dtype(values) and values.isna().all()):
                columns.append(automatic_column)
    display = board[columns].copy().rename(
        columns={
            "Proj_K": "Proj K", "Proj_ER": "Proj ER", "Proj_Outs": "Proj Outs",
            "K_Score": "K Score", "Run_Prevention_Score": "Run Prevention",
            "Outs_Score": "Outs Score", "Overall_Score": "Overall", "Confidence_Level": "Confidence",
            "Lineup_Status": "Lineup", "K_Line": "K Line", "K_Projection_Edge": "K Edge",
            "ER_Line": "ER Line", "ER_Projection_Edge": "ER Edge", "Outs_Line": "Outs Line",
            "Outs_Projection_Edge": "Outs Edge", "K_Line_Source": "Line Source",
        }
    )
    styled = style_board(display, ["Proj K", "K Score", "Run Prevention", "Outs Score", "Overall"], ["Proj ER"])
    formatters = {"Proj K": "{:.2f}", "Proj ER": "{:.2f}", "Proj Outs": "{:.1f}", "K Score": "{:.0f}", "Run Prevention": "{:.0f}", "Outs Score": "{:.0f}", "Overall": "{:.0f}"}
    for column in ["K Line", "K Edge", "ER Line", "ER Edge", "Outs Line", "Outs Edge"]:
        if column in display.columns:
            formatters[column] = "{:.2f}"
    styled = styled.format(formatters)
    st.dataframe(styled, width="stretch", hide_index=True, height=650)

with k_tab:
    columns = ["K_Rank", "Pitcher", "Opponent", "Proj_K", "Proj_BF", "Adj_K_Rate", "Pitcher_K_Rate", "Opponent_K_Rate", "Whiff_Pct", "CSW_Pct", "PitchTypeScore", "K_Score", "Confidence_Level"]
    display = board.sort_values("Proj_K", ascending=False)[columns].copy().rename(columns={
        "K_Rank": "Rank", "Proj_K": "Proj K", "Proj_BF": "Proj BF", "Adj_K_Rate": "Game K%",
        "Pitcher_K_Rate": "Pitcher K%", "Opponent_K_Rate": "Opponent K%", "Whiff_Pct": "Whiff%",
        "CSW_Pct": "CSW%", "PitchTypeScore": "Pitch Match", "K_Score": "K Score", "Confidence_Level": "Confidence",
    })
    styled = style_board(display, ["Proj K", "Game K%", "Pitcher K%", "Opponent K%", "Whiff%", "CSW%", "Pitch Match", "K Score"])
    styled = styled.format({"Proj K": "{:.2f}", "Proj BF": "{:.1f}", "Game K%": "{:.1%}", "Pitcher K%": "{:.1%}", "Opponent K%": "{:.1%}", "Whiff%": "{:.1%}", "CSW%": "{:.1%}", "Pitch Match": "{:.0f}", "K Score": "{:.0f}"})
    st.dataframe(styled, width="stretch", hide_index=True, height=650)

with er_tab:
    columns = ["ER_Rank", "Pitcher", "Opponent", "Proj_ER", "P_0_1_ER", "P_2_3_ER", "P_4plus_ER", "xwOBA_Allowed", "BB_Rate", "Barrel_Allowed", "Opp_xwOBA", "Park_Run_Factor", "Weather_Factor", "Run_Prevention_Score"]
    display = board.sort_values("Proj_ER")[columns].copy().rename(columns={
        "ER_Rank": "Rank", "Proj_ER": "Proj ER", "P_0_1_ER": "0–1 ER", "P_2_3_ER": "2–3 ER", "P_4plus_ER": "4+ ER",
        "xwOBA_Allowed": "xwOBA Allowed", "BB_Rate": "BB%", "Barrel_Allowed": "Barrel% Allowed",
        "Opp_xwOBA": "Opp xwOBA", "Park_Run_Factor": "Park", "Weather_Factor": "Weather", "Run_Prevention_Score": "Run Prevention",
    })
    styled = style_board(display, ["0–1 ER", "Run Prevention"], ["Proj ER", "4+ ER", "xwOBA Allowed", "BB%", "Barrel% Allowed", "Opp xwOBA", "Park", "Weather"])
    styled = styled.format({"Proj ER": "{:.2f}", "0–1 ER": "{:.1%}", "2–3 ER": "{:.1%}", "4+ ER": "{:.1%}", "xwOBA Allowed": "{:.3f}", "BB%": "{:.1%}", "Barrel% Allowed": "{:.1%}", "Opp xwOBA": "{:.3f}", "Park": "{:.0f}", "Weather": "{:.3f}", "Run Prevention": "{:.0f}"})
    st.dataframe(styled, width="stretch", hide_index=True, height=650)

with outs_tab:
    columns = ["Outs_Rank", "Pitcher", "Opponent", "Proj_Outs", "Proj_Innings", "Proj_BF", "Proj_Pitches", "Recent_Outs", "BB_Rate", "Outs_Score", "Confidence_Level"]
    display = board.sort_values("Proj_Outs", ascending=False)[columns].copy().rename(columns={
        "Outs_Rank": "Rank", "Proj_Outs": "Proj Outs", "Proj_Innings": "Proj IP", "Proj_BF": "Proj BF",
        "Proj_Pitches": "Proj Pitches", "Recent_Outs": "Recent Outs", "BB_Rate": "BB%", "Outs_Score": "Outs Score", "Confidence_Level": "Confidence",
    })
    styled = style_board(display, ["Proj Outs", "Proj IP", "Proj BF", "Proj Pitches", "Recent Outs", "Outs Score"], ["BB%"])
    styled = styled.format({"Proj Outs": "{:.1f}", "Proj IP": "{:.2f}", "Proj BF": "{:.1f}", "Proj Pitches": "{:.0f}", "Recent Outs": "{:.1f}", "BB%": "{:.1%}", "Outs Score": "{:.0f}"})
    st.dataframe(styled, width="stretch", hide_index=True, height=650)

pitcher_options = board["Pitcher"].tolist()

with lineup_tab:
    selected = st.selectbox("Pitcher", pitcher_options, key="lineup_pitcher")
    row = board[board["Pitcher"].eq(selected)].iloc[0]
    detail = details.get(row["DetailKey"], {})
    st.markdown(f"### {selected} vs {row['Opponent']} · {row['Lineup_Status']} lineup")
    lineup_frame = detail.get("lineup", pd.DataFrame())
    if lineup_frame is None or lineup_frame.empty:
        st.info("No opponent lineup profile was available.")
    else:
        display = lineup_frame.rename(columns={"LineupSpot": "Order", "EffectiveStand": "Bats", "K_Pct": "K%", "BB_Pct": "BB%", "HR_Pct": "HR%", "Brl_BBE": "Barrel%", "HH_BBE": "Hard Hit%", "Whiff_Pct": "Whiff%"})
        styled = style_board(display, ["K%", "Whiff%"], ["BB%", "HR%", "xwOBA", "Barrel%", "Hard Hit%"])
        styled = styled.format({"Order": "{:.0f}", "PA": "{:.0f}", "K%": "{:.1%}", "BB%": "{:.1%}", "HR%": "{:.1%}", "xwOBA": "{:.3f}", "Barrel%": "{:.1%}", "Hard Hit%": "{:.1%}", "Whiff%": "{:.1%}"})
        st.dataframe(styled, width="stretch", hide_index=True)

with pitch_tab:
    selected = st.selectbox("Pitcher", pitcher_options, key="pitch_match_pitcher")
    row = board[board["Pitcher"].eq(selected)].iloc[0]
    detail = details.get(row["DetailKey"], {})
    st.metric("Overall pitch-type matchup", f"{row['PitchTypeScore']:.0f}", score_label(row["PitchTypeScore"]))
    pitch_frame = detail.get("pitch_types", pd.DataFrame())
    if pitch_frame is None or pitch_frame.empty:
        st.info("No pitch-type matchup table was available.")
    else:
        styled = style_board(pitch_frame, ["Pitcher Whiff%", "Pitcher CSW%", "Opponent K%", "Opponent Whiff%", "Match Score"], ["Pitcher xwOBA", "Opponent xwOBA"])
        styled = styled.format({"Usage": "{:.1%}", "Velocity": "{:.1f}", "Pitcher Whiff%": "{:.1%}", "Pitcher CSW%": "{:.1%}", "Pitcher xwOBA": "{:.3f}", "Opponent K%": "{:.1%}", "Opponent Whiff%": "{:.1%}", "Opponent xwOBA": "{:.3f}", "Match Score": "{:.0f}"})
        st.dataframe(styled, width="stretch", hide_index=True)

with logs_tab:
    selected = st.selectbox("Pitcher", pitcher_options, key="logs_pitcher")
    row = board[board["Pitcher"].eq(selected)].iloc[0]
    detail = details.get(row["DetailKey"], {})
    logs = detail.get("game_log", pd.DataFrame())
    if logs is None or logs.empty:
        st.info("No starter game log was returned.")
        if detail.get("log_error"):
            st.code(str(detail["log_error"]))
    else:
        display = logs[[column for column in ["Date", "Opponent", "IP", "Outs", "K", "ER", "BF", "Pitches", "Hits", "BB", "HR"] if column in logs.columns]].sort_values("Date", ascending=False)
        st.dataframe(display.style.format({"Date": lambda value: value.strftime("%Y-%m-%d") if not pd.isna(value) else "", "Outs": "{:.0f}", "K": "{:.0f}", "ER": "{:.0f}", "BF": "{:.0f}", "Pitches": "{:.0f}"}), width="stretch", hide_index=True)

with lines_tab:
    selected = st.selectbox("Pitcher", pitcher_options, key="lines_pitcher")
    row = board[board["Pitcher"].eq(selected)].iloc[0]
    st.markdown(f"### {selected} · {row['Game']}")
    st.caption(
        "Automatic values come from the selected source above. You can still overwrite any line manually for comparison. "
        "A positive projection edge means the model projects above the line."
    )

    def render_prop_comparison(
        prefix: str,
        title: str,
        projection_column: str,
        minimum: float,
        maximum: float,
        default_step: float = 0.5,
    ) -> None:
        projection = float(row[projection_column])
        auto_line = pd.to_numeric(pd.Series([row.get(f"{prefix}_Line")]), errors="coerce").iloc[0]
        default_line = float(auto_line) if not pd.isna(auto_line) else float(np.clip(round(projection * 2) / 2, minimum, maximum))
        source = str(row.get(f"{prefix}_Line_Source", "") or "")
        over_odds = row.get(f"{prefix}_Over_Odds", np.nan)
        under_odds = row.get(f"{prefix}_Under_Odds", np.nan)
        line_key_value = f"{default_line:.2f}".replace(".", "_")
        prop_line = st.number_input(
            f"{title} line",
            min_value=minimum,
            max_value=maximum,
            value=float(np.clip(default_line, minimum, maximum)),
            step=default_step,
            key=f"{prefix}_manual_line_{row['PitcherID']}_{line_key_value}_{normalize_lookup_text(odds_source_mode)}",
        )
        if prefix == "Outs":
            over_probability = 1.0 - normal_cdf(prop_line, projection, max(float(row.get("Outs_SD", 2.5)), 0.75))
        else:
            over_probability = probability_over_line(projection, prop_line)
        fair_market_over = no_vig_over_probability(over_odds, under_odds)
        projection_edge = projection - prop_line
        st.metric(f"Projected {title.lower()}", f"{projection:.2f}", delta=f"{projection_edge:+.2f} vs line")
        st.metric("Model over probability", f"{over_probability:.1%}")
        st.metric("Model under probability", f"{1.0 - over_probability:.1%}")
        if source:
            st.caption(
                f"Auto source: **{source}** · Over {format_american_odds(over_odds)} · "
                f"Under {format_american_odds(under_odds)}"
            )
        else:
            st.caption("No automatic line was matched for this pitcher and market; the model-rounded line is shown.")
        if np.isfinite(fair_market_over):
            st.metric("No-vig market over", f"{fair_market_over:.1%}", delta=f"{over_probability - fair_market_over:+.1%} model edge")
        if source.lower().startswith("prizepicks"):
            st.caption("PrizePicks prices are indicative; use the line and model probability as the primary comparison.")

    c1, c2, c3 = st.columns(3)
    with c1:
        render_prop_comparison("K", "Strikeouts", "Proj_K", 0.5, 14.5)
    with c2:
        render_prop_comparison("ER", "Earned runs", "Proj_ER", 0.5, 7.5)
    with c3:
        render_prop_comparison("Outs", "Outs", "Proj_Outs", 8.5, 23.5)

    st.caption(
        "K and ER probabilities use a Poisson approximation. Outs use the pitcher's recent-start variance with a normal approximation. "
        "Use the Backtest & calibration tab to measure and recalibrate these probabilities over time."
    )

with backtest_tab:
    st.markdown("### Save the current slate before first pitch")
    st.caption(
        "Enter any available prop lines, then download the snapshot. Keep these CSV files and upload them here after games finish. "
        "The app flags snapshots generated after first pitch so they can be excluded from testing."
    )
    snapshot_generated = st.session_state.get("pitcher_generated_at", pd.Timestamp.now(tz="UTC").isoformat())
    snapshot_lookback = int(st.session_state.get("pitcher_lookback_days", lookback_days))
    full_snapshot = build_projection_snapshot(board, slate_date, snapshot_lookback, snapshot_generated)
    editor_columns = [
        "SlateDate", "Pitcher", "Team", "Opponent", "Lineup_Status", "Proj_K", "Proj_ER", "Proj_Outs",
        "K_Score", "Run_Prevention_Score", "Outs_Score", "K_Line", "ER_Line", "Outs_Line",
        "K_Line_Source", "ER_Line_Source", "Outs_Line_Source",
        "K_Closing_Line", "ER_Closing_Line", "Outs_Closing_Line", "Notes",
    ]
    editor_columns = [column for column in editor_columns if column in full_snapshot.columns]
    editable = full_snapshot[editor_columns].copy()
    disabled_columns = [column for column in editor_columns if column not in {"K_Line", "ER_Line", "Outs_Line", "K_Closing_Line", "ER_Closing_Line", "Outs_Closing_Line", "Notes"}]
    edited = st.data_editor(
        editable,
        disabled=disabled_columns,
        width="stretch",
        hide_index=True,
        key=f"pitcher_snapshot_editor_{slate_date}_{snapshot_generated}",
        column_config={
            "K_Line": st.column_config.NumberColumn("K line", min_value=0.0, step=0.5, format="%.1f"),
            "ER_Line": st.column_config.NumberColumn("ER line", min_value=0.0, step=0.5, format="%.1f"),
            "Outs_Line": st.column_config.NumberColumn("Outs line", min_value=0.0, step=0.5, format="%.1f"),
            "K_Closing_Line": st.column_config.NumberColumn("K close", min_value=0.0, step=0.5, format="%.1f"),
            "ER_Closing_Line": st.column_config.NumberColumn("ER close", min_value=0.0, step=0.5, format="%.1f"),
            "Outs_Closing_Line": st.column_config.NumberColumn("Outs close", min_value=0.0, step=0.5, format="%.1f"),
        },
    )
    export_snapshot = full_snapshot.copy()
    for column in ["K_Line", "ER_Line", "Outs_Line", "K_Closing_Line", "ER_Closing_Line", "Outs_Closing_Line", "Notes"]:
        if column in edited.columns:
            export_snapshot[column] = edited[column].to_numpy()
    snapshot_filename = f"pitcher_predictions_{pd.Timestamp(slate_date).strftime('%Y-%m-%d')}.csv"
    st.download_button(
        "Download pregame projection snapshot",
        export_snapshot.to_csv(index=False).encode("utf-8"),
        file_name=snapshot_filename,
        mime="text/csv",
        width="stretch",
    )

    st.divider()
    st.markdown("### Upload prediction history")
    uploaded_history = st.file_uploader(
        "Upload one or more pitcher prediction CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="pitcher_history_uploads",
    )
    if uploaded_history:
        upload_signature = tuple((uploaded.name, len(uploaded.getvalue())) for uploaded in uploaded_history)
        if st.session_state.get("pitcher_history_signature") != upload_signature:
            combined_history, upload_errors = combine_history_uploads(uploaded_history)
            st.session_state["pitcher_backtest_history"] = combined_history
            st.session_state["pitcher_history_signature"] = upload_signature
            st.session_state["pitcher_history_upload_errors"] = upload_errors
    history = normalize_backtest_history(st.session_state.get("pitcher_backtest_history", pd.DataFrame()))
    upload_errors = st.session_state.get("pitcher_history_upload_errors", [])
    if upload_errors:
        with st.expander("Files that could not be read"):
            for error in upload_errors:
                st.code(error)

    action_columns = st.columns(2)
    with action_columns[0]:
        if st.button(
            "Add current snapshot to session master",
            width="stretch",
            key="pitcher_add_current_snapshot",
        ):
            history = append_snapshot_to_history(history, export_snapshot)
            st.session_state["pitcher_backtest_history"] = history
            st.success(
                "Current pitcher snapshot added to the session master. "
                "Download the updated master CSV before leaving or rebooting."
            )
    with action_columns[1]:
        if st.button(
            "Fetch official completed-game results",
            type="primary",
            width="stretch",
            key="pitcher_fetch_completed_results",
            disabled=history.empty,
        ):
            history, result_summary = fill_actual_results(history)
            st.session_state["pitcher_backtest_history"] = history
            st.session_state["pitcher_result_summary"] = result_summary

    result_summary = st.session_state.get("pitcher_result_summary", {})
    if result_summary:
        st.caption(
            f"Result update: {result_summary.get('matched', 0)} rows matched · "
            f"{result_summary.get('not_final', 0)} games not final · "
            f"{result_summary.get('unmatched', 0)} unmatched/scratched pitchers"
            + (
                f" · {result_summary.get('fallback_matched', 0)} date-fallback matches"
                if result_summary.get('fallback_matched', 0)
                else ""
            )
            + (
                f" · {result_summary.get('missing_game_pk', 0)} rows missing GamePK"
                if result_summary.get('missing_game_pk', 0)
                else ""
            )
        )
        if result_summary.get("errors"):
            with st.expander("Result-fetch details"):
                for error in result_summary["errors"][:50]:
                    st.code(error)

    if history.empty:
        st.info(
            "Add the current snapshot to the session master or upload a saved snapshot/master CSV. "
            "Then fetch official results after the games finish."
        )
    else:
        option_columns = st.columns(3)
        with option_columns[0]:
            latest_only = st.checkbox("Use latest pregame snapshot per pitcher/game", value=True)
        with option_columns[1]:
            exclude_after_start = st.checkbox("Exclude snapshots created after first pitch", value=True)
        with option_columns[2]:
            minimum_calibration_rows = st.number_input("Minimum starts for calibration", min_value=20, max_value=250, value=30, step=10)

        analysis_history = deduplicate_history(history, latest_only=latest_only, exclude_after_start=exclude_after_start)
        completed_any = analysis_history[["Actual_K", "Actual_ER", "Actual_Outs"]].notna().any(axis=1)
        completed_history = analysis_history[completed_any].copy()

        counts = st.columns(4)
        counts[0].metric("Master rows", f"{len(history):,}")
        counts[1].metric("Analysis rows", f"{len(analysis_history):,}")
        counts[2].metric("Rows with results", f"{len(completed_history):,}")
        counts[3].metric("After-start snapshots", f"{int(history['SnapshotAfterStart'].fillna(False).sum()):,}")

        st.download_button(
            "Download updated master history",
            normalize_backtest_history(history).to_csv(index=False).encode("utf-8"),
            file_name="pitcher_backtest_master.csv",
            mime="text/csv",
            width="stretch",
        )

        if completed_history.empty:
            st.warning("No completed results are available yet. Use the result-fetch button after the games are final.")
        else:
            st.markdown("### Projection accuracy")
            metric_rows: list[dict] = []
            for target_name, config in BACKTEST_TARGETS.items():
                metrics = projection_metrics(completed_history, config["projection"], config["actual"], config["baseline"])
                metric_rows.append({"Target": target_name, **metrics})
            metric_table = pd.DataFrame(metric_rows)
            st.dataframe(
                metric_table.style.format({
                    "MAE": "{:.3f}", "RMSE": "{:.3f}", "Bias": "{:+.3f}",
                    "Correlation": "{:.3f}", "Baseline_MAE": "{:.3f}",
                }),
                width="stretch",
                hide_index=True,
            )
            st.caption("Bias is projection minus actual. Positive bias means the model is projecting too high.")

            target_for_scatter = st.selectbox("Projection chart target", list(BACKTEST_TARGETS), key="pitcher_scatter_target")
            scatter_config = BACKTEST_TARGETS[target_for_scatter]
            scatter = completed_history[[scatter_config["projection"], scatter_config["actual"]]].dropna().rename(
                columns={scatter_config["projection"]: "Projection", scatter_config["actual"]: "Actual"}
            )
            if not scatter.empty:
                st.scatter_chart(scatter, x="Projection", y="Actual", width="stretch")

            st.markdown("### Linear calibration")
            calibration_bundle, calibration_table = make_calibration_bundle(completed_history, int(minimum_calibration_rows))
            display_calibration = calibration_table.copy()
            if not display_calibration.empty:
                st.dataframe(
                    display_calibration.style.format({
                        "slope": "{:.4f}", "intercept": "{:+.4f}", "raw_mae": "{:.3f}",
                        "calibrated_mae": "{:.3f}", "holdout_raw_mae": "{:.3f}",
                        "holdout_calibrated_mae": "{:.3f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )
            st.caption(
                "The holdout columns fit coefficients on the earlier 70% of starts and test them on the later 30%. "
                "Use holdout improvement—not only in-sample improvement—before applying calibration."
            )
            if calibration_bundle.get("targets"):
                calibration_json = json.dumps(calibration_bundle, indent=2).encode("utf-8")
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "Download calibration JSON",
                        calibration_json,
                        file_name="pitcher_calibration.json",
                        mime="application/json",
                        width="stretch",
                    )
                with c2:
                    if st.button("Apply fitted calibration to current board", width="stretch"):
                        st.session_state["pitcher_calibration"] = calibration_bundle
                        recalibrated = apply_projection_calibration(st.session_state["pitcher_board"], calibration_bundle)
                        st.session_state["pitcher_board"] = assign_scores(recalibrated)
                        st.rerun()

            st.markdown("### Probability calibration against prop lines")
            prop_target = st.selectbox("Prop target", list(BACKTEST_TARGETS), key="pitcher_prop_calibration_target")
            prop_records = prop_calibration_records(completed_history, prop_target)
            if prop_records.empty:
                st.info(f"Enter {BACKTEST_TARGETS[prop_target]['line']} values in your saved snapshots to evaluate probabilities.")
            else:
                prop_table, brier_score = probability_calibration_table(prop_records)
                p1, p2 = st.columns(2)
                p1.metric("Decisions", f"{len(prop_records):,}")
                p2.metric("Brier score", f"{brier_score:.3f}", help="Lower is better; 0 is perfect.")
                st.dataframe(
                    prop_table.style.format({
                        "Average_Probability": "{:.1%}", "Actual_Win_Rate": "{:.1%}", "Calibration_Gap": "{:+.1%}",
                    }),
                    width="stretch",
                    hide_index=True,
                )

            st.markdown("### Score buckets")
            bucket_target = st.selectbox("Score target", list(BACKTEST_TARGETS), key="pitcher_score_bucket_target")
            bucket_table = score_bucket_table(completed_history, bucket_target)
            if not bucket_table.empty:
                st.dataframe(
                    bucket_table.style.format({
                        "Average_Projection": "{:.2f}", "Average_Actual": "{:.2f}",
                        "Average_Score": "{:.1f}", "Over_Rate": "{:.1%}",
                    }),
                    width="stretch",
                    hide_index=True,
                )

            st.markdown("### Segment and lookback checks")
            segment_choice = st.selectbox(
                "Break results down by",
                [column for column in ["LookbackDays", "Lineup_Status", "Confidence_Level", "PitcherLocation", "Hand", "ModelVersion"] if column in completed_history.columns],
                key="pitcher_segment_choice",
            )
            segment_table = grouped_accuracy(completed_history, segment_choice)
            if not segment_table.empty:
                st.dataframe(
                    segment_table.style.format({
                        "MAE": "{:.3f}", "RMSE": "{:.3f}", "Bias": "{:+.3f}",
                        "Correlation": "{:.3f}", "Baseline_MAE": "{:.3f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )

            with st.expander("Completed history used in this report"):
                st.dataframe(completed_history, width="stretch", hide_index=True, height=500)

with notes_tab:
    st.markdown(
        """
        ### How to read the Pitcher Lab

        - **Projected K** combines the pitcher's recent/season strikeout rate, Statcast K and whiff skills, projected batters faced, the opponent lineup's strikeout profile and the pitch-type matchup.
        - **Projected ER** blends recent and season ER rates with xwOBA, walks, barrels, opponent quality, park factor and first-pitch weather.
        - **Projected outs** starts with recent workload and adjusts for walks and opponent lineup difficulty.
        - **K Score**, **Run Prevention Score** and **Outs Score** are 0–100 relative scores within the selected slate. They are not probabilities.
        - **Overall Score** is 40% K Score, 35% Run Prevention Score and 25% Outs Score.
        - Confirmed lineups are preferred. Before posting, the app uses the opponent's most recent observed lineup from Statcast.
        - Small samples are shrunk toward league averages, and the confidence label falls when a pitcher has few starts or little Statcast history.

        - The **Backtest & calibration** tab exports timestamped pregame snapshots, fills official K/ER/outs results, measures MAE/RMSE/bias, compares simple recent-form baselines, checks score buckets and tests probability calibration.
        - Use **Add current snapshot to session master** to append today's edited pregame board without uploading it first. Download the updated master-history CSV before leaving because Streamlit Community Cloud storage is temporary.
        - A calibration JSON can be downloaded after a sufficient sample and loaded from the sidebar on future slates. Prefer coefficients that improve the chronological holdout sample.

        This remains a projection framework. Use several hundred completed starts before making large weight changes, and never include snapshots generated after first pitch in a fair backtest.
        """
    )
