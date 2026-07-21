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

MODEL_VERSION = "pitcher-lab-two-stage-k-v3"
K_MODEL_VERSION = "two-stage-k-v1"
BF_MODEL_FEATURES = [
    "Season_BF", "Recent_BF", "Last_Start_Pitches", "Last3_Pitches",
    "Recent_Outs", "Outs_SD", "Short_Hook_Rate", "Prior_Starts",
]
K_RATE_MODEL_FEATURES = [
    "Pitcher_K_Rate", "Recent_K_Rate", "Whiff_Pct", "CSW_Pct",
    "Opponent_K_Rate", "Hand_L",
]
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
        row["SelectedSource"] = f"Best over | {row.get('Bookmaker')}"
        row["BookCount"] = int(books["BookmakerKey"].nunique())
        return row
    if source_mode == "Best sportsbook under line":
        best_line = pd.to_numeric(books["Line"], errors="coerce").max()
        subset = books[pd.to_numeric(books["Line"], errors="coerce").eq(best_line)].copy()
        subset["_Price"] = pd.to_numeric(subset["UnderOdds"], errors="coerce").fillna(-100000)
        row = subset.sort_values("_Price").iloc[-1].drop(labels="_Price").to_dict()
        row["SelectedSource"] = f"Best under | {row.get('Bookmaker')}"
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


def apply_odds_quotes_to_board(
    board: pd.DataFrame,
    quotes: pd.DataFrame,
    source_mode: str,
    independent_model_weight: float = 0.35,
    minimum_bet_edge: float = 0.75,
) -> pd.DataFrame:
    result = board.copy()
    result["Independent_Proj_K"] = pd.to_numeric(result.get("Proj_K"), errors="coerce")
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
    result["Market_Blend_K"] = np.nan
    valid_k_line = pd.to_numeric(result.get("K_Line"), errors="coerce").notna()
    weight = float(np.clip(independent_model_weight, 0.0, 1.0))
    result.loc[valid_k_line, "Market_Blend_K"] = (
        weight * pd.to_numeric(result.loc[valid_k_line, "Independent_Proj_K"], errors="coerce")
        + (1.0 - weight) * pd.to_numeric(result.loc[valid_k_line, "K_Line"], errors="coerce")
    )
    result["Final_Proj_K"] = result["Market_Blend_K"].where(valid_k_line, result["Independent_Proj_K"])
    result["Blended_K_Edge"] = result["Final_Proj_K"] - pd.to_numeric(result.get("K_Line"), errors="coerce")
    for index in result.index[valid_k_line]:
        line = float(result.at[index, "K_Line"])
        final_projection = float(result.at[index, "Final_Proj_K"])
        result.at[index, "K_Model_Over_Prob"] = probability_over_line(final_projection, line)
        market_probability = pd.to_numeric(pd.Series([result.at[index, "K_Market_Over_Prob"]]), errors="coerce").iloc[0]
        result.at[index, "K_Probability_Edge"] = (
            float(result.at[index, "K_Model_Over_Prob"] - market_probability)
            if pd.notna(market_probability) else np.nan
        )
    role_ok = result.get("Role_Eligible", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    confirmed = result.get("Lineup_Status", pd.Series("", index=result.index)).astype(str).eq("Confirmed")
    edge_ok = result["Blended_K_Edge"].abs().ge(float(minimum_bet_edge)).fillna(False)
    result["Bet_Eligible"] = role_ok & confirmed & valid_k_line & edge_ok
    result["K_Bet_Signal"] = np.where(
        result["Bet_Eligible"], np.where(result["Blended_K_Edge"].gt(0), "OVER", "UNDER"), "PASS"
    )
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
# Chronological two-stage strikeout model
# -----------------------------------------------------------------------------

def _mode_text(series: pd.Series, default: str = "") -> str:
    values = series.dropna().astype(str)
    return str(values.mode().iloc[0]) if not values.empty else default


def build_historical_start_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Create one leakage-safe result row per starting pitcher/game from Statcast."""
    if df is None or df.empty:
        return pd.DataFrame()
    work = df[df["game_date"].notna() & df["game_pk"].notna() & df["pitcher"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work["at_bat_number"] = pd.to_numeric(work["at_bat_number"], errors="coerce")
    first_ab = (
        work.groupby(["game_pk", "pitcher_team", "pitcher"], dropna=False)["at_bat_number"]
        .min().reset_index(name="FirstAB")
    )
    starters = (
        first_ab.sort_values(["game_pk", "pitcher_team", "FirstAB"])
        .drop_duplicates(["game_pk", "pitcher_team"], keep="first")
        [["game_pk", "pitcher_team", "pitcher"]]
    )
    work = work.merge(starters.assign(IsStarter=True), on=["game_pk", "pitcher_team", "pitcher"], how="left")
    work = work[work["IsStarter"].fillna(False)].copy()
    if work.empty:
        return pd.DataFrame()
    summaries = (
        work.groupby(["game_date", "game_pk", "pitcher", "pitcher_team"], dropna=False)
        .agg(
            Opponent=("batter_team", lambda s: _mode_text(s, "")),
            Hand=("p_throws", lambda s: _mode_text(s, "R")),
            Actual_BF=("pa_key", lambda s: int(s[work.loc[s.index, "is_pa_end"]].nunique())),
            Actual_K=("is_k", "sum"),
            Actual_Outs=("outs_recorded", "sum"),
            Actual_Pitches=("pitcher", "size"),
            Swings=("is_swing", "sum"),
            Whiffs=("is_whiff", "sum"),
            CalledStrikes=("is_called_strike", "sum"),
        )
        .reset_index()
        .rename(columns={"game_date": "Date", "game_pk": "GamePK", "pitcher": "PitcherID", "pitcher_team": "Team"})
    )
    for column in ["Actual_BF", "Actual_K", "Actual_Outs", "Actual_Pitches", "Swings", "Whiffs", "CalledStrikes"]:
        summaries[column] = pd.to_numeric(summaries[column], errors="coerce").fillna(0.0)
    # Remove openers and scoring artifacts from the training target. These are
    # still useful later as examples for the live role-risk gate.
    summaries["StarterLengthOK"] = summaries["Actual_Pitches"].ge(40) | summaries["Actual_Outs"].ge(9)
    return summaries.sort_values(["Date", "GamePK", "Team"]).reset_index(drop=True)


def build_chronological_k_backfill(df: pd.DataFrame, target_start: object, target_end: object) -> pd.DataFrame:
    """Build pregame features using only games strictly before each target start."""
    starts = build_historical_start_summaries(df)
    if starts.empty:
        return starts
    target_start = pd.Timestamp(target_start).normalize()
    target_end = pd.Timestamp(target_end).normalize()
    rows: list[dict] = []
    for current in starts.itertuples(index=False):
        current_date = pd.Timestamp(current.Date).normalize()
        if current_date < target_start or current_date > target_end or not bool(current.StarterLengthOK):
            continue
        prior = starts[starts["Date"].lt(current_date) & starts["StarterLengthOK"]].copy()
        pitcher_prior = prior[prior["PitcherID"].eq(current.PitcherID)].sort_values("Date")
        if len(pitcher_prior) < 2:
            continue
        recent = pitcher_prior.tail(5)
        league_bf = max(float(prior["Actual_BF"].sum()), 1.0)
        league_k_rate = float(prior["Actual_K"].sum() / league_bf)
        league_whiff = float(prior["Whiffs"].sum() / max(prior["Swings"].sum(), 1.0))
        league_csw = float((prior["Whiffs"].sum() + prior["CalledStrikes"].sum()) / max(prior["Actual_Pitches"].sum(), 1.0))
        pitcher_bf = float(pitcher_prior["Actual_BF"].sum())
        recent_bf = float(recent["Actual_BF"].sum())
        pitcher_k = (float(pitcher_prior["Actual_K"].sum()) + league_k_rate * 120.0) / (pitcher_bf + 120.0)
        recent_k = (float(recent["Actual_K"].sum()) + pitcher_k * 55.0) / (recent_bf + 55.0)
        whiff = (float(pitcher_prior["Whiffs"].sum()) + league_whiff * 180.0) / (float(pitcher_prior["Swings"].sum()) + 180.0)
        csw = (float(pitcher_prior["Whiffs"].sum() + pitcher_prior["CalledStrikes"].sum()) + league_csw * 300.0) / (float(pitcher_prior["Actual_Pitches"].sum()) + 300.0)
        opponent_prior = prior[prior["Team"].eq(current.Opponent) & prior["Hand"].eq(current.Hand)]
        opponent_bf = float(opponent_prior["Actual_BF"].sum())
        opponent_k = (float(opponent_prior["Actual_K"].sum()) + league_k_rate * 220.0) / (opponent_bf + 220.0)
        outs = pd.to_numeric(pitcher_prior["Actual_Outs"], errors="coerce").dropna()
        recent_outs = float(recent["Actual_Outs"].mean())
        short_hook = float(pitcher_prior["Actual_Outs"].lt(15).mean())
        rows.append({
            "Date": str(current_date.date()), "GamePK": int(current.GamePK), "PitcherID": int(current.PitcherID),
            "Team": str(current.Team), "Opponent": str(current.Opponent), "Hand": str(current.Hand),
            "Season_BF": float(pitcher_prior["Actual_BF"].mean()),
            "Recent_BF": float(recent["Actual_BF"].mean()),
            "Last_Start_Pitches": float(pitcher_prior.iloc[-1]["Actual_Pitches"]),
            "Last3_Pitches": float(pitcher_prior.tail(3)["Actual_Pitches"].mean()),
            "Recent_Outs": recent_outs,
            "Outs_SD": float(outs.tail(8).std(ddof=0)) if len(outs) > 1 else 3.0,
            "Short_Hook_Rate": short_hook,
            "Prior_Starts": int(len(pitcher_prior)),
            "Pitcher_K_Rate": pitcher_k, "Recent_K_Rate": recent_k,
            "Whiff_Pct": whiff, "CSW_Pct": csw,
            "Opponent_K_Rate": opponent_k, "Hand_L": 1.0 if str(current.Hand) == "L" else 0.0,
            "Actual_BF": float(current.Actual_BF), "Actual_K": float(current.Actual_K),
            "Actual_K_Rate": float(current.Actual_K / max(current.Actual_BF, 1.0)),
            "Actual_Outs": float(current.Actual_Outs), "Actual_Pitches": float(current.Actual_Pitches),
        })
    return pd.DataFrame(rows).sort_values(["Date", "GamePK", "PitcherID"]).reset_index(drop=True) if rows else pd.DataFrame()


def _fit_ridge_payload(frame: pd.DataFrame, features: list[str], target: str, alpha: float = 8.0) -> dict:
    clean = frame[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean) < max(40, len(features) * 5):
        return {}
    x = clean[features].to_numpy(float)
    y = clean[target].to_numpy(float)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    z = (x - means) / scales
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    return {
        "features": features, "means": means.tolist(), "scales": scales.tolist(),
        "intercept": float(coefficients[0]), "coefficients": coefficients[1:].tolist(),
    }


def _ridge_predict(payload: dict, values: dict, default: float) -> float:
    if not isinstance(payload, dict) or not payload.get("features"):
        return float(default)
    try:
        x = np.array([float(values.get(name, np.nan)) for name in payload["features"]], dtype=float)
        means = np.array(payload["means"], dtype=float)
        scales = np.array(payload["scales"], dtype=float)
        x = np.where(np.isfinite(x), x, means)
        return float(payload["intercept"] + np.dot((x - means) / scales, np.array(payload["coefficients"], dtype=float)))
    except Exception:
        return float(default)


def train_two_stage_k_model(backfill: pd.DataFrame, alpha: float = 8.0) -> tuple[dict, dict]:
    """Fit BF and K/BF separately and report a strict chronological holdout."""
    if backfill is None or len(backfill) < 100:
        return {}, {"ok": False, "message": "At least 100 chronological starts are required."}
    data = backfill.sort_values(["Date", "GamePK", "PitcherID"]).reset_index(drop=True)
    split = max(80, int(len(data) * 0.80))
    train, test = data.iloc[:split], data.iloc[split:]
    if len(test) < 20:
        return {}, {"ok": False, "message": "At least 20 holdout starts are required."}
    bf_model = _fit_ridge_payload(train, BF_MODEL_FEATURES, "Actual_BF", alpha)
    rate_model = _fit_ridge_payload(train, K_RATE_MODEL_FEATURES, "Actual_K_Rate", alpha)
    if not bf_model or not rate_model:
        return {}, {"ok": False, "message": "The backfill did not contain enough complete feature rows."}
    def predict_rows(frame: pd.DataFrame) -> np.ndarray:
        predictions = []
        for row in frame.to_dict("records"):
            bf = np.clip(_ridge_predict(bf_model, row, row.get("Recent_BF", 22.0)), 12.0, 31.0)
            rate = np.clip(_ridge_predict(rate_model, row, row.get("Pitcher_K_Rate", 0.22)), 0.08, 0.42)
            predictions.append(float(bf * rate))
        return np.array(predictions)
    train_pred = predict_rows(train)
    test_pred = predict_rows(test)
    model = {
        "version": K_MODEL_VERSION,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "training_rows": int(len(train)), "holdout_rows": int(len(test)), "alpha": float(alpha),
        "bf_model": bf_model, "rate_model": rate_model,
    }
    metrics = {
        "ok": True, "training_n": int(len(train)), "holdout_n": int(len(test)),
        "training_mae": float(np.mean(np.abs(train_pred - train["Actual_K"].to_numpy(float)))),
        "holdout_mae": float(np.mean(np.abs(test_pred - test["Actual_K"].to_numpy(float)))),
        "holdout_bias": float(np.mean(test_pred - test["Actual_K"].to_numpy(float))),
        "holdout_baseline_mae": float(np.mean(np.abs((test["Recent_BF"] * test["Pitcher_K_Rate"]).to_numpy(float) - test["Actual_K"].to_numpy(float)))),
    }
    return model, metrics


def parse_two_stage_k_model(payload: object) -> dict:
    if not isinstance(payload, dict) or payload.get("version") != K_MODEL_VERSION:
        return {}
    if not isinstance(payload.get("bf_model"), dict) or not isinstance(payload.get("rate_model"), dict):
        return {}
    return payload


def validate_starter_role(log: dict) -> tuple[str, bool, str]:
    starts = int(log.get("Starts", 0) or 0)
    last_pitches = float(log.get("Last_Start_Pitches", 0.0) or 0.0)
    last3 = float(log.get("Last3_Pitches", 0.0) or 0.0)
    recent_bf = float(log.get("Recent_BF", 0.0) or 0.0)
    if starts >= 3 and last_pitches >= 55 and last3 >= 55 and recent_bf >= 16:
        return "Verified starter", True, "Recent starter workload is established."
    if starts >= 1 and last_pitches >= 60 and recent_bf >= 15:
        return "Emerging starter", True, "Limited start history; require a market line before betting."
    return "Opener / role risk", False, "Recent workload does not verify a normal starter role."


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


def build_hitter_zone_matchups(
    df: pd.DataFrame,
    base: pd.DataFrame,
    pitcher_id: int,
    pitcher_hand: str,
    league: dict,
) -> pd.DataFrame:
    """Score how well each hitter's zone damage overlaps this pitcher's locations.

    Hitter results are limited to the pitcher's handedness. Rates are regressed
    toward league average in small samples, then weighted by the selected
    pitcher's actual zone usage. This is a matchup indicator, not a home-run
    probability.
    """
    result = base.copy()
    if result.empty:
        return result

    valid_zones = list(range(1, 10))
    pitcher_pitches = df[df["pitcher"].eq(int(pitcher_id)) & df["zone"].isin(valid_zones)].copy()
    if pitcher_pitches.empty:
        zone_usage = pd.Series(1.0 / len(valid_zones), index=valid_zones, dtype=float)
    else:
        zone_usage = pitcher_pitches["zone"].astype(int).value_counts(normalize=True).reindex(valid_zones, fill_value=0.0)

    pitcher_bbe = pitcher_pitches[pitcher_pitches["is_bbe"]].copy()
    pitcher_zone = {}
    for zone in valid_zones:
        zone_bbe = pitcher_bbe[pitcher_bbe["zone"].eq(zone)]
        pitcher_zone[zone] = (
            float(zone_bbe["is_hard_hit"].sum()) + league["HH_BBE"] * 18.0
        ) / (len(zone_bbe) + 18.0)

    hitter_ids = pd.to_numeric(result["player_id"], errors="coerce").dropna().astype(int).tolist()
    hitter_pitches = df[
        df["batter"].isin(hitter_ids)
        & df["p_throws"].eq(pitcher_hand)
        & df["zone"].isin(valid_zones)
    ].copy()
    hitter_bbe = hitter_pitches[hitter_pitches["is_bbe"]].copy()

    matched_hh, zone_fit, pitcher_vulnerability, zone_samples = [], [], [], []
    for player_id in pd.to_numeric(result["player_id"], errors="coerce"):
        player_bbe = hitter_bbe[hitter_bbe["batter"].eq(player_id)]
        weighted_hh = 0.0
        weighted_fit = 0.0
        weighted_pitcher_hh = 0.0
        for zone in valid_zones:
            usage = float(zone_usage.loc[zone])
            zone_bbe = player_bbe[player_bbe["zone"].eq(zone)]
            hitter_hh = (
                float(zone_bbe["is_hard_hit"].sum()) + league["HH_BBE"] * 10.0
            ) / (len(zone_bbe) + 10.0)
            pitcher_hh = float(pitcher_zone[zone])
            weighted_hh += usage * hitter_hh
            weighted_pitcher_hh += usage * pitcher_hh
            # Both sides must be strong in the same zones for a premium fit.
            overlap = math.sqrt(
                max(hitter_hh / max(league["HH_BBE"], 0.001), 0.05)
                * max(pitcher_hh / max(league["HH_BBE"], 0.001), 0.05)
            )
            weighted_fit += usage * overlap
        matched_hh.append(float(weighted_hh))
        pitcher_vulnerability.append(float(weighted_pitcher_hh))
        zone_fit.append(float(np.clip(50.0 + (weighted_fit - 1.0) * 62.0, 0.0, 100.0)))
        zone_samples.append(int(len(player_bbe)))

    result["Matched_HH_Pct"] = matched_hh
    result["Zone_Fit_Score"] = zone_fit
    result["Pitcher_Zone_HH_Allowed"] = pitcher_vulnerability
    result["Zone_BBE"] = zone_samples

    hh_component = np.clip(
        50.0 + (result["Matched_HH_Pct"] - league["HH_BBE"]) * 145.0,
        0.0,
        100.0,
    )
    hr_component = np.clip(
        50.0 + (result["HR_Pct"] - league["HR_PA"]) * 650.0,
        0.0,
        100.0,
    )
    barrel_component = np.clip(
        50.0 + (result["Brl_BBE"] - league["Brl_BBE"]) * 300.0,
        0.0,
        100.0,
    )
    result["HR_Threat_Score"] = np.clip(
        0.42 * hh_component
        + 0.35 * result["Zone_Fit_Score"]
        + 0.15 * barrel_component
        + 0.08 * hr_component,
        0.0,
        100.0,
    )
    result["HR_Threat"] = pd.cut(
        result["HR_Threat_Score"],
        bins=[-np.inf, 44.999, 59.999, 74.999, np.inf],
        labels=["Low", "Elevated", "High", "Extreme"],
    ).astype(str)
    result["Matchup_Flag"] = ""
    if result["HR_Threat_Score"].notna().any():
        top_index = result["HR_Threat_Score"].idxmax()
        result.loc[top_index, "Matchup_Flag"] = "TOP HR THREAT"
    return result


def build_lineup_profile(
    df: pd.DataFrame,
    lineup: pd.DataFrame,
    opponent_team: str,
    pitcher_hand: str,
    league: dict,
    pitcher_id: int | None = None,
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

    if pitcher_id is not None:
        base = build_hitter_zone_matchups(df, base, int(pitcher_id), pitcher_hand, league)

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
        "Top6_K_Pct": float(np.average(base.loc[base["LineupSpot"].le(6), "K_Pct"], weights=base.loc[base["LineupSpot"].le(6), "Weight"])) if len(base.loc[base["LineupSpot"].le(6)]) else league["K_PA"],
        "Bottom3_K_Pct": float(np.average(base.loc[base["LineupSpot"].ge(7), "K_Pct"], weights=base.loc[base["LineupSpot"].ge(7), "Weight"])) if len(base.loc[base["LineupSpot"].ge(7)]) else league["K_PA"],
        "Lineup_Contact_Risk": float(1.0 - np.average(base["Whiff_Pct"], weights=weights)) if len(base) else 1.0 - league["Whiff"],
        "Top_HR_Threat": str(base.loc[base["HR_Threat_Score"].idxmax(), "Player"]) if "HR_Threat_Score" in base and base["HR_Threat_Score"].notna().any() else "—",
        "Top_HR_Threat_Score": float(base["HR_Threat_Score"].max()) if "HR_Threat_Score" in base and base["HR_Threat_Score"].notna().any() else 50.0,
        "Lineup_Zone_Fit": float(np.average(base["Zone_Fit_Score"], weights=weights)) if "Zone_Fit_Score" in base and len(base) else 50.0,
    }
    keep = ["LineupSpot", "Player", "Position", "EffectiveStand", "PA", "K_Pct", "BB_Pct", "HR_Pct", "xwOBA", "Brl_BBE", "HH_BBE", "Whiff_Pct", "Matched_HH_Pct", "Pitcher_Zone_HH_Allowed", "Zone_Fit_Score", "HR_Threat_Score", "HR_Threat", "Matchup_Flag", "Zone_BBE"]
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
    return float(factor), f"park_factors.csv | {int(row.get('rolling_years', 3))}-year"


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
            "Last_Start_Pitches": 88.0, "Last3_Pitches": 88.0, "Pitch_Count_Trend": 0.0,
            "Quality_Start_Rate": 0.45, "Stable_Workload_Rate": 0.55, "Short_Hook_Rate": 0.18,
            "Blowup_Hook_Rate": 0.15, "Recent_BB_BF": league["BB_PA"],
            "Season_K_BF": league["K_PA"], "Recent_K_BF": league["K_PA"],
            "Season_ER9": league["ER9"], "Recent_ER9": league["ER9"],
        }
    clean = logs.dropna(subset=["Outs"]).copy()
    if clean.empty:
        return game_log_profile(pd.DataFrame(), league)
    clean["K_BF"] = safe_divide(clean["K"], clean["BF"], league["K_PA"])
    clean["ER9"] = safe_divide(clean["ER"] * 27.0, clean["Outs"], league["ER9"])
    clean["BB_BF"] = safe_divide(clean["BB"], clean["BF"], league["BB_PA"])
    last3_pitches = float(pd.to_numeric(clean["Pitches"], errors="coerce").tail(3).mean()) if len(clean) else 88.0
    season_pitches = float(pd.to_numeric(clean["Pitches"], errors="coerce").mean()) if len(clean) else 88.0
    last_start_pitches = float(pd.to_numeric(clean["Pitches"], errors="coerce").dropna().iloc[-1]) if pd.to_numeric(clean["Pitches"], errors="coerce").dropna().size else season_pitches
    return {
        "Starts": int(len(clean)),
        "Season_Outs": float(clean["Outs"].mean()),
        "Recent_Outs": weighted_recent_mean(clean["Outs"], 5, clean["Outs"].mean()),
        "Outs_SD": float(max(clean["Outs"].tail(10).std(ddof=1) if len(clean.tail(10)) > 1 else 2.5, 1.5)),
        "Season_BF": float(clean["BF"].mean()),
        "Recent_BF": weighted_recent_mean(clean["BF"], 5, clean["BF"].mean()),
        "Season_Pitches": season_pitches,
        "Recent_Pitches": weighted_recent_mean(clean["Pitches"], 5, clean["Pitches"].mean()),
        "Last_Start_Pitches": last_start_pitches,
        "Last3_Pitches": last3_pitches,
        "Pitch_Count_Trend": float(last3_pitches - season_pitches),
        "Quality_Start_Rate": float((clean["Outs"].ge(18)).mean()),
        "Stable_Workload_Rate": float((clean["Outs"].tail(5).ge(15)).mean()),
        "Short_Hook_Rate": float((clean["Outs"].le(12)).mean()),
        "Blowup_Hook_Rate": float(((clean["ER"].ge(4)) & (clean["Outs"].le(15))).mean()),
        "Recent_BB_BF": weighted_recent_mean(clean["BB_BF"], 5, clean["BB_BF"].mean()),
        "Season_K_BF": float(clean["K"].sum() / max(clean["BF"].sum(), 1.0)),
        "Recent_K_BF": weighted_recent_mean(clean["K_BF"], 5, clean["K_BF"].mean()),
        "Season_ER9": float(clean["ER"].sum() * 27.0 / max(clean["Outs"].sum(), 1.0)),
        "Recent_ER9": weighted_recent_mean(clean["ER9"], 5, clean["ER9"].mean()),
    }




def pitcher_recent_arsenal_profile(df: pd.DataFrame, pitcher_id: int, league: dict) -> dict:
    """Recent-form arsenal signals from currently loaded Statcast data.

    These features are intentionally self-contained so the dashboard still works
    without paid feeds: velocity trend, recent whiff/CSW, and pitch-mix change.
    """
    pitches = df[df["pitcher"].eq(int(pitcher_id))].copy()
    defaults = {
        "Recent_Whiff_Pct": league["Whiff"],
        "Recent_CSW_Pct": league["CSW"],
        "Fastball_Velo": np.nan,
        "Fastball_Velo_Trend": 0.0,
        "Pitch_Mix_Change": 0.0,
        "Arsenal_Form_Score": 50.0,
    }
    if pitches.empty or "game_date" not in pitches.columns:
        return defaults
    pitches["game_date"] = pd.to_datetime(pitches["game_date"], errors="coerce")
    pitches = pitches[pitches["game_date"].notna()].copy()
    if pitches.empty:
        return defaults
    max_date = pitches["game_date"].max()
    recent = pitches[pitches["game_date"].ge(max_date - pd.Timedelta(days=14))].copy()
    prior = pitches[pitches["game_date"].lt(max_date - pd.Timedelta(days=14))].copy()
    if recent.empty:
        recent = pitches.tail(min(len(pitches), 250)).copy()
    recent_swings = max(int(recent["is_swing"].sum()), 1)
    recent_whiff = float(recent["is_whiff"].sum() / recent_swings)
    recent_csw = float((recent["is_whiff"].sum() + recent["is_called_strike"].sum()) / max(len(recent), 1))
    fb_mask = recent["pitch_name"].astype(str).str.contains("4-Seam|Fastball|Sinker|Cutter", case=False, na=False)
    recent_fb = float(pd.to_numeric(recent.loc[fb_mask, "release_speed"], errors="coerce").mean()) if fb_mask.any() else np.nan
    if prior.empty:
        prior_fb = recent_fb
        mix_change = 0.0
    else:
        prior_fb_mask = prior["pitch_name"].astype(str).str.contains("4-Seam|Fastball|Sinker|Cutter", case=False, na=False)
        prior_fb = float(pd.to_numeric(prior.loc[prior_fb_mask, "release_speed"], errors="coerce").mean()) if prior_fb_mask.any() else recent_fb
        recent_mix = recent["pitch_name"].value_counts(normalize=True)
        prior_mix = prior["pitch_name"].value_counts(normalize=True)
        all_pitches = recent_mix.index.union(prior_mix.index)
        mix_change = float((recent_mix.reindex(all_pitches, fill_value=0.0) - prior_mix.reindex(all_pitches, fill_value=0.0)).abs().sum() / 2.0)
    velo_trend = 0.0 if pd.isna(recent_fb) or pd.isna(prior_fb) else float(recent_fb - prior_fb)
    arsenal_score = float(np.clip(50.0 + (recent_whiff - league["Whiff"]) * 170.0 + (recent_csw - league["CSW"]) * 145.0 + velo_trend * 5.0 - mix_change * 18.0, 0.0, 100.0))
    return {
        "Recent_Whiff_Pct": recent_whiff,
        "Recent_CSW_Pct": recent_csw,
        "Fastball_Velo": recent_fb,
        "Fastball_Velo_Trend": velo_trend,
        "Pitch_Mix_Change": mix_change,
        "Arsenal_Form_Score": arsenal_score,
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
    k_model: dict | None = None,
) -> tuple[dict, dict]:
    logs, log_error = fetch_pitcher_game_log(int(pitcher_id), int(season))
    log = game_log_profile(logs, league)
    pitcher = pitcher_statcast_profile(df, int(pitcher_id), league)
    opponent, hitter_table = build_lineup_profile(
        df, lineup, opponent_team, pitcher["Hand"], league, pitcher_id=int(pitcher_id)
    )
    pitch_match_score, pitch_match_table = build_pitch_type_matchup(df, int(pitcher_id), lineup, opponent_team, pitcher["Hand"], league)
    arsenal = pitcher_recent_arsenal_profile(df, int(pitcher_id), league)
    if pitch_match_table is not None and not pitch_match_table.empty:
        pitch_mix_weights = pd.to_numeric(pitch_match_table.get("Usage"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if pitch_mix_weights.sum() <= 0:
            pitch_mix_weights = np.ones(len(pitch_match_table), dtype=float)
        pitch_mix_whiff = float(np.average(pd.to_numeric(pitch_match_table.get("Opponent Whiff%"), errors="coerce").fillna(league["Whiff"]), weights=pitch_mix_weights))
        pitch_mix_xwoba = float(np.average(pd.to_numeric(pitch_match_table.get("Opponent xwOBA"), errors="coerce").fillna(league["xwOBA"]), weights=pitch_mix_weights))
        pitch_mix_damage = float(np.average(pd.to_numeric(pitch_match_table.get("Pitcher xwOBA"), errors="coerce").fillna(league["xwOBA"]), weights=pitch_mix_weights))
    else:
        pitch_mix_whiff = league["Whiff"]
        pitch_mix_xwoba = league["xwOBA"]
        pitch_mix_damage = league["xwOBA"]
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
    rule_projected_bf = float(np.clip(projected_outs * bf_per_out, 13.0, 31.0))
    projected_pitches = float(np.clip(0.60 * log["Recent_Pitches"] + 0.40 * log["Season_Pitches"], 65.0, 112.0))

    # Stable additive K-rate construction. Additive deltas avoid the runaway
    # compounding that occurred when several noisy ratios were multiplied.
    whiff_k_equivalent = league["K_PA"] + 0.72 * (pitcher["Whiff_Pct"] - league["Whiff"])
    csw_k_equivalent = league["K_PA"] + 0.78 * (pitcher["CSW_Pct"] - league["CSW"])
    skill_rate = (
        0.46 * pitcher["K_PA"]
        + 0.14 * log["Recent_K_BF"]
        + 0.18 * log["Season_K_BF"]
        + 0.12 * whiff_k_equivalent
        + 0.10 * csw_k_equivalent
    )
    matchup_delta = (
        0.52 * (opponent["Opp_K_PA"] - league["K_PA"])
        + 0.12 * (opponent["Top6_K_Pct"] - opponent["Opp_K_PA"])
        + 0.08 * (pitch_mix_whiff - league["Whiff"])
        + 0.00018 * (pitch_match_score - 50.0)
    )
    hand_adjustment = -0.020 if pitcher["Hand"] == "L" else 0.0
    rule_k_rate = float(np.clip(skill_rate + matchup_delta + hand_adjustment, 0.08, 0.42))

    role_status, role_eligible, role_note = validate_starter_role(log)
    model_payload = parse_two_stage_k_model(k_model or {})
    feature_values = {
        "Season_BF": log["Season_BF"], "Recent_BF": log["Recent_BF"],
        "Last_Start_Pitches": log["Last_Start_Pitches"], "Last3_Pitches": log["Last3_Pitches"],
        "Recent_Outs": log["Recent_Outs"], "Outs_SD": log["Outs_SD"],
        "Short_Hook_Rate": log["Short_Hook_Rate"], "Prior_Starts": log["Starts"],
        "Pitcher_K_Rate": pitcher["K_PA"], "Recent_K_Rate": log["Recent_K_BF"],
        "Whiff_Pct": pitcher["Whiff_Pct"], "CSW_Pct": pitcher["CSW_Pct"],
        "Opponent_K_Rate": opponent["Opp_K_PA"], "Hand_L": 1.0 if pitcher["Hand"] == "L" else 0.0,
    }
    if model_payload:
        learned_bf = float(np.clip(_ridge_predict(model_payload["bf_model"], feature_values, rule_projected_bf), 12.0, 31.0))
        learned_rate = float(np.clip(_ridge_predict(model_payload["rate_model"], feature_values, rule_k_rate), 0.08, 0.42))
        projected_bf = float(0.75 * learned_bf + 0.25 * rule_projected_bf)
        adjusted_k_rate = float(0.75 * learned_rate + 0.25 * rule_k_rate)
        k_model_source = f"Trained {model_payload.get('version', K_MODEL_VERSION)}"
    else:
        projected_bf = rule_projected_bf
        adjusted_k_rate = rule_k_rate
        k_model_source = "Stabilized rules"
    if not role_eligible:
        projected_bf = min(projected_bf, 16.0)
    projected_k = float(np.clip(projected_bf * adjusted_k_rate, 0.5, 13.5))

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
    workload_confidence = float(np.clip(
        0.25 * starts_conf
        + 0.25 * (100.0 - min(100.0, max(log["Outs_SD"], 0.0) * 14.0))
        + 0.20 * log["Stable_Workload_Rate"] * 100.0
        + 0.15 * log["Quality_Start_Rate"] * 100.0
        + 0.15 * (100.0 - log["Short_Hook_Rate"] * 100.0)
        - max(0.0, 72.0 - log["Last_Start_Pitches"]) * 0.45,
        0.0, 100.0,
    ))
    command_score = float(np.clip(
        55.0
        + (league["BB_PA"] - pitcher["BB_PA"]) * 430.0
        + (pitcher["CSW_Pct"] - league["CSW"]) * 185.0
        + (league["BB_PA"] - log["Recent_BB_BF"]) * 185.0,
        0.0, 100.0,
    ))
    k_confidence_score = float(np.clip(
        0.28 * confidence
        + 0.22 * pitch_match_score
        + 0.18 * arsenal["Arsenal_Form_Score"]
        + 0.17 * (100.0 * (1.0 - math.exp(-max(opponent["Lineup_PA_Sample"], 0) / 240.0)))
        + 0.15 * command_score,
        0.0, 100.0,
    ))
    run_environment_risk = float(np.clip(
        50.0
        + (park_factor - 100.0) * 1.1
        + (weather_factor - 1.0) * 115.0
        + (opponent["Opp_xwOBA"] - league["xwOBA"]) * 235.0
        + (pitch_mix_xwoba - league["xwOBA"]) * 185.0,
        0.0, 100.0,
    ))
    projection_quality_score = float(np.clip(0.36 * k_confidence_score + 0.34 * workload_confidence + 0.30 * command_score, 0.0, 100.0))

    row = {
        "PitcherID": int(pitcher_id),
        "Pitcher": str(pitcher_name),
        "Team": pitcher_team,
        "Opponent": opponent_team,
        "Matchup": f"{game.get('away_abbr')} @ {game.get('home_abbr')}",
        "Game": f"{game.get('away_abbr')} @ {game.get('home_abbr')} | {game.get('time_et')}",
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
        "Rule_Proj_BF": rule_projected_bf,
        "Rule_K_Rate": rule_k_rate,
        "K_Model_Source": k_model_source,
        "Role_Status": role_status,
        "Role_Eligible": role_eligible,
        "Role_Note": role_note,
        "Projection_Eligible": bool(role_eligible and lineup_status == "Confirmed"),
        "Bet_Eligible": False,
        "Adj_K_Rate": adjusted_k_rate,
        "Pitcher_K_Rate": pitcher["K_PA"],
        "Opponent_K_Rate": opponent["Opp_K_PA"],
        "Top6_K_Rate": opponent["Top6_K_Pct"],
        "Bottom3_K_Rate": opponent["Bottom3_K_Pct"],
        "Lineup_Contact_Risk": opponent["Lineup_Contact_Risk"],
        "Whiff_Pct": pitcher["Whiff_Pct"],
        "CSW_Pct": pitcher["CSW_Pct"],
        "Recent_Whiff_Pct": arsenal["Recent_Whiff_Pct"],
        "Recent_CSW_Pct": arsenal["Recent_CSW_Pct"],
        "Fastball_Velo": arsenal["Fastball_Velo"],
        "Fastball_Velo_Trend": arsenal["Fastball_Velo_Trend"],
        "Pitch_Mix_Change": arsenal["Pitch_Mix_Change"],
        "Arsenal_Form_Score": arsenal["Arsenal_Form_Score"],
        "PitchTypeScore": pitch_match_score,
        "PitchMix_Opp_Whiff": pitch_mix_whiff,
        "PitchMix_Opp_xwOBA": pitch_mix_xwoba,
        "PitchMix_Damage_xwOBA": pitch_mix_damage,
        "xwOBA_Allowed": pitcher["xwOBA_Allowed"],
        "BB_Rate": pitcher["BB_PA"],
        "K_minus_BB": pitcher["K_PA"] - pitcher["BB_PA"],
        "Barrel_Allowed": pitcher["Brl_BBE_Allowed"],
        "HardHit_Allowed": pitcher["HH_BBE_Allowed"],
        "Opp_xwOBA": opponent["Opp_xwOBA"],
        "Opp_Barrel": opponent["Opp_Brl_BBE"],
        "Lineup_Zone_Fit": opponent["Lineup_Zone_Fit"],
        "Top_HR_Threat": opponent["Top_HR_Threat"],
        "Top_HR_Threat_Score": opponent["Top_HR_Threat_Score"],
        "Recent_Outs": log["Recent_Outs"],
        "Season_Pitches": log["Season_Pitches"],
        "Recent_Pitches": log["Recent_Pitches"],
        "Last_Start_Pitches": log["Last_Start_Pitches"],
        "Last3_Pitches": log["Last3_Pitches"],
        "Pitch_Count_Trend": log["Pitch_Count_Trend"],
        "Quality_Start_Rate": log["Quality_Start_Rate"],
        "Stable_Workload_Rate": log["Stable_Workload_Rate"],
        "Short_Hook_Rate": log["Short_Hook_Rate"],
        "Blowup_Hook_Rate": log["Blowup_Hook_Rate"],
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
        "K_Confidence_Score": k_confidence_score,
        "Command_Score": command_score,
        "Workload_Confidence_Score": workload_confidence,
        "Run_Environment_Risk": run_environment_risk,
        "Projection_Quality_Score": projection_quality_score,
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
        if target_name == "Strikeouts":
            result[projection_col] = result[raw_col]
            continue
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
        percentile(result["Proj_K"]) * 0.24
        + percentile(result["Adj_K_Rate"]) * 0.13
        + percentile(result["Whiff_Pct"]) * 0.09
        + percentile(result["CSW_Pct"]) * 0.08
        + percentile(result["Opponent_K_Rate"]) * 0.10
        + percentile(result.get("Top6_K_Rate", result["Opponent_K_Rate"])) * 0.06
        + result["PitchTypeScore"].clip(0, 100) * 0.08
        + percentile(result.get("PitchMix_Opp_Whiff", result["Opponent_K_Rate"])) * 0.05
        + percentile(result.get("Arsenal_Form_Score", pd.Series(50, index=result.index))) * 0.07
        + percentile(result["Proj_BF"]) * 0.06
        + percentile(result["Recent_K_Rate"]) * 0.04
    )
    result["Run_Prevention_Score"] = (
        percentile(result["Proj_ER"], higher_is_better=False) * 0.30
        + percentile(result["xwOBA_Allowed"], higher_is_better=False) * 0.15
        + percentile(result["K_minus_BB"]) * 0.10
        + percentile(result["Barrel_Allowed"], higher_is_better=False) * 0.10
        + percentile(result["HardHit_Allowed"], higher_is_better=False) * 0.08
        + percentile(result["Opp_xwOBA"], higher_is_better=False) * 0.12
        + percentile(result["Park_Run_Factor"], higher_is_better=False) * 0.06
        + percentile(result.get("Run_Environment_Risk", pd.Series(50, index=result.index)), higher_is_better=False) * 0.06
        + percentile(result.get("Command_Score", pd.Series(50, index=result.index))) * 0.06
        + percentile(result["Recent_ER9"], higher_is_better=False) * 0.05
    )
    result["Outs_Score"] = (
        percentile(result["Proj_Outs"]) * 0.45
        + percentile(result["Recent_Outs"]) * 0.20
        + percentile(result["Proj_Pitches"]) * 0.08
        + percentile(result.get("Workload_Confidence_Score", pd.Series(50, index=result.index))) * 0.14
        + percentile(result["BB_Rate"], higher_is_better=False) * 0.08
        + percentile(result["Opp_xwOBA"], higher_is_better=False) * 0.06
        + percentile(result["Starts"]) * 0.04
    )
    result["Overall_Score"] = 0.36 * result["K_Score"] + 0.30 * result["Run_Prevention_Score"] + 0.22 * result["Outs_Score"] + 0.12 * result.get("Projection_Quality_Score", pd.Series(50, index=result.index))
    risk_notes = []
    for _, row in result.iterrows():
        notes = []
        proj_k = pd.to_numeric(pd.Series([row.get("Proj_K")]), errors="coerce").iloc[0]
        proj_er = pd.to_numeric(pd.Series([row.get("Proj_ER")]), errors="coerce").iloc[0]
        proj_outs = pd.to_numeric(pd.Series([row.get("Proj_Outs")]), errors="coerce").iloc[0]
        if pd.notna(proj_k) and 5.0 <= proj_k <= 6.9:
            notes.append("K bucket has run high")
        if pd.notna(row.get("K_Score")) and float(row.get("K_Score")) >= 85:
            notes.append("High K-score volatility")
        if pd.notna(proj_er) and proj_er >= 3.5:
            notes.append("High ER range has run high")
        if pd.notna(proj_outs) and proj_outs >= 18:
            notes.append("High outs hook risk")
        if pd.notna(row.get("Workload_Confidence_Score")) and float(row.get("Workload_Confidence_Score")) < 50:
            notes.append("Workload caution")
        if pd.notna(row.get("Command_Score")) and float(row.get("Command_Score")) < 45:
            notes.append("Command risk")
        if not bool(row.get("Role_Eligible", False)):
            notes.append("Opener / role risk — no bet")
        if str(row.get("Lineup_Status", "")) != "Confirmed":
            notes.append("Lineup not confirmed")
        risk_notes.append("; ".join(notes) if notes else "Clean")
    result["Projection_Risk_Note"] = risk_notes
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
        "Adj_K_Rate", "Pitcher_K_Rate", "Opponent_K_Rate", "Top6_K_Rate", "Bottom3_K_Rate", "PitchMix_Opp_Whiff", "Whiff_Pct", "CSW_Pct", "Recent_Whiff_Pct", "Recent_CSW_Pct", "Fastball_Velo_Trend", "PitchTypeScore",
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
        "Outs_SD", "Workload_Confidence_Score", "Command_Score", "K_Confidence_Score", "Projection_Quality_Score", "Projection_Risk_Note", "Baseline_K", "Baseline_ER", "Baseline_Outs", "K_Score", "Run_Prevention_Score",
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
        if fit.get("ok") and target_name != "Strikeouts":
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


def _bucket_accuracy_summary(frame: pd.DataFrame, bucket_col: str, projection_col: str, actual_col: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    frame["Error"] = frame[projection_col] - frame[actual_col]
    frame["Abs_Error"] = frame["Error"].abs()
    frame["Squared_Error"] = np.square(frame["Error"])
    frame["Actual_Above_Projection"] = frame[actual_col] > frame[projection_col]
    grouped = frame.groupby(bucket_col, observed=False).agg(
        Sample=(actual_col, "size"),
        Average_Projection=(projection_col, "mean"),
        Average_Actual=(actual_col, "mean"),
        MAE=("Abs_Error", "mean"),
        RMSE=("Squared_Error", lambda values: float(np.sqrt(np.mean(values))) if len(values) else np.nan),
        Bias=("Error", "mean"),
        Actual_Above_Projection_Rate=("Actual_Above_Projection", "mean"),
    ).reset_index()
    return grouped


def projection_bucket_edges(target_name: str) -> tuple[list[float], list[str]]:
    if target_name == "Strikeouts":
        return [0, 3, 4, 5, 6, 7, 8, 30], ["0–2.9", "3.0–3.9", "4.0–4.9", "5.0–5.9", "6.0–6.9", "7.0–7.9", "8.0+"]
    if target_name == "Earned runs":
        return [0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 20], ["0.0–1.4", "1.5–1.9", "2.0–2.4", "2.5–2.9", "3.0–3.4", "3.5–3.9", "4.0+"]
    return [0, 12, 15, 18, 21, 24, 40], ["0–11", "12–14", "15–17", "18–20", "21–23", "24+"]


def model_projection_bucket_table(history: pd.DataFrame, target_name: str) -> pd.DataFrame:
    config = BACKTEST_TARGETS[target_name]
    projection_col, actual_col = config["projection"], config["actual"]
    required = [projection_col, actual_col]
    if any(column not in history.columns for column in required):
        return pd.DataFrame()
    frame = history[required].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required)
    if frame.empty:
        return frame
    bins, labels = projection_bucket_edges(target_name)
    frame["Projection_Bucket"] = pd.cut(frame[projection_col], bins=bins, labels=labels, include_lowest=True, right=False)
    return _bucket_accuracy_summary(frame, "Projection_Bucket", projection_col, actual_col)


def score_bucket_table(history: pd.DataFrame, target_name: str) -> pd.DataFrame:
    config = BACKTEST_TARGETS[target_name]
    score_col, projection_col, actual_col = config["score"], config["projection"], config["actual"]
    required = [score_col, projection_col, actual_col]
    if any(column not in history.columns for column in required):
        return pd.DataFrame()
    frame = history[required].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required)
    if frame.empty:
        return frame
    bins = [-0.001, 30, 45, 55, 70, 85, 100.001]
    labels = ["0–29", "30–44", "45–54", "55–69", "70–84", "85–100"]
    frame["Score_Bucket"] = pd.cut(frame[score_col], bins=bins, labels=labels, include_lowest=True, right=False)
    grouped = _bucket_accuracy_summary(frame, "Score_Bucket", projection_col, actual_col)
    if not grouped.empty:
        avg_score = frame.groupby("Score_Bucket", observed=False)[score_col].mean().reset_index(name="Average_Score")
        grouped = grouped.merge(avg_score, on="Score_Bucket", how="left")
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


# -----------------------------------------------------------------------------
# Table color system
# -----------------------------------------------------------------------------

FAVORABLE_BG = "background-color:#dcfce7;color:#14532d;font-weight:800;"
FAVORABLE_SOFT_BG = "background-color:#ecfdf5;color:#166534;font-weight:750;"
WARNING_BG = "background-color:#fef3c7;color:#92400e;font-weight:760;"
UNFAVORABLE_BG = "background-color:#fee2e2;color:#991b1b;font-weight:800;"
UNFAVORABLE_SOFT_BG = "background-color:#fff1f2;color:#9f1239;font-weight:750;"
POWER_BG = "background-color:#dbeafe;color:#1e3a8a;font-weight:820;"
NEUTRAL_BG = "background-color:#fffdf8;color:#162033;"
HEADER_BG = "background-color:#102033;color:#f8fafc;font-weight:900;"


def _column_styles_by_percentile(series: pd.Series, higher_is_better: bool = True) -> list[str]:
    """Red/yellow/green table colors where green always means favorable for the pitcher."""
    numeric = pd.to_numeric(series, errors="coerce")
    styles = [NEUTRAL_BG] * len(series)
    if numeric.notna().sum() < 2 or numeric.nunique(dropna=True) < 2:
        return styles
    ranks = numeric.rank(pct=True, method="average")
    if not higher_is_better:
        ranks = 1.0 - ranks
    output: list[str] = []
    for value, rank in zip(numeric, ranks):
        if pd.isna(value) or pd.isna(rank):
            output.append(NEUTRAL_BG)
        elif rank >= 0.80:
            output.append(FAVORABLE_BG)
        elif rank >= 0.62:
            output.append(FAVORABLE_SOFT_BG)
        elif rank <= 0.20:
            output.append(UNFAVORABLE_BG)
        elif rank <= 0.38:
            output.append(WARNING_BG)
        else:
            output.append(NEUTRAL_BG)
    return output


def _edge_styles(series: pd.Series, *, lower_is_better_target: bool = False) -> list[str]:
    """Style prop edges. For K/Outs, positive edge favors over; for ER, positive edge is run-risk."""
    numeric = pd.to_numeric(series, errors="coerce")
    styles: list[str] = []
    for value in numeric:
        if pd.isna(value):
            styles.append(NEUTRAL_BG)
            continue
        # ER edge above line means the pitcher is projected for more runs: risky for pitcher.
        if lower_is_better_target:
            if value <= -0.50:
                styles.append(FAVORABLE_BG)
            elif value < 0:
                styles.append(FAVORABLE_SOFT_BG)
            elif value >= 0.75:
                styles.append(UNFAVORABLE_BG)
            elif value > 0:
                styles.append(WARNING_BG)
            else:
                styles.append(NEUTRAL_BG)
        else:
            if value >= 1.00:
                styles.append(FAVORABLE_BG)
            elif value > 0:
                styles.append(FAVORABLE_SOFT_BG)
            elif value <= -1.00:
                styles.append(UNFAVORABLE_BG)
            elif value < 0:
                styles.append(WARNING_BG)
            else:
                styles.append(NEUTRAL_BG)
    return styles


def _risk_note_styles(series: pd.Series) -> list[str]:
    styles: list[str] = []
    for value in series.fillna("").astype(str):
        text = value.lower()
        if text == "clean" or text.strip() == "":
            styles.append(FAVORABLE_SOFT_BG)
        elif any(token in text for token in ["high", "risk", "caution", "hook", "command", "volatility"]):
            styles.append(UNFAVORABLE_SOFT_BG)
        else:
            styles.append(WARNING_BG)
    return styles


def style_board(frame: pd.DataFrame, score_columns: list[str], lower_better: list[str] | None = None):
    """Pitcher-friendly table styling.

    Green always means favorable for the pitcher or for the displayed prop side.
    Red always means unfavorable / contact / hit / HR / run-risk / short-hook risk.
    This avoids matplotlib-dependent gradients and works on Streamlit Cloud.
    """
    lower_better = lower_better or []
    styler = frame.style.set_table_styles([
        {"selector": "th", "props": [("background-color", "#102033"), ("color", "#f8fafc"), ("font-weight", "900"), ("border", "1px solid #d9cdb9")]},
        {"selector": "td", "props": [("background-color", "#fffdf8"), ("color", "#162033"), ("border", "1px solid #efe4d3")]},
        {"selector": "tbody tr:hover td", "props": [("background-color", "#eef8f7")]},
    ])

    # Projection/score/signal columns: higher values are good unless explicitly passed as lower_better.
    for column in score_columns:
        if column in frame.columns:
            styler = styler.apply(
                lambda s, col=column: _column_styles_by_percentile(s, higher_is_better=col not in lower_better),
                subset=[column],
                axis=0,
            )

    # Risk columns: lower is favorable for the pitcher. These are hit/HR/run/contact risk signals.
    for column in lower_better:
        if column in frame.columns and column not in score_columns:
            styler = styler.apply(
                lambda s: _column_styles_by_percentile(s, higher_is_better=False),
                subset=[column],
                axis=0,
            )

    # Projection edge columns get explicit red/green side logic.
    for column in frame.columns:
        col_text = str(column).lower()
        if "edge" in col_text:
            er_like = col_text.startswith("er") or "earned" in col_text or "run" in col_text
            styler = styler.apply(
                lambda s, er_like=er_like: _edge_styles(s, lower_is_better_target=er_like),
                subset=[column],
                axis=0,
            )
        elif "risk note" in col_text or "projection_risk_note" in col_text:
            styler = styler.apply(_risk_note_styles, subset=[column], axis=0)

    return styler


def render_color_legend(location: str = "dashboard") -> None:
    """Render a compact legend explaining the table colors."""
    st.markdown(
        f"""
        <div class="legend-card">
            <div class="legend-title">Table Color Key | {escape(str(location).title())}</div>
            <div class="legend-grid">
                <div class="legend-item"><span class="legend-swatch favorable"></span><b>Green</b><span>Favorable for pitcher / strong support</span></div>
                <div class="legend-item"><span class="legend-swatch warning"></span><b>Yellow</b><span>Caution or mixed signal</span></div>
                <div class="legend-item"><span class="legend-swatch unfavorable"></span><b>Red</b><span>Unfavorable: hit, HR, run, walk, contact, or hook risk</span></div>
                <div class="legend-item"><span class="legend-swatch blue"></span><b>Blue</b><span>Projection/score strength or model power signal</span></div>
            </div>
            <div class="legend-note">
                For K/Outs tables, higher green values usually support pitcher overs. For ER/run-risk tables, red means more damage risk; green means run prevention.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Sleek Edge theme overlay
# -----------------------------------------------------------------------------
def inject_edge_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg:#f6f1e9;
            --panel:#fffdf8;
            --panel-2:#fff8ee;
            --ink:#162033;
            --muted:#6b7280;
            --line:#eadfce;
            --nav:#102033;
            --nav-2:#17324d;
            --blue:#2563eb;
            --teal:#0f9f9a;
            --green:#15803d;
            --orange:#d97706;
            --red:#dc2626;
            --purple:#7c3aed;
            --shadow:0 10px 28px rgba(38,28,12,.08);
        }
        html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background:
                radial-gradient(circle at 7% 2%, rgba(37,99,235,.10), transparent 28rem),
                radial-gradient(circle at 94% 6%, rgba(15,159,154,.11), transparent 26rem),
                linear-gradient(180deg, #faf7f0 0%, var(--bg) 46%, #fbfaf7 100%) !important;
            color:var(--ink) !important;
        }
        [data-testid="stHeader"] { background:rgba(250,247,240,.88) !important; backdrop-filter:blur(12px); }
        .block-container { max-width:1840px; padding-top:.85rem; padding-bottom:2.6rem; }
        h1,h2,h3,h4,p,span,label,div { color:inherit; }
        h1 { letter-spacing:-.045em; font-weight:950 !important; color:#101b2d !important; }
        h2,h3 { letter-spacing:-.025em; font-weight:875 !important; color:#101b2d !important; }
        .app-kicker {
            display:inline-flex; align-items:center; gap:.45rem; padding:.34rem .72rem;
            border-radius:999px; background:linear-gradient(90deg,var(--teal),var(--blue));
            color:white !important; font-size:.72rem; font-weight:900; letter-spacing:.10em;
            text-transform:uppercase; box-shadow:0 8px 18px rgba(15,159,154,.20); margin-bottom:.35rem;
        }
        .edge-topbar {
            display:flex; align-items:center; justify-content:space-between; gap:1rem;
            background:linear-gradient(90deg,var(--nav) 0%,var(--nav-2) 100%);
            border:1px solid rgba(255,255,255,.10); border-radius:18px; padding:.68rem .82rem;
            box-shadow:var(--shadow); margin:.1rem 0 1rem;
        }
        .edge-brand { display:flex; align-items:center; gap:.55rem; color:white !important; font-weight:950; letter-spacing:-.02em; font-size:1.08rem; }
        .edge-brand .ball { width:26px; height:26px; border-radius:999px; display:grid; place-items:center; background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.22); font-size:.9rem; }
        .edge-nav { display:flex; gap:.35rem; flex-wrap:wrap; }
        .edge-pill { color:#e5eefc !important; font-weight:750; font-size:.82rem; padding:.42rem .7rem; border-radius:999px; background:rgba(255,255,255,.055); border:1px solid rgba(255,255,255,.08); }
        .edge-pill.active { background:#ffffff; color:var(--blue) !important; }
        .edge-status { color:#dbeafe !important; font-size:.78rem; font-weight:650; }
        div[data-testid="stMetric"] {
            background:linear-gradient(145deg,var(--panel) 0%, #fef7ea 100%) !important;
            border:1px solid var(--line) !important; border-radius:18px !important; padding:.9rem 1rem !important;
            box-shadow:var(--shadow) !important;
        }
        div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color:#42526b !important; font-weight:800 !important; }
        div[data-testid="stMetricValue"] { color:#0f3460 !important; font-weight:950 !important; }
        div[data-testid="stMetricDelta"] { color:var(--green) !important; font-weight:850 !important; }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) { gap:.8rem; }
        .stButton > button, .stDownloadButton > button {
            border-radius:12px !important; border:1px solid #d9cdb9 !important;
            background:#fffdf8 !important; color:var(--ink) !important; font-weight:800 !important;
            box-shadow:0 5px 14px rgba(38,28,12,.06) !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--teal) !important; color:var(--teal) !important; transform:translateY(-1px); }
        .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background:linear-gradient(180deg,#2563eb,#1d4ed8) !important; color:#fff !important; border-color:#2563eb !important; }
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stDateInput"] input,
        div[data-testid="stTextInput"] input,
        textarea {
            background:#fffdf8 !important; color:var(--ink) !important; border:1px solid #d9cdb9 !important; border-radius:12px !important;
        }
        div[data-testid="stNumberInput"] button { background:#f3eadc !important; color:var(--ink) !important; border-color:#d9cdb9 !important; border-radius:10px !important; }
        [data-testid="stSidebar"] { background:linear-gradient(180deg,#102033 0%,#17324d 62%,#102033 100%) !important; border-right:1px solid rgba(255,255,255,.08) !important; }
        [data-testid="stSidebar"] * { color:#edf6ff !important; }
        [data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea { background:#fffdf8 !important; color:#162033 !important; }
        [data-testid="stSidebar"] .stButton > button { background:rgba(255,255,255,.08) !important; color:#fff !important; border-color:rgba(255,255,255,.16) !important; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.45rem; border-bottom:1px solid #e4d7c4; }
        [data-baseweb="tab"] { height:42px !important; border-radius:12px 12px 0 0 !important; background:#fff8ee !important; color:#475569 !important; font-weight:850 !important; border:1px solid #eadfce !important; border-bottom:0 !important; }
        [aria-selected="true"][data-baseweb="tab"] { background:#fffdf8 !important; color:var(--teal) !important; box-shadow:inset 0 -3px 0 var(--teal) !important; }
        .hero, .leader, .note, .slate-card, .info-card, .soft-card {
            background:linear-gradient(145deg,#fffdf8,#fff7ea) !important; border:1px solid var(--line) !important;
            border-radius:18px !important; box-shadow:var(--shadow) !important; color:var(--ink) !important;
        }
        .hero h2, .hero h1, .hero p { color:var(--ink) !important; }
        .leader .label, .slate-time, .muted { color:var(--muted) !important; }
        .leader .name, .leader .value { color:#0f3460 !important; }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border:1px solid var(--line) !important; border-radius:15px !important; overflow:hidden !important;
            background:#fffdf8 !important; box-shadow:0 7px 20px rgba(38,28,12,.045) !important;
        }
        [data-testid="stDataFrame"] *, [data-testid="stTable"] * { color:#162033 !important; }
        [data-testid="stDataFrame"] [role="gridcell"],
        [data-testid="stDataFrame"] [role="columnheader"],
        [data-testid="stDataFrame"] [role="rowheader"] {
            background:#fffdf8 !important; color:#162033 !important; border-color:#efe4d3 !important;
        }
        [data-testid="stDataFrame"] [role="columnheader"], [data-testid="stTable"] thead tr th {
            background:#f5eadb !important; color:#42526b !important; font-weight:900 !important;
        }
        [data-testid="stTable"] tbody tr:nth-child(even) { background:#fbf6ec !important; }
        [data-testid="stTable"] tbody tr:hover, [data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"] { background:#eef8f7 !important; }
        .streamlit-expanderHeader { background:#fffdf8 !important; border:1px solid var(--line) !important; border-radius:14px !important; font-weight:850 !important; color:var(--ink) !important; }
        .stAlert { border-radius:14px !important; }
        .legend-card {
            background:linear-gradient(145deg,#fffdf8,#fff7ea); border:1px solid #eadfce;
            border-radius:18px; padding:.85rem 1rem; box-shadow:0 8px 22px rgba(38,28,12,.07);
            margin:.45rem 0 1rem;
        }
        .legend-title { font-weight:950; color:#102033; margin-bottom:.55rem; letter-spacing:-.01em; }
        .legend-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.55rem; }
        .legend-item { display:flex; align-items:center; gap:.45rem; padding:.48rem .55rem; border-radius:12px; background:#ffffff; border:1px solid #efe4d3; font-size:.82rem; }
        .legend-item span:last-child { color:#64748b; font-size:.75rem; }
        .legend-swatch { width:18px; height:18px; border-radius:6px; display:inline-block; border:1px solid rgba(15,23,42,.10); flex:0 0 auto; }
        .legend-swatch.favorable { background:#22c55e; }
        .legend-swatch.warning { background:#f59e0b; }
        .legend-swatch.unfavorable { background:#ef4444; }
        .legend-swatch.blue { background:#3b82f6; }
        .legend-note { margin-top:.5rem; color:#64748b; font-size:.78rem; }
        @media (max-width: 1100px) { .legend-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
        hr { border-color:#e6dac7 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_edge_header() -> None:
    st.markdown(
        """
        <div class="edge-topbar">
            <div class="edge-brand">
                <div class="edge-logo">⚾</div>
                <div>
                    <div class="edge-brand-title">PITCH EDGE</div>
                    <div class="edge-brand-subtitle">Pitcher prop projections & model accuracy</div>
                </div>
            </div>
            <div class="edge-nav">
                <span class="edge-nav-item active">Dashboard</span><span class="edge-nav-item">Matchups</span><span class="edge-nav-item">Pitchers</span><span class="edge-nav-item">Lineups</span><span class="edge-nav-item">Park Factors</span><span class="edge-nav-item">Backtest</span><span class="edge-nav-item">Settings</span>
            </div>
            <div class="edge-meta">
                <span>☾</span>
                <span>Data updated: session live</span>
                <span class="edge-dot"></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

inject_css()
inject_edge_theme()
render_edge_header()
st.caption('Sleek model dashboard for strikeouts, earned runs, outs, score buckets, and projection accuracy.')
render_color_legend('pitcher dashboard')
st.caption("Projected strikeouts, earned runs, outs, opponent-lineup fit, pitch-type matchups and 0–100 category scores.")

with st.sidebar:
    st.header("Pitcher slate")
    slate_date = st.date_input("Slate date", value=date.today(), key="pitcher_slate_date")
    lookback_days = st.slider("Statcast lookback days", 21, 90, 45, 3)
    min_starts = st.slider("Minimum starts for full confidence", 1, 8, 3)
    include_tbd = st.checkbox("Show games with one probable pitcher TBD", value=False)
    st.subheader("Two-stage K model")
    k_model_upload = st.file_uploader("Optional trained K model JSON", type=["json"], key="pitcher_k_model_upload")
    uploaded_k_model: dict = {}
    if k_model_upload is not None:
        try:
            uploaded_k_model = parse_two_stage_k_model(json.loads(k_model_upload.getvalue().decode("utf-8-sig")))
            if uploaded_k_model:
                st.success(f"Two-stage K model loaded ({uploaded_k_model.get('training_rows', 0)} training starts).")
            else:
                st.warning("That JSON is not a compatible two-stage K model.")
        except Exception as exc:
            st.warning(f"K model could not be read: {type(exc).__name__}: {exc}")
    session_k_model = parse_two_stage_k_model(st.session_state.get("two_stage_k_model", {}))
    active_k_model = uploaded_k_model or session_k_model
    market_model_weight = st.slider(
        "Independent-model weight when K line exists", 0.0, 1.0, 0.35, 0.05,
        help="Final K = this weight × independent model + remaining weight × market line.",
    )
    minimum_bet_edge = st.slider("Minimum blended K edge", 0.25, 2.0, 0.75, 0.25)
    st.caption("K linear calibration is disabled because it did not improve the chronological holdout sample.")
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
            <h2>{len(schedule)} games | {probable_count} probable starters</h2>
            <p>{escape(str(slate_date))} | MLB schedule, posted lineups, park context and first-pitch weather.</p>
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
                        lineup, lineup_status, weather, int(pd.Timestamp(slate_date).year), active_k_model,
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

with st.expander("Historical K backfill & two-stage training", expanded=False):
    st.markdown("### Chronological Statcast backfill")
    st.caption(
        "Every training row uses only games played before that start. The first 60 days are warmup history, "
        "not training targets. Download both the backfill CSV and trained JSON because Streamlit session storage is temporary."
    )
    default_backfill_end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    default_backfill_start = max(pd.Timestamp(f"{default_backfill_end.year}-04-01"), default_backfill_end - pd.Timedelta(days=150))
    backfill_cols = st.columns(2)
    with backfill_cols[0]:
        backfill_start = st.date_input("Backfill target start", value=default_backfill_start.date(), key="k_backfill_start")
    with backfill_cols[1]:
        backfill_end = st.date_input("Backfill target end", value=default_backfill_end.date(), key="k_backfill_end")
    build_backfill = st.button("Build chronological K backfill", type="primary", key="build_k_backfill", width="stretch")
    if build_backfill:
        target_start = pd.Timestamp(backfill_start)
        target_end = pd.Timestamp(backfill_end)
        if target_end < target_start:
            st.error("Backfill end must be on or after the start date.")
        elif (target_end - target_start).days > 200:
            st.error("Build at most 200 target days at once to keep a single-file Streamlit run manageable.")
        else:
            warmup_start = target_start - pd.Timedelta(days=60)
            with st.spinner(f"Loading Statcast {warmup_start.date()} through {target_end.date()} and building leakage-safe starts..."):
                try:
                    historical_raw = load_statcast(str(warmup_start.date()), str(target_end.date()))
                    historical_prepared = prepare_statcast(historical_raw)
                    k_backfill = build_chronological_k_backfill(historical_prepared, target_start, target_end)
                    st.session_state["k_backfill"] = k_backfill
                except Exception as exc:
                    st.error(f"Backfill failed: {type(exc).__name__}: {exc}")
    uploaded_backfill = st.file_uploader("Or upload an existing K backfill CSV", type=["csv"], key="k_backfill_upload")
    if uploaded_backfill is not None:
        try:
            uploaded_frame = pd.read_csv(StringIO(uploaded_backfill.getvalue().decode("utf-8-sig")))
            if not uploaded_frame.empty:
                st.session_state["k_backfill"] = uploaded_frame
        except Exception as exc:
            st.warning(f"Backfill CSV could not be read: {type(exc).__name__}: {exc}")
    k_backfill = st.session_state.get("k_backfill", pd.DataFrame())
    if k_backfill is not None and not k_backfill.empty:
        b1, b2, b3 = st.columns(3)
        b1.metric("Training candidates", f"{len(k_backfill):,}")
        b2.metric("Pitchers", f"{k_backfill['PitcherID'].nunique():,}" if "PitcherID" in k_backfill else "—")
        b3.metric("Date range", f"{k_backfill['Date'].min()} to {k_backfill['Date'].max()}" if "Date" in k_backfill else "—")
        st.download_button(
            "Download chronological K backfill CSV", k_backfill.to_csv(index=False).encode("utf-8"),
            file_name="pitcher_k_chronological_backfill.csv", mime="text/csv", width="stretch",
        )
        ridge_alpha = st.slider("Model shrinkage", 1.0, 30.0, 8.0, 1.0, key="k_ridge_alpha")
        if st.button("Train two-stage K model", key="train_two_stage_k", width="stretch"):
            trained_model, training_metrics = train_two_stage_k_model(k_backfill, ridge_alpha)
            st.session_state["k_training_metrics"] = training_metrics
            if trained_model:
                st.session_state["two_stage_k_model"] = trained_model
                st.success("Two-stage model trained and activated. Rebuild the pitcher board to use it.")
            else:
                st.error(training_metrics.get("message", "Model training failed."))
        training_metrics = st.session_state.get("k_training_metrics", {})
        if training_metrics.get("ok"):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Training starts", training_metrics["training_n"])
            m2.metric("Holdout starts", training_metrics["holdout_n"])
            m3.metric("Holdout MAE", f"{training_metrics['holdout_mae']:.3f}")
            m4.metric("Baseline MAE", f"{training_metrics['holdout_baseline_mae']:.3f}")
            if training_metrics["holdout_mae"] >= training_metrics["holdout_baseline_mae"]:
                st.warning("This trained model did not beat the baseline on the later holdout. Do not use it for betting yet.")
            trained_model = parse_two_stage_k_model(st.session_state.get("two_stage_k_model", {}))
            if trained_model:
                st.download_button(
                    "Download trained K model JSON", json.dumps(trained_model, indent=2).encode("utf-8"),
                    file_name="pitcher_two_stage_k_model.json", mime="application/json", width="stretch",
                )

base_board = st.session_state.get("pitcher_board", pd.DataFrame())
details = st.session_state.get("pitcher_details", {})

new_board_columns = {"K_Model_Source", "Role_Status", "Role_Eligible", "Rule_K_Rate"}
if not base_board.empty and not new_board_columns.issubset(base_board.columns):
    st.warning(
        "This session still contains a pitcher board built by an older dashboard version. "
        "Press **Build / refresh pitcher board** once to calculate the new two-stage K and role fields."
    )

if base_board.empty:
    st.info("Choose a slate date and press **Build / refresh pitcher board**.")
    st.stop()

loaded_quotes = st.session_state.get("pitcher_odds_quotes", pd.DataFrame())
board = apply_odds_quotes_to_board(
    base_board, loaded_quotes, odds_source_mode, market_model_weight, minimum_bet_edge
) if loaded_quotes is not None and not loaded_quotes.empty else base_board.copy()

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
calibration_note = " | Saved calibration applied" if bool(board.get("Calibration_Applied", pd.Series(False, index=board.index)).any()) else ""
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
    "Pitcher board", "Strikeouts", "Earned runs", "Outs / innings", "Pitcher vs opponent",
    "Pitch-type matchup", "Recent starts", "Line comparison", "Backtest & calibration", "Model notes",
])

with board_tab:
    render_color_legend('main board')
    columns = [
        "Rank", "Pitcher", "Team", "Opponent", "Game", "Proj_K", "Proj_ER", "Proj_Outs",
        "Top_HR_Threat", "Top_HR_Threat_Score", "Lineup_Zone_Fit",
        "K_Model_Source", "Role_Status", "Bet_Eligible",
        "K_Score", "Run_Prevention_Score", "Outs_Score", "Overall_Score", "Confidence_Level", "Lineup_Status",
    ]
    automatic_columns = [
        "K_Line", "K_Projection_Edge", "ER_Line", "ER_Projection_Edge",
        "Outs_Line", "Outs_Projection_Edge", "K_Line_Source", "Final_Proj_K", "Blended_K_Edge", "K_Bet_Signal",
    ]
    for automatic_column in automatic_columns:
        if automatic_column in board.columns:
            values = board[automatic_column]
            if values.astype(str).str.strip().ne("").any() and not (pd.api.types.is_numeric_dtype(values) and values.isna().all()):
                columns.append(automatic_column)
    columns = [column for column in columns if column in board.columns]
    display = board[columns].copy().rename(
        columns={
            "Proj_K": "Proj K", "Proj_ER": "Proj ER", "Proj_Outs": "Proj Outs",
            "Top_HR_Threat": "Top HR Threat", "Top_HR_Threat_Score": "Threat Score",
            "Lineup_Zone_Fit": "Lineup Zone Fit",
            "Final_Proj_K": "Final K", "Blended_K_Edge": "Blended K Edge", "K_Bet_Signal": "K Signal",
            "K_Score": "K Score", "Run_Prevention_Score": "Run Prevention",
            "Outs_Score": "Outs Score", "Overall_Score": "Overall", "Confidence_Level": "Confidence",
            "Lineup_Status": "Lineup", "K_Line": "K Line", "K_Projection_Edge": "K Edge",
            "ER_Line": "ER Line", "ER_Projection_Edge": "ER Edge", "Outs_Line": "Outs Line",
            "Outs_Projection_Edge": "Outs Edge", "K_Line_Source": "Line Source",
        }
    )
    styled = style_board(display, ["Proj K", "K Score", "Run Prevention", "Outs Score", "Overall"], ["Proj ER", "Threat Score", "Lineup Zone Fit"])
    formatters = {"Proj K": "{:.2f}", "Proj ER": "{:.2f}", "Proj Outs": "{:.1f}", "Threat Score": "{:.0f}", "Lineup Zone Fit": "{:.0f}", "K Score": "{:.0f}", "Run Prevention": "{:.0f}", "Outs Score": "{:.0f}", "Overall": "{:.0f}", "K Conf": "{:.0f}", "Workload": "{:.0f}", "Command": "{:.0f}"}
    for column in ["K Line", "K Edge", "Final K", "Blended K Edge", "ER Line", "ER Edge", "Outs Line", "Outs Edge"]:
        if column in display.columns:
            formatters[column] = "{:.2f}"
    formatters = {column: formatter for column, formatter in formatters.items() if column in display.columns}
    styled = styled.format(formatters)
    st.dataframe(styled, width="stretch", hide_index=True, height=650)

with k_tab:
    render_color_legend('strikeouts')
    columns = ["K_Rank", "Pitcher", "Opponent", "Proj_K", "Proj_BF", "Adj_K_Rate", "Pitcher_K_Rate", "Opponent_K_Rate", "Top6_K_Rate", "Bottom3_K_Rate", "PitchMix_Opp_Whiff", "Whiff_Pct", "CSW_Pct", "Recent_Whiff_Pct", "Recent_CSW_Pct", "Fastball_Velo_Trend", "PitchTypeScore", "K_Model_Source", "Role_Status", "Role_Note", "Bet_Eligible", "K_Score", "Confidence_Level"]
    columns = [column for column in columns if column in board.columns]
    display = board.sort_values("Proj_K", ascending=False)[columns].copy().rename(columns={
        "K_Rank": "Rank", "Proj_K": "Proj K", "Proj_BF": "Proj BF", "Adj_K_Rate": "Game K%",
        "Pitcher_K_Rate": "Pitcher K%", "Opponent_K_Rate": "Opponent K%", "Whiff_Pct": "Whiff%",
        "CSW_Pct": "CSW%", "Top6_K_Rate": "Top6 K%", "PitchMix_Opp_Whiff": "Pitch-Mix Whiff%",
        "Arsenal_Form_Score": "Arsenal Form", "K_Confidence_Score": "K Conf",
        "PitchTypeScore": "Pitch Match", "K_Score": "K Score", "Projection_Risk_Note": "Risk Note", "Confidence_Level": "Confidence",
    })
    styled = style_board(display, ["Proj K", "Game K%", "Pitcher K%", "Opponent K%", "Top6 K%", "Pitch-Mix Whiff%", "Whiff%", "CSW%", "Arsenal Form", "K Conf", "Pitch Match", "K Score"])
    k_formatters = {"Proj K": "{:.2f}", "Proj BF": "{:.1f}", "Game K%": "{:.1%}", "Pitcher K%": "{:.1%}", "Opponent K%": "{:.1%}", "Top6 K%": "{:.1%}", "Pitch-Mix Whiff%": "{:.1%}", "Whiff%": "{:.1%}", "CSW%": "{:.1%}", "Arsenal Form": "{:.0f}", "K Conf": "{:.0f}", "Pitch Match": "{:.0f}", "K Score": "{:.0f}"}
    styled = styled.format({column: formatter for column, formatter in k_formatters.items() if column in display.columns})
    st.dataframe(styled, width="stretch", hide_index=True, height=650)

with er_tab:
    render_color_legend('earned runs / damage risk')
    columns = ["ER_Rank", "Pitcher", "Opponent", "Proj_ER", "P_0_1_ER", "P_2_3_ER", "P_4plus_ER", "xwOBA_Allowed", "BB_Rate", "Command_Score", "Barrel_Allowed", "PitchMix_Opp_xwOBA", "Opp_xwOBA", "Run_Environment_Risk", "Park_Run_Factor", "Weather_Factor", "Run_Prevention_Score", "Projection_Risk_Note"]
    display = board.sort_values("Proj_ER")[columns].copy().rename(columns={
        "ER_Rank": "Rank", "Proj_ER": "Proj ER", "P_0_1_ER": "0–1 ER", "P_2_3_ER": "2–3 ER", "P_4plus_ER": "4+ ER",
        "xwOBA_Allowed": "xwOBA Allowed", "BB_Rate": "BB%", "Barrel_Allowed": "Barrel% Allowed",
        "Opp_xwOBA": "Opp xwOBA", "Park_Run_Factor": "Park", "Weather_Factor": "Weather", "Run_Prevention_Score": "Run Prevention",
    })
    styled = style_board(display, ["0–1 ER", "Run Prevention"], ["Proj ER", "4+ ER", "xwOBA Allowed", "BB%", "Barrel% Allowed", "Opp xwOBA", "Park", "Weather"])
    styled = styled.format({"Proj ER": "{:.2f}", "0–1 ER": "{:.1%}", "2–3 ER": "{:.1%}", "4+ ER": "{:.1%}", "xwOBA Allowed": "{:.3f}", "BB%": "{:.1%}", "Barrel% Allowed": "{:.1%}", "Opp xwOBA": "{:.3f}", "Park": "{:.0f}", "Weather": "{:.3f}", "Run Prevention": "{:.0f}"})
    st.dataframe(styled, width="stretch", hide_index=True, height=650)

with outs_tab:
    render_color_legend('outs / workload')
    columns = ["Outs_Rank", "Pitcher", "Opponent", "Proj_Outs", "Proj_Innings", "Proj_BF", "Proj_Pitches", "Last_Start_Pitches", "Pitch_Count_Trend", "Recent_Outs", "Workload_Confidence_Score", "Short_Hook_Rate", "BB_Rate", "Outs_Score", "Projection_Risk_Note", "Confidence_Level"]
    display = board.sort_values("Proj_Outs", ascending=False)[columns].copy().rename(columns={
        "Outs_Rank": "Rank", "Proj_Outs": "Proj Outs", "Proj_Innings": "Proj IP", "Proj_BF": "Proj BF",
        "Proj_Pitches": "Proj Pitches", "Last_Start_Pitches": "Last Start Pitches", "Pitch_Count_Trend": "Pitch Trend", "Recent_Outs": "Recent Outs",
        "Workload_Confidence_Score": "Workload", "Short_Hook_Rate": "Short Hook%", "BB_Rate": "BB%", "Outs_Score": "Outs Score", "Projection_Risk_Note": "Risk Note", "Confidence_Level": "Confidence",
    })
    styled = style_board(display, ["Proj Outs", "Proj IP", "Proj BF", "Proj Pitches", "Last Start Pitches", "Recent Outs", "Workload", "Outs Score"], ["Pitch Trend", "Short Hook%", "BB%"])
    styled = styled.format({"Proj Outs": "{:.1f}", "Proj IP": "{:.2f}", "Proj BF": "{:.1f}", "Proj Pitches": "{:.0f}", "Last Start Pitches": "{:.0f}", "Pitch Trend": "{:+.1f}", "Recent Outs": "{:.1f}", "Workload": "{:.0f}", "Short Hook%": "{:.1%}", "BB%": "{:.1%}", "Outs Score": "{:.0f}"})
    st.dataframe(styled, width="stretch", hide_index=True, height=650)

pitcher_options = board["Pitcher"].tolist()

with lineup_tab:
    render_color_legend('pitcher vs opponent')
    selected = st.selectbox("Pitcher", pitcher_options, key="lineup_pitcher")
    row = board[board["Pitcher"].eq(selected)].iloc[0]
    detail = details.get(row["DetailKey"], {})
    st.markdown(f"### {selected} vs {row['Opponent']} | {row['Lineup_Status']} lineup")
    lineup_frame = detail.get("lineup", pd.DataFrame())
    if lineup_frame is None or lineup_frame.empty:
        st.info("No opponent lineup profile was available.")
    elif not {"HR_Threat_Score", "HR_Threat", "Zone_Fit_Score", "Matched_HH_Pct"}.issubset(lineup_frame.columns):
        st.info("This lineup was cached by an older version. Rebuild the pitcher board to calculate zone fit and HR threat.")
        display = lineup_frame.rename(columns={"LineupSpot": "Order", "EffectiveStand": "Bats", "K_Pct": "K%", "BB_Pct": "BB%", "HR_Pct": "HR%", "Brl_BBE": "Barrel%", "HH_BBE": "Hard Hit%", "Whiff_Pct": "Whiff%"})
        st.dataframe(display, width="stretch", hide_index=True)
    else:
        ranked = lineup_frame.sort_values("HR_Threat_Score", ascending=False)
        top = ranked.iloc[0]
        summary_cols = st.columns(4)
        summary_cols[0].metric("Top HR threat", str(top["Player"]), str(top["HR_Threat"]))
        summary_cols[1].metric("Threat score", f"{top['HR_Threat_Score']:.0f}/100")
        summary_cols[2].metric("Zone fit", f"{top['Zone_Fit_Score']:.0f}/100")
        summary_cols[3].metric("Matched hard hit", f"{top['Matched_HH_Pct']:.1%}")
        st.warning(
            f"Top matchup alert: {top['Player']} combines a {top['Matched_HH_Pct']:.1%} "
            f"pitcher-location-weighted hard-hit rate with a {top['Zone_Fit_Score']:.0f}/100 zone fit."
        )
        display = lineup_frame.rename(columns={"LineupSpot": "Order", "EffectiveStand": "Bats", "K_Pct": "K%", "BB_Pct": "BB%", "HR_Pct": "HR%", "Brl_BBE": "Barrel%", "HH_BBE": "Hard Hit%", "Whiff_Pct": "Whiff%", "Matched_HH_Pct": "Matched Hard Hit%", "Pitcher_Zone_HH_Allowed": "Pitcher Zone HH%", "Zone_Fit_Score": "Zone Fit", "HR_Threat_Score": "HR Threat Score", "HR_Threat": "HR Threat", "Matchup_Flag": "Alert", "Zone_BBE": "Zone BBE"})
        preferred = ["Order", "Player", "Position", "Bats", "Alert", "HR Threat", "HR Threat Score", "Zone Fit", "Matched Hard Hit%", "Pitcher Zone HH%", "Zone BBE", "Hard Hit%", "Barrel%", "HR%", "xwOBA", "K%", "Whiff%", "PA"]
        display = display[[column for column in preferred if column in display.columns]]
        styled = style_board(display, ["K%", "Whiff%"], ["HR Threat Score", "Zone Fit", "Matched Hard Hit%", "Pitcher Zone HH%", "HR%", "xwOBA", "Barrel%", "Hard Hit%"])
        lineup_formatters = {"Order": "{:.0f}", "PA": "{:.0f}", "Zone BBE": "{:.0f}", "K%": "{:.1%}", "BB%": "{:.1%}", "HR%": "{:.1%}", "xwOBA": "{:.3f}", "Barrel%": "{:.1%}", "Hard Hit%": "{:.1%}", "Matched Hard Hit%": "{:.1%}", "Pitcher Zone HH%": "{:.1%}", "Zone Fit": "{:.0f}", "HR Threat Score": "{:.0f}", "Whiff%": "{:.1%}"}
        styled = styled.format({column: formatter for column, formatter in lineup_formatters.items() if column in display.columns})
        st.dataframe(styled, width="stretch", hide_index=True)
        st.caption("Zone Fit measures overlap between each hitter's hard-contact zones versus this pitcher's handedness and the selected pitcher's actual zone usage and hard-hit locations. HR Threat is a matchup score, not a projected home-run probability.")

with pitch_tab:
    render_color_legend('pitch-type matchup')
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
    st.markdown(f"### {selected} | {row['Game']}")
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
                f"Auto source: **{source}** | Over {format_american_odds(over_odds)} | "
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
        "Use the Backtest & calibration tab to measure projection accuracy first; prop-line probability calibration is optional."
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
            f"Result update: {result_summary.get('matched', 0)} rows matched | "
            f"{result_summary.get('not_final', 0)} games not final | "
            f"{result_summary.get('unmatched', 0)} unmatched/scratched pitchers"
            + (
                f" | {result_summary.get('fallback_matched', 0)} date-fallback matches"
                if result_summary.get('fallback_matched', 0)
                else ""
            )
            + (
                f" | {result_summary.get('missing_game_pk', 0)} rows missing GamePK"
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
            st.markdown("### Model projection accuracy — no prop lines required")
            st.caption(
                "This section compares each model projection directly to the official result. "
                "Rows are included even when K/ER/Outs prop lines are blank."
            )
            metric_rows: list[dict] = []
            for target_name, config in BACKTEST_TARGETS.items():
                metrics = projection_metrics(completed_history, config["projection"], config["actual"], config["baseline"])
                baseline_mae = metrics.get("Baseline_MAE", np.nan)
                mae = metrics.get("MAE", np.nan)
                improvement = baseline_mae - mae if np.isfinite(baseline_mae) and np.isfinite(mae) else np.nan
                improvement_pct = improvement / baseline_mae if np.isfinite(improvement) and baseline_mae else np.nan
                metric_rows.append({
                    "Target": target_name,
                    **metrics,
                    "MAE_Edge_vs_Baseline": improvement,
                    "MAE_Improvement_Pct": improvement_pct,
                })
            metric_table = pd.DataFrame(metric_rows)
            st.dataframe(
                metric_table.style.format({
                    "MAE": "{:.3f}", "RMSE": "{:.3f}", "Bias": "{:+.3f}",
                    "Correlation": "{:.3f}", "Baseline_MAE": "{:.3f}",
                    "MAE_Edge_vs_Baseline": "{:+.3f}", "MAE_Improvement_Pct": "{:+.1%}",
                }),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Bias is projection minus actual. Positive bias means the model is projecting too high. "
                "MAE edge vs baseline is positive when the model beats the simple baseline."
            )

            st.markdown("### K error decomposition and betting gate")
            k_components = completed_history.copy()
            for column in ["Proj_K", "Actual_K", "Proj_BF", "Actual_BF", "Adj_K_Rate"]:
                k_components[column] = pd.to_numeric(k_components.get(column), errors="coerce")
            k_components = k_components.dropna(subset=["Proj_K", "Actual_K", "Proj_BF", "Actual_BF", "Adj_K_Rate"])
            k_components = k_components[k_components["Actual_BF"].gt(0)].copy()
            if not k_components.empty:
                k_components["Actual_K_Rate"] = k_components["Actual_K"] / k_components["Actual_BF"]
                k_components["K_Error"] = k_components["Proj_K"] - k_components["Actual_K"]
                k_components["BF_Error"] = k_components["Proj_BF"] - k_components["Actual_BF"]
                k_components["Rate_Error"] = k_components["Adj_K_Rate"] - k_components["Actual_K_Rate"]
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("K MAE", f"{k_components['K_Error'].abs().mean():.3f}")
                d2.metric("BF MAE", f"{k_components['BF_Error'].abs().mean():.3f}")
                d3.metric("K-rate MAE", f"{k_components['Rate_Error'].abs().mean():.3f}")
                rate_corr = k_components[["Rate_Error", "K_Error"]].corr().iloc[0, 1] if len(k_components) > 2 else np.nan
                d4.metric("Rate-error link", f"{rate_corr:.3f}" if np.isfinite(rate_corr) else "—")
                hand_table = (
                    k_components.groupby("Hand", dropna=False)
                    .agg(N=("Actual_K", "size"), K_MAE=("K_Error", lambda s: s.abs().mean()), K_Bias=("K_Error", "mean"))
                    .reset_index()
                ) if "Hand" in k_components.columns else pd.DataFrame()
                if not hand_table.empty:
                    st.dataframe(hand_table.style.format({"K_MAE": "{:.3f}", "K_Bias": "{:+.3f}"}), width="stretch", hide_index=True)
            line_check = completed_history.dropna(subset=["K_Line", "Proj_K", "Actual_K"]).copy()
            if not line_check.empty:
                line_check["Decision"] = np.where(line_check["Proj_K"].gt(line_check["K_Line"]), "Over", "Under")
                line_check["Result"] = np.where(line_check["Actual_K"].gt(line_check["K_Line"]), "Over", np.where(line_check["Actual_K"].lt(line_check["K_Line"]), "Under", "Push"))
                decisions = line_check[line_check["Result"].ne("Push")]
                win_rate = float(decisions["Decision"].eq(decisions["Result"]).mean()) if not decisions.empty else np.nan
                st.caption(f"Saved-line model-side accuracy: {win_rate:.1%} across {len(decisions):,} non-push decisions." if np.isfinite(win_rate) else "No non-push K-line decisions yet.")

            target_for_scatter = st.selectbox("Projection chart target", list(BACKTEST_TARGETS), key="pitcher_scatter_target")
            scatter_config = BACKTEST_TARGETS[target_for_scatter]
            scatter = completed_history[[scatter_config["projection"], scatter_config["actual"]]].dropna().rename(
                columns={scatter_config["projection"]: "Projection", scatter_config["actual"]: "Actual"}
            )
            if not scatter.empty:
                st.scatter_chart(scatter, x="Projection", y="Actual", width="stretch")

            st.markdown("### Projection buckets — no prop lines required")
            projection_bucket_target = st.selectbox(
                "Projection bucket target",
                list(BACKTEST_TARGETS),
                key="pitcher_projection_bucket_target",
            )
            projection_buckets = model_projection_bucket_table(completed_history, projection_bucket_target)
            if projection_buckets.empty:
                st.info("No completed projections are available for this target yet.")
            else:
                st.dataframe(
                    projection_buckets.style.format({
                        "Average_Projection": "{:.2f}", "Average_Actual": "{:.2f}",
                        "MAE": "{:.3f}", "RMSE": "{:.3f}", "Bias": "{:+.3f}",
                        "Actual_Above_Projection_Rate": "{:.1%}",
                    }),
                    width="stretch",
                    hide_index=True,
                )
            st.caption(
                "These buckets show where the model is too high or too low by projection range, without using sportsbook lines."
            )

            st.markdown("### Score accuracy buckets — no prop lines required")
            bucket_target = st.selectbox("Score target", list(BACKTEST_TARGETS), key="pitcher_score_bucket_target")
            bucket_table = score_bucket_table(completed_history, bucket_target)
            if not bucket_table.empty:
                st.dataframe(
                    bucket_table.style.format({
                        "Average_Projection": "{:.2f}", "Average_Actual": "{:.2f}",
                        "MAE": "{:.3f}", "RMSE": "{:.3f}", "Bias": "{:+.3f}",
                        "Actual_Above_Projection_Rate": "{:.1%}", "Average_Score": "{:.1f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )
            st.caption(
                "Score buckets grade whether higher model scores are producing tighter projections, not whether they beat a prop line."
            )

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

            with st.expander("Optional prop-line calibration — requires saved pregame lines", expanded=False):
                st.caption(
                    "This is only for betting-line decisions. It uses rows where a valid pregame prop line was saved, "
                    "so it can be biased if only some pitchers/books have lines. It does not measure pure projection accuracy."
                )
                prop_target = st.selectbox("Prop target", list(BACKTEST_TARGETS), key="pitcher_prop_calibration_target")
                prop_config = BACKTEST_TARGETS[prop_target]
                target_completed = completed_history[[prop_config["projection"], prop_config["actual"]]].dropna().shape[0]
                prop_records = prop_calibration_records(completed_history, prop_target)
                if prop_records.empty:
                    st.info(f"Enter {BACKTEST_TARGETS[prop_target]['line']} values in your saved snapshots to evaluate probabilities.")
                else:
                    prop_table, brier_score = probability_calibration_table(prop_records)
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Decisions", f"{len(prop_records):,}")
                    p2.metric("Brier score", f"{brier_score:.3f}", help="Lower is better; 0 is perfect.")
                    coverage = len(prop_records) / target_completed if target_completed else np.nan
                    p3.metric("Line coverage", f"{coverage:.1%}" if np.isfinite(coverage) else "N/A")
                    st.dataframe(
                        prop_table.style.format({
                            "Average_Probability": "{:.1%}", "Actual_Win_Rate": "{:.1%}", "Calibration_Gap": "{:+.1%}",
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

        - **Projected K** is now a two-stage estimate: expected batters faced × stabilized K probability per batter. Recent form is deliberately limited, matchup adjustments are additive, and left-handed pitchers receive a conservative correction unless a trained model learns a better value.
        - **Role validation** blocks normal bet eligibility when recent starts, pitch counts and batters faced do not verify a starter workload. Confirmed lineups and a saved K line are also required for a bet signal.
        - **Market blend** uses 35% independent model and 65% sportsbook line by default. The independent projection remains visible, and a signal requires the selected minimum blended edge.
        - **Historical K backfill** reconstructs starters from Statcast and creates every feature from games strictly before the target start. The later 20% is never used for fitting and becomes the chronological holdout.
        - **Projected ER** blends recent and season ER rates with xwOBA, walks, barrels, opponent quality, park factor and first-pitch weather.
        - **Projected outs** starts with recent workload and now adds leash context: last-start pitch count, last-three pitch trend, stable workload rate, quality-start rate, short-hook risk and command risk.
        - **K Score**, **Run Prevention Score** and **Outs Score** are 0–100 relative scores within the selected slate. They are not probabilities.
        - **Overall Score** is 40% K Score, 35% Run Prevention Score and 25% Outs Score.
        - Confirmed lineups are preferred. Before posting, the app uses the opponent's most recent observed lineup from Statcast.
        - Small samples are shrunk toward league averages, and the confidence label falls when a pitcher has few starts or little Statcast history.

        - The **Backtest & calibration** tab exports timestamped pregame snapshots, fills official K/ER/outs results, measures MAE/RMSE/bias, compares simple recent-form baselines, checks score buckets and tests probability calibration.
        - Use **Add current snapshot to session master** to append today's edited pregame board without uploading it first. Download the updated master-history CSV before leaving because Streamlit Community Cloud storage is temporary.
        - K linear calibration is disabled because it did not improve the supplied chronological holdout. ER and outs calibration remain optional.
        - Download the two-stage K model JSON after training and upload it from the sidebar after a Streamlit restart or repository deployment.

        This remains a projection framework. Use several hundred completed starts before making large weight changes, and never include snapshots generated after first pitch in a fair backtest.
        """
    )
