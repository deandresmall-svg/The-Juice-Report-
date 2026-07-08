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

# [All your original constants and utility functions go here - keep them unchanged]
# For brevity, I'm showing the key updated parts. Replace your old functions with these.

# ==================== IMPROVED FUNCTIONS ====================

def game_log_profile(logs: pd.DataFrame, league: dict) -> dict:
    if logs is None or logs.empty:
        return {
            "Starts": 0, "Season_Outs": 16.5, "Recent_Outs": 16.5, "Outs_SD": 2.6,
            "Season_BF": 22.5, "Recent_BF": 22.5, "Season_Pitches": 88.0, "Recent_Pitches": 88.0,
            "Season_K_BF": league["K_PA"], "Recent_K_BF": league["K_PA"],
            "Recent_K_BF_Exp": league["K_PA"], "Season_ER9": league["ER9"], 
            "Recent_ER9": league["ER9"], "Pitch_Efficiency": 3.85,
        }
    clean = logs.dropna(subset=["Outs"]).copy()
    clean["K_BF"] = safe_divide(clean["K"], clean["BF"], league["K_PA"])
    clean["ER9"] = safe_divide(clean["ER"] * 27.0, clean["Outs"], league["ER9"])
    
    clean = clean.sort_values("Date").reset_index(drop=True)
    weights = np.exp(np.linspace(-2.2, 0, len(clean)))
    weights /= weights.sum()
    
    def wmean(s):
        return float(np.average(s.dropna(), weights=weights[-len(s.dropna()):]))
    
    recent_5 = clean.tail(5)
    
    return {
        "Starts": int(len(clean)),
        "Season_Outs": float(clean["Outs"].mean()),
        "Recent_Outs": wmean(clean["Outs"]),
        "Outs_SD": float(max(clean["Outs"].tail(12).std(ddof=1) if len(clean) > 1 else 2.6, 1.55)),
        "Season_BF": float(clean["BF"].mean()),
        "Recent_BF": wmean(clean["BF"]),
        "Season_Pitches": float(clean["Pitches"].mean()),
        "Recent_Pitches": wmean(clean["Pitches"]),
        "Season_K_BF": float(clean["K"].sum() / max(clean["BF"].sum(), 1.0)),
        "Recent_K_BF": wmean(clean["K_BF"]),
        "Recent_K_BF_Exp": float(recent_5["K_BF"].mean()) if not recent_5.empty else wmean(clean["K_BF"]),
        "Season_ER9": float(clean["ER"].sum() * 27.0 / max(clean["Outs"].sum(), 1.0)),
        "Recent_ER9": wmean(clean["ER9"]),
        "Pitch_Efficiency": float((clean["Outs"] / (clean["Pitches"] / 100.0)).mean()),
    }

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
    
    k_rate = shrink_rate(pa["is_k"].sum(), plate_appearances, league["K_PA"], 180)
    bb_rate = shrink_rate(pa["is_bb"].sum(), plate_appearances, league["BB_PA"], 160)
    hr_rate = shrink_rate(pa["is_hr"].sum(), plate_appearances, league["HR_PA"], 200)
    xwoba = shrink_rate(pa["xwoba_value"].sum(), plate_appearances, league["xwOBA"], 200)
    barrel = shrink_rate(pitches["is_barrel"].sum(), bbe, league["Brl_BBE"], 130)
    hard_hit = shrink_rate(pitches["is_hard_hit"].sum(), bbe, league["HH_BBE"], 130)
    whiff = shrink_rate(pitches["is_whiff"].sum(), swings, league["Whiff"], 230)
    csw = shrink_rate(pitches["is_whiff"].sum() + pitches["is_called_strike"].sum(), pitch_count, league["CSW"], 380)
    
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

def build_pitcher_projection(...):
    # Use the improved version I provided earlier
    # (Insert the full build_pitcher_projection function from my previous message)
    pass  # Replace with the full function

# ==================== YOUR ORIGINAL CODE CONTINUES HERE ====================
# Paste the rest of your original code (from the big pasted-text.txt) after this point.
# Keep everything else exactly the same except for the three functions above.

print("Updated MLB Pitcher Dashboard loaded successfully!")
