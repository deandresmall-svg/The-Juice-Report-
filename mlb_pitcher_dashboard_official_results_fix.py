def game_log_profile(logs: pd.DataFrame, league: dict) -> dict:
    if logs is None or logs.empty:
        return { ... }  # fallback values

    clean = logs.dropna(subset=["Outs"]).copy()
    clean["K_BF"] = safe_divide(clean["K"], clean["BF"], league["K_PA"])
    clean["ER9"] = safe_divide(clean["ER"] * 27.0, clean["Outs"], league["ER9"])

    # Exponential decay for recency
    clean = clean.sort_values("Date")
    weights = np.exp(np.linspace(-2.2, 0, len(clean)))
    weights /= weights.sum()

    def weighted_mean(s):
        return float(np.average(s.dropna(), weights=weights[-len(s.dropna()):]))

    recent_5 = clean.tail(5)

    return {
        "Starts": int(len(clean)),
        "Season_Outs": float(clean["Outs"].mean()),
        "Recent_Outs": weighted_mean(clean["Outs"]),
        "Outs_SD": float(max(clean["Outs"].tail(12).std(ddof=1) if len(clean) > 1 else 2.55, 1.55)),
        "Season_BF": float(clean["BF"].mean()),
        "Recent_BF": weighted_mean(clean["BF"]),
        "Season_Pitches": float(clean["Pitches"].mean()),
        "Recent_Pitches": weighted_mean(clean["Pitches"]),
        "Season_K_BF": float(clean["K"].sum() / max(clean["BF"].sum(), 1)),
        "Recent_K_BF": weighted_mean(clean["K_BF"]),
        "Recent_K_BF_Exp": float(recent_5["K_BF"].mean()) if not recent_5.empty else weighted_mean(clean["K_BF"]),
        "Season_ER9": float(clean["ER"].sum() * 27 / max(clean["Outs"].sum(), 1)),
        "Recent_ER9": weighted_mean(clean["ER9"]),
        "Pitch_Efficiency": float((clean["Outs"] / (clean["Pitches"] / 100)).mean()),
    }
