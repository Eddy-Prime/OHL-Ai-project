from flask import Flask, request, jsonify, send_from_directory
import pickle
import json
import difflib
import requests
import numpy as np
from datetime import datetime, date
import xml.etree.ElementTree as ET

import os

app = Flask(__name__)
# Fix hardcoded paths for Docker compatibility
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'rb') as f:
    artifacts = pickle.load(f)
model = artifacts['model']
scaler = artifacts['scaler']
std_dev = 1000.0  # Fixed value since not saved in pickle
feats = artifacts['features']

lookup_path = os.path.join(os.path.dirname(__file__), 'opponentlookup.json')
with open(lookup_path) as f:
    opponent_lookup = json.load(f)
GLOBAL_AVG = opponent_lookup.pop('globalavg', 5000.0)

STADIUM_CAPACITY  = 10_000
LOG_CLIP_MIN, LOG_CLIP_MAX = 6.0, 10.5
SHORT_TERM_DAYS   = 14

_opp_avgs   = list(opponent_lookup.values())
OPP_AVG_MIN = float(min(_opp_avgs))
OPP_AVG_MAX = float(max(_opp_avgs))

SELLOUT_OPPONENTS = {"club brugge", "anderlecht"}
SELLOUT_BLEND     = 0.5
MAX_UPLIFT        = 0.10   # regular opponents: max 10% above their opp_avg
GLOBAL_SOFT_CAP   = 8_000  # hard ceiling for all non-sellout opponents


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_academic_week(d: date) -> int:
    start = date(d.year if d.month >= 9 else d.year - 1, 9, 15)
    return max(1, (d - start).days // 7 + 1)

def get_matchday(d: date) -> int:
    start = date(d.year if d.month >= 7 else d.year - 1, 7, 25)
    return min(34, max(1, round((d - start).days / 7)))

def get_season_progress(matchday: int) -> float:
    return round(matchday / 34, 4)

def get_season_enc(d: date) -> int:
    season_start = d.year if d.month >= 7 else d.year - 1
    return max(0, season_start - 2022)

def fuzzy_match(name: str):
    matches = difflib.get_close_matches(name, opponent_lookup.keys(), n=1, cutoff=0.3)
    if matches:
        return matches[0], float(opponent_lookup[matches[0]])
    for team in opponent_lookup:
        if name.lower() in team.lower() or team.lower() in name.lower():
            return team, float(opponent_lookup[team])
    return None, float(GLOBAL_AVG)

def opp_strength_norm(opp_avg: float) -> float:
    denom = OPP_AVG_MAX - OPP_AVG_MIN
    if denom == 0:
        return 0.5
    return round((opp_avg - OPP_AVG_MIN) / denom, 4)


# ── Weather ───────────────────────────────────────────────────────────────────
def get_rain(match_date: date) -> tuple:
    today      = date.today()
    days_ahead = (match_date - today).days
    if days_ahead > 16:
        return 0.0, "forecast_unavailable"
    if days_ahead < 0:
        url = (f"https://archive-api.open-meteo.com/v1/archive"
               f"?latitude=50.8798&longitude=4.7005&daily=precipitation_sum"
               f"&timezone=Europe/Brussels&start_date={match_date}&end_date={match_date}")
    else:
        url = (f"https://api.open-meteo.com/v1/forecast"
               f"?latitude=50.8798&longitude=4.7005&daily=precipitation_sum"
               f"&timezone=Europe/Brussels&start_date={match_date}&end_date={match_date}")
    try:
        data = requests.get(url, timeout=3).json()
        rain = data['daily']['precipitation_sum'][0]
        return (float(rain) if rain is not None else 0.0), "live"
    except Exception:
        return 0.0, "unavailable"

def rain_label(mm: float) -> str:
    if mm == 0:   return "☀️ No rain forecast"
    elif mm < 2:  return "🌦️ Light rain (minimal impact)"
    elif mm < 10: return "🌧️ Moderate rain (some fans may stay home)"
    else:         return "⛈️ Heavy rain (expect lower attendance)"


# ── News ──────────────────────────────────────────────────────────────────────
def looks_like_match_article(title: str, opponent: str) -> bool:
    t   = title.lower()
    opp = opponent.lower()
    if "ohl" not in t and "oud-heverlee" not in t and "oud heverlee" not in t:
        return False
    if opp not in t:
        return False
    match_keywords = [" - ", "vs", "tegen", "match", "wedstrijd",
                      "verslaat", "verliest", "gelijkspel", "samenvatting",
                      "voorbeschouwing", "speeldag", "jupiler pro league"]
    return any(k in t for k in match_keywords)

def get_article_count_7d(opponent: str) -> tuple:
    queries = [f"\"OH Leuven\" \"{opponent}\" voetbal",
               f"\"Oud-Heverlee Leuven\" \"{opponent}\" wedstrijd",
               f"OHL {opponent} Jupiler Pro League"]
    headers   = {'User-Agent': 'Mozilla/5.0'}
    cutoff_ts = datetime.now().timestamp() - 7 * 86400
    all_titles, relevant = set(), []
    for q in queries:
        url = ("https://news.google.com/rss/search"
               f"?q={q.replace(' ', '+')}&hl=nl&gl=BE&ceid=BE:nl")
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            root = ET.fromstring(resp.content)
            for item in root.findall('.//item'):
                title    = item.findtext('title', '') or ''
                pub_date = item.findtext('pubDate', '')
                if title in all_titles:
                    continue
                all_titles.add(title)
                try:
                    pub_ts = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z").timestamp()
                except Exception:
                    continue
                if pub_ts < cutoff_ts:
                    continue
                if looks_like_match_article(title, opponent):
                    relevant.append(title)
        except Exception:
            continue
    return len(relevant[:5]), relevant[:5]

def get_ohl_interest_proxy(opponent: str, days_ahead: int) -> float:
    if days_ahead <= SHORT_TERM_DAYS:
        count, _ = get_article_count_7d(opponent)
        return min(50.0, max(2.0, float(count) * 1.5))
    return 5.0


# ── Form helpers ──────────────────────────────────────────────────────────────
def parse_form(form_str: str):
    """Parse last 3 results. Returns (points, wins_last_3, goal_diff)."""
    form_str   = form_str.upper().strip()
    tokens     = form_str.replace(',', ' ').split()
    results, goal_diffs = [], []
    for tok in tokens:
        result, gdiff = None, 0
        for r in ('W', 'D', 'L'):
            if tok.startswith(r):
                result = r
                rest   = tok[1:]
                if '-' in rest:
                    try:
                        parts = rest.split('-')
                        gdiff = int(parts[0]) - int(parts[1])
                    except Exception:
                        gdiff = 0
                break
        if result:
            results.append(result)
            goal_diffs.append(gdiff)
    if not results:
        return None
    points_map = {'W': 3, 'D': 1, 'L': 0}
    last3      = results[-3:]
    goal_last3 = goal_diffs[-3:]
    points     = sum(points_map[r] for r in last3)
    wins       = sum(1 for r in last3 if r == 'W')
    goal_diff  = sum(goal_last3)
    return points, wins, goal_diff

TOP_TEAMS = {"club brugge", "anderlecht", "stvv", "kv mechelen", "westerlo"}

def is_top_opponent(name: str) -> int:
    return int(name.lower() in TOP_TEAMS)

def is_playoff_match(d: date) -> int:
    return int((d.month == 3 and d.day >= 20) or (d.month in [4, 5] and d.day <= 20))


# ── Feature vector ────────────────────────────────────────────────────────────
def build_feature_vector(session: dict, points_last_5: float,
                         goal_diff_last_5: float, rain_mm: float,
                         article_count_7d: int, ohl_interest: float) -> np.ndarray:

    match_date   = datetime.strptime(session['date'], "%Y-%m-%d").date()
    matchday     = get_matchday(match_date)
    season_prog  = get_season_progress(matchday)
    season_enc   = get_season_enc(match_date)
    kickoff_hour = session['kickoff_hour']
    opp_avg      = session['opp_avg']
    is_top       = session['is_top']

    attendance_lag_1  = session.get('attendance_lag_1',  opp_avg)
    attendance_roll_3 = session.get('attendance_roll_3', opp_avg)
    attendance_roll_5 = session.get('attendance_roll_5', opp_avg)

    tickets_combined = session.get('tickets_combined', None)
    if tickets_combined is not None:
        tickets_sold_b2c = tickets_combined * 0.73
        tickets_sold_b2b = tickets_combined * 0.27
    else:
        tickets_sold_b2c = opp_avg * 0.35
        tickets_sold_b2b = opp_avg * 0.13

    pct_free_tickets = session.get('pct_free_tickets', 0.0)
    has_promotion    = session.get('has_promotion', 0)

    opp_str_norm_val = opp_strength_norm(opp_avg)
    form_norm        = points_last_5 / 15
    match_importance = round(0.5 * form_norm + 0.5 * season_prog, 4)
    match_attract    = round(
        0.4 * match_importance + 0.4 * opp_str_norm_val + 0.2 * is_top, 4)
    form_x_opp = points_last_5 * is_top
    lag_x_form = attendance_lag_1 * points_last_5

    feature_map = {
        'attendance_lag_1':            attendance_lag_1,
        'attendance_roll_3':           attendance_roll_3,
        'attendance_roll_5':           attendance_roll_5,
        'tickets_sold_b2c':            tickets_sold_b2c,
        'tickets_sold_b2b':            tickets_sold_b2b,
        'opponent_avg_attendance_raw': opp_avg,
        'is_top_opponent':             is_top,
        'points_last_5':               points_last_5,
        'goal_diff_last_5':            goal_diff_last_5,
        'form_x_opponent':             form_x_opp,
        'match_attractiveness':        match_attract,
        'season_progress':             season_prog,
        'season_enc':                  season_enc,
        'is_playoff':                  is_playoff_match(match_date),
        'has_promotion':               has_promotion,
        'pct_free_tickets':            pct_free_tickets,
        'kickoff_hour':                kickoff_hour,
        'weather_rain_mm':             rain_mm,
        'ohl_interest':                ohl_interest,
        'article_count_7d':            float(article_count_7d),
        'lag_x_form':                  lag_x_form,
    }

    return np.array([[feature_map.get(f, 0.0) for f in feats]])


# ── Session ───────────────────────────────────────────────────────────────────
session = {}


# ── Reset ─────────────────────────────────────────────────────────────────────
@app.route('/reset', methods=['POST'])
def reset():
    session.clear()
    session['step'] = 'ask_opponent'
    return jsonify({'ok': True})


# ── Chat ──────────────────────────────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message', '').strip()
    step     = session.get('step', 'ask_opponent')

    # ── Step 1: Opponent ──────────────────────────────────────────────────────
    if step == 'ask_opponent':
        matched, opp_avg = fuzzy_match(user_msg)
        if not matched:
            reply = (
                f"❌ Opponent <b>'{user_msg}'</b> not recognised.<br>"
                f"Try a name like <b>Club Brugge</b>, <b>Anderlecht</b>, <b>Westerlo</b>...<br><br>"
                f"👤 Who is the <b>opponent</b>?"
            )
        else:
            session['opponent'] = matched
            session['opp_avg']  = opp_avg
            session['is_top']   = is_top_opponent(user_msg)
            session['step']     = 'ask_date'
            reply = (
                f"Got it — <b>{matched}</b> "
                f"(historical avg: {opp_avg:.0f} tickets).<br>"
                f"📅 What is the match date? <i>(YYYY-MM-DD)</i>"
            )

    # ── Step 2: Date ──────────────────────────────────────────────────────────
    elif step == 'ask_date':
        try:
            match_date = datetime.strptime(user_msg, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({'reply': "❌ Invalid date format. Please use <b>YYYY-MM-DD</b> (e.g. 2026-04-15)."})
        days_ahead = (match_date - date.today()).days
        session['date']       = user_msg
        session['days_ahead'] = days_ahead
        session['step']       = 'ask_kickoff'
        mode_label = (
            f"🔴 <b>Short-term mode</b> (≤{SHORT_TERM_DAYS} days) — "
            "live weather + live news + ticket sales used."
            if days_ahead <= SHORT_TERM_DAYS
            else f"🔵 <b>Long-term mode</b> (>{SHORT_TERM_DAYS} days) — "
            "weather estimated, news baseline used."
        )
        reply = (
            f"📅 Date set: <b>{match_date.strftime('%A, %d %b %Y')}</b> "
            f"({days_ahead} days away).<br>{mode_label}<br><br>"
            f"⏰ What is the kickoff time? <i>(HH:MM, e.g. 20:45)</i>"
        )

    # ── Step 3: Kickoff ───────────────────────────────────────────────────────
    elif step == 'ask_kickoff':
        try:
            kickoff_hour = int(user_msg.split(':')[0])
        except ValueError:
            return jsonify({'reply': "❌ Invalid time. Please use <b>HH:MM</b> format (e.g. 20:45)."})
        session['kickoff_hour'] = kickoff_hour

        if session['days_ahead'] <= SHORT_TERM_DAYS:
            session['step'] = 'ask_tickets_sold'
            reply = (
                "⏰ Kickoff set.<br><br>"
                "🎟️ How many <b>single-match tickets have been sold</b> so far "
                "<i>(excluding season tickets)</i>?<br>"
                "<i>(e.g. 2400)</i><br>"
                "<small style='color:#888'>Type <b>skip</b> if unknown.</small>"
            )
        else:
            session['points_last_5']    = 7.5
            session['goal_diff_last_5'] = 0.0
            session['form_note']        = "<i style='color:#888;font-size:12px;'>* Using season average form.</i><br>"
            session['step'] = 'ask_free_tickets'
            reply = (
                "⏰ Kickoff set.<br><br>"
                "🎫 What <b>% of tickets are complimentary</b> (free) for this match?<br>"
                "<i>(e.g. 5 for 5%, or 0)</i><br>"
                "<small style='color:#888'>Type <b>skip</b> to use 0%.</small>"
            )

    # ── Step 4 (short-term only): Tickets sold ────────────────────────────────
    elif step == 'ask_tickets_sold':
        if user_msg.lower() != 'skip':
            try:
                session['tickets_combined'] = float(
                    user_msg.replace(',', '').replace('.', ''))
            except ValueError:
                return jsonify({'reply': "❌ Please enter a number (e.g. 2400) or type <b>skip</b>."})
        session['step'] = 'ask_attendance'
        reply = (
            "🏟️ What was OHL's attendance at the <b>last 3–5 home matches</b>?<br>"
            "<i>Enter up to 5 numbers, most recent first (e.g. 7200 6800 8100 7500 6200)</i><br>"
            "<small style='color:#888'>Type <b>skip</b> to use historical average.</small>"
        )

    # ── Step 5 (short-term only): Recent home attendance ─────────────────────
    elif step == 'ask_attendance':
        if user_msg.lower() == 'skip':
            att_note = "<i style='color:#888;font-size:12px;'>* Using historical avg for recent attendance.</i><br>"
        else:
            try:
                nums = [int(x.replace(',', '').replace('.', '')) for x in user_msg.split()]
                nums = [max(0, min(STADIUM_CAPACITY + 2000, n)) for n in nums[:5]]
            except ValueError:
                return jsonify({'reply': "❌ Please enter numbers only (e.g. 7200 6800 8100) or type <b>skip</b>."})
            if not nums:
                nums = [int(session['opp_avg'])]
            session['attendance_lag_1']  = float(nums[0])
            session['attendance_roll_3'] = float(np.mean(nums[:3]))
            session['attendance_roll_5'] = float(np.mean(nums[:5]))
            att_note = (
                f"<i style='color:#a0c4ff;font-size:12px;'>"
                f"✅ Last attendance: {nums[0]:,} | "
                f"3-match avg: {session['attendance_roll_3']:,.0f}"
                f"</i><br>"
            )
        session['att_note'] = att_note
        session['step']     = 'ask_form'
        reply = (
            f"{att_note}"
            "📋 What are OHL's <b>last 3 match results</b>?<br>"
            "<i>With scores for best accuracy: W3-1 D1-1 L0-2<br>"
            "Or simple letters: W L D (oldest → newest)</i><br>"
            "<small style='color:#888'>Type <b>skip</b> to use season averages.</small>"
        )

    # ── Step 6 (short-term only): Form ───────────────────────────────────────
    elif step == 'ask_form':
        form_note = ""
        if user_msg.lower() == 'skip':
            session['points_last_5']    = 7.5
            session['goal_diff_last_5'] = 0.0
            form_note = "<i style='color:#888;font-size:12px;'>* Using season average form.</i><br>"
        else:
            parsed = parse_form(user_msg)
            if parsed:
                session['points_last_5']    = parsed[0]
                session['goal_diff_last_5'] = float(parsed[2])
                form_note = (
                    f"<i style='color:#a0c4ff;font-size:12px;'>"
                    f"✅ Form: {parsed[0]} pts last 3 | {parsed[1]} wins | "
                    f"GD: {parsed[2]:+d}"
                    f"</i><br>"
                )
            else:
                session['points_last_5']    = 7.5
                session['goal_diff_last_5'] = 0.0
                form_note = "<i style='color:#f4a261;font-size:12px;'>⚠️ Could not parse — using season averages.</i><br>"
        session['form_note'] = form_note
        session['step']      = 'ask_free_tickets'
        reply = (
            f"{form_note}"
            "🎫 What <b>% of tickets are complimentary</b> (free) for this match?<br>"
            "<i>(e.g. 5 for 5%, or 0)</i><br>"
            "<small style='color:#888'>Type <b>skip</b> to use 0%.</small>"
        )

    # ── Step 7: Free ticket % ─────────────────────────────────────────────────
    elif step == 'ask_free_tickets':
        if user_msg.lower() == 'skip' or user_msg.strip() == '0':
            session['pct_free_tickets'] = 0.0
        else:
            try:
                pct = float(user_msg.replace('%', '').strip())
                session['pct_free_tickets'] = max(0.0, min(100.0, pct))
            except ValueError:
                return jsonify({'reply': "❌ Please enter a number like <b>5</b> (for 5%) or type <b>skip</b>."})
        session['step'] = 'ask_promotion'
        reply = (
            "🎟️ Is there an active <b>ticket promotion</b> for this match?<br>"
            "<i>(yes / no)</i>"
        )

    # ── Step 8: Promotion → Predict ───────────────────────────────────────────
    elif step == 'ask_promotion':
        session['has_promotion'] = 1 if user_msg.lower() in ('yes', 'y', 'ja') else 0
        session['step']          = 'predict'

        match_date = datetime.strptime(session['date'], "%Y-%m-%d").date()
        days_ahead = session['days_ahead']
        opponent   = session['opponent']
        opp_avg    = session['opp_avg']

        rain_mm, rain_source     = get_rain(match_date)
        article_count, headlines = get_article_count_7d(opponent)
        ohl_interest             = get_ohl_interest_proxy(opponent, days_ahead)
        points_last_5            = session['points_last_5']
        goal_diff_last_5         = session['goal_diff_last_5']

        X_vec    = build_feature_vector(
            session, points_last_5, goal_diff_last_5,
            rain_mm, article_count, ohl_interest
        )
        X_scaled = scaler.transform(X_vec)
        log_pred = float(model.predict(X_scaled)[0])
        log_pred = np.clip(log_pred, LOG_CLIP_MIN, LOG_CLIP_MAX)

        # ── Apply ceiling in log-space BEFORE computing interval ──────────────
        if opponent.lower() in SELLOUT_OPPONENTS:
            # Blend toward opp_avg for sellout opponents
            raw_pred = round(np.exp(log_pred))  
            raw_low  = max(0,                raw_pred - int(std_dev))
            raw_high = min(STADIUM_CAPACITY, raw_pred + int(std_dev))
            pred = min(STADIUM_CAPACITY, round(raw_pred * (1 - SELLOUT_BLEND) + opp_avg * SELLOUT_BLEND))
            low  = min(STADIUM_CAPACITY, round(raw_low  * (1 - SELLOUT_BLEND) + opp_avg * SELLOUT_BLEND))
            high = min(STADIUM_CAPACITY, round(raw_high * (1 - SELLOUT_BLEND) + opp_avg * SELLOUT_BLEND))
        else:
            # Cap in log-space — interval stays ±std_dev wide around capped value
            ceiling     = min(GLOBAL_SOFT_CAP, round(opp_avg * (1 + MAX_UPLIFT)))
            log_ceiling = np.log(max(ceiling, 1))
            log_pred    = min(log_pred, log_ceiling)   # cap here, once
            pred = round(np.exp(log_pred))
            low  = max(0,                pred - int(std_dev))
            high = min(STADIUM_CAPACITY, pred + int(std_dev))
        pct      = round(pred / STADIUM_CAPACITY * 100, 1)
        pct_low  = round(low  / STADIUM_CAPACITY * 100, 1)
        pct_high = round(high / STADIUM_CAPACITY * 100, 1)

        if pred >= 8500:
            tier = "🔥 Near sell-out — activate waitlist"
        elif pred >= 7000:
            tier = "✅ High attendance — full staffing recommended"
        elif pred >= 5000:
            tier = "🟡 Moderate attendance — standard staffing"
        else:
            tier = "🔴 Low attendance — consider promotional push"

        if days_ahead <= SHORT_TERM_DAYS:
            mode_badge   = "🔴 <b>Short-term prediction</b> — live weather + match-relevant news used"
            weather_line = (
                f"{rain_label(rain_mm)} ({rain_mm:.1f}mm)"
                if rain_source == "live"
                else "⚠️ Weather data unavailable — using 0mm"
            )
        else:
            mode_badge   = "🔵 <b>Long-term prediction</b> — estimated weather, news baseline"
            weather_line = (
                "🌤️ Weather forecast unavailable (>16 days away) — using 0mm"
                if rain_source == "forecast_unavailable"
                else f"{rain_label(rain_mm)} ({rain_mm:.1f}mm)"
            )

        if headlines:
            news_items = "".join(
                f"<li style='font-size:12px;color:#ccc;margin:2px 0;'>{h}</li>"
                for h in headlines)
            news_block = (
                f"<br>📰 <b>Match-relevant news ({article_count} articles, last 7 days):</b>"
                f"<ul style='margin:4px 0 0 16px;padding:0;'>{news_items}</ul>")
        else:
            news_block = (
                f"<br>📰 <b>News:</b> {article_count} relevant articles in last 7 days "
                f"<i style='color:#888;'>(based on title & date filtering)</i>")

        matchday     = get_matchday(match_date)
        academic_w   = get_academic_week(match_date)
        form_note    = session.get('form_note', '')
        att_note     = session.get('att_note', '')
        pct_free     = session.get('pct_free_tickets', 0.0)
        tickets_line = (
            f"🎟️ Tickets sold so far: {int(session['tickets_combined']):,}<br>"
            if 'tickets_combined' in session else ""
        )
        sellout_note = (
            f"<i style='color:#f4a261;font-size:12px;'>⚡ Sellout adjustment applied "
            f"(blended with {opp_avg:.0f} historical avg)</i><br>"
            if opponent.lower() in SELLOUT_OPPONENTS else ""
        )

        reply = f"""
        {mode_badge}<br>
        ✅ Prediction for <b>{opponent}</b>
        on <b>{session['date']}</b> at <b>{session['kickoff_hour']}:00</b>:<br><br>

        {weather_line}<br>
        📅 Matchday: ~{matchday} &nbsp;|&nbsp; Academic week: {academic_w}<br>
        🏆 Top opponent: {"Yes" if session['is_top'] else "No"}
        &nbsp;|&nbsp; 🎟️ Promotion: {"Yes" if session['has_promotion'] else "No"}
        &nbsp;|&nbsp; 🎫 Free tickets: {pct_free:.0f}%<br>
        {tickets_line}{att_note}{form_note}
        📋 Form: {points_last_5:.0f} pts | GD: {goal_diff_last_5:+.0f}
        {news_block}

        <div style='background:#0d2a4a;padding:14px;border-radius:8px;margin-top:10px;'>
            {sellout_note}
            📊 <b>Predicted attendance: {pred:,}</b> ({pct}% capacity)<br>
            📉 Low estimate:  {low:,} ({pct_low}%)<br>
            📈 High estimate: {high:,} ({pct_high}%)<br><br>
            <span style='color:#f4a261;'>{tier}</span>
        </div><br>

        🔄 Want to predict another match? Type an opponent name to start again!
        """

        session.clear()
        session['step'] = 'ask_opponent'

    else:
        session.clear()
        session['step'] = 'ask_opponent'
        reply = "👤 Who is the <b>opponent</b>?"

    return jsonify({'reply': reply})

# ── Stateless API Endpoint for Dashboard ──────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # 1. Opponent
    raw_opponent = data.get('opponent', '')
    matched, opp_avg = fuzzy_match(raw_opponent)
    if not matched:
        return jsonify({"error": f"Opponent '{raw_opponent}' not recognised."}), 400
    is_top = is_top_opponent(raw_opponent)

    # 2. Date
    raw_date = data.get('date', '')
    try:
        match_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    days_ahead = (match_date - date.today()).days

    # 3. Kickoff
    raw_kickoff = data.get('kickoff', '20:00')
    try:
        kickoff_hour = int(raw_kickoff.split(':')[0])
    except ValueError:
        kickoff_hour = 20

    # 4. Form
    raw_form = data.get('form', 'skip')
    points_last_5 = 7.5
    goal_diff_last_5 = 0.0
    if raw_form.lower() != 'skip':
        parsed = parse_form(raw_form)
        if parsed:
            points_last_5 = parsed[0]

    # 5. Promotion
    has_promotion = 1 if data.get('promotion') else 0

    # Create session-like dict for build_feature_vector
    temp_session = {
        'date': raw_date,
        'kickoff_hour': kickoff_hour,
        'opp_avg': opp_avg,
        'is_top': is_top,
        'opponent': matched,
        'has_promotion': has_promotion
    }

    # Fetch dynamic data
    rain_mm, rain_source = get_rain(match_date)
    article_count, headlines = get_article_count_7d(matched)
    ohl_interest = get_ohl_interest_proxy(matched, days_ahead)

    # Prediction
    try:
        X_vec = build_feature_vector(
            temp_session, points_last_5, goal_diff_last_5,
            rain_mm, article_count, ohl_interest
        )
        X_scaled = scaler.transform(X_vec)
        pred = float(model.predict(X_scaled)[0])
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    low = max(0, round(pred - std_dev))
    high = min(STADIUM_CAPACITY, round(pred + std_dev))
    pred = round(pred)

    if pred >= 8500:
        tier = "Near sell-out"
        color = "#e63946" # Red
    elif pred >= 7000:
        tier = "High attendance"
        color = "#2a9d8f" # Green
    elif pred >= 5000:
        tier = "Moderate attendance"
        color = "#e9c46a" # Yellow
    else:
        tier = "Low attendance"
        color = "#f4a261" # Orange

    return jsonify({
        "prediction": pred,
        "low": low,
        "high": high,
        "capacity_pct": round(pred / STADIUM_CAPACITY * 100, 1),
        "tier": tier,
        "tier_color": color,
        "opponent": matched,
        "match_date": match_date.strftime('%A, %d %b %Y'),
        "weather": {
            "rain_mm": rain_mm,
            "label": rain_label(rain_mm)
        },
        "news": {
            "count": article_count,
            "headlines": headlines
        }
    })

import os
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
PUBLIC_DIR = os.path.join(PROJECT_ROOT, 'public')

@app.route('/public/<path:filename>')
def public_files(filename):
    return send_from_directory(PUBLIC_DIR, filename)

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    metrics = {}
    csv_path = os.path.join(os.path.dirname(__file__), '../../data/raw/model_results_combined.csv')
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics['MAE'] = round(float(row['MAE']), 1)
                metrics['RMSE'] = round(float(row['RMSE']), 1)
                metrics['R2'] = round(float(row['R2']), 2)
                metrics['MAPE'] = f"{round(float(row['MAPE']), 1)}%"
                break # Only one row expected
    except Exception as e:
        return jsonify({"error": f"Could not load metrics: {e}"}), 500

    return jsonify({
        "metrics": metrics,
        "features": feats,
        "features_count": len(feats),
        "global_avg": round(GLOBAL_AVG)
    })

@app.route('/')
def index():
    # Serve dashboard.html from the root directory of the project
    return send_from_directory(os.path.join(os.path.dirname(__file__), '../../'), 'dashboard.html')

@app.route('/chat')
def chat_page():
    # Serve the original chatbot interface
    return send_from_directory(os.path.dirname(__file__), 'index.html')

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5800, debug=True)