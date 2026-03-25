from flask import Flask, request, jsonify, send_from_directory
import pickle
import json
import difflib
import requests
import numpy as np
from datetime import datetime, date
import xml.etree.ElementTree as ET

app = Flask(__name__)
with open('/Users/nachatissa/Desktop/SCHOOL/SEM2/Advanced AI/BUSit week/Attendance AI/MODEL/OHL-Ai-project/src/interactive_interface/model.pkl', 'rb') as f:
    artifacts = pickle.load(f)
model = artifacts['model']
scaler = artifacts['scaler']
std_dev = 1000.0  # Fixed value since not saved in pickle
feats = artifacts['features']

with open('/Users/nachatissa/Desktop/SCHOOL/SEM2/Advanced AI/BUSit week/Attendance AI/MODEL/OHL-Ai-project/src/interactive_interface/opponentlookup.json') as f:
    opponent_lookup = json.load(f)
GLOBAL_AVG = opponent_lookup.pop('globalavg', 5000.0)

STADIUM_CAPACITY = 10_000

# All known opponent dummy columns
ALL_OPP_DUMMIES = [f for f in feats if f.startswith('opp_')]

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
    # 2022/23 = 0, 2023/24 = 1, 2024/25 = 2, 2025/26 = 3 ...
    base_year = 2022
    season_start = d.year if d.month >= 7 else d.year - 1
    return max(0, season_start - base_year)

def fuzzy_match(name: str):
    matches = difflib.get_close_matches(name, opponent_lookup.keys(), n=1, cutoff=0.3)
    if matches:
        return matches[0], float(opponent_lookup[matches[0]])
    for team in opponent_lookup:
        if name.lower() in team.lower() or team.lower() in name.lower():
            return team, float(opponent_lookup[team])
    return None, float(GLOBAL_AVG)

def get_opp_dummy_vector(matched_name: str) -> dict:
    vec = {col: 0 for col in ALL_OPP_DUMMIES}
    key = f"opp_{matched_name}"
    if key in vec:
        vec[key] = 1
    return vec

# ── Weather ───────────────────────────────────────────────────────────────────
def get_rain(match_date: date) -> tuple:
    today      = date.today()
    days_ahead = (match_date - today).days

    if days_ahead > 16:
        return 0.0, "forecast_unavailable"

    if days_ahead < 0:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude=50.8798&longitude=4.7005"
            f"&daily=precipitation_sum"
            f"&timezone=Europe/Brussels"
            f"&start_date={match_date}&end_date={match_date}"
        )
    else:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude=50.8798&longitude=4.7005"
            f"&daily=precipitation_sum"
            f"&timezone=Europe/Brussels"
            f"&start_date={match_date}&end_date={match_date}"
        )

    try:
        data = requests.get(url, timeout=3).json()
        rain = data['daily']['precipitation_sum'][0]
        return (float(rain) if rain is not None else 0.0), "live"
    except Exception:
        return 0.0, "unavailable"

def rain_label(mm: float) -> str:
    if mm == 0:    return "☀️ No rain forecast"
    elif mm < 2:   return "🌦️ Light rain (minimal impact)"
    elif mm < 10:  return "🌧️ Moderate rain (some fans may stay home)"
    else:          return "⛈️ Heavy rain (expect lower attendance)"

# ── News scraping: match‑relevant filter ──────────────────────────────────────
def looks_like_match_article(title: str, opponent: str) -> bool:
    """Heuristics to keep only match-relevant articles."""
    t   = title.lower()
    opp = opponent.lower()

    # Must mention OHL or Oud-Heverlee Leuven
    if "ohl" not in t and "oud-heverlee" not in t and "oud heverlee" not in t:
        return False

    # Must mention opponent name
    if opp not in t:
        return False

    # Prefer match-related language
    match_keywords = [
        " - ", "vs", "tegen", "match", "wedstrijd",
        "verslaat", "verliest", "gelijkspel", "samenvatting",
        "voorbeschouwing", "speeldag", "jupiler pro league"
    ]
    return any(k in t for k in match_keywords)

def get_article_count_7d(opponent: str) -> tuple:
    """
    Google News RSS for OHL vs opponent, last 7 days, filtered to
    match‑relevant football articles only. Returns (count, headlines_list).
    """
    queries = [
        f"\"OH Leuven\" \"{opponent}\" voetbal",
        f"\"Oud-Heverlee Leuven\" \"{opponent}\" wedstrijd",
        f"OHL {opponent} Jupiler Pro League"
    ]

    headers   = {'User-Agent': 'Mozilla/5.0'}
    cutoff_ts = datetime.now().timestamp() - 7 * 86400

    all_titles = set()
    relevant   = []

    for q in queries:
        url = (
            "https://news.google.com/rss/search"
            f"?q={q.replace(' ', '+')}"
            "&hl=nl&gl=BE&ceid=BE:nl"
        )
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            root = ET.fromstring(resp.content)
            items = root.findall('.//item')

            for item in items:
                title    = item.findtext('title', '') or ''
                pub_date = item.findtext('pubDate', '')

                if title in all_titles:
                    continue
                all_titles.add(title)

                try:
                    pub_ts = datetime.strptime(
                        pub_date, "%a, %d %b %Y %H:%M:%S %Z"
                    ).timestamp()
                except Exception:
                    continue

                # Only last 7 days
                if pub_ts < cutoff_ts:
                    continue

                # Filter to match‑relevant football content
                if not looks_like_match_article(title, opponent):
                    continue

                relevant.append(title)

        except Exception:
            continue

    relevant = relevant[:5]
    return len(relevant), relevant

def get_ohl_interest_proxy(opponent: str, days_ahead: int) -> float:
    """
    For short-term: derive from article_count.
    For long-term: use median baseline (5.0).
    """
    if days_ahead <= 7:
        count, _ = get_article_count_7d(opponent)
        return min(50.0, max(2.0, float(count) * 1.5))
    else:
        return 5.0

# ── Calendar & form helpers ───────────────────────────────────────────────────
def is_weekend(d: date) -> int:
    return int(d.weekday() >= 5)

def parse_form(form_str: str):
    form_str = form_str.upper().replace(',', ' ').replace('-', ' ')
    results  = [c for c in form_str if c in ('W', 'D', 'L')]
    if not results:
        return None
    points_map    = {'W': 3, 'D': 1, 'L': 0}
    last5         = results[-5:]
    last3         = results[-3:]
    points_last_5 = sum(points_map[r] for r in last5)
    wins_last_3   = sum(1 for r in last3 if r == 'W')
    return points_last_5, wins_last_3

TOP_TEAMS = {"club brugge", "anderlecht", "stvv", "kv mechelen", "westerlo"}

def is_top_opponent(name: str) -> int:
    return int(name.lower() in TOP_TEAMS)

def is_playoff_match(d: date) -> int:
    # Rough window: late March → mid-May
    return int((d.month == 3 and d.day >= 20) or
               (d.month in [4, 5] and d.day <= 20))

# ── Feature vector builder ────────────────────────────────────────────────────
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
    matched_name = session['opponent']

    # Rolling attendance — proxy with global average
    attendance_lag_1  = float(GLOBAL_AVG)
    attendance_roll_3 = float(GLOBAL_AVG)
    attendance_roll_5 = float(GLOBAL_AVG)

    # Composites
    opp_str_norm     = 0.5
    form_norm        = points_last_5 / 15
    match_importance = round(0.5 * form_norm + 0.5 * season_prog, 4)
    match_attract    = round(0.4 * match_importance + 0.4 * opp_str_norm + 0.2 * is_top, 4)
    form_x_opp       = points_last_5 * is_top
    lag_x_form       = attendance_lag_1 * points_last_5

    has_promotion    = session.get('has_promotion', 0)
    is_playoff_flag  = is_playoff_match(match_date)

    tickets_sold_b2c = session.get('tickets_sold_b2c', 2500.0)
    tickets_sold_b2b = session.get('tickets_sold_b2b', 800.0)

    opp_vec = get_opp_dummy_vector(matched_name)

    feature_map = {
        'attendance_lag_1':            attendance_lag_1,
        'attendance_roll_3':           attendance_roll_3,
        'attendance_roll_5':           attendance_roll_5,
        'tickets_sold_b2c':            tickets_sold_b2c,
        'tickets_sold_b2b':            tickets_sold_b2b,
        'opponent_avg_attendance_raw': opp_avg,
        'is_top_opponent':             is_top,
        'last_result_vs_opponent_enc': session.get('last_result_enc', 0),
        'points_last_5':               points_last_5,
        'goal_diff_last_5':            goal_diff_last_5,
        'form_x_opponent':             form_x_opp,
        'match_importance':            match_importance,
        'match_attractiveness':        match_attract,
        'season_progress':             season_prog,
        'season_enc':                  season_enc,
        'is_playoff':                  is_playoff_flag,
        'has_promotion':               has_promotion,
        'pct_free_tickets':            session.get('pct_free_tickets', 0.0),
        'is_weekend':                  is_weekend(match_date),
        'is_midweek':                  int(not is_weekend(match_date)),
        'kickoff_hour':                kickoff_hour,
        'is_school_holiday_flanders':  session.get('is_school_holiday', 0),
        'is_public_holiday':           0,
        'academic_week':               get_academic_week(match_date),
        'weather_score':               max(0.0, 10.0 - rain_mm),
        'weather_rain_mm':             rain_mm,
        'ohl_interest':                ohl_interest,
        'article_count_7d':            float(article_count_7d),
        'lag_x_form':                  lag_x_form,
        'promo_x_top_opponent':        has_promotion * is_top,
        'rain_x_weekend':              rain_mm * is_weekend(match_date),
        'playoff_x_top':               is_playoff_flag * is_top,
        **opp_vec
    }

    return np.array([[feature_map.get(f, 0.0) for f in feats]])

# ── Session state ─────────────────────────────────────────────────────────────
session = {}

# ── Reset endpoint ────────────────────────────────────────────────────────────
@app.route('/reset', methods=['POST'])
def reset():
    session.clear()
    session['step'] = 'ask_opponent'
    return jsonify({'ok': True})

# ── Chat endpoint ─────────────────────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message', '').strip()
    step     = session.get('step', 'ask_opponent')

    # Step 1: Opponent
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

    # Step 2: Date
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
            "🔴 <b>Short-term mode</b> (≤7 days) — live weather + live news will be used."
            if days_ahead <= 7
            else "🔵 <b>Long-term mode</b> (>7 days) — weather estimated, news baseline used."
        )

        reply = (
            f"📅 Date set: <b>{match_date.strftime('%A, %d %b %Y')}</b> "
            f"({days_ahead} days away).<br>"
            f"{mode_label}<br><br>"
            f"⏰ What is the kickoff time? <i>(HH:MM, e.g. 20:45)</i>"
        )

    # Step 3: Kickoff
    elif step == 'ask_kickoff':
        try:
            kickoff_hour = int(user_msg.split(':')[0])
        except ValueError:
            return jsonify({'reply': "❌ Invalid time. Please use <b>HH:MM</b> format (e.g. 20:45)."})

        session['kickoff_hour'] = kickoff_hour
        session['step']         = 'ask_form'
        reply = (
            "⏰ Kickoff set.<br><br>"
            "📋 What are OHL's <b>last 5 match results</b>? "
            "<i>(e.g. W W L D W — oldest to newest)</i><br>"
            "<small style='color:#888'>Type <b>skip</b> to use season averages instead.</small>"
        )

    # Step 4: Form
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
                session['goal_diff_last_5'] = 0.0
                form_note = (
                    f"<i style='color:#a0c4ff;font-size:12px;'>"
                    f"✅ Form: {parsed[0]} pts last 5 | {parsed[1]} wins last 3"
                    f"</i><br>"
                )
            else:
                session['points_last_5']    = 7.5
                session['goal_diff_last_5'] = 0.0
                form_note = "<i style='color:#f4a261;font-size:12px;'>⚠️ Could not parse — using season averages.</i><br>"

        session['form_note'] = form_note
        session['step']      = 'ask_promotion'
        reply = (
            f"{form_note}"
            "🎟️ Is there an active <b>ticket promotion</b> for this match?<br>"
            "<i>(yes / no)</i>"
        )

    # Step 5: Promotion → Predict
    elif step == 'ask_promotion':
        session['has_promotion'] = 1 if user_msg.lower() in ('yes', 'y', 'ja') else 0
        session['step']          = 'predict'

        match_date = datetime.strptime(session['date'], "%Y-%m-%d").date()
        days_ahead = session['days_ahead']
        opponent   = session['opponent']

        rain_mm, rain_source      = get_rain(match_date)
        article_count, headlines  = get_article_count_7d(opponent)
        ohl_interest              = get_ohl_interest_proxy(opponent, days_ahead)
        points_last_5             = session['points_last_5']
        goal_diff_last_5          = session['goal_diff_last_5']

        X_vec    = build_feature_vector(
            session, points_last_5, goal_diff_last_5,
            rain_mm, article_count, ohl_interest
        )
        X_scaled = scaler.transform(X_vec)
        pred     = float(model.predict(X_scaled)[0])

        low      = max(0, round(pred - std_dev))
        high     = min(STADIUM_CAPACITY, round(pred + std_dev))
        pred     = round(pred)

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

        if days_ahead <= 7:
            mode_badge = "🔴 <b>Short-term prediction</b> — live weather + match-relevant news used"
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
                for h in headlines
            )
            news_block = (
                f"<br>📰 <b>Match-relevant news ({article_count} articles, last 7 days):</b>"
                f"<ul style='margin:4px 0 0 16px;padding:0;'>{news_items}</ul>"
            )
        else:
            news_block = (
                f"<br>📰 <b>News:</b> {article_count} relevant articles in last 7 days "
                f"<i style='color:#888;'>(based on title & date filtering)</i>"
            )

        matchday   = get_matchday(match_date)
        academic_w = get_academic_week(match_date)

        reply = f"""
        {mode_badge}<br>
        ✅ Prediction for <b>{session['opponent']}</b>
        on <b>{session['date']}</b> at <b>{session['kickoff_hour']}:00</b>:<br><br>

        {weather_line}<br>
        📅 Matchday: ~{matchday} &nbsp;|&nbsp; Academic week: {academic_w}<br>
        🏆 Top opponent: {"Yes" if session['is_top'] else "No"}
        &nbsp;|&nbsp;
        🎟️ Promotion: {"Yes" if session['has_promotion'] else "No"}
        &nbsp;|&nbsp;
        📋 Form: {points_last_5:.0f} pts last 5
        <br>

        {session['form_note']}
        {news_block}

        <div style='background:#0d2a4a;padding:14px;border-radius:8px;margin-top:10px;'>
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

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)