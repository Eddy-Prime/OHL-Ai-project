from flask import Flask, request, jsonify, send_from_directory
import pickle, json, difflib, requests
import numpy as np
from datetime import datetime, date

app = Flask(__name__)

# ── Load model & lookup ──────────────────────────────────────────────────────
with open('model.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model   = artifacts['model']
scaler  = artifacts['scaler']
std_dev = artifacts['std']
feats   = artifacts['features']

with open('opponent_lookup.json') as f:
    opponent_lookup = json.load(f)
GLOBAL_AVG = opponent_lookup.pop('__global_avg__')

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_academic_week(d: date) -> int:
    start = date(d.year if d.month >= 9 else d.year - 1, 9, 15)
    return max(1, (d - start).days // 7 + 1)

def get_matchday(d: date) -> int:
    start = date(d.year if d.month >= 7 else d.year - 1, 7, 25)
    return min(34, max(1, round((d - start).days / 7)))

def get_season_progress(matchday: int) -> float:
    return round(matchday / 34, 4)

def fuzzy_match(name: str):
    matches = difflib.get_close_matches(name, opponent_lookup.keys(), n=1, cutoff=0.3)
    if matches:
        return matches[0], float(opponent_lookup[matches[0]])
    for team in opponent_lookup:
        if name.lower() in team.lower() or team.lower() in name.lower():
            return team, float(opponent_lookup[team])
    return None, float(GLOBAL_AVG)

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
    if mm == 0:   return "☀️ No rain forecast"
    elif mm < 2:  return "🌦️ Light rain (minimal impact)"
    elif mm < 10: return "🌧️ Moderate rain (some fans may stay home)"
    else:         return "⛈️ Heavy rain (expect lower attendance)"

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
        session['date'] = user_msg
        session['step'] = 'ask_kickoff'
        reply = (
            f"📅 Date set: <b>{match_date.strftime('%A, %d %b %Y')}</b>.<br>"
            f"⏰ What is the kickoff time? <i>(HH:MM, e.g. 20:45)</i>"
        )

    # ── Step 3: Kickoff ───────────────────────────────────────────────────────
    elif step == 'ask_kickoff':
        try:
            kickoff_hour = int(user_msg.split(':')[0])
        except ValueError:
            return jsonify({'reply': "❌ Invalid time. Please use <b>HH:MM</b> format (e.g. 20:45)."})

        session['kickoff_hour'] = kickoff_hour
        session['step']         = 'ask_form'
        reply = (
            "⏰ Kickoff set.<br><br>"
            "📋 What are OHL's <b>last 5 match results</b>? <i>(e.g. W W L D W — oldest to newest)</i><br>"
            "<small style='color:#888'>Type <b>skip</b> to use season averages instead.</small>"
        )

    # ── Step 4: Form → Predict ────────────────────────────────────────────────
    elif step == 'ask_form':
        kickoff_hour  = session['kickoff_hour']
        match_date    = datetime.strptime(session['date'], "%Y-%m-%d").date()
        academic_week = get_academic_week(match_date)
        matchday      = get_matchday(match_date)
        season_prog   = get_season_progress(matchday)
        rain_mm, rain_source = get_rain(match_date)
        opp_avg       = session['opp_avg']
        is_top        = session['is_top']
        weekend       = is_weekend(match_date)

        # Form — parse or use defaults
        form_note = ""
        if user_msg.lower() == 'skip':
            points_last_5    = 7.5
            wins_last_3      = 1.0
            goal_diff_last_5 = 0.0
            form_note        = "<i style='color:#888;font-size:12px;'>* Using season average form.</i><br>"
        else:
            parsed = parse_form(user_msg)
            if parsed:
                points_last_5, wins_last_3 = parsed
                goal_diff_last_5 = 0.0
                form_note = (
                    f"<i style='color:#a0c4ff;font-size:12px;'>"
                    f"✅ Form used: {points_last_5} pts last 5 | {wins_last_3} wins last 3"
                    f"</i><br>"
                )
            else:
                points_last_5    = 7.5
                wins_last_3      = 1.0
                goal_diff_last_5 = 0.0
                form_note        = "<i style='color:#f4a261;font-size:12px;'>⚠️ Could not parse form — using season averages.</i><br>"

        # Composite features
        opp_str_norm     = 0.5
        form_norm        = points_last_5 / 15
        match_importance = round(0.5 * form_norm + 0.5 * season_prog, 4)
        match_attract    = round(0.4 * match_importance + 0.4 * opp_str_norm + 0.2 * is_top, 4)
        form_x_opp       = points_last_5 * is_top
        attendance_lag_1 = float(GLOBAL_AVG)
        has_promotion    = 0

        X = np.array([[
            opp_avg,
            academic_week,
            matchday,
            rain_mm,
            kickoff_hour,
            points_last_5,
            wins_last_3,
            goal_diff_last_5,
            match_importance,
            match_attract,
            form_x_opp,
            weekend,
            season_prog,
            has_promotion,
            attendance_lag_1,
        ]])

        X_scaled = scaler.transform(X)
        pred     = model.predict(X_scaled)[0]
        low      = max(0, round(pred - std_dev))
        high     = round(pred + std_dev)
        pred     = round(pred)

        weather_line = (
            "🌤️ Weather unavailable (>16 days away) — using 0mm rain"
            if rain_source == "forecast_unavailable"
            else f"{rain_label(rain_mm)} ({rain_mm:.1f}mm)"
        )

        reply = f"""
        ✅ Prediction for <b>{session['opponent']}</b>
        on <b>{session['date']}</b> at <b>{kickoff_hour}:00</b>:<br><br>
        {weather_line}<br>
        📅 Academic week: {academic_week} &nbsp;|&nbsp; Est. matchday: {matchday}<br>
        🏆 Top opponent: {"Yes" if is_top else "No"} &nbsp;|&nbsp; Weekend: {"Yes" if weekend else "No"}<br><br>
        {form_note}
        <div style='background:#0d2a4a;padding:12px;border-radius:8px;margin-top:8px;'>
            📊 <b>Predicted attendance: {pred:,}</b><br>
            📉 Minimum estimate: {low:,}<br>
            📈 Maximum estimate: {high:,}
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