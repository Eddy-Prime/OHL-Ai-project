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

# ── Session state ─────────────────────────────────────────────────────────────
session = {}

# ── Chat endpoint ─────────────────────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message', '').strip()
    step     = session.get('step', 'ask_opponent')

    if step == 'ask_opponent':
        matched, opp_avg = fuzzy_match(user_msg)
        if matched:
            session['opponent'] = matched
            session['opp_avg']  = opp_avg
            session['step']     = 'ask_date'
            reply = f"Got it — <b>{matched}</b> (historical avg: {opp_avg:.0f} tickets).<br>📅 What is the match date? <i>(YYYY-MM-DD)</i>"
        else:
            session['opponent'] = 'Unknown'
            session['opp_avg']  = float(GLOBAL_AVG)
            session['step']     = 'ask_date'
            reply = f"Unknown opponent — using global average ({GLOBAL_AVG:.0f} tickets).<br>📅 What is the match date? <i>(YYYY-MM-DD)</i>"

    elif step == 'ask_date':
        try:
            match_date = datetime.strptime(user_msg, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({'reply': "❌ Invalid date format. Please use <b>YYYY-MM-DD</b> (e.g. 2026-04-15)."})
        session['step'] = 'ask_kickoff'
        session['date'] = user_msg
        reply = f"📅 Date set: <b>{match_date.strftime('%A, %d %b %Y')}</b>.<br>⏰ What is the kickoff time? <i>(HH:MM, e.g. 20:45)</i>"

    elif step == 'ask_kickoff':
        try:
            kickoff_hour = int(user_msg.split(':')[0])
        except ValueError:
            return jsonify({'reply': "❌ Invalid time. Please use <b>HH:MM</b> format (e.g. 20:45)."})

        match_date    = datetime.strptime(session['date'], "%Y-%m-%d").date()
        academic_week = get_academic_week(match_date)
        matchday      = get_matchday(match_date)
        rain_mm, rain_source = get_rain(match_date)
        opp_avg       = session['opp_avg']

        if rain_source == "forecast_unavailable":
            weather_line = "🌤️ Weather unavailable (>16 days away) — using 0mm rain"
        else:
            weather_line = f"{rain_label(rain_mm)} ({rain_mm:.1f}mm)"

        X        = np.array([[opp_avg, academic_week, matchday, rain_mm, kickoff_hour]])
        X_scaled = scaler.transform(X)
        pred     = model.predict(X_scaled)[0]
        low      = max(0, round(pred - std_dev))
        high     = round(pred + std_dev)
        pred     = round(pred)

        reply = f"""
        ✅ Here's the prediction for <b>{session['opponent']}</b>
        on <b>{session['date']}</b> at <b>{user_msg}</b>:<br><br>
        {weather_line}<br>
        📅 Academic week: {academic_week} &nbsp;|&nbsp; Est. matchday: {matchday}<br><br>
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