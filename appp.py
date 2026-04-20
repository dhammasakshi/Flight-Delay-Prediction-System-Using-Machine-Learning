import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import requests

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="✈️ Airline AI", layout="wide")

# ======================
# 🎥 VIDEO BACKGROUND
# ======================
st.markdown("""
<style>
video {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    z-index: -2;
}
.overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(0,0,0,0.55);
    z-index: -1;
}
[data-testid="stAppViewContainer"] { background: transparent; }

.block-container {
    background: rgba(0,0,0,0.35);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 2rem;
}

/* Premium Button */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 55px;
    font-size: 18px;
    font-weight: 600;
}
</style>

<video autoplay muted loop playsinline>
<source src="https://res.cloudinary.com/duan6rknb/video/upload/v1776722344/15374383_3840_2160_32fps_gh3i0o.mp4" type="video/mp4">
</video>

<div class="overlay"></div>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ======================
# FLIGHTS
# ======================
flights = [
    "IndiGo 6E-203",
    "Air India AI-101",
    "SpiceJet SG-456",
    "Vistara UK-707",
    "Akasa Air QP-112",
    "Emirates EK-500",
    "Qatar Airways QR-901"
]

# ======================
# WEATHER
# ======================
API_KEY = "8f411872094b57953205b69903aa797a"

def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
        res = requests.get(url).json()
        return res.get("weather", [{}])[0].get("main", "Clear")
    except:
        return "Clear"

# ======================
# UI
# ======================
st.title("✈️ Flight Delay Prediction System")

col1, col2 = st.columns(2)

with col1:
    flight = st.selectbox("✈️ Flight", flights)
    origin = st.text_input("🛫 Origin (City / Country)", "Nashik, India")

with col2:
    destination = st.text_input("🛬 Destination (City / Country)", "Pune, India")
    time_input = st.time_input("⏰ Departure Time")

# CENTER BUTTON
c1, c2, c3 = st.columns([1,2,1])
with c2:
    predict = st.button("🚀 Predict Delay", use_container_width=True)

# ======================
# PREDICTION
# ======================
if predict:

    if not origin or not destination:
        st.warning("Please enter both origin and destination")
        st.stop()

    hour = time_input.hour
    crs_dep_time = hour * 100

    flight_val = flights.index(flight)

    # WEATHER
    weather = get_weather(origin).lower()
    st.info(f"🌦️ Weather at {origin}: {weather.upper()}")

    weather_risk = 0
    if weather in ["rain", "storm"]:
        weather_risk = 0.25
    elif weather in ["fog", "mist"]:
        weather_risk = 0.15
    elif weather in ["clouds"]:
        weather_risk = 0.10

    # MODEL INPUT
    df = pd.DataFrame([[ 
        flight_val, 0, 1, crs_dep_time, 0, 0, 0
    ]], columns=columns)

    prob = model.predict_proba(df)[0][1]
    final_prob = min(prob + weather_risk, 1)

    st.divider()

    # RESULT
    if final_prob > 0.5:
        st.error(f"⚠️ Flight Delayed ({round(final_prob*100,2)}%)")
    else:
        st.success(f"✅ Flight On Time ({round((1-final_prob)*100,2)}%)")

    # ======================
    # 📊 GRAPH (MAIN FEATURE)
    # ======================
    # ======================
# 📊 MODERN SMALL DONUT GRAPH
# ======================
st.subheader("📊 Prediction Overview")

labels = ["On Time", "Delayed"]
values = [1 - final_prob, final_prob]

# Smaller figure
fig, ax = plt.subplots(figsize=(3.5, 3.5))

# Donut style
wedges, texts, autotexts = ax.pie(
    values,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops=dict(width=0.4),  # 🔥 makes donut
)

# Clean look
ax.axis('equal')

# Remove background
fig.patch.set_alpha(0)
ax.set_facecolor('none')

# Center it
c1, c2, c3 = st.columns([1,2,1])
with c2:
    st.pyplot(fig)

st.caption("AI-based probability of delay vs on-time performance.")