import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import requests


model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
flight_classes = pickle.load(open("flight_classes.pkl", "rb"))
route_classes = pickle.load(open("route_classes.pkl", "rb"))


flights = {
    "IndiGo 6E-203": "IndiGo 6E-203",
    "Air India AI-101": "Air India AI-101",
    "SpiceJet SG-456": "SpiceJet SG-456",
    "Vistara UK-707": "Vistara UK-707",
    "Akasa Air QP-112": "Akasa Air QP-112",
    "Emirates EK-500": "Emirates EK-500",
    "Qatar Airways QR-901": "Qatar Airways QR-901"
}


API_KEY = st.secrets["OPENWEATHER_API_KEY"]

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
    try:
        res = requests.get(url, timeout=5).json()
        return res.get("weather", [{}])[0].get("main", "Clear")
    except:
        return "Clear"


city_list = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad",
    "Chennai", "Kolkata", "Pune", "Ahmedabad","Nashik",
    "New York", "London", "Dubai", "Singapore","Korea","Bangkok",
    "Thailand","Kashmir","Jaipur","Spain","Pakistan","UK"
]


st.set_page_config(page_title="✈️ Flight AI", layout="wide")


st.markdown("""
<style>

.stApp {
    background: transparent;
}

.video-background {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -10;
    overflow: hidden;
}

.video-background video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
}

.main {
    position: relative;
    z-index: 1;
}

</style>

<div class="video-background">
    <video autoplay loop muted playsinline>
        <source src="https://res.cloudinary.com/duan6rknb/video/upload/v1776722344/15374383_3840_2160_32fps_gh3i0o.mp4" type="video/mp4">
    </video>
</div>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align:center;color:white;'>Flight Delay Prediction</h1>", unsafe_allow_html=True)

st.markdown('<div class="glass">', unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    flight = st.selectbox("Flight", list(flights.keys()))
    origin = st.selectbox("Origin City", city_list)

with col2:
    destination = st.selectbox("Destination City", city_list)
    departure_time = st.time_input("Departure Time")


day_map = {
    "Monday":1,"Tuesday":2,"Wednesday":3,
    "Thursday":4,"Friday":5,"Saturday":6,"Sunday":7
}

day_name = st.selectbox("Day", list(day_map.keys()))
day = day_map[day_name]


predict = st.button("Predict Delay")

if predict:

    hour_24 = departure_time.hour
    minute = departure_time.minute
    crs_dep_time = hour_24 * 100 + minute

    flight_val = abs(hash(flights[flight])) % len(flight_classes)

    route = origin + "_" + destination
    route_val = abs(hash(route)) % len(route_classes)

    is_peak = 1 if (7 <= hour_24 <= 10 or 17 <= hour_24 <= 21) else 0
    is_weekend = 1 if day in [6, 7] else 0

    tb = 0 if hour_24 <= 6 else 1 if hour_24 <= 12 else 2 if hour_24 <= 18 else 3

  
    weather_text = get_weather(origin)


    df = pd.DataFrame([[flight_val, route_val, day, crs_dep_time, is_peak, is_weekend, tb]],
                      columns=columns)

    prob = model.predict_proba(df)[0][1]

    st.divider()

    delay_prob = prob * 100
    ontime_prob = (1 - prob) * 100

    if prob > 0.5:
        st.error(f"⚠️ Flight Likely Delayed ({delay_prob:.2f}%)")
    else:
        st.success(f"✅ Flight Likely On Time ({ontime_prob:.2f}%)")

    st.info(f"🌦️ Weather in {origin}: {weather_text}")


    fig, ax = plt.subplots(figsize=(2.8, 2.8))

    ax.pie(
        [1-prob, prob],
        labels=["On Time", "Delayed"],
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 8}
    )

    centre = plt.Circle((0,0), 0.55, fc="white")
    fig.gca().add_artist(centre)

    ax.axis("equal")

    st.pyplot(fig, use_container_width=False)

st.markdown('</div>', unsafe_allow_html=True)
