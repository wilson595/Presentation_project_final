import streamlit as st
from streamlit_folium import st_folium
import folium
import requests
from datetime import datetime, timedelta
import pytz
import pandas as pd
import plotly.express as px
from Simulacion_final_final import simuler_trajectoire

# 🌡️ Température et pression standard en fonction de l'altitude
def temperature_standard(h):
    T0 = 288.15  # en K
    lapse_rate = 0.0065  # K/m
    return T0 - lapse_rate * h


def pression_standard(h):
    T0 = 288.15
    P0 = 1013.25
    lapse_rate = 0.0065
    g = 9.80665
    M = 0.0289644
    R = 8.31447
    return P0 * (1 - (lapse_rate * h) / T0) ** ((g * M) / (R * lapse_rate))


def set_background_image():
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExMzIxcWV6OHVhb25yNDB6NXBoY29uaGU3a2tnMWFvdDRxbXQ5Z2lsZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7qDKdHAqamtq0uBi/giphy.gif");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


# 🧭 Convertir un angle en direction cardinale
def angle_to_direction(angle):
    directions = ['Nord', 'Nord-Est', 'Est', 'Sud-Est', 'Sud', 'Sud-Ouest', 'Ouest', 'Nord-Ouest']
    idx = int((angle + 22.5) % 360 / 45)
    return directions[idx]


# 🌍 Interface
st.set_page_config(layout="centered", page_title="Météo Drone Delivery")
set_background_image()
st.title("🌍 Sélectionnez un point de livraison sur la carte")

# 🗺️ Carte interactive
m = folium.Map(location=[48.85, 2.35], zoom_start=4)
folium.LatLngPopup().add_to(m)
st.markdown('<p style="color:blue">Cliquez sur la carte pour sélectionner les coordonnées.</p>', unsafe_allow_html=True)
map_data = st_folium(m, width=700, height=500)

# 📍 Si l'utilisateur clique
if map_data and map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    st.success(f"📍 Coordonnées sélectionnées : {lat:.4f}, {lon:.4f}")

    # 📡 Appel à l'API
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=wind_speed_10m,wind_direction_10m,"
        f"wind_speed_80m,wind_direction_80m,"
        f"wind_speed_120m,wind_direction_120m,"
        f"wind_speed_180m,wind_direction_180m"
        f"&timezone=auto"
    )

    try:
        response = requests.get(url).json()
        heures_disponibles = response["hourly"]["time"]
        heures_disponibles_dt = [datetime.fromisoformat(h) for h in heures_disponibles]

        # 📅 Sélection de la date de livraison
        st.subheader("🗓️ Sélection la date de la livraison")
        min_date = datetime.now().date()
        max_date = (datetime.now() + timedelta(days=7)).date()
        date_selectionnee = st.date_input(
            "Choisissez la date de livraison",
            min_value=min_date,
            max_value=max_date,
            value=min_date

        )

        # ⏰ Filtrage des heures disponibles
        heures_du_jour = [h for h in heures_disponibles_dt if h.date() == date_selectionnee]

        if heures_du_jour:
            heure_selectionnee = st.selectbox("Sélectionnez l'heure de la livraison", heures_du_jour)
            index_horaire = heures_disponibles_dt.index(heure_selectionnee)

            # 📊 Préparation des données
            altitudes = [10, 80, 120, 180]
            meteo_multi_alt = []

            for alt in altitudes:
                vitesse = response["hourly"].get(f"wind_speed_{alt}m", [None])[index_horaire]
                direction = response["hourly"].get(f"wind_direction_{alt}m", [None])[index_horaire]

                meteo_multi_alt.append({
                    "Altitude (m)": alt,
                    "Vitesse (m/s)": round(vitesse, 2) if vitesse else None,
                    "Direction (°)": round(direction) if direction else None,
                    "Direction": angle_to_direction(direction) if direction else None,
                    "Température (°C)": round(temperature_standard(alt) - 273.15, 2),
                    "Pression (kPa)": round(pression_standard(alt) / 10, 2)
                })

            # 🔽 Tri par altitude
            meteo_multi_alt.sort(key=lambda x: x['Altitude (m)'], reverse=True)
            df_meteo = pd.DataFrame(meteo_multi_alt)

            # 📊 Affichage du tableau
            st.subheader("📊 Données météorologiques")
            st.dataframe(
                df_meteo.style
                .background_gradient(subset=["Vitesse (m/s)"], cmap="Blues")
                .background_gradient(subset=["Température (°C)"], cmap="Reds"),
                width=800
            )

            # 📈 Graphiques
            st.subheader("📈 Visualisations")

            # Graphique de vitesse
            fig_vitesse = px.line(
                df_meteo,
                x="Altitude (m)",
                y="Vitesse (m/s)",
                title="Vitesse du vent par altitude",
                markers=True,
                color_discrete_sequence=["#3498DB"]
            )
            st.plotly_chart(fig_vitesse, use_container_width=True)

            # Rose des vents
            if all(df_meteo["Direction (°)"].notna()):
                fig_rose = px.bar_polar(
                    df_meteo,
                    r="Vitesse (m/s)",
                    theta="Direction (°)",
                    color="Altitude (m)",
                    title="Tendance du vent",
                    template="plotly_dark",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_rose, use_container_width=True)

            # 📍 Carte de localisation
            st.subheader("📍 Position final de livraison")
            m = folium.Map(location=[lat, lon], zoom_start=10)
            folium.Marker(
                [lat, lon],
                popup=f"Livraison: {date_selectionnee} {heure_selectionnee.strftime('%H:%M')}",
                icon=folium.Icon(color="green", icon="truck")
            ).add_to(m)
            st_folium(m, width=700, height=300)

        else:
            st.warning("Aucune donnée disponible pour cette date.")

    except Exception as e:
        st.error("Erreur lors de la récupération des données météo.")
        st.exception(e)
else:
    st.info("Veuillez sélectionner un point sur la carte pour commencer.")
# Initialisation d'un conteneur de session pour retenir le point sélectionné
if "clicked_point" not in st.session_state:
    st.session_state.clicked_point = None

# Si clic détecté → stocker dans session_state
if map_data["last_clicked"] is not None:
    st.session_state.clicked_point = map_data["last_clicked"]
    st.success(f"📍 Point sélectionné : lat = {st.session_state.clicked_point['lat']:.4f}, "
               f"lon = {st.session_state.clicked_point['lng']:.4f}")

if st.session_state.clicked_point:
    if st.button("🚀 Lancer la simulation"):
        lat = st.session_state.clicked_point["lat"]
        lon = st.session_state.clicked_point["lng"]

        with st.spinner("Simulation en cours..."):
            # Appelle ici ta vraie fonction de simulation
            x_star, erreur, (xf, yf), z_t, time = simuler_trajectoire(lat=lat, lon=lon)

        st.write(f"📍 Point d'atterrissage : ({xf:.2f}, {yf:.2f})")
        st.write(f"🎯 Erreur par rapport à la cible : {erreur:.2f} m")
        st.image("trajectoire.gif", caption="Animation 3D de la trajectoire")
        st.image("graph2D.png", caption="📉 Trajectoire au sol (2D)")
        st.image("graph3D.png", caption="📊 Trajectoire complète (3D)")
# 🎨 Style CSS personnalisé
st.markdown("""
<style>
/* Texte en rouge partout */
html, body, [class*="st-"], .stApp {
    color: red !important;
}

/* Éléments spécifiques */
h1, h2, h3, h4, h5, h6, p, div, span {
    color: red !important;
}

/* Forcer la couleur dans les tableaux */
thead tr th, tbody tr td {
    color: red !important;
}
</style>
""", unsafe_allow_html=True)
