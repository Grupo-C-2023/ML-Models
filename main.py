import streamlit as st
from multiapp import MultiApp
from deploy import home, svc_model

app = MultiApp()

st.markdown("# Inteligencia de Negocios - Equipo C - Semana 14 ")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("SVC Model", svc_model.app)

# Add more models
# app.add_app("Scrappe Twitter y Mineria de Textos", scrapping_twitter.app)

# The main app
app.run()
