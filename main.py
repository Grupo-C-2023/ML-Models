import streamlit as st
from multiapp import MultiApp
from deploy import home, svc_model, kmeans_model, random_forest_regression_model, decision_tree_model
app = MultiApp()

st.markdown("# Inteligencia de Negocios - Equipo C - Semana 14 ")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("SVC Model", svc_model.app)
app.add_app("KMeans Model", kmeans_model.app)
app.add_app("Random Forest Regression Model", random_forest_regression_model.app)
app.add_app("Decision Tree Model", decision_tree_model.app)

# Add more models
# app.add_app("Scrappe Twitter y Mineria de Textos", scrapping_twitter.app)

# The main app
app.run()
