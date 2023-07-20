# Proyecto de Modelos de Machine Learning con Streamlit

Este proyecto tiene como objetivo implementar varios modelos de machine learning utilizando Streamlit para crear una interfaz interactiva y visualizar los resultados.
Versión desplegada del sistema en: https://streamlit-bi.fly.dev/

## Equipo: C-2023-1
Docente Líder del Proyecto: Mg. Ing. Ernesto Cancho-Rodríguez, MBA de la George Washington University ecr@gwu.edu
- Vasquez Palomino, Ashel Joseph
- Gonza Soto, Raquel Stacy
- Cardenas Ramirez, Jean Carlo
- Alvarado Arroyo, Diego Akira
- Huerta Villalta, Jasmin Amparo
- Pairazaman Arias, Oscar Eduardo
- Zavala Sanchez, Diego Alonso

## Descripción

En este proyecto, desarrollamos una aplicación web utilizando Streamlit para implementar los siguientes modelos de machine learning:

- K-Means
- Árbol de Decisión (Decision Tree)
- LSTM (Long Short-Term Memory)
- Random Forest Regression
- SVM (Support Vector Machine)

Cada modelo se ha implementado en un módulo separado y se ha creado una interfaz interactiva en Streamlit para cargar los datos, ajustar los parámetros del modelo y visualizar los resultados.

## Requisitos de Instalación

- Python 3.7 o superior
- Streamlit
- Scikit-learn
- TensorFlow (para el modelo LSTM)
- Otros paquetes de Python mencionados en el archivo requirements.txt

Puedes instalar los requisitos ejecutando el siguiente comando:

```bash
pip install -r requirements.txt
```

## Instrucciones de Uso
Clona este repositorio en tu máquina local:

```bash
git clone https://github.com/Grupo-C-2023/ML-Models.git
```

Navega hasta el directorio del proyecto:
```bash
cd ML-Models
```

Instala las dependencias mencionadas en el archivo requirements.txt (revisa los requisitos de instalación):
```bash
pip install -r requirements.txt
```
Ejecuta la aplicación de Streamlit:

```bash
streamlit run app.py
```
Abre tu navegador web y visita la siguiente dirección:
```bash
http://localhost:8501
```
Interactúa con la aplicación seleccionando el modelo de machine learning, cargando los datos y ajustando los parámetros según sea necesario.
