import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import streamlit as st

def app():
    st.title("K-Means Clustering - Análisis Bidimensional")

    st.write("Este es un ejemplo de aplicación de K-Means Clustering en un análisis bidimensional utilizando Streamlit.")

    pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S') # Se capta la fecha y hora actual

    # Commented out IPython magic to ensure Python compatibility.
    # # Conectando con Drive
    # %%time
    # from google.colab import drive
    # drive.mount("/content/drive")

    # Carga del dataset
    df = pd.read_csv('deploy/data/ClusteringBidimensional.csv')
    st.write('Dataset:')
    st.write(df.describe()) 

    # Perform MinMax normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[['Risk', 'Return']] = scaler.fit_transform(df[['Risk', 'Return']])

    # Commented out IPython magic to ensure Python compatibility.
    # # Perform clustering using K-means
    # %%time
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(df[['Risk', 'Return']])
    st.write("Número de clusters:", num_clusters)
    st.write("Centroides:")
    st.write(kmeans.cluster_centers_)
    st.write("Etiquetas de cluster asignadas a cada muestra:")
    st.write(kmeans.labels_)

    # Get the cluster labels
    labels = kmeans.labels_

    # Add the cluster labels to the DataFrame
    df['Cluster_identificado'] = labels

    st.write("Dataset con etiquetas de cluster:")
    st.write(df)

    # Get the cluster centers
    centroids = kmeans.cluster_centers_

    # Crea el gráfico de dispersión con los clusters
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['Risk'], df['Return'], c=df['Cluster_identificado'], cmap='viridis')
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='red')
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.title('Clustering Result')
    ax.legend(*scatter.legend_elements(), title='Clusters')

    # Muestra el gráfico en la aplicación de Streamlit
    st.write("Gráfico de dispersión con los clusters:")
    st.pyplot(fig)

    # Plot the geometric polygon for each cluster
    for cluster in range(num_clusters):
        cluster_points = df[df['Cluster_identificado'] == cluster][['Risk', 'Return']].values
        hull = ConvexHull(cluster_points)
        polygon = Polygon(cluster_points[hull.vertices], edgecolor='blue', linewidth=1, fill=None)
        ax.add_patch(polygon)

    # Set the labels and title
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.title('Clustering Result')

    # Create a legend
    legend_elements = scatter.legend_elements()[0]
    legend_labels = ['Cluster {}'.format(i) for i in range(num_clusters)]
    ax.legend(legend_elements, legend_labels, loc='upper left')

    # Show the plot
    st.write("Gráfico de dispersión con los clusters y sus polígonos:")
    st.pyplot(fig)
    
    st.write("Recomendación:")
    st.write("El análisis de clustering realizado puede ayudar a identificar diferentes grupos en función de los niveles de riesgo y rendimiento. Esta información puede ser útil para la toma de decisiones y la segmentación de clientes o inversiones.")

