
import sys
import numpy
import pandas
import rasterio
import geopandas
import matplotlib
import plotly
import streamlit
import folium

from src.config import PROJECT_ROOT

print("=" * 50)
print("NMMC ECOLOGICAL ANALYTICS")
print("=" * 50)

print("Python:", sys.version.split()[0])
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Rasterio:", rasterio.__version__)
print("GeoPandas:", geopandas.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Plotly:", plotly.__version__)
print("Streamlit:", streamlit.__version__)
print("Folium:", folium.__version__)

print("Project directory:", PROJECT_ROOT)
print("Environment test successful!")
