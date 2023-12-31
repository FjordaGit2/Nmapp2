import streamlit as st
import subprocess
from PIL import Image
import pandas as pd



df = pd.read_csv("sample.csv")
# Define the column names you want to select
print(df.head())
columns_to_select = df.columns

# Construct a string with the column names
columns_to_select_str = ', '.join([f'"{col}"' for col in columns_to_select])

print(columns_to_select)

process3 = subprocess.Popen(["Rscript", "networkplot.R"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
result3 = process3.communicate()
print("Complated")

plot_image = Image.open("qgraph_plot.png")
#st.image(plot_image, caption='qgraph Plot', use_column_width=True)
