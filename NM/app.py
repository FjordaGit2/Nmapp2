import streamlit as st
import pandas as pd
import os
os.environ['R_HOME'] = '/usr/lib/R'
import threading
import rpy2.robjects as robjects
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from igraph import Graph
import matplotlib.pyplot as plt
import igraph as ig
from PIL import Image

# List of packages to install and load
packages_to_install = ["networktools", "smacof", "MPsychoR", "psych", "eigenmodel", "dplyr", "NetworkComparisonTest"]
libraries_to_load = ["networktools", "MPsychoR", "smacof", "qgraph", "psych", "eigenmodel", "dplyr", "ggplot2", "IsingFit"]

# Install necessary packages
for package in packages_to_install:
    robjects.r(f'if(!("{package}" %in% installed.packages())) install.packages("{package}")')

# Load necessary R libraries
for library in libraries_to_load:
    robjects.r(f'library({library})')


def load_file(uploaded_file):
    """Load the uploaded file as a DataFrame."""
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

def display_dataframe_preview(df):
    """Display a preview of the DataFrame."""
    st.write("Preview of the DataFrame:")
    st.write(df)

def display_selected_columns(df, selected_columns):
    """Display the selected columns from the DataFrame."""
    st.write("You selected the following columns:")
    st.write(selected_columns)

    # Display the preview of selected columns
    st.write("Preview of selected columns:")
    st.write(df[selected_columns])

def generate_network_plot(cor_matrix):
    # Create a directed graph from the correlation matrix
    G = nx.DiGraph()
    nodes = list(cor_matrix.columns)
    G.add_nodes_from(nodes)
    
    
    threshold = 0.2  # Adjust the threshold as needed
    #add edges
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if abs(cor_matrix.iloc[i, j]) > threshold: #calculate correlation matrix
                G.add_edge(nodes[i], nodes[j])

    # Position nodes using a spring layout
    pos = nx.spring_layout(G)

    # Create a Plotly figure for network visualization 
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=list(G.nodes()),
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # Color node points by the number of connections.
    node_colors = [len(adjacencies[1]) for node, adjacencies in enumerate(G.adjacency())]
    node_trace.marker.color = node_colors

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Network Plot',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        text="Network visualization",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002)]))
    
    st.plotly_chart(fig, use_container_width=True)

# New function to visualize network using igraph (translated from R to Python)
def visualize_network_with_igraph(cor_matrix):
    nodes = list(cor_matrix.columns)
    edges = [(nodes[i], nodes[j]) 
             for i in range(len(nodes)) 
             for j in range(i + 1, len(nodes)) 
             if abs(cor_matrix.iloc[i, j]) > 0.2]

    g = Graph.TupleList(edges, directed=True, weights=True)

    # Set the layout
    layout = g.layout_auto()

    # Plot using igraph
    ig_plot = go.Figure(go.Scatter(x=[layout[i][0] for i in range(len(layout))],
                                   y=[layout[i][1] for i in range(len(layout))],
                                   mode='markers+text',
                                   text=nodes,
                                   hoverinfo='text',
                                   marker=dict(size=10, color='blue')
                                   ))
    for edge in edges:
        src, tgt = edge
        src_index = nodes.index(src)
        tgt_index = nodes.index(tgt)
        ig_plot.add_trace(
            go.Scatter(x=[layout[src_index][0], layout[tgt_index][0]],
                       y=[layout[src_index][1], layout[tgt_index][1]],
                       mode='lines',
                       line=dict(width=1, color='gray')
                       ))
    ig_plot.update_layout(title='Network Plot (igraph)',
                          titlefont_size=16,
                          showlegend=False,
                          hovermode='closest',
                          margin=dict(b=20, l=5, r=5, t=40),
                          annotations=[dict(
                              text="Network visualization using igraph",
                              showarrow=False,
                              xref="paper", yref="paper",
                              x=0.005, y=-0.002)])
    st.plotly_chart(ig_plot, use_container_width=True)


def main():
    st.title("Network Visualizer")

    # File upload section
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        st.info("File successfully uploaded!")
        with open(uploaded_file.name, 'wb') as f:
            f.write(uploaded_file.getvalue())

        # Load the uploaded file as a DataFrame
        df = load_file(uploaded_file)
        display_dataframe_preview(df)

        # Allow users to select columns
        selected_columns = st.multiselect("Select columns", df.columns)

        if selected_columns:
            display_selected_columns(df, selected_columns)

            st.header('Network Visualization using NetworkX:')
            cor_matrix = df[selected_columns].corr()
            generate_network_plot(cor_matrix)

            st.header('Network Visualization using igraph:')
            visualize_network_with_igraph(cor_matrix)

            if len(selected_columns)>1:
                # Read the CSV file
                robjects.r(f'dt <- read.csv("{uploaded_file.name}", header=TRUE)')

                # Define the column names you want to select
                columns_to_select = selected_columns

                # Construct a string with the column names
                columns_to_select_str = ', '.join([f'"{col}"' for col in columns_to_select])

                # Use the constructed string in the R code to select columns
                robjects.r(f'netdt1 <- select(dt, {columns_to_select_str})')

                robjects.r('net1 <- qgraph(cor_auto(netdt1), n = nrow(netdt1), lambda.min.ratio = 0.05, default = "EBICglasso", layout="spring", vsize = 16, gamma = 0.2, tuning = 0.2, refit = TRUE)')


                # # Define the file path for the image
                # image_path = "graph_plot.png"

                # # Delete the existing image file if it exists
                # if os.path.exists(image_path):
                #     os.remove(image_path)

               # Plot the qgraph for the correlation difference and save it as an image
               
                try:
                    robjects.r(f'png("graph_plot.png", width=600, height=400)')
                    robjects.r('qgraph(net1, maximum=0.29)')
                    robjects.r('dev.off()')  # Close the PNG device
                except:
                  pass        
                        
                if os.path.exists("graph_plot.png"):
                    fig = px.imshow(Image.open("graph_plot.png"))
                    st.plotly_chart(fig, use_container_width=True)

                if os.path.exists("graph_plot.png"):
                    plot_image = Image.open("graph_plot.png")
                    st.image(plot_image, caption='Network Plot', use_column_width=True)

                # Integrate the provided R code only for net1
                robjects.r('centralityPlot(net1, include = c("Strength","Betweenness","Closeness"), orderBy = "Betweenness")')

                robjects.r('out1 <- expectedInf(cor_auto(netdt1))')

                robjects.r('par(mfrow=c(1,2))')
                robjects.r('plot(out1, order="value", zscore=TRUE, title = "Controls")')

                # Save and display the last graph
                try:
                    robjects.r(f'png("centrality.png", width=800, height=600)')
                    robjects.r('qgraph(net1, maximum=0.29)')
                    robjects.r('centralityPlot(net1, include = c("Strength","Betweenness","Closeness"), orderBy = "Betweenness")')
                    robjects.r('dev.off()')  # Close the PNG device
                except:
                    pass

                if os.path.exists("centrality.png"):
                    centrality_fig = px.imshow(Image.open("centrality.png"))
                    st.plotly_chart(centrality_fig, use_container_width=True)

                if os.path.exists("centrality.png"):
                    plot_image = Image.open("centrality.png")
                    st.image(plot_image, caption='Centrality Plot', use_column_width=True)

if __name__ == "__main__":
    main()