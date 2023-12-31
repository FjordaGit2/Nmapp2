import streamlit as st
import pandas as pd


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


def main():
    st.title("Network Visualizer")

    # File upload section
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        st.info("File successfully uploaded!")

        # Load the uploaded file as a DataFrame
        df = load_file(uploaded_file)
        display_dataframe_preview(df)

        # Allow users to select columns
        selected_columns = st.multiselect("Select columns", df.columns)

        if selected_columns:
            print("Uploaded FIle: ", uploaded_file)
            print("Files Name: ", uploaded_file.name )
            display_selected_columns(df, selected_columns)

            print("Seelcted_Columns: ", selected_columns)



        



if __name__ == "__main__":
    main()