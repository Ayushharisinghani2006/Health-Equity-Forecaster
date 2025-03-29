import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import pickle
import plotly.express as px
import folium
from streamlit_folium import st_folium
import shap
import matplotlib.pyplot as plt  # Added missing import
from src.visualizations import generate_visualizations
from src.report_generator import generate_report
from src.data_processor import clean_data
from src.ml_models import train_model

# Function to calculate Gini coefficient
def gini_coefficient(x):
    """Calculate the Gini coefficient for a given array."""
    x = np.array(x)
    n = len(x)
    if n == 0:
        return 0
    diffsum = np.sum([np.sum(np.abs(xi - x)) for xi in x])
    avg = np.mean(x)
    return diffsum / (2 * n * n * avg) if avg != 0 else 0

# Caching function for SHAP values
@st.cache_data
def compute_shap_values(X, model):
    """Compute SHAP values for the given data and model."""
    start_time = time.time()
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    st.write(f"Debug: SHAP computation took {time.time() - start_time:.2f} seconds")
    return shap_values

# Set page config for better presentation
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Health Equity Forecaster",
    page_icon="🏥"
)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    session_vars = {
        "df": None,
        "filtered_df": pd.DataFrame(),
        "model": None,
        "gini_hospitals": None,
        "fig_elbow": None,
        "fig1": None,
        "fig2": None,
        "fig3": None,
        "shap_values": None,
        "shap_plot_path": None
    }
    for key, value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Load data (precomputed or runtime)
def load_data():
    """Load precomputed data or compute at runtime if necessary."""
    start_time = time.time()
    ROOT_DIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
    data_dir = os.path.join(ROOT_DIR, "data")
    preprocessed_dir = os.path.join(data_dir, "preprocessed")
    data_path = os.path.join(preprocessed_dir, "processed_data.csv")

    st.write(f"Debug: Checking for precomputed data at: {data_path}")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)

    if os.path.exists(data_path):
        # Load precomputed data
        st.write("Debug: Loading precomputed data...")
        df = pd.read_csv(data_path)
        st.write(f"Debug: Number of rows in dataset: {len(df)}")

        required_columns = ["Region", "Population", "Hospitals", "Disease_Rate", "Latitude", "Longitude", "Anomaly", "Risk_Score", "Cluster"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Precomputed dataset is missing required columns: {missing_columns}")
            st.stop()

        # Load model and other artifacts
        with open(os.path.join(preprocessed_dir, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(preprocessed_dir, "shap_values.pkl"), "rb") as f:
            shap_values = pickle.load(f)

        # Compute Gini coefficient
        gini_hospitals = gini_coefficient(df["Hospitals_per_Capita"])
    else:
        # Fallback: Compute at runtime
        st.warning("Precomputed data not found. Computing data at runtime (this may take a while)...")
        if uploaded_file is None:
            st.error("No precomputed data found and no file uploaded. Please upload a dataset or run preprocess.py.")
            st.stop()

        # Save uploaded file
        with open(os.path.join(data_dir, "health_data.csv"), "wb") as f:
            f.write(uploaded_file.getbuffer())
        data_path = os.path.join(data_dir, "health_data.csv")

        # Load and validate data
        df = pd.read_csv(data_path)
        st.write(f"Debug: Number of rows in dataset: {len(df)}")

        required_columns = ["Region", "Population", "Hospitals", "Disease_Rate", "Latitude", "Longitude"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Uploaded dataset is missing required columns: {missing_columns}")
            st.stop()
        if (df["Hospitals"] < 0).any():
            st.error("Dataset contains invalid data: 'Hospitals' column has negative values.")
            st.stop()
        if (df["Population"] <= 0).any():
            st.error("Dataset contains invalid data: 'Population' column has zero or negative values.")
            st.stop()
        if df[["Latitude", "Longitude"]].isnull().any().any():
            st.error("Dataset contains invalid data: 'Latitude' or 'Longitude' column has missing values.")
            st.stop()

        # Process data
        step_start = time.time()
        df = clean_data(data_path)
        st.write(f"Debug: Data cleaning took {time.time() - step_start:.2f} seconds")

        # Calculate Risk_Score
        step_start = time.time()
        df["Risk_Score"] = (df["Disease_Rate"] * df["Population"]) / (df["Hospitals"] + 1)
        st.write(f"Debug: Risk_Score calculation took {time.time() - step_start:.2f} seconds")

        # Train model
        step_start = time.time()
        model = train_model(df)
        st.write(f"Debug: Model training took {time.time() - step_start:.2f} seconds")

        # SHAP Explainability
        step_start = time.time()
        X = df[["Population", "Hospitals_per_Capita", "Disease_Burden"]]
        X_sample = X.sample(n=min(5, len(X)), random_state=42) if len(X) > 5 else X
        shap_values = compute_shap_values(X_sample, model["best_model"])
        st.write(f"Debug: SHAP computation took {time.time() - step_start:.2f} seconds")

        # Compute Gini coefficient
        gini_hospitals = gini_coefficient(df["Hospitals_per_Capita"])

    st.write(f"Debug: Initial data loading took {time.time() - start_time:.2f} seconds")
    return df, model, shap_values, gini_hospitals

# Apply filters to the DataFrame
def apply_filters(df):
    """Apply filters to the DataFrame based on user selections."""
    st.sidebar.title("Filters")
    regions = df["Region"].unique().tolist()
    selected_regions = st.sidebar.multiselect("Select Regions", regions, default=regions)
    risk_score_range = st.sidebar.slider(
        "Risk Score Range",
        float(df["Risk_Score"].min()),
        float(df["Risk_Score"].max()),
        (float(df["Risk_Score"].min()), float(df["Risk_Score"].max()))
    )
    clusters = df["Cluster"].unique().tolist()
    selected_clusters = st.sidebar.multiselect("Select Clusters", clusters, default=clusters)

    filtered_df = df[
        (df["Region"].isin(selected_regions)) &
        (df["Risk_Score"].between(risk_score_range[0], risk_score_range[1])) &
        (df["Cluster"].isin(selected_clusters))
    ]

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust the filters.")
        st.stop()

    return filtered_df

# Main app logic
def main():
    """Main function to run the Streamlit app."""
    # Initialize session state
    initialize_session_state()

    # File uploader
    st.title("Health Equity Forecaster")
    st.write("The app uses precomputed data for faster loading. If you upload a new dataset, please run preprocess.py first.")
    global uploaded_file
    uploaded_file = st.file_uploader("Upload health data CSV (optional)", type="csv", accept_multiple_files=False, key="health_data")

    # Load data
    if st.session_state.df is None:
        with st.spinner("Loading data..."):
            df, model, shap_values, gini_hospitals = load_data()
            st.session_state.df = df
            st.session_state.model = model
            st.session_state.shap_values = shap_values
            st.session_state.gini_hospitals = gini_hospitals

    # Use session state data
    df = st.session_state.df
    model = st.session_state.model
    shap_values = st.session_state.shap_values
    gini_hospitals = st.session_state.gini_hospitals

    # Apply filters
    filtered_df = apply_filters(df)
    st.session_state.filtered_df = filtered_df

    # Define paths
    ROOT_DIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
    output_dir = os.path.join(ROOT_DIR, "output")
    feedback_path = os.path.join(ROOT_DIR, "feedback.txt")
    report_path = os.path.join(output_dir, "social_impact_report.pdf")
    preprocessed_dir = os.path.join(ROOT_DIR, "data", "preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Overview", "Visualizations", "Geospatial Map", "Model Results", "Recommendations", "Feedback"])

    # Page Navigation
    if page == "Home":
        st.header("Welcome to Health Equity Forecaster")
        st.write("""
            This app identifies regions with limited healthcare access and provides actionable recommendations using machine learning, clustering, and explainability.
            - **Data Overview**: Explore the processed dataset and anomalies.
            - **Visualizations**: View clustering and risk visualizations.
            - **Geospatial Map**: See risk and hospital distribution on a map.
            - **Model Results**: Check model performance and feature importance.
            - **Recommendations**: Get actionable insights for resource allocation.
            - **Feedback**: Share your thoughts and suggestions.
        """)

    elif page == "Data Overview":
        st.header("Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Processed Data")
            st.dataframe(filtered_df)
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data",
                data=csv,
                file_name="filtered_health_data.csv",
                mime="text/csv"
            )
        with col2:
            st.write("### Anomalies Detected")
            anomalies = filtered_df[filtered_df["Anomaly"] == -1][["Region", "Hospitals_per_Capita", "Disease_Burden"]]
            if not anomalies.empty:
                st.dataframe(anomalies)
            else:
                st.write("No anomalies detected in the filtered data.")
        st.write("### Healthcare Equity Metric")
        st.write(f"Gini Coefficient of Hospital Distribution: {gini_hospitals:.2f}")
        st.write("A higher Gini coefficient indicates greater inequality in hospital distribution across regions.")

    elif page == "Visualizations":
        st.header("Visualizations")
        # Load precomputed Elbow Method plot
        if st.session_state.fig_elbow is None:
            start_time = time.time()
            with st.spinner("Loading Elbow Method plot..."):
                elbow_plot_path = os.path.join(preprocessed_dir, "elbow_method.pkl")
                st.write(f"Debug: Loading Elbow Method plot from: {elbow_plot_path}")
                if os.path.exists(elbow_plot_path):
                    with open(elbow_plot_path, "rb") as f:
                        fig_elbow = pickle.load(f)
                    st.session_state.fig_elbow = fig_elbow
                else:
                    st.error("Precomputed Elbow Method plot not found. Please run preprocess.py to generate it.")
                    st.stop()
            st.write(f"Debug: Elbow Method plot loading took {time.time() - start_time:.2f} seconds")
        st.write("### Optimal Number of Clusters (Elbow Method)")
        st.plotly_chart(st.session_state.fig_elbow)

        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.fig1 is None:
                start_time = time.time()
                fig1 = px.bar(filtered_df, x="Region", y="Hospitals_per_Capita", color="Cluster", title="Hospitals per Capita by Region")
                fig1.add_scatter(x=filtered_df["Region"], y=filtered_df["Population"]/1000, name="Population (k)", yaxis="y2")
                fig1.update_layout(
                    yaxis=dict(title="Hospitals per Capita"),
                    yaxis2=dict(title="Population (thousands)", overlaying="y", side="right"),
                    title="Hospitals per Capita and Population by Region"
                )
                st.session_state.fig1 = fig1
                st.write(f"Debug: Bar plot generation took {time.time() - start_time:.2f} seconds")
            st.plotly_chart(st.session_state.fig1)
        with col2:
            if st.session_state.fig2 is None:
                start_time = time.time()
                fig2 = px.scatter(filtered_df, x="Region", y="Risk_Score", color="Cluster", size="Hospitals", title="Healthcare Risk by Region (Clustered)")
                st.session_state.fig2 = fig2
                st.write(f"Debug: Scatter plot generation took {time.time() - start_time:.2f} seconds")
            st.plotly_chart(st.session_state.fig2)

    elif page == "Geospatial Map":
        st.header("Geospatial Risk Map")
        start_time = time.time()
        m = folium.Map(location=[filtered_df["Latitude"].mean(), filtered_df["Longitude"].mean()], zoom_start=5)

        # Layer 1: Risk Score
        risk_layer = folium.FeatureGroup(name="Risk Score", show=True)
        for idx, row in filtered_df.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=row["Risk_Score"] / filtered_df["Risk_Score"].max() * 20,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.6,
                popup=f"Region: {row['Region']}<br>Risk Score: {row['Risk_Score']:.2f}"
            ).add_to(risk_layer)
        risk_layer.add_to(m)

        # Layer 2: Hospitals per Capita
        hospitals_layer = folium.FeatureGroup(name="Hospitals per Capita", show=False)
        for idx, row in filtered_df.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=row["Hospitals_per_Capita"] / filtered_df["Hospitals_per_Capita"].max() * 20,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.6,
                popup=f"Region: {row['Region']}<br>Hospitals per Capita: {row['Hospitals_per_Capita']:.2f}"
            ).add_to(hospitals_layer)
        hospitals_layer.add_to(m)

        # Add Layer Control
        folium.LayerControl().add_to(m)

        st_folium(m, width=700, height=500)
        st.write(f"Debug: Geospatial map generation took {time.time() - start_time:.2f} seconds")

    elif page == "Model Results":
        st.header("Model Results")
        start_time = time.time()
        st.write(f"Model Accuracy: {model['best_score_']:.2f}")
        st.write(f"Precision: {model['precision']:.2f}")
        st.write(f"Recall: {model['recall']:.2f}")
        st.write(f"F1-Score: {model['f1']:.2f}")

        if st.session_state.fig3 is None:
            plot_start = time.time()
            features = ["Population", "Hospitals_per_Capita", "Disease_Burden"]
            coef = model["best_model"].coef_
            if len(coef.shape) == 1:
                importance = coef
            else:
                importance = coef[0]
            if len(importance) != len(features):
                st.warning(f"Warning: Number of features ({len(features)}) does not match number of coefficients ({len(importance)}).")
                importance = importance[:len(features)]
            feature_importance = pd.DataFrame({
                "Feature": features,
                "Importance": importance
            })
            fig3 = px.bar(feature_importance, x="Feature", y="Importance", title="Feature Importance")
            st.session_state.fig3 = fig3
            st.write(f"Debug: Feature importance plot generation took {time.time() - plot_start:.2f} seconds")
        st.write("### Feature Importance")
        st.plotly_chart(st.session_state.fig3)

        if st.session_state.shap_plot_path is None:
            shap_plot_start = time.time()
            X = df[["Population", "Hospitals_per_Capita", "Disease_Burden"]]
            X_sample = X.sample(n=min(5, len(X)), random_state=42) if len(X) > 5 else X
            st.write(f"Debug: SHAP sample size: {len(X_sample)} rows")
            shap_plot_path = os.path.join(output_dir, "visualizations", "shap_summary.png")
            os.makedirs(os.path.dirname(shap_plot_path), exist_ok=True)
            st.write(f"Debug: Generating SHAP summary plot at: {shap_plot_path}")
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            save_start = time.time()
            plt.savefig(shap_plot_path, dpi=150, bbox_inches="tight")
            st.write(f"Debug: Saving SHAP plot took {time.time() - save_start:.2f} seconds")
            plt.close()
            st.session_state.shap_plot_path = shap_plot_path
            st.write(f"Debug: SHAP plot generation took {time.time() - shap_plot_start:.2f} seconds")

        st.write("### Model Explainability (SHAP Values)")
        if os.path.exists(st.session_state.shap_plot_path):
            st.image(st.session_state.shap_plot_path)
        else:
            st.warning("SHAP plot not found. Please ensure it was generated successfully.")
        st.write(f"Debug: Model Results page took {time.time() - start_time:.2f} seconds")

    elif page == "Recommendations":
        st.header("Recommendations")
        start_time = time.time()
        high_risk_regions = filtered_df[filtered_df["Prediction"] == 1]["Region"].tolist()
        anomaly_regions = filtered_df[filtered_df["Anomaly"] == -1]["Region"].tolist()
        high_risk_clusters = filtered_df[filtered_df["Cluster"] == filtered_df["Cluster"].max()]["Region"].tolist()

        st.write("#### High-Risk Regions (Model Prediction)")
        st.write(f"Deploy clinics to: {', '.join(high_risk_regions) if high_risk_regions else 'None'}")

        st.write("#### Anomalous Regions (Outliers)")
        st.write(f"Investigate healthcare access in: {', '.join(anomaly_regions) if anomaly_regions else 'None'}")

        st.write("#### High-Risk Clusters (KMeans)")
        st.write(f"Prioritize resource allocation to: {', '.join(high_risk_clusters) if high_risk_clusters else 'None'}")
        st.write(f"Debug: Recommendations generation took {time.time() - start_time:.2f} seconds")

        if st.button("Generate Visualizations and Report"):
            with st.spinner("Generating visualizations..."):
                start_time = time.time()
                try:
                    generate_visualizations(df)
                    st.success("Visualizations generated successfully.")
                except Exception as e:
                    st.error(f"Error in generating visualizations: {str(e)}")
                    st.stop()
                st.write(f"Debug: Visualization generation took {time.time() - start_time:.2f} seconds")

            with st.spinner("Generating report..."):
                start_time = time.time()
                try:
                    generate_report(df, model)
                    time.sleep(1)
                except Exception as e:
                    st.error(f"Failed to generate the report: {str(e)}")
                    st.stop()
                st.write(f"Debug: Report generation took {time.time() - start_time:.2f} seconds")

            max_attempts = 3
            for attempt in range(max_attempts):
                if os.path.exists(report_path):
                    st.success(f"Report generated at {report_path}")
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="Download Report",
                            data=f,
                            file_name="social_impact_report.pdf",
                            mime="application/pdf"
                        )
                    break
                else:
                    st.warning(f"Attempt {attempt + 1}/{max_attempts}: Report file not found at {report_path}. Retrying in 1 second...")
                    st.write(f"Debug: Contents of output directory ({output_dir}): {os.listdir(output_dir)}")
                    time.sleep(1)
            else:
                st.error(f"Report file not found at {report_path} after {max_attempts} attempts. Please ensure the report was generated successfully.")

    elif page == "Feedback":
        st.header("Community Feedback")
        start_time = time.time()
        feedback = st.text_area("Provide feedback or suggest additional regions/data:")
        if st.button("Submit Feedback"):
            with open(feedback_path, "a") as f:
                f.write(f"{feedback}\n")
            st.success(f"Thank you for your feedback! Saved to {feedback_path}")

        st.write("### Feedback Dashboard")
        if os.path.exists(feedback_path):
            with open(feedback_path, "r") as f:
                feedback_list = f.readlines()
            if feedback_list:
                feedback_df = pd.DataFrame({"Feedback": [fb.strip() for fb in feedback_list]})
                st.write("#### Submitted Feedback")
                st.dataframe(feedback_df)
            else:
                st.write("No feedback submitted yet.")
        else:
            st.write("No feedback submitted yet.")
        st.write(f"Debug: Feedback page took {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")