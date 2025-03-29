import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import os
import streamlit as st
import time

@st.cache_data
def generate_visualizations(df):
    """
    Generate visualizations for healthcare equity and save them to the output directory.
    
    Args:
        df (pd.DataFrame): Input DataFrame with columns including 
            "Region", "Population", "Hospitals", "Disease_Rate", "Cluster", "Risk_Score".
    
    Raises:
        ValueError: If required columns are missing.
        Exception: If visualization generation fails.
    """
    try:
        start_time = time.time()
        print("Debug: Starting generate_visualizations...")

        # Validate input DataFrame
        print("Debug: Validating input DataFrame...")
        required_columns = ["Region", "Population", "Hospitals", "Disease_Rate", "Cluster", "Risk_Score"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Debug: Missing columns detected: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        print(f"Debug: DataFrame validation completed. Shape: {df.shape}")

        # Define the root directory (move up one level from src to the project root)
        ROOT_DIR = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        print(f"Debug: ROOT_DIR in visualizations: {ROOT_DIR}")

        # Define the output directory for visualizations
        output_dir = os.path.join(ROOT_DIR, "output", "visualizations")
        print(f"Debug: Visualizations output directory: {output_dir}")

        # Create the output directory if it doesn't exist
        print("Debug: Creating output directory if it doesn't exist...")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Debug: Output directory created/exists: {output_dir}")

        # hospitals_per_region.png (stacked bar)
        print("Debug: Generating hospitals_per_region.png...")
        hospital_plot_start = time.time()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(df["Region"], df["Hospitals"], color="skyblue", label="Hospitals")
        ax2 = ax1.twinx()
        ax2.plot(df["Region"], df["Population"]/1000, color="orange", label="Population (k)")
        ax1.set_xlabel("Region")
        ax1.set_ylabel("Hospitals", color="skyblue")
        ax2.set_ylabel("Population (thousands)", color="orange")
        plt.title("Hospitals and Population per Region")
        plt.xticks(rotation=45, ha="right")
        fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
        hospital_plot_path = os.path.join(output_dir, "hospitals_per_region.png")
        plt.tight_layout()
        # Save the plot
        save_start = time.time()
        plt.savefig(hospital_plot_path, dpi=150, bbox_inches="tight")  # Reduced DPI for faster saving
        print(f"Debug: Saving hospitals_per_region.png took {time.time() - save_start:.2f} seconds")
        plt.close()
        print(f"Debug: Generated hospitals_per_region.png at {hospital_plot_path}")
        print(f"Debug: Total time for hospitals_per_region.png: {time.time() - hospital_plot_start:.2f} seconds")

        # risk_map.png (scatter with clustering)
        print("Debug: Generating risk_map.png...")
        risk_map_start = time.time()
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df["Region"], df["Risk_Score"], c=df["Cluster"], cmap="viridis", s=100)
        plt.colorbar(label="Risk Cluster")
        plt.xlabel("Region")
        plt.ylabel("Healthcare Risk Score")
        plt.title("Healthcare Risk by Region (Clustered)")
        plt.xticks(rotation=45, ha="right")
        risk_map_path = os.path.join(output_dir, "risk_map.png")
        plt.tight_layout()
        # Save the plot
        save_start = time.time()
        plt.savefig(risk_map_path, dpi=150, bbox_inches="tight")  # Reduced DPI for faster saving
        print(f"Debug: Saving risk_map.png took {time.time() - save_start:.2f} seconds")
        plt.close()
        print(f"Debug: Generated risk_map.png at {risk_map_path}")
        print(f"Debug: Total time for risk_map.png: {time.time() - risk_map_start:.2f} seconds")

        print(f"Debug: Total time for generate_visualizations: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"Error in generate_visualizations: {str(e)}")
        raise

if __name__ == "__main__":
    print("Debug: Running visualizations.py as main...")
    df = pd.read_csv("data/health_data.csv")
    required_columns = ["Region", "Population", "Hospitals", "Disease_Rate"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Debug: Missing columns in raw data: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Simulate preprocessing done in app.py
    print("Debug: Preprocessing data...")
    from src.data_processor import clean_data
    df = clean_data("data/health_data.csv")
    df["Risk_Score"] = (df["Disease_Rate"] * df["Population"]) / (df["Hospitals"] + 1)
    kmeans = KMeans(n_clusters=3, n_init=3, random_state=42)  # Reduced n_init for faster computation
    df["Cluster"] = kmeans.fit_predict(df[["Risk_Score"]])
    
    print("Debug: Generating visualizations...")
    generate_visualizations(df)