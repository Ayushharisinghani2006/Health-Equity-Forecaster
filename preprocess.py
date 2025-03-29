import pandas as pd
import numpy as np
import os
import pickle
import time
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import shap
from src.data_processor import clean_data
from src.ml_models import train_model

def preprocess_data(input_path, output_dir):
    """
    Preprocess the health data and save precomputed artifacts for faster app loading.

    Args:
        input_path (str): Path to the input CSV file.
        output_dir (str): Directory to save precomputed artifacts.

    Raises:
        Exception: If any preprocessing step fails.
    """
    start_time = time.time()
    print(f"Debug: Starting preprocessing with input_path={input_path}, output_dir={output_dir}")

    # Load and clean data
    print("Debug: Loading and cleaning data...")
    try:
        load_start = time.time()
        df = pd.read_csv(input_path)
        print(f"Debug: Loaded raw data with {len(df)} rows and columns: {df.columns.tolist()}")
        print(f"Debug: Loading raw data took {time.time() - load_start:.2f} seconds")
    except Exception as e:
        print(f"Error: Failed to load CSV file: {str(e)}")
        raise

    try:
        clean_start = time.time()
        df = clean_data(input_path)
        print(f"Debug: Cleaned data with {len(df)} rows and columns: {df.columns.tolist()}")
        print(f"Debug: Data cleaning took {time.time() - clean_start:.2f} seconds")
    except Exception as e:
        print(f"Error: Failed to clean data: {str(e)}")
        raise

    # Anomaly Detection
    print("Debug: Performing anomaly detection...")
    try:
        anomaly_start = time.time()
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_features = df[["Hospitals_per_Capita", "Disease_Burden"]].to_numpy()
        print(f"Debug: Anomaly features shape: {anomaly_features.shape}")
        df["Anomaly"] = iso_forest.fit_predict(anomaly_features)
        print(f"Debug: Anomaly detection completed. Anomaly counts: {df['Anomaly'].value_counts().to_dict()}")
        print(f"Debug: Anomaly detection took {time.time() - anomaly_start:.2f} seconds")
    except Exception as e:
        print(f"Error: Failed during anomaly detection: {str(e)}")
        raise

    # Calculate Risk_Score
    print("Debug: Calculating Risk_Score...")
    try:
        risk_score_start = time.time()
        df["Risk_Score"] = (df["Disease_Rate"] * df["Population"]) / (df["Hospitals"] + 1)
        print(f"Debug: Risk_Score calculated. Sample values: {df['Risk_Score'].head().tolist()}")
        print(f"Debug: Risk_Score calculation took {time.time() - risk_score_start:.2f} seconds")
    except Exception as e:
        print(f"Error: Failed to calculate Risk_Score: {str(e)}")
        raise

    # Clustering with KMeans
    print("Debug: Performing clustering...")
    try:
        clustering_start = time.time()
        kmeans = KMeans(n_clusters=3, n_init=3, random_state=42)  # Reduced n_init for faster computation
        # Scale Risk_Score for better clustering
        scaler = StandardScaler()
        risk_score_scaled = scaler.fit_transform(df[["Risk_Score"]])
        df["Cluster"] = kmeans.fit_predict(risk_score_scaled)
        print(f"Debug: Clustering completed. Cluster counts: {df['Cluster'].value_counts().to_dict()}")
        print(f"Debug: Clustering took {time.time() - clustering_start:.2f} seconds")
    except Exception as e:
        print(f"Error: Failed during clustering: {str(e)}")
        raise

    # Precompute Elbow Method plot
    print("Debug: Precomputing Elbow Method plot...")
    try:
        elbow_start = time.time()
        inertias = []
        K = range(2, min(5, len(df)))
        print(f"Debug: K range for Elbow Method: {list(K)}")
        for k in K:
            kmeans_elbow = KMeans(n_clusters=k, n_init=3, random_state=42)
            kmeans_elbow.fit(risk_score_scaled)
            inertias.append(kmeans_elbow.inertia_)
        fig_elbow = px.line(x=list(K), y=inertias, title="Elbow Method for Optimal Clusters")
        elbow_plot_path = os.path.join(output_dir, "elbow_method.pkl")
        with open(elbow_plot_path, "wb") as f:
            pickle.dump(fig_elbow, f)
        print(f"Debug: Saved Elbow Method plot to {elbow_plot_path}")
        print(f"Debug: Elbow Method plot precomputation took {time.time() - elbow_start:.2f} seconds")
    except Exception as e:
        print(f"Error: Failed to precompute Elbow Method plot: {str(e)}")
        raise

    # Train model
    print("Debug: Training model...")
    try:
        model_start = time.time()
        model = train_model(df)
        print(f"Debug: Model training completed. Best score: {model['best_score_']:.2f}")
        print(f"Debug: Model training took {time.time() - model_start:.2f} seconds")
    except Exception as e:
        print(f"Error: Failed to train model: {str(e)}")
        raise

    # SHAP Explainability
    print("Debug: Computing SHAP values...")
    try:
        shap_start = time.time()
        X = df[["Population", "Hospitals_per_Capita", "Disease_Burden"]]
        # Adjust sample size for small dataset
        X_sample = X.sample(n=min(5, len(X)), random_state=42) if len(X) > 5 else X
        print(f"Debug: SHAP sample size: {len(X_sample)} rows")
        explainer = shap.LinearExplainer(model["best_model"], X_sample)
        shap_values = explainer.shap_values(X_sample)
        print(f"Debug: SHAP values computed. Shape: {shap_values.shape}")
        print(f"Debug: SHAP computation took {time.time() - shap_start:.2f} seconds")
    except Exception as e:
        print(f"Error: Failed to compute SHAP values: {str(e)}")
        raise

    # Save results
    print("Debug: Saving precomputed data...")
    try:
        save_start = time.time()
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "processed_data.csv"), index=False)
        print(f"Debug: Saved processed_data.csv to {output_dir}")
        
        with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        print(f"Debug: Saved model.pkl to {output_dir}")
        
        with open(os.path.join(output_dir, "shap_values.pkl"), "wb") as f:
            pickle.dump(shap_values, f)
        print(f"Debug: Saved shap_values.pkl to {output_dir}")
        
        with open(os.path.join(output_dir, "kmeans.pkl"), "wb") as f:
            pickle.dump(kmeans, f)
        print(f"Debug: Saved kmeans.pkl to {output_dir}")
        
        with open(os.path.join(output_dir, "iso_forest.pkl"), "wb") as f:
            pickle.dump(iso_forest, f)
        print(f"Debug: Saved iso_forest.pkl to {output_dir}")
        
        print(f"Debug: Saving precomputed data took {time.time() - save_start:.2f} seconds")
    except Exception as e:
        print(f"Error: Failed to save precomputed data: {str(e)}")
        raise

    print(f"Debug: Total preprocessing time: {time.time() - start_time:.2f} seconds")
    print("Debug: Preprocessing completed successfully.")

if __name__ == "__main__":
    input_path = "data/health_data.csv"
    output_dir = "data/preprocessed"
    preprocess_data(input_path, output_dir)