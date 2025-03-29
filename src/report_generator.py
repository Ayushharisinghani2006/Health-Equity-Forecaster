from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
from src.data_processor import clean_data
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def generate_report(df, model):
    """
    Generate a 10-slide PDF report for the Health Equity Forecaster project.
    
    Args:
        df (pd.DataFrame): Processed DataFrame with predictions and clusters.
        model (dict): Dictionary containing model metrics and the trained model.
    """
    print("Debug: Starting report generation")
    try:
        # Define the root directory (move up one level from src to the project root)
        ROOT_DIR = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        print(f"Debug: ROOT_DIR: {ROOT_DIR}")

        # Use absolute path for output directory
        output_dir = os.path.join(ROOT_DIR, "output")
        print(f"Debug: Output directory: {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Debug: Created output directory: {output_dir}")

        # Define the PDF path
        pdf_path = os.path.join(output_dir, "social_impact_report.pdf")
        print(f"Debug: Creating PDF at {pdf_path}")

        # Define the visualizations path
        viz_path = os.path.join(output_dir, "visualizations")
        hospital_img = os.path.join(viz_path, "hospitals_per_region.png")
        risk_img = os.path.join(viz_path, "risk_map.png")
        elbow_img = os.path.join(viz_path, "elbow_method.png")
        feature_img = os.path.join(viz_path, "feature_importance.png")
        shap_img = os.path.join(viz_path, "shap_summary.png")
        conf_matrix_path = os.path.join(viz_path, "confusion_matrix.png")

        # Wait for visualizations to be generated
        for _ in range(5):
            if all(os.path.exists(img) for img in [hospital_img, risk_img, elbow_img, feature_img, shap_img]):
                break
            print(f"Debug: Waiting for visualizations to be generated...")
            time.sleep(1)

        # Create the PDF
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Slide 1: Cover Page
        elements.append(Paragraph("Health Equity Forecaster Report", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", styles["Normal"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("An analysis of healthcare equity to identify high-risk regions and provide recommendations for resource allocation.", styles["Normal"]))
        elements.append(Spacer(1, 36))

        # Slide 2: Analysis Overview
        elements.append(Paragraph("Analysis Overview", styles["Heading1"]))
        elements.append(Paragraph(f"Number of regions analyzed: {len(df)}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Slide 3-4: EDA
        elements.append(Paragraph("Healthcare Distribution", styles["Heading1"]))
        if os.path.exists(hospital_img):
            elements.append(Image(hospital_img, width=400, height=300))
        else:
            elements.append(Paragraph("Hospital visualization not available.", styles["Normal"]))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Risk Analysis", styles["Heading1"]))
        if os.path.exists(risk_img):
            elements.append(Image(risk_img, width=400, height=300))
        else:
            elements.append(Paragraph("Risk map visualization not available.", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Slide 5: Clustering Analysis
        elements.append(Paragraph("Clustering Analysis", styles["Heading1"]))
        if os.path.exists(elbow_img):
            elements.append(Paragraph("Elbow Method for Optimal Clusters", styles["Heading2"]))
            elements.append(Image(elbow_img, width=400, height=300))
        else:
            elements.append(Paragraph("Elbow method visualization not available.", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Slide 6-7: Model Results
        elements.append(Paragraph("Prediction Results", styles["Heading1"]))
        elements.append(Paragraph(f"Model Accuracy: {model['best_score_']:.2f}", styles["Normal"]))
        elements.append(Paragraph(f"Precision: {model['precision']:.2f}", styles["Normal"]))
        elements.append(Paragraph(f"Recall: {model['recall']:.2f}", styles["Normal"]))
        elements.append(Paragraph(f"F1-Score: {model['f1']:.2f}", styles["Normal"]))
        if os.path.exists(conf_matrix_path):
            elements.append(Paragraph("Confusion Matrix", styles["Heading2"]))
            elements.append(Image(conf_matrix_path, width=400, height=300))
        else:
            # Generate confusion matrix if not already saved
            fig, ax = plt.subplots()
            sns.heatmap(model['confusion_matrix'], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            plt.savefig(conf_matrix_path, dpi=300, bbox_inches="tight")
            plt.close()
            if os.path.exists(conf_matrix_path):
                elements.append(Paragraph("Confusion Matrix", styles["Heading2"]))
                elements.append(Image(conf_matrix_path, width=400, height=300))
            else:
                elements.append(Paragraph("Confusion matrix visualization not available.", styles["Normal"]))
        elements.append(Spacer(1, 12))

        if os.path.exists(feature_img):
            elements.append(Paragraph("Feature Importance", styles["Heading2"]))
            elements.append(Image(feature_img, width=400, height=300))
        else:
            elements.append(Paragraph("Feature importance visualization not available.", styles["Normal"]))
        elements.append(Spacer(1, 12))

        if os.path.exists(shap_img):
            elements.append(Paragraph("Feature Importance (SHAP)", styles["Heading2"]))
            elements.append(Image(shap_img, width=400, height=300))
        else:
            elements.append(Paragraph("SHAP visualization not available.", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Slide 8-9: Recommendations
        elements.append(Paragraph("Recommendations", styles["Heading1"]))
        high_risk = df[df["Prediction"] == 1]["Region"].tolist()
        anomaly_regions = df[df["Anomaly"] == -1]["Region"].tolist()
        high_risk_clusters = df[df["Cluster"] == df["Cluster"].max()]["Region"].tolist()
        elements.append(Paragraph(f"Deploy clinics to: {', '.join(high_risk) if high_risk else 'None'}", styles["Normal"]))
        elements.append(Paragraph(f"Investigate healthcare access in: {', '.join(anomaly_regions) if anomaly_regions else 'None'}", styles["Normal"]))
        elements.append(Paragraph(f"Prioritize resource allocation to: {', '.join(high_risk_clusters) if high_risk_clusters else 'None'}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Slide 10: Summary
        elements.append(Paragraph("Summary", styles["Heading1"]))
        elements.append(Paragraph(f"Analyzed {len(df)} regions, identifying {len(high_risk)} high-risk regions for clinic deployment.", styles["Normal"]))
        elements.append(Paragraph(f"Model accuracy achieved: {model['best_score_']:.2f}.", styles["Normal"]))
        elements.append(Paragraph("Use these insights to improve healthcare equity by targeting resource allocation effectively.", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Build the PDF
        doc.build(elements)
        print(f"Debug: PDF created successfully at {pdf_path}")
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        raise

if __name__ == "__main__":
    df = pd.read_csv("data/health_data.csv")
    # Check for required columns
    required_columns = ["Region", "Population", "Hospitals", "Disease_Rate", "Latitude", "Longitude"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    df = clean_data("data/health_data.csv")
    from src.ml_models import train_model  # Import inside the block to avoid circular imports
    model = train_model(df)
    generate_report(df, model)