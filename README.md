# Health Equity Forecaster 🏥

**Health Equity Forecaster** is a data-driven web application designed to address healthcare disparities by identifying regions with limited healthcare access. Built for a hackathon, this app leverages machine learning, clustering, anomaly detection, and explainability techniques to provide actionable insights for policymakers and healthcare organizations. The app is optimized for fast loading using precomputed data and offers an interactive interface for exploring healthcare equity metrics, visualizations, and recommendations.

## Features

- **Data Overview**: View processed health data, including anomalies and healthcare equity metrics (Gini coefficient).
- **Visualizations**: Explore interactive plots, including the Elbow Method for clustering, hospitals per capita, and healthcare risk by region.
- **Geospatial Map**: Visualize healthcare risk and hospital distribution on an interactive map with toggleable layers.
- **Model Results**: Analyze machine learning model performance (Logistic Regression) with metrics, feature importance, and SHAP explainability.
- **Recommendations**: Get actionable insights on high-risk regions, anomalous regions, and clusters for resource allocation.
- **Feedback**: Submit feedback to suggest additional regions or data for future improvements.
- **Performance Optimization**: Uses precomputed data for fast loading (initial load time < 3 seconds).

## Tech Stack

- **Python**: Core programming language.
- **Streamlit**: Web app framework for the interactive interface.
- **Scikit-learn**: Machine learning (LogisticRegression, KMeans, IsolationForest).
- **SHAP**: Model explainability.
- **Plotly**: Interactive visualizations.
- **Folium**: Geospatial mapping.
- **Pandas/NumPy**: Data processing.
- **Matplotlib/Seaborn**: Static visualizations.


## Setup
### Prerequisites

- Python 3.8 or higher
- Git (for version control and deployment)

### Clone the Repository
```bashS
git clone https://github.com/<your-username>/HealthEquityForecaster.git

1. Create a virtual environment: `python -m venv venv`
2. Activate it: `venv\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`
4. python preprocess.py
5.. Run the app: `streamlit run app.py`

## Data
- `data/health_data.csv`: Input dataset with columns `Region`, `Population`, `Hospitals`, `Disease_Rate`, `Latitude`, `Longitude`.

## Output
- Visualizations: `output/visualizations/`
- Report: `output/social_impact_report.pdf`
- Feedback: `feedback.txt`

## Social Impact
This app addresses healthcare disparities by identifying underserved regions and providing data-driven recommendations for clinic deployment, aiming to improve access to healthcare services in high-risk areas.