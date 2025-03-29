import pandas as pd

class HealthDatabase:
    """Simple in-memory database for storing and retrieving data."""
    def __init__(self):
        self.data = None

    def load_data(self, file_path):
        """Load data into memory."""
        self.data = pd.read_csv(file_path)
        return self.data

    def save_predictions(self, predictions):
        """Save predictions to a file."""
        predictions.to_csv("../output/predictions.csv", index=False)

if __name__ == "__main__":
    db = HealthDatabase()
    df = db.load_data("../data/health_data.csv")
    print(df.head())