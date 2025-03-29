import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_model(df):
    """
    Train a Logistic Regression model to predict high-need areas with hyperparameter tuning.
    
    Args:
        df (pd.DataFrame): Input DataFrame with columns including 
            "Population", "Hospitals_per_Capita", "Disease_Burden".
    
    Returns:
        dict: Dictionary containing model metrics and the trained model.
    """
    try:
        # Validate input DataFrame
        required_columns = ["Population", "Hospitals_per_Capita", "Disease_Burden"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
        os.makedirs(output_dir, exist_ok=True)

        # Features and target (High Need if Hospitals_per_Capita is below median)
        X = df[["Population", "Hospitals_per_Capita", "Disease_Burden"]]
        y = (df["Hospitals_per_Capita"] < df["Hospitals_per_Capita"].median()).astype(int)

        # Debug statements
        print("Debug: Features used for training in train_model", X.columns.tolist())
        print("Debug: Shape of X", X.shape)
        print("Debug: Unique values in y", np.unique(y))

        # Check class distribution
        class_counts = np.bincount(y)
        print(f"Debug: Class distribution in y: {class_counts}")

        # Split the data
        # Use stratify to maintain class distribution in train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Define the model and hyperparameters
        model = LogisticRegression(random_state=42)
        param_grid = {
            "C": [0.1, 1, 10],
            "solver": ["liblinear"],  # Use only "liblinear" to support both l1 and l2 penalties
            "penalty": ["l1", "l2"]
        }

        # Determine the number of splits for cross-validation
        min_class_count = min(class_counts)  # Number of samples in the smallest class
        n_splits = min(3, min_class_count)  # Use at most 3 splits, but reduce if min_class_count is smaller
        print(f"Debug: Minimum class count: {min_class_count}, n_splits: {n_splits}")

        if n_splits < 2:
            print("Warning: Dataset is too small for cross-validation. Training without GridSearchCV.")
            model.fit(X_train, y_train)
            best_model = model
            best_score = accuracy_score(y_test, model.predict(X_test))
        else:
            print(f"Debug: Using {n_splits}-fold cross-validation.")
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=n_splits,  # Adjusted number of splits
                scoring="accuracy",
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_

        # Debug model coefficients
        print("Debug: Model coefficients shape after training", best_model.coef_.shape)
        print("Debug: Model coefficients", best_model.coef_)

        # Predict on test set
        y_pred = best_model.predict(X_test)

        # Compute metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Store metrics in a dictionary
        metrics = {
            "best_model": best_model,  # Store the model directly, not the GridSearchCV object
            "best_score_": best_score,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix
        }

        # Add predictions to DataFrame and save
        df["Prediction"] = best_model.predict(X)
        output_path = os.path.join(output_dir, "predictions.csv")
        df.to_csv(output_path, index=False)
        print(f"Debug: Predictions saved to {output_path}")

        return metrics

    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        raise

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "health_data.csv")
    df = pd.read_csv(data_path)
    metrics = train_model(df)
    print(f"Best model accuracy: {metrics['best_score_']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1-Score: {metrics['f1']:.2f}")