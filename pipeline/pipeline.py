import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Initialize Local MLflow Tracking System
mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))
mlflow.set_experiment("Iris_Kubeflow_Pipeline")

print("--- STARTING AUTOMATED ML WORKFLOW ---")

# ==========================================
# COMPONENT A: LOAD AND PREPROCESS DATA
# ==========================================
print("[Step 1/4] Executing Component A: Data Preprocessing...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split successfully. Training shapes: {X_train.shape}, Testing shapes: {X_test.shape}")

# Start MLflow Tracking Run
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"Active MLflow Run ID: {run_id}")

    # ==========================================
    # COMPONENT B: TRAIN MODEL
    # ==========================================
    print("[Step 2/4] Executing Component B: Model Training...")
    n_estimators = 100
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    mlflow.log_param("n_estimators", n_estimators)
    print("Model training completed successfully.")

    # ==========================================
    # COMPONENT C: EVALUATE MODEL
    # ==========================================
    print("[Step 3/4] Executing Component C: Model Evaluation...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    
    # Log metrics to tracker
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    print(f"Evaluation Metrics Logged -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")

    # ==========================================
    # COMPONENT D: MODEL PACKAGING & REGISTRY
    # ==========================================
    print("[Step 4/4] Executing Component D: Model Registry & Packaging...")
    model_name = "IrisRandomForestModel"
    
    # Log and register model artifact
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=model_name
    )
    print(f"Model successfully saved and sent to MLflow Registry as '{model_name}'.")

    # Connect client to manage lifecycle states (Staging -> Production)
    client = MlflowClient()
    latest_versions = client.get_latest_versions(model_name, stages=["None"])
    if latest_versions:
        current_version = latest_versions[0].version
        print(f"Transitioning Model Version {current_version} from Staging to Production...")
        
        # Transition stage as explicitly requested on Page 5
        client.transition_model_version_stage(
            name=model_name,
            version=current_version,
            stage="Production"
        )
        print("Lifecycle transition complete! Model status is now set to 'Production'.")

print("--- PIPELINE RUN COMPLETED PERFECTLY ---")
