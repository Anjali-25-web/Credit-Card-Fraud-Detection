import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)

from preprocessing import preprocess_data, apply_smote
import utils

def train_and_evaluate_all():
    """
    Downloads the dataset if needed, cleans it, splits it,
    oversamples the training set with SMOTE, trains 4 models,
    evaluates them, and saves the models and comparison metrics.
    """
    print("--- Starting Credit Card Fraud Detection Training Pipeline ---")
    
    # 1. Download and load data
    csv_path = utils.download_dataset()
    print("Loading dataset into memory...")
    col_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    df = pd.read_csv(csv_path, header=None, names=col_names)
    print(f"Dataset shape: {df.shape}")
    
    # Keep a small sample of original test data for the Streamlit preset loader
    # This ensures we have a few actual fraud and legitimate cases saved separately
    print("Extracting presets for app...")
    fraud_presets = df[df['Class'] == 1].sample(n=10, random_state=42)
    legit_presets = df[df['Class'] == 0].sample(n=10, random_state=42)
    presets = pd.concat([fraud_presets, legit_presets]).reset_index(drop=True)
    utils.save_artifact(presets, "test_presets.joblib")
    
    # 2. Preprocessing & Splitting
    print("Preprocessing data (scaling & duplicate removal)...")
    # Preprocess training data (will fit RobustScaler and remove duplicates)
    df_clean, scaler = preprocess_data(df, is_training=True)
    utils.save_artifact(scaler, "scaler.joblib")
    
    # Split features and labels
    X = df_clean.drop('Class', axis=1)
    y = df_clean['Class']
    
    # 80/20 Train/Test split
    # Stratified split is essential due to extreme class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save the test set for external validation/app evaluation
    test_data = {'X_test': X_test, 'y_test': y_test}
    utils.save_artifact(test_data, "test_data.joblib")
    
    # 3. Apply SMOTE to training split
    print("Applying SMOTE to balance the training set...")
    X_train_res, y_train_res = apply_smote(X_train, y_train, random_state=42)
    
    # 4. Define models to train
    # Optimized hyper-parameters for speed and performance
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            random_state=42, 
            solver='liblinear'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, 
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=50, 
            max_depth=12, 
            n_jobs=-1, 
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            n_jobs=-1, 
            random_state=42,
            eval_metric='logloss'
        )
    }
    
    metrics_summary = {}
    roc_curves_data = {}
    
    # 5. Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining model: {name}...")
        start_time = time.time()
        model.fit(X_train_res, y_train_res)
        train_time = time.time() - start_time
        print(f"Finished training in {train_time:.2f} seconds.")
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Check if model has predict_proba
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # For models without proba (though all these have it), use decision function
            y_prob = model.decision_function(X_test)
            
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC curve points (downsample points to keep artifact size small)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        # Downsample ROC points for charting efficiency
        if len(fpr) > 500:
            indices = np.linspace(0, len(fpr) - 1, 500, dtype=int)
            fpr = fpr[indices]
            tpr = tpr[indices]
            
        # Save evaluation metrics
        metrics_summary[name] = {
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1-Score': float(f1),
            'ROC-AUC': float(roc_auc),
            'Training Time (s)': float(train_time),
            'Confusion Matrix': cm.tolist() # JSON serializable
        }
        
        # Save ROC points separately for the plotting utility
        roc_curves_data[name] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'roc_auc': float(roc_auc)
        }
        
        # Save individual model
        filename = f"{name.lower().replace(' ', '_')}.joblib"
        utils.save_artifact(model, filename)
        print(f"Saved {name} metrics: Acc={acc:.4f}, Prec={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")

    # 6. Rank and select best model based on F1-Score
    best_model_name = max(metrics_summary, key=lambda k: metrics_summary[k]['F1-Score'])
    best_f1 = metrics_summary[best_model_name]['F1-Score']
    print(f"\n>>> Best model selected by F1-Score: {best_model_name} (F1 = {best_f1:.4f}) <<<")
    
    # Save best model to a dedicated path
    best_model = models[best_model_name]
    utils.save_artifact(best_model, "best_model.joblib")
    
    # Save metadata dictionary
    meta_info = {
        'best_model_name': best_model_name,
        'metrics': metrics_summary,
        'roc_curves': roc_curves_data,
        'timestamp': time.time()
    }
    utils.save_artifact(meta_info, "training_metadata.joblib")
    
    # Print comparison table
    df_metrics = pd.DataFrame(metrics_summary).T
    print("\nModel Comparison Table:")
    print(df_metrics[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Training Time (s)']])
    print("\n--- Training Pipeline Completed Successfully! ---")
    
if __name__ == "__main__":
    train_and_evaluate_all()
