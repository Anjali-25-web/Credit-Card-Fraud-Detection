import os
import pandas as pd
import numpy as np
import argparse
import json
import utils
from preprocessing import preprocess_data

class FraudPredictor:
    """Class to manage loading models and performing fraud predictions on transactions."""
    
    def __init__(self, model_name="best_model.joblib", scaler_name="scaler.joblib"):
        """
        Initializes the predictor by loading the classifier model and RobustScaler.
        
        Args:
            model_name (str): The filename of the model in models/.
            scaler_name (str): The filename of the scaler in models/.
        """
        self.model = utils.load_artifact(model_name)
        self.scaler = utils.load_artifact(scaler_name)
        
        if self.model is None or self.scaler is None:
            raise FileNotFoundError(
                "Model or Scaler not found! Please run train_model.py first to generate the artifacts."
            )
            
        # Get feature names from model if possible, otherwise use standard schema
        if hasattr(self.model, "feature_names_in_"):
            self.expected_features = list(self.model.feature_names_in_)
        else:
            self.expected_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

    def predict_single(self, transaction_dict):
        """
        Predicts fraud for a single transaction.
        
        Args:
            transaction_dict (dict): Keys must match V1-V28, Time, Amount.
            
        Returns:
            dict: Prediction, probabilities, confidence score, and risk level.
        """
        # Convert dictionary to DataFrame
        df_raw = pd.DataFrame([transaction_dict])
        
        # Perform prediction
        result = self.predict_batch(df_raw)
        return result.iloc[0].to_dict()

    def predict_batch(self, df_transactions):
        """
        Predicts fraud for a batch of transactions.
        
        Args:
            df_transactions (pd.DataFrame): Must contain V1-V28, Time, and Amount.
            
        Returns:
            pd.DataFrame: DataFrame containing original features plus prediction,
                          probability, confidence, and risk level.
        """
        # Align columns to match the expected features of the model
        missing_cols = [col for col in self.expected_features if col not in df_transactions.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required features: {missing_cols}")
            
        # Reorder columns to match training schema
        df_features = df_transactions[self.expected_features].copy()
        
        # Preprocess using the loaded scaler
        df_processed = preprocess_data(df_features, is_training=False, scaler=self.scaler)
        
        # Predict Class and Probability
        predictions = self.model.predict(df_processed)
        
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(df_processed)[:, 1]
        else:
            # Fallback if model doesn't support probabilities (should not happen for our set)
            decision = self.model.decision_function(df_processed)
            probabilities = 1 / (1 + np.exp(-decision)) # Sigmoid mapping
            
        results_df = df_transactions.copy()
        results_df['Fraud_Prediction'] = predictions.astype(int)
        results_df['Fraud_Probability'] = probabilities
        
        # Confidence score:
        # If predicted Fraud (1), confidence is the fraud probability
        # If predicted Legitimate (0), confidence is (1 - fraud probability)
        results_df['Confidence'] = np.where(
            predictions == 1, 
            probabilities, 
            1.0 - probabilities
        )
        
        # Risk level classification
        def get_risk_level(prob):
            if prob < 0.2:
                return 'Low Risk'
            elif prob < 0.7:
                return 'Medium Risk'
            else:
                return 'High Risk (Potential Fraud)'
                
        results_df['Risk_Level'] = results_df['Fraud_Probability'].apply(get_risk_level)
        
        return results_df

def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Prediction CLI")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV file or JSON string of single transaction.")
    parser.add_argument('--output', type=str, default=None, help="Path to save output CSV (for batch inputs).")
    parser.add_argument('--model', type=str, default="best_model.joblib", help="Model filename under models/ directory.")
    
    args = parser.parse_args()
    
    try:
        predictor = FraudPredictor(model_name=args.model)
        
        # Check if input is a JSON string
        if args.input.strip().startswith('{') and args.input.strip().endswith('}'):
            transaction = json.loads(args.input)
            res = predictor.predict_single(transaction)
            print("\nPrediction Result:")
            print(json.dumps(res, indent=4))
        else:
            # Assume file path
            if not os.path.exists(args.input):
                print(f"Error: Input file '{args.input}' not found.")
                return
            df_in = pd.read_csv(args.input)
            print(f"Loaded {len(df_in)} transactions from {args.input}. Predicting...")
            df_out = predictor.predict_batch(df_in)
            
            if args.output:
                df_out.to_csv(args.output, index=False)
                print(f"Results successfully saved to: {args.output}")
            else:
                # Print summary
                fraud_count = (df_out['Fraud_Prediction'] == 1).sum()
                print(f"Batch prediction finished. Found {fraud_count} potential fraud cases out of {len(df_out)} transactions.")
                print(df_out[['Fraud_Prediction', 'Fraud_Probability', 'Risk_Level']].head(10))
                
    except Exception as e:
        print(f"Execution failed: {str(e)}")

if __name__ == "__main__":
    main()
