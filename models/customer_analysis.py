import pandas as pd
import numpy as np
from models.churn_model import load_churn_model, predict_churn
import os

def analyze_customers():
    """
    Analyze all customers in the dataset and return their churn probabilities and top influencing features.
    
    Returns:
        DataFrame: Contains customer ID, churn risk, and top 3 influencing features
    """
    try:
        # Load the model
        model = load_churn_model()
        
        # Load the data
        data_path = "models/E Commerce Dataset2.xlsx"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found at: {data_path}")
            
        df = pd.read_excel(data_path)
        
        # Check if CustomerID column exists
        if 'CustomerID' not in df.columns:
            raise ValueError("CustomerID column not found in the dataset")
        
        # Get feature columns (excluding CustomerID and target column if exists)
        feature_cols = [col for col in df.columns if col not in ['CustomerID', 'Churn']]
        
        if not feature_cols:
            raise ValueError("No feature columns found in the dataset")
        
        # Get predictions
        X = df[feature_cols]
        probabilities = predict_churn(model, X)
        
        # Get feature importance
        feature_importance = pd.Series(model.feature_importances_, index=feature_cols)
        
        # Create results DataFrame
        results = []
        for idx, row in df.iterrows():
            customer_id = row['CustomerID']
            churn_risk = probabilities[idx]
            
            # Get top 3 features for this customer
            top_features = feature_importance.nlargest(3)
            
            results.append({
                'CustomerID': customer_id,
                'Churn Risk': churn_risk,
                'Top Feature 1': top_features.index[0],
                'Importance 1': top_features.iloc[0],
                'Top Feature 2': top_features.index[1],
                'Importance 2': top_features.iloc[1],
                'Top Feature 3': top_features.index[2],
                'Importance 3': top_features.iloc[2]
            })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"Error in analyze_customers: {str(e)}")
        raise  # Re-raise the exception to see the full traceback