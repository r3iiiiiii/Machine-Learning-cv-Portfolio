import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def load_trained_model(model_path='results/models.pkl'):
    """Load trained models"""
    return joblib.load(model_path)

def load_scaler(scaler_path='results/scaler.pkl'):
    """Load data scaler"""
    return joblib.load(scaler_path)

def predict_churn(model_dict, X_new):
    """Make predictions using ensemble"""
    xgb_model = model_dict['xgb']
    rf_model = model_dict['rf']
    lr_model = model_dict['lr']
    
    xgb_pred = xgb_model.predict_proba(X_new)[:, 1]
    rf_pred = rf_model.predict_proba(X_new)[:, 1]
    lr_pred = lr_model.predict_proba(X_new)[:, 1]
    
    ensemble_pred = 0.5 * xgb_pred + 0.3 * rf_pred + 0.2 * lr_pred
    
    return ensemble_pred

def get_churn_report(y_true, y_pred_proba, threshold=0.5):
    """Get classification report with business metrics"""
    y_pred = (y_pred_proba > threshold).astype(int)
    
    report = classification_report(y_true, y_pred, 
                                  target_names=['No Churn', 'Churn'])
    
    print(f"Classification Report (threshold={threshold}):")
    print(report)
    
    total_customers = len(y_true)
    churned_identified = (y_pred == 1).sum()
    churned_actual = (y_true == 1).sum()
    correct_identification = ((y_pred == 1) & (y_true == 1)).sum()
    
    print(f"\nBusiness Metrics:")
    print(f"  Total customers: {total_customers}")
    print(f"  Actual churners: {churned_actual} ({churned_actual/total_customers*100:.1f}%)")
    print(f"  Identified as churners: {churned_identified}")
    print(f"  Correctly identified: {correct_identification} ({correct_identification/churned_actual*100:.1f}%)")
    
    return {
        'total_customers': total_customers,
        'actual_churners': churned_actual,
        'identified_churners': churned_identified,
        'correct_identifications': correct_identification,
        'recall': correct_identification / churned_actual if churned_actual > 0 else 0
    }

def calculate_roi(correct_identifications, total_customers, 
                 retention_success_rate=0.5, 
                 customer_lifetime_value=100000,
                 retention_cost_per_customer=5000):
    """Calculate ROI of churn prediction system"""
    
    customers_retained = int(correct_identifications * retention_success_rate)
    revenue_saved = customers_retained * customer_lifetime_value
    retention_program_cost = correct_identifications * retention_cost_per_customer
    net_benefit = revenue_saved - retention_program_cost
    roi_ratio = revenue_saved / retention_program_cost if retention_program_cost > 0 else 0
    
    print(f"\nROI Analysis:")
    print(f"  Identified churners: {correct_identifications}")
    print(f"  Retention success rate: {retention_success_rate*100:.0f}%")
    print(f"  Customers retained: {customers_retained}")
    print(f"  Revenue saved: {revenue_saved:,} rubles")
    print(f"  Retention cost: {retention_program_cost:,} rubles")
    print(f"  Net benefit: {net_benefit:,} rubles")
    print(f"  ROI Ratio: {roi_ratio:.1f}:1")
    
    return {
        'customers_retained': customers_retained,
        'revenue_saved': revenue_saved,
        'retention_cost': retention_program_cost,
        'net_benefit': net_benefit,
        'roi_ratio': roi_ratio
    }
