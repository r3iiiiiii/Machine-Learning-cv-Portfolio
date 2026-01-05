import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ChurnPredictor:
    def __init__(self):
        self.xgb_model = None
        self.rf_model = None
        self.lr_model = None
        self.metrics = {}
    
    def train_xgboost(self, X_train, y_train):
        print("\n=== ОБУЧЕНИЕ XGBoost ===")
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Вес позитивного класса: {scale_pos_weight:.2f}")
        
        self.xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=0
        )
        
        self.xgb_model.fit(X_train, y_train)
        print("✅ XGBoost обучена!")
        return self.xgb_model
    
    def train_random_forest(self, X_train, y_train):
        print("\n=== ОБУЧЕНИЕ Random Forest ===")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train, y_train)
        print("✅ Random Forest обучена!")
        return self.rf_model
    
    def train_logistic_regression(self, X_train, y_train):
        print("\n=== ОБУЧЕНИЕ Логистической регрессии ===")
        
        self.lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        
        self.lr_model.fit(X_train, y_train)
        print("✅ Логистическая регрессия обучена!")
        return self.lr_model
    
    def predict_ensemble(self, X):
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
        rf_pred = self.rf_model.predict_proba(X)[:, 1]
        lr_pred = self.lr_model.predict_proba(X)[:, 1]
        
        ensemble_pred = 0.5 * xgb_pred + 0.3 * rf_pred + 0.2 * lr_pred
        return ensemble_pred
    
    def evaluate_models(self, X_test, y_test):
        print("\n=== ОЦЕНКА МОДЕЛЕЙ ===")
        
        models = {
            'XGBoost': self.xgb_model,
            'Random Forest': self.rf_model,
            'Логистическая регрессия': self.lr_model
        }
        
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'точность': accuracy,
                'точность_для_уходящих': precision,
                'полнота': recall,
                'f1': f1,
                'auc': auc
            }
            
            print(f"\n{name}:")
            print(f"  Точность: {accuracy:.4f}")
            print(f"  Точность (для класса 1): {precision:.4f}")
            print(f"  Полнота: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC-ROC: {auc:.4f}")
        
        y_pred_ensemble = (self.predict_ensemble(X_test) > 0.5).astype(int)
        y_pred_proba_ensemble = self.predict_ensemble(X_test)
        
        accuracy = accuracy_score(y_test, y_pred_ensemble)
        precision = precision_score(y_test, y_pred_ensemble, zero_division=0)
        recall = recall_score(y_test, y_pred_ensemble, zero_division=0)
        f1 = f1_score(y_test, y_pred_ensemble, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba_ensemble)
        
        results['Ансамбль'] = {
            'точность': accuracy,
            'точность_для_уходящих': precision,
            'полнота': recall,
            'f1': f1,
            'auc': auc
        }
        
        print(f"\nАнсамбль (50% XGB + 30% RF + 20% LR):")
        print(f"  Точность: {accuracy:.4f}")
        print(f"  Точность (для класса 1): {precision:.4f}")
        print(f"  Полнота: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")
        
        self.metrics = results
        return results
    
    def plot_feature_importance(self, feature_names, output_path='results/важность_признаков.png'):
        print("\n=== СОЗДАНИЕ ГРАФИКА ВАЖНОСТИ ===")
        
        importance = self.xgb_model.feature_importances_
        indices = np.argsort(importance)[::-1][:10]
        
        plt.figure(figsize=(12, 6))
        plt.title('Важность признаков (XGBoost)', fontsize=16, fontweight='bold')
        bars = plt.bar(range(len(indices)), importance[indices], color='#1f77b4')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Важность', fontsize=12)
        plt.xlabel('Признаки', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ График сохранён: {output_path}")
    
    def plot_roc_curves(self, X_test, y_test, output_path='results/roc_кривые.png'):
        print("\n=== СОЗДАНИЕ ROC КРИВЫХ ===")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = {
            'XGBoost': self.xgb_model,
            'Random Forest': self.rf_model,
            'Логистическая регрессия': self.lr_model
        }
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for (name, model), color in zip(models.items(), colors):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{name} (AUC={auc:.4f})')
        
        y_pred_ensemble = self.predict_ensemble(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred_ensemble)
        auc = roc_auc_score(y_test, y_pred_ensemble)
        ax.plot(fpr, tpr, 'r-', linewidth=3, label=f'Ансамбль (AUC={auc:.4f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Случайное угадывание')
        ax.set_xlabel('False Positive Rate (Ложные срабатывания)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Верно найдено)', fontsize=12)
        ax.set_title('ROC Кривые - Все модели', fontsize=16, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ ROC кривые сохранены: {output_path}")
    
    def save_models(self, model_path='results/модели.pkl', metrics_path='results/метрики.json'):
        joblib.dump({
            'xgb': self.xgb_model,
            'rf': self.rf_model,
            'lr': self.lr_model
        }, model_path)
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Модели сохранены: {model_path}")
        print(f"✅ Метрики сохранены: {metrics_path}")
    
    def load_models(self, model_path='results/модели.pkl'):
        models_dict = joblib.load(model_path)
        self.xgb_model = models_dict['xgb']
        self.rf_model = models_dict['rf']
        self.lr_model = models_dict['lr']
        print(f"✅ Модели загружены: {model_path}")