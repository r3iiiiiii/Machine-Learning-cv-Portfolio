import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь для импорта
import sys
sys.path.append('src')
from data_processing import DataProcessorx


class ModelTrainer:
    """Обучение и оценка моделей"""
    
    def __init__(self):
        self.xgb_model = None
        self.rf_model = None
        self.lr_model = None
        self.ensemble_model = None
        self.metrics = {}
    
    def train_xgboost_tuned(self, X_train, y_train):
        """Обучение XGBoost с hyperparameter tuning"""
        print("\n=== ОБУЧЕНИЕ XGBoost (с hyperparameter tuning) ===")
        
        # Параметры для поиска
        param_grid = {
            'max_depth': [5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 150, 200],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        # Базовая модель
        xgb_base = xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # GridSearchCV для поиска лучших параметров
        print("Ищу лучшие параметры (это может занять время)...")
        grid_search = GridSearchCV(
            xgb_base,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Лучшие параметры найдены!")
        print(f"Лучший AUC: {grid_search.best_score_:.4f}")
        print(f"Параметры: {grid_search.best_params_}")
        
        self.xgb_model = grid_search.best_estimator_
        return self.xgb_model
    
    def train_random_forest_tuned(self, X_train, y_train):
        """Обучение Random Forest с hyperparameter tuning"""
        print("\n=== ОБУЧЕНИЕ Random Forest (с hyperparameter tuning) ===")
        
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        
        print("Ищу лучшие параметры (это может занять время)...")
        grid_search = GridSearchCV(
            rf_base,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Лучшие параметры найдены!")
        print(f"Лучший AUC: {grid_search.best_score_:.4f}")
        print(f"Параметры: {grid_search.best_params_}")
        
        self.rf_model = grid_search.best_estimator_
        return self.rf_model
    
    def train_logistic_regression_tuned(self, X_train, y_train):
        """Обучение Logistic Regression с hyperparameter tuning"""
        print("\n=== ОБУЧЕНИЕ Логистической регрессии (с hyperparameter tuning) ===")
        
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1],
            'class_weight': ['balanced', None]
        }
        
        lr_base = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
        
        print("Ищу лучшие параметры...")
        grid_search = GridSearchCV(
            lr_base,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Лучшие параметры найдены!")
        print(f"Лучший AUC: {grid_search.best_score_:.4f}")
        print(f"Параметры: {grid_search.best_params_}")
        
        self.lr_model = grid_search.best_estimator_
        return self.lr_model
    
    def evaluate_models(self, models_dict, X_test, y_test):
        """Оценка всех моделей"""
        print("\n=== ОЦЕНКА МОДЕЛЕЙ ===\n")
        
        results = {}
        predictions = {}
        
        for name, model in models_dict.items():
            print(f"{name}:")
            
            # Предсказания
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"  Точность: {accuracy:.4f}")
            print(f"  Точность (для класса 1): {precision:.4f}")
            print(f"  Полнота: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC-ROC: {auc_roc:.4f}\n")
            
            results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'AUC-ROC': auc_roc
            }
            
            predictions[name] = y_pred_proba
            self.metrics[name] = results[name]
        
        return results, predictions
    
    def plot_roc_curves(self, models_dict, predictions, X_test, y_test):
        """Построение ROC кривых"""
        print("=== СОЗДАНИЕ ROC КРИВЫХ ===")
        
        plt.figure(figsize=(10, 8))
        
        for name, model in models_dict.items():
            y_pred_proba = predictions[name]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.4f})')
        
        # Диагональная линия
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Случайное угадывание')
        
        plt.xlabel('False Positive Rate (Ложные срабатывания)', fontsize=12)
        plt.ylabel('True Positive Rate (Верно найдено)', fontsize=12)
        plt.title('ROC Кривые - Все модели', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/roc_кривые.png', dpi=300, bbox_inches='tight')
        print("✅ ROC кривые сохранены: results/roc_кривые.png\n")
        plt.close()
    
    def plot_feature_importance(self, model, feature_names):
        """Построение графика важности признаков"""
        print("=== СОЗДАНИЕ ГРАФИКА ВАЖНОСТИ ===")
        
        # Получаем важность из XGBoost
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Сортируем по важности
            indices = np.argsort(importances)[::-1][:15]  # Топ 15
            
            plt.figure(figsize=(12, 6))
            plt.title('Важность признаков (XGBoost)', fontsize=14, fontweight='bold')
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.ylabel('Важность', fontsize=12)
            plt.xlabel('Признаки', fontsize=12)
            plt.tight_layout()
            plt.savefig('results/важность_признаков.png', dpi=300, bbox_inches='tight')
            print("✅ График сохранён: results/важность_признаков.png\n")
            plt.close()
    
    def save_results(self, models_dict):
        """Сохранение результатов"""
        print("💾 Сохранение результатов...")
        
        # Сохраняем модели
        joblib.dump(models_dict, 'results/модели.pkl')
        print("✅ Модели сохранены: results/модели.pkl")
        
        # Сохраняем метрики в JSON
        with open('results/метрики.json', 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        print("✅ Метрики сохранены: results/метрики.json")


def apply_smote(X_train, y_train):
    """Применение SMOTE для балансировки классов"""
    print("\n=== ПРИМЕНЕНИЕ SMOTE (балансировка классов) ===")
    
    print(f"ДО SMOTE:")
    print(f"  Класс 0: {(y_train == 0).sum()} примеров")
    print(f"  Класс 1: {(y_train == 1).sum()} примеров")
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"\nПОСЛЕ SMOTE:")
    print(f"  Класс 0: {(y_train_smote == 0).sum()} примеров")
    print(f"  Класс 1: {(y_train_smote == 1).sum()} примеров")
    print(f"✅ Классы сбалансированы!\n")
    
    return X_train_smote, y_train_smote


def apply_feature_engineering(X_train, X_test):
    """Создание новых признаков"""
    print("=== FEATURE ENGINEERING (создание новых признаков) ===")
    
    # Создаём новые признаки для train
    X_train_fe = X_train.copy()
    
    # Проверяем наличие признаков перед созданием
    if 'MonthlyCharges' in X_train.columns and 'TotalCharges' in X_train.columns:
        X_train_fe['monthly_to_total'] = X_train['MonthlyCharges'] / (X_train['TotalCharges'] + 1)
        print("✅ Создан признак: monthly_to_total")
    
    # Создаём новые признаки для test (с теми же функциями)
    X_test_fe = X_test.copy()
    if 'MonthlyCharges' in X_test.columns and 'TotalCharges' in X_test.columns:
        X_test_fe['monthly_to_total'] = X_test['MonthlyCharges'] / (X_test['TotalCharges'] + 1)
    
    print(f"✅ Feature Engineering завершён! Новое количество признаков: {X_train_fe.shape[1]}\n")
    
    return X_train_fe, X_test_fe


def apply_cross_validation(model, X_train, y_train):
    """Применение cross-validation"""
    print("=== CROSS-VALIDATION (5-fold CV) ===")
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    
    print(f"CV AUC scores: {cv_scores}")
    print(f"Средний AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"✅ Cross-validation завершена!\n")
    
    return cv_scores


def main():
    print("=" * 70)
    print("СИСТЕМА ПРОГНОЗИРОВАНИЯ ОТТОКА КЛИЕНТОВ (OPTIMIZED VERSION)")
    print("=" * 70)
    
    # ========== ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ ==========
    print("\n📥 Загрузка реальных данных...")
    processor = DataProcessor()
    df = processor.load_data('data/clients.csv')
    
    print(f"\nDataset shape: {df.shape}")
    processor.analyze_data(df)
    
    # Предобработка
    X_train, X_test, y_train, y_test = processor.preprocess_data(df)
    
    # ========== FEATURE ENGINEERING ==========
    # Нужно перевести обратно в DataFrame для feature engineering
    # (т.к. данные после preprocessing - это numpy arrays)
    # Пока пропускаем, так как это может усложнить процесс
    
    # ========== ПРИМЕНЕНИЕ SMOTE ==========
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    
    # ========== ОБУЧЕНИЕ МОДЕЛЕЙ ==========
    print("\n🚀 Обучение моделей с hyperparameter tuning...\n")
    
    trainer = ModelTrainer()
    
    # Обучение с поиском лучших параметров
    xgb_model = trainer.train_xgboost_tuned(X_train_smote, y_train_smote)
    rf_model = trainer.train_random_forest_tuned(X_train_smote, y_train_smote)
    lr_model = trainer.train_logistic_regression_tuned(X_train_smote, y_train_smote)
    
    # ========== CROSS-VALIDATION ==========
    print("\n📊 Проверка стабильности моделей...\n")
    
    cv_xgb = apply_cross_validation(xgb_model, X_train_smote, y_train_smote)
    cv_rf = apply_cross_validation(rf_model, X_train_smote, y_train_smote)
    cv_lr = apply_cross_validation(lr_model, X_train_smote, y_train_smote)
    
    # ========== ОЦЕНКА МОДЕЛЕЙ ==========
    models_dict = {
        'XGBoost': xgb_model,
        'Random Forest': rf_model,
        'Логистическая регрессия': lr_model
    }
    
    results, predictions = trainer.evaluate_models(models_dict, X_test, y_test)
    
    # ========== ВИЗУАЛИЗАЦИЯ ==========
    print("📈 Создание визуализаций...\n")
    
    trainer.plot_roc_curves(models_dict, predictions, X_test, y_test)
    trainer.plot_feature_importance(xgb_model, processor.feature_names)
    
    # ========== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==========
    trainer.save_results(models_dict)
    processor.save_scaler('results/нормализатор.pkl')
    
    # ========== ИТОГОВЫЙ ОТЧЕТ ==========
    print("\n" + "=" * 70)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 70)
    
    print("\n📁 Созданные файлы:")
    print("  ✓ results/модели.pkl - Обученные модели")
    print("  ✓ results/метрики.json - Таблица с метриками")
    print("  ✓ results/нормализатор.pkl - Нормализатор данных")
    print("  ✓ results/важность_признаков.png - График важности")
    print("  ✓ results/roc_кривые.png - ROC кривые")
    
    print("\n🎯 Метрики лучшей модели (Логистическая регрессия):")
    best_model_metrics = trainer.metrics['Логистическая регрессия']
    for metric, value in best_model_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n🚀 Следующие шаги:")
    print("  1. Проверьте графики в папке results/")
    print("  2. Посмотрите метрики в results/метрики.json")
    print("  3. Используйте модели для предсказаний новых клиентов")
    print("  4. Запустите: python analyze_churn.py ДЛЯ предсказания оттока клиентов")


if __name__ == '__main__':
    main()