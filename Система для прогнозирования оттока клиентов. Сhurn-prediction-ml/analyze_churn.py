import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('src')
from data_processing import DataProcessor


class ChurnAnalyzer:
    """Анализ оттока клиентов для ВСЕХ клиентов из датасета"""
    
    def __init__(self):
        self.xgb_model = None
        self.rf_model = None
        self.lr_model = None
        self.scaler = None
        self.processor = DataProcessor()
    
    def load_models(self, models_path):
        """Загрузка моделей"""
        models_dict = joblib.load(models_path)
        self.xgb_model = models_dict.get('XGBoost')
        self.rf_model = models_dict.get('Random Forest')
        self.lr_model = models_dict.get('Логистическая регрессия')
        print("✅ Модели загружены успешно!")
    
    def load_scaler(self, scaler_path):
        """Загрузка нормализатора"""
        self.scaler = joblib.load(scaler_path)
        print("✅ Нормализатор загружен!")
    
    def preprocess_new_data(self, df):
        """Предобработка новых данных"""
        print("\n📥 Предобработка данных...")
        
        df_processed = df.copy()
        
        # Конвертируем Churn если есть
        if 'Churn' in df_processed.columns:
            if df_processed['Churn'].dtype == 'object':
                if df_processed['Churn'].isin(['Yes', 'No']).all():
                    df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})
        
        # Конвертируем TotalCharges
        if 'TotalCharges' in df_processed.columns:
            if df_processed['TotalCharges'].dtype == 'object':
                df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
                df_processed['TotalCharges'].fillna(0, inplace=True)
        
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Удаляем ID столбцы
        id_cols = [col for col in categorical_cols if 'id' in col.lower()]
        for col in id_cols:
            if col in categorical_cols:
                categorical_cols.remove(col)
            if col in df_processed.columns:
                df_processed.drop(col, axis=1, inplace=True)
        
        # Если есть Churn, удаляем его из признаков
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        
        # Конвертируем Yes/No в 1/0
        for col in categorical_cols:
            if df_processed[col].dtype == 'object':
                if df_processed[col].isin(['Yes', 'No']).all():
                    df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})
        
        remaining_categorical = [col for col in categorical_cols 
                                if df_processed[col].dtype == 'object']
        
        # One-Hot Encoding
        if remaining_categorical:
            df_encoded = pd.get_dummies(df_processed, 
                                       columns=remaining_categorical, 
                                       drop_first=True)
        else:
            df_encoded = df_processed.copy()
        
        # Отделяем Churn если есть
        if 'Churn' in df_encoded.columns:
            X = df_encoded.drop('Churn', axis=1)
            y = df_encoded['Churn']
        else:
            X = df_encoded
            y = None
        
        print(f"✅ Данные подготовлены! Размер: {X.shape}")
        
        return X, y
    
    def predict(self, X):
        """Предсказание для всех моделей"""
        print("\n🚀 Предсказание моделей...\n")
        
        X_scaled = self.scaler.transform(X)
        
        # XGBoost
        try:
            xgb_pred = self.xgb_model.predict(X_scaled)
            xgb_proba = self.xgb_model.predict_proba(X_scaled)[:, 1]
            print("✅ XGBoost предсказал результаты")
        except Exception as e:
            print(f"❌ Ошибка XGBoost: {e}")
            xgb_pred = np.zeros(len(X_scaled))
            xgb_proba = np.zeros(len(X_scaled))
        
        # Random Forest
        try:
            rf_pred = self.rf_model.predict(X_scaled)
            rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
            print("✅ Random Forest предсказал результаты")
        except Exception as e:
            print(f"❌ Ошибка Random Forest: {e}")
            rf_pred = np.zeros(len(X_scaled))
            rf_proba = np.zeros(len(X_scaled))
        
        # Logistic Regression
        try:
            lr_pred = self.lr_model.predict(X_scaled)
            lr_proba = self.lr_model.predict_proba(X_scaled)[:, 1]
            print("✅ Логистическая регрессия предсказала результаты")
        except Exception as e:
            print(f"❌ Ошибка Логистической регрессии: {e}")
            lr_pred = np.zeros(len(X_scaled))
            lr_proba = np.zeros(len(X_scaled))
        
        # Ансамбль
        ensemble_proba = (xgb_proba + rf_proba + lr_proba) / 3
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        results = {
            'XGBoost_pred': xgb_pred,
            'XGBoost_proba': xgb_proba,
            'RandomForest_pred': rf_pred,
            'RandomForest_proba': rf_proba,
            'LogisticRegression_pred': lr_pred,
            'LogisticRegression_proba': lr_proba,
            'Ensemble_pred': ensemble_pred,
            'Ensemble_proba': ensemble_proba
        }
        
        return results
    
    def save_predictions_csv(self, df, predictions, output_path):
        """Сохранение результатов в CSV"""
        print("\n💾 Сохранение CSV результатов...\n")
        
        result_df = df.copy()
        
        result_df['XGBoost_Прогноз'] = ['УХОДИТ' if p == 1 else 'ОСТАЁТСЯ' for p in predictions['XGBoost_pred']]
        result_df['XGBoost_Вероятность'] = (predictions['XGBoost_proba'] * 100).round(2)
        
        result_df['RandomForest_Прогноз'] = ['УХОДИТ' if p == 1 else 'ОСТАЁТСЯ' for p in predictions['RandomForest_pred']]
        result_df['RandomForest_Вероятность'] = (predictions['RandomForest_proba'] * 100).round(2)
        
        result_df['LogisticRegression_Прогноз'] = ['УХОДИТ' if p == 1 else 'ОСТАЁТСЯ' for p in predictions['LogisticRegression_pred']]
        result_df['LogisticRegression_Вероятность'] = (predictions['LogisticRegression_proba'] * 100).round(2)
        
        result_df['Ensemble_Прогноз'] = ['УХОДИТ' if p == 1 else 'ОСТАЁТСЯ' for p in predictions['Ensemble_pred']]
        result_df['Ensemble_Вероятность'] = (predictions['Ensemble_proba'] * 100).round(2)
        
        result_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ CSV сохранен: {output_path}")
        
        return result_df
    
    def save_text_report(self, df, predictions, output_path):
        """Сохранение красивого текстового отчёта"""
        print(f"📄 Создание текстового отчёта...")
        
        ensemble_churn = (predictions['Ensemble_pred'] == 1).sum()
        ensemble_stay = (predictions['Ensemble_pred'] == 0).sum()
        avg_risk = predictions['Ensemble_proba'].mean() * 100
        
        critical = (predictions['Ensemble_proba'] >= 0.90).sum()
        high = ((predictions['Ensemble_proba'] >= 0.70) & (predictions['Ensemble_proba'] < 0.90)).sum()
        medium = ((predictions['Ensemble_proba'] >= 0.50) & (predictions['Ensemble_proba'] < 0.70)).sum()
        low = (predictions['Ensemble_proba'] < 0.50).sum()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("ОТЧЁТ О ПРОГНОЗИРОВАНИИ ОТТОКА КЛИЕНТОВ\n")
            f.write("="*100 + "\n\n")
            
            # Общая статистика
            f.write("📊 ОБЩАЯ СТАТИСТИКА:\n")
            f.write("-"*100 + "\n")
            f.write(f"Всего клиентов анализировано: {len(df)}\n")
            f.write(f"Предсказано УХОДЯЩИХ: {ensemble_churn} ({ensemble_churn/len(df)*100:.1f}%)\n")
            f.write(f"Предсказано ВЕРНЫХ: {ensemble_stay} ({ensemble_stay/len(df)*100:.1f}%)\n")
            f.write(f"Средний риск ухода: {avg_risk:.2f}%\n\n")
            
            # Распределение по риску
            f.write("📋 РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ РИСКА:\n")
            f.write("-"*100 + "\n")
            f.write(f"🔴 КРИТИЧЕСКИЙ (>90%):    {critical:6d} ({critical/len(df)*100:5.1f}%) ← СРОЧНО ТРЕБУЮТ ВНИМАНИЯ!\n")
            f.write(f"🟠 ВЫСОКИЙ (70-90%):      {high:6d} ({high/len(df)*100:5.1f}%) ← ВЫСОКИЙ ПРИОРИТЕТ\n")
            f.write(f"🟡 СРЕДНИЙ (50-70%):      {medium:6d} ({medium/len(df)*100:5.1f}%) ← ВНИМАНИЕ\n")
            f.write(f"🟢 НИЗКИЙ (<50%):         {low:6d} ({low/len(df)*100:5.1f}%) ← СТАБИЛЬНЫ\n\n")
            
            # Топ-100 рисков
            f.write("\n" + "="*100 + "\n")
            f.write("🔴 ТОП-100 КЛИЕНТОВ С НАИВЫСШИМ РИСКОМ УХОДА:\n")
            f.write("="*100 + "\n\n")
            
            sorted_indices = np.argsort(predictions['Ensemble_proba'])[::-1][:min(100, len(df))]
            
            for i, idx in enumerate(sorted_indices, 1):
                customer_id = df.iloc[idx].get('customerID', f'ID_{idx}')
                risk = predictions['Ensemble_proba'][idx] * 100
                xgb_risk = predictions['XGBoost_proba'][idx] * 100
                rf_risk = predictions['RandomForest_proba'][idx] * 100
                lr_risk = predictions['LogisticRegression_proba'][idx] * 100
                
                # Дополнительная информация о клиенте (если есть)
                gender = df.iloc[idx].get('gender', 'N/A')
                tenure = df.iloc[idx].get('tenure', 'N/A')
                monthly = df.iloc[idx].get('MonthlyCharges', 'N/A')
                
                f.write(f"{i:3d}. Клиент: {customer_id:15s} | Риск ухода (Ансамбль): {risk:6.2f}%\n")
                f.write(f"     Пол: {str(gender):8s} | Стаж (мес): {str(tenure):6s} | Платёж/мес: {str(monthly):8s}\n")
                f.write(f"     XGBoost: {xgb_risk:6.2f}% | Random Forest: {rf_risk:6.2f}% | Логрег: {lr_risk:6.2f}%\n")
                
                # Определяем цвет риска
                if risk >= 90:
                    risk_level = "🔴 КРИТИЧЕСКИЙ - СРОЧНО!"
                elif risk >= 70:
                    risk_level = "🟠 ВЫСОКИЙ - ПРИОРИТЕТ"
                elif risk >= 50:
                    risk_level = "🟡 СРЕДНИЙ - ВНИМАНИЕ"
                else:
                    risk_level = "🟢 НИЗКИЙ - СТАБИЛЕН"
                
                f.write(f"     Статус: {risk_level}\n\n")
        
        print(f"✅ Текстовый отчёт сохранен: {output_path}")
    
    def print_console_report(self, df, predictions, total_clients):
        """Красивый отчёт в консоль"""
        ensemble_churn = (predictions['Ensemble_pred'] == 1).sum()
        ensemble_stay = (predictions['Ensemble_pred'] == 0).sum()
        
        critical = (predictions['Ensemble_proba'] >= 0.90).sum()
        high = ((predictions['Ensemble_proba'] >= 0.70) & (predictions['Ensemble_proba'] < 0.90)).sum()
        medium = ((predictions['Ensemble_proba'] >= 0.50) & (predictions['Ensemble_proba'] < 0.70)).sum()
        low = (predictions['Ensemble_proba'] < 0.50).sum()
        
        print("\n" + "="*100)
        print("📊 ИТОГОВАЯ СТАТИСТИКА АНАЛИЗА:")
        print("="*100)
        print(f"\n✅ Всего клиентов анализировано: {total_clients}")
        print(f"\n📈 ПРОГНОЗЫ АНСАМБЛЯ:")
        print(f"  🔴 УХОДЯЩИЕ:  {ensemble_churn:6d} ({ensemble_churn/total_clients*100:6.1f}%)")
        print(f"  🟢 ВЕРНЫЕ:    {ensemble_stay:6d} ({ensemble_stay/total_clients*100:6.1f}%)")
        
        print(f"\n📋 РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ РИСКА:")
        print(f"  🔴 КРИТИЧЕСКИЙ (>90%):   {critical:6d} ({critical/total_clients*100:6.1f}%) ← СРОЧНЫЕ МЕРЫ!")
        print(f"  🟠 ВЫСОКИЙ (70-90%):     {high:6d} ({high/total_clients*100:6.1f}%) ← ВЫСОКИЙ ПРИОРИТЕТ")
        print(f"  🟡 СРЕДНИЙ (50-70%):     {medium:6d} ({medium/total_clients*100:6.1f}%) ← ВНИМАНИЕ")
        print(f"  🟢 НИЗКИЙ (<50%):        {low:6d} ({low/total_clients*100:6.1f}%) ← СТАБИЛЬНЫ")
        
        print("\n" + "-"*100)
        print("ТОП-20 КЛИЕНТОВ С НАИВЫСШИМ РИСКОМ:")
        print("-"*100)
        
        sorted_indices = np.argsort(predictions['Ensemble_proba'])[::-1][:20]
        
        for i, idx in enumerate(sorted_indices, 1):
            customer_id = df.iloc[idx].get('customerID', f'ID_{idx}')
            risk = predictions['Ensemble_proba'][idx] * 100
            
            print(f"{i:2d}. {customer_id:15s} | Риск: {risk:6.2f}% | XGB: {predictions['XGBoost_proba'][idx]*100:6.2f}% | "
                  f"RF: {predictions['RandomForest_proba'][idx]*100:6.2f}% | LR: {predictions['LogisticRegression_proba'][idx]*100:6.2f}%")


def analyze_full_dataset():
    """Анализ ВСЕХ клиентов из датасета"""
    print("="*100)
    print("СИСТЕМА ПРОГНОЗИРОВАНИЯ ОТТОКА КЛИЕНТОВ - ПОЛНЫЙ АНАЛИЗ ДАТАСЕТА")
    print("="*100)
    
    analyzer = ChurnAnalyzer()
    
    # Загружаем модели
    print("\n📥 Загрузка обученных моделей...")
    analyzer.load_models('results/модели.pkl')
    analyzer.load_scaler('results/нормализатор.pkl')
    
    # Загружаем ВСЕ клиентов
    print(f"📥 Загрузка ВСЕХ клиентов из data/clients.csv...")
    df = pd.read_csv('data/clients.csv')
    total_clients = len(df)
    print(f"✅ Загружено {total_clients} клиентов")
    
    # Предобработка
    X, y = analyzer.preprocess_new_data(df)
    
    # Предсказание
    predictions = analyzer.predict(X)
    
    # Сохранения
    print("\n" + "="*100)
    print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ:")
    print("="*100)
    
    # CSV
    analyzer.save_predictions_csv(df, predictions, 'results/predictions.csv')
    
    # Текстовый отчёт
    analyzer.save_text_report(df, predictions, 'results/churn_report.txt')
    
    # Консоль
    analyzer.print_console_report(df, predictions, total_clients)
    
    print("\n" + "="*100)
    print("✅ АНАЛИЗ ЗАВЕРШЕН!")
    print("="*100)
    print(f"\n📁 CSV результаты сохранены в:     results/predictions.csv")
    print(f"📄 Текстовый отчёт сохранен в:    results/churn_report.txt")
    print(f"\n💡 Откройте эти файлы для детального анализа!\n")


if __name__ == '__main__':
    analyze_full_dataset()
