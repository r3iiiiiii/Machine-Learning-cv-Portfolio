import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json


class DataProcessor:
    """Обработчик данных для прогнозирования оттока"""
    
    def __init__(self):
        self.scaler = None
        self.feature_names = []
    
    def load_data(self, filepath):
        """Загрузка данных из CSV файла"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        return df
    
    def analyze_data(self, df):
        """Анализ загруженных данных"""
        print("\n=== DATA ANALYSIS ===\n")
        print("Dataset Info:")
        print(df.info())
        print("\nBasic Statistics:")
        print(df.describe())
        print("\nMissing Values:")
        print(df.isnull().sum())
        print("\nTarget Distribution:")
        print(df['Churn'].value_counts())
    
    def preprocess_data(self, df):
        """Подготовка данных для обучения"""
        
        print("\n=== DATA PREPROCESSING ===\n")
        
        #  КОНВЕРТАЦИЯ CHURN (ОЧЕНЬ ВАЖНО!)
        if 'Churn' in df.columns:
            # Проверяем формат Churn
            if df['Churn'].dtype == 'object':
                # Если текст ('Yes'/'No' или 'Да'/'Нет')
                if df['Churn'].isin(['Yes', 'No']).all():
                    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
                elif df['Churn'].isin(['Да', 'Нет']).all():
                    df['Churn'] = df['Churn'].map({'Да': 1, 'Нет': 0})
                else:
                    # Если другой формат, пробуем конвертировать
                    df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')
            
            # Заполняем пропущенные значения (если есть)
            if df['Churn'].isnull().any():
                df['Churn'].fillna(0, inplace=True)
        
        #  КОНВЕРТАЦИЯ TotalCharges (если строка)
        if 'TotalCharges' in df.columns:
            if df['TotalCharges'].dtype == 'object':
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                df['TotalCharges'].fillna(0, inplace=True)
        
        # Копируем dataframe для обработки
        df_processed = df.copy()
        
        # Определяем категориальные и числовые столбцы
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Удаляем ID столбцы (не нужны для обучения)
        id_cols = [col for col in categorical_cols if 'id' in col.lower() or 'ID' in col]
        for col in id_cols:
            if col in categorical_cols:
                categorical_cols.remove(col)
            df_processed.drop(col, axis=1, inplace=True)
        
        # Удаляем Churn из признаков (это целевая переменная)
        if 'Churn' in numerical_cols:
            numerical_cols.remove('Churn')
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        
        print(f"Categorical columns: {categorical_cols}")
        print(f"Numerical columns: {numerical_cols}")
        
        #  КОНВЕРТАЦИЯ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
        # Преобразуем 'Yes'/'No' в 1/0 для всех столбцов
        for col in categorical_cols:
            if df_processed[col].dtype == 'object':
                if df_processed[col].isin(['Yes', 'No']).all():
                    df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})
                elif df_processed[col].isin(['Да', 'Нет']).all():
                    df_processed[col] = df_processed[col].map({'Да': 1, 'Нет': 0})
        
        # One-Hot Encoding для оставшихся категориальных признаков
        remaining_categorical = [col for col in categorical_cols 
                                if df_processed[col].dtype == 'object']
        
        if remaining_categorical:
            df_encoded = pd.get_dummies(df_processed, 
                                       columns=remaining_categorical, 
                                       drop_first=True)
        else:
            df_encoded = df_processed.copy()
        
        # Разделяем на X и y
        X = df_encoded.drop('Churn', axis=1)
        y = df_encoded['Churn']
        
        # Сохраняем названия признаков
        self.feature_names = X.columns.tolist()
        
        print(f"\nDataset shape: {X.shape}")
        
        # Разбиваем на train и test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Class distribution (train): {dict(y_train.value_counts())}")
        print(f"Class distribution (test): {dict(y_test.value_counts())}")
        
        # Нормализуем данные (StandardScaler)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def save_scaler(self, filepath):
        """Сохранение нормализатора"""
        import joblib
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")


def get_sample_data(n_samples=1000):
    """Генерация синтетических данных для тестирования"""
    np.random.seed(42)
    
    data = {
        'numerical_50': np.random.randint(0, 100, n_samples),
        'Month': np.random.randint(1, 13, n_samples),
        'Month_12': np.random.randint(0, 2, n_samples),
        'CompanySize': np.random.randint(0, 3, n_samples),
        'Month_6': np.random.randint(0, 2, n_samples),
        'Churn': np.random.randint(0, 2, n_samples)
    }
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    # Тестирование на синтетических данных
    print("Testing with synthetic data...\n")
    df = get_sample_data()
    
    processor = DataProcessor()
    processor.analyze_data(df)
    
    X_train, X_test, y_train, y_test = processor.preprocess_data(df)
    print(f"\n✅ Preprocessing completed!")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")