"""
🎯 СИСТЕМА ПРОГНОЗИРОВАНИЯ ОТТОКА КЛИЕНТОВ - ИСПРАВЛЕННАЯ ВЕРСИЯ
Made by Poroshin SA ©

Ошибка исправлена: Загрузка моделей при инициализации
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    os.system('mode con: cols=120 lines=40')
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        pass


class Colors:
    """Палитра цветов"""
    DARK_BG = '\033[38;2;17;24;39m'
    DARK_SURFACE = '\033[38;2;30;41;59m'
    PRIMARY = '\033[38;2;102;178;255m'
    PRIMARY_HOVER = '\033[38;2;137;196;255m'
    ACCENT = '\033[38;2;255;179;102m'
    SUCCESS = '\033[38;2;102;255;179m'
    WARNING = '\033[38;2;255;214;102m'
    DANGER = '\033[38;2;255;102;102m'
    CRITICAL = '\033[38;2;255;77;77m'
    TEXT_PRIMARY = '\033[38;2;237;241;245m'
    TEXT_SECONDARY = '\033[38;2;176;190;197m'
    TEXT_MUTED = '\033[38;2;120;135;150m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


class Animations:
    """Анимации"""
    
    @staticmethod
    def loading_spinner():
        chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        for i in range(20):
            sys.stdout.write(f"\r{Colors.PRIMARY}⟳ {chars[i % len(chars)]} Загрузка...{Colors.RESET}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()


class Box:
    """Боксы и рамки"""
    
    @staticmethod
    def header(text: str, width: int = 100):
        print(f"\n{Colors.PRIMARY}{'═' * width}{Colors.RESET}")
        print(f"{Colors.PRIMARY_HOVER}{Colors.BOLD}{text.center(width)}{Colors.RESET}")
        print(f"{Colors.PRIMARY}{'═' * width}{Colors.RESET}\n")
    
    @staticmethod
    def error_card(title: str, content: str):
        print(f"\n{Colors.DANGER}✗ {Colors.BOLD}{title}{Colors.RESET}")
        print(f"{Colors.DANGER}{content}{Colors.RESET}")
    
    @staticmethod
    def success_card(title: str, content: str = ""):
        print(f"\n{Colors.SUCCESS}✓ {Colors.BOLD}{title}{Colors.RESET}")
        if content:
            print(f"{Colors.SUCCESS}{content}{Colors.RESET}")
    
    @staticmethod
    def info_card(title: str, content: str):
        print(f"\n{Colors.PRIMARY}{Colors.BOLD}{title}{Colors.RESET}")
        print(f"{Colors.TEXT_SECONDARY}{content}{Colors.RESET}")


class ChurnPredictionSystem:
    """Система прогнозирования оттока"""
    
    def __init__(self):
        self.models = None
        self.scaler = None
        self.data = None
        self.predictions = None
        self.paths = {
            'models': 'results/модели.pkl',
            'scaler': 'results/нормализатор.pkl',
            'data': 'data/clients.csv',
            'output_csv': 'results/predictions.csv',
            'output_report': 'results/churn_report.txt'
        }
    
    def clear_screen(self):
        os.system('cls' if sys.platform == 'win32' else 'clear')
    
    def check_environment(self) -> bool:
        print(f"\n{Colors.TEXT_SECONDARY}Проверка окружения...{Colors.RESET}\n")
        missing = []
        
        for name, path in self.paths.items():
            if name in ['output_csv', 'output_report']:
                continue
            if os.path.exists(path):
                print(f"{Colors.SUCCESS}✓{Colors.RESET} {path}")
            else:
                print(f"{Colors.DANGER}✗{Colors.RESET} {path}")
                missing.append(name)
        
        if missing:
            Box.error_card("Отсутствуют файлы", "Поместите недостающие файлы в нужные папки")
            return False
        return True
    
    def load_models(self) -> bool:
        print(f"\n{Colors.TEXT_SECONDARY}Загрузка моделей и данных...{Colors.RESET}\n")
        
        try:
            Animations.loading_spinner()
            
            # Загрузить модели
            models_dict = joblib.load(self.paths['models'])
            self.models = {
                'xgb': models_dict.get('XGBoost'),
                'rf': models_dict.get('Random Forest'),
                'lr': models_dict.get('Логистическая регрессия')
            }
            
            # Загрузить скейлер
            self.scaler = joblib.load(self.paths['scaler'])
            
            # Загрузить данные
            self.data = pd.read_csv(self.paths['data'])
            
            Box.success_card("Модели загружены!", f"{len(self.data):,} клиентов")
            return True
            
        except Exception as e:
            Box.error_card("Ошибка загрузки", str(e))
            return False
    
    def preprocess_data(self):
        print(f"\n{Colors.TEXT_SECONDARY}Предобработка данных...{Colors.RESET}\n")
        
        try:
            df = self.data.copy()
            
            # Yes/No → 1/0
            if 'Churn' in df.columns and df['Churn'].dtype == 'object':
                df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
            
            if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == 'object':
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
            
            # Категориальные переменные
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Удалить ID
            id_cols = [c for c in categorical_cols if 'id' in c.lower()]
            for col in id_cols:
                if col in df.columns:
                    df = df.drop(col, axis=1)
                if col in categorical_cols:
                    categorical_cols.remove(col)
            
            # Удалить Churn из признаков
            if 'Churn' in categorical_cols:
                categorical_cols.remove('Churn')
            
            # Yes/No в признаках
            for col in categorical_cols:
                if df[col].dtype == 'object' and df[col].isin(['Yes', 'No']).all():
                    df[col] = df[col].map({'Yes': 1, 'No': 0})
            
            # One-Hot Encoding
            remaining = [c for c in categorical_cols if df[c].dtype == 'object']
            if remaining:
                df = pd.get_dummies(df, columns=remaining, drop_first=True)
            
            # Отделить целевую переменную
            if 'Churn' in df.columns:
                X = df.drop('Churn', axis=1)
            else:
                X = df
            
            print(f"{Colors.SUCCESS}✓{Colors.RESET} Готово: {X.shape[0]} клиентов, {X.shape[1]} признаков")
            return X
            
        except Exception as e:
            Box.error_card("Ошибка предобработки", str(e))
            return None
    
    def predict(self, X: pd.DataFrame) -> Dict:
        print(f"\n{Colors.TEXT_SECONDARY}Прогнозирование...{Colors.RESET}\n")
        
        try:
            X_scaled = self.scaler.transform(X)
            
            xgb_proba = self.models['xgb'].predict_proba(X_scaled)[:, 1]
            print(f"{Colors.SUCCESS}✓ XGBoost{Colors.RESET}")
            
            rf_proba = self.models['rf'].predict_proba(X_scaled)[:, 1]
            print(f"{Colors.SUCCESS}✓ Random Forest{Colors.RESET}")
            
            lr_proba = self.models['lr'].predict_proba(X_scaled)[:, 1]
            print(f"{Colors.SUCCESS}✓ Logistic Regression{Colors.RESET}")
            
            ensemble_proba = (xgb_proba + rf_proba + lr_proba) / 3
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)
            
            return {
                'xgb_proba': xgb_proba,
                'rf_proba': rf_proba,
                'lr_proba': lr_proba,
                'ensemble_pred': ensemble_pred,
                'ensemble_proba': ensemble_proba
            }
        except Exception as e:
            Box.error_card("Ошибка предсказания", str(e))
            return None
    
    def save_results(self, predictions: Dict):
        print(f"\n{Colors.TEXT_SECONDARY}Сохранение результатов...{Colors.RESET}\n")
        
        os.makedirs('results', exist_ok=True)
        
        result_df = self.data.copy()
        result_df['Риск_%'] = (predictions['ensemble_proba'] * 100).round(2)
        result_df['Прогноз'] = ['🔴 УХОДИТ' if p == 1 else '🟢 ОСТАЁТСЯ' for p in predictions['ensemble_pred']]
        result_df['Уровень'] = result_df['Риск_%'].apply(
            lambda x: "🔴 КРИТИЧЕСКИЙ" if x >= 90 else ("🟠 ВЫСОКИЙ" if x >= 70 else ("🟡 СРЕДНИЙ" if x >= 50 else "🟢 НИЗКИЙ"))
        )
        
        result_df.to_csv(self.paths['output_csv'], index=False, encoding='utf-8')
        print(f"{Colors.SUCCESS}✓{Colors.RESET} CSV: {self.paths['output_csv']}")
        
        # Текстовый отчёт
        total = len(result_df)
        critical = (predictions['ensemble_proba'] >= 0.90).sum()
        high = ((predictions['ensemble_proba'] >= 0.70) & (predictions['ensemble_proba'] < 0.90)).sum()
        
        with open(self.paths['output_report'], 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("ОТЧЁТ: ПРОГНОЗИРОВАНИЕ ОТТОКА КЛИЕНТОВ\n".center(100))
            f.write("="*100 + "\n\n")
            f.write(f"Всего клиентов: {total}\n")
            f.write(f"🔴 КРИТИЧЕСКИЙ (>90%): {critical} ({critical/total*100:.1f}%)\n")
            f.write(f"🟠 ВЫСОКИЙ (70-90%): {high} ({high/total*100:.1f}%)\n\n")
            
            sorted_idx = np.argsort(predictions['ensemble_proba'])[::-1][:50]
            f.write("ТОП-50 КЛИЕНТОВ:\n")
            for i, idx in enumerate(sorted_idx, 1):
                cid = result_df.iloc[idx].get('customerID', f'ID_{idx}')
                risk = predictions['ensemble_proba'][idx] * 100
                f.write(f"{i}. {cid} | Риск: {risk:.2f}%\n")
        
        print(f"{Colors.SUCCESS}✓{Colors.RESET} Отчет: {self.paths['output_report']}")
    
    def show_results(self, predictions: Dict):
        self.clear_screen()
        Box.header("📊 РЕЗУЛЬТАТЫ")
        
        total = len(self.data)
        churn = (predictions['ensemble_pred'] == 1).sum()
        
        critical = (predictions['ensemble_proba'] >= 0.90).sum()
        high = ((predictions['ensemble_proba'] >= 0.70) & (predictions['ensemble_proba'] < 0.90)).sum()
        medium = ((predictions['ensemble_proba'] >= 0.50) & (predictions['ensemble_proba'] < 0.70)).sum()
        low = (predictions['ensemble_proba'] < 0.50).sum()
        
        print(f"{Colors.PRIMARY}{Colors.BOLD}Статистика:{Colors.RESET}")
        print(f"  Всего: {total:,} | Будут уходить: {churn} ({churn/total*100:.1f}%)")
        
        print(f"\n{Colors.PRIMARY}{Colors.BOLD}По уровню риска:{Colors.RESET}")
        print(f"  {Colors.CRITICAL}🔴 КРИТИЧЕСКИЙ (>90%):  {critical:4d} ({critical/total*100:5.1f}%){Colors.RESET}")
        print(f"  {Colors.DANGER}🟠 ВЫСОКИЙ (70-90%):    {high:4d} ({high/total*100:5.1f}%){Colors.RESET}")
        print(f"  {Colors.WARNING}🟡 СРЕДНИЙ (50-70%):    {medium:4d} ({medium/total*100:5.1f}%){Colors.RESET}")
        print(f"  {Colors.SUCCESS}🟢 НИЗКИЙ (<50%):       {low:4d} ({low/total*100:5.1f}%){Colors.RESET}")
        
        print(f"\n{Colors.PRIMARY}{Colors.BOLD}ТОП-20:{Colors.RESET}\n")
        
        sorted_idx = np.argsort(predictions['ensemble_proba'])[::-1][:20]
        
        for i, idx in enumerate(sorted_idx, 1):
            cid = self.data.iloc[idx].get('customerID', f'ID_{idx}')
            risk = predictions['ensemble_proba'][idx] * 100
            
            if risk >= 90:
                color, level = Colors.CRITICAL, "🔴 КРИТИЧЕСКИЙ"
            elif risk >= 70:
                color, level = Colors.DANGER, "🟠 ВЫСОКИЙ"
            elif risk >= 50:
                color, level = Colors.WARNING, "🟡 СРЕДНИЙ"
            else:
                color, level = Colors.SUCCESS, "🟢 НИЗКИЙ"
            
            print(f"{i:2d}. {cid:15s} | {color}{level}{Colors.RESET} | Риск: {color}{risk:6.2f}%{Colors.RESET}")
        
        print(f"\n{Colors.TEXT_SECONDARY}[ENTER - вернуться]{Colors.RESET}")
        input()


class ChurnPredictionApp:
    """Главное приложение - ИСПРАВЛЕННОЕ"""
    
    def __init__(self):
        self.system = ChurnPredictionSystem()
        self.running = True
    
    def show_splash_screen(self):
        self.system.clear_screen()
        splash_text = """
        ╔═══════════════════════════════════════════════════════════════════════════════════════╗
        ║                                                                                       ║
        ║         🎯 СИСТЕМА ПРОГНОЗИРОВАНИЯ ОТТОКА КЛИЕНТОВ                                   ║
        ║                                                                                       ║
        ║              Предсказывайте и удерживайте важных клиентов                           ║
        ║                                                                                       ║
        ╚═══════════════════════════════════════════════════════════════════════════════════════╝
        """
        print(f"{Colors.PRIMARY}{splash_text}{Colors.RESET}")
        print(f"\n{Colors.TEXT_SECONDARY}Инициализация системы...{Colors.RESET}\n")
        Animations.loading_spinner()
        
        # ✅ ИСПРАВКА: Проверить окружение
        if not self.system.check_environment():
            print(f"\n{Colors.DANGER}[ENTER - выход]{Colors.RESET}")
            input()
            self.running = False
            return
        
        # ✅ ИСПРАВКА: Загрузить модели И ДАННЫЕ при запуске!
        if not self.system.load_models():
            print(f"\n{Colors.DANGER}[ENTER - выход]{Colors.RESET}")
            input()
            self.running = False
            return
        
        time.sleep(1)
        self.show_main_menu()
    
    def show_main_menu(self):
        while self.running:
            self.system.clear_screen()
            Box.header("🏠 ГЛАВНОЕ МЕНЮ")
            
            # ✅ ИСПРАВКА: self.system.data теперь не None!
            print(f"{Colors.TEXT_PRIMARY}Текущий датасет:{Colors.RESET}")
            print(f"  📁 data/clients.csv")
            print(f"  📊 Клиентов: {len(self.system.data):,}")
            
            print(f"\n{Colors.PRIMARY}{Colors.BOLD}Выберите действие:{Colors.RESET}\n")
            
            options = [
                ("1", "🚀 Начать прогнозирование оттока", self.run_prediction),
                ("2", "📊 Просмотр последних результатов", self.show_results),
                ("3", "⚙️  Настройки", self.show_settings),
                ("4", "ℹ️  О системе", self.show_about),
                ("5", "❌ Выход", self.exit_app),
            ]
            
            for key, text, _ in options:
                print(f"  {Colors.PRIMARY}{key}{Colors.RESET} - {text}")
            
            choice = input(f"\n{Colors.TEXT_SECONDARY}Выберите (1-5): {Colors.RESET}").strip()
            
            for key, _, callback in options:
                if choice == key:
                    callback()
                    break
    
    def run_prediction(self):
        self.system.clear_screen()
        Box.header("🚀 ЗАПУСК ПРОГНОЗИРОВАНИЯ")
        
        X = self.system.preprocess_data()
        if X is None:
            print(f"\n{Colors.DANGER}[ENTER]{Colors.RESET}")
            input()
            return
        
        predictions = self.system.predict(X)
        if predictions is None:
            print(f"\n{Colors.DANGER}[ENTER]{Colors.RESET}")
            input()
            return
        
        self.system.save_results(predictions)
        self.system.predictions = predictions
        
        Box.success_card("Прогнозирование завершено!")
        time.sleep(1)
        self.system.show_results(predictions)
    
    def show_results(self):
        if self.system.predictions is None:
            self.system.clear_screen()
            Box.error_card("Результаты не найдены", "Сначала запустите прогнозирование (опция 1)")
            print(f"\n{Colors.TEXT_SECONDARY}[ENTER]{Colors.RESET}")
            input()
            return
        
        self.system.show_results(self.system.predictions)
    
    def show_settings(self):
        self.system.clear_screen()
        Box.header("⚙️  НАСТРОЙКИ")
        
        Box.info_card("📁 Датасет", "data/clients.csv")
        Box.info_card("📦 Модели", "results/модели.pkl")
        Box.info_card("🔧 Скейлер", "results/нормализатор.pkl")
        
        print(f"\n{Colors.TEXT_SECONDARY}[ENTER]{Colors.RESET}")
        input()
    
    def show_about(self):
        self.system.clear_screen()
        Box.header("ℹ️  О СИСТЕМЕ")
        
        Box.info_card("Версия", "1.0 (Исправленная)")
        Box.info_card("Точность", "AUC-ROC: 83.4%")
        Box.info_card("Разработка", "Poroshin SA © 2026")
        Box.info_card("Статус", "✅ Production Ready")
        
        print(f"\n{Colors.TEXT_SECONDARY}[ENTER]{Colors.RESET}")
        input()
    
    def exit_app(self):
        self.system.clear_screen()
        print(f"\n{Colors.PRIMARY}{'═' * 100}{Colors.RESET}")
        print(f"{Colors.PRIMARY_HOVER}{Colors.BOLD}Спасибо за использование!{Colors.RESET}".center(100))
        print(f"{Colors.PRIMARY}{'═' * 100}{Colors.RESET}")
        print(f"\n{Colors.TEXT_SECONDARY}{Colors.DIM}Made by Poroshin SA © 2026{Colors.RESET}\n")
        self.running = False


def main():
    try:
        app = ChurnPredictionApp()
        app.show_splash_screen()
    except KeyboardInterrupt:
        print(f"\n{Colors.DANGER}Программа прервана.{Colors.RESET}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.DANGER}Ошибка: {e}{Colors.RESET}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
