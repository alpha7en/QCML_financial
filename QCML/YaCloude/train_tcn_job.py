import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# >>> иМПОРТиРУЕМ MATPLOTLIB С УКАЗАНиЕМ BACKEND <<<
import matplotlib
matplotlib.use('Agg') # используем неинтерактивный бэкенд Agg для сохранения в файл
import matplotlib.pyplot as plt
# <<<------------------------------------------------->>>
import os
import time
import copy
import pickle
import argparse
import json

# --- Определение МОДЕЛи (Классическая NN 32-16-8) ---
class FinancialNN(nn.Module):
    def __init__(self, input_dim):
        super(FinancialNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 8), nn.BatchNorm1d(8), nn.ReLU()
        )
        self.output_layer = nn.Linear(8, 1)

    def forward(self, x):
        if x.dim() != 2: raise ValueError(f"Expected 2D input (N, C), got {x.dim()}")
        if x.shape[0] <= 1 and self.training:
            print("Предупреждение: Пропуск батча размером 1 (BatchNorm1d).")
            return torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

# --- Основная функция ---
def main(config, data_csv_path, output_model_path, output_scaler_path, output_plot_path): # <<< Добавлен путь для графика
    """Основная функция обучения и оценки MLP с сохранением графика."""

    print("--- Конфигурация MLP ---")
    # ... (вывод конфига без изменений) ...
    for key, value in config.items(): print(f"{key}: {value}")
    print("-" * 30)

    # Параметры из конфигурации
    # ... (все параметры без изменений) ...
    target_col = config['target_col']; final_test_set_size = config['final_test_set_size']
    k_folds = config['k_folds']; val_split_for_final = config['val_split_for_final']
    batch_size = config['batch_size']; learning_rate = config['learning_rate']
    epochs = config['epochs']; weight_decay = config['weight_decay']
    early_stopping_patience = config['early_stopping_patience']
    scheduler_patience = config['scheduler_patience']; scheduler_factor = config['scheduler_factor']
    random_seed = config['random_seed']


    # Настройка устройства и seed
    # ... (без изменений) ...
    if torch.cuda.is_available(): device = torch.device("cuda"); print(f"GPU: {torch.cuda.get_device_name(0)}")
    else: device = torch.device("cpu"); print("CPU.")
    torch.manual_seed(random_seed); np.random.seed(random_seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(random_seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    # --- 1. Загрузка и обработка данных ---
    # ... (без изменений) ...
    try:
        df = pd.read_csv(data_csv_path, index_col='date', parse_dates=True)
        print(f"Данные загружены. Форма: {df.shape}")
    except FileNotFoundError: raise SystemExit(f"Ошибка: Файл не найден: {data_csv_path}")
    except Exception as e: raise SystemExit(f"Ошибка чтения CSV: {e}")
    if target_col not in df.columns: raise ValueError(f"Целевой столбец '{target_col}' не найден!")
    feature_cols = [col for col in df.columns if col != target_col]
    print(f"Целевая переменная: {target_col}, Признаков: {len(feature_cols)}")
    if not feature_cols: raise ValueError("Нет признаков.")
    print(f"Размер данных до NaN: {df.shape}")
    df_cleaned = df.dropna(subset=[target_col] + feature_cols).copy()
    print(f"Размер данных после NaN: {df_cleaned.shape}")
    if df_cleaned.empty: raise ValueError("Нет данных после NaN.")

    # --- 2. Подготовка X, y ---
    # ... (без изменений) ...
    X_cleaned = df_cleaned[feature_cols].values
    y_cleaned = df_cleaned[target_col].values.reshape(-1, 1)
    original_indices_cleaned = df_cleaned.index

    # --- 3. Разделение данных ---
    # ... (без изменений) ...
    print("-" * 30); print(f"Разделение данных...")
    if X_cleaned.shape[0] < 2: raise ValueError("Недостаточно данных для разделения.")
    X_temp, X_final_test, y_temp, y_final_test, temp_indices, final_test_indices = train_test_split(
        X_cleaned, y_cleaned, original_indices_cleaned, test_size=final_test_set_size, random_state=random_seed, shuffle=True
    )
    if X_temp.shape[0] == 0 or X_final_test.shape[0] == 0: raise ValueError("Ошибка разделения.")
    print(f"Размер ТЕМП X: {X_temp.shape}, ФиНАЛЬНЫЙ ТЕСТ X: {X_final_test.shape}"); print("-" * 30)

    input_dim = X_cleaned.shape[1]

    # --- 4. Кросс-валидация ---
    # ... (Код КФ остается БЕЗ иЗМЕНЕНиЙ, он не генерирует итоговый график) ...
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    fold_mses, fold_r2s = [], []; best_epochs_per_fold = []
    print(f"\n--- Начало {k_folds}-кратной кросс-валидации (MLP + Улучшения) ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp, y_temp)):
        # ... (весь код КФ) ...
        pass # Placeholder, т.к. код КФ не меняется
    print("\n--- Сводные результаты К-кратной кросс-валидации (MLP) ---")
    if fold_mses: print(f"Среднее MSE: {np.mean(fold_mses):.4f} (+/- {np.std(fold_mses):.4f})"); print(f"Среднее R²:  {np.mean(fold_r2s):.4f} (+/- {np.std(fold_r2s):.4f})")
    else: print("Кросс-валидация не дала результатов.")


    # --- 5. Обучение ФиНАЛЬНОЙ модели ---
    print("\n--- Обучение финальной модели на ТЕМП данных (MLP) ---")
    # ... (Разделение на train_final/val_final без изменений) ...
    if X_temp.shape[0] < 2: raise ValueError("Недостаточно данных для финального обучения.")
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(X_temp, y_temp, test_size=val_split_for_final, random_state=random_seed, shuffle=True)
    if X_train_final.shape[0] == 0 or X_val_final.shape[0] == 0: raise ValueError("Ошибка разделения для финального обучения.")

    # >>> Обучаем финальный Scaler и сохраняем <<<
    scaler_final = StandardScaler()
    X_train_final_scaled = scaler_final.fit_transform(X_train_final)
    X_val_final_scaled = scaler_final.transform(X_val_final)
    try:
        with open(output_scaler_path, 'wb') as f: pickle.dump(scaler_final, f)
        print(f"Scaler сохранен: {output_scaler_path}")
    except Exception as e: print(f"Ошибка сохранения scaler: {e}")

    # ... (Подготовка тензоров и DataLoader'ов без изменений) ...
    X_train_final_tensor = torch.tensor(X_train_final_scaled, dtype=torch.float32).to(device); y_train_final_tensor = torch.tensor(y_train_final, dtype=torch.float32).to(device)
    X_val_final_tensor = torch.tensor(X_val_final_scaled, dtype=torch.float32).to(device); y_val_final_tensor = torch.tensor(y_val_final, dtype=torch.float32).to(device)
    final_train_dataset = TensorDataset(X_train_final_tensor, y_train_final_tensor);
    if len(final_train_dataset) == 0: raise ValueError("Нет данных для финального обучения.")
    final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True)
    final_val_dataset = TensorDataset(X_val_final_tensor, y_val_final_tensor); final_val_loader = DataLoader(final_val_dataset, batch_size=batch_size, shuffle=False) if len(final_val_dataset) > 0 else []

    # ... (инициализация финальной модели, критерия, оптимизатора, шедулера без изменений) ...
    final_model = FinancialNN(input_dim=input_dim).to(device)
    final_criterion = nn.MSELoss(); final_optimizer = AdamW(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    final_scheduler = ReduceLROnPlateau(final_optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=False)


    # ... (Цикл обучения финальной модели с Early Stopping без изменений) ...
    best_val_loss_final = float('inf'); epochs_no_improve_final = 0; best_model_state_final = None; best_epoch_final = -1; final_train_losses_history = []
    print(f"  Начало финального обучения..."); start_time_final = time.time()
    for epoch in range(epochs):
         # ... (код цикла) ...
        pass # Placeholder
    end_time_final = time.time(); print(f"Финальное обучение завершено (лучший {best_epoch_final}) за {end_time_final - start_time_final:.2f} сек.")


    # --- 6. Сохранение ЛУЧШЕЙ ФиНАЛЬНОЙ модели ---
    # ... (код сохранения модели без изменений, использует output_model_path) ...
    if best_model_state_final:
        try:
            torch.save(best_model_state_final, output_model_path)
            print(f"Лучшая финальная модель (эпоха {best_epoch_final}) сохранена в: {output_model_path}")
            final_model.load_state_dict(best_model_state_final)
        except Exception as e: print(f"Ошибка сохранения/загрузки лучшей модели: {e}. используется последняя.")
    else:
        print("Предупреждение: Нет лучшего состояния, сохраняется последняя.")
        try: torch.save(final_model.state_dict(), output_model_path); print(f"Последняя модель сохранена: {output_model_path}")
        except Exception as e: print(f"Ошибка сохранения последней модели: {e}")


    # --- 7. Оценка ФиНАЛЬНОЙ модели на ОТДЕЛЬНОМ ТЕСТЕ ---
    print("\n--- Оценка ЛУЧШЕЙ ФиНАЛЬНОЙ модели на ОТДЕЛЬНОМ ФиНАЛЬНОМ ТЕСТЕ (MLP) ---")
    final_model.eval()
    if X_final_test.shape[0] > 0:
        # >>> Масштабируем тестовые данные <<<
        X_final_test_scaled = scaler_final.transform(X_final_test)
        X_final_test_tensor_eval = torch.tensor(X_final_test_scaled, dtype=torch.float32).to(device)
        with torch.no_grad(): y_pred_nn = final_model(X_final_test_tensor_eval).cpu().numpy()
        # >>> Расчет метрик <<<
        mse_final_nn = mean_squared_error(y_final_test, y_pred_nn); r2_final_nn = r2_score(y_final_test, y_pred_nn)
        print("\n--- Результаты оценки (MLP) ---")
        print(f"MSE MLP:               {mse_final_nn:.4f}")
        print(f"RMSE MLP:             {np.sqrt(mse_final_nn):.4f}")
        print(f"MAE MLP:              {mean_absolute_error(y_final_test, y_pred_nn):.4f}")
        print(f"R² MLP (vs Mean):     {r2_final_nn:.4f}")

        # --- 8. Визуализация и СОХРАНЕНиЕ ГРАФиКА ---
        print("\n--- Построение и сохранение графика результатов ---")
        try:
             plt.figure(figsize=(10, 5)) # Уменьшим размер для одного графика
             # График Предсказания vs Реальность (Scatter plot)
             plt.scatter(y_final_test, y_pred_nn, alpha=0.6, label='Предсказание vs Реальность')
             plt.plot([y_final_test.min(), y_final_test.max()], [y_final_test.min(), y_final_test.max()],
                      'r--', lw=2, label='идеальное совпадение')
             plt.xlabel("Реальные значения")
             plt.ylabel("Предсказанные значения")
             plt.title(f"Сравнение Предсказаний и Реальности (Финальный тест, MLP)\nЦель: {target_col}\nR²={r2_final_nn:.4f}")
             plt.legend()
             plt.grid(True)
             plt.tight_layout()
             # >>> Сохраняем график в файл <<<
             plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
             print(f"График сохранен в: {output_plot_path}")
             plt.close() # Закрываем фигуру, чтобы она не "висела" в памяти

             # Опционально: График потерь (если нужен)
             # plt.figure(figsize=(10, 5))
             # plt.plot(range(1, len(final_train_losses_history) + 1), final_train_losses_history, label='MSE Обучения')
             # plt.xlabel("Эпоха"), plt.ylabel("Средняя ошибка"); plt.title("Динамика ошибки обучения (MLP)")
             # plt.legend(), plt.grid(True)
             # output_loss_plot_path = os.path.join(os.path.dirname(output_plot_path), 'loss_plot_mlp.png')
             # plt.savefig(output_loss_plot_path)
             # print(f"График потерь сохранен в: {output_loss_plot_path}")
             # plt.close()

        except Exception as e:
             print(f"Ошибка при построении/сохранении графика: {e}")

    else:
        print("Финальный тестовый набор пуст. Оценка и визуализация невозможны.")


# --- Точка входа для скрипта ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MLP model using DataSphere Jobs.')
    parser.add_argument('--data_csv', type=str, required=True, help='Path to input CSV')
    parser.add_argument('--config_json', type=str, required=True, help='Path to JSON config')
    parser.add_argument('--output_model', type=str, required=True, help='Path to save model state_dict')
    parser.add_argument('--output_scaler', type=str, required=True, help='Path to save scaler')
    # >>> Добавляем аргумент для пути к графику <<<
    parser.add_argument('--output_plot', type=str, required=True, help='Path to save the results plot image')

    args = parser.parse_args()

    print("--- Параметры запуска ---")
    print(f"Данные CSV: {args.data_csv}"); print(f"Конфиг JSON: {args.config_json}")
    print(f"Выход модель: {args.output_model}"); print(f"Выход Scaler: {args.output_scaler}")
    print(f"Выход График: {args.output_plot}") # <<< Выводим путь к графику
    print("-" * 30)

    try:
        with open(args.config_json, 'r') as f: config_params = json.load(f)
        print(f"Конфигурация загружена из {args.config_json}")
    except Exception as e: raise SystemExit(f"Ошибка чтения конфигурации {args.config_json}: {e}")

    # >>> Передаем путь к графику в main <<<
    main(config_params, args.data_csv, args.output_model, args.output_scaler, args.output_plot)

    print("\n--- Задание DataSphere Job (MLP) завершено успешно ---")