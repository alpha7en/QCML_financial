import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import TensorDataset, DataLoader, Subset # Убедимся, что импорты есть
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import time
import copy
import pickle
import argparse # Для аргументов командной строки
import json     # Для чтения конфигурации

# --- Модели TCN (без изменений) ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super(Chomp1d, self).__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding); self.relu1 = nn.ReLU(); self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding); self.relu2 = nn.ReLU(); self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU(); self.init_weights()
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01); self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None: self.downsample.weight.data.normal_(0, 0.01)
    def forward(self, x):
        out = self.net(x); res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_dim)
    def forward(self, x):
        if x.dim() != 3: raise ValueError(f"Expected input with 3 dimensions (N, L, C), but got {x.dim()}")
        x = x.permute(0, 2, 1); y = self.network(x)
        o = self.linear(y[:, :, -1]); return o

# --- Функции (без изменений) ---
def create_sequences(input_data_2d, target_data_2d, sequence_length):
    sequences, targets = [], []
    if input_data_2d.ndim != 2 or target_data_2d.ndim != 2: raise ValueError("Input/target must be 2D")
    if input_data_2d.shape[0] != target_data_2d.shape[0]: raise ValueError("Input/target rows mismatch")
    num_samples = input_data_2d.shape[0]
    if num_samples <= sequence_length: return np.array([]), np.array([])
    for i in range(sequence_length, num_samples):
        sequences.append(input_data_2d[i-sequence_length:i, :])
        targets.append(target_data_2d[i, :])
    return (np.array(sequences), np.array(targets)) if sequences else (np.array([]), np.array([]))

def main(config, data_csv_path, output_model_path, output_scaler_path):
    """Основная функция обучения и оценки."""

    print("--- Конфигурация ---")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-" * 30)

    # Параметры из конфигурации
    target_col = config['target_col']; sequence_length = config['sequence_length']
    final_test_set_size = config['final_test_set_size']; k_folds = config['k_folds']
    val_split_for_final = config['val_split_for_final']; batch_size = config['batch_size']
    max_learning_rate = config['max_learning_rate']; epochs = config['epochs']
    tcn_num_channels = config['tcn_num_channels']; tcn_kernel_size = config['tcn_kernel_size']
    tcn_dropout = config['tcn_dropout']; weight_decay = config['weight_decay']
    max_grad_norm = config['max_grad_norm']; early_stopping_patience = config['early_stopping_patience']
    scheduler_patience = config['scheduler_patience']; scheduler_factor = config['scheduler_factor']
    random_seed = config['random_seed']

    # Настройка устройства и seed
    if torch.cuda.is_available(): device = torch.device("cuda"); print(f"GPU: {torch.cuda.get_device_name(0)}")
    else: device = torch.device("cpu"); print("CPU.")
    torch.manual_seed(random_seed); np.random.seed(random_seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(random_seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    # --- 1. Загрузка и обработка данных ---
    try:
        df = pd.read_csv(data_csv_path, index_col='date', parse_dates=True)
        print(f"Данные загружены из {data_csv_path}. Форма: {df.shape}")
    except FileNotFoundError: raise SystemExit(f"Ошибка: Файл не найден: {data_csv_path}")
    except Exception as e: raise SystemExit(f"Ошибка чтения CSV: {e}")
    if target_col not in df.columns: raise ValueError(f"Целевой столбец '{target_col}' не найден!")
    feature_cols = [col for col in df.columns if col != target_col]
    print(f"Целевая переменная: {target_col}, Признаков: {len(feature_cols)}")
    if not feature_cols: raise ValueError("Нет признаков для обучения.")
    print(f"Размер данных до удаления NaN: {df.shape}")
    df_cleaned = df.dropna(subset=[target_col] + feature_cols).copy()
    print(f"Размер данных после удаления NaN: {df_cleaned.shape}")
    if df_cleaned.empty: raise ValueError("Нет данных после удаления NaN.")

    # --- 2. Масштабирование признаков (2D) ---
    scaler = StandardScaler()
    X_scaled_2d = scaler.fit_transform(df_cleaned[feature_cols].values)
    y_unscaled_2d = df_cleaned[target_col].values.reshape(-1, 1)
    try:
        # >>> Сохраняем scaler по пути из аргументов <<<
        with open(output_scaler_path, 'wb') as f: pickle.dump(scaler, f)
        print(f"Scaler сохранен: {output_scaler_path}")
    except Exception as e: print(f"Ошибка сохранения scaler: {e}")

    # --- 3. Создание последовательностей ---
    X_seq, y_seq = create_sequences(X_scaled_2d, y_unscaled_2d, sequence_length)
    if X_seq.shape[0] == 0: raise ValueError("Не удалось создать последовательности.")
    original_indices_seq = df_cleaned.index[sequence_length:]
    print(f"Созданы последовательности: X_seq shape={X_seq.shape}, y_seq shape={y_seq.shape}")

    
    print("-" * 30); print(f"Разделение последовательностей")
    if X_seq.shape[0] < 2: raise ValueError("Недостаточно данных для разделения")
    X_temp, X_final_test, y_temp, y_final_test, temp_indices, final_test_indices = train_test_split(
        X_seq, y_seq, original_indices_seq, test_size=final_test_set_size, random_state=random_seed, shuffle=True
    )
    if X_temp.shape[0] == 0 or X_final_test.shape[0] == 0: raise ValueError("Ошибка разделения одна из выборок пуста")
    print(f"РазмерТЕМП X: {X_temp.shape} НАЛЬНЫЙ ТЕСТ  {X_final_test.shape}"); print("-" * 30)

    input_dim = X_seq.shape[2]

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    fold_mses, fold_r2s = [], []; best_epochs_per_fold = []
    print(f"\n-- Начало {k_folds}-кратной кросс-валидации --")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp, y_temp)):
        print(f"\n--- ФОЛД {fold+1}/{k_folds} ---")
        
        X_train_fold, X_val_fold = X_temp[train_idx], X_temp[val_idx]; y_train_fold, y_val_fold = y_temp[train_idx], y_temp[val_idx]
        X_train_fold_tensor = torch.tensor(X_train_fold, dtype=torch.float32).to(device); y_train_fold_tensor = torch.tensor(y_train_fold, dtype=torch.float32).to(device)
        X_val_fold_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device); y_val_fold_tensor = torch.tensor(y_val_fold, dtype=torch.float32).to(device)
        train_fold_dataset = TensorDataset(X_train_fold_tensor, y_train_fold_tensor)
        if len(train_fold_dataset) == 0: print(f"  Пропуск фолда {fold+1}: нет данных."); continue
        train_fold_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_fold_dataset = TensorDataset(X_val_fold_tensor, y_val_fold_tensor)
        val_fold_loader = DataLoader(val_fold_dataset, batch_size=batch_size, shuffle=False, drop_last=False) if len(val_fold_dataset) > 0 else []

        model = TCNModel(input_dim=input_dim, output_dim=1, num_channels=tcn_num_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout).to(device)
        criterion = nn.MSELoss(); optimizer = AdamW(model.parameters(), lr=max_learning_rate, weight_decay=weight_decay)
        steps_per_epoch = len(train_fold_loader);
        if steps_per_epoch == 0: print(f"  Пропуск фолда {fold+1}: нет батчей."); continue
        scheduler = OneCycleLR(optimizer, max_lr=max_learning_rate, epochs=epochs, steps_per_epoch=steps_per_epoch)

        best_val_loss = float('inf'); epochs_no_improve = 0; best_model_state = None; best_epoch = -1
        start_time_fold = time.time(); print(f"  Начало обучения фолда {fold+1}...")
        for epoch in range(epochs):
            # ... (цикл обучения и валидации фолда с early stopping и scheduler.step()) ...
            model.train(); train_loss_epoch = 0.0; batches_processed_train = 0
            for batch_X, batch_y in train_fold_loader:
                if batch_X.dim() != 3: continue
                optimizer.zero_grad(); outputs = model(batch_X); loss = criterion(outputs, batch_y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm); optimizer.step(); scheduler.step()
                train_loss_epoch += loss.item(); batches_processed_train += 1
            if batches_processed_train == 0: continue
            avg_train_loss = train_loss_epoch / batches_processed_train
            model.eval(); val_loss_epoch = 0.0; batches_processed_val = 0
            if len(val_fold_loader) > 0:
                with torch.no_grad():
                    for batch_X_val, batch_y_val in val_fold_loader:
                        if batch_X_val.dim() != 3: continue
                        outputs_val = model(batch_X_val); loss_val = criterion(outputs_val, batch_y_val)
                        val_loss_epoch += loss_val.item(); batches_processed_val += 1
                avg_val_loss = val_loss_epoch / batches_processed_val if batches_processed_val > 0 else float('inf')
            else: avg_val_loss = float('inf')
            if (epoch + 1) % 50 == 0: print(f'    Эпоха [{epoch+1}/{epochs}], T Loss: {avg_train_loss:.4f}, V Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.1e}')
            if avg_val_loss < best_val_loss: best_val_loss = avg_val_loss; best_model_state = copy.deepcopy(model.state_dict()); best_epoch = epoch + 1; epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience and len(val_fold_loader) > 0: print(f"    Ранняя остановка!"); break
        end_time_fold = time.time(); print(f"  Обучение фолда {fold+1} завершено (лучший {best_epoch}) за {end_time_fold - start_time_fold:.2f} сек.")
        best_epochs_per_fold.append(best_epoch)

        # ... (оценка лучшей модели фолда) ...
        if best_model_state: model.load_state_dict(best_model_state)
        else: print("  Предупреждение: спользуется последняя модель фолда.");
        model.eval()
        if X_val_fold_tensor.shape[0] > 0:
            with torch.no_grad(): y_val_pred = model(X_val_fold_tensor).cpu().numpy()
            mse_fold = mean_squared_error(y_val_fold, y_val_pred); r2_fold = r2_score(y_val_fold, y_val_pred)
            print(f"  Метрики валидации фолда {fold+1}: MSE={mse_fold:.4f}, R²={r2_fold:.4f}")
            fold_mses.append(mse_fold); fold_r2s.append(r2_fold)
        else: print(f"  Пропуск оценки фолда {fold+1}.")

    # ... (вывод сводных результатов КФ) ...
    print("\n--- Сводные результаты К-кратной кросс-валидации ---")
    if fold_mses:
        print(f"Среднее MSE: {np.mean(fold_mses):.4f} (+/- {np.std(fold_mses):.4f})"); print(f"Среднее R²:  {np.mean(fold_r2s):.4f} (+/- {np.std(fold_r2s):.4f})")
        if best_epochs_per_fold: print(f"Ср. 'лучших' эпох: {np.mean(best_epochs_per_fold):.1f} (+/- {np.std(best_epochs_per_fold):.1f})")
    else: print("Кросс-валидация не дала результатов.")

    # --- 6. Обучение ФиНАЛЬНОЙ модели ---
    # (КОД ОБУЧЕНиЯ ФиНАЛЬНОЙ МОДЕЛи С ВАЛиДАЦиЕЙ и EARLY STOPPING ОСТАЕТСЯ ПРЕЖНиМ)
    print("\n--- Обучение финальной модели на ТЕМП данных ---")
    if X_temp.shape[0] < 2: raise ValueError("Недостаточно данных для финального обучения.")
    X_train_final_seq, X_val_final_seq, y_train_final_seq, y_val_final_seq = train_test_split(X_temp, y_temp, test_size=val_split_for_final, random_state=random_seed, shuffle=True)
    if X_train_final_seq.shape[0] == 0 or X_val_final_seq.shape[0] == 0: raise ValueError("Ошибка разделения для финального обучения.")
    X_train_final_tensor = torch.tensor(X_train_final_seq, dtype=torch.float32).to(device); y_train_final_tensor = torch.tensor(y_train_final_seq, dtype=torch.float32).to(device)
    X_val_final_tensor = torch.tensor(X_val_final_seq, dtype=torch.float32).to(device); y_val_final_tensor = torch.tensor(y_val_final_seq, dtype=torch.float32).to(device)
    final_train_dataset = TensorDataset(X_train_final_tensor, y_train_final_tensor);
    if len(final_train_dataset) == 0: raise ValueError("Нет данных для финального обучения.")
    final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    final_val_dataset = TensorDataset(X_val_final_tensor, y_val_final_tensor); final_val_loader = DataLoader(final_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False) if len(final_val_dataset) > 0 else []
    final_model = TCNModel(input_dim=input_dim, output_dim=1, num_channels=tcn_num_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout).to(device)
    final_criterion = nn.MSELoss(); final_optimizer = AdamW(final_model.parameters(), lr=max_learning_rate, weight_decay=weight_decay)
    final_steps_per_epoch = len(final_train_loader);
    if final_steps_per_epoch == 0: raise ValueError("Нет батчей для финального обучения.")
    final_scheduler = OneCycleLR(final_optimizer, max_lr=max_learning_rate, epochs=epochs, steps_per_epoch=final_steps_per_epoch)
    best_val_loss_final = float('inf'); epochs_no_improve_final = 0; best_model_state_final = None; best_epoch_final = -1; final_train_losses_history = []
    print(f"  Начало финального обучения..."); start_time_final = time.time()
    for epoch in range(epochs):
        # ... (цикл обучения и валидации финальной модели) ...
        final_model.train(); train_loss_epoch = 0.0; batches_processed_train = 0
        for batch_X, batch_y in final_train_loader:
            if batch_X.dim() != 3: continue
            final_optimizer.zero_grad(); outputs = final_model(batch_X); loss = final_criterion(outputs, batch_y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_grad_norm); final_optimizer.step(); final_scheduler.step()
            train_loss_epoch += loss.item(); batches_processed_train += 1
        if batches_processed_train == 0: continue
        avg_train_loss = train_loss_epoch / batches_processed_train; final_train_losses_history.append(avg_train_loss)
        final_model.eval(); val_loss_epoch = 0.0; batches_processed_val = 0
        if len(final_val_loader) > 0:
            with torch.no_grad():
                for batch_X_val, batch_y_val in final_val_loader:
                    if batch_X_val.dim() != 3: continue
                    outputs_val = final_model(batch_X_val); loss_val = final_criterion(outputs_val, batch_y_val)
                    val_loss_epoch += loss_val.item(); batches_processed_val += 1
            avg_val_loss = val_loss_epoch / batches_processed_val if batches_processed_val > 0 else float('inf')
        else: avg_val_loss = float('inf')
        if (epoch + 1) % 50 == 0: print(f'    Эпоха [{epoch+1}/{epochs}], T Loss: {avg_train_loss:.4f}, V Loss: {avg_val_loss:.4f}, LR: {final_optimizer.param_groups[0]["lr"]:.1e}')
        if avg_val_loss < best_val_loss_final:
            best_val_loss_final = avg_val_loss; best_model_state_final = copy.deepcopy(final_model.state_dict()); best_epoch_final = epoch + 1; epochs_no_improve_final = 0
            # Сохранение лучшего чекпоинта (опционально, для Jobs можно не сохранять промежуточные)
            # best_checkpoint_path = os.path.join(checkpoint_dir, f'best_model_tcn_final.pth')
            # try: torch.save({'epoch': best_epoch_final, ...}, best_checkpoint_path)
            # except Exception as e: print(f"    Ошибка сохранения TCN чекпоинта: {e}")
        else:
            epochs_no_improve_final += 1
            if epochs_no_improve_final >= early_stopping_patience and len(final_val_loader) > 0: print(f"    Ранняя остановка!"); break
    end_time_final = time.time(); print(f"Финальное обучение завершено (лучший {best_epoch_final}) за {end_time_final - start_time_final:.2f} сек.")

    # --- 7. Сохранение ЛУЧШЕЙ ФиНАЛЬНОЙ модели ---
    if best_model_state_final:
        try:
            # >>> Сохраняем модель по пути из аргументов <<<
            torch.save(best_model_state_final, output_model_path)
            print(f"Лучшая финальная модель (эпоха {best_epoch_final}) сохранена в: {output_model_path}")
            # Загружаем лучшую для последующей оценки
            final_model.load_state_dict(best_model_state_final)
        except Exception as e: print(f"Ошибка сохранения лучшей финальной модели: {e}. используется последняя.")
    else:
        print("Предупреждение: Нет лучшего состояния, сохраняется последняя модель.")
        try:
             # >>> Сохраняем модель по пути из аргументов <<<
            torch.save(final_model.state_dict(), output_model_path)
            print(f"Последняя финальная модель сохранена в: {output_model_path}")
        except Exception as e: print(f"Ошибка сохранения последней финальной модели: {e}")

    # --- 8. Оценка ФиНАЛЬНОЙ модели на ОТДЕЛЬНОМ ТЕСТЕ ---
    print("\n--- Оценка ЛУЧШЕЙ ФиНАЛЬНОЙ модели на ОТДЕЛЬНОМ ФиНАЛЬНОМ ТЕСТЕ ---")
    final_model.eval()
    if X_final_test.shape[0] > 0:
        X_final_test_tensor_eval = torch.tensor(X_final_test, dtype=torch.float32).to(device)
        with torch.no_grad(): y_pred_nn = final_model(X_final_test_tensor_eval).cpu().numpy()
        mse_final_nn = mean_squared_error(y_final_test, y_pred_nn); r2_final_nn = r2_score(y_final_test, y_pred_nn)
        print("\n--- Результаты оценки (TCN) ---")
        print(f"MSE TCN:               {mse_final_nn:.4f}")
        print(f"RMSE TCN:             {np.sqrt(mse_final_nn):.4f}")
        print(f"MAE TCN:              {mean_absolute_error(y_final_test, y_pred_nn):.4f}")
        print(f"R² TCN (vs Mean):     {r2_final_nn:.4f}")

        # --- 9. Визуализация (Опционально, может не работать/сохраняться в Jobs) ---
        # В среде Jobs графики обычно не отображаются. Можно сохранять в файлы,
        # но нужно убедиться, что путь для сохранения доступен и указан в outputs job.yaml
        # Для простоты сейчас визуализация отключена для Jobs
        print("\n--- Визуализация отключена для запуска в Jobs ---")
        # try:
        #      plt.figure(figsize=(15, 6))
        #      # ... код графиков ...
        #      # output_plot_path = os.path.join(os.path.dirname(output_model_path), 'results_plot.png') # Пример пути
        #      # plt.savefig(output_plot_path)
        #      # plt.close()
        # except Exception as e: print(f"Ошибка визуализации: {e}")
    else:
        print("Финальный тестовый набор пуст. Оценка невозможна.")

# --- Точка входа для скрипта ---
if __name__ == "__main__":
    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(description='Train TCN model for financial time series using DataSphere Jobs.')
    # --- Входные аргументы (будут заменены DataSphere на реальные пути) ---
    parser.add_argument('--data_csv', type=str, required=True, help='Path to the input CSV data file (provided by DataSphere as ${DATA_CSV})')
    parser.add_argument('--config_json', type=str, required=True, help='Path to the JSON configuration file (provided by DataSphere as ${CONFIG_JSON})')
    # --- Выходные аргументы (будут заменены DataSphere на реальные пути) ---
    parser.add_argument('--output_model', type=str, required=True, help='Path to save the trained model state_dict (provided by DataSphere as ${OUTPUT_MODEL})')
    parser.add_argument('--output_scaler', type=str, required=True, help='Path to save the fitted scaler (provided by DataSphere as ${OUTPUT_SCALER})')

    # Парсим аргументы командной строки
    args = parser.parse_args()

    print("--- Параметры запуска ---")
    print(f"Входные данные CSV: {args.data_csv}")
    print(f"Конфигурация JSON: {args.config_json}")
    print(f"Выходная модель: {args.output_model}")
    print(f"Выходной Scaler: {args.output_scaler}")
    print("-" * 30)

    # Загрузка конфигурации из JSON
    try:
        with open(args.config_json, 'r') as f:
            config_params = json.load(f)
            print(f"Конфигурация успешно загружена из {args.config_json}")
    except FileNotFoundError: raise SystemExit(f"Ошибка: Файл конфигурации не найден: {args.config_json}")
    except json.JSONDecodeError: raise SystemExit(f"Ошибка: Некорректный формат JSON: {args.config_json}")
    except Exception as e: raise SystemExit(f"Ошибка при чтении конфигурации: {e}")

    # Запуск основного процесса с загруженной конфигурацией и путями
    main(config_params, args.data_csv, args.output_model, args.output_scaler)

    print("\n--- Задание DataSphere Job завершено успешно ---")