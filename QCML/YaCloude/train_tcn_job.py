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


# --- Определения МОДЕЛЕЙ ---

# Оставляем FinancialNN на случай, если захочется сравнить, но использовать не будем
# --- Новая, УСЛОЖНЕННАЯ модель: CNN-GRU V2 ---
class CNNGRUModelV2(nn.Module):
    def __init__(self, input_dim, sequence_length,
                 cnn_out_channels_1, cnn_kernel_size_1, # Параметры 1го CNN слоя
                 cnn_out_channels_2, cnn_kernel_size_2, # Параметры 2го CNN слоя
                 gru_hidden_dim, gru_num_layers, gru_bidirectional, # Параметры GRU
                 mlp_hidden_1, mlp_hidden_2, # Параметры MLP (2 скрытых слоя)
                 output_dim=1, dropout_rate=0.0):
        super(CNNGRUModelV2, self).__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.gru_bidirectional = gru_bidirectional

        # --- 1-й Блок CNN ---
        self.conv1 = nn.Conv1d(in_channels=self.input_dim,
                               out_channels=cnn_out_channels_1,
                               kernel_size=cnn_kernel_size_1,
                               padding='same')
        self.bn1 = nn.BatchNorm1d(cnn_out_channels_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        # --- 2-й Блок CNN ---
        self.conv2 = nn.Conv1d(in_channels=cnn_out_channels_1, # Вход = выход предыдущего
                               out_channels=cnn_out_channels_2,
                               kernel_size=cnn_kernel_size_2,
                               padding='same')
        self.bn2 = nn.BatchNorm1d(cnn_out_channels_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        # --- Skip Connection (опционально, но полезно) ---
        # Если кол-во каналов меняется между слоями, нужна проекция (1x1 свертка)
        # --- Skip Connection (опционально, но полезно) ---
        # Если кол-во каналов меняется между слоями, нужна проекция (1x1 свертка)
        if cnn_out_channels_1 != cnn_out_channels_2:
            # НЕПРАВиЛЬНО БЫЛО: in_channels=cnn_out_channels_1
            # ПРАВиЛЬНО: in_channels=self.input_dim (т.к. проецируем 'identity')
            self.skip_projection = nn.Conv1d(in_channels=self.input_dim,
                                             out_channels=cnn_out_channels_2,
                                             kernel_size=1, padding='same')
        else:
            self.skip_projection = None  # Каналы совпадают, проекция не нужна

        # --- GRU Слой ---
        # Входной размер GRU = кол-во каналов после второго CNN блока
        self.gru = nn.GRU(input_size=cnn_out_channels_2,
                          hidden_size=gru_hidden_dim,
                          num_layers=gru_num_layers,
                          batch_first=True,
                          dropout=dropout_rate if gru_num_layers > 1 else 0.0,
                          bidirectional=gru_bidirectional) # <--- Добавлено bidirectional

        # --- MLP Голова (с двумя скрытыми слоями) ---
        # Входной размер MLP зависит от GRU: hidden_dim * 2 если bidirectional, иначе hidden_dim
        gru_output_features = gru_hidden_dim * 2 if gru_bidirectional else gru_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(gru_output_features, mlp_hidden_1), # <--- Вход зависит от GRU
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_1, mlp_hidden_2), # <--- Второй скрытый слой MLP
            nn.ReLU(),
            nn.Dropout(dropout_rate), # <--- Дополнительный Dropout
            nn.Linear(mlp_hidden_2, output_dim)
        )

    def forward(self, x):
        # Ожидаемый вход x shape: (batch_size, sequence_length, input_dim)
        if x.dim() != 3 or x.shape[1] != self.sequence_length or x.shape[2] != self.input_dim:
             raise ValueError(f"CNNGRUModelV2 ожидает вход (N={x.shape[0]}, L={self.sequence_length}, C_in={self.input_dim}), получено {x.shape}")

        # --- CNN Часть ---
        # (N, C_in, L)
        x_cnn = x.permute(0, 2, 1)

        # Блок 1
        identity = x_cnn # Сохраняем для skip connection (если нужна проекция)
        x_cnn1 = self.conv1(x_cnn)
        x_cnn1 = self.bn1(x_cnn1)
        x_cnn1 = self.relu1(x_cnn1)
        x_cnn1 = self.dropout1(x_cnn1) # Выход Блока 1: (N, C1, L)

        # Блок 2
        x_cnn2 = self.conv2(x_cnn1)
        x_cnn2 = self.bn2(x_cnn2)
        x_cnn2 = self.relu2(x_cnn2)
        x_cnn2 = self.dropout2(x_cnn2) # Выход Блока 2: (N, C2, L)

        # Применение Skip Connection
        if self.skip_projection is not None:
             identity_projected = self.skip_projection(identity) # Проецируем вход conv1
        else:
             # Если проекция не нужна (C1==C2), то identity для сложения должен быть x_cnn1
             # Но сложение выхода conv2 с выходом conv1 менее стандартно, чем сложение с входом conv1
             # Давайте сложим выход Блока 2 с ПРОЕЦиРОВАННЫМ ВХОДОМ Блока 1 (классический ResNet стиль)
             # Если каналы совпали случайно, а архитектурно должны бы отличаться, то skip не добавим,
             # чтобы не усложнять чрезмерно. Если хотите skip всегда, убедитесь, что skip_projection есть всегда.
             # В данном варианте: если C1 != C2, складываем x_cnn2 и identity_projected.
             # Если C1 == C2, просто используем x_cnn2 (без skip).
             # Для простоты, если C1==C2, то skip_projection = None, и skip не используется.
             # Если хотите skip и при C1==C2, то измените логику в __init__
             pass # identity_projected не определен, skip не будет

        if self.skip_projection is not None:
            x_combined = x_cnn2 + identity_projected
            x_combined = self.relu2(x_combined) # Часто применяют ReLU после сложения
            # Если skip не было, используем выход второго блока
            x_gru_input_features = x_combined
        else:
            x_gru_input_features = x_cnn2


        # Меняем местами для GRU: (N, L, C2)
        x_gru_input = x_gru_input_features.permute(0, 2, 1)

        # --- GRU Часть ---
        self.gru.flatten_parameters()
        # out_gru shape: (N, L, H_out * num_directions)
        out_gru, hn = self.gru(x_gru_input)

        # --- MLP Часть ---
        # Берем выход GRU с последнего временного шага
        # out_gru[:, -1, :] будет иметь размер (N, gru_hidden_dim * num_directions)
        last_time_step_out = out_gru[:, -1, :]

        output = self.mlp(last_time_step_out)
        return output
# --- Функция для создания последовательностей ---
# (Она понадобится после масштабирования)
def create_sequences(input_data, target_data, sequence_length):
    sequences, targets = [], []
    if len(input_data) <= sequence_length:
        print(f"Предупреждение: Длина данных ({len(input_data)}) меньше или равна sequence_length ({sequence_length}). Последовательности не будут созданы.")
        return np.array([]), np.array([])
    for i in range(sequence_length, len(input_data)):
        sequences.append(input_data[i-sequence_length:i])
        targets.append(target_data[i])
    if not sequences: # Дополнительная проверка
        return np.array([]), np.array([])
    return np.array(sequences), np.array(targets)


# --- Основная функция ---
def main(config, data_csv_path, output_model_path, output_scaler_path, output_plot_path): # <<< Добавлен путь для графика
    """Основная функция обучения и оценки CNN-GRU модели с сохранением графика."""

    print("--- Конфигурация CNN-GRU ---") # <--- изменено название
    for key, value in config.items(): print(f"{key}: {value}")
    print("-" * 30)

    # --- Параметры из конфигурации ---
    target_col = config['target_col']
    final_test_set_size = config['final_test_set_size']
    k_folds = config['k_folds']
    val_split_for_final = config['val_split_for_final'] # Для разделения ПОСЛЕ KFold
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    weight_decay = config['weight_decay']
    early_stopping_patience = config['early_stopping_patience']
    scheduler_patience = config['scheduler_patience']
    scheduler_factor = config['scheduler_factor']
    # --- Внутри функции main ---

    # ... (старые параметры) ...
    random_seed = config['random_seed']

    # >>> Параметры для CNN-GRU V2 <<<
    sequence_length = config.get('sequence_length', 20)
    dropout_rate = config.get('dropout_rate', 0.3)
    # Параметры CNN
    cnn_out_channels_1 = config.get('cnn_out_channels_1', 64)  # Параметр для 1го слоя
    cnn_kernel_size_1 = config.get('cnn_kernel_size_1', 5)  # Параметр для 1го слоя
    cnn_out_channels_2 = config.get('cnn_out_channels_2', 128)  # Параметр для 2го слоя
    cnn_kernel_size_2 = config.get('cnn_kernel_size_2', 3)  # Параметр для 2го слоя
    # Параметры GRU
    gru_hidden_dim = config.get('gru_hidden_dim', 128)
    gru_num_layers = config.get('gru_num_layers', 2)
    gru_bidirectional = config.get('gru_bidirectional', True)  # Новый параметр
    # Параметры MLP
    mlp_hidden_1 = config.get('mlp_hidden_1', 64)  # Новый параметр для 1го MLP слоя
    mlp_hidden_2 = config.get('mlp_hidden_2', 32)  # Новый параметр для 2го MLP слоя

    print("--- Параметры CNN-GRU V2 ---")  # Обновлено
    print(f"sequence_length: {sequence_length}")
    print(f"dropout_rate: {dropout_rate}")
    print(f"cnn_out_channels_1: {cnn_out_channels_1}")
    print(f"cnn_kernel_size_1: {cnn_kernel_size_1}")
    print(f"cnn_out_channels_2: {cnn_out_channels_2}")
    print(f"cnn_kernel_size_2: {cnn_kernel_size_2}")
    print(f"gru_hidden_dim: {gru_hidden_dim}")
    print(f"gru_num_layers: {gru_num_layers}")
    print(f"gru_bidirectional: {gru_bidirectional}")  # Выводим новый параметр
    print(f"mlp_hidden_1: {mlp_hidden_1}")  # Выводим новые параметры
    print(f"mlp_hidden_2: {mlp_hidden_2}")
    print("-" * 30)
    # <<<--------------------------->>>

    # Настройка устройства и seed
    if torch.cuda.is_available(): device = torch.device("cuda"); print(f"GPU: {torch.cuda.get_device_name(0)}")
    else: device = torch.device("cpu"); print("CPU.")
    torch.manual_seed(random_seed); np.random.seed(random_seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(random_seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    # --- 1. Загрузка и начальная обработка данных ---
    print("\n--- 1. Загрузка и начальная обработка ---")
    try:
        df = pd.read_parquet(data_csv_path)
        if 'TRADEDATE' not in df.columns: raise ValueError("Нет колонки TRADEDATE")
        if not pd.api.types.is_datetime64_any_dtype(df['TRADEDATE']):
            df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
        df = df.sort_values(by='TRADEDATE').reset_index(drop=True) # Сортировка и сброс индекса
        print(f"Данные загружены. Форма: {df.shape}")
    except Exception as e: raise SystemExit(f"Ошибка чтения {data_csv_path}: {e}")

    if target_col not in df.columns: raise ValueError(f"Целевой столбец '{target_col}' не найден!")
    feature_cols = [col for col in df.columns if col not in [target_col, 'TRADEDATE']]
    print(f"Целевая переменная: {target_col}, Признаков иСХОДНЫХ: {len(feature_cols)}")
    if not feature_cols: raise ValueError("Нет признаков для использования.")

    print(f"Размер данных до проверки NaN: {df.shape}")
    df_cleaned = df.dropna(subset=[target_col] + feature_cols).copy()
    print(f"Размер данных после проверки NaN: {df_cleaned.shape}")
    if df_cleaned.empty: raise ValueError("Нет данных после удаления NaN.")

    # --- 2. Подготовка X, y (Все еще 2D) и дат ---
    print("\n--- 2. Подготовка X, y (2D) и дат ---")
    X_cleaned = df_cleaned[feature_cols].values # Numpy array (N, num_features)
    y_cleaned = df_cleaned[target_col].values.reshape(-1, 1) # Numpy array (N, 1)
    dates_cleaned = df_cleaned['TRADEDATE'].values # Numpy array (N,) дат

    # <<< иЗМЕНЕНО: Определяем input_dim для модели СЕЙЧАС (кол-во признаков на шаге) >>>
    input_dim_model = X_cleaned.shape[1]
    print(f"Определен input_dim для модели (признаков на шаге): {input_dim_model}")

    # --- 3. Разделение данных на временные (temp) и финальный тест ---
    #    Делаем ДО масштабирования и создания последовательностей! shuffle=False важно!
    print("\n--- 3. Разделение на Temp / Final Test (до масштабирования) ---")
    if X_cleaned.shape[0] < 2: raise ValueError("Недостаточно данных для разделения.")

    n_samples_total = X_cleaned.shape[0]
    n_test = int(n_samples_total * final_test_set_size)
    if n_test == 0 and final_test_set_size > 0: n_test = 1 # Хотя бы 1 образец в тесте, если возможно
    n_temp = n_samples_total - n_test
    if n_temp <= 0 or n_test <= 0: raise ValueError(f"Некорректные размеры после расчета разделения: temp={n_temp}, test={n_test}")

    # Разделяем вручную, т.к. shuffle=False
    X_temp = X_cleaned[:n_temp]
    X_final_test_orig = X_cleaned[n_temp:] # Сохраним немасштабированный для информации
    y_temp = y_cleaned[:n_temp]
    y_final_test_orig = y_cleaned[n_temp:]
    dates_temp = dates_cleaned[:n_temp]
    dates_final_test = dates_cleaned[n_temp:]

    print(f"Разделение выполнено: Temp={X_temp.shape[0]}, Final Test={X_final_test_orig.shape[0]}")

    # --- 4. Масштабирование ---
    #    Обучаем Scaler ТОЛЬКО на X_temp и применяем ко всему
    print("\n--- 4. Масштабирование данных ---")
    scaler = StandardScaler()
    print("Обучение Scaler на X_temp...")
    X_temp_scaled = scaler.fit_transform(X_temp)
    print("Масштабирование X_final_test...")
    X_final_test_scaled = scaler.transform(X_final_test_orig)

    # Сохраняем ОДиН обученный scaler
    try:
        with open(output_scaler_path, 'wb') as f: pickle.dump(scaler, f)
        print(f"Scaler (обученный на X_temp) сохранен в: {output_scaler_path}")
    except Exception as e: print(f"Ошибка сохранения scaler: {e}")

    # --- 5. Создание ПОСЛЕДОВАТЕЛЬНОСТЕЙ ---
    print("\n--- 5. Создание последовательностей ---")
    print(f"Создание последовательностей для Temp данных (длина {sequence_length})...")
    X_temp_seq, y_temp_seq = create_sequences(X_temp_scaled, y_temp, sequence_length)
    # Даты для temp последовательностей (соответствуют y_temp_seq)
    dates_temp_seq = dates_temp[sequence_length:]

    print(f"Создание последовательностей для Final Test данных (длина {sequence_length})...")
    X_final_test_seq, y_final_test_seq = create_sequences(X_final_test_scaled, y_final_test_orig, sequence_length)
     # Даты для final test последовательностей (соответствуют y_final_test_seq)
    dates_final_test_seq = dates_final_test[sequence_length:]

    if X_temp_seq.shape[0] == 0: raise ValueError("Не удалось создать обучающие последовательности (Temp).")
    if X_final_test_seq.shape[0] == 0: print("Предупреждение: Не удалось создать тестовые последовательности (Final Test). Финальная оценка будет пропущена.") # Не фатально, если тест пуст

    print(f"Размеры после создания последовательностей:")
    print(f"  Temp Seq: X={X_temp_seq.shape}, y={y_temp_seq.shape}, dates={dates_temp_seq.shape}")
    print(f"  Test Seq: X={X_final_test_seq.shape}, y={y_final_test_seq.shape}, dates={dates_final_test_seq.shape}")


    # --- 6. Кросс-валидация (на последовательностях Temp) ---
    #    Теперь работает с X_temp_seq, y_temp_seq
    kf = KFold(n_splits=k_folds, shuffle=False) # shuffle=False для временных данных
    fold_mses, fold_r2s = [], []; best_epochs_per_fold = []
    print(f"\n--- Начало {k_folds}-кратной кросс-валидации (CNN-GRU) ---") # <--- изменено название

    # Проверяем, достаточно ли данных для KFold
    if X_temp_seq.shape[0] < k_folds:
        print(f"Предупреждение: Количество образцов в Temp ({X_temp_seq.shape[0]}) меньше k_folds ({k_folds}). KFold будет пропущен.")
        k_folds = 0 # Пропускаем цикл KFold

    for fold, (train_indices, val_indices) in enumerate(kf.split(X_temp_seq, y_temp_seq)):
        print(f"\n--- ФОЛД {fold + 1}/{k_folds} ---")
        # Подготовка данных фолда (уже последовательности и масштабированы)
        X_train_fold_seq, X_val_fold_seq = X_temp_seq[train_indices], X_temp_seq[val_indices]
        y_train_fold_seq, y_val_fold_seq = y_temp_seq[train_indices], y_temp_seq[val_indices]

        # Конвертация в тензоры
        X_train_fold_tensor = torch.tensor(X_train_fold_seq, dtype=torch.float32).to(device)
        y_train_fold_tensor = torch.tensor(y_train_fold_seq, dtype=torch.float32).to(device)
        X_val_fold_tensor = torch.tensor(X_val_fold_seq, dtype=torch.float32).to(device)
        y_val_fold_tensor = torch.tensor(y_val_fold_seq, dtype=torch.float32).to(device)

        # Создание DataLoader'ов
        train_fold_dataset = TensorDataset(X_train_fold_tensor, y_train_fold_tensor)
        # shuffle=True в обучающем загрузчике допустимо, т.к. перемешиваются уже готовые последовательности
        train_fold_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)
        val_fold_dataset = TensorDataset(X_val_fold_tensor, y_val_fold_tensor)
        val_fold_loader = DataLoader(val_fold_dataset, batch_size=batch_size, shuffle=False)

        # >>> инициализация CNN-GRU модели для фолда <<<
        model = CNNGRUModelV2(input_dim=input_dim_model, sequence_length=sequence_length,
                              cnn_out_channels_1=cnn_out_channels_1, cnn_kernel_size_1=cnn_kernel_size_1,
                              cnn_out_channels_2=cnn_out_channels_2, cnn_kernel_size_2=cnn_kernel_size_2,
                              gru_hidden_dim=gru_hidden_dim, gru_num_layers=gru_num_layers,
                              gru_bidirectional=gru_bidirectional,  # <-- Передаем
                              mlp_hidden_1=mlp_hidden_1, mlp_hidden_2=mlp_hidden_2,  # <-- Передаем
                              dropout_rate=dropout_rate).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=False)

        # --- Цикл обучения фолда (структура та же, но модель другая) ---
        best_val_loss = float('inf'); epochs_no_improve = 0; best_model_state = None; best_epoch = -1
        start_time_fold = time.time(); print(f"  Начало обучения фолда {fold + 1}...")
        for epoch in range(epochs):
            model.train(); train_loss_epoch = 0.0; batches_processed_train = 0
            for batch_X_seq, batch_y_seq in train_fold_loader: # Данные теперь 3D
                optimizer.zero_grad()
                outputs = model(batch_X_seq) # Модель принимает 3D
                loss = criterion(outputs, batch_y_seq)
                loss.backward(); optimizer.step()
                train_loss_epoch += loss.item(); batches_processed_train += 1
            avg_train_loss = train_loss_epoch / batches_processed_train if batches_processed_train > 0 else 0

            model.eval(); val_loss_epoch = 0.0; batches_processed_val = 0
            with torch.no_grad():
                for batch_X_val_seq, batch_y_val_seq in val_fold_loader:
                    outputs_val = model(batch_X_val_seq) # Модель принимает 3D
                    loss_val = criterion(outputs_val, batch_y_val_seq)
                    val_loss_epoch += loss_val.item(); batches_processed_val += 1
            avg_val_loss = val_loss_epoch / batches_processed_val if batches_processed_val > 0 else float('inf')

            if (epoch + 1) % 10 == 0: # Печатаем реже
                 print(f'    Фолд {fold + 1}, Эпоха [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.1e}')

            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss; best_model_state = copy.deepcopy(model.state_dict()); best_epoch = epoch + 1; epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience: print(f"    Ранняя остановка на эпохе {epoch + 1}!"); break
        # --- Конец цикла обучения фолда ---
        end_time_fold = time.time()
        print(f"  Обучение фолда {fold + 1} завершено (лучшая эпоха {best_epoch}) за {end_time_fold - start_time_fold:.2f} сек.")
        best_epochs_per_fold.append(best_epoch)

        # Оценка лучшей модели фолда (на валидационных данных фолда)
        if best_model_state: model.load_state_dict(best_model_state)
        else: print("  Предупреждение: используется последняя модель фолда.")
        model.eval()
        with torch.no_grad(): y_val_pred_seq = model(X_val_fold_tensor).cpu().numpy() # Предсказание для 3D данных
        mse_fold = mean_squared_error(y_val_fold_seq, y_val_pred_seq); r2_fold = r2_score(y_val_fold_seq, y_val_pred_seq)
        print(f"  Метрики валидации фолда {fold + 1} (лучшая): MSE={mse_fold:.6f}, R²={r2_fold:.6f}")
        fold_mses.append(mse_fold); fold_r2s.append(r2_fold)
    # --- Конец цикла KFold ---

    print("\n--- Сводные результаты К-кратной кросс-валидации (CNN-GRU) ---") # <--- изменено
    if fold_mses:
         print(f"Среднее MSE по фолдам: {np.mean(fold_mses):.6f} (+/- {np.std(fold_mses):.6f})")
         print(f"Среднее R² по фолдам:  {np.mean(fold_r2s):.6f} (+/- {np.std(fold_r2s):.6f})")
         print(f"Среднее количество 'лучших' эпох: {np.mean(best_epochs_per_fold):.1f} (+/- {np.std(best_epochs_per_fold):.1f})")
    else: print("Кросс-валидация была пропущена или не дала результатов.")


    # --- 7. Обучение ФиНАЛЬНОЙ модели ---
    #    используем ВЕСЬ X_temp_seq для обучения (с небольшой валидацией для early stopping)
    print("\n--- 7. Обучение финальной модели на ПОСЛЕДОВАТЕЛЬНОСТЯХ Temp (CNN-GRU) ---")

    # Разделение Temp последовательностей на train_final/val_final (shuffle=False)
    if X_temp_seq.shape[0] < 2: raise ValueError("Недостаточно данных для финального обучения.")
    # используем train_test_split для удобства, но с shuffle=False
    X_train_final_seq, X_val_final_seq, y_train_final_seq, y_val_final_seq = train_test_split(
        X_temp_seq, y_temp_seq, test_size=val_split_for_final, shuffle=False
    )
    if X_train_final_seq.shape[0] == 0: raise ValueError("Ошибка разделения: нет данных для финального обучения.")
    print(f"Размеры для финального обучения: Train={X_train_final_seq.shape}, Val={X_val_final_seq.shape}")

    # <<< Удален блок с scaler_final - масштабирование уже сделано >>>

    # Подготовка тензоров и DataLoader'ов для финального обучения (на последовательностях)
    X_train_final_tensor = torch.tensor(X_train_final_seq, dtype=torch.float32).to(device)
    y_train_final_tensor = torch.tensor(y_train_final_seq, dtype=torch.float32).to(device)
    X_val_final_tensor = torch.tensor(X_val_final_seq, dtype=torch.float32).to(device)
    y_val_final_tensor = torch.tensor(y_val_final_seq, dtype=torch.float32).to(device)

    final_train_dataset = TensorDataset(X_train_final_tensor, y_train_final_tensor)
    final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True) # Shuffle=True для обучения
    final_val_dataset = TensorDataset(X_val_final_tensor, y_val_final_tensor)
    final_val_loader = DataLoader(final_val_dataset, batch_size=batch_size, shuffle=False) if len(final_val_dataset) > 0 else []


    # >>> инициализация ФиНАЛЬНОЙ CNN-GRU модели <<<
    final_model = CNNGRUModelV2(input_dim=input_dim_model, sequence_length=sequence_length,
                       cnn_out_channels_1=cnn_out_channels_1, cnn_kernel_size_1=cnn_kernel_size_1,
                       cnn_out_channels_2=cnn_out_channels_2, cnn_kernel_size_2=cnn_kernel_size_2,
                       gru_hidden_dim=gru_hidden_dim, gru_num_layers=gru_num_layers,
                       gru_bidirectional=gru_bidirectional,
                       mlp_hidden_1=mlp_hidden_1, mlp_hidden_2=mlp_hidden_2,
                       dropout_rate=dropout_rate).to(device)

    final_criterion = nn.MSELoss()
    final_optimizer = optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    final_scheduler = ReduceLROnPlateau(final_optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=False)

    # --- Цикл обучения финальной модели (структура та же, модель другая) ---
    best_val_loss_final = float('inf'); epochs_no_improve_final = 0; best_model_state_final = None; best_epoch_final = -1; final_train_losses_history = []
    print(f"  Начало финального обучения (CNN-GRU)..."); start_time_final = time.time()
    for epoch in range(epochs):
        final_model.train(); train_loss_epoch = 0.0; batches_processed_train = 0
        for batch_X_seq, batch_y_seq in final_train_loader: # 3D данные
            final_optimizer.zero_grad()
            outputs = final_model(batch_X_seq) # 3D вход
            loss = final_criterion(outputs, batch_y_seq)
            loss.backward(); final_optimizer.step()
            train_loss_epoch += loss.item(); batches_processed_train += 1
        avg_train_loss = train_loss_epoch / batches_processed_train if batches_processed_train > 0 else 0
        final_train_losses_history.append(avg_train_loss)

        final_model.eval(); val_loss_epoch = 0.0; batches_processed_val = 0
        if final_val_loader: # Проверяем, есть ли валидационный загрузчик
             with torch.no_grad():
                  for batch_X_val_seq, batch_y_val_seq in final_val_loader:
                      outputs_val = final_model(batch_X_val_seq) # 3D вход
                      loss_val = final_criterion(outputs_val, batch_y_val_seq)
                      val_loss_epoch += loss_val.item(); batches_processed_val += 1
             avg_val_loss = val_loss_epoch / batches_processed_val if batches_processed_val > 0 else float('inf')
        else: # Если нет валидации, используем train loss для scheduler и early stopping
             avg_val_loss = avg_train_loss
             if epoch==0: print("  Предупреждение: Валидационный набор пуст, early stopping и scheduler будут работать по train loss.")

        if (epoch + 1) % 10 == 0: # Печатаем реже
             print(f'    Финал. обуч., Эпоха [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {final_optimizer.param_groups[0]["lr"]:.1e}')

        final_scheduler.step(avg_val_loss) # Передаем val_loss (или train_loss если val нет)
        # Early stopping по val_loss (или train_loss если val нет)
        if avg_val_loss < best_val_loss_final:
            best_val_loss_final = avg_val_loss; best_model_state_final = copy.deepcopy(final_model.state_dict()); best_epoch_final = epoch + 1; epochs_no_improve_final = 0
        else:
            epochs_no_improve_final += 1
            if epochs_no_improve_final >= early_stopping_patience: print(f"    Ранняя остановка фин. обучения на эпохе {epoch + 1}!"); break
    # --- Конец цикла обучения финальной модели ---
    end_time_final = time.time()
    print(f"Финальное обучение завершено (лучшая эпоха {best_epoch_final}) за {end_time_final - start_time_final:.2f} сек.")

    # --- 8. Сохранение ЛУЧШЕЙ ФиНАЛЬНОЙ модели ---
    #    (Код сохранения модели без изменений)
    if best_model_state_final:
        try:
            torch.save(best_model_state_final, output_model_path)
            print(f"Лучшая финальная модель (эпоха {best_epoch_final}) сохранена в: {output_model_path}")
            # Загружаем лучшее состояние в final_model для последующей оценки
            final_model.load_state_dict(best_model_state_final)
        except Exception as e: print(f"Ошибка сохранения/загрузки лучшей модели: {e}. используется последняя.")
    else:
        print("Предупреждение: Нет лучшего состояния, сохраняется последняя модель.")
        try: torch.save(final_model.state_dict(), output_model_path); print(f"Последняя модель сохранена: {output_model_path}")
        except Exception as e: print(f"Ошибка сохранения последней модели: {e}")

    # --- 9. Оценка ФиНАЛЬНОЙ модели на ОТДЕЛЬНОМ ТЕСТЕ ---
    #    используем X_final_test_seq, y_final_test_seq
    print("\n--- 9. Оценка ЛУЧШЕЙ ФиНАЛЬНОЙ модели на ОТДЕЛЬНОМ ФиНАЛЬНОМ ТЕСТЕ (CNN-GRU) ---") # <--- изменено
    final_model.eval() # Переводим модель в режим оценки

    if X_final_test_seq.shape[0] > 0:
        # <<< Масштабирование тестовых данных УЖЕ сделано в шаге 4 >>>
        # <<< Последовательности УЖЕ созданы в шаге 5 >>>

        # Конвертация в тензор
        X_final_test_tensor_eval = torch.tensor(X_final_test_seq, dtype=torch.float32).to(device)
        print(f"Подготовлен тензор для финальной оценки: {X_final_test_tensor_eval.shape}")

        # Получение предсказаний (можно использовать батчинг для больших тестов, но здесь делаем одним махом)
        with torch.no_grad():
            # используем модель, в которую загружено лучшее состояние (или последнее)
            y_pred_nn_seq = final_model(X_final_test_tensor_eval).cpu().numpy()

        print(f"Предсказания на финальном тесте получены: {y_pred_nn_seq.shape}")
        print(f"Реальные значения финального теста: {y_final_test_seq.shape}")

        # >>> Расчет метрик (на последовательностях) <<<
        mse_final_nn = mean_squared_error(y_final_test_seq, y_pred_nn_seq)
        r2_final_nn = r2_score(y_final_test_seq, y_pred_nn_seq)
        mae_final_nn = mean_absolute_error(y_final_test_seq, y_pred_nn_seq)

        print("\n--- Результаты оценки (CNN-GRU) ---") # <--- изменено
        print(f"MSE CNN-GRU:          {mse_final_nn:.6f}")
        print(f"RMSE CNN-GRU:         {np.sqrt(mse_final_nn):.6f}")
        print(f"MAE CNN-GRU:          {mae_final_nn:.6f}")
        print(f"R² CNN-GRU (vs Mean): {r2_final_nn:.6f}")

        # --- 10. Визуализация и СОХРАНЕНиЕ ГРАФиКА ---
        print("\n--- 10. Построение и сохранение графика результатов ---")
        try:
             plt.figure(figsize=(10, 5))
             # График Предсказания vs Реальность (Scatter plot)
             plt.scatter(y_final_test_seq, y_pred_nn_seq, alpha=0.6, label='Предсказание vs Реальность', s=10)
             min_val = min(np.nanmin(y_final_test_seq), np.nanmin(y_pred_nn_seq))
             max_val = max(np.nanmax(y_final_test_seq), np.nanmax(y_pred_nn_seq))
             plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='идеальное совпадение')
             plt.xlabel("Реальные значения (y_final_test_seq)")
             plt.ylabel("Предсказанные значения (y_pred_nn_seq)")
             plt.title(f"Сравнение Предсказаний (Финальный тест, CNN-GRU)\nЦель: {target_col}\nR²={r2_final_nn:.4f}") # <--- изменено
             plt.legend(); plt.grid(True); plt.tight_layout()
             plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
             print(f"График сравнения сохранен в: {output_plot_path}")
             plt.close() # Закрываем фигуру

             # График потерь (остается таким же)
             if final_train_losses_history: # Проверяем, что история не пуста
                 plt.figure(figsize=(10, 5))
                 plt.plot(range(1, len(final_train_losses_history) + 1), final_train_losses_history, label='MSE Обучения (финал)')
                 plt.xlabel("Эпоха"); plt.ylabel("Средняя ошибка (MSE)"); plt.title("Динамика ошибки финального обучения (CNN-GRU)") # <--- изменено
                 plt.legend(); plt.grid(True); plt.tight_layout()
                 # Генерируем имя файла для графика потерь
                 loss_plot_filename = f"loss_plot_cnngru_{os.path.splitext(os.path.basename(output_model_path))[0]}.png"
                 output_loss_plot_path = os.path.join(os.path.dirname(output_plot_path), loss_plot_filename)
                 plt.savefig(output_loss_plot_path, dpi=150, bbox_inches='tight')
                 print(f"График потерь сохранен в: {output_loss_plot_path}")
                 plt.close()
             else:
                 print("история потерь финального обучения пуста, график не создан.")

        except Exception as e:
             print(f"Ошибка при построении/сохранении графика: {e}")

    else:
        print("Финальный тестовый набор (последовательности) пуст. Оценка и визуализация невозможны.")

# --- Точка входа для скрипта ---
if __name__ == "__main__":
    # Добавьте в ваш config2.json параметры:
    # "sequence_length": 20,
    # "cnn_out_channels": 32,
    # "cnn_kernel_size": 3,
    # "gru_hidden_dim": 64,
    # "gru_num_layers": 1,
    # "mlp_hidden": 32,
    # "dropout_rate": 0.3

    parser = argparse.ArgumentParser(description='Train CNN-GRU model using DataSphere Jobs.') # <--- изменено
    parser.add_argument('--data_csv', type=str, required=True, help='Path to input Parquet file') # изменено на Parquet
    parser.add_argument('--config_json', type=str, required=True, help='Path to JSON config')
    parser.add_argument('--output_model', type=str, required=True, help='Path to save model state_dict')
    parser.add_argument('--output_scaler', type=str, required=True, help='Path to save the single scaler object')
    parser.add_argument('--output_plot', type=str, required=True, help='Path to save the results plot image')

    args = parser.parse_args()

    print("--- Параметры запуска ---")
    print(f"Данные Parquet: {args.data_csv}"); print(f"Конфиг JSON: {args.config_json}") # изменено
    print(f"Выход модель: {args.output_model}"); print(f"Выход Scaler: {args.output_scaler}")
    print(f"Выход График: {args.output_plot}")
    print("-" * 30)

    try:
        with open(args.config_json, 'r') as f: config_params = json.load(f)
        print(f"Конфигурация загружена из {args.config_json}")
    except Exception as e: raise SystemExit(f"Ошибка чтения конфигурации {args.config_json}: {e}")

    main(config_params, args.data_csv, args.output_model, args.output_scaler, args.output_plot)

    print("\n--- Задание DataSphere Job (CNN-GRU) завершено успешно ---") # <--- изменено