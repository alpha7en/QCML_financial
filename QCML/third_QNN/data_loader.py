# data_loader.py
import pandas as pd
import numpy as np # vanilla_np в вашем примере, здесь используем стандартный numpy
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

import config

def load_and_split_data(data_path, datetime_col, target_col, other_exclude_cols, test_size, batch_size):
    """
    Загружает, предварительно обрабатывает и разделяет данные на обучающую и тестовую выборки
    с учетом хронологии.
    """
    try:
        print(f"\nLoading data from: {data_path}")
        df = pd.read_parquet(data_path)
        print(f"Original data loaded. Shape: {df.shape}")

        if datetime_col not in df.columns:
            raise ValueError(f"Missing date column: {datetime_col}")

        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            print(f"Converting '{datetime_col}' to datetime...")
            df[datetime_col] = pd.to_datetime(df[datetime_col])

        print(f"Sorting data by '{datetime_col}'...")
        df = df.sort_values(by=datetime_col).reset_index(drop=True)

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        # Определение признаковых колонок
        exclude_cols = [target_col, datetime_col] + other_exclude_cols
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        if not feature_cols:
            raise ValueError("No feature columns found (excluding target, date, and other specified columns).")
        print(f"Input feature columns ({len(feature_cols)}): {feature_cols}")

        cols_to_check_nan = [target_col] + feature_cols
        initial_rows = len(df)
        df_cleaned = df.dropna(subset=cols_to_check_nan).copy()
        removed_rows = initial_rows - len(df_cleaned)
        if removed_rows > 0:
            print(f"Removed {removed_rows} rows due to NaN values in features or target.")
        if df_cleaned.empty:
            raise ValueError("No data left after NaN removal.")
        print(f"Data shape after NaN removal: {df_cleaned.shape}")

        X_input_np = df_cleaned[feature_cols].values.astype(np.float64)
        y_target_np = df_cleaned[target_col].values.astype(np.float64).reshape(-1, 1) # reshape for scaler

        k_features = X_input_np.shape[1]
        print(f"Input Features (K'): {k_features}, Target Variable: '{target_col}'")

        # Хронологическое разделение
        n_samples_total = len(df_cleaned)
        n_test = int(n_samples_total * test_size)
        n_train = n_samples_total - n_test

        if n_train <= 0 or n_test <= 0:
            raise ValueError(f"Not enough data to create train/test split. Train samples: {n_train}, Test samples: {n_test}")

        X_input_train_np = X_input_np[:n_train]
        X_input_test_np = X_input_np[n_train:]
        y_target_train_np = y_target_np[:n_train]
        y_target_test_np = y_target_np[n_train:]

        print(f"Train data shape: X={X_input_train_np.shape}, y={y_target_train_np.shape}")
        print(f"Test data shape: X={X_input_test_np.shape}, y={y_target_test_np.shape}")
        print(f"Train period: {df_cleaned[datetime_col].iloc[0]} to {df_cleaned[datetime_col].iloc[n_train-1]}")
        print(f"Test period: {df_cleaned[datetime_col].iloc[n_train]} to {df_cleaned[datetime_col].iloc[n_samples_total-1]}")


        # Масштабирование признаков и целевой переменной
        # Признаки масштабируем в [0, 1], что хорошо для AngleEmbedding (углы могут быть pi * значение)
        # Целевую переменную масштабируем в [-1, 1], так как выход QNN qml.expval(PauliZ) находится в этом диапазоне
        x_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaler = MinMaxScaler(feature_range=(-1, 1))

        X_train_scaled = x_scaler.fit_transform(X_input_train_np)
        X_test_scaled = x_scaler.transform(X_input_test_np)

        y_train_scaled = y_scaler.fit_transform(y_target_train_np)
        y_test_scaled = y_scaler.transform(y_target_test_np)

        # Преобразование в тензоры PyTorch
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

        # Создание DataLoader'ов
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # Перемешиваем только обучающую выборку
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        # Тестовую выборку не перемешиваем и не отбрасываем последний батч
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        return train_loader, test_loader, k_features, y_scaler, feature_cols

    except Exception as e:
        print(f"Error in load_and_split_data: {e}")
        raise # Перевыбрасываем исключение, чтобы остановить выполнение, если данные не загрузятся

if __name__ == '__main__':
    # Пример использования (для отладки)
    # Убедитесь, что файл config.py находится в той же директории
    # и в нем указаны корректные DATA_PATH, DATETIME_COLUMN, TARGET_COLUMN
    try:
        train_loader, test_loader, num_features, scaler, features = load_and_split_data(
            data_path=config.DATA_PATH,
            datetime_col=config.DATETIME_COLUMN,
            target_col=config.TARGET_COLUMN,
            other_exclude_cols=config.OTHER_EXCLUDE_COLS,
            test_size=config.TEST_SIZE,
            batch_size=config.BATCH_SIZE
        )
        print(f"\nData loaded successfully. Number of features: {num_features}")
        print(f"Feature names: {features}")
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Test loader: {len(test_loader)} batches")

        # Проверка одного батча
        for X_batch, y_batch in train_loader:
            print(f"X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}")
            break
    except ValueError as ve:
        print(f"ValueError during data loading: {ve}")
    except FileNotFoundError:
        print(f"Data file not found at {config.DATA_PATH}. Please check config.py.")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")