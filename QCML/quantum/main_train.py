import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
# Убираем train_test_split, будем делить вручную
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
import sys

from qcml_model import QCMLModel

# --- Вспомогательная функция для VQE (без изменений) ---
def find_ground_state_vqe(model, thetas_input, xs_input, initial_phi,
                          vqe_steps=100, vqe_lr=0.05, vqe_tol=1e-4, verbose=True):
    # ... (код без изменений) ...
    if verbose: print(f"    Starting VQE optimization ({vqe_steps} steps, lr={vqe_lr})...")
    phi = np.copy(initial_phi); opt_phi = qml.AdamOptimizer(stepsize=vqe_lr); energy_prev = np.inf
    for i in range(vqe_steps):
        phi, energy = opt_phi.step_and_cost(lambda v: model.energy_vqe(v, thetas_input, xs_input), phi)
        if verbose and (i % 20 == 0 or i == vqe_steps - 1): print(f"      VQE iter {i:>3}: energy = {energy:.6f}")
        if np.abs(energy - energy_prev) < vqe_tol:
            if verbose: print(f"      VQE converged at step {i}, energy = {energy:.6f}"); break
        energy_prev = energy
    if verbose: final_energy = model.energy_vqe(phi, thetas_input, xs_input); print(f"    VQE finished. Final energy = {final_energy:.6f}")
    return phi

# --- Функция Предсказания (без изменений) ---
def predict(model, thetas_input_trained, X_input_features, initial_phi_guess, vqe_config):
    # ... (код без изменений) ...
    print(f"\n--- Generating Target Predictions for {len(X_input_features)} samples ---")
    predictions = []; phi_guess = np.copy(initial_phi_guess); pred_start_time = time.time()
    if thetas_input_trained.shape[0] != model.k_features: raise ValueError(f"Dim mismatch: thetas ({thetas_input_trained.shape[0]}) vs model k ({model.k_features})")
    if X_input_features.shape[1] != model.k_features: raise ValueError(f"Dim mismatch: features ({X_input_features.shape[1]}) vs model k ({model.k_features})")
    for i, xs_input_t in enumerate(X_input_features):
        if (i+1) % 50 == 0 or i == len(X_input_features) - 1: print(f"  Predicting sample {i+1}/{len(X_input_features)}...")
        phi_pred = find_ground_state_vqe(model, thetas_input_trained, xs_input_t, phi_guess, verbose=False, **vqe_config)
        phi_guess = phi_pred; pred_value = model.predict_target(phi_pred); predictions.append(pred_value)
    pred_end_time = time.time(); print(f"--- Predictions finished in {pred_end_time - pred_start_time:.2f}s ---")
    return np.array(predictions)


# --- Основная Функция Обучения и Оценки ---
def train_and_evaluate(config_path):
    """
    Загружает конфигурацию, данные, обучает QCML модель с разделением по дате,
    оценивает и строит графики.
    """
    print("--- Starting Training and Evaluation ---")

    # --- 0. Загрузка Конфигурации ---
    try:
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f: config = json.load(f)
        print("Configuration loaded successfully.")
        data_path = config['data']['data_path']
        target_col = config['data']['target_col']
        test_size = config['training']['test_size'] # Доля для теста
        # random_state теперь не нужен для разделения, но оставим для воспроизводимости VQE/theta init
        random_state = config['training'].get('random_state', 42)
        np.random.seed(random_state) # Установка seed для NumPy

        n_qubits = config['model']['n_qubits']; n_layers_vqe = config['model']['n_layers_vqe']; l_u_layers = config['model']['l_u_layers']
        n_epochs = config['training']['n_epochs']; batch_size = config['training']['batch_size']; vqe_steps = config['training']['vqe_steps']; vqe_lr = config['training']['vqe_lr']; theta_lr = config['training']['theta_lr']; vqe_tol = config['training']['vqe_tol']
        results_prefix = config['output']['results_prefix']

    except Exception as e: print(f"Error loading configuration: {e}"); return

    # --- 1. Загрузка, Сортировка и Подготовка Данных ---
    try:
        print(f"Loading data from: {data_path}")
        df = pd.read_parquet(data_path)
        print(f"Original data loaded. Shape: {df.shape}")

        # Проверка и обработка TRADEDATE
        date_col = 'TRADEDATE' # Колонка с датой
        if date_col not in df.columns: raise ValueError(f"Missing date column: {date_col}")
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            print(f"Converting '{date_col}' to datetime...")
            df[date_col] = pd.to_datetime(df[date_col])
        print(f"Sorting data by '{date_col}'...")
        df = df.sort_values(by=date_col).reset_index(drop=True) # Сортировка

        # Определение признаков (ИСКЛЮЧАЯ target И date)
        if target_col not in df.columns: raise ValueError(f"Target column '{target_col}' not found.")
        feature_cols = [col for col in df.columns if col not in [target_col, date_col]]
        if not feature_cols: raise ValueError("No feature columns found (excluding target and date).")
        print(f"Input feature columns: {feature_cols}")

        # Обработка NaN (после определения колонок)
        cols_to_check_nan = [target_col] + feature_cols
        initial_rows = len(df)
        df_cleaned = df.dropna(subset=cols_to_check_nan).copy()
        removed_rows = initial_rows - len(df_cleaned)
        if removed_rows > 0: print(f"Removed {removed_rows} rows due to NaN values in relevant columns.")
        if df_cleaned.empty: raise ValueError("No data left after removing NaN values.")
        print(f"Data shape after NaN removal: {df_cleaned.shape}")

        # Извлечение X и y из ОЧИЩЕННЫХ и ОТСОРТИРОВАННЫХ данных
        X_input = df_cleaned[feature_cols].values # K' признаков
        y_target = df_cleaned[target_col].values # Цель
        dates_cleaned = df_cleaned[date_col].values # Сохраним даты для информации
        k_features = X_input.shape[1] # K'

        print(f"Input Features (K'): {k_features}, Target Variable: '{target_col}'")

    except Exception as e:
        print(f"Error loading/preparing data: {e}")
        return

    # --- 1.5. Разделение Train/Test по Дате (ВРУЧНУЮ) ---
    n_samples_total = len(df_cleaned)
    if n_samples_total < 2: raise ValueError("Not enough data samples to split.")

    n_test = int(n_samples_total * test_size)
    if test_size > 0 and n_test == 0: n_test = 1 # Гарантируем хотя бы 1 тестовый сэмпл
    n_train = n_samples_total - n_test

    if n_train <= 0 or n_test <= 0:
        raise ValueError(f"Invalid train/test split sizes: train={n_train}, test={n_test}")

    # Берем первые n_train для обучения, последние n_test для теста
    X_input_train = X_input[:n_train]
    X_input_test  = X_input[n_train:]
    y_target_train = y_target[:n_train]
    y_target_test  = y_target[n_train:]
    dates_train = dates_cleaned[:n_train]
    dates_test = dates_cleaned[n_train:]

    print(f"\nData split by date ({1-test_size:.0%} train / {test_size:.0%} test):")
    print(f"  Train: {n_train} samples (until {pd.to_datetime(dates_train[-1]).date()})")
    print(f"  Test:  {n_test} samples (from {pd.to_datetime(dates_test[0]).date()})")
    print(f"  Train input features shape: {X_input_train.shape}")
    print(f"  Test input features shape: {X_input_test.shape}")


    # --- 2. Инициализация Модели и Параметров ---
    dev = qml.device("lightning.qubit", wires=n_qubits)
    model = QCMLModel(
        n_qubits=n_qubits, k_input_features=k_features, # K' признаков
        n_layers_vqe=n_layers_vqe, l_u_layers=l_u_layers, dev=dev
    )

    initial_phi = np.random.normal(0, 0.01, size=model.phi_shape)
    initial_thetas = np.random.normal(0, 0.1, size=model.thetas_shape) # Форма (K', ...)
    thetas = np.copy(initial_thetas); phi_guess = np.copy(initial_phi)
    opt_theta = qml.AdamOptimizer(stepsize=theta_lr)
    theta_grad_fn = qml.grad(model.cost_for_theta_gradient, argnum=0)


    # --- 3. Цикл Обучения с Батчингом (используем K' признаков) ---
    print("\n--- Starting QCML Training (using K' features/operators for H) ---")
    start_time_total = time.time(); history = {'epoch': [], 'avg_epoch_cost': [], 'train_mse': []}
    vqe_config = {'vqe_steps': vqe_steps, 'vqe_lr': vqe_lr, 'vqe_tol':vqe_tol}

    # Получаем индексы для обучающей выборки
    train_indices_original_order = np.arange(n_train) # Индексы от 0 до n_train-1

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        epoch_start_time = time.time(); total_cost_epoch = 0; num_batches = 0

        # ПЕРЕМЕШИВАЕМ индексы ТРЕНИРОВОЧНОЙ выборки для батчинга
        shuffled_train_indices = np.random.permutation(train_indices_original_order)

        for i in range(0, n_train, batch_size):
            # Берем батч ПЕРЕМЕШАННЫХ индексов
            batch_indices = shuffled_train_indices[i:min(i + batch_size, n_train)]
            # Выбираем данные по этим индексам из ТРЕНИРОВОЧНЫХ массивов
            batch_X_input = X_input_train[batch_indices]

            if len(batch_X_input) == 0: continue

            num_batches += 1; batch_start_time = time.time()
            print(f"  Batch {num_batches}/{int(np.ceil(n_train/batch_size))}")

            batch_grads_sum = np.zeros_like(thetas); batch_cost_sum = 0.0

            for xs_input_t in batch_X_input: # K' признаков
                # VQE Фаза
                phi_opt = find_ground_state_vqe(model, thetas, xs_input_t, phi_guess, verbose=False, **vqe_config)
                phi_guess = phi_opt
                # Theta Gradient Фаза
                psi_t = model.get_state_vector(phi_opt)
                current_cost = model.cost_for_theta_gradient(thetas, psi_t, xs_input_t)
                grad_theta = theta_grad_fn(thetas, psi_t, xs_input_t)
                batch_grads_sum += grad_theta; batch_cost_sum += current_cost

            avg_batch_grad = batch_grads_sum / len(batch_X_input)
            avg_batch_cost = batch_cost_sum / len(batch_X_input)
            thetas = opt_theta.apply_grad(avg_batch_grad, [thetas])[0]

            batch_end_time = time.time()
            print(f"    Batch finished. Avg Cost E_t = {avg_batch_cost:.6f}. Time: {batch_end_time - batch_start_time:.2f}s")
            total_cost_epoch += avg_batch_cost

        # Оценка MSE на Обучающей Выборке после Эпохи (на НЕ перемешанных train данных)
        print("  Evaluating MSE on Training set (original order) for epoch logging...")
        train_preds = predict(model, thetas, X_input_train, initial_phi, vqe_config)
        train_mse = mean_squared_error(y_target_train, train_preds)
        print(f"  Training MSE (vs {target_col}) after epoch {epoch+1}: {train_mse:.6f}")

        # Логирование истории
        avg_cost_epoch = total_cost_epoch / num_batches if num_batches > 0 else 0
        history['epoch'].append(epoch + 1); history['avg_epoch_cost'].append(avg_cost_epoch); history['train_mse'].append(train_mse)
        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} finished. Average Batch Cost (E_t) = {avg_cost_epoch:.6f}. Epoch time: {epoch_end_time - epoch_start_time:.2f}s")

    end_time_total = time.time()
    print(f"\n--- Training Finished ---"); print(f"Total training time: {end_time_total - start_time_total:.2f}s")
    trained_thetas = thetas

    # --- 4. Оценка на Тестовой Выборке (НЕ перемешанной) ---
    print("\n--- Evaluating on Test Set ---")
    test_preds = predict(model, trained_thetas, X_input_test, initial_phi, vqe_config)
    test_mse = mean_squared_error(y_target_test, test_preds)
    print(f"Final Test MSE (vs {target_col}): {test_mse:.6f}")

    # --- 5. Построение Графиков ---
    print("\n--- Plotting Results ---")
    # ... (код графиков без изменений) ...
    try:
        plt.figure(figsize=(12, 5)); ax1 = plt.subplot(1, 2, 1); ax1.plot(history['epoch'], history['avg_epoch_cost'], 'bo-', label='Avg. Batch Cost (E_t)'); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Avg. Cost (E_t)", color='b'); ax1.tick_params(axis='y', labelcolor='b'); ax1.set_title("Training Progress per Epoch"); ax1.grid(True); ax2 = ax1.twinx(); ax2.plot(history['epoch'], history['train_mse'], 'ro-', label=f'Train MSE (vs {target_col})'); ax2.set_ylabel("Train MSE", color='r'); ax2.tick_params(axis='y', labelcolor='r'); lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); ax2.legend(lines + lines2, labels + labels2, loc='best'); plt.subplot(1, 2, 2); plt.scatter(y_target_test, test_preds, alpha=0.6, label=f"Test MSE: {test_mse:.4f}"); min_val = min(np.min(y_target_test), np.min(test_preds)) - 0.1 ; max_val = max(np.max(y_target_test), np.max(test_preds)) + 0.1 ; plt.plot([min_val, max_val], [min_val, max_val], '--k', label="Ideal (y=x)"); plt.xlabel(f"True Values ({target_col})"); plt.ylabel(f"Predicted Values ({target_col})"); plt.title("Prediction vs True Values (Test Set)"); plt.legend(); plt.grid(True); plt.axis('equal'); plt.xlim(min_val, max_val); plt.ylim(min_val, max_val); plt.tight_layout(); plot_filename = f"{results_prefix}_plots.png"; plt.savefig(plot_filename); print(f"Plots saved to {plot_filename}"); plt.show()
    except Exception as e: print(f"Error during plotting: {e}")

    print("\n--- Process Finished ---")
    return trained_thetas, history


# --- Точка входа ---
if __name__ == "__main__":
    config_file_path = 'config.json'
    train_and_evaluate(config_file_path)