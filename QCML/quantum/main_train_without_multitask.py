
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import json
import sys
import os
# import multiprocessing # REMOVED
# from functools import partial # REMOVED
import torch

# --- Проверка и заглушка QCMLModel (остается без изменений) ---
try:
    from qcml_model import QCMLModel
except ImportError:
    print("Ошибка: Не удалось импортировать QCMLModel из qcml_model.")
    print("Убедитесь, что файл qcml_model2.py находится в той же директории или доступен в PYTHONPATH.")
    # Заглушка (как в предыдущем варианте)
    class QCMLModel:
        def __init__(self, n_qubits, k_input_features, n_layers_vqe, l_u_layers, dev):
            print("ПРЕДУПРЕЖДЕНИЕ: Используется заглушка для QCMLModel.")
            self.n_qubits = n_qubits
            self.k_input_features = k_input_features
            self.n_layers_vqe = n_layers_vqe
            self.l_u_layers = l_u_layers
            self.dev = dev
            self.phi_shape = (n_layers_vqe, n_qubits, 3) # Пример
            self.thetas_shape = (l_u_layers, k_input_features) # Пример
            print(f"Создана заглушка QCMLModel с phi_shape={self.phi_shape}, thetas_shape={self.thetas_shape}")
            # Методы-заглушки (как в предыдущем варианте)
            def energy_vqe(self, phi, thetas_input, xs_input): return np.random.rand()
            def get_state_vector(self, phi): return np.random.rand(2**self.n_qubits) + 1j * np.random.rand(2**self.n_qubits)
            def cost_for_theta_gradient(self, thetas, psi_t, xs_input): return np.random.rand()
            def predict_target(self, phi_pred): return np.random.rand() # Просто случайное число для предсказания
    # sys.exit(1) # Не выходим, чтобы код мог работать с заглушкой


# --- Вспомогательная функция find_ground_state_vqe (без изменений) ---
def find_ground_state_vqe(model, thetas_input, xs_input, initial_phi,
                          vqe_steps=100, vqe_lr=0.05, vqe_tol=1e-5, verbose=True):
    """Находит основное состояние с помощью VQE."""
    if verbose: print(f" Starting VQE optimization (steps={vqe_steps}, lr={vqe_lr}, tol={vqe_tol})...")
    phi = np.copy(initial_phi)
    opt_phi = qml.AdamOptimizer(stepsize=vqe_lr)
    energy_prev = np.inf
    steps_done = 0
    for i in range(vqe_steps):
        steps_done=i+1
        phi, energy = opt_phi.step_and_cost(lambda v: model.energy_vqe(v, thetas_input, xs_input), phi)
        if verbose and (i % 10 == 0 or i == vqe_steps - 1): # Реже выводим VQE
             print(f"  VQE iter {i:>3}: energy = {energy:.6f}")
        if np.abs(energy - energy_prev) < vqe_tol:
            if verbose: print(f" VQE converged at step {i+1}, energy = {energy:.6f}")
            break
        energy_prev = energy
    final_energy = model.energy_vqe(phi, thetas_input, xs_input)
    return phi, final_energy


# --- Вспомогательная функция find_ground_state_vqe с PyTorch оптимизатором (без изменений) ---
def find_ground_state_vqe_pytorch(model, thetas_input_np, xs_input_np, initial_phi_np,
                                  vqe_steps=100, vqe_lr=0.05, vqe_tol=1e-5, verbose=True):
    """
    Находит основное состояние с помощью VQE, используя PyTorch оптимизатор (torch.optim.Adam)
    и PyTorch-интерфейс QNode в модели.
    """
    if verbose: print(f" Starting VQE optimization (PyTorch Adam, steps={vqe_steps}, lr={vqe_lr}, tol={vqe_tol})...")
    device = torch.device("cpu")
    phi = torch.tensor(initial_phi_np, dtype=torch.float64, device=device, requires_grad=True)
    thetas_input_torch = torch.tensor(thetas_input_np, dtype=torch.float64, device=device)
    xs_input_torch = torch.tensor(xs_input_np, dtype=torch.float64, device=device)
    opt_phi_torch = torch.optim.Adam([phi], lr=vqe_lr)
    energy_prev = float('inf')
    steps_done = 0
    for i in range(vqe_steps):
        steps_done = i + 1
        opt_phi_torch.zero_grad()
        current_energy = model.energy_vqe_pytorch(phi, thetas_input_torch, xs_input_torch)
        current_energy.backward()
        opt_phi_torch.step()
        energy_val = current_energy.item()
        if verbose and (i % 10 == 0 or i == vqe_steps - 1):
            print(f"  VQE (PyTorch) iter {i:>3}: energy = {energy_val:.6f}")
        if abs(energy_val - energy_prev) < vqe_tol:
            if verbose: print(f" VQE (PyTorch) converged at step {i + 1}, energy = {energy_val:.6f}")
            break
        energy_prev = energy_val
    with torch.no_grad():
        final_energy_val = model.energy_vqe_pytorch(phi, thetas_input_torch, xs_input_torch).item()
    return phi.detach().cpu().numpy(), final_energy_val

# --- Глобальная переменная для хранения модели в воркере (REMOVED) ---
# worker_model = None

# --- Функция инициализации воркера (REMOVED) ---
# def init_worker(model_config):
#     ...

# --- ФУНКЦИЯ для обработки одного сэмпла ОБУЧЕНИЯ (ИЗМЕНЕНО: принимает model) ---
def process_single_sample(model, xs_input_t, current_thetas, phi_guess, vqe_config): # Added model argument
    """
    Обрабатывает один сэмпл для ОБУЧЕНИЯ: использует переданную 'model'.
    Возвращает кортеж (gradient_theta, cost_e_t, final_phi).
    """
    # global worker_model # REMOVED

    if model is None: # Should not happen if model is passed correctly
        print(f"Error: model argument is None in process_single_sample.")
        try: thetas_shape = current_thetas.shape
        except: thetas_shape = (1,)
        try: phi_shape = phi_guess.shape
        except: phi_shape = (1,)
        return np.zeros(thetas_shape), np.inf, np.zeros(phi_shape)

    try:
        # --- VQE Фаза ---
        # Используем model напрямую
        phi_opt, _ = find_ground_state_vqe_pytorch(model, current_thetas, xs_input_t, phi_guess, verbose=False, **vqe_config)

        # --- Theta Gradient Фаза ---
        psi_t = model.get_state_vector(phi_opt)

        try:
            # Используем model напрямую
            local_theta_grad_fn = qml.grad(model.cost_for_theta_gradient, argnum=0)
            grad_theta = local_theta_grad_fn(current_thetas, psi_t, xs_input_t)
            current_cost = model.cost_for_theta_gradient(current_thetas, psi_t, xs_input_t)
        except AttributeError:
            print(f"Warning: cost_for_theta_gradient not fully implemented in QCMLModel заглушка.")
            grad_theta = np.zeros_like(current_thetas)
            current_cost = np.random.rand()

        return grad_theta, current_cost, phi_opt

    except Exception as e:
        print(f"Error processing training sample: {e}")
        try: thetas_shape = model.thetas_shape
        except: thetas_shape = current_thetas.shape
        try: phi_shape = model.phi_shape
        except: phi_shape = phi_guess.shape
        return np.zeros(thetas_shape), np.inf, np.zeros(phi_shape)

# --- ФУНКЦИЯ для обработки одного сэмпла ПРЕДСКАЗАНИЯ (ИЗМЕНЕНО: принимает model) ---
def predict_single_sample(model, xs_input_t, thetas_input_trained, initial_phi_guess, vqe_config): # Added model argument
    """
    Обрабатывает один сэмпл для ПРЕДСКАЗАНИЯ: использует переданную 'model'.
    Возвращает одно значение предсказания.
    """
    # global worker_model # REMOVED

    if model is None: # Should not happen
        print(f"Error: model argument is None in predict_single_sample.")
        return np.nan

    try:
        # --- VQE Фаза для предсказания ---
        phi_pred, _ = find_ground_state_vqe_pytorch(model, thetas_input_trained, xs_input_t, initial_phi_guess, verbose=False, **vqe_config)

        # --- Предсказание ---
        try:
             pred_value = model.predict_target(phi_pred)
        except AttributeError:
             print(f"Warning: predict_target not fully implemented in QCMLModel заглушка.")
             pred_value = np.random.rand()

        return pred_value

    except Exception as e:
        print(f"Error processing prediction sample: {e}")
        return np.nan


# --- Функция Предсказания (SEQUENTIAL) ---
def predict(model, thetas_input_trained, X_input_features, initial_phi_guess, vqe_config): # Added model, removed num_workers, model_config
    """
    Генерирует предсказания для X_input_features, используя последовательную обработку.
    """
    print(f"\n--- Generating Target Predictions for {len(X_input_features)} samples (SEQUENTIAL) ---")
    pred_start_time = time.time()

    if not isinstance(X_input_features, np.ndarray):
        X_input_features = np.array(X_input_features)
    if X_input_features.ndim == 1:
        X_input_features = X_input_features.reshape(1, -1)

    predictions = []
    for xs_input_t in X_input_features:
        pred_value = predict_single_sample(model, # Pass model
                                           xs_input_t,
                                           thetas_input_trained,
                                           initial_phi_guess,
                                           vqe_config)
        predictions.append(pred_value)

    predictions = np.array(predictions)
    num_failed = np.isnan(predictions).sum()
    if num_failed > 0:
        print(f"\n Warning: {num_failed} predictions failed (returned NaN).")

    pred_end_time = time.time()
    print(f"--- Predictions finished in {pred_end_time - pred_start_time:.2f}s ---")
    return predictions


# --- Основная Функция Обучения и Оценки (с изменениями) ---
def train_and_evaluate(config_path):
    # --- 1. Загрузка Конфигурации и Данных ---
    try:
        print(f"Loading configuration from: {config_path}");
        with open(config_path, 'r') as f: config = json.load(f); print("Configuration loaded.")
        data_path = config['data']['data_path']; target_col = config['data']['target_col']; test_size = config['training']['test_size']; random_state = config['training'].get('random_state', 42); np.random.seed(random_state)
        n_qubits = config['model']['n_qubits']; n_layers_vqe = config['model']['n_layers_vqe']; l_u_layers = config['model']['l_u_layers']
        n_epochs = config['training']['n_epochs']; batch_size = config['training']['batch_size']; vqe_steps = config['training']['vqe_steps']; vqe_lr = config['training']['vqe_lr']; theta_lr = config['training']['theta_lr']; vqe_tol = config['training'].get('vqe_tol', 1e-5)
        # num_workers = config['training'].get('num_workers', os.cpu_count()); # REMOVED
        results_prefix = config['output']['results_prefix']
        # print(f"Using num_workers for multiprocessing: {num_workers}") # REMOVED
    except Exception as e: print(f"Error loading config: {e}"); return

    # --- Загрузка и подготовка данных (без изменений) ---
    try:
        print(f"\nLoading data from: {data_path}"); df = pd.read_parquet(data_path); print(f"Original data loaded. Shape: {df.shape}"); date_col = 'TRADEDATE'
        if date_col not in df.columns: raise ValueError(f"Missing date column: {date_col}")
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]): print(f"Converting '{date_col}' to datetime..."); df[date_col] = pd.to_datetime(df[date_col])
        print(f"Sorting data by '{date_col}'..."); df = df.sort_values(by=date_col).reset_index(drop=True)
        if target_col not in df.columns: raise ValueError(f"Target column '{target_col}' not found.")
        feature_cols = [col for col in df.columns if col not in [target_col, date_col]]
        if not feature_cols: raise ValueError("No feature columns found (excluding target and date).")
        print(f"Input feature columns: {feature_cols}")
        cols_to_check_nan = [target_col] + feature_cols; initial_rows = len(df); df_cleaned = df.dropna(subset=cols_to_check_nan).copy(); removed_rows = initial_rows - len(df_cleaned)
        if removed_rows > 0: print(f"Removed {removed_rows} rows due to NaN values.");
        if df_cleaned.empty: raise ValueError("No data left after NaN removal."); print(f"Data shape after NaN removal: {df_cleaned.shape}")
        X_input = df_cleaned[feature_cols].values; y_target = df_cleaned[target_col].values; dates_cleaned = df_cleaned[date_col].values; k_features = X_input.shape[1]
        print(f"Input Features (K'): {k_features}, Target Variable: '{target_col}'")
    except Exception as e: print(f"Error loading/preparing data: {e}"); return

    n_samples_total = len(df_cleaned);
    if n_samples_total < 2: raise ValueError("Not enough data samples to split.")
    n_test = int(n_samples_total * test_size);
    if test_size > 0 and n_test == 0: n_test = 1
    n_train = n_samples_total - n_test;
    if n_train <= 0 or (test_size > 0 and n_test <= 0):
        raise ValueError(f"Invalid train/test split sizes: train={n_train}, test={n_test} from total={n_samples_total} and test_size={test_size}")
    X_input_train = X_input[:n_train]; X_input_test = X_input[n_train:]; y_target_train = y_target[:n_train]; y_target_test = y_target[n_train:]; dates_train = dates_cleaned[:n_train]; dates_test = dates_cleaned[n_train:]
    print(f"\nData split by date ({1-test_size:.0%} train / {test_size:.0%} test):"); print(f" Train: {n_train} samples (until {pd.to_datetime(dates_train[-1]).date()})");
    if n_test > 0: print(f" Test: {n_test} samples (from {pd.to_datetime(dates_test[0]).date()})");
    else: print(" Test: 0 samples");
    print(f" Train input features shape: {X_input_train.shape}"); print(f" Test input features shape: {X_input_test.shape}")

    # --- 2. Инициализация Модели и Параметров ---
    print("\n--- Initializing Model and Parameters ---")
    try:
        # Создаем основное устройство и модель здесь
        dev = qml.device("lightning.qubit", wires=n_qubits)
        model = QCMLModel(n_qubits=n_qubits, k_input_features=k_features,
                          n_layers_vqe=n_layers_vqe, l_u_layers=l_u_layers, dev=dev)
        phi_shape = model.phi_shape
        thetas_shape = model.thetas_shape
        print("QCMLModel instance created successfully.")
    except Exception as e:
        print(f"Error initializing QCMLModel: {e}")
        return

    initial_phi = np.random.normal(0, 0.01, size=phi_shape)
    initial_thetas = np.random.normal(0, 0.1, size=thetas_shape)
    thetas = np.copy(initial_thetas)
    phi_guess = np.copy(initial_phi)
    opt_theta = qml.AdamOptimizer(stepsize=theta_lr)

    start_epoch = 0
    history = {'epoch': [], 'avg_epoch_cost': [], 'train_mse': []}
    checkpoint_file = f"{results_prefix}_checkpoint.npz"

    print(f"\nChecking for checkpoint file: {checkpoint_file}")
    if os.path.exists(checkpoint_file):
        print(f"Checkpoint file found. Attempting to load state...")
        try:
            data = np.load(checkpoint_file, allow_pickle=True)
            loaded_thetas = data['thetas']
            loaded_epoch = data['epoch'].item()
            loaded_history = data['history'].item()
            if loaded_thetas.shape == thetas_shape:
                thetas = loaded_thetas
                start_epoch = loaded_epoch
                history = loaded_history
                if 'phi_guess' in data:
                    loaded_phi_guess = data['phi_guess']
                    if loaded_phi_guess.shape == phi_shape:
                        phi_guess = loaded_phi_guess
                        print("Loaded phi_guess from checkpoint.")
                    else:
                        print("Warning: Checkpoint 'phi_guess' shape mismatch. Using initial guess.")
                print(f"Successfully loaded state. Resuming training from epoch {start_epoch + 1}/{n_epochs}")
            else:
                print(f"Warning: Checkpoint 'thetas' shape {loaded_thetas.shape} does not match model shape {thetas_shape}. Ignoring checkpoint.")
        except Exception as e:
            print(f"Error loading checkpoint file '{checkpoint_file}': {e}. Starting training from scratch.")
    else:
        print("No checkpoint file found. Starting training from scratch.")

    # --- 3. Цикл Обучения с Батчингом (SEQUENTIAL) ---
    print("\n--- Starting QCML Training (SEQUENTIAL) ---")
    start_time_total = time.time();
    vqe_config = {'vqe_steps': vqe_steps, 'vqe_lr': vqe_lr, 'vqe_tol': vqe_tol}
    print(f"Using VQE config: {vqe_config}")
    train_indices_original_order = np.arange(n_train)

    # model_config_for_worker - REMOVED

    for epoch in range(start_epoch, n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}"); epoch_start_time = time.time(); total_cost_epoch = 0; num_batches = 0
        shuffled_train_indices = np.random.permutation(train_indices_original_order)

        # No pool creation here

        for i in range(0, n_train, batch_size):
            batch_indices = shuffled_train_indices[i:min(i + batch_size, n_train)]
            batch_X_input = X_input_train[batch_indices]
            current_batch_size = len(batch_X_input)
            if current_batch_size == 0: continue
            num_batches += 1; batch_start_time = time.time()
            sys.stdout.write(f" Batch {num_batches}/{int(np.ceil(n_train/batch_size))} (size {current_batch_size}) processing... \r")
            sys.stdout.flush()

            # --- Sequential processing of batch ---
            results = []
            for xs_input_t_sample in batch_X_input:
                res = process_single_sample(model, # Pass model
                                            xs_input_t_sample,
                                            thetas,
                                            phi_guess,
                                            vqe_config)
                results.append(res)

            valid_results = [res for res in results if isinstance(res, tuple) and len(res) == 3 and np.isfinite(res[1])]

            if len(valid_results) < current_batch_size:
                failed_count = current_batch_size - len(valid_results)
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                print(f"\n Warning: {failed_count}/{current_batch_size} samples failed processing in batch {num_batches}")
            if not valid_results:
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                print(f"\n Error: All samples failed in batch {num_batches}. Skipping theta update for this batch.")
                continue

            batch_grads_sum = np.sum([res[0] for res in valid_results], axis=0)
            batch_cost_sum = np.sum([res[1] for res in valid_results])
            avg_final_phi = np.mean([res[2] for res in valid_results], axis=0)
            phi_guess = avg_final_phi

            avg_batch_grad = batch_grads_sum / len(valid_results)
            avg_batch_cost = batch_cost_sum / len(valid_results)

            thetas = opt_theta.apply_grad(avg_batch_grad, [thetas])[0]
            batch_end_time = time.time()
            total_cost_epoch += avg_batch_cost

            sys.stdout.write('\r' + ' ' * 80 + '\r')
            print(f" Batch {num_batches}/{int(np.ceil(n_train/batch_size))}: Avg Cost E_t = {avg_batch_cost:.6f}. Time: {batch_end_time - batch_start_time:.2f}s")

        # --- Оценка MSE на Обучающей Выборке после Эпохи ---
        print("\n Evaluating MSE on Training set for epoch logging (using sequential predict)...")
        # Pass model to predict, remove num_workers and model_config_for_worker
        train_preds = predict(model, thetas, X_input_train, initial_phi, vqe_config)
        valid_train_indices = ~np.isnan(train_preds)
        if np.any(~valid_train_indices):
             print(f" Warning: {np.sum(~valid_train_indices)} NaN predictions found on training set evaluation. Excluding them from MSE.")
        if np.sum(valid_train_indices) > 0:
             train_mse = mean_squared_error(y_target_train[valid_train_indices], train_preds[valid_train_indices])
             print(f" Training MSE (vs {target_col}) after epoch {epoch+1}: {train_mse:.6f}")
        else:
             train_mse = np.nan
             print(f" Training MSE (vs {target_col}) after epoch {epoch+1}: NaN (all predictions failed)")

        avg_cost_epoch = total_cost_epoch / num_batches if num_batches > 0 else 0
        if not history['epoch'] or history['epoch'][-1] != epoch + 1:
            history['epoch'].append(epoch + 1)
            history['avg_epoch_cost'].append(avg_cost_epoch)
            history['train_mse'].append(train_mse)
        else:
            history['avg_epoch_cost'][-1] = avg_cost_epoch
            history['train_mse'][-1] = train_mse

        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} finished. Average Batch Cost (E_t) = {avg_cost_epoch:.6f}. Epoch time: {epoch_end_time - epoch_start_time:.2f}s")

        print(f" Saving checkpoint to {checkpoint_file} after epoch {epoch + 1}...")
        try:
            np.savez_compressed(checkpoint_file,
                                thetas=thetas,
                                phi_guess=phi_guess,
                                epoch=np.array(epoch + 1),
                                history=np.array(history, dtype=object))
            print(" Checkpoint saved successfully.")
        except Exception as e:
            print(f" Error saving checkpoint: {e}")

    # --- 4. Оценка на Тестовой Выборке ---
    end_time_total = time.time();
    print(f"\n--- Training Finished ---");
    total_execution_time = end_time_total - start_time_total
    print(f"Total training execution time (this run): {total_execution_time:.2f}s");
    trained_thetas = thetas

    test_mse = np.nan
    if n_test > 0:
        print("\n--- Evaluating on Test Set (using sequential predict) ---");
        # Pass model to predict, remove num_workers and model_config_for_worker
        test_preds = predict(model, trained_thetas, X_input_test, initial_phi, vqe_config)
        valid_test_indices = ~np.isnan(test_preds)
        if np.any(~valid_test_indices):
            print(f" Warning: {np.sum(~valid_test_indices)} NaN predictions found on test set. Excluding them from MSE.")
        if np.sum(valid_test_indices) > 0:
            test_mse = mean_squared_error(y_target_test[valid_test_indices], test_preds[valid_test_indices])
            print(f"Final Test MSE (vs {target_col}): {test_mse:.6f}")
        else:
            test_mse = np.nan
            print(f"Final Test MSE (vs {target_col}): NaN (all predictions failed)")
    else:
        print("\n--- No Test Set Evaluation (test_size = 0) ---")

    # --- 5. Построение Графиков (без изменений) ---
    print("\n--- Plotting Results ---");
    try:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        valid_cost_epochs = [h for i, h in enumerate(history['avg_epoch_cost']) if np.isfinite(h)]
        valid_cost_epoch_nums = [history['epoch'][i] for i, h in enumerate(history['avg_epoch_cost']) if np.isfinite(h)]
        if valid_cost_epoch_nums:
            axs[0].plot(valid_cost_epoch_nums, valid_cost_epochs, 'bo-', label='Average Batch Cost (E_t)')
        axs[0].set_ylabel("Average Cost (E_t)")
        axs[0].legend()
        axs[0].set_title(f'Training History ({results_prefix})')
        axs[0].grid(True)

        valid_mse_epochs = [h for i, h in enumerate(history['train_mse']) if np.isfinite(h)]
        valid_mse_epoch_nums = [history['epoch'][i] for i, h in enumerate(history['train_mse']) if np.isfinite(h)]
        if valid_mse_epoch_nums:
            axs[1].plot(valid_mse_epoch_nums, valid_mse_epochs, 'ro-', label=f'Train MSE (vs {target_col})')
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Mean Squared Error (MSE)")
        axs[1].legend()

        if n_test > 0 and not np.isnan(test_mse):
            axs[1].axhline(test_mse, color='g', linestyle='--', label=f'Final Test MSE = {test_mse:.6f}')
            axs[1].legend()

        axs[1].grid(True)
        plt.tight_layout()
        plot_filename = f"{results_prefix}_training_history.png"
        plt.savefig(plot_filename)
        print(f"Training history plot saved to {plot_filename}")
        plt.close(fig)

    except Exception as e:
        print(f"Error during plotting: {e}")

    print("\n--- Process Finished ---");
    return trained_thetas, history


# --- Точка входа (без изменений в основной логике, удалено multiprocessing setup) ---
if __name__ == "__main__":
    # try: # REMOVED multiprocessing specific setup
    #     multiprocessing.set_start_method('spawn', force=True)
    #     print("Set multiprocessing start method to 'spawn'.")
    # except RuntimeError:
    #     print(f"Multiprocessing start method already set to '{multiprocessing.get_start_method()}'.")

    enable_profiling = False
    profiler = None
    if enable_profiling:
        import cProfile, pstats
        print("--- Enabling Profiling ---")
        profiler = cProfile.Profile(); profiler.enable()

    config_file_path = 'config.json'
    if not os.path.exists(config_file_path):
        print(f"Ошибка: Файл конфигурации '{config_file_path}' не найден.")
        sys.exit(1)

    train_and_evaluate(config_file_path)

    if enable_profiling and profiler:
        profiler.disable(); stats = pstats.Stats(profiler).sort_stats('cumulative');
        print("\n--- Profiling Results (Top 30 Cumulative Time) ---");
        stats.print_stats(30);
        try:
            with open(config_file_path, 'r') as f:
                config_data = json.load(f)
            prefix = config_data.get('output', {}).get('results_prefix', 'qcml')
        except:
            prefix = 'qcml'
        profile_filename = f"{prefix}_profile_sequential.prof" # Changed filename
        stats.dump_stats(profile_filename);
        print(f"\nProfiling data saved to {profile_filename}");
        print(f"Use 'snakeviz {profile_filename}' to visualize.")
