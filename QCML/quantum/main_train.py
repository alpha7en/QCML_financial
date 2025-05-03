import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import json
import sys
import os
import multiprocessing
from functools import partial

# --- Проверка и заглушка QCMLModel (остается без изменений) ---
try:
    from qcml_model import QCMLModel
except ImportError:
    print("Ошибка: Не удалось импортировать QCMLModel из qcml_model.")
    print("Убедитесь, что файл qcml_model.py находится в той же директории или доступен в PYTHONPATH.")
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
        def predict_target(self, phi_pred): return np.random.rand()
    # sys.exit(1)


# --- Вспомогательные функции find_ground_state_vqe и predict (без изменений) ---
def find_ground_state_vqe(model, thetas_input, xs_input, initial_phi,
                          vqe_steps=100, vqe_lr=0.05, vqe_tol=1e-5, verbose=True):
    # ... (код без изменений) ...
    if verbose: print(f"    Starting VQE optimization (steps={vqe_steps}, lr={vqe_lr}, tol={vqe_tol})...")
    phi = np.copy(initial_phi); opt_phi = qml.AdamOptimizer(stepsize=vqe_lr); energy_prev = np.inf; steps_done = 0
    for i in range(vqe_steps):
        steps_done=i+1; phi, energy = opt_phi.step_and_cost(lambda v: model.energy_vqe(v, thetas_input, xs_input), phi)
        if verbose and (i % 1 == 0 or i == vqe_steps - 1): print(f"      VQE iter {i:>3}: energy = {energy:.6f}")
        if np.abs(energy - energy_prev) < vqe_tol:
            if verbose: print(f"      VQE converged at step {i+1}, energy = {energy:.6f}"); break
        energy_prev = energy
    final_energy = model.energy_vqe(phi, thetas_input, xs_input)
    return phi, final_energy

def predict(model, thetas_input_trained, X_input_features, initial_phi_guess, vqe_config):
    # ... (код без изменений) ...
    print(f"\n--- Generating Target Predictions for {len(X_input_features)} samples ---")
    predictions = []; phi_guess = np.copy(initial_phi_guess); pred_start_time = time.time()
    if not isinstance(X_input_features, np.ndarray):
        X_input_features = np.array(X_input_features)
    if X_input_features.ndim == 1:
         X_input_features = X_input_features.reshape(1, -1)
    for i, xs_input_t in enumerate(X_input_features):
        if (i+1) % 1 == 0 or i == len(X_input_features) - 1: print(f"  Predicting sample {i+1}/{len(X_input_features)}...")
        phi_pred, _ = find_ground_state_vqe(model, thetas_input_trained, xs_input_t, phi_guess, verbose=False, **vqe_config)
        phi_guess = phi_pred; pred_value = model.predict_target(phi_pred); predictions.append(pred_value)
    pred_end_time = time.time(); print(f"--- Predictions finished in {pred_end_time - pred_start_time:.2f}s ---")
    return np.array(predictions)


# --- *НОВОЕ*: Глобальная переменная для хранения модели в воркере ---
worker_model = None

# --- *НОВОЕ*: Функция инициализации воркера ---
def init_worker(model_config):
    """Инициализирует модель и устройство в каждом процессе воркера."""
    global worker_model
    n_qubits = model_config['n_qubits']
    k_features = model_config['k_features']
    n_layers_vqe = model_config['n_layers_vqe']
    l_u_layers = model_config['l_u_layers']

    try:
        # Создаем устройство и модель ЗДЕСЬ, в дочернем процессе
        dev = qml.device("lightning.qubit", wires=n_qubits)
        worker_model = QCMLModel(n_qubits=n_qubits, k_input_features=k_features,
                                 n_layers_vqe=n_layers_vqe, l_u_layers=l_u_layers, dev=dev)
        print(f"Worker PID {os.getpid()}: Initialized QCMLModel.")
    except Exception as e:
        print(f"Worker PID {os.getpid()}: Error initializing QCMLModel: {e}")
        worker_model = None # Убедимся, что None, если инициализация не удалась

# --- ФУНКЦИЯ для обработки одного сэмпла (ИЗМЕНЕНО: использует worker_model) ---
def process_single_sample(xs_input_t, current_thetas, phi_guess, vqe_config):
    """
    Обрабатывает один сэмпл: использует ГЛОБАЛЬНУЮ worker_model.
    Возвращает кортеж (gradient_theta, cost_e_t, final_phi).
    """
    global worker_model # Используем модель, созданную в init_worker

    # Проверка, что модель была успешно инициализирована
    if worker_model is None:
        print(f"Error in PID {os.getpid()}: worker_model is not initialized.")
        # Возвращаем ошибку с корректными (хотя бы по типу) значениями
        # Попытка получить формы из аргументов
        try: thetas_shape = current_thetas.shape
        except: thetas_shape = (1,)
        try: phi_shape = phi_guess.shape
        except: phi_shape = (1,)
        return np.zeros(thetas_shape), np.inf, np.zeros(phi_shape)

    try:
        # --- VQE Фаза ---
        phi_opt, _ = find_ground_state_vqe(worker_model, current_thetas, xs_input_t, phi_guess, verbose=False, **vqe_config)

        # --- Theta Gradient Фаза ---
        psi_t = worker_model.get_state_vector(phi_opt)

        # СОЗДАЕМ функцию градиента ЗДЕСЬ, используя worker_model
        local_theta_grad_fn = qml.grad(worker_model.cost_for_theta_gradient, argnum=0)

        # Вычисляем градиент и стоимость
        grad_theta = local_theta_grad_fn(current_thetas, psi_t, xs_input_t)
        current_cost = worker_model.cost_for_theta_gradient(current_thetas, psi_t, xs_input_t)

        return grad_theta, current_cost, phi_opt
    except Exception as e:
        # Выводим ошибку вместе с PID для идентификации воркера
        print(f"Error processing sample (PID {os.getpid()}): {e}")
        # Используем формы из worker_model, если доступно, иначе из аргументов
        try: thetas_shape = worker_model.thetas_shape
        except: thetas_shape = current_thetas.shape
        try: phi_shape = worker_model.phi_shape
        except: phi_shape = phi_guess.shape
        return np.zeros(thetas_shape), np.inf, np.zeros(phi_shape)


# --- Основная Функция Обучения и Оценки ---
def train_and_evaluate(config_path):
    # --- 1. Загрузка Конфигурации и Данных (без изменений) ---
    try:
        print(f"Loading configuration from: {config_path}");
        with open(config_path, 'r') as f: config = json.load(f); print("Configuration loaded.")
        data_path = config['data']['data_path']; target_col = config['data']['target_col']; test_size = config['training']['test_size']; random_state = config['training'].get('random_state', 42); np.random.seed(random_state)
        n_qubits = config['model']['n_qubits']; n_layers_vqe = config['model']['n_layers_vqe']; l_u_layers = config['model']['l_u_layers']
        n_epochs = config['training']['n_epochs']; batch_size = config['training']['batch_size']; vqe_steps = config['training']['vqe_steps']; vqe_lr = config['training']['vqe_lr']; theta_lr = config['training']['theta_lr']; vqe_tol = config['training'].get('vqe_tol', 1e-5)
        num_workers = config['training'].get('num_workers', os.cpu_count()); results_prefix = config['output']['results_prefix']
        print(f"Using num_workers for multiprocessing: {num_workers}")
    except Exception as e: print(f"Error loading config: {e}"); return

    # --- Загрузка и подготовка данных (без изменений) ---
    try:
        print(f"\nLoading data from: {data_path}"); df = pd.read_parquet(data_path)[:64]; print(f"Original data loaded. Shape: {df.shape}"); date_col = 'TRADEDATE'
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
        X_input = df_cleaned[feature_cols].values; y_target = df_cleaned[target_col].values; dates_cleaned = df_cleaned[date_col].values; k_features = X_input.shape[1] # k_features определяется здесь
        print(f"Input Features (K'): {k_features}, Target Variable: '{target_col}'")
    except Exception as e: print(f"Error loading/preparing data: {e}"); return

    # Разделение данных на train/test (без изменений)
    n_samples_total = len(df_cleaned);
    if n_samples_total < 2: raise ValueError("Not enough data samples to split.")
    n_test = int(n_samples_total * test_size);
    if test_size > 0 and n_test == 0: n_test = 1
    n_train = n_samples_total - n_test;
    if n_train <= 0 or (test_size > 0 and n_test <= 0):
         raise ValueError(f"Invalid train/test split sizes: train={n_train}, test={n_test} from total={n_samples_total} and test_size={test_size}")
    X_input_train = X_input[:n_train]; X_input_test  = X_input[n_train:]; y_target_train = y_target[:n_train]; y_target_test  = y_target[n_train:]; dates_train = dates_cleaned[:n_train]; dates_test = dates_cleaned[n_train:]
    print(f"\nData split by date ({1-test_size:.0%} train / {test_size:.0%} test):"); print(f"  Train: {n_train} samples (until {pd.to_datetime(dates_train[-1]).date()})");
    if n_test > 0: print(f"  Test:  {n_test} samples (from {pd.to_datetime(dates_test[0]).date()})");
    else: print("  Test:  0 samples");
    print(f"  Train input features shape: {X_input_train.shape}"); print(f"  Test input features shape: {X_input_test.shape}")

    # --- 2. Инициализация Параметров (Модель создается в воркерах!) ---
    # Создаем временную модель ТОЛЬКО для получения форм параметров
    # и для использования в функциях вне пула (predict, начальная оценка)
    temp_dev = qml.device("lightning.qubit", wires=n_qubits)
    try:
        temp_model = QCMLModel(n_qubits=n_qubits, k_input_features=k_features, n_layers_vqe=n_layers_vqe, l_u_layers=l_u_layers, dev=temp_dev)
        phi_shape = temp_model.phi_shape
        thetas_shape = temp_model.thetas_shape
        del temp_model # Удаляем временную модель, она больше не нужна здесь
        del temp_dev
        print("Temporary model created for shape info and deleted.")
    except Exception as e:
        print(f"Error initializing temporary QCMLModel for shape info: {e}")
        return # Не можем продолжить без форм параметров

    initial_phi = np.random.normal(0, 0.01, size=phi_shape)
    initial_thetas = np.random.normal(0, 0.1, size=thetas_shape)
    thetas = np.copy(initial_thetas)
    phi_guess = np.copy(initial_phi)
    opt_theta = qml.AdamOptimizer(stepsize=theta_lr)

    # --- Инициализация чекпоинтов (без изменений) ---
    start_epoch = 0
    history = {'epoch': [], 'avg_epoch_cost': [], 'train_mse': []}
    checkpoint_file = f"{results_prefix}_checkpoint.npz"

    # --- Загрузка чекпоинта (без изменений) ---
    print(f"\nChecking for checkpoint file: {checkpoint_file}")
    if os.path.exists(checkpoint_file):
        print(f"Checkpoint file found. Attempting to load state...")
        try:
            data = np.load(checkpoint_file, allow_pickle=True)
            loaded_thetas = data['thetas']
            loaded_epoch = data['epoch'].item()
            loaded_history = data['history'].item()

            # Проверка совместимости формы thetas (полученной из временной модели)
            if loaded_thetas.shape == thetas_shape:
                thetas = loaded_thetas
                start_epoch = loaded_epoch
                history = loaded_history
                print(f"Successfully loaded state. Resuming training from epoch {start_epoch + 1}/{n_epochs}")
            else:
                 print(f"Warning: Checkpoint 'thetas' shape {loaded_thetas.shape} does not match model shape {thetas_shape}. Ignoring checkpoint.")
        except Exception as e:
            print(f"Error loading checkpoint file '{checkpoint_file}': {e}. Starting training from scratch.")
    else:
        print("No checkpoint file found. Starting training from scratch.")

    # --- 3. Цикл Обучения с Батчингом и MULTIPROCESSING ---
    print("\n--- Starting QCML Training (USING MULTIPROCESSING) ---")
    start_time_total = time.time();
    vqe_config = {'vqe_steps': vqe_steps, 'vqe_lr': vqe_lr, 'vqe_tol': vqe_tol}
    print(f"Using VQE config: {vqe_config}")
    train_indices_original_order = np.arange(n_train)
    mp_context = multiprocessing.get_context('spawn')

    # --- *НОВОЕ*: Сборка конфигурации модели для передачи инициализатору ---
    model_config_for_worker = {
        'n_qubits': n_qubits,
        'k_features': k_features, # Убедимся, что k_features определено
        'n_layers_vqe': n_layers_vqe,
        'l_u_layers': l_u_layers
    }

    # --- Цикл по эпохам (start_epoch учитывается) ---
    for epoch in range(start_epoch, n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}"); epoch_start_time = time.time(); total_cost_epoch = 0; num_batches = 0
        shuffled_train_indices = np.random.permutation(train_indices_original_order)

        # --- *ИЗМЕНЕНО*: Создание пула с инициализатором ---
        # Передаем функцию init_worker и аргументы для нее (конфигурацию модели)
        with mp_context.Pool(processes=num_workers,
                             initializer=init_worker,
                             initargs=(model_config_for_worker,)) as pool:
            print(f"  Created process pool with {num_workers} workers for epoch {epoch+1}.")
            for i in range(0, n_train, batch_size):
                batch_indices = shuffled_train_indices[i:min(i + batch_size, n_train)]
                batch_X_input = X_input_train[batch_indices]
                if len(batch_X_input) == 0: continue
                num_batches += 1; batch_start_time = time.time()
                print(f"  Batch {num_batches}/{int(np.ceil(n_train/batch_size))} processing...", end='\r', flush=True)

                # --- *ИЗМЕНЕНО*: Параллельная обработка батча (НЕ передаем model) ---
                process_func = partial(process_single_sample,
                                       # model НЕ передается!
                                       current_thetas=thetas,
                                       phi_guess=phi_guess,
                                       vqe_config=vqe_config)

                results = pool.map(process_func, batch_X_input)

                # --- Сбор и усреднение результатов (без изменений) ---
                valid_results = [res for res in results if isinstance(res, tuple) and np.isfinite(res[1])]
                if len(valid_results) < len(results):
                    sys.stdout.write('\r' + ' ' * 80 + '\r')
                    print(f"\n  Warning: {len(results) - len(valid_results)} samples failed in batch {num_batches}")
                if not valid_results:
                    sys.stdout.write('\r' + ' ' * 80 + '\r')
                    print(f"\n  Error: All samples failed in batch {num_batches}. Skipping update."); continue

                batch_grads_sum = np.sum([res[0] for res in valid_results], axis=0)
                batch_cost_sum = np.sum([res[1] for res in valid_results])
                avg_final_phi = np.mean([res[2] for res in valid_results], axis=0)
                phi_guess = avg_final_phi

                avg_batch_grad = batch_grads_sum / len(valid_results)
                avg_batch_cost = batch_cost_sum / len(valid_results)
                thetas = opt_theta.apply_grad(avg_batch_grad, [thetas])[0]
                batch_end_time = time.time()

                if num_batches % 1 == 0 or i + batch_size >= n_train:
                    sys.stdout.write('\r' + ' ' * 80 + '\r')
                    print(f"  Batch {num_batches}/{int(np.ceil(n_train/batch_size))}: Avg Cost E_t = {avg_batch_cost:.6f}. Time: {batch_end_time - batch_start_time:.2f}s")
                total_cost_epoch += avg_batch_cost

        # --- Оценка MSE на Обучающей Выборке после Эпохи ---
        # Используем временную модель, созданную заново для этой цели
        print("\n  Evaluating MSE on Training set for epoch logging...");
        eval_dev = qml.device("lightning.qubit", wires=n_qubits)
        eval_model = QCMLModel(n_qubits=n_qubits, k_input_features=k_features,
                               n_layers_vqe=n_layers_vqe, l_u_layers=l_u_layers, dev=eval_dev)
        train_preds = predict(eval_model, thetas, X_input_train, initial_phi, vqe_config);
        train_mse = mean_squared_error(y_target_train, train_preds);
        print(f"  Training MSE (vs {target_col}) after epoch {epoch+1}: {train_mse:.6f}")
        del eval_model # Удаляем модель после оценки
        del eval_dev

        # --- Логирование истории (без изменений) ---
        avg_cost_epoch = total_cost_epoch / num_batches if num_batches > 0 else 0
        if not (history['epoch'] and history['epoch'][-1] == epoch + 1):
             history['epoch'].append(epoch + 1)
             history['avg_epoch_cost'].append(avg_cost_epoch)
             history['train_mse'].append(train_mse)
        else:
             history['avg_epoch_cost'][-1] = avg_cost_epoch
             history['train_mse'][-1] = train_mse
        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} finished. Average Batch Cost (E_t) = {avg_cost_epoch:.6f}. Epoch time: {epoch_end_time - epoch_start_time:.2f}s")

        # --- Сохранение чекпоинта (без изменений) ---
        print(f"  Saving checkpoint to {checkpoint_file} after epoch {epoch + 1}...")
        try:
            np.savez_compressed(checkpoint_file,
                                thetas=thetas,
                                epoch=np.array(epoch + 1),
                                history=np.array(history, dtype=object))
            print("  Checkpoint saved successfully.")
        except Exception as e:
            print(f"  Error saving checkpoint: {e}")


    # --- 4. Оценка на Тестовой Выборке ---
    end_time_total = time.time();
    print(f"\n--- Training Finished ---");
    total_execution_time = end_time_total - start_time_total
    print(f"Total training execution time (this run): {total_execution_time:.2f}s");
    trained_thetas = thetas

    test_mse = np.nan
    if n_test > 0:
        print("\n--- Evaluating on Test Set ---");
        # Используем временную модель, созданную заново
        test_eval_dev = qml.device("lightning.qubit", wires=n_qubits)
        test_eval_model = QCMLModel(n_qubits=n_qubits, k_input_features=k_features,
                                    n_layers_vqe=n_layers_vqe, l_u_layers=l_u_layers, dev=test_eval_dev)
        test_preds = predict(test_eval_model, trained_thetas, X_input_test, initial_phi, vqe_config);
        test_mse = mean_squared_error(y_target_test, test_preds);
        print(f"Final Test MSE (vs {target_col}): {test_mse:.6f}")
        del test_eval_model
        del test_eval_dev
    else:
        print("\n--- No Test Set Evaluation (test_size = 0) ---")


    # --- 5. Построение Графиков (без изменений) ---
    print("\n--- Plotting Results ---");
    try:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        if history['epoch']:
             axs[0].plot(history['epoch'], history['avg_epoch_cost'], 'bo-', label='Average Batch Cost (E_t)')
        axs[0].set_ylabel("Average Cost (E_t)")
        axs[0].legend()
        axs[0].set_title(f'Training History ({results_prefix})')
        axs[0].grid(True)
        if history['epoch']:
            axs[1].plot(history['epoch'], history['train_mse'], 'ro-', label=f'Train MSE (vs {target_col})')
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Mean Squared Error (MSE)")
        axs[1].legend()
        axs[1].grid(True)
        if n_test > 0 and not np.isnan(test_mse):
             axs[1].axhline(test_mse, color='g', linestyle='--', label=f'Final Test MSE = {test_mse:.6f}')
             axs[1].legend()
        plt.tight_layout()
        plot_filename = f"{results_prefix}_training_history.png"
        plt.savefig(plot_filename)
        print(f"Training history plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error during plotting: {e}")

    print("\n--- Process Finished ---");
    return trained_thetas, history


# --- Точка входа (без изменений) ---
if __name__ == "__main__":
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
        # Попытка получить префикс из конфига для имени файла профиля
        try:
            with open(config_file_path, 'r') as f:
                config_data = json.load(f)
            prefix = config_data.get('output', {}).get('results_prefix', 'qcml')
        except:
            prefix = 'qcml' # Значение по умолчанию, если конфиг не прочитался
        profile_filename = f"{prefix}_profile_mp_chkpt_workerinit.prof"
        stats.dump_stats(profile_filename);
        print(f"\nProfiling data saved to {profile_filename}");
        print(f"Use 'snakeviz {profile_filename}' to visualize.")