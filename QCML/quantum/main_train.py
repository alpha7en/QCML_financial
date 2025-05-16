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
import torch

import traceback

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
def find_ground_state_vqe_spsa(model, thetas_input_np, xs_input_np, initial_phi_np,
                               vqe_steps=100,  # Максимальное количество итераций SPSA
                               # Гиперпараметры SPSA, встроенные в функцию
                               spsa_a=0.8,  # Начальный размер шага Adam (SPSA 'a')
                               spsa_c=0.1,  # Размер возмущения (SPSA 'c')
                               spsa_A=0.0,  # Стабилизирующий параметр (часто 0 или небольшой % от max_iter)
                               # PennyLane SPSAOptimizer по умолчанию A=0
                               spsa_alpha=0.602,  # Экспонента для уменьшения 'a'
                               spsa_gamma=0.101,  # Экспонента для уменьшения 'c'
                               vqe_tol=1e-5,  # Толерантность для проверки сходимости
                               verbose=1,  # 0-нет, 1-основной, 2-детальный
                               vqe_lr=None):  # vqe_lr не используется SPSA, но оставим для совместимости сигнатуры
    """
    Находит основное состояние (минимизирует model.energy_vqe_for_spsa) с помощью SPSA.
    """
    if verbose > 0:
        print(f"  Starting VQE optimization with SPSA (max_steps={vqe_steps}, tol={vqe_tol})...")
        print(f"  SPSA H-params: a={spsa_a}, c={spsa_c}, A={spsa_A}, alpha={spsa_alpha}, gamma={spsa_gamma}")

    vqe_total_time_start = time.time()
    iteration_times = []
    cost_fn_eval_count = 0  # Счетчик вызовов основной функции стоимости

    # SPSA работает с параметрами, которые могут быть pennylane.numpy или обычными numpy
    # Важно, чтобы они не имели requires_grad=True с точки зрения PyTorch, если не смешиваем
    phi = np.copy(initial_phi_np)  # Работаем с копией NumPy

    # Оборачиваем функцию стоимости model.energy_vqe_for_spsa
    cost_fn_for_spsa_raw = partial(model.energy_vqe_for_spsa,
                                   thetas_input_np=thetas_input_np,
                                   xs_input_np=xs_input_np)

    # Обертка для подсчета вызовов
    def cost_fn_spsa_counted(phi_params_np):
        nonlocal cost_fn_eval_count
        cost_fn_eval_count += 1
        return cost_fn_for_spsa_raw(phi_params_np)

    # Инициализация оптимизатора SPSA из PennyLane.
    # Передаем гиперпараметры при инициализации.
    opt_spsa = qml.SPSAOptimizer(maxiter=vqe_steps,
                                 a=spsa_a,
                                 c=spsa_c,
                                 A=spsa_A,
                                 alpha=spsa_alpha,
                                 gamma=spsa_gamma)

    energy_prev = cost_fn_spsa_counted(phi)  # Начальная стоимость (1-й вызов)
    if verbose > 1: print(f"    SPSA iter -1: Initial Cost={energy_prev:.6f} (1 eval)")

    steps_done = 0
    converged = False

    for i in range(vqe_steps):
        iter_time_start = time.time()
        steps_done = i + 1

        # opt_spsa.step требует функцию стоимости и текущие параметры.
        # Внутри step будут сделаны 2 вызова cost_fn_spsa_counted.
        phi_new = opt_spsa.step(cost_fn_spsa_counted, phi)

        # Важно: SPSAOptimizer.step может вернуть тот же объект phi, измененный на месте,
        # или новый. Для безопасности присваиваем.
        phi = phi_new

        # Для логирования и проверки сходимости, нам нужно значение энергии ПОСЛЕ шага.
        # Если мы хотим строго 2 вызова на шаг SPSA, мы не должны вызывать cost_fn_spsa_counted здесь снова.
        # Однако, для логирования и проверки сходимости это часто делают.
        # Для чистоты SPSA, можно использовать последнее значение из внутренних вычислений,
        # но qml.SPSAOptimizer не возвращает его из step().
        # Поэтому делаем еще один вызов для логирования.
        energy_current_for_log = cost_fn_spsa_counted(phi)  # +1 вызов для лога/сходимости

        iter_time_end = time.time()
        iteration_times.append(iter_time_end - iter_time_start)

        if verbose > 1 and (i < 3 or i % (vqe_steps // 10 if vqe_steps > 10 else 1) == 0 or i == vqe_steps - 1):
            print(f"    SPSA iter {i:>3}: Cost={energy_current_for_log:.6f}, iter_t={iteration_times[-1]:.4f}s "
                  f"(cost_evals for this step: {opt_spsa.num_evals_per_step})")  # num_evals_per_step должно быть 2

        if np.abs(
                energy_current_for_log - energy_prev) < vqe_tol and i > opt_spsa.A / 2:  # Даем SPSA немного времени на стабилизацию
            if verbose > 0:
                print(f"  SPSA converged at step {steps_done}, Cost={energy_current_for_log:.6f}")
            converged = True
            break
        energy_prev = energy_current_for_log

    final_energy = energy_prev  # Последняя вычисленная энергия
    if not converged and steps_done == vqe_steps and verbose > 0:
        print(f"  SPSA finished after {steps_done} (max_steps). Final Cost={final_energy:.6f}")

    vqe_total_time_end = time.time()
    vqe_total_duration = vqe_total_time_end - vqe_total_time_start

    if verbose > 0:
        print(f"  SPSA VQE finished in {vqe_total_duration:.4f}s. Total cost evals: {cost_fn_eval_count}")
        if steps_done > 0:
            print(
                f"    Avg iter time: {np.mean(iteration_times):.4f}s (включая доп. вызов cost для лога/сходимости)")

    return phi, final_energy  # phi - это NumPy массив


# --- *НОВОЕ*: Вспомогательная функция find_ground_state_vqe с PyTorch оптимизатором ---
def find_ground_state_vqe_pytorch(model, thetas_input_np, xs_input_np, initial_phi_np,
                                  vqe_steps=100, vqe_lr=0.05, vqe_tol=1e-5, verbose=True):
    """
    Находит основное состояние с помощью VQE, используя PyTorch оптимизатор (torch.optim.Adam)
    и PyTorch-интерфейс QNode в модели.
    """
    if verbose: print(f" Starting VQE optimization (PyTorch Adam, steps={vqe_steps}, lr={vqe_lr}, tol={vqe_tol})...")

    # Определяем устройство PyTorch. Для параметров QNode, которые обрабатываются
    # lightning.qubit (CPU-симулятор), параметры обычно должны оставаться на CPU.
    device = torch.device("cpu")

    # 1. Преобразование NumPy массивов в PyTorch тензоры.
    #    phi - это параметры, которые мы будем оптимизировать, поэтому requires_grad=True.
    #    Используем torch.float64 для совместимости с PennyLane, который часто использует float64.
    phi = torch.tensor(initial_phi_np, dtype=torch.float64, device=device, requires_grad=True)

    #    thetas_input и xs_input в контексте этой VQE-оптимизации являются константами
    #    (они не оптимизируются *этой* функцией VQE). Градиенты по ним здесь не нужны.
    thetas_input_torch = torch.tensor(thetas_input_np, dtype=torch.float64, device=device)
    xs_input_torch = torch.tensor(xs_input_np, dtype=torch.float64, device=device)

    # 2. Инициализация оптимизатора PyTorch (Adam).
    #    Передаем в оптимизатор список тензоров, которые нужно оптимизировать (в данном случае, только [phi]).
    opt_phi_torch = torch.optim.Adam([phi], lr=vqe_lr)

    energy_prev = float('inf')  # Используем float('inf') для корректного первого сравнения
    steps_done = 0

    for i in range(vqe_steps):
        steps_done = i + 1

        # 3. Цикл оптимизации PyTorch:
        opt_phi_torch.zero_grad()  # Обнуляем градиенты параметров phi перед новым вычислением.
        # Это стандартный шаг в PyTorch.

        # Вызываем новую функцию energy_vqe_pytorch из вашей модели.
        # Она принимает тензоры PyTorch и возвращает тензор PyTorch (скалярную энергию).
        # Поскольку phi имеет requires_grad=True, PyTorch будет отслеживать операции
        # для вычисления градиента current_energy по phi.
        current_energy = model.energy_vqe_pytorch(phi, thetas_input_torch, xs_input_torch)

        # Вычисляем градиенты (d_energy/d_phi) с помощью механизма autograd PyTorch.
        # Градиенты будут сохранены в атрибуте .grad тензора phi (т.е., phi.grad).
        current_energy.backward()

        # Обновляем параметры phi на основе вычисленных градиентов.
        # Оптимизатор Adam использует phi.grad для выполнения шага обновления.
        opt_phi_torch.step()

        # Получаем скалярное значение энергии из тензора PyTorch для логирования и проверки сходимости.
        energy_val = current_energy.item()

        if verbose and (i % 10 == 0 or i == vqe_steps - 1):
            print(f"  VQE (PyTorch) iter {i:>3}: energy = {energy_val:.6f}")

        if abs(energy_val - energy_prev) < vqe_tol:
            if verbose: print(f" VQE (PyTorch) converged at step {i + 1}, energy = {energy_val:.6f}")
            break
        energy_prev = energy_val

    # Получаем финальную энергию после последнего шага.
    # Используем .detach() для phi, чтобы создать новый тензор, не требующий градиентов,
    # если phi будет дальше использоваться в вычислениях, где градиенты не нужны.
    # Это хорошая практика перед финальным вычислением или возвратом.
    with torch.no_grad():  # Гарантирует, что операции внутри блока не будут отслеживаться для градиентов
        final_energy_val = model.energy_vqe_pytorch(phi, thetas_input_torch, xs_input_torch).item()

    # Возвращаем оптимизированные параметры phi как NumPy массив для совместимости
    # с остальной частью вашего кода, которая ожидает NumPy.
    # .detach() убирает тензор из графа вычислений.
    # .cpu() перемещает тензор на CPU (если он был на GPU, здесь он уже на CPU).
    # .numpy() конвертирует тензор CPU в NumPy массив.
    return phi.detach().cpu().numpy(), final_energy_val

# --- Глобальная переменная для хранения модели в воркере (без изменений) ---
worker_model = None

# --- Функция инициализации воркера (без изменений) ---
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
        # print(f"Worker PID {os.getpid()}: Initialized QCMLModel.") # Можно убрать для чистоты вывода
    except Exception as e:
        print(f"Worker PID {os.getpid()}: Error initializing QCMLModel: {e}")
        worker_model = None # Убедимся, что None, если инициализация не удалась


# --- ФУНКЦИЯ для обработки одного сэмпла ОБУЧЕНИЯ (ИЗМЕНЕНО: использует worker_model) ---
def process_single_sample(xs_input_t, current_thetas, phi_guess, vqe_config):
    """
    Обрабатывает один сэмпл для ОБУЧЕНИЯ: использует ГЛОБАЛЬНУЮ worker_model.
    Возвращает кортеж (gradient_theta, cost_e_t, final_phi).
    """
    global worker_model # Используем модель, созданную в init_worker

    # Проверка, что модель была успешно инициализирована
    if worker_model is None:
        print(f"Error in PID {os.getpid()}: worker_model (train) is not initialized.")
        try: thetas_shape = current_thetas.shape
        except: thetas_shape = (1,)
        try: phi_shape = phi_guess.shape
        except: phi_shape = (1,)
        return np.zeros(thetas_shape), np.inf, np.zeros(phi_shape) # Возврат ошибки

    try:
        # --- VQE Фаза ---
        phi_opt, _ = find_ground_state_vqe_spsa(worker_model, current_thetas, xs_input_t, phi_guess, verbose=False, **vqe_config)

        # --- Theta Gradient Фаза ---
        psi_t = worker_model.get_state_vector(phi_opt)

        # СОЗДАЕМ функцию градиента ЗДЕСЬ, используя worker_model
        # Используем try-except, так как cost_for_theta_gradient может не быть в заглушке
        try:
            local_theta_grad_fn = qml.grad(worker_model.cost_for_theta_gradient, argnum=0)
            # Вычисляем градиент и стоимость
            grad_theta = local_theta_grad_fn(current_thetas, psi_t, xs_input_t)
            current_cost = worker_model.cost_for_theta_gradient(current_thetas, psi_t, xs_input_t)
        except AttributeError:
            print(f"Warning (PID {os.getpid()}): cost_for_theta_gradient not fully implemented in QCMLModel заглушка.")
            grad_theta = np.zeros_like(current_thetas) # Возвращаем нулевой градиент
            current_cost = np.random.rand() # Возвращаем случайную стоимость


        return grad_theta, current_cost, phi_opt

    except Exception as e:
        # Выводим ошибку вместе с PID для идентификации воркера
        print(f"Error processing training sample (PID {os.getpid()}): {e}")
        # Используем формы из worker_model, если доступно, иначе из аргументов
        try: thetas_shape = worker_model.thetas_shape
        except: thetas_shape = current_thetas.shape
        try: phi_shape = worker_model.phi_shape
        except: phi_shape = phi_guess.shape
        return np.zeros(thetas_shape), np.inf, np.zeros(phi_shape) # Возврат ошибки

# --- *НОВОЕ*: ФУНКЦИЯ для обработки одного сэмпла ПРЕДСКАЗАНИЯ ---
def predict_single_sample(xs_input_t, thetas_input_trained, initial_phi_guess, vqe_config):
    """
    Обрабатывает один сэмпл для ПРЕДСКАЗАНИЯ: использует ГЛОБАЛЬНУЮ worker_model.
    Возвращает одно значение предсказания.
    """
    global worker_model # Используем модель, созданную в init_worker

    # Проверка, что модель была успешно инициализирована
    if worker_model is None:
        print(f"Error in PID {os.getpid()}: worker_model (predict) is not initialized.")
        return np.nan # Возвращаем NaN в случае ошибки инициализации

    try:
        # --- VQE Фаза для предсказания ---
        # Используем initial_phi_guess для каждого сэмпла независимо
        phi_pred, _ = find_ground_state_vqe_spsa(worker_model, thetas_input_trained, xs_input_t, initial_phi_guess, verbose=False, **vqe_config)

        # --- Предсказание ---
        # Используем try-except, так как predict_target может не быть в заглушке
        try:
             pred_value = worker_model.predict_target(phi_pred)
        except AttributeError:
             print(f"Warning (PID {os.getpid()}): predict_target not fully implemented in QCMLModel заглушка.")
             pred_value = np.random.rand() # Возвращаем случайное значение

        return pred_value

    except Exception as e:
        print(f"Error processing prediction sample (PID {os.getpid()}): {e}")
        return np.nan # Возвращаем NaN в случае ошибки обработки


# --- *ИЗМЕНЕНО*: Функция Предсказания с MULTIPROCESSING ---
def predict(thetas_input_trained, X_input_features, initial_phi_guess, vqe_config, model_config, num_workers):
    """
    Генерирует предсказания для X_input_features, используя multiprocessing.
    """
    print(f"\n--- Generating Target Predictions for {len(X_input_features)} samples (USING MULTIPROCESSING: {num_workers} workers) ---")
    pred_start_time = time.time()

    if not isinstance(X_input_features, np.ndarray):
        X_input_features = np.array(X_input_features)
    if X_input_features.ndim == 1:
        X_input_features = X_input_features.reshape(1, -1) # Обрабатываем один сэмпл как массив

    mp_context = multiprocessing.get_context('spawn')

    # --- Создание пула с инициализатором ---
    # Передаем функцию init_worker и аргументы для нее (конфигурацию модели)
    with mp_context.Pool(processes=num_workers,
                         initializer=init_worker,
                         initargs=(model_config,)) as pool:
        print(f" Created process pool with {num_workers} workers for prediction.")

        # --- Параллельная обработка предсказаний ---
        predict_func = partial(predict_single_sample,
                               thetas_input_trained=thetas_input_trained,
                               initial_phi_guess=initial_phi_guess,
                               vqe_config=vqe_config)

        # Используем map для распределения работы
        predictions = pool.map(predict_func, X_input_features)

    # --- Сбор результатов ---
    # pool.map возвращает список результатов в том же порядке, что и входные данные
    predictions = np.array(predictions)

    # Проверка на ошибки (NaN)
    num_failed = np.isnan(predictions).sum()
    if num_failed > 0:
        print(f"\n Warning: {num_failed} predictions failed (returned NaN).")
        # Можно решить, как обрабатывать NaN дальше (например, оставить их или заменить средним)

    pred_end_time = time.time()
    print(f"--- Predictions finished in {pred_end_time - pred_start_time:.2f}s ---")
    return predictions


# --- Основная Функция Обучения и Оценки (с изменениями в вызовах predict) ---
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
        print(f"\nLoading data from: {data_path}"); df = pd.read_parquet(data_path); print(f"Original data loaded. Shape: {df.shape}"); date_col = 'TRADEDATE' # Предполагаем имя колонки с датой
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
    if test_size > 0 and n_test == 0: n_test = 1 # Ensure at least one test sample if test_size > 0
    n_train = n_samples_total - n_test;
    if n_train <= 0 or (test_size > 0 and n_test <= 0):
        raise ValueError(f"Invalid train/test split sizes: train={n_train}, test={n_test} from total={n_samples_total} and test_size={test_size}")
    X_input_train = X_input[:n_train]; X_input_test = X_input[n_train:]; y_target_train = y_target[:n_train]; y_target_test = y_target[n_train:]; dates_train = dates_cleaned[:n_train]; dates_test = dates_cleaned[n_train:]
    print(f"\nData split by date ({1-test_size:.0%} train / {test_size:.0%} test):"); print(f" Train: {n_train} samples (until {pd.to_datetime(dates_train[-1]).date()})");
    if n_test > 0: print(f" Test: {n_test} samples (from {pd.to_datetime(dates_test[0]).date()})");
    else: print(" Test: 0 samples");
    print(f" Train input features shape: {X_input_train.shape}"); print(f" Test input features shape: {X_input_test.shape}")

    # --- 2. Инициализация Параметров (Модель создается в воркерах!) ---
    # Создаем временную модель ТОЛЬКО для получения форм параметров
    temp_dev = qml.device("lightning.qubit", wires=n_qubits) # Можно использовать 'default.qubit' для форм
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
    phi_guess = np.copy(initial_phi) # Начальное предположение для phi в VQE
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
            loaded_epoch = data['epoch'].item() # .item() для скаляра numpy
            loaded_history = data['history'].item() # .item() для object array

            # Проверка совместимости формы thetas (полученной из временной модели)
            if loaded_thetas.shape == thetas_shape:
                thetas = loaded_thetas
                start_epoch = loaded_epoch # Эпоха, *после* которой сохранили
                history = loaded_history
                # Восстанавливаем phi_guess, если он сохранялся (опционально)
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


    # --- 3. Цикл Обучения с Батчингом и MULTIPROCESSING ---
    print("\n--- Starting QCML Training (USING MULTIPROCESSING) ---")
    start_time_total = time.time();
    vqe_config = {'vqe_steps': vqe_steps, 'vqe_lr': vqe_lr, 'vqe_tol': vqe_tol}
    print(f"Using VQE config: {vqe_config}")
    train_indices_original_order = np.arange(n_train)
    mp_context = multiprocessing.get_context('spawn') # 'spawn' рекомендуется для кроссплатформенности

    # --- Сборка конфигурации модели для передачи инициализатору ---
    model_config_for_worker = {
        'n_qubits': n_qubits,
        'k_features': k_features, # Убедимся, что k_features определено
        'n_layers_vqe': n_layers_vqe,
        'l_u_layers': l_u_layers
    }

    # --- Цикл по эпохам (start_epoch учитывается) ---
    for epoch in range(start_epoch, n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}"); epoch_start_time = time.time(); total_cost_epoch = 0; num_batches = 0
        # Перемешивание данных для каждой эпохи
        shuffled_train_indices = np.random.permutation(train_indices_original_order)

        # --- Создание пула с инициализатором для ЭПОХИ ---
        # Пул создается и закрывается для каждой эпохи, чтобы воркеры перезапускались
        # Это может помочь с утечками памяти или другими проблемами в долгоживущих воркерах
        with mp_context.Pool(processes=num_workers,
                             initializer=init_worker,
                             initargs=(model_config_for_worker,)) as pool:
            print(f" Created process pool with {num_workers} workers for epoch {epoch+1}.")
            for i in range(0, n_train, batch_size):
                batch_indices = shuffled_train_indices[i:min(i + batch_size, n_train)]
                batch_X_input = X_input_train[batch_indices]
                current_batch_size = len(batch_X_input) # Фактический размер батча
                if current_batch_size == 0: continue # Пропустить пустой батч
                num_batches += 1; batch_start_time = time.time()
                sys.stdout.write(f" Batch {num_batches}/{int(np.ceil(n_train/batch_size))} (size {current_batch_size}) processing... \r")
                sys.stdout.flush()


                # --- Параллельная обработка батча (НЕ передаем model) ---
                process_func = partial(process_single_sample,
                                       # model НЕ передается!
                                       current_thetas=thetas,
                                       phi_guess=phi_guess, # Передаем текущее phi_guess
                                       vqe_config=vqe_config)

                # Используем map для распределения работы
                results = pool.map(process_func, batch_X_input)

                # --- Сбор и усреднение результатов ---
                # Фильтруем результаты, где стоимость не бесконечна (индикатор ошибки)
                valid_results = [res for res in results if isinstance(res, tuple) and len(res) == 3 and np.isfinite(res[1])]

                if len(valid_results) < current_batch_size:
                    failed_count = current_batch_size - len(valid_results)
                    # Очищаем строку и выводим предупреждение
                    sys.stdout.write('\r' + ' ' * 80 + '\r')
                    print(f"\n Warning: {failed_count}/{current_batch_size} samples failed processing in batch {num_batches}")
                if not valid_results:
                    # Очищаем строку и выводим ошибку
                    sys.stdout.write('\r' + ' ' * 80 + '\r')
                    print(f"\n Error: All samples failed in batch {num_batches}. Skipping theta update for this batch.")
                    continue # Переходим к следующему батчу

                # Суммируем градиенты и стоимости только по валидным результатам
                batch_grads_sum = np.sum([res[0] for res in valid_results], axis=0)
                batch_cost_sum = np.sum([res[1] for res in valid_results])
                # Усредняем phi из валидных результатов для следующего phi_guess
                avg_final_phi = np.mean([res[2] for res in valid_results], axis=0)
                phi_guess = avg_final_phi # Обновляем phi_guess на основе среднего по батчу

                # Усредняем градиент и стоимость по количеству *успешно обработанных* сэмплов
                avg_batch_grad = batch_grads_sum / len(valid_results)
                avg_batch_cost = batch_cost_sum / len(valid_results)

                # Обновление параметров theta
                thetas = opt_theta.apply_grad(avg_batch_grad, [thetas])[0] # opt_theta возвращает кортеж
                batch_end_time = time.time()
                total_cost_epoch += avg_batch_cost # Используем среднюю стоимость батча

                # Выводим информацию о батче, очищая предыдущую строку
                sys.stdout.write('\r' + ' ' * 80 + '\r') # Очистка строки
                print(f" Batch {num_batches}/{int(np.ceil(n_train/batch_size))}: Avg Cost E_t = {avg_batch_cost:.6f}. Time: {batch_end_time - batch_start_time:.2f}s")

        # --- Оценка MSE на Обучающей Выборке после Эпохи ---
        # *ИЗМЕНЕНО*: Используем новую функцию predict с мультипроцессингом
        print("\n Evaluating MSE on Training set for epoch logging (using parallel predict)...")
        # Используем initial_phi как отправную точку для VQE в предсказаниях этой эпохи
        train_preds = predict(thetas, X_input_train, initial_phi, vqe_config, model_config_for_worker, num_workers)
        # Обработка возможных NaN в предсказаниях перед вычислением MSE
        valid_train_indices = ~np.isnan(train_preds)
        if np.any(~valid_train_indices):
             print(f" Warning: {np.sum(~valid_train_indices)} NaN predictions found on training set evaluation. Excluding them from MSE.")
        if np.sum(valid_train_indices) > 0:
             train_mse = mean_squared_error(y_target_train[valid_train_indices], train_preds[valid_train_indices])
             print(f" Training MSE (vs {target_col}) after epoch {epoch+1}: {train_mse:.6f}")
        else:
             train_mse = np.nan # Не удалось посчитать MSE
             print(f" Training MSE (vs {target_col}) after epoch {epoch+1}: NaN (all predictions failed)")


        # --- Логирование истории (без изменений) ---
        avg_cost_epoch = total_cost_epoch / num_batches if num_batches > 0 else 0
        if not history['epoch'] or history['epoch'][-1] != epoch + 1: # Добавляем, если новая эпоха
            history['epoch'].append(epoch + 1)
            history['avg_epoch_cost'].append(avg_cost_epoch)
            history['train_mse'].append(train_mse)
        else: # Обновляем, если перезапускаем эпоху из чекпоинта
            history['avg_epoch_cost'][-1] = avg_cost_epoch
            history['train_mse'][-1] = train_mse

        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} finished. Average Batch Cost (E_t) = {avg_cost_epoch:.6f}. Epoch time: {epoch_end_time - epoch_start_time:.2f}s")

        # --- Сохранение чекпоинта (добавлено сохранение phi_guess) ---
        print(f" Saving checkpoint to {checkpoint_file} after epoch {epoch + 1}...")
        try:
            np.savez_compressed(checkpoint_file,
                                thetas=thetas,
                                phi_guess=phi_guess, # Сохраняем последнее phi_guess
                                epoch=np.array(epoch + 1), # Сохраняем номер *завершенной* эпохи
                                history=np.array(history, dtype=object))
            print(" Checkpoint saved successfully.")
        except Exception as e:
            print(f" Error saving checkpoint: {e}")


    # --- 4. Оценка на Тестовой Выборке ---
    end_time_total = time.time();
    print(f"\n--- Training Finished ---");
    total_execution_time = end_time_total - start_time_total
    print(f"Total training execution time (this run): {total_execution_time:.2f}s");
    trained_thetas = thetas # Финальные обученные параметры

    test_mse = np.nan
    if n_test > 0:
        print("\n--- Evaluating on Test Set (using parallel predict) ---");
        # *ИЗМЕНЕНО*: Используем новую функцию predict с мультипроцессингом
        test_preds = predict(trained_thetas, X_input_test, initial_phi, vqe_config, model_config_for_worker, num_workers)
        # Обработка NaN
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
        # График стоимости
        valid_cost_epochs = [h for i, h in enumerate(history['avg_epoch_cost']) if np.isfinite(h)]
        valid_cost_epoch_nums = [history['epoch'][i] for i, h in enumerate(history['avg_epoch_cost']) if np.isfinite(h)]
        if valid_cost_epoch_nums:
            axs[0].plot(valid_cost_epoch_nums, valid_cost_epochs, 'bo-', label='Average Batch Cost (E_t)')
        axs[0].set_ylabel("Average Cost (E_t)")
        axs[0].legend()
        axs[0].set_title(f'Training History ({results_prefix})')
        axs[0].grid(True)

        # График MSE
        valid_mse_epochs = [h for i, h in enumerate(history['train_mse']) if np.isfinite(h)]
        valid_mse_epoch_nums = [history['epoch'][i] for i, h in enumerate(history['train_mse']) if np.isfinite(h)]
        if valid_mse_epoch_nums:
            axs[1].plot(valid_mse_epoch_nums, valid_mse_epochs, 'ro-', label=f'Train MSE (vs {target_col})')
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Mean Squared Error (MSE)")
        axs[1].legend() # Легенда для Train MSE

        # Добавляем линию Test MSE, если она есть
        if n_test > 0 and not np.isnan(test_mse):
            axs[1].axhline(test_mse, color='g', linestyle='--', label=f'Final Test MSE = {test_mse:.6f}')
            axs[1].legend() # Обновляем легенду, чтобы включить Test MSE

        axs[1].grid(True)
        plt.tight_layout()
        plot_filename = f"{results_prefix}_training_history.png"
        plt.savefig(plot_filename)
        print(f"Training history plot saved to {plot_filename}")
        # plt.show() # Опционально показать график
        plt.close(fig) # Закрыть фигуру после сохранения

    except Exception as e:
        print(f"Error during plotting: {e}")

    print("\n--- Process Finished ---");
    return trained_thetas, history


# --- Точка входа (без изменений) ---
if __name__ == "__main__":
    # Устанавливаем метод старта 'spawn' глобально для multiprocessing, если еще не установлен
    # Это важно для избежания проблем с некоторыми бэкендами (особенно GPU) и для консистентности
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        # Может возникнуть, если метод уже установлен
        print(f"Multiprocessing start method already set to '{multiprocessing.get_start_method()}'.")


    enable_profiling = False # По умолчанию выключено
    profiler = None
    if enable_profiling:
        import cProfile, pstats
        print("--- Enabling Profiling ---")
        profiler = cProfile.Profile(); profiler.enable()

    config_file_path = 'config.json'
    if not os.path.exists(config_file_path):
        print(f"Ошибка: Файл конфигурации '{config_file_path}' не найден.")
        sys.exit(1)

    # Запуск основного процесса
    train_and_evaluate(config_file_path)

    # Обработка профилирования, если было включено
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
        profile_filename = f"{prefix}_profile_mp_predict.prof" # Новое имя файла профиля
        stats.dump_stats(profile_filename);
        print(f"\nProfiling data saved to {profile_filename}");
        print(f"Use 'snakeviz {profile_filename}' to visualize.")