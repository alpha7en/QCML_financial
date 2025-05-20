# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pennylane as qml
import time
from collections import deque
import multiprocessing
from functools import partial
import os
import matplotlib.pyplot as plt
import json

import config
from data_loader import load_and_split_data
from quantum_model import QNN

# --- Глобальная переменная для модели и loss_fn в воркере (без изменений) ---
worker_qnn_model = None
worker_loss_fn = None


# --- init_worker_cpu (без изменений) ---
def init_worker_cpu(model_params_for_worker, loss_fn_for_worker_str):
    global worker_qnn_model, worker_loss_fn
    try:
        worker_qnn_model = QNN(
            n_features_for_embedding=model_params_for_worker['features_for_embedding_input'],
            n_qubits_for_ansatz=model_params_for_worker['actual_n_qubits_for_model'],
            n_layers=model_params_for_worker['n_layers'],
            embedding_type=model_params_for_worker['embedding_type'],
            rotation_gate=model_params_for_worker['rotation_gate'],
            device_name=model_params_for_worker['device_name']  # Воркеры CPU
        )
        worker_qnn_model.eval()
    except Exception as e:
        print(f"Worker PID {os.getpid()}: Error initializing QNN model: {e}")
        worker_qnn_model = None
    if loss_fn_for_worker_str == "MSELoss":
        worker_loss_fn = nn.MSELoss()
    else:  # По умолчанию MSELoss
        # print(f"Worker PID {os.getpid()}: Unknown loss function string '{loss_fn_for_worker_str}'. Defaulting to MSELoss.")
        worker_loss_fn = nn.MSELoss()


# --- process_sample_for_grads (без изменений, но с небольшими уточнениями для надежности) ---
def process_sample_for_grads(data_sample, model_state_dict_numpy):
    global worker_qnn_model, worker_loss_fn
    # Определяем ожидаемое количество градиентов из model_state_dict_numpy
    # Это важно для создания корректных dummy_grads
    num_trainable_params_in_state_dict = 0
    if model_state_dict_numpy:
        # Подсчет параметров, которые обычно обучаемы (например, 'q_weights')
        # Это упрощение, в реальном QNN могут быть и другие обучаемые параметры
        if 'q_weights' in model_state_dict_numpy:  # Предполагаем, что q_weights главный обучаемый параметр
            num_trainable_params_in_state_dict = 1
        # Если у вас есть другие обучаемые параметры в state_dict, которые передаются,
        # их нужно учесть здесь или передать их имена/количество.
        # Для простоты, если q_weights нет, но state_dict не пуст, считаем 1.
        elif model_state_dict_numpy:
            num_trainable_params_in_state_dict = len(model_state_dict_numpy)

    if worker_qnn_model is None or worker_loss_fn is None:
        dummy_grads = [np.array([0.0]) for _ in
                       range(num_trainable_params_in_state_dict if num_trainable_params_in_state_dict > 0 else 1)]
        return dummy_grads, float('inf')

    x_sample_np, y_sample_np = data_sample

    # Используем y до его преобразования в тензор и squeeze
    # weight = 1.0
    # if not (-0.2 <= y_sample_np <= 0.0):  # Если значение вне доминирующего диапазона
    #     weight =15.0
    try:
        model_dtype = next(worker_qnn_model.parameters()).dtype
        state_dict_torch = {name: torch.tensor(param_np, dtype=model_dtype)
                            for name, param_np in model_state_dict_numpy.items()}
        worker_qnn_model.load_state_dict(state_dict_torch,
                                         strict=False)  # strict=False если не все ключи state_dict есть в модели
        worker_qnn_model.zero_grad()

        x_sample_tensor = torch.tensor(x_sample_np, dtype=model_dtype).unsqueeze(0)
        y_sample_tensor = torch.tensor(y_sample_np, dtype=model_dtype)
        if y_sample_tensor.ndim == 0:
            y_sample_tensor = y_sample_tensor.unsqueeze(0)
        # elif y_sample_tensor.ndim == 1 and y_sample_tensor.shape[0] != 1:
        #      y_sample_tensor = y_sample_tensor[0].unsqueeze(0) # Берем первый для сэмпла, если это батч пришел

    except Exception as e_prepare:
        # print(f"Worker PID {os.getpid()}: Error preparing data/model: {e_prepare}")
        dummy_grads = [np.zeros_like(param_np) for param_np in
                       model_state_dict_numpy.values()] if model_state_dict_numpy else [np.array([0.0])]
        return dummy_grads, float('inf')

    try:
        prediction_raw = worker_qnn_model(x_sample_tensor)
        if prediction_raw.dtype != model_dtype:

            #print(f" INFO - Casting prediction dtype from {prediction_raw.dtype} to {model_dtype} to ensure consistency.")
            prediction = prediction_raw.to(model_dtype)  # Явное приведение типа
        else:
            prediction = prediction_raw  # Тип уже совпадает

        # Коррекция форм для loss (оставляем ваш вариант, он должен работать для сэмпла)
        if y_sample_tensor.ndim == 2 and y_sample_tensor.shape[1] == 1 and prediction.ndim == 1:
            y_squeezed = y_sample_tensor.squeeze(1)
        elif y_sample_tensor.ndim == 1 and prediction.ndim == 1 and y_sample_tensor.shape[0] == prediction.shape[0]:
            y_squeezed = y_sample_tensor
        elif prediction.ndim > 1 and prediction.shape[0] == 1:  # prediction был (1,N), стал (N,)
            y_squeezed = y_sample_tensor  # y_sample_tensor (1,) или (N,)
            prediction = prediction.squeeze(0)
        else:
            y_squeezed = y_sample_tensor

        # Если prediction скаляр, а y_squeezed (1,), делаем prediction (1,)
        if prediction.ndim == 0 and y_squeezed.ndim == 1 and y_squeezed.shape[0] == 1:
            prediction = prediction.unsqueeze(0)

        loss = worker_loss_fn(prediction, y_squeezed)
        loss_val = loss.item()
    except Exception as e_loss:
        # print(f"Worker PID {os.getpid()}: Error in loss calc: {e_loss}, pred_shape={prediction.shape if 'prediction' in locals() else 'N/A'}, y_sq_shape={y_squeezed.shape if 'y_squeezed' in locals() else 'N/A'}")
        dummy_grads = [np.zeros_like(param_np) for param_np in
                       model_state_dict_numpy.values()] if model_state_dict_numpy else [np.array([0.0])]
        return dummy_grads, float('inf')
    try:
        loss.backward()
    except Exception as e_backward:
        # print(f"Worker PID {os.getpid()}: Error in backward: {e_backward}")
        dummy_grads = [np.zeros_like(param_np) for param_np in
                       model_state_dict_numpy.values()] if model_state_dict_numpy else [np.array([0.0])]
        return dummy_grads, loss_val  # Возвращаем loss, но пустые градиенты

    grads_numpy = []
    # Собираем градиенты только для параметров, которые есть в model_state_dict_numpy
    # и которые требуют градиентов в worker_qnn_model
    for param_name, param in worker_qnn_model.named_parameters():
        if param_name in model_state_dict_numpy and param.requires_grad:  # Убедимся, что параметр был в state_dict
            if param.grad is not None:
                grads_numpy.append(param.grad.cpu().numpy().copy())
            else:
                # print(f"Worker PID {os.getpid()}: Grad is None for param {param_name}")
                grads_numpy.append(np.zeros_like(param.data.cpu().numpy()))  # Отправляем нулевой градиент

    # Если grads_numpy пуст, но должны были быть градиенты
    if not grads_numpy and num_trainable_params_in_state_dict > 0:
        # print(f"Worker PID {os.getpid()}: grads_numpy is empty but expected {num_trainable_params_in_state_dict} grad arrays.")
        # Создаем dummy_grads на основе структуры model_state_dict_numpy
        dummy_grads_list = []
        for name, param_np_template in model_state_dict_numpy.items():
            # Проверяем, есть ли такой параметр в worker_qnn_model и требует ли он градиент
            # Это сложно проверить здесь без доступа к структуре worker_qnn_model до state_dict load
            # Просто создадим нулевые массивы по шаблону state_dict
            dummy_grads_list.append(np.zeros_like(param_np_template))
        return dummy_grads_list, loss_val

    # Проверка: главный процесс ожидает список градиентов в том же порядке,
    # в котором param.requires_grad для main_model.named_parameters()
    # worker_qnn_model.named_parameters() должен иметь тот же порядок и те же обучаемые параметры.

    return grads_numpy, loss_val


# --- set_random_seeds (без изменений) ---
def set_random_seeds(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    qml.numpy.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed_value}")


def train_model():
    set_random_seeds(config.RANDOM_SEED)

    results_dir = config.RESULTS_DIR if hasattr(config, 'RESULTS_DIR') else "results"
    model_name_base = config.MODEL_NAME if hasattr(config, 'MODEL_NAME') else "qnn_model"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    checkpoint_path = os.path.join(results_dir, f"{model_name_base}_checkpoint.pth")
    final_model_path = os.path.join(results_dir, f"{model_name_base}_final.pth")
    history_plot_path = os.path.join(results_dir, f"{model_name_base}_training_history.png")
    history_json_path = os.path.join(results_dir, f"{model_name_base}_training_history.json")

    print("Starting training process...")
    # Убрано предупреждение о GPU, так как воркеры всегда CPU-based в этом скрипте
    # Но device_name в model_params_for_worker важен для консистентности QNN
    worker_device_for_qnn_init = "default.qubit"  # Явно для QNN в воркере

    print(f"Using quantum device for main model: {config.QUANTUM_DEVICE}")
    num_mp_workers = config.NUM_MP_WORKERS if hasattr(config, 'NUM_MP_WORKERS') else os.cpu_count()
    # Ограничим число воркеров, если оно слишком большое или не указано разумно
    if num_mp_workers is None or num_mp_workers <= 0:
        num_mp_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    elif num_mp_workers > os.cpu_count() * 2:  # Простое ограничение
        num_mp_workers = os.cpu_count()
        print(f"Warning: NUM_MP_WORKERS too high, reduced to {num_mp_workers}")

    print(
        f"Using {num_mp_workers} worker processes for gradient computation (workers will use CPU device for QNN init: {worker_device_for_qnn_init}).")

    try:
        train_loader, test_loader, num_features, y_scaler, feature_names = load_and_split_data(
            data_path=config.DATA_PATH,
            datetime_col=config.DATETIME_COLUMN,
            target_col=config.TARGET_COLUMN,
            other_exclude_cols=config.OTHER_EXCLUDE_COLS,
            test_size=config.TEST_SIZE,
            batch_size=config.BATCH_SIZE
        )
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Определение параметров модели (оставляем вашу логику)
    if config.EMBEDDING_TYPE == "Amplitude":
        actual_n_qubits_for_model = config.N_QUBITS_AMPLITUDE
        features_for_embedding_input = num_features
    elif config.EMBEDDING_TYPE == "Angle":
        if hasattr(config, 'USE_PCA') and config.USE_PCA and hasattr(config,
                                                                     'N_COMPONENTS_PCA'):  # Добавил проверку на N_COMPONENTS_PCA
            actual_n_qubits_for_model = config.N_COMPONENTS_PCA
            features_for_embedding_input = config.N_COMPONENTS_PCA
        else:
            actual_n_qubits_for_model = num_features
            features_for_embedding_input = num_features
        # print(f"Info: Using AngleEmbedding with {actual_n_qubits_for_model} qubits.") # Убрано для краткости
    else:
        raise ValueError(f"Unsupported EMBEDDING_TYPE in config: {config.EMBEDDING_TYPE}")
    print(
        f"Main model EMBEDDING_TYPE: {config.EMBEDDING_TYPE}, Ansatz Qubits: {actual_n_qubits_for_model}, Embedding Input Features: {features_for_embedding_input}")

    main_model = QNN(
        n_features_for_embedding=features_for_embedding_input,
        n_qubits_for_ansatz=actual_n_qubits_for_model,
        n_layers=config.N_LAYERS,
        embedding_type=config.EMBEDDING_TYPE,
        rotation_gate=config.ROTATION_GATE_EMBEDDING,
        device_name=config.QUANTUM_DEVICE  # Устройство для основной модели
    )

    # Перемещение основной модели на GPU, если это настроено
    if "gpu" in str(config.QUANTUM_DEVICE).lower() and torch.cuda.is_available():
        main_model.to(torch.device("cuda"))
        print("Main model moved to CUDA device.")
    elif "gpu" in str(config.QUANTUM_DEVICE).lower() and not torch.cuda.is_available():
        print(f"Warning: CUDA not available, main model using CPU despite QUANTUM_DEVICE='{config.QUANTUM_DEVICE}'.")

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(main_model.parameters(), lr=config.LEARNING_RATE,
                           weight_decay=getattr(config, 'WEIGHT_DECAY', 0.0))  # Добавлен weight_decay
    print("\nMain Model, Loss Function, and Optimizer initialized.")
    print(f"Main Model Trainable Parameters: {sum(p.numel() for p in main_model.parameters() if p.requires_grad)}")

    start_epoch = 0
    best_test_mse = float('inf')
    training_history = {
        'epoch': [], 'avg_train_loss': [], 'test_mse_original': [],
        'test_mae_original': [], 'actual_epoch_time_sec': []
    }

    if os.path.exists(checkpoint_path):  # Загрузка чекпоинта (оставляем вашу логику)
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        try:
            checkpoint_device = 'cuda' if "gpu" in str(
                config.QUANTUM_DEVICE).lower() and torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(checkpoint_path, map_location=checkpoint_device)  # Загружаем на целевое устройство

            main_model.load_state_dict(checkpoint['model_state_dict'])
            # Если оптимизатор тоже на GPU, его нужно переместить
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint_device == 'cuda':  # Если оптимизатор был на GPU, его состояние тоже может быть на GPU
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

            start_epoch = checkpoint['epoch'] + 1
            training_history = checkpoint['training_history']
            best_test_mse = checkpoint.get('best_test_mse', float('inf'))
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}. Best test MSE: {best_test_mse:.6f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            training_history = {k: [] for k in training_history.keys()}
            best_test_mse = float('inf')
    else:
        print("No checkpoint found. Starting from scratch.")

    total_batches_per_epoch = len(train_loader)
    print(f"Total batches per epoch: {total_batches_per_epoch}")
    # batch_times уже был deque(maxlen=10), изменил на 20 для соответствия ТЗ
    batch_times = deque(maxlen=20)  # <--- ИЗМЕНЕНО maxlen

    model_params_for_worker = {
        'features_for_embedding_input': features_for_embedding_input,
        'actual_n_qubits_for_model': actual_n_qubits_for_model,
        'n_layers': config.N_LAYERS,
        'embedding_type': config.EMBEDDING_TYPE,
        'rotation_gate': config.ROTATION_GATE_EMBEDDING,
        'device_name': worker_device_for_qnn_init  # Передаем CPU-device для QNN в воркерах
    }
    loss_fn_for_worker_str = "MSELoss"  # Можно сделать конфигурируемым

    print("\nStarting training loop with multiprocessing...")
    mp_context = multiprocessing.get_context(config.MP_START_METHOD if hasattr(config, 'MP_START_METHOD') else 'spawn')

    for epoch in range(start_epoch, config.EPOCHS):
        main_model.train()
        epoch_loss_sum = 0.0
        num_samples_processed_in_epoch = 0
        start_time_epoch = time.time()

        # Словарь для хранения усредненных градиентов для логирования
        # Ключ - имя параметра, значение - numpy массив градиента
        # Он будет обновляться перед optimizer.step()
        applied_avg_grads_for_log = {}

        with mp_context.Pool(processes=num_mp_workers,
                             initializer=init_worker_cpu,
                             initargs=(model_params_for_worker, loss_fn_for_worker_str)) as pool:

            for batch_idx, (X_batch_torch, y_batch_torch) in enumerate(train_loader):
                start_time_batch_logging = time.time()
                main_model_state_dict_numpy = {name: param.cpu().detach().numpy()
                                               # Всегда на CPU для передачи в воркеры
                                               for name, param in main_model.state_dict().items()}

                batch_data_for_pool = []
                # Данные для воркеров всегда CPU numpy
                for i in range(X_batch_torch.size(0)):
                    batch_data_for_pool.append(
                        (X_batch_torch[i].cpu().numpy(), y_batch_torch[i].cpu().numpy())
                    )

                process_func = partial(process_sample_for_grads,
                                       model_state_dict_numpy=main_model_state_dict_numpy)
                results = pool.map(process_func, batch_data_for_pool)

                valid_results = [res for res in results if
                                 res is not None and np.isfinite(res[1]) and isinstance(res[0], list) and len(
                                     res[0]) > 0]

                if not valid_results:
                    current_batch_time_logging = time.time() - start_time_batch_logging
                    batch_times.append(current_batch_time_logging if current_batch_time_logging > 0.001 else 0.001)
                    if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                        print(
                            f"  Epoch {epoch + 1}/{config.EPOCHS} | Batch {batch_idx + 1}/{total_batches_per_epoch} SKIPPED (all samples failed in workers or returned invalid grads)")
                    continue

                # Усреднение градиентов
                num_grad_arrays_from_worker = len(valid_results[0][0])
                summed_grads_numpy = [np.zeros_like(valid_results[0][0][i]) for i in range(num_grad_arrays_from_worker)]

                for res_grads_list, _ in valid_results:
                    if len(res_grads_list) == num_grad_arrays_from_worker:
                        for i in range(num_grad_arrays_from_worker):
                            summed_grads_numpy[i] += res_grads_list[i]

                avg_grads_numpy = [s_grad / len(valid_results) for s_grad in summed_grads_numpy]

                # Очищаем applied_avg_grads_for_log перед заполнением новыми значениями
                applied_avg_grads_for_log.clear()

                optimizer.zero_grad()
                param_idx = 0
                for param_name, param in main_model.named_parameters():
                    if param.requires_grad:
                        if param_idx < len(avg_grads_numpy):
                            grad_tensor = torch.tensor(avg_grads_numpy[param_idx], dtype=param.dtype,
                                                       device=param.device)
                            param.grad = grad_tensor
                            # Сохраняем усредненный градиент для логирования
                            applied_avg_grads_for_log[param_name] = avg_grads_numpy[param_idx].flatten()
                            param_idx += 1
                        # else:
                        #     print(f"Warning: Not enough avg_grads for param {param_name}")

                # ---> НАЧАЛО БЛОКА ДЕТАЛЬНОГО ЛОГИРОВАНИЯ (КАЖДЫЕ 20 БАТЧЕЙ) <---
                log_details_this_batch = (batch_idx + 1) % 20 == 0 or (batch_idx == 0 and total_batches_per_epoch > 0)
                current_batch_avg_loss_for_log = sum(res[1] for res in valid_results) / len(
                    valid_results) if valid_results else float('nan')

                if log_details_this_batch:
                    print(
                        f"\n--- Detailed Stats at Epoch {epoch + 1}, Batch {batch_idx + 1}/{total_batches_per_epoch} (Avg Batch Loss: {current_batch_avg_loss_for_log:.6f}) ---")
                    for param_name, param in main_model.named_parameters():
                        if param.requires_grad:  # Логируем только обучаемые параметры
                            weights_data = param.data.cpu().numpy().flatten()  # Веса всегда с CPU для numpy
                            print(f"Param '{param_name}' (Shape: {list(param.shape)}):")
                            print(f"  Weights: Min: {weights_data.min():.4f}, Max: {weights_data.max():.4f}, "
                                  f"Mean: {weights_data.mean():.4f}, Std: {weights_data.std():.4f}")

                            if param_name in applied_avg_grads_for_log:
                                grads_data_log = applied_avg_grads_for_log[param_name]  # Это уже numpy flatten
                                print(
                                    f"  Applied Avg Grads: Min: {grads_data_log.min():.4e}, Max: {grads_data_log.max():.4e}, "
                                    f"Mean Abs: {np.mean(np.abs(grads_data_log)):.4e}, Std: {grads_data_log.std():.4e}")
                            elif param.grad is not None:  # Резервный вариант, если applied_avg_grads_for_log не заполнился
                                grads_data_log_fallback = param.grad.data.cpu().numpy().flatten()
                                print(
                                    f"  param.grad Fallback: Min: {grads_data_log_fallback.min():.4e}, Max: {grads_data_log_fallback.max():.4e}, "
                                    f"Mean Abs: {np.mean(np.abs(grads_data_log_fallback)):.4e}, Std: {grads_data_log_fallback.std():.4e}")
                            else:
                                print(f"  Applied Avg Grads: Grad was None or not logged for this param.")
                    print("---")
                # ---> КОНЕЦ БЛОКА ДЕТАЛЬНОГО ЛОГИРОВАНИЯ <---

                optimizer.step()

                epoch_loss_sum += current_batch_avg_loss_for_log * len(valid_results)
                num_samples_processed_in_epoch += len(valid_results)

                end_time_batch_logging = time.time()
                current_batch_time_logging = end_time_batch_logging - start_time_batch_logging
                batch_times.append(current_batch_time_logging if current_batch_time_logging > 0.001 else 0.001)

                if log_details_this_batch:  # Выводим обычное логирование времени вместе с детальным
                    avg_recent_batch_time = sum(batch_times) / len(batch_times)
                    remaining_batches_in_epoch = total_batches_per_epoch - (batch_idx + 1)
                    estimated_epoch_remaining_time = remaining_batches_in_epoch * avg_recent_batch_time
                    estimated_total_epoch_time = total_batches_per_epoch * avg_recent_batch_time
                    print(f"Batch Time: {current_batch_time_logging:.3f}s | "
                          f"Avg Last {len(batch_times)}: {avg_recent_batch_time:.3f}s/batch | "
                          f"Est. Epoch Time: {estimated_total_epoch_time / 60:.2f} min | "
                          f"Est. Epoch Rem: {estimated_epoch_remaining_time / 60:.2f} min")
                    print("--- End Detailed Stats Block ---")

        avg_train_loss_epoch = epoch_loss_sum / num_samples_processed_in_epoch if num_samples_processed_in_epoch > 0 else float(
            'inf')
        end_time_epoch = time.time()
        actual_epoch_time = end_time_epoch - start_time_epoch
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS} Summary:")
        print(f"  Avg Training Loss: {avg_train_loss_epoch:.6f}")
        print(f"  Actual Epoch Time: {actual_epoch_time / 60:.2f} min ({actual_epoch_time:.2f}s)\n")

        current_test_mse_orig, current_test_mae_orig = float('nan'), float('nan')
        if len(test_loader.dataset) > 0:
            main_model.eval()
            test_loss_sum_scaled = 0.0
            all_predictions_scaled_epoch = []
            all_targets_scaled_epoch = []
            with torch.no_grad():
                for X_batch_test_cpu, y_batch_test_cpu in test_loader:  # Данные с CPU
                    # Перемещение на устройство основной модели
                    target_device = next(main_model.parameters()).device
                    X_batch_test = X_batch_test_cpu.to(target_device, non_blocking=True)
                    y_batch_test = y_batch_test_cpu.to(target_device, non_blocking=True)

                    predictions_scaled = main_model(X_batch_test)

                    # Ваша логика обработки форм
                    if y_batch_test.ndim == 2 and y_batch_test.shape[1] == 1 and predictions_scaled.ndim == 1:
                        y_squeezed_test = y_batch_test.squeeze(1)
                        predictions_final_test = predictions_scaled
                    elif y_batch_test.ndim == 1 and predictions_scaled.ndim == 1 and y_batch_test.shape[0] == \
                            predictions_scaled.shape[0]:
                        y_squeezed_test = y_batch_test
                        predictions_final_test = predictions_scaled
                    elif predictions_scaled.ndim > 1 and predictions_scaled.shape[0] == y_batch_test.shape[0]:
                        y_squeezed_test = y_batch_test.squeeze(1) if y_batch_test.ndim > 1 and y_batch_test.shape[
                            1] == 1 else y_batch_test
                        predictions_final_test = predictions_scaled.squeeze(1) if predictions_scaled.ndim > 1 and \
                                                                                  predictions_scaled.shape[
                                                                                      1] == 1 else predictions_scaled
                    else:
                        y_squeezed_test = y_batch_test.squeeze() if y_batch_test.ndim > predictions_scaled.ndim else y_batch_test
                        predictions_final_test = predictions_scaled.squeeze() if predictions_scaled.ndim > y_squeezed_test.ndim else predictions_scaled
                        if y_squeezed_test.ndim == 0: y_squeezed_test = y_squeezed_test.unsqueeze(0)
                        if predictions_final_test.ndim == 0: predictions_final_test = predictions_final_test.unsqueeze(
                            0)

                    try:
                        loss_test_batch = loss_fn(predictions_final_test, y_squeezed_test)
                        test_loss_sum_scaled += loss_test_batch.item()
                    except Exception as e_test_loss:
                        # print(f"Error calc test loss: {e_test_loss}, pred: {predictions_final_test.shape}, target: {y_squeezed_test.shape}")
                        pass  # Пропускаем, если ошибка

                    all_predictions_scaled_epoch.extend(predictions_final_test.cpu().numpy().flatten())
                    all_targets_scaled_epoch.extend(y_squeezed_test.cpu().numpy().flatten())

            avg_test_loss_scaled_epoch = test_loss_sum_scaled / len(test_loader) if len(test_loader) > 0 else float(
                'inf')
            print(f"Epoch {epoch + 1} - Avg Test Loss (scaled): {avg_test_loss_scaled_epoch:.6f}")

            if all_predictions_scaled_epoch and y_scaler is not None:
                preds_np_epoch = np.array(all_predictions_scaled_epoch).reshape(-1, 1)
                targets_np_epoch = np.array(all_targets_scaled_epoch).reshape(-1, 1)
                try:
                    preds_orig_epoch = y_scaler.inverse_transform(preds_np_epoch)
                    targets_orig_epoch = y_scaler.inverse_transform(targets_np_epoch)
                    current_test_mse_orig = np.mean((preds_orig_epoch - targets_orig_epoch) ** 2)
                    current_test_mae_orig = np.mean(np.abs(preds_orig_epoch - targets_orig_epoch))
                    print(
                        f"Epoch {epoch + 1} - Test MSE (original): {current_test_mse_orig:.6f}, Test MAE (original): {current_test_mae_orig:.6f}")

                    if np.isfinite(
                            current_test_mse_orig) and current_test_mse_orig < best_test_mse:  # <--- Добавил isfinite
                        best_test_mse = current_test_mse_orig
                        torch.save({
                            'epoch': epoch, 'model_state_dict': main_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_train_loss_epoch,
                            'test_mse_original': current_test_mse_orig, 'test_mae_original': current_test_mae_orig,
                            'config': {k: v for k, v in vars(config).items() if not k.startswith('__')}
                        }, os.path.join(results_dir, f"{model_name_base}_best.pth"))
                        print(f"Epoch {epoch + 1}: New best model saved with Test MSE: {best_test_mse:.6f}")
                except Exception as e_scale:
                    print(f"Error during inverse scaling epoch test metrics: {e_scale}")
            elif y_scaler is None:
                print("y_scaler is None, skipping original scale test metrics.")
        else:
            print(f"Epoch {epoch + 1} - No test data for evaluation.")

        # Обновление и сохранение истории и чекпоинта (оставляем вашу логику, но с float для json)
        training_history['epoch'].append(epoch + 1)
        training_history['avg_train_loss'].append(
            float(avg_train_loss_epoch) if np.isfinite(avg_train_loss_epoch) else None)
        training_history['test_mse_original'].append(
            float(current_test_mse_orig) if np.isfinite(current_test_mse_orig) else None)
        training_history['test_mae_original'].append(
            float(current_test_mae_orig) if np.isfinite(current_test_mae_orig) else None)
        training_history['actual_epoch_time_sec'].append(float(actual_epoch_time))

        print(f"Saving checkpoint to {checkpoint_path} after epoch {epoch + 1}...")
        try:
            torch.save({
                'epoch': epoch, 'model_state_dict': main_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'training_history': training_history,
                'best_test_mse': float(best_test_mse),  # Убедимся что float
                'config': {k: v for k, v in vars(config).items() if not k.startswith('__')}
            }, checkpoint_path)
            print("Checkpoint saved successfully.")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

        try:  # Сохранение истории в JSON (оставляем вашу логику)
            with open(history_json_path, 'w') as f_json:
                serializable_history = {}
                for key, values in training_history.items():
                    serializable_history[key] = [
                        (float(v) if isinstance(v, (np.float32, np.float64, np.ndarray, torch.Tensor)) and np.isfinite(
                            v) else
                         (None if isinstance(v, (np.float32, np.float64, np.ndarray, torch.Tensor)) and not np.isfinite(
                             v) else v))
                        for v in values]
                json.dump(serializable_history, f_json, indent=4)
            # print(f"Training history saved to {history_json_path}") # Можно убрать для краткости
        except Exception as e_json:
            print(f"Error saving training history to JSON: {e_json}")

    print("\nTraining finished.")
    # Сохранение финальной модели (оставляем вашу логику)
    print(f"\nSaving final model to {final_model_path}...")
    try:
        torch.save({
            'epoch': config.EPOCHS - 1, 'model_state_dict': main_model.state_dict(),
            'training_history': training_history, 'best_test_mse': float(best_test_mse),
            'config': {k: v for k, v in vars(config).items() if not k.startswith('__')}
        }, final_model_path)
        print("Final model saved successfully.")
    except Exception as e:
        print(f"Error saving final model: {e}")

    # Построение графика (оставляем вашу логику)
    print(f"\nGenerating and saving training history plot to {history_plot_path}...")
    try:
        plt.figure(figsize=(12, 8))
        ax1 = plt.gca()
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Avg Train Loss (Scaled)', color=color)
        epochs_for_plot = np.array(training_history['epoch'])
        train_losses_for_plot = np.array(training_history['avg_train_loss'], dtype=float)
        test_mse_for_plot = np.array(training_history['test_mse_original'], dtype=float)

        valid_train_mask = np.isfinite(train_losses_for_plot)
        if np.any(valid_train_mask):
            ax1.plot(epochs_for_plot[valid_train_mask], train_losses_for_plot[valid_train_mask], color=color,
                     marker='o', linestyle='-', label='Avg Train Loss (Scaled)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        plt.grid(True, axis='y', linestyle=':', alpha=0.7)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Test MSE (Original Scale)', color=color)
        valid_test_mask = np.isfinite(test_mse_for_plot)
        if np.any(valid_test_mask):
            ax2.plot(epochs_for_plot[valid_test_mask], test_mse_for_plot[valid_test_mask], color=color, marker='x',
                     linestyle='--', label='Test MSE (Original Scale)')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        plt.title(f'Training History - {model_name_base} (Epochs: {len(epochs_for_plot)})')
        fig = plt.gcf()
        fig.tight_layout()
        plt.savefig(history_plot_path)
        plt.close(fig)
        print("Training history plot saved.")
    except Exception as e_plot:
        print(f"Error generating plot: {e_plot}")


if __name__ == '__main__':
    # Установка метода start для multiprocessing (оставляем вашу логику)
    current_start_method = multiprocessing.get_start_method(allow_none=True)
    desired_start_method = getattr(config, 'MP_START_METHOD', 'spawn')  # По умолчанию 'spawn' из конфига

    if current_start_method is None:
        try:
            multiprocessing.set_start_method(desired_start_method, force=True)
            print(f"Multiprocessing start method set to '{desired_start_method}'.")
        except RuntimeError as e:  # Если уже установлен контекст
            print(
                f"Warning: Could not set start method to '{desired_start_method}' (may be already set by context): {e}. Current: {multiprocessing.get_start_method(allow_none=True)}")

    elif current_start_method != desired_start_method:
        try:
            multiprocessing.set_start_method(desired_start_method, force=True)
            print(f"Multiprocessing start method changed from '{current_start_method}' to '{desired_start_method}'.")
        except RuntimeError as e:
            print(
                f"Warning: Could not change multiprocessing start method from '{current_start_method}' to '{desired_start_method}': {e}. Using current method.")
    else:
        print(f"Multiprocessing start method is already '{desired_start_method}'.")

    train_model()