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
import matplotlib.pyplot as plt  # <--- ДОБАВЛЕНО для графиков
import json  # <--- ДОБАВЛЕНО для сохранения истории как JSON

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
            device_name=model_params_for_worker['device_name']
        )
        worker_qnn_model.eval()
    except Exception as e:
        print(f"Worker PID {os.getpid()}: Error initializing QNN model: {e}")
        worker_qnn_model = None
    if loss_fn_for_worker_str == "MSELoss":
        worker_loss_fn = nn.MSELoss()
    else:
        worker_loss_fn = nn.MSELoss()


# --- process_sample_for_grads (без изменений) ---
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

    #Используем y до его преобразования в тензор и squeeze
    weight = 1.0
    if not (-0.2 <= y_sample_np <= 0.0):  # Если значение вне доминирующего диапазона
        weight =15.0
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
        loss_val = weight*loss.item()
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

    # --- Создание директории для результатов ---
    results_dir = config.RESULTS_DIR if hasattr(config, 'RESULTS_DIR') else "results"
    model_name_base = config.MODEL_NAME if hasattr(config, 'MODEL_NAME') else "qnn_model"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    # --- Файлы для сохранения ---
    checkpoint_path = os.path.join(results_dir, f"{model_name_base}_checkpoint.pth")
    final_model_path = os.path.join(results_dir, f"{model_name_base}_final.pth")
    history_plot_path = os.path.join(results_dir, f"{model_name_base}_training_history.png")
    history_json_path = os.path.join(results_dir,
                                     f"{model_name_base}_training_history.json")  # Для сохранения данных графика

    print("Starting training process...")
    if "gpu" in config.QUANTUM_DEVICE.lower():
        print(
            f"WARNING: QUANTUM_DEVICE is set to '{config.QUANTUM_DEVICE}'. Multiprocessing CPU parallelism is intended for CPU devices like 'lightning.qubit' or 'default.qubit'.")
    print(f"Using quantum device for main model: {config.QUANTUM_DEVICE}")
    num_mp_workers = config.NUM_MP_WORKERS if hasattr(config, 'NUM_MP_WORKERS') else os.cpu_count()
    print(f"Using {num_mp_workers} worker processes for gradient computation.")

    # 1. Загрузка и подготовка данных (без изменений)
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

    # Определение параметров модели (без изменений)
    if config.EMBEDDING_TYPE == "Amplitude":
        actual_n_qubits_for_model = config.N_QUBITS_AMPLITUDE
        features_for_embedding_input = num_features
    elif config.EMBEDDING_TYPE == "Angle":
        if hasattr(config, 'USE_PCA') and config.USE_PCA:
            actual_n_qubits_for_model = config.N_COMPONENTS_PCA
            features_for_embedding_input = config.N_COMPONENTS_PCA
        else:
            actual_n_qubits_for_model = num_features
            features_for_embedding_input = num_features
        print(f"Info: Using AngleEmbedding with {actual_n_qubits_for_model} qubits.")
    else:
        raise ValueError(f"Unsupported EMBEDDING_TYPE in config: {config.EMBEDDING_TYPE}")
    print(f"Main model will use {actual_n_qubits_for_model} qubits for the ansatz.")
    print(f"Embedding layer will receive {features_for_embedding_input} features as input.")

    # --- Основная модель (в главном процессе) (без изменений) ---
    main_model = QNN(
        n_features_for_embedding=features_for_embedding_input,
        n_qubits_for_ansatz=actual_n_qubits_for_model,
        n_layers=config.N_LAYERS,
        embedding_type=config.EMBEDDING_TYPE,
        rotation_gate=config.ROTATION_GATE_EMBEDDING,
        device_name=config.QUANTUM_DEVICE
    )
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(main_model.parameters(), lr=config.LEARNING_RATE)
    print("\nMain Model, Loss Function, and Optimizer initialized.")
    print(f"Main Model Parameters: {sum(p.numel() for p in main_model.parameters() if p.requires_grad)}")

    # --- Инициализация для чекпоинтов и истории ---
    start_epoch = 0
    best_test_mse = float('inf')  # Для сохранения лучшей модели по тестовому MSE
    training_history = {
        'epoch': [],
        'avg_train_loss': [],  # Будем хранить средний лосс на трейне за эпоху
        'test_mse_original': [],  # MSE на тесте (оригинальный масштаб) после каждой эпохи
        'test_mae_original': [],  # MAE на тесте (оригинальный масштаб) после каждой эпохи
        'actual_epoch_time_sec': []
    }

    # --- Загрузка чекпоинта ---
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path)
            main_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # Начинаем со следующей эпохи
            training_history = checkpoint['training_history']
            best_test_mse = checkpoint.get('best_test_mse', float('inf'))  # Загружаем best_test_mse
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
            print(f"Last recorded best_test_mse: {best_test_mse}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            training_history = {k: [] for k in training_history}  # Сброс истории
            best_test_mse = float('inf')
    else:
        print("No checkpoint found. Starting from scratch.")

    total_batches_per_epoch = len(train_loader)
    print(f"Total batches per epoch: {total_batches_per_epoch}")
    batch_times = deque(maxlen=10)
    model_params_for_worker = {
        'features_for_embedding_input': features_for_embedding_input,
        'actual_n_qubits_for_model': actual_n_qubits_for_model,
        'n_layers': config.N_LAYERS,
        'embedding_type': config.EMBEDDING_TYPE,
        'rotation_gate': config.ROTATION_GATE_EMBEDDING,
        'device_name': config.QUANTUM_DEVICE
    }
    loss_fn_for_worker_str = "MSELoss"

    # 3. Цикл обучения
    print("\nStarting training loop with multiprocessing...")
    mp_context = multiprocessing.get_context('spawn')

    for epoch in range(start_epoch, config.EPOCHS):
        main_model.train()
        epoch_loss_sum = 0.0
        num_samples_processed_in_epoch = 0
        start_time_epoch = time.time()

        with mp_context.Pool(processes=num_mp_workers,
                             initializer=init_worker_cpu,
                             initargs=(model_params_for_worker, loss_fn_for_worker_str)) as pool:
            # print(f" Epoch {epoch + 1}/{config.EPOCHS}: Process pool with {num_mp_workers} workers created.") # Можно убрать для краткости

            for batch_idx, (X_batch_torch, y_batch_torch) in enumerate(train_loader):
                start_time_batch_logging = time.time()  # Для логирования времени батча
                main_model_state_dict_numpy = {name: param.cpu().detach().numpy()
                                               for name, param in main_model.state_dict().items()}
                batch_data_for_pool = []
                for i in range(X_batch_torch.size(0)):
                    batch_data_for_pool.append(
                        (X_batch_torch[i].cpu().numpy(), y_batch_torch[i].cpu().numpy())
                    )
                process_func = partial(process_sample_for_grads,
                                       model_state_dict_numpy=main_model_state_dict_numpy)
                results = pool.map(process_func, batch_data_for_pool)
                valid_results = [res for res in results if res is not None and np.isfinite(res[1])]
                if not valid_results:
                    current_batch_time_logging = time.time() - start_time_batch_logging
                    batch_times.append(current_batch_time_logging if current_batch_time_logging > 0 else 0.1)
                    if (batch_idx + 1) % 10 == 0 or batch_idx == 0:  # Выводим лог даже если батч провален
                        print(
                            f"  Epoch {epoch + 1}/{config.EPOCHS} | Batch {batch_idx + 1}/{total_batches_per_epoch} SKIPPED (all samples failed)")
                    continue

                num_model_params = len(valid_results[0][0])
                summed_grads_numpy = [np.zeros_like(grad_template) for grad_template in valid_results[0][0]]
                for res_grads_list, _ in valid_results:
                    for i in range(num_model_params):
                        summed_grads_numpy[i] += res_grads_list[i]
                avg_grads_numpy = [s_grad / len(valid_results) for s_grad in summed_grads_numpy]

                optimizer.zero_grad()
                param_idx = 0
                for param_name, param in main_model.named_parameters():
                    if param.requires_grad:
                        if param_idx < len(avg_grads_numpy):
                            grad_tensor = torch.tensor(avg_grads_numpy[param_idx], dtype=param.dtype,
                                                       device=param.device)
                            param.grad = grad_tensor
                            param_idx += 1
                        # else: # Убрано предупреждение для краткости
                        #     print(f"Warning: Mismatch in number of gradients and model parameters. Param: {param_name}")
                optimizer.step()
                current_batch_loss_sum = sum(res[1] for res in valid_results)
                epoch_loss_sum += current_batch_loss_sum
                num_samples_processed_in_epoch += len(valid_results)

                # Логирование времени (без изменений)
                end_time_batch_logging = time.time()
                current_batch_time_logging = end_time_batch_logging - start_time_batch_logging
                batch_times.append(current_batch_time_logging)
                if (batch_idx + 1) % 10 == 0 or batch_idx == 0 and total_batches_per_epoch > 0:
                    avg_recent_batch_time = sum(batch_times) / len(batch_times)
                    remaining_batches_in_epoch = total_batches_per_epoch - (batch_idx + 1)
                    estimated_epoch_remaining_time = remaining_batches_in_epoch * avg_recent_batch_time
                    estimated_total_epoch_time = total_batches_per_epoch * avg_recent_batch_time
                    print(f"  Epoch {epoch + 1}/{config.EPOCHS} | Batch {batch_idx + 1}/{total_batches_per_epoch} "
                          f"| Batch Time: {current_batch_time_logging:.3f}s "
                          f"| Avg Last {len(batch_times)}: {avg_recent_batch_time:.3f}s/batch "
                          f"| Est. Epoch Time: {estimated_total_epoch_time / 60:.2f} min "
                          f"| Est. Epoch Rem: {estimated_epoch_remaining_time / 60:.2f} min")

        avg_train_loss_epoch = epoch_loss_sum / num_samples_processed_in_epoch if num_samples_processed_in_epoch > 0 else float(
            'inf')
        end_time_epoch = time.time()
        actual_epoch_time = end_time_epoch - start_time_epoch
        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} - Avg Training Loss: {avg_train_loss_epoch:.6f} - Actual Epoch Time: {actual_epoch_time / 60:.2f} min ({actual_epoch_time:.2f}s)")

        # --- Оценка на тестовой выборке ПОСЛЕ КАЖДОЙ ЭПОХИ для истории и лучшей модели ---
        current_test_mse_orig, current_test_mae_orig = float('nan'), float('nan')
        if len(test_loader.dataset) > 0:  # Проверяем, есть ли тестовые данные
            main_model.eval()  # Переводим модель в режим оценки для теста
            test_loss_sum_scaled = 0.0
            all_predictions_scaled_epoch = []
            all_targets_scaled_epoch = []
            with torch.no_grad():
                for X_batch_test, y_batch_test in test_loader:
                    predictions_scaled = main_model(X_batch_test)
                    # ... (код обработки форм y_batch_test и predictions_scaled как в финальной оценке)
                    if y_batch_test.ndim == 2 and y_batch_test.shape[1] == 1 and predictions_scaled.ndim == 1:
                        y_squeezed_test = y_batch_test.squeeze(1)
                    else:
                        y_squeezed_test = y_batch_test
                    if predictions_scaled.ndim == 2 and predictions_scaled.shape[1] == 1 and y_squeezed_test.ndim == 1:
                        predictions_final_test = predictions_scaled.squeeze(1)
                    else:
                        predictions_final_test = predictions_scaled

                    loss_test_batch = loss_fn(predictions_final_test, y_squeezed_test)
                    test_loss_sum_scaled += loss_test_batch.item()
                    all_predictions_scaled_epoch.extend(predictions_final_test.cpu().numpy())
                    all_targets_scaled_epoch.extend(y_squeezed_test.cpu().numpy())

            avg_test_loss_scaled_epoch = test_loss_sum_scaled / len(test_loader) if len(test_loader) > 0 else float(
                'inf')
            print(f"Epoch {epoch + 1} - Avg Test Loss (scaled): {avg_test_loss_scaled_epoch:.6f}")

            if all_predictions_scaled_epoch:  # Если были предсказания
                preds_np_epoch = np.array(all_predictions_scaled_epoch).reshape(-1, 1)
                targets_np_epoch = np.array(all_targets_scaled_epoch).reshape(-1, 1)
                try:
                    preds_orig_epoch = y_scaler.inverse_transform(preds_np_epoch)
                    targets_orig_epoch = y_scaler.inverse_transform(targets_np_epoch)
                    current_test_mse_orig = np.mean((preds_orig_epoch - targets_orig_epoch) ** 2)
                    current_test_mae_orig = np.mean(np.abs(preds_orig_epoch - targets_orig_epoch))
                    print(
                        f"Epoch {epoch + 1} - Test MSE (original): {current_test_mse_orig:.6f}, Test MAE (original): {current_test_mae_orig:.6f}")

                    # --- Сохранение лучшей модели ---
                    if current_test_mse_orig < best_test_mse:
                        best_test_mse = current_test_mse_orig
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': main_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_train_loss_epoch,  # Можно сохранить и другие метрики
                            'test_mse_original': current_test_mse_orig,
                            'test_mae_original': current_test_mae_orig,
                            'config': {k: v for k, v in vars(config).items() if not k.startswith('__')}
                            # Сохраняем конфиг
                        }, os.path.join(results_dir, f"{model_name_base}_best.pth"))
                        print(f"Epoch {epoch + 1}: New best model saved with Test MSE: {best_test_mse:.6f}")

                except Exception as e_scale:
                    print(f"Error during inverse scaling for epoch test metrics: {e_scale}")
        else:
            print(f"Epoch {epoch + 1} - No test data for evaluation.")

        # --- Обновление истории ---
        training_history['epoch'].append(epoch + 1)
        training_history['avg_train_loss'].append(avg_train_loss_epoch)
        training_history['test_mse_original'].append(current_test_mse_orig)
        training_history['test_mae_original'].append(current_test_mae_orig)
        training_history['actual_epoch_time_sec'].append(actual_epoch_time)

        # --- Сохранение чекпоинта после каждой эпохи ---
        print(f"Saving checkpoint to {checkpoint_path} after epoch {epoch + 1}...")
        try:
            torch.save({
                'epoch': epoch,  # Сохраняем номер завершенной эпохи
                'model_state_dict': main_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_history': training_history,
                'best_test_mse': best_test_mse,
                'config': {k: v for k, v in vars(config).items() if not k.startswith('__')}  # Сохраняем конфиг
            }, checkpoint_path)
            print("Checkpoint saved successfully.")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

        # --- Сохранение данных истории в JSON ---
        try:
            with open(history_json_path, 'w') as f_json:
                # Преобразуем numpy float32 в float для JSON сериализации, если они есть в истории
                serializable_history = {}
                for key, values in training_history.items():
                    serializable_history[key] = [float(v) if isinstance(v, (np.float32, np.float64, np.ndarray)) else v
                                                 for v in values]
                json.dump(serializable_history, f_json, indent=4)
            print(f"Training history saved to {history_json_path}")
        except Exception as e_json:
            print(f"Error saving training history to JSON: {e_json}")

    print("\nTraining finished.")

    # --- Сохранение финальной модели ---
    print(f"\nSaving final model to {final_model_path}...")
    try:
        torch.save({
            'epoch': config.EPOCHS - 1,  # Последняя выполненная эпоха
            'model_state_dict': main_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),  # Можно не сохранять, если модель только для инференса
            'training_history': training_history,
            'best_test_mse': best_test_mse,
            'config': {k: v for k, v in vars(config).items() if not k.startswith('__')}
        }, final_model_path)
        print("Final model saved successfully.")
    except Exception as e:
        print(f"Error saving final model: {e}")

    # --- Построение и сохранение графика обучения ---
    print(f"\nGenerating and saving training history plot to {history_plot_path}...")
    try:
        plt.figure(figsize=(12, 8))

        # График потерь на обучении
        ax1 = plt.gca()
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Avg Train Loss (Scaled)', color=color)
        # Фильтруем nan/inf перед построением
        valid_train_loss = [loss for loss in training_history['avg_train_loss'] if np.isfinite(loss)]
        valid_epochs_train_loss = [training_history['epoch'][i] for i, loss in
                                   enumerate(training_history['avg_train_loss']) if np.isfinite(loss)]
        if valid_epochs_train_loss:
            ax1.plot(valid_epochs_train_loss, valid_train_loss, color=color, marker='o', linestyle='-',
                     label='Avg Train Loss (Scaled)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        plt.grid(True, axis='y', linestyle=':', alpha=0.7)

        # График MSE на тесте (оригинальный масштаб)
        ax2 = ax1.twinx()  # разделяемая ось X
        color = 'tab:blue'
        ax2.set_ylabel('Test MSE (Original Scale)', color=color)
        # Фильтруем nan/inf перед построением
        valid_test_mse = [mse for mse in training_history['test_mse_original'] if np.isfinite(mse)]
        valid_epochs_test_mse = [training_history['epoch'][i] for i, mse in
                                 enumerate(training_history['test_mse_original']) if np.isfinite(mse)]
        if valid_epochs_test_mse:
            ax2.plot(valid_epochs_test_mse, valid_test_mse, color=color, marker='x', linestyle='--',
                     label='Test MSE (Original Scale)')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        plt.title(f'Training History - {model_name_base}')
        fig = plt.gcf()  # Получаем текущую фигуру
        fig.tight_layout()  # чтобы уместить все метки
        plt.savefig(history_plot_path)
        plt.close(fig)  # Закрываем фигуру, чтобы не отображалась в блокнотах и не накапливалась
        print("Training history plot saved.")
    except Exception as e_plot:
        print(f"Error generating plot: {e_plot}")


if __name__ == '__main__':
    try:
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn')
        print(f"Multiprocessing start method: {multiprocessing.get_start_method()}.")
    except RuntimeError as e:
        print(f"Could not set multiprocessing start method (may be already set): {e}")
        print(f"Current start method: {multiprocessing.get_start_method()}.")
    train_model()