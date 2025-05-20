# analyze_model.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import time  # <--- ДОБАВЛЕНО для логирования времени
from collections import deque  # <--- ДОБАВЛЕНО для логирования времени
import multiprocessing  # <--- ДОБАВЛЕНО для распараллеливания
from functools import partial  # <--- ДОБАВЛЕНО для распараллеливания

import config
from data_loader import load_and_split_data
from quantum_model import QNN

# --- Глобальные переменные для воркера инференса ---
worker_qnn_model_inference = None
worker_model_params_g_inference = None


# --- Инициализатор для воркера инференса ---
def init_worker_inference(model_params_for_worker):
    global worker_qnn_model_inference, worker_model_params_g_inference
    worker_model_params_g_inference = model_params_for_worker
    pid = os.getpid()
    # print(f"DEBUG_INFERENCE_WORKER (PID:{pid}): init_worker_inference started. Params: {model_params_for_worker}")
    print(model_params_for_worker)
    try:
        worker_qnn_model_inference = QNN(
            n_features_for_embedding=model_params_for_worker['n_features_for_embedding'],
            n_qubits_for_ansatz=model_params_for_worker['n_qubits_for_ansatz'],
            n_layers=model_params_for_worker['n_layers'],
            embedding_type=model_params_for_worker['embedding_type'],
            rotation_gate=model_params_for_worker['rotation_gate'],
            device_name=model_params_for_worker['device_name']  # Должно быть CPU-устройство для воркеров
        )
        worker_qnn_model_inference.eval()  # Устанавливаем режим оценки
        # print(f"DEBUG_INFERENCE_WORKER (PID:{pid}): Worker QNN for inference initialized and set to eval mode.")
    except Exception as e:
        print(f"ERROR_INFERENCE_WORKER (PID:{pid}): Error initializing QNN model: {e}")
        worker_qnn_model_inference = None


# --- Функция воркера для инференса одного сэмпла ---
def process_sample_for_inference(data_sample_x_np, model_state_dict_numpy):
    global worker_qnn_model_inference
    pid = os.getpid()

    if worker_qnn_model_inference is None:
        print(
            f"ERROR_INFERENCE_WORKER (PID:{pid}): worker_qnn_model_inference is None in process_sample. Returning None.")
        return None  # Возвращаем None, чтобы можно было отфильтровать в основном процессе

    try:
        # Определяем dtype из параметров уже инициализированной и загруженной модели в воркере
        model_dtype_proc = next(worker_qnn_model_inference.parameters()).dtype

        # data_sample_x_np - это уже numpy array для одного сэмпла X
        # print(f"DEBUG_INFERENCE_WORKER (PID:{pid}): Processing sample with shape {data_sample_x_np.shape}, dtype {data_sample_x_np.dtype}")

        x_sample_tensor = torch.tensor(data_sample_x_np, dtype=model_dtype_proc).unsqueeze(0)  # Добавляем batch_dim
        # print(f"DEBUG_INFERENCE_WORKER (PID:{pid}): x_sample_tensor created, shape {x_sample_tensor.shape}, dtype {x_sample_tensor.dtype}")

        with torch.no_grad():  # Убедимся, что градиенты не считаются
            prediction_tensor = worker_qnn_model_inference(x_sample_tensor)

        # print(f"DEBUG_INFERENCE_WORKER (PID:{pid}): Sample processed, prediction_tensor shape: {prediction_tensor.shape}")
        # Предсказание обычно имеет форму (1, 1) или (1,) для одного сэмпла, если выход модели скалярный.
        # flatten() сделает его (N,) если выход многомерный, или оставит (1,) -> () -> (1,)
        return prediction_tensor.cpu().numpy().flatten()  # Возвращаем numpy array предсказаний

    except Exception as e:
        print(
            f"ERROR_INFERENCE_WORKER (PID:{pid}): Error during inference for a sample: {e}. Input shape: {data_sample_x_np.shape if isinstance(data_sample_x_np, np.ndarray) else 'N/A'}")
        return None

# --- Функции для построения графиков (остаются БЕЗ ИЗМЕНЕНИЙ) ---
# ... (plot_predicted_vs_actual, plot_residuals, plot_distributions, plot_qcml_forecast_surface) ...
# (код этих функций из предыдущего ответа)
def plot_predicted_vs_actual(y_true_orig, y_pred_orig, results_dir, model_name_base, plot_suffix="_analysis"):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_orig, y_pred_orig, alpha=0.5, edgecolors='k', s=20)
    min_val = min(np.min(y_true_orig), np.min(y_pred_orig))
    max_val = max(np.max(y_true_orig), np.max(y_pred_orig))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal y=x')
    plt.xlabel("Actual Values (Original Scale)")
    plt.ylabel("Predicted Values (Original Scale)")
    plt.title(f"Predicted vs. Actual Values - {model_name_base}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{model_name_base}_predicted_vs_actual{plot_suffix}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Predicted vs. Actual plot saved to {plot_path}")


def plot_residuals(y_true_orig, y_pred_orig, results_dir, model_name_base, plot_suffix="_analysis"):
    residuals = y_true_orig.flatten() - y_pred_orig.flatten()
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_orig.flatten(), residuals, alpha=0.5, edgecolors='k', s=20)
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.xlabel("Predicted Values (Original Scale)")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"Residuals Plot - {model_name_base}")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{model_name_base}_residuals{plot_suffix}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Residuals plot saved to {plot_path}")


def plot_distributions(y_true_orig, y_pred_orig, results_dir, model_name_base, plot_suffix="_analysis"):
    plt.figure(figsize=(10, 6))
    plt.hist(y_true_orig.flatten(), bins=50, alpha=0.7, label='Actual Values', color='blue', density=True)
    plt.hist(y_pred_orig.flatten(), bins=50, alpha=0.7, label='Predicted Values', color='orange', density=True)
    plt.xlabel("Values (Original Scale)")
    plt.ylabel("Density")
    plt.title(f"Distribution of Actual vs. Predicted Values - {model_name_base}")
    plt.legend()
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{model_name_base}_distributions{plot_suffix}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Distributions plot saved to {plot_path}")


def plot_qcml_forecast_surface(model, X_data_scaled_for_surface, feature_indices, feature_names,
                               results_dir, model_name_base, n_points=20, plot_suffix="_analysis",
                               model_output_scaler=None):
    if X_data_scaled_for_surface is None or X_data_scaled_for_surface.shape[0] == 0:
        print("Warning: X_data_scaled_for_surface is None or empty. Cannot generate QCML forecast surface plot.")
        return
    if len(feature_indices) != 2 or len(feature_names) != 2:
        print(
            "Warning: feature_indices and feature_names must contain two elements. Cannot generate QCML forecast surface plot.")
        return

    idx_f1, idx_f2 = feature_indices;
    name_f1, name_f2 = feature_names
    num_total_features = X_data_scaled_for_surface.shape[1]
    print(
        f"Generating QCML forecast surface for features: '{name_f1}' (idx {idx_f1}) and '{name_f2}' (idx {idx_f2})...")
    f1_vals_train = X_data_scaled_for_surface[:, idx_f1];
    f2_vals_train = X_data_scaled_for_surface[:, idx_f2]
    f1_range = np.linspace(f1_vals_train.min(), f1_vals_train.max(), n_points)
    f2_range = np.linspace(f2_vals_train.min(), f2_vals_train.max(), n_points)
    other_features_mean = np.mean(X_data_scaled_for_surface, axis=0)
    F1, F2 = np.meshgrid(f1_range, f2_range)
    Z_preds_scaled = np.zeros_like(F1);
    Z_preds_original = np.zeros_like(F1)
    model.eval();
    target_device = next(model.parameters()).device;
    model_input_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for i in range(n_points):
            for j in range(n_points):
                input_vector = other_features_mean.copy();
                input_vector[idx_f1] = F1[i, j];
                input_vector[idx_f2] = F2[i, j]
                input_tensor = torch.tensor(input_vector, dtype=model_input_dtype).unsqueeze(0).to(target_device)
                prediction_scaled_tensor = model(input_tensor)
                prediction_scaled_val = prediction_scaled_tensor.cpu().item()
                Z_preds_scaled[i, j] = prediction_scaled_val
                if model_output_scaler:
                    try:
                        Z_preds_original[i, j] = \
                        model_output_scaler.inverse_transform(np.array([[prediction_scaled_val]]))[0, 0]
                    except Exception:
                        Z_preds_original[i, j] = prediction_scaled_val
                else:
                    Z_preds_original[i, j] = prediction_scaled_val
    fig = plt.figure(figsize=(12, 8));
    ax = fig.add_subplot(111, projection='3d')
    surf_data = Z_preds_original if model_output_scaler else Z_preds_scaled
    z_label_text = 'QCML Forecast (Original Scale)' if model_output_scaler else 'QCML Forecast (Scaled Output [-1,1])'
    surf = ax.plot_surface(F1, F2, surf_data, cmap='magma', edgecolor='none', antialiased=True)
    ax.set_xlabel(f"{name_f1} (scaled input)");
    ax.set_ylabel(f"{name_f2} (scaled input)");
    ax.set_zlabel(z_label_text)
    ax.set_title(f'QCML Forecast Surface - {model_name_base}');
    try:
        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    except Exception:
        pass
    ax.view_init(elev=20, azim=-120);
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{model_name_base}_qcml_forecast_surface{plot_suffix}.png");
    plt.savefig(plot_path);
    plt.close(fig)
    print(f"QCML forecast surface plot saved to {plot_path}")


def analyze_trained_model():
    print("--- Starting Model Analysis ---")
    results_dir = config.RESULTS_DIR if hasattr(config, 'RESULTS_DIR') else "results"
    model_name_base = config.MODEL_NAME if hasattr(config, 'MODEL_NAME') else "qnn_model"

    model_load_path = getattr(config, 'MODEL_TO_ANALYZE_PATH', os.path.join(results_dir, f"{model_name_base}_best.pth"))
    if not os.path.exists(model_load_path):
        model_load_path = os.path.join(results_dir, f"{model_name_base}_final.pth")
        if not os.path.exists(model_load_path):
            print(
                f"ERROR: Model file not found. Searched:\n- {getattr(config, 'MODEL_TO_ANALYZE_PATH', 'N/A')}\n- {os.path.join(results_dir, f'{model_name_base}_best.pth')}\n- {os.path.join(results_dir, f'{model_name_base}_final.pth')}\nExiting analysis.")
            return
    print(f"Analyzing model from: {model_load_path}")

    # --- Загрузка данных ---
    X_train_scaled_for_surface_plot = None
    # Устанавливаем BATCH_SIZE для test_loader (может быть больше, чем при обучении)
    analysis_batch_size = getattr(config, 'ANALYSIS_BATCH_SIZE', config.BATCH_SIZE)
    try:
        _, test_loader, num_features_from_data, y_scaler, feature_names, X_train_scaled_for_surface_plot = load_and_split_data(
            data_path=config.DATA_PATH, datetime_col=config.DATETIME_COLUMN, target_col=config.TARGET_COLUMN,
            other_exclude_cols=config.OTHER_EXCLUDE_COLS, test_size=config.TEST_SIZE,
            batch_size=analysis_batch_size, return_X_train_scaled=True
        )
    except TypeError:
        try:
            _, test_loader, num_features_from_data, y_scaler, feature_names = load_and_split_data(
                data_path=config.DATA_PATH, datetime_col=config.DATETIME_COLUMN, target_col=config.TARGET_COLUMN,
                other_exclude_cols=config.OTHER_EXCLUDE_COLS, test_size=config.TEST_SIZE,
                batch_size=analysis_batch_size
            )
        except Exception as e_data:
            print(f"ERROR: Failed to load data for analysis: {e_data}"); return
    except Exception as e_data_general:
        print(f"ERROR: Failed to load data (general error): {e_data_general}"); return

    # --- Загрузка конфигурации модели ---
    checkpoint = None;
    model_config_source = {}
    try:
        checkpoint = torch.load(model_load_path, map_location='cpu', weights_only=False)
        if 'config' in checkpoint:
            model_config_source = checkpoint['config']
        else:
            model_config_source = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    except Exception:
        model_config_source = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    embedding_type_cfg = model_config_source.get('EMBEDDING_TYPE', 'Angle')
    n_layers_cfg = model_config_source.get('N_LAYERS', 1)
    rotation_gate_cfg = model_config_source.get('ROTATION_GATE_EMBEDDING', 'X')
    if embedding_type_cfg == "Amplitude":
        actual_n_qubits_for_model_cfg = model_config_source.get('N_QUBITS_AMPLITUDE', 5)
        features_for_embedding_input_cfg = num_features_from_data
    elif embedding_type_cfg == "Angle":
        if model_config_source.get('USE_PCA', False) and model_config_source.get('N_COMPONENTS_PCA') is not None:
            actual_n_qubits_for_model_cfg = model_config_source.get('N_COMPONENTS_PCA')
            features_for_embedding_input_cfg = model_config_source.get('N_COMPONENTS_PCA')
        else:
            actual_n_qubits_for_model_cfg = num_features_from_data; features_for_embedding_input_cfg = num_features_from_data
    else:
        print(f"ERROR: Unsupported EMBEDDING_TYPE '{embedding_type_cfg}'."); return
    print(
        f"Model arch to load: Emb={embedding_type_cfg}, AnsatzQubits={actual_n_qubits_for_model_cfg}, EmbInFeat={features_for_embedding_input_cfg}, Layers={n_layers_cfg}")

    # --- Инициализация и загрузка модели ---
    # Устройство для воркеров инференса всегда CPU
    worker_device_for_qnn_init_inference = "default.qubit"
    num_mp_workers_inference = getattr(config, 'NUM_MP_WORKERS_INFERENCE', os.cpu_count())
    if num_mp_workers_inference is None or num_mp_workers_inference <= 0: num_mp_workers_inference = max(1,
                                                                                                         os.cpu_count() // 2 if os.cpu_count() else 1)
    print(
        f"Using {num_mp_workers_inference} worker processes for inference (workers use CPU QNN: {worker_device_for_qnn_init_inference}).")

    model_params_for_worker_inference = {
        'n_features_for_embedding': features_for_embedding_input_cfg,
        'n_qubits_for_ansatz': actual_n_qubits_for_model_cfg,
        'n_layers': n_layers_cfg,
        'embedding_type': embedding_type_cfg,
        'rotation_gate': rotation_gate_cfg,
        'device_name': worker_device_for_qnn_init_inference
    }

    # Загружаем state_dict основной модели для передачи в воркеры
    # Это state_dict будет от модели, которая была обучена (возможно на GPU)
    # но воркеры будут CPU, поэтому state_dict передается как numpy
    main_model_state_dict_numpy_for_inference = None
    if checkpoint and 'model_state_dict' in checkpoint:
        # Загружаем state_dict на CPU, затем конвертируем в numpy
        temp_model = QNN(**model_params_for_worker_inference)  # Временная модель для загрузки state_dict
        temp_model.load_state_dict(checkpoint['model_state_dict'])
        main_model_state_dict_numpy_for_inference = {name: param.cpu().detach().numpy()
                                                     for name, param in temp_model.state_dict().items()}
        print("Model state_dict prepared for inference workers.")
    else:
        print("ERROR: Could not get model_state_dict from checkpoint for inference workers.")
        return

    # --- 5. Получение предсказаний на тестовой выборке с распараллеливанием ---
    all_predictions_scaled_mp = []
    all_targets_scaled_mp = []  # Собираем таргеты для сопоставления

    total_test_batches = len(test_loader)
    inference_batch_times = deque(maxlen=20)
    processed_batches_count = 0

    print(f"Generating predictions on test set using {num_mp_workers_inference} workers...")
    start_time_inference_total = time.time()

    # Установка метода старта для multiprocessing (аналогично train.py)
    current_start_method_before_pool_an = multiprocessing.get_start_method(allow_none=True)
    desired_start_method_before_pool_an = getattr(config, 'MP_START_METHOD', 'spawn')  # Из конфига или spawn
    if current_start_method_before_pool_an is None or current_start_method_before_pool_an != desired_start_method_before_pool_an:
        try:
            multiprocessing.set_start_method(desired_start_method_before_pool_an, force=True)
        except RuntimeError:
            pass
    mp_context_inference = multiprocessing.get_context(multiprocessing.get_start_method(allow_none=True))

    with mp_context_inference.Pool(processes=num_mp_workers_inference,
                                   initializer=init_worker_inference,
                                   initargs=(model_params_for_worker_inference,)) as pool:
        for batch_idx, (X_batch_torch_cpu, y_batch_torch_cpu) in enumerate(test_loader):
            start_time_batch_inference = time.time()

            batch_data_x_for_pool = [X_batch_torch_cpu[i].numpy() for i in range(X_batch_torch_cpu.size(0))]

            process_func_inference = partial(process_sample_for_inference,
                                             model_state_dict_numpy=main_model_state_dict_numpy_for_inference)

            batch_predictions_list = pool.map(process_func_inference, batch_data_x_for_pool)

            # Отфильтровываем None результаты (если были ошибки в воркерах)
            valid_batch_predictions = [pred for pred in batch_predictions_list if pred is not None]

            if valid_batch_predictions:
                all_predictions_scaled_mp.extend(np.concatenate(valid_batch_predictions, axis=0).flatten())
                # Сохраняем соответствующие таргеты
                # Важно: нужно убедиться, что порядок таргетов соответствует предсказаниям
                # Если какие-то сэмплы в batch_predictions_list были None, их таргеты нужно пропустить
                corresponding_targets = []
                for i, pred_res in enumerate(batch_predictions_list):
                    if pred_res is not None:
                        corresponding_targets.append(y_batch_torch_cpu[i].numpy())
                if corresponding_targets:  # Если есть валидные таргеты
                    all_targets_scaled_mp.extend(np.array(corresponding_targets).flatten())

            processed_batches_count += 1
            end_time_batch_inference = time.time()
            current_batch_inference_time = end_time_batch_inference - start_time_batch_inference
            inference_batch_times.append(
                current_batch_inference_time if current_batch_inference_time > 0.001 else 0.001)

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or (batch_idx + 1) == total_test_batches:
                avg_recent_inference_time = sum(inference_batch_times) / len(inference_batch_times)
                remaining_batches_inference = total_test_batches - (batch_idx + 1)
                estimated_inference_remaining_time = remaining_batches_inference * avg_recent_inference_time
                print(f"  Inference Batch {batch_idx + 1}/{total_test_batches} "
                      f"| Batch Time: {current_batch_inference_time:.3f}s "
                      f"| Avg Last {len(inference_batch_times)}: {avg_recent_inference_time:.3f}s/batch "
                      f"| Est. Rem. Time: {estimated_inference_remaining_time:.2f}s")

    end_time_inference_total = time.time()
    total_inference_time = end_time_inference_total - start_time_inference_total
    print(
        f"Predictions generated for {len(all_predictions_scaled_mp)} test samples in {total_inference_time:.2f}s using MP.")

    # --- 6. Построение графиков ---
    if all_predictions_scaled_mp and y_scaler is not None:
        preds_np_scaled = np.array(all_predictions_scaled_mp).reshape(-1, 1)
        targets_np_scaled = np.array(all_targets_scaled_mp).reshape(-1, 1)  # Используем собранные таргеты

        # Убедимся, что размеры совпадают перед inverse_transform
        min_len = min(len(preds_np_scaled), len(targets_np_scaled))
        if min_len < len(preds_np_scaled) or min_len < len(targets_np_scaled):
            print(
                f"Warning: Trimming predictions/targets for plotting due to length mismatch after MP errors (preds: {len(preds_np_scaled)}, targets: {len(targets_np_scaled)}, using: {min_len})")
        preds_np_scaled = preds_np_scaled[:min_len]
        targets_np_scaled = targets_np_scaled[:min_len]

        if min_len == 0:
            print("No valid predictions/targets available for plotting after MP.")
            return

        try:
            preds_orig = y_scaler.inverse_transform(preds_np_scaled)
            targets_orig = y_scaler.inverse_transform(targets_np_scaled)

            print("\n--- Generating Evaluation Plots ---")
            plot_predicted_vs_actual(targets_orig, preds_orig, results_dir, model_name_base)
            plot_residuals(targets_orig, preds_orig, results_dir, model_name_base)
            plot_distributions(targets_orig, preds_orig, results_dir, model_name_base)

            if X_train_scaled_for_surface_plot is not None and feature_names is not None and len(feature_names) >= 2:
                idx_f1_3d = getattr(config, 'SURFACE_PLOT_FEATURE1_IDX', 0)
                idx_f2_3d = getattr(config, 'SURFACE_PLOT_FEATURE2_IDX', 1)
                if idx_f1_3d >= X_train_scaled_for_surface_plot.shape[1] or idx_f2_3d >= \
                        X_train_scaled_for_surface_plot.shape[1] or idx_f1_3d == idx_f2_3d:
                    print(
                        f"Warning: Feature indices for 3D plot ({idx_f1_3d}, {idx_f2_3d}) are invalid or out of bounds. Using first two valid & distinct features.")
                    idx_f1_3d = 0;
                    idx_f2_3d = 1 if X_train_scaled_for_surface_plot.shape[1] > 1 else 0
                name_f1_3d = feature_names[idx_f1_3d] if idx_f1_3d < len(feature_names) else f"Feature_{idx_f1_3d}"
                name_f2_3d = feature_names[idx_f2_3d] if idx_f2_3d < len(feature_names) else f"Feature_{idx_f2_3d}"

                # Загружаем 'чистую' модель еще раз для 3D-графика, чтобы она была на CPU для простоты
                # или используем model_to_analyze, если она уже на CPU
                model_for_surface = QNN(  # Инициализируем на CPU
                    n_features_for_embedding=features_for_embedding_input_cfg,
                    n_qubits_for_ansatz=actual_n_qubits_for_model_cfg, n_layers=n_layers_cfg,
                    embedding_type=embedding_type_cfg, rotation_gate=rotation_gate_cfg,
                    device_name="default.qubit"
                )
                if checkpoint and 'model_state_dict' in checkpoint:
                    model_for_surface.load_state_dict(checkpoint['model_state_dict'])
                model_for_surface.eval()

                plot_qcml_forecast_surface(model_for_surface, X_train_scaled_for_surface_plot,
                                           feature_indices=(idx_f1_3d, idx_f2_3d),
                                           feature_names=(name_f1_3d, name_f2_3d),
                                           results_dir=results_dir, model_name_base=model_name_base,
                                           model_output_scaler=y_scaler)
            else:
                print(
                    "Skipping QCML forecast surface plot: X_train_scaled_for_surface_plot or feature_names not available/sufficient.")
        except Exception as e_final_plots:
            print(f"Error during final advanced plotting: {e_final_plots}")
    elif y_scaler is None:
        print("y_scaler is None. Cannot generate original scale plots.")
    else:
        print("No test predictions available to generate plots.")
    print("\n--- Model Analysis Finished ---")


if __name__ == '__main__':
    # Установка метода start для multiprocessing
    # ... (ваш код установки метода start, можно скопировать из train.py) ...
    current_start_method = multiprocessing.get_start_method(allow_none=True)
    desired_start_method = getattr(config, 'MP_START_METHOD', 'spawn')
    if current_start_method is None or current_start_method != desired_start_method:
        try:
            multiprocessing.set_start_method(desired_start_method, force=True)
        except RuntimeError:
            pass
    print(f"Analyze_model.py: MP Start method is '{multiprocessing.get_start_method(allow_none=True)}'.")

    analyze_trained_model()