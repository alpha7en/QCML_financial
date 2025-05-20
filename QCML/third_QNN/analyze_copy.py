# analyze_model.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import time
from collections import deque # Возвращаем для скользящего среднего времени обработки чанка
import multiprocessing
# from functools import partial # Не используется

import config
from data_loader import load_and_split_data
from quantum_model import QNN

# --- Глобальные переменные для воркера инференса ---
worker_qnn_model_inference = None

# --- Инициализатор для воркера инференса ---
def init_worker_inference(model_params_for_worker, model_state_dict_numpy_for_worker):
    global worker_qnn_model_inference
    pid = os.getpid()
    # print(f"DEBUG_INFERENCE_WORKER (PID:{pid}): Initializing. Params: {model_params_for_worker}") # Можно раскомментировать для детального дебага

    try:
        worker_qnn_model_inference = QNN(
            n_features_for_embedding=model_params_for_worker['n_features_for_embedding'],
            n_qubits_for_ansatz=model_params_for_worker['n_qubits_for_ansatz'],
            n_layers=model_params_for_worker['n_layers'],
            embedding_type=model_params_for_worker['embedding_type'],
            rotation_gate=model_params_for_worker['rotation_gate'],
            device_name=model_params_for_worker['device_name']
        )

        if model_state_dict_numpy_for_worker:
            state_dict_torch = {
                name: torch.tensor(param_np)
                for name, param_np in model_state_dict_numpy_for_worker.items()
            }
            worker_qnn_model_inference.load_state_dict(state_dict_torch)
            # print(f"DEBUG_INFERENCE_WORKER (PID:{pid}): Model state_dict loaded successfully.")
        else:
            print(f"WARNING_INFERENCE_WORKER (PID:{pid}): No model_state_dict provided. Model weights are uninitialized.")

        worker_qnn_model_inference.eval()
        # print(f"DEBUG_INFERENCE_WORKER (PID:{pid}): Worker QNN initialized and set to eval mode.")
    except Exception as e:
        print(f"ERROR_INFERENCE_WORKER (PID:{pid}): Error initializing/loading QNN model: {e}")
        worker_qnn_model_inference = None


# --- Функция воркера для инференса ЧАНКА данных ---
def process_data_chunk_for_inference(data_chunk_x_np):
    global worker_qnn_model_inference
    pid = os.getpid()

    if worker_qnn_model_inference is None:
        print(f"ERROR_INFERENCE_WORKER (PID:{pid}): worker_qnn_model_inference is None. Returning None.")
        return None

    if not isinstance(data_chunk_x_np, np.ndarray) or data_chunk_x_np.size == 0:
        return np.array([]) 

    try:
        model_dtype_proc = next(worker_qnn_model_inference.parameters()).dtype
        
        expected_numpy_dtype = np.float32 if model_dtype_proc == torch.float32 else np.float64
        if data_chunk_x_np.dtype != expected_numpy_dtype:
             data_chunk_x_np_typed = np.asarray(data_chunk_x_np, dtype=expected_numpy_dtype)
        else:
             data_chunk_x_np_typed = data_chunk_x_np

        x_chunk_tensor = torch.tensor(data_chunk_x_np_typed, dtype=model_dtype_proc)
        
        with torch.no_grad():
            predictions_tensor = worker_qnn_model_inference(x_chunk_tensor)

        return predictions_tensor.cpu().numpy()

    except Exception as e:
        print(
            f"ERROR_INFERENCE_WORKER (PID:{pid}): Error during inference for a chunk: {e}. Input chunk shape: {data_chunk_x_np.shape}")
        return None

# --- Функции для построения графиков (остаются БЕЗ ИЗМЕНЕНИЙ) ---
def plot_predicted_vs_actual(y_true_orig, y_pred_orig, results_dir, model_name_base, plot_suffix="_analysis"):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_orig, y_pred_orig, alpha=0.5, edgecolors='k', s=20)
    if y_true_orig.size > 0 and y_pred_orig.size > 0: # Защита от пустых массивов
        min_val = min(np.min(y_true_orig), np.min(y_pred_orig))
        max_val = max(np.max(y_true_orig), np.max(y_pred_orig))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal y=x')
    else:
        plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Ideal y=x (no data)') # Плейсхолдер
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
    if y_true_orig.size == 0 or y_pred_orig.size == 0: # Защита
        print("Skipping residuals plot: no data.")
        return
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
    if y_true_orig.size == 0 and y_pred_orig.size == 0: # Защита
        print("Skipping distributions plot: no data.")
        return
    plt.figure(figsize=(10, 6))
    if y_true_orig.size > 0:
        plt.hist(y_true_orig.flatten(), bins=50, alpha=0.7, label='Actual Values', color='blue', density=True)
    if y_pred_orig.size > 0:
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
        print("Warning: feature_indices and feature_names must contain two elements. Cannot generate QCML forecast surface plot.")
        return

    idx_f1, idx_f2 = feature_indices
    name_f1, name_f2 = feature_names
    print(f"Generating QCML forecast surface for features: '{name_f1}' (idx {idx_f1}) and '{name_f2}' (idx {idx_f2})...")
    f1_vals_train = X_data_scaled_for_surface[:, idx_f1]
    f2_vals_train = X_data_scaled_for_surface[:, idx_f2]
    f1_range = np.linspace(f1_vals_train.min(), f1_vals_train.max(), n_points)
    f2_range = np.linspace(f2_vals_train.min(), f2_vals_train.max(), n_points)
    other_features_mean = np.mean(X_data_scaled_for_surface, axis=0)
    F1, F2 = np.meshgrid(f1_range, f2_range)
    Z_preds_scaled = np.zeros_like(F1)
    Z_preds_original = np.zeros_like(F1)
    model.eval()
    target_device = next(model.parameters()).device
    model_input_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for i in range(n_points):
            for j in range(n_points):
                input_vector = other_features_mean.copy()
                input_vector[idx_f1] = F1[i, j]
                input_vector[idx_f2] = F2[i, j]
                input_tensor = torch.tensor(input_vector, dtype=model_input_dtype).unsqueeze(0).to(target_device)
                prediction_scaled_tensor = model(input_tensor)
                prediction_scaled_val = prediction_scaled_tensor.cpu().item()
                Z_preds_scaled[i, j] = prediction_scaled_val
                if model_output_scaler:
                    try:
                        Z_preds_original[i, j] = model_output_scaler.inverse_transform(np.array([[prediction_scaled_val]]))[0, 0]
                    except Exception:
                        Z_preds_original[i, j] = prediction_scaled_val
                else:
                    Z_preds_original[i, j] = prediction_scaled_val
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf_data = Z_preds_original if model_output_scaler else Z_preds_scaled
    z_label_text = 'QCML Forecast (Original Scale)' if model_output_scaler else 'QCML Forecast (Scaled Output [-1,1])'
    surf = ax.plot_surface(F1, F2, surf_data, cmap='magma', edgecolor='none', antialiased=True)
    ax.set_xlabel(f"{name_f1} (scaled input)")
    ax.set_ylabel(f"{name_f2} (scaled input)")
    ax.set_zlabel(z_label_text)
    ax.set_title(f'QCML Forecast Surface - {model_name_base}')
    try:
        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    except Exception:
        pass
    ax.view_init(elev=20, azim=-120)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{model_name_base}_qcml_forecast_surface{plot_suffix}.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"QCML forecast surface plot saved to {plot_path}")


def analyze_trained_model():
    print("--- Starting Model Analysis ---")
    # ... (Код загрузки модели и конфигурации остается таким же) ...
    results_dir = config.RESULTS_DIR if hasattr(config, 'RESULTS_DIR') else "results"
    model_name_base = config.MODEL_NAME if hasattr(config, 'MODEL_NAME') else "qnn_model"

    model_load_path = getattr(config, 'MODEL_TO_ANALYZE_PATH', os.path.join(results_dir, f"{model_name_base}_best.pth"))
    if not os.path.exists(model_load_path):
        model_load_path = os.path.join(results_dir, f"{model_name_base}_final.pth")
        if not os.path.exists(model_load_path):
            print(f"ERROR: Model file not found. Searched paths were not valid.\nExiting analysis.")
            return
    print(f"Analyzing model from: {model_load_path}")

    X_train_scaled_for_surface_plot = None
    analysis_batch_size = getattr(config, 'ANALYSIS_BATCH_SIZE', config.BATCH_SIZE)
    try:
        _, test_loader, num_features_from_data, y_scaler, feature_names, X_train_scaled_for_surface_plot = load_and_split_data(
            data_path=config.DATA_PATH, datetime_col=config.DATETIME_COLUMN, target_col=config.TARGET_COLUMN,
            other_exclude_cols=config.OTHER_EXCLUDE_COLS, test_size=config.TEST_SIZE,
            batch_size=analysis_batch_size, return_X_train_scaled=True, shuffle_test=False
        )
    except TypeError:
        try:
            print("Note: Using older load_and_split_data. Assuming test data is not shuffled for consistency.")
            _, test_loader, num_features_from_data, y_scaler, feature_names = load_and_split_data(
                data_path=config.DATA_PATH, datetime_col=config.DATETIME_COLUMN, target_col=config.TARGET_COLUMN,
                other_exclude_cols=config.OTHER_EXCLUDE_COLS, test_size=config.TEST_SIZE,
                batch_size=analysis_batch_size
            )
        except Exception as e_data:
            print(f"ERROR: Failed to load data for analysis (fallback): {e_data}"); return
    except Exception as e_data_general:
        print(f"ERROR: Failed to load data (general): {e_data_general}"); return

    checkpoint = None; model_config_source = {}
    try:
        checkpoint = torch.load(model_load_path, map_location='cpu', weights_only=False)
        if 'config' in checkpoint: model_config_source = checkpoint['config']
        else: model_config_source = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    except Exception: model_config_source = {k: v for k, v in vars(config).items() if not k.startswith('__')}

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
    else: print(f"ERROR: Unsupported EMBEDDING_TYPE '{embedding_type_cfg}'."); return
    print(f"Model arch to load: Emb={embedding_type_cfg}, AnsatzQubits={actual_n_qubits_for_model_cfg}, EmbInFeat={features_for_embedding_input_cfg}, Layers={n_layers_cfg}")

    worker_device_for_qnn_init_inference = "default.qubit"
    num_mp_workers_inference = getattr(config, 'NUM_MP_WORKERS_INFERENCE', os.cpu_count())
    if num_mp_workers_inference is None or num_mp_workers_inference <= 0:
        num_mp_workers_inference = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)

    model_params_for_worker_inference = {
        'n_features_for_embedding': features_for_embedding_input_cfg,
        'n_qubits_for_ansatz': actual_n_qubits_for_model_cfg, 'n_layers': n_layers_cfg,
        'embedding_type': embedding_type_cfg, 'rotation_gate': rotation_gate_cfg,
        'device_name': worker_device_for_qnn_init_inference
    }
    main_model_state_dict_numpy_for_inference = None
    if checkpoint and 'model_state_dict' in checkpoint:
        temp_model_for_state_dict = QNN(**model_params_for_worker_inference)
        temp_model_for_state_dict.load_state_dict(checkpoint['model_state_dict'])
        main_model_state_dict_numpy_for_inference = {name: param.cpu().detach().numpy()
                                                     for name, param in temp_model_for_state_dict.state_dict().items()}
        print("Model state_dict prepared for inference workers.")
    else:
        print("ERROR: Could not get model_state_dict from checkpoint. Worker models will be uninitialized.")
        # Не выходим, но это важно


    print("Collecting all test data for chunk-based parallel processing...")
    all_X_data_list = []
    all_y_data_list = []
    for X_batch_torch_cpu, y_batch_torch_cpu in test_loader:
        all_X_data_list.append(X_batch_torch_cpu.numpy())
        all_y_data_list.append(y_batch_torch_cpu.numpy())

    all_predictions_scaled_mp = np.array([])
    all_targets_scaled_mp = np.array([]) # Инициализация

    if not all_X_data_list:
        print("WARNING: No data found in test_loader. Skipping inference.")
    else:
        all_X_data_np = np.concatenate(all_X_data_list, axis=0)
        all_y_data_np = np.concatenate(all_y_data_list, axis=0)
        print(f"Collected {all_X_data_np.shape[0]} test samples.")

        num_samples = all_X_data_np.shape[0]
        actual_num_workers_for_pool = min(num_mp_workers_inference, num_samples if num_samples > 0 else 1)
        if actual_num_workers_for_pool == 0 and num_samples > 0 : actual_num_workers_for_pool = 1

        if actual_num_workers_for_pool < num_mp_workers_inference and num_mp_workers_inference > 1:
             print(f"  (Adjusted from configured {num_mp_workers_inference} workers to {actual_num_workers_for_pool} due to sample size: {num_samples})")
        
        # Делим на чанки только если есть данные и воркеры
        data_chunks_X = []
        if num_samples > 0 and actual_num_workers_for_pool > 0:
            data_chunks_X = np.array_split(all_X_data_np, actual_num_workers_for_pool)
            data_chunks_X = [chunk for chunk in data_chunks_X if chunk.size > 0] # Убираем пустые чанки, если np.array_split их создал
            if not data_chunks_X and num_samples > 0: # Если после фильтрации не осталось чанков, но были данные
                print(f"Warning: No non-empty data chunks after splitting {num_samples} samples into {actual_num_workers_for_pool} parts. Processing sequentially if possible.")
                if num_samples > 0: data_chunks_X = [all_X_data_np] # Обработать всё одним "чанком"
                actual_num_workers_for_pool = 1 if data_chunks_X else 0


        start_time_overall_inference_processing = time.time() # Общее время инференса с этого момента

        all_predictions_scaled_mp_chunks_ordered = []
        num_total_chunks = len(data_chunks_X)
        processed_chunks_count = 0
        chunk_processing_times = deque(maxlen=max(1, num_total_chunks // 10, 5)) # Скользящее среднее для ETA, минимум 5
        last_chunk_time_stamp = time.time() # Начальная точка для первого чанка

        if num_total_chunks > 0 and actual_num_workers_for_pool > 0:
            print(f"Generating predictions for {num_samples} samples in {num_total_chunks} chunk(s) using {actual_num_workers_for_pool} worker(s)...")
            
            current_start_method_before_pool_an = multiprocessing.get_start_method(allow_none=True)
            desired_start_method_before_pool_an = getattr(config, 'MP_START_METHOD', 'spawn')
            if current_start_method_before_pool_an is None or current_start_method_before_pool_an != desired_start_method_before_pool_an:
                try: multiprocessing.set_start_method(desired_start_method_before_pool_an, force=True)
                except RuntimeError: pass
            mp_context_inference = multiprocessing.get_context(multiprocessing.get_start_method(allow_none=True))

            with mp_context_inference.Pool(processes=actual_num_workers_for_pool,
                                           initializer=init_worker_inference,
                                           initargs=(model_params_for_worker_inference, main_model_state_dict_numpy_for_inference)) as pool:
                
                results_iterator = pool.imap(process_data_chunk_for_inference, data_chunks_X)
                
                for i, result_chunk_np in enumerate(results_iterator):
                    time_chunk_processed_at = time.time()
                    processed_chunks_count += 1
                    all_predictions_scaled_mp_chunks_ordered.append(result_chunk_np)

                    current_chunk_duration = time_chunk_processed_at - last_chunk_time_stamp
                    last_chunk_time_stamp = time_chunk_processed_at
                    if current_chunk_duration > 0.0001: chunk_processing_times.append(current_chunk_duration)

                    print_interval = max(1, num_total_chunks // 20) # Каждые 5% или каждый чанк
                    if processed_chunks_count % print_interval == 0 or processed_chunks_count == num_total_chunks or (num_total_chunks <= 5 and processed_chunks_count >=1):
                        elapsed_total_inference_time = time.time() - start_time_overall_inference_processing
                        if chunk_processing_times:
                            avg_recent_chunk_time = sum(chunk_processing_times) / len(chunk_processing_times)
                            remaining_chunks = num_total_chunks - processed_chunks_count
                            estimated_remaining_time = remaining_chunks * avg_recent_chunk_time
                            print(f"  Processed chunk {processed_chunks_count}/{num_total_chunks} "
                                  f"| Last chunk: {current_chunk_duration:.3f}s "
                                  f"| Avg last {len(chunk_processing_times)}: {avg_recent_chunk_time:.3f}s/chunk "
                                  f"| Elapsed: {elapsed_total_inference_time:.2f}s "
                                  f"| Est. Rem.: {estimated_remaining_time:.2f}s")
                        else:
                            print(f"  Processed chunk {processed_chunks_count}/{num_total_chunks} "
                                  f"| Elapsed: {elapsed_total_inference_time:.2f}s (Calculating ETA...)")
        elif num_samples > 0: # Если чанков нет, но данные есть (например, num_workers = 0)
            print(f"Warning: No chunks to process with multiprocessing for {num_samples} samples. Check worker/chunk logic.")


        total_inference_duration = time.time() - start_time_overall_inference_processing

        # Сбор и обработка результатов
        valid_prediction_chunks_data = []
        for chunk_res in all_predictions_scaled_mp_chunks_ordered:
            if chunk_res is not None and chunk_res.size > 0:
                valid_prediction_chunks_data.append(chunk_res)
        
        if valid_prediction_chunks_data:
            all_predictions_scaled_mp = np.concatenate(valid_prediction_chunks_data, axis=0)
            if all_predictions_scaled_mp.ndim > 1:
                if all_predictions_scaled_mp.shape[1] == 1: all_predictions_scaled_mp = all_predictions_scaled_mp.flatten()
                else:
                    print(f"Warning: Predictions have unexpected shape {all_predictions_scaled_mp.shape}. Using only the first column.")
                    all_predictions_scaled_mp = all_predictions_scaled_mp[:, 0].flatten()

            # Реконструкция таргетов
            reconstructed_targets_list = []
            current_sample_idx_in_all_y = 0
            if data_chunks_X : # Убедимся что data_chunks_X не пустой
                for i, original_x_chunk in enumerate(data_chunks_X):
                    num_samples_in_original_x_chunk = original_x_chunk.shape[0]
                    if i < len(all_predictions_scaled_mp_chunks_ordered) and \
                       all_predictions_scaled_mp_chunks_ordered[i] is not None and \
                       all_predictions_scaled_mp_chunks_ordered[i].size > 0:
                        target_chunk = all_y_data_np[current_sample_idx_in_all_y : current_sample_idx_in_all_y + num_samples_in_original_x_chunk]
                        reconstructed_targets_list.append(target_chunk)
                    current_sample_idx_in_all_y += num_samples_in_original_x_chunk
            
            if reconstructed_targets_list:
                all_targets_scaled_mp = np.concatenate(reconstructed_targets_list, axis=0).flatten()
            else: all_targets_scaled_mp = np.array([])
            
            num_preds = len(all_predictions_scaled_mp)
            num_targets = len(all_targets_scaled_mp)
            print(f"Predictions generated for {num_preds} (targets: {num_targets}) test samples in {total_inference_duration:.2f}s using MP (chunk-based strategy).")

            if num_preds != num_targets and num_preds > 0 and num_targets > 0 : # Печатаем предупреждение только если есть и то и другое, но не совпадает
                 print(f"CRITICAL WARNING: Mismatch between number of predictions and reconstructed targets. Preds: {num_preds}, Targets: {num_targets}. Trimming to shortest length for plotting.")
                 min_len_sync = min(num_preds, num_targets)
                 all_predictions_scaled_mp = all_predictions_scaled_mp[:min_len_sync]
                 all_targets_scaled_mp = all_targets_scaled_mp[:min_len_sync]
        else:
            print("No valid predictions were generated from any worker process.")
            all_predictions_scaled_mp = np.array([])
            all_targets_scaled_mp = np.array([])


    # --- 6. Построение графиков ---
    if len(all_predictions_scaled_mp) > 0 and y_scaler is not None:
        if len(all_targets_scaled_mp) == 0:
            print("Warning: Predictions exist, but no corresponding targets found for plotting.")
        else:
            preds_np_scaled = np.array(all_predictions_scaled_mp).reshape(-1, 1)
            targets_np_scaled = np.array(all_targets_scaled_mp).reshape(-1, 1)
            
            min_len_final = min(preds_np_scaled.shape[0], targets_np_scaled.shape[0])
            if min_len_final < preds_np_scaled.shape[0] or min_len_final < targets_np_scaled.shape[0]:
                print(f"Final check: Trimming preds/targets to {min_len_final} for plotting due to length mismatch.")
            preds_np_scaled = preds_np_scaled[:min_len_final]
            targets_np_scaled = targets_np_scaled[:min_len_final]

            if preds_np_scaled.shape[0] == 0:
                print("No valid predictions/targets available for plotting after all processing.")
            else:
                try:
                    preds_orig = y_scaler.inverse_transform(preds_np_scaled)
                    targets_orig = y_scaler.inverse_transform(targets_np_scaled)

                    print("\n--- Generating Evaluation Plots ---")
                    if preds_orig.size > 0 and targets_orig.size > 0 :
                        plot_predicted_vs_actual(targets_orig, preds_orig, results_dir, model_name_base)
                        plot_residuals(targets_orig, preds_orig, results_dir, model_name_base)
                        plot_distributions(targets_orig, preds_orig, results_dir, model_name_base)
                    else:
                        print("Skipping plots as there are no valid original scale predictions/targets after inverse transform.")

                    if X_train_scaled_for_surface_plot is not None and feature_names is not None and len(feature_names) >= 2:
                        idx_f1_3d = getattr(config, 'SURFACE_PLOT_FEATURE1_IDX', 0)
                        idx_f2_3d = getattr(config, 'SURFACE_PLOT_FEATURE2_IDX', 1)
                        num_avail_features = X_train_scaled_for_surface_plot.shape[1]
                        if not (0 <= idx_f1_3d < num_avail_features and 0 <= idx_f2_3d < num_avail_features and idx_f1_3d != idx_f2_3d):
                            print(f"Warning: Feature indices for 3D plot ({idx_f1_3d}, {idx_f2_3d}) are invalid for {num_avail_features} features. Adjusting.")
                            idx_f1_3d = 0; idx_f2_3d = 1 if num_avail_features > 1 else 0
                        
                        name_f1_3d = feature_names[idx_f1_3d] if idx_f1_3d < len(feature_names) else f"Feature_{idx_f1_3d}"
                        name_f2_3d = feature_names[idx_f2_3d] if idx_f2_3d < len(feature_names) else f"Feature_{idx_f2_3d}"
                        
                        model_for_surface = QNN(
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
                        print("Skipping QCML forecast surface plot: X_train_scaled_for_surface_plot or feature_names not available/sufficient.")
                except Exception as e_final_plots:
                    print(f"Error during final advanced plotting: {e_final_plots}")
                    import traceback
                    traceback.print_exc()
    elif y_scaler is None:
        print("y_scaler is None. Cannot generate original scale plots.")
    else:
        print("No test predictions available to generate plots.")
    print("\n--- Model Analysis Finished ---")


if __name__ == '__main__':
    current_start_method = multiprocessing.get_start_method(allow_none=True)
    desired_start_method = getattr(config, 'MP_START_METHOD', 'spawn')
    if current_start_method is None or current_start_method != desired_start_method:
        try:
            multiprocessing.set_start_method(desired_start_method, force=True)
        except RuntimeError as e:
            print(f"Note: Could not set multiprocessing start method to '{desired_start_method}' (current: '{current_start_method}'). Error: {e}")
    print(f"Analyze_model.py: MP Start method is '{multiprocessing.get_start_method(allow_none=True)}'.")

    analyze_trained_model()