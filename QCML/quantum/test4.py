import pennylane as qml
from pennylane import numpy as np
import time

# --- Настройки (как в вашем коде) ---
n = 5                # число кубитов
K = 4                # число признаков (любой K)
n_layers_vqe = 3     # число слоёв в ansatz-е VQE
L_U = 1              # число слоёв в U для каждого A_k
dev = qml.device("lightning.qubit", wires=n)
print(f"Device: {dev.name}, Qubits: {n}, Features: {K}, VQE Layers: {n_layers_vqe}, U Layers: {L_U}")

# --- 1. Базовый оператор O_base (как в вашем коде) ---
dim = 2**n
vals = np.arange(dim, dtype=float)
# Нормируем спектр в диапазон [-1,1] - ВАЖНО для стабильности
vals = 2 * (vals - vals.mean()) / (vals.max() - vals.min() + 1e-9) # Добавим epsilon для избежания деления на ноль, если все vals одинаковые
O_base = np.diag(vals)
O2_base = np.diag(vals**2) # O_base^2

# Квантовые объекты для измерений
O_op  = qml.Hermitian(O_base, wires=range(n))
O2_op = qml.Hermitian(O2_base, wires=range(n))
# I_op  = qml.Identity(wires=0) # Не используется явно в расчетах энергии

print("O_base и O2_base созданы.")
print(f"  Spectrum O_base: min={vals.min():.2f}, max={vals.max():.2f}")
print(f"  Spectrum O2_base: min={vals.min()**2:.2f}, max={vals.max()**2:.2f}")

# --- 2. Функция apply_U (как в вашем коде) ---
def apply_U(theta_k):
    # weights=theta_k ожидает форму (L_U, n, 3)
    qml.templates.StronglyEntanglingLayers(weights=theta_k, wires=range(n))

# --- 3. QNode для одного оператора A_k и заданного phi (для VQE) ---
# Этот QNode нужен для вычисления энергии VQE
@qml.qnode(dev) # Используем device по умолчанию для VQE
def exp_vals_for_A_vqe(phi, theta_k):
    qml.templates.StronglyEntanglingLayers(weights=phi, wires=range(n))
    apply_U(theta_k)
    # Возвращаем <A_k^2> и <A_k>
    return qml.expval(O2_op), qml.expval(O_op)

# --- 4. Энергия VQE (как в вашем коде) ---
def energy_vqe(phi, thetas_fixed, xs_fixed):
    E = 0.0
    for k in range(K):
        # Вызываем QNode с текущими phi и ФИКСИРОВАННЫМИ thetas[k]
        exp_O2, exp_O = exp_vals_for_A_vqe(phi, thetas_fixed[k])
        E += exp_O2 - 2 * xs_fixed[k] * exp_O + xs_fixed[k] ** 2
    return E

# --- 5. VQE-фаза: Оптимизация phi ---
# Обернем в функцию для использования в цикле обучения
def find_ground_state_vqe(thetas_current, xs_current, initial_phi, vqe_steps=300, vqe_lr=0.05, vqe_tol=1e-5):
    """Находит оптимальные параметры phi для VQE."""
    print(f"  Starting VQE optimization for {vqe_steps} steps (lr={vqe_lr})...")
    phi = np.copy(initial_phi) # Работаем с копией
    opt_phi = qml.AdamOptimizer(stepsize=vqe_lr)
    energy_prev = np.inf

    for i in range(vqe_steps):
        # Шаг оптимизации: передаем ФИКСИРОВАННЫЕ thetas_current и xs_current
        phi, energy = opt_phi.step_and_cost(lambda v: energy_vqe(v, thetas_current, xs_current), phi)

        if i % 25 == 0 or i == vqe_steps - 1:
            print(f"    VQE iter {i:>3}: energy = {energy:.6f}")

        # Простая проверка сходимости
        if np.abs(energy - energy_prev) < vqe_tol:
            print(f"    VQE converged at step {i}, energy = {energy:.6f}")
            break

        if i % 2 == 0: energy_prev = energy

    print(f"  VQE finished. Final energy = {energy:.6f}")
    return phi # Возвращаем оптимальные параметры phi

# --- 6. QNode для извлечения состояния psi_t (как в вашем коде) ---
# Этот QNode необязателен, если StatePrep умеет работать с параметрами напрямую,
# но извлечение вектора состояния более явно соответствует статье.
# ВНИМАНИЕ: Может быть затратным по памяти для большого n!
@qml.qnode(dev)
def get_state_vector(phi):
    qml.templates.StronglyEntanglingLayers(weights=phi, wires=range(n))
    # Возвращает вектор состояния высокой точности
    return qml.state()

# --- 7. QNode для оценки A_k на фиксированном psi (для градиента по theta) ---
# ВАЖНО: Убран некорректный gradient_kwargs. Используем adjoint.
@qml.qnode(dev, diff_method="adjoint")
def exp_vals_for_A_theta_grad(psi_vector, theta_k):
    # Готовим найденное состояние psi с помощью StatePrep
    qml.StatePrep(psi_vector, wires=range(n))
    # Применяем U(theta_k) - по параметрам theta_k будет считаться градиент
    apply_U(theta_k)
    # Возвращаем <A_k^2> и <A_k>
    return qml.expval(O2_op), qml.expval(O_op)

# --- 8. Функция стоимости для градиента по thetas ---
# Эта функция будет дифференцироваться по своему первому аргументу `thetas_var`
def cost_for_theta_gradient(thetas_var, psi_fixed, xs_fixed):
    E_theta = 0.0
    # Используем psi_fixed, который не зависит от thetas_var в этой функции
    for k in range(K):
        # Вызываем QNode, который будет дифференцироваться по theta_k = thetas_var[k]
        exp_O2, exp_O = exp_vals_for_A_theta_grad(psi_fixed, thetas_var[k])
        E_theta += exp_O2 - 2 * xs_fixed[k] * exp_O + xs_fixed[k] ** 2
    return E_theta

# --- Основной Цикл Обучения ---

def train_qcml(X_train, n_epochs, initial_thetas, initial_phi_guess,
               vqe_steps=50, vqe_lr=0.05, theta_lr=0.01):
    """
    Полный цикл обучения QCML модели.

    Args:
        X_train (list or np.array): Обучающий датасет, где каждый элемент - вектор xs_t размерности K.
        n_epochs (int): Количество эпох обучения.
        initial_thetas (np.array): Начальные параметры для операторов A_k.
        initial_phi_guess (np.array): Начальное предположение для параметров VQE (может переиспользоваться).
        vqe_steps (int): Количество шагов в VQE оптимизации на каждой итерации.
        vqe_lr (float): Скорость обучения для VQE.
        theta_lr (float): Скорость обучения для параметров theta_k.

    Returns:
        np.array: Обученные параметры thetas.
    """
    thetas = np.copy(initial_thetas)
    phi_guess = np.copy(initial_phi_guess) # Начальное состояние для VQE

    # Оптимизатор для параметров theta_k
    opt_theta = qml.AdamOptimizer(stepsize=theta_lr)

    # Функция градиента для theta (дифференцируем cost_for_theta_gradient по первому аргументу)
    theta_grad_fn = qml.grad(cost_for_theta_gradient, argnum=0)

    print("\n--- Starting QCML Training ---")
    start_time_total = time.time()

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        epoch_start_time = time.time()
        total_cost_epoch = 0

        # Итерация по данным (можно добавить батчинг при необходимости)
        for i, xs_t in enumerate(X_train):
            iter_start_time = time.time()
            print(f"  Training Step {i+1}/{len(X_train)}")

            # --- VQE Фаза ---
            # Находим оптимальное phi_opt для текущих thetas и xs_t
            # В качестве начального phi можно использовать phi_guess с предыдущего шага
            # или всегда начинать с initial_phi_guess
            phi_opt = find_ground_state_vqe(thetas, xs_t, phi_guess, vqe_steps=vqe_steps, vqe_lr=vqe_lr)
            phi_guess = phi_opt # Используем найденное состояние как начальное для следующего VQE

            # --- Theta Update Фаза ---
            # 1. Получаем вектор состояния psi_t
            print("    Extracting state vector psi_t...")
            psi_t = get_state_vector(phi_opt)
            # Проверка типа и формы (опционально)
            # print(f"    State vector type: {type(psi_t)}, shape: {psi_t.shape}, dtype: {psi_t.dtype}")

            # 2. Вычисляем градиент dE_t / d(theta_k)
            #    Передаем ФИКСИРОВАННЫЙ psi_t и xs_t
            print("    Calculating gradient w.r.t. thetas...")
            grad_theta = theta_grad_fn(thetas, psi_t, xs_t) # Вычисляем градиент по thetas

            # 3. Обновляем thetas с помощью оптимизатора
            thetas = opt_theta.apply_grad(grad_theta, [thetas])[0]
            # Или можно так: thetas, _ = opt_theta.step_and_cost(lambda v: cost_for_theta_gradient(v, psi_t, xs_t), thetas) # Оптимизатор сам вычислит градиент

            # Логирование стоимости (опционально, может замедлить)
            current_cost = cost_for_theta_gradient(thetas, psi_t, xs_t)
            total_cost_epoch += current_cost
            iter_end_time = time.time()
            print(f"    Theta updated. Current Cost E_t = {current_cost:.6f}. Step time: {iter_end_time - iter_start_time:.2f}s")


        epoch_end_time = time.time()
        avg_cost_epoch = total_cost_epoch / len(X_train)
        print(f"Epoch {epoch+1} finished. Average Cost = {avg_cost_epoch:.6f}. Epoch time: {epoch_end_time - epoch_start_time:.2f}s")

    end_time_total = time.time()
    print(f"\n--- Training Finished ---")
    print(f"Total training time: {end_time_total - start_time_total:.2f}s")

    return thetas

# --- Пример Запуска Обучения ---

# 1. Генерируем случайные обучающие данные (замените на реальные)
num_samples = 10 # Небольшое количество для примера
X_train_data = [np.random.normal(0, 1, size=(K,)) for _ in range(num_samples)]

# 2. Инициализируем параметры
shape_phi = qml.templates.StronglyEntanglingLayers.shape(n_layers=n_layers_vqe, n_wires=n)
initial_phi = np.random.normal(0, 0.01, size=shape_phi)

shape_thetas = (K, L_U, n, 3) # Форма для весов StronglyEntanglingLayers
initial_thetas = np.random.normal(0, 0.1, size=shape_thetas)

# 3. Запускаем обучение
trained_thetas = train_qcml(
    X_train=X_train_data,
    n_epochs=2,         # Небольшое количество эпох для примера
    initial_thetas=initial_thetas,
    initial_phi_guess=initial_phi,
    vqe_steps=20,       # Уменьшим шаги VQE для скорости примера
    vqe_lr=0.05,
    theta_lr=0.02
)

print("\nОбученные параметры thetas:")
print(trained_thetas.shape)
# Можно сохранить trained_thetas для дальнейшего использования