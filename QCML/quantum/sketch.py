import pennylane as qml
from pennylane import numpy as np

# --- 1. Константы и Настройки ---
N_QUBITS = 5
PQC_WIRES = range(N_QUBITS)

# Гиперпараметры PQC U(θk)
N_LAYERS_PQC = 2
PARAMS_SHAPE_PER_OPERATOR = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS_PQC, n_wires=N_QUBITS)
PARAMS_PER_OPERATOR = PARAMS_SHAPE_PER_OPERATOR[0] * PARAMS_SHAPE_PER_OPERATOR[1]

# Устройство
DEV = qml.device("default.qubit", wires=N_QUBITS)

# --- 2. PQC Анзатц U(θk) ---
def pqc_ansatz(params_k, wires):
    """Параметризованная унитарная схема U(θk)."""
    params_reshaped = params_k.reshape(PARAMS_SHAPE_PER_OPERATOR)
    qml.StronglyEntanglingLayers(params_reshaped, wires=wires)

# --- 3. Выбор Базового Оператора O_base ---
# O_base должен быть объектом qml.Observable
# Пример 1: Pauli Z на первом кубите
O_base = qml.PauliZ(0)
# Пример 2: Сумма Z по всем кубитам
# O_base = qml.sum(*(qml.PauliZ(i) for i in PQC_WIRES))
# Пример 3: Более сложный, но все еще фиксированный оператор
# O_base = qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0)])
print(f"Выбран базовый оператор O_base: {O_base}")

# --- 4. QNode для вычисления ожидания <ψ| Âk(θk) |ψ> ---
# <ψ| Âk |ψ> = <ψ| U(θk)† O_base U(θk) |ψ>
# Это можно вычислить, приготовив состояние |ψ'> = U(θk)|ψ>
# и затем измерив ожидание O_base в этом состоянии: <ψ'| O_base |ψ'>

@qml.qnode(DEV, interface="autograd")
def calculate_Ak_expectation(params_k, input_state_vector):
    """
    Вычисляет <ψ| Âk(θk) |ψ>, где Âk = U(θk)† O_base U(θk).
    params_k: параметры θk для схемы U.
    input_state_vector: Вектор состояния |ψ> (например, |ψt>).
    """
    # Шаг 1: Приготовить входное состояние |ψ>
    # Используем MottonenStatePreparation или AmplitudeEmbedding, если нужно,
    # но если |ψt> уже известно как вектор, используем qml.StatePrep
    qml.StatePrep(input_state_vector, wires=PQC_WIRES)

    # Шаг 2: Применить схему U(θk)
    pqc_ansatz(params_k, wires=PQC_WIRES) # Теперь состояние |ψ'> = U(θk)|ψ>

    # Шаг 3: Измерить ожидание O_base
    return qml.expval(O_base)

# --- 5. Пример Использования ---
# Случайные параметры для оператора k
theta_k_init = np.random.uniform(0, 2 * np.pi, size=PARAMS_PER_OPERATOR, requires_grad=True)

# Некоторое входное состояние |ψ> (например, найденное |ψt>)
# Для примера возьмем |0...0>
psi_input = np.zeros(2**N_QUBITS)
psi_input[0] = 1.0
# Или случайное нормированное состояние:
# psi_input = np.random.rand(2**N_QUBITS) + 1j * np.random.rand(2**N_QUBITS)
# psi_input /= np.linalg.norm(psi_input)

# Вычисляем ожидаемое значение Âk(θk) в состоянии |ψ>
expval_Ak = calculate_Ak_expectation(theta_k_init, psi_input)

print(f"Параметры θk: {theta_k_init.shape}")
print(f"Входное состояние |ψ>: {psi_input.shape}")
print(f"Ожидаемое значение <ψ| Âk(θk) |ψ>: {expval_Ak}")

# Градиент expval_Ak по theta_k_init можно вычислить с помощью qml.grad
grad_fn_expval = qml.grad(calculate_Ak_expectation, argnum=0)
gradients_expval = grad_fn_expval(theta_k_init, psi_input)
print(f"Градиент ожидания по θk: {gradients_expval.shape}")

# --- 6. Как это использовать в QCML? ---
# Нам нужно вычислять E_t = <ψt| H |ψt> = <ψt| Σ (Âk - xt_k*I)² |ψt>
# E_t = Σ [ <ψt| Âk² |ψt> - 2*xt_k * <ψt| Âk |ψt> + xt_k² * <ψt|I|ψt> ]
# E_t = Σ [ <ψt| Âk² |ψt> - 2*xt_k * <ψt| Âk |ψt> + xt_k² ]

# Мы научились вычислять <ψt| Âk |ψt> с помощью calculate_Ak_expectation.
# Теперь нам нужно научиться вычислять <ψt| Âk² |ψt>.

# Âk² = (U† O_base U) (U† O_base U) = U† O_base² U
# Значит, <ψt| Âk² |ψt> = <ψt| U(θk)† O_base² U(θk) |ψt>

# Нужно вычислить матрицу O_base_squared = O_base @ O_base
# и создать для нее наблюдаемую qml.Observable.
O_base_matrix = qml.matrix(O_base)
O_base_squared_matrix = qml.math.matmul(O_base_matrix, O_base_matrix)
# Убедимся, что результат эрмитов (должен быть, если O_base эрмитов)
O_base_squared_matrix = 0.5 * (O_base_squared_matrix + qml.math.conj(qml.math.T(O_base_squared_matrix)))
O_base_squared_obs = qml.Hermitian(O_base_squared_matrix, wires=PQC_WIRES)

@qml.qnode(DEV, interface="autograd")
def calculate_Ak_squared_expectation(params_k, input_state_vector):
    """Вычисляет <ψ| Âk(θk)² |ψ>."""
    qml.StatePrep(input_state_vector, wires=PQC_WIRES)
    pqc_ansatz(params_k, wires=PQC_WIRES)
    # Измеряем ожидание O_base²
    return qml.expval(O_base_squared_obs)

# Теперь мы можем собрать функцию стоимости E_t и дифференцировать ее по θk
def calculate_qcml_cost_term_k(params_k, xt_k, psi_t_detached):
    """Вычисляет один член суммы для E_t: <Âk²> - 2*xt_k*<Âk> + xt_k²"""
    expval_Ak_sq = calculate_Ak_squared_expectation(params_k, psi_t_detached)
    expval_Ak_val = calculate_Ak_expectation(params_k, psi_t_detached)
    cost_k = expval_Ak_sq - 2 * xt_k * expval_Ak_val + xt_k**2
    return cost_k

# Полная стоимость E_t будет суммой cost_k по всем k
# Градиент E_t по всем theta_all = {θk} можно будет найти через autograd



# ... (определения функций calculate_Ak_expectation, calculate_Ak_squared_expectation как раньше)

# Случайные параметры и входное состояние
theta_k_init = np.random.uniform(0, 2 * np.pi, size=PARAMS_PER_OPERATOR, requires_grad=True)
psi_input = np.zeros(2**N_QUBITS)
psi_input[0] = 1.0

# Создаем функции для вычисления градиентов по theta_k_init (argnum=0)
grad_fn_expval_Ak = qml.grad(calculate_Ak_expectation, argnum=0)
grad_fn_expval_Ak_sq = qml.grad(calculate_Ak_squared_expectation, argnum=0)

# Вычисляем градиенты
gradients_Ak = grad_fn_expval_Ak(theta_k_init, psi_input)
gradients_Ak_sq = grad_fn_expval_Ak_sq(theta_k_init, psi_input)

print(f"Градиент <Âk> по θk: {gradients_Ak.shape}")
print(f"Градиент <Âk²> по θk: {gradients_Ak_sq.shape}")

# В основном цикле обучения, градиент полной E_t по всем theta_all
# будет автоматически собран из градиентов этих компонентов
# при использовании qml.grad для всей функции стоимости или через
# интерфейс ML фреймворка.