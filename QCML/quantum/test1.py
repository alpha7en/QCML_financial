import pennylane as qml
from pennylane import numpy as np

# --- Настройки ---
n = 5               # число кубитов (2^5 = 32 размер гильбертова пространства)
K = 1               # число признаков (K=1 для начала)
n_layers_vqe = 3    # число слоёв в ansatz-е VQE

dev = qml.device("default.qubit", wires=n)  # квантовый симулятор

# --- 1. Базовый оператор O_base ---
# Создаём диагональную матрицу 32x32 с уникальными значениями на диагонали (0..31).
dim = 2**n

vals = np.arange(dim, dtype=float)
vals = 2*(vals - vals.mean()) / (vals.max() - vals.min())  # диапазон [-0.5,0.5]
O_base = np.diag(vals)
O2_base = np.diag(vals**2)
print(O_base)

# Объявляем квантовые операторы для измерения O и O^2.
# qml.Hermitian позволяет измерять произвольную эрмитову матрицу
O_op  = qml.Hermitian(O_base, wires=range(n))
O2_op = qml.Hermitian(O2_base, wires=range(n))

# --- 2. Параметризованный оператор A(θ) = U† O_base U ---
# Функция для применения U(θ) к квантовым проводам.
# В данной реализации используем шаблон StronglyEntanglingLayers
def apply_U(theta):
    # theta.shape = (L_U, n, 3). Здесь L_U = число слоёв в U, выберем 1 слой.
    qml.templates.StronglyEntanglingLayers(weights=theta, wires=range(n))
    # После применения этой схемы, соответствующий оператор A будет U† O U.

# Начальные параметры θ для U (random init).
# Выбираем 1 слой SL с n=5, по 3 угла на кубит => форма (1,5,3).
L_U = 1
theta = np.random.normal(0, 0.1, size=(L_U, n, 3))
# Для простоты считаем один оператор A (k=1), то есть один массив параметров theta.

# --- 3. Функция для вычисления энергии E(φ;θ) для VQE (fixed θ) ---
# Ansatz для поиска основного состояния |psi>:
@qml.qnode(dev)
def ansatz_circuit(phi, theta):
    """
    Входы:
        phi  - параметры ansatz VQE (форма: (n_layers_vqe, n, 3))
        theta - параметры для U (форма: (L_U, n, 3)), считаем константой здесь.
    Действия:
        - Применяем StronglyEntanglingLayers(phi) -> состояние |phi>
        - Применяем U(theta) (является частью измеряемого оператора)
        - Возвращаем ⟨O^2⟩ и ⟨O⟩ для финального состояния
    """
    # Ansatс VQE
    qml.templates.StronglyEntanglingLayers(weights=phi, wires=range(n))
    # Применяем U(theta)
    apply_U(theta)
    # Измеряем ожидания O^2 и O. Они соответствуют ожиданиям A^2 и A
    return qml.expval(O2_op), qml.expval(O_op)

# Стоимость VQE как комбинация этих ожиданий.
def energy_vqe(phi, theta, x):
    """
    Возвращает энергию ⟨H⟩ = ⟨A^2 - 2xA + x^2 I⟩ на состоянии |phi>⊗U.
    """
    exp_O2, exp_O = ansatz_circuit(phi, theta)
    return exp_O2 - 2 * x * exp_O + x**2

# --- 4. Поиск основного состояния (VQE) ---
# Пример входного вектора x (K=1).
x = 1.23  # любое реальное число

# Инициализируем параметры ansatz для VQE.
shape_phi = qml.templates.StronglyEntanglingLayers.shape(n_layers_vqe, n)
phi = np.random.normal(0, 0.1, size=shape_phi)

# Оптимизируем φ, минимизируя energy_vqe.
opt_phi = qml.GradientDescentOptimizer(stepsize=0.1)
max_iter = 600
for i in range(max_iter):
    # Шаг оптимизации φ при фиксированном θ
    phi, energy = opt_phi.step_and_cost(lambda v: energy_vqe(v, theta, x), phi)
    if i % 20 == 0:
        print(f"VQE iter {i}: energy = {energy:.6f}")

# После оптимизации получаем приближённое основное состояние |ψ> = |phi_opt>.
phi_opt = phi
print("Final VQE energy:", energy)

# Сохраняем вектор состояния ψ_t в массив (для вычисления градиентов по θ).
# QNode с возвращаемым состоянием qml.state() из PennyLane
@qml.qnode(dev)
def psi_state(phi):
    qml.templates.StronglyEntanglingLayers(weights=phi, wires=range(n))
    return qml.state()

psi = psi_state(phi_opt)  # комплексный вектор размерности 2^n

# --- 5. Вычисление энергии и градиента по θ при фиксированном |ψ> ---
# Используем другой QNode: готовим состояние ψ через StatePrep, потом применяем U(θ) и измеряем O^2, O.
@qml.qnode(dev)
def expectation_A(theta):
    """
    QNode: готовим фиксированное состояние psi (наш найденный ground state),
    затем применяем U(theta) и измеряем O^2 и O.
    """
    # Подготавливаем состояние |ψ> (StatePrep)
    qml.StatePrep(psi, wires=range(n))
    # Применяем U(theta)
    apply_U(theta)
    # Возвращаем ожидания
    return qml.expval(O2_op), qml.expval(O_op)

# Функция энергии H(θ) = ⟨ψ|H|ψ⟩ при фиксированном ψ и x.
def energy_A(theta, x):
    exp_O2, exp_O = expectation_A(theta)
    return exp_O2 - 2 * x * exp_O + x**2

# Значение энергии до обновления θ
E_before = energy_A(theta, x)

# Вычисляем градиенты dE/dθ при фиксированном ψ (используем встроенный grad)
grad_fun = qml.grad(energy_A)
grad_theta = grad_fun(theta, x)  # форма (L_U, n, 3)
grad_norm = np.linalg.norm(grad_theta)
print("Grad norm:", grad_norm)

# --- 6. Шаг оптимизации θ ---
# Градиентный шаг (например, шаг простого градиентного спуска или Adam).
learning_rate = 0.1
theta = theta - learning_rate * grad_theta

# Энергия после обновления θ (с тем же ψ)
E_after = energy_A(theta, x)

# Вывод результатов
print("\nПараметры θ (до шага):\n", theta + learning_rate * grad_theta)  # показать старое значение
print("Градиент dE/dθ (норма):", grad_norm)
print("E before =", E_before, "E after =", E_after)
