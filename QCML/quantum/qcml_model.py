import pennylane as qml
# Используем ванильный NumPy для функции разложения и начальных данных
import numpy as vanilla_np
import torch
from itertools import product  # Для функции разложения


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Функция разложения диагонального оператора на сумму произведений I и Z +++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_pauli_decomposition_for_diag_operator(diag_elements_np, num_qubits):
    """
    Раскладывает диагональный оператор, заданный его диагональными элементами,
    на линейную комбинацию произведений операторов Паули Z и Identity.

    Args:
        diag_elements_np (numpy.ndarray): 1D массив диагональных элементов (длина 2**num_qubits).
        num_qubits (int): Количество кубит.

    Returns:
        tuple: (coeffs, ops) для qml.Hamiltonian
               coeffs (list[float]): Список коэффициентов.
               ops (list[qml.Observable]): Список операторов (произведения I и Z).
    """
    if not isinstance(diag_elements_np, vanilla_np.ndarray):
        diag_elements_np = vanilla_np.array(diag_elements_np)

    expected_len = 2 ** num_qubits
    if len(diag_elements_np) != expected_len:
        raise ValueError(f"Длина diag_elements_np ({len(diag_elements_np)}) "
                         f"должна быть 2**num_qubits (2**{num_qubits}={expected_len})")

    coeffs = []
    pauli_observables = []

    # Итерируем по всем 2**num_qubits возможным конфигурациям Паули-строк (I или Z на каждом кубите)
    for pauli_config_bits in product([0, 1], repeat=num_qubits):
        # pauli_config_bits: кортеж из 0 и 1, где 0 -> I, 1 -> Z

        current_pauli_ops_list = []
        is_all_identity = True
        for qubit_idx in range(num_qubits):
            if pauli_config_bits[qubit_idx] == 0:  # Identity
                current_pauli_ops_list.append(qml.Identity(qubit_idx))
            else:  # PauliZ
                current_pauli_ops_list.append(qml.PauliZ(qubit_idx))
                is_all_identity = False

        # Создаем оператор произведения (например, I(0) @ Z(1) @ I(2))
        if not current_pauli_ops_list:  # Случай 0 кубит
            if num_qubits == 0 and expected_len == 1:  # Скаляр
                coeffs.append(diag_elements_np[0])
                # Для qml.Hamiltonian нужен наблюдаемый. Можно использовать Identity(0), если N=1
                # или специальный qml.Identity() без проводов, но это не наш случай.
                # Этот блок не должен выполняться для num_qubits > 0.
                # В PennyLane для скаляра лучше просто добавить его к стоимости.
                # Мы предполагаем num_qubits >= 1
                if num_qubits == 0:  # Для случая 0 кубит, если бы он был главным
                    # coeffs.append(diag_elements_np[0])
                    # pauli_observables.append(qml.Identity(0)) # Заглушка, некорректно для N=0
                    # Для N=0, это просто скаляр, не гамильтониан в обычном смысле.
                    # Пропускаем, так как наши num_qubits > 0
                    pass
                continue

        # Формируем объект Observable для произведения Паули
        # Если все Identity, то это глобальный Identity оператор
        if is_all_identity and num_qubits > 0:
            # Для N кубит, это I(0)@I(1)@...@I(N-1).
            # qml.Hamiltonian может принять qml.Identity(0) если другие члены есть,
            # или можно просто использовать qml.Identity(0) как один из членов, если он единственный.
            # PennyLane "умно" создаст оператор Identity на всех нужных кубитах.
            final_pauli_term_op = qml.Identity(0)  # PennyLane распространит это на все провода гамильтониана
            # или можно взять Identity на первом проводе, если так яснее.
            # Для ясности можно собрать полный Identity:
            # final_pauli_term_op = current_pauli_ops_list[0]
            # for i in range(1, len(current_pauli_ops_list)):
            #     final_pauli_term_op = final_pauli_term_op @ current_pauli_ops_list[i]

        elif num_qubits > 0:
            # Убираем Identity из списка, если есть другие операторы Паули,
            # так как qml.PauliZ(0) @ qml.Identity(1) === qml.PauliZ(0)
            # (если в Hamiltonian передавать список отдельных Паули на кубитах)
            # Но мы строим тензорное произведение, так что все Identity нужны.
            final_pauli_term_op = current_pauli_ops_list[0]
            for i in range(1, len(current_pauli_ops_list)):
                final_pauli_term_op = final_pauli_term_op @ current_pauli_ops_list[i]
        else:  # num_qubits == 0, уже обработано выше (пропущено)
            continue

        # Вычисляем диагональные элементы этого оператора произведения Паули P_s
        diag_Ps = vanilla_np.ones(expected_len)
        for state_idx in range(expected_len):  # Итерируем по базисным состояниям |0...0> до |1...1>
            eigenvalue = 1.0
            for qubit_j in range(num_qubits):
                if pauli_config_bits[qubit_j] == 1:  # Если это PauliZ на j-м кубите
                    s_j = (state_idx >> qubit_j) & 1  # j-й бит state_idx (справа налево)
                    if s_j == 1:  # если кубит j в состоянии |1>
                        eigenvalue *= -1.0
            diag_Ps[state_idx] = eigenvalue

        # Коэффициент c_s = (1/2^N) * Trace(D @ P_s) = (1/2^N) * sum(diag(D) * diag(P_s))
        coeff = (1.0 / (expected_len)) * vanilla_np.sum(diag_elements_np * diag_Ps)

        if abs(coeff) > 1e-10:
            coeffs.append(coeff)
            pauli_observables.append(final_pauli_term_op)

    # Если список коэффициентов пуст (например, для нулевого оператора),
    # qml.Hamiltonian([0.0], [qml.Identity(0)]) является стандартной практикой.
    if not coeffs:
        # print("Warning: Pauli decomposition resulted in all zero coefficients. Returning 0.0 * I.")
        coeffs.append(0.0)
        if num_qubits > 0:
            pauli_observables.append(qml.Identity(0))  # Для любого провода
        else:  # Скалярный случай (0 кубит), вернем Identity на "несуществующем" проводе 0 для консистентности
            # Но это не должно вызываться с num_qubits=0, если есть проверка в начале.
            # Если diag_elements_np = [c] для N=0, то coeffs=[c], ops=[qml.Identity(0)] - нужно обработать.
            # Эта функция не предназначена для N=0.
            pass  # Не должно сюда попадать для N > 0

    return coeffs, pauli_observables


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class QCMLModel:
    def __init__(self, n_qubits, k_input_features, n_layers_vqe, l_u_layers, dev):
        self.n_qubits = n_qubits
        self.k_features = k_input_features
        self.n_layers_vqe = n_layers_vqe
        self.l_u_layers = l_u_layers
        self.dev = dev

        # --- Базовый оператор O_base ---
        dim = 2 ** self.n_qubits
        # Используем vanilla_np для создания исходных данных для разложения
        vals_vanilla_np = vanilla_np.arange(dim, dtype=float)
        vals_vanilla_np = 2 * (vals_vanilla_np - vals_vanilla_np.mean()) / (
                    vals_vanilla_np.max() - vals_vanilla_np.min() + 1e-9)

        o_base_diag_vanilla_np = vals_vanilla_np
        o2_base_diag_vanilla_np = vals_vanilla_np ** 2

        print(f"Decomposing O_op for {n_qubits} qubits...")
        coeffs_O, ops_O = get_pauli_decomposition_for_diag_operator(o_base_diag_vanilla_np, n_qubits)
        print(f"Decomposing O2_op for {n_qubits} qubits...")
        coeffs_O2, ops_O2 = get_pauli_decomposition_for_diag_operator(o2_base_diag_vanilla_np, n_qubits)

        # Коэффициенты для qml.Hamiltonian.
        # PennyLane обычно может обрабатывать Python float или NumPy float для коэффициентов.
        # Преобразование в тензоры PyTorch здесь не обязательно, если они не обучаемые.
        self.O_op_hamiltonian = qml.Hamiltonian(coeffs_O, ops_O)
        self.O2_op_hamiltonian = qml.Hamiltonian(coeffs_O2, ops_O2)

        print(f"  O_op_hamiltonian: {len(coeffs_O)} terms. Example terms:")
        for i in range(min(3, len(coeffs_O))):
            print(f"    Coeff: {coeffs_O[i]}, Op: {ops_O[i]}")
        print(f"  O2_op_hamiltonian: {len(coeffs_O2)} terms. Example terms:")
        for i in range(min(3, len(coeffs_O2))):
            print(f"    Coeff: {coeffs_O2[i]}, Op: {ops_O2[i]}")

        # A_target_op - если это тоже произвольный Hermitian, его тоже нужно разложить.
        # PauliZ уже является оператором Паули.
        self.A_target_op = qml.PauliZ(0)

        print("QCMLModel Initialized (using Pauli Hamiltonians):")
        print(f"  Qubits: {self.n_qubits}, Input Features/Operators (K'): {self.k_features}")

        self.phi_shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=self.n_layers_vqe, n_wires=self.n_qubits)
        self.theta_k_shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=self.l_u_layers,
                                                                          n_wires=self.n_qubits)
        self.thetas_shape = (self.k_features,) + self.theta_k_shape

        self._vqe_ansatz_template = qml.templates.StronglyEntanglingLayers
        self._apply_U_template = qml.templates.StronglyEntanglingLayers

        self._theta_grad_qnode_cache = {}
        self._vqe_qnode_pytorch_cached = None
        self._state_vector_qnode_cached = None
        self._predict_qnode_cached = None  # Добавил для консистентности

    def _get_vqe_qnode_pytorch(self):
        if self._vqe_qnode_pytorch_cached is None:
            # Пытаемся использовать 'backprop' с qml.Hamiltonian из Паули-членов
            # Это должно быть эффективно на lightning.qubit/gpu
            @qml.qnode(self.dev, interface='torch', diff_method='adjoint')
            def qnode_func(phi, theta_k):
                self._vqe_ansatz_template(weights=phi, wires=range(self.n_qubits))
                self._apply_U_template(weights=theta_k, wires=range(self.n_qubits))
                # Измеряем гамильтонианы, разложенные на Паули
                expval_O2 = qml.expval(self.O2_op_hamiltonian)
                expval_O = qml.expval(self.O_op_hamiltonian)
                return expval_O2, expval_O

            self._vqe_qnode_pytorch_cached = qnode_func
            print(
                f"Created VQE QNode with diff_method='adjoint' (using Pauli Hamiltonians) and interface='torch' for device {self.dev.name}")
        return self._vqe_qnode_pytorch_cached

    # --- Энергия VQE с PyTorch тензорами (использует _get_vqe_qnode_pytorch) ---
    def energy_vqe_pytorch(self, phi_torch, thetas_input_torch, xs_input_torch):
        # Убедимся, что входные тензоры имеют правильный dtype, если необходимо
        # (Обычно float64 для параметров PennyLane)
        phi_torch = phi_torch.to(torch.float64)
        thetas_input_torch = thetas_input_torch.to(torch.float64)
        xs_input_torch = xs_input_torch.to(torch.float64)  # Если они приходят как другие типы

        E = torch.tensor(0.0, dtype=torch.float64, device=phi_torch.device)

        if not (len(thetas_input_torch) == len(xs_input_torch) == self.k_features):
            raise ValueError(f"Несоответствие размеров в energy_vqe_pytorch")

        qnode_pytorch = self._get_vqe_qnode_pytorch()

        for k_idx in range(self.k_features):
            theta_k_slice = thetas_input_torch[k_idx]
            xs_k_val = xs_input_torch[k_idx]
            exp_O2, exp_O = qnode_pytorch(phi_torch, theta_k_slice)

            # qml.expval(qml.Hamiltonian(...)) должен возвращать вещественный тензор PyTorch
            term_cost = exp_O2 - 2 * xs_k_val * exp_O + xs_k_val.pow(2)
            E = E + term_cost
        return E

    # --- Методы для градиента theta, состояния и предсказания ---
    # (Эти методы не менялись принципиально, но используемые O_op/O2_op теперь гамильтонианы)
    # Важно: если _get_theta_grad_qnode использует O_op/O2_op, они теперь гамильтонианы.
    # diff_method для них должен быть совместим. 'adjoint' или 'backprop' должны подойти.

    def _get_state_vector_qnode(self):
        if self._state_vector_qnode_cached is None:
            @qml.qnode(self.dev)
            def qnode_func(phi):
                self._vqe_ansatz_template(weights=phi, wires=range(self.n_qubits))
                return qml.state()

            self._state_vector_qnode_cached = qnode_func
        return self._state_vector_qnode_cached

    def _get_theta_grad_qnode(self, diff_method="adjoint"):  # 'adjoint' должен хорошо работать с Паули-гамильтонианами
        cache_key = diff_method
        if cache_key not in self._theta_grad_qnode_cache:
            @qml.qnode(self.dev, diff_method=diff_method)
            def qnode_func(psi_vector_np, theta_k_weights_np):
                qml.StatePrep(psi_vector_np, wires=range(self.n_qubits))
                self._apply_U_template(weights=theta_k_weights_np, wires=range(self.n_qubits))
                # Используем новые гамильтонианы
                expval_O2 = qml.expval(self.O2_op_hamiltonian)
                expval_O = qml.expval(self.O_op_hamiltonian)
                return expval_O2, expval_O

            self._theta_grad_qnode_cache[cache_key] = qnode_func
            print(
                f"Created Theta Grad QNode with diff_method='{diff_method}' (using Pauli Hamiltonians) for device {self.dev.name}")
        return self._theta_grad_qnode_cache[cache_key]

    def _get_predict_qnode(self):
        if self._predict_qnode_cached is None:
            @qml.qnode(self.dev)
            def qnode_func(phi):
                self._vqe_ansatz_template(weights=phi, wires=range(self.n_qubits))
                return qml.expval(self.A_target_op)  # A_target_op остался PauliZ

            self._predict_qnode_cached = qnode_func
        return self._predict_qnode_cached

    # --- Старые методы с qml.Hermitian (можно удалить или оставить для сравнения) ---
    # def energy_vqe(self, phi, thetas_input, xs_input): ...
    # def _get_vqe_qnode(self): ...
    # (Убедитесь, что все вызовы используют новые _pytorch версии и гамильтонианы там, где нужно)

    def cost_for_theta_gradient(self, current_thetas_np, psi_fixed_np, xs_input_np_sample, diff_method="adjoint"):
        E_theta = 0.0
        if not (len(current_thetas_np) == len(xs_input_np_sample) == self.k_features):
            raise ValueError(f"Несоответствие размеров в cost_for_theta_gradient")

        # qml.grad будет дифференцировать по current_thetas_np (argnum=0).
        # QNode внутри должен быть способен вычислить градиенты по своим параметрам (theta_k_weights_np)
        # чтобы qml.grad мог применить правило цепи.
        # diff_method='adjoint' для _get_theta_grad_qnode должен это обеспечить.

        # psi_fixed_np уже должен быть NumPy array.
        # current_thetas_np - (K', L_U, N, 3) NumPy array
        # xs_input_np_sample - (K',) NumPy array

        # Эта функция суммирует стоимости, а qml.grad затем дифференцирует эту сумму
        # по каждому элементу current_thetas_np.

        # ВАЖНО: qml.grad(cost_fn, argnum) ожидает, что cost_fn вернет СКАЛЯР.
        # Поэтому _get_theta_grad_qnode вызывается внутри цикла.

        # Мы не можем передать весь current_thetas_np в один QNode и ожидать градиент по нему,
        # если QNode сам по себе не построен для такой батчевой обработки параметров и возврата градиента.
        # Вместо этого, qml.grad будет многократно вызывать cost_for_theta_gradient,
        # сдвигая по одному параметру из current_thetas_np (если используется parameter-shift для qml.grad),
        # или используя другой метод, если cost_for_theta_gradient содержит QNode с другим diff_method.

        # Для ясности: grad_fn = qml.grad(self.cost_for_theta_gradient, argnum=0)
        # grad_theta = grad_fn(current_thetas_np, psi_fixed_np, xs_input_np_sample)
        # Здесь qml.grad будет управлять дифференцированием cost_for_theta_gradient.
        # Если _get_theta_grad_qnode использует 'adjoint', он эффективно посчитает градиент
        # по theta_k_weights_np для одного вызова. qml.grad использует эту информацию.

        for k in range(self.k_features):
            # Вспомогательный QNode для вычисления частной стоимости для k-го признака.
            # Этот QNode НЕ дифференцируется по current_thetas_np[k] напрямую здесь.
            # Он просто вычисляет значения, которые потом суммируются.
            # Дифференцирование по current_thetas_np[k] обеспечивается внешним qml.grad.
            # Однако, чтобы qml.grad работал эффективно (например, с adjoint),
            # QNode, используемый *внутри* дифференцируемой функции, должен поддерживать нужный diff_method.
            qnode_for_cost_k = self._get_theta_grad_qnode(diff_method=diff_method)

            theta_k_weights = current_thetas_np[k]  # (L_U, N, 3)
            xs_val_k = xs_input_np_sample[k]  # скаляр

            exp_O2, exp_O = qnode_for_cost_k(psi_fixed_np, theta_k_weights)
            E_theta += exp_O2 - 2 * xs_val_k * exp_O + xs_val_k ** 2
        return E_theta  # Возвращает скаляр (NumPy float)

    def get_state_vector(self, phi_np):
        qnode = self._get_state_vector_qnode()
        return qnode(phi_np)

    def predict_target(self, phi_np):
        qnode = self._get_predict_qnode()
        return qnode(phi_np)