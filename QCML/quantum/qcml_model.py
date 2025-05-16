import pennylane as qml
# Используем ванильный NumPy для функции разложения и начальных данных
import numpy as vanilla_np
import torch
from itertools import product  # Для функции разложения


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Функция разложения диагонального оператора на сумму произведений I и Z +++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_pauli_decomposition_for_diag_operator(diag_elements_np, num_qubits, dev_for_matrix=None, debug_print=False):
    if not isinstance(diag_elements_np, vanilla_np.ndarray):
        diag_elements_np = vanilla_np.array(diag_elements_np, dtype=float)

    expected_len = 2 ** num_qubits
    if len(diag_elements_np) != expected_len:
        raise ValueError(f"Длина diag_elements_np ({len(diag_elements_np)}) "
                         f"должна быть 2**num_qubits (2**{num_qubits}={expected_len})")

    coeffs = []
    pauli_observables = []
    all_wires = list(range(num_qubits))

    # Если dev_for_matrix не передан, создадим временный default.qubit для qml.matrix,
    # так как qml.matrix может требовать активное устройство.
    # Однако, qml.matrix(Operator) обычно работает и без явного устройства, если оно просто
    # строит математическую матрицу.
    # Проверим, нужен ли он. Если да, то default.qubit достаточно.

    if debug_print: print(f"  Target diagonal D: {diag_elements_np}")

    for pauli_config_bits in product([0, 1], repeat=num_qubits):
        current_term_ops_list_for_hamiltonian = []
        pauli_string_repr_list_for_debug = []
        for qubit_idx in range(num_qubits):
            if pauli_config_bits[qubit_idx] == 0:
                current_term_ops_list_for_hamiltonian.append(qml.Identity(qubit_idx))
                pauli_string_repr_list_for_debug.append(f"I({qubit_idx})")
            else:
                current_term_ops_list_for_hamiltonian.append(qml.PauliZ(qubit_idx))
                pauli_string_repr_list_for_debug.append(f"Z({qubit_idx})")

        if num_qubits == 0: continue

        pauli_term_op_for_hamiltonian = current_term_ops_list_for_hamiltonian[0]
        if num_qubits > 1:
            for i in range(1, num_qubits):
                pauli_term_op_for_hamiltonian @= current_term_ops_list_for_hamiltonian[i]

        current_pauli_str_debug = " @ ".join(pauli_string_repr_list_for_debug)
        if debug_print: print(f"    Testing Pauli term P: {current_pauli_str_debug} (config: {pauli_config_bits})")

        # --- ИСПОЛЬЗУЕМ qml.matrix для получения diag_P ---
        try:
            # wire_order здесь должен соответствовать тому, как упорядочены кубиты в pauli_config_bits
            # и как state_idx интерпретируется (0-й бит state_idx -> wires[0])
            matrix_P_s_complex = qml.matrix(pauli_term_op_for_hamiltonian, wire_order=all_wires)
            diag_P = vanilla_np.diag(matrix_P_s_complex.real)
        except Exception as e_mat_p:
            print(f"      ERROR: Could not compute matrix for {pauli_term_op_for_hamiltonian}: {e_mat_p}")
            # Fallback на ручное вычисление, если qml.matrix не сработал (но он должен)
            # Однако, если qml.matrix не работает, то и H_reconstructed.matrix тоже не сработает.
            # Это указывает на более глубокую проблему, если qml.matrix падает.
            # Для теста пока оставим ручное, если qml.matrix падает.
            diag_P = vanilla_np.ones(expected_len, dtype=float)  # Заглушка в случае ошибки
            for state_idx in range(expected_len):
                eigenvalue_P_for_state = 1.0
                for wire_idx in range(num_qubits):
                    if pauli_config_bits[wire_idx] == 1:
                        state_of_wire_idx = (state_idx >> wire_idx) & 1
                        if state_of_wire_idx == 1:
                            eigenvalue_P_for_state *= -1.0
                diag_P[state_idx] = eigenvalue_P_for_state
            if debug_print: print(f"      Used fallback manual diag_P calculation.")

        if debug_print: print(f"      diag_P (from qml.matrix or fallback): {diag_P}")

        coeff = (1.0 / expected_len) * vanilla_np.sum(diag_elements_np * diag_P)
        if debug_print: print(
            f"      Calculated coeff = (1/{expected_len}) * sum({diag_elements_np} * {diag_P}) = {coeff:.4f}")

        if abs(coeff) > 1e-9:
            if debug_print: print(f"      ADDING TERM: coeff={coeff:.4f}, op={pauli_term_op_for_hamiltonian}")
            coeffs.append(coeff)
            pauli_observables.append(pauli_term_op_for_hamiltonian)

    if not coeffs and num_qubits > 0:
        coeffs.append(0.0)
        id_op = qml.Identity(all_wires[0]) if all_wires else qml.Identity(0)
        if num_qubits > 1:
            for i in range(1, len(all_wires)): id_op @= qml.Identity(all_wires[i])
        pauli_observables.append(id_op)

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