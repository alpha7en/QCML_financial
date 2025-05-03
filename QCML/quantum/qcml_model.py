import pennylane as qml
from pennylane import numpy as np

class QCMLModel:
    """
    QCML модель. Версия с QNodes как МЕТОДАМИ КЛАССА для совместимости с multiprocessing.
    """
    def __init__(self, n_qubits, k_input_features, n_layers_vqe, l_u_layers, dev):
        self.n_qubits = n_qubits
        self.k_features = k_input_features
        self.n_layers_vqe = n_layers_vqe
        self.l_u_layers = l_u_layers
        self.dev = dev # Сохраняем устройство

        # --- Базовый оператор O_base ---
        dim = 2**self.n_qubits; vals = np.arange(dim, dtype=float)
        vals = 2 * (vals - vals.mean()) / (vals.max() - vals.min() + 1e-9)
        self._o_base_matrix = np.diag(vals); self._o2_base_matrix = np.diag(vals**2)
        self.O_op  = qml.Hermitian(self._o_base_matrix, wires=range(self.n_qubits))
        self.O2_op = qml.Hermitian(self._o2_base_matrix, wires=range(self.n_qubits))
        self.A_target_op = qml.PauliZ(0)

        print("QCMLModel Initialized (QNodes as methods for multiprocessing):")
        print(f"  Qubits: {self.n_qubits}, Input Features/Operators (K'): {self.k_features}")
        # ... (остальные print) ...

        # Формы параметров
        self.phi_shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=self.n_layers_vqe, n_wires=self.n_qubits)
        self.theta_k_shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=self.l_u_layers, n_wires=self.n_qubits)
        self.thetas_shape = (self.k_features,) + self.theta_k_shape

        # Шаблоны схем
        self._vqe_ansatz_template = qml.templates.StronglyEntanglingLayers
        self._apply_U_template = qml.templates.StronglyEntanglingLayers

        # --- QNodes теперь МЕТОДЫ КЛАССА ---
        # Динамически создаваемые QNode для градиента theta и другие
        self._theta_grad_qnode_cache = {} # Кэш для созданных QNode

    # --- Методы QNode ---
    # Используем декоратор qml.QNode для создания QNode при вызове метода
    # Это позволяет передать self.dev

    # Метод для QNode VQE
    def _get_vqe_qnode(self):
        # Этот QNode не требует дифференцировки внутри VQE по параметрам схемы,
        # градиент будет считаться по phi снаружи.
        @qml.qnode(self.dev)
        def qnode_func(phi, theta_k):
            self._vqe_ansatz_template(weights=phi, wires=range(self.n_qubits))
            self._apply_U_template(weights=theta_k, wires=range(self.n_qubits))
            return qml.expval(self.O2_op), qml.expval(self.O_op)
        return qnode_func

    # Метод для QNode извлечения состояния
    def _get_state_vector_qnode(self):
        @qml.qnode(self.dev)
        def qnode_func(phi):
            self._vqe_ansatz_template(weights=phi, wires=range(self.n_qubits))
            return qml.state()
        return qnode_func

    # Метод для QNode градиента theta (кэшируем)
    def _get_theta_grad_qnode(self, diff_method="adjoint"):
        cache_key = diff_method
        if cache_key not in self._theta_grad_qnode_cache:
            @qml.qnode(self.dev, diff_method=diff_method)
            def qnode_func(psi_vector, theta_k):
                qml.StatePrep(psi_vector, wires=range(self.n_qubits))
                self._apply_U_template(weights=theta_k, wires=range(self.n_qubits))
                return qml.expval(self.O2_op), qml.expval(self.O_op)
            self._theta_grad_qnode_cache[cache_key] = qnode_func
        return self._theta_grad_qnode_cache[cache_key]

    # Метод для QNode предсказания
    def _get_predict_qnode(self):
        @qml.qnode(self.dev)
        def qnode_func(phi):
            self._vqe_ansatz_template(weights=phi, wires=range(self.n_qubits))
            return qml.expval(self.A_target_op)
        return qnode_func

    # --- Энергия VQE ---
    def energy_vqe(self, phi, thetas_input, xs_input):
        E = 0.0
        if len(thetas_input) != len(xs_input) or len(thetas_input) != self.k_features: raise ValueError("Mismatch")
        qnode = self._get_vqe_qnode() # Получаем QNode
        for k_idx in range(self.k_features):
            exp_O2, exp_O = qnode(phi, thetas_input[k_idx]) # Вызов QNode
            E += exp_O2 - 2 * xs_input[k_idx] * exp_O + xs_input[k_idx] ** 2
        return E

    # --- Функция стоимости для градиента по thetas ---
    def cost_for_theta_gradient(self, thetas_input, psi_fixed, xs_input, diff_method="adjoint"):
        E_theta = 0.0
        if len(thetas_input) != len(xs_input) or len(thetas_input) != self.k_features: raise ValueError("Mismatch")
        grad_qnode = self._get_theta_grad_qnode(diff_method=diff_method) # Получаем QNode
        for k in range(self.k_features):
            exp_O2, exp_O = grad_qnode(psi_fixed, thetas_input[k]) # Вызов QNode
            E_theta += exp_O2 - 2 * xs_input[k] * exp_O + xs_input[k] ** 2
        return E_theta

    # --- Методы для состояния и предсказания ---
    def get_state_vector(self, phi):
        qnode = self._get_state_vector_qnode() # Получаем QNode
        return qnode(phi) # Вызов QNode

    def predict_target(self, phi):
        qnode = self._get_predict_qnode() # Получаем QNode
        return qnode(phi) # Вызов QNode