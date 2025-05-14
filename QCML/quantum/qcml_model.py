import pennylane as qml
from pennylane import numpy as np
import torch
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
        self._vqe_qnode_pytorch_cached = None # Для кэширования PyTorch VQE QNode
    # --- Методы QNode ---
    # Используем декоратор qml.QNode для создания QNode при вызове метода
    # Это позволяет передать self.dev
    # Метод для QNode VQE с интерфейсом PyTorch
    def _get_vqe_qnode_pytorch(self):
        """
        Возвращает QNode для VQE с интерфейсом PyTorch.
        Использует diff_method='adjoint' для потенциально лучшей производительности с lightning.qubit.
        """
        # Кэшируем, чтобы не пересоздавать QNode без необходимости
        if self._vqe_qnode_pytorch_cached is None:
            # interface='torch': Указывает PennyLane использовать PyTorch для обработки параметров
            #                    и градиентов этого QNode.
            # diff_method='adjoint': Предпочтительный метод дифференцирования для симуляторов,
            #                        таких как lightning.qubit, из-за его эффективности.
            #                        Он вычисляет градиенты всех параметров за два прохода.
            #                        Требует проверки совместимости с qml.Hermitian в вашей версии.
            # diff_method='backprop': Альтернатива, если 'adjoint' не работает. Использует
            #                         обратное распространение ошибки через всю схему.
            @qml.qnode(self.dev, interface='torch', diff_method='backprop')
            def qnode_func(phi, theta_k):  # phi и theta_k будут тензорами PyTorch
                # _vqe_ansatz_template и _apply_U_template - это стандартные шаблоны PennyLane,
                # которые совместимы с интерфейсом PyTorch, если их параметры (weights)
                # являются тензорами PyTorch.
                self._vqe_ansatz_template(weights=phi, wires=range(self.n_qubits))
                self._apply_U_template(weights=theta_k, wires=range(self.n_qubits))
                # qml.expval(self.O2_op) и qml.expval(self.O_op) вернут тензоры PyTorch,
                # когда QNode имеет интерфейс 'torch'.
                return qml.expval(self.O2_op), qml.expval(self.O_op)

            self._vqe_qnode_pytorch_cached = qnode_func
        return self._vqe_qnode_pytorch_cached

    # Метод для QNode извлечения состояния (без изменений)
    def _get_state_vector_qnode(self):
        @qml.qnode(self.dev)
        def qnode_func(phi):
            self._vqe_ansatz_template(weights=phi, wires=range(self.n_qubits))
            return qml.state()

        return qnode_func

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
    def _get_theta_grad_qnode(self, diff_method='backprop'):
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
    def cost_for_theta_gradient(self, thetas_input, psi_fixed, xs_input, diff_method='backprop'):
        E_theta = 0.0
        if len(thetas_input) != len(xs_input) or len(thetas_input) != self.k_features: raise ValueError("Mismatch")
        grad_qnode = self._get_theta_grad_qnode(diff_method=diff_method) # Получаем QNode
        for k in range(self.k_features):
            exp_O2, exp_O = grad_qnode(psi_fixed, thetas_input[k]) # Вызов QNode
            E_theta += exp_O2 - 2 * xs_input[k] * exp_O + xs_input[k] ** 2
        return E_theta

    # Энергия VQE с использованием PyTorch тензоров и QNode
    def energy_vqe_pytorch(self, phi_torch, thetas_input_torch, xs_input_torch):
        """
        Вычисляет энергию VQE, принимая на вход PyTorch тензоры.
        phi_torch: torch.Tensor (параметры, которые оптимизируются, requires_grad=True).
        thetas_input_torch: torch.Tensor (параметры theta_k для каждого признака).
        xs_input_torch: torch.Tensor (значения признаков xs_k).
        """
        # Инициализируем энергию как тензор PyTorch с тем же типом данных и устройством,
        # что и phi_torch, для обеспечения совместимости операций и корректной работы autograd.
        # grad_fn у E будет отслеживать операции для вычисления градиента по phi_torch.
        E = torch.tensor(0.0, dtype=phi_torch.dtype, device=phi_torch.device)

        if not (len(thetas_input_torch) == len(xs_input_torch) == self.k_features):
            raise ValueError(f"Несоответствие размеров в energy_vqe_pytorch: "
                             f"len(thetas_input_torch)={len(thetas_input_torch)}, "
                             f"len(xs_input_torch)={len(xs_input_torch)}, "
                             f"self.k_features={self.k_features}")

        # Получаем PyTorch-совместимый QNode.
        qnode_pytorch = self._get_vqe_qnode_pytorch()

        for k_idx in range(self.k_features):
            # Извлекаем срезы для текущего признака k.
            # Они уже должны быть тензорами PyTorch, если a_input_features_torch и x_target_values_torch
            # являются тензорами.
            theta_k_slice = thetas_input_torch[k_idx]
            xs_k_val = xs_input_torch[k_idx]

            # Вызываем QNode. phi_torch - это тензор, по которому будут считаться градиенты.
            # theta_k_slice здесь передается как константа для данного вызова (не оптимизируется внутри VQE).
            exp_O2, exp_O = qnode_pytorch(phi_torch, theta_k_slice)

            # Все математические операции выполняются с использованием тензоров PyTorch.
            # Это гарантирует, что PyTorch сможет отслеживать граф вычислений
            # и корректно вычислить градиенты для phi_torch.
            term_cost = exp_O2 - 2 * xs_k_val * exp_O + xs_k_val.pow(2)  # Используем .pow(2) для тензора
            E = E + term_cost
        return E


    # --- Методы для состояния и предсказания ---
    def get_state_vector(self, phi):
        qnode = self._get_state_vector_qnode() # Получаем QNode
        return qnode(phi) # Вызов QNode

    def predict_target(self, phi):
        qnode = self._get_predict_qnode() # Получаем QNode
        return qnode(phi) # Вызов QNode