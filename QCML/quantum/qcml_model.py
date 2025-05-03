import pennylane as qml
from pennylane import numpy as np

class QCMLModel:
    """
    QCML модель, где только ВХОДНЫЕ признаки участвуют в обучении H.
    Предсказание использует входные признаки для поиска состояния и отдельный A_target.
    """
    def __init__(self, n_qubits, k_input_features, n_layers_vqe, l_u_layers, dev):
        # k_input_features = K' - число только ВХОДНЫХ признаков
        self.n_qubits = n_qubits
        self.k_features = k_input_features # Число операторов Ak = числу входных признаков
        self.n_layers_vqe = n_layers_vqe
        self.l_u_layers = l_u_layers
        self.dev = dev

        # --- 1. Базовый оператор O_base ---
        dim = 2**self.n_qubits
        vals = np.arange(dim, dtype=float)
        vals = 2 * (vals - vals.mean()) / (vals.max() - vals.min() + 1e-9)
        self._o_base_matrix = np.diag(vals)
        self._o2_base_matrix = np.diag(vals**2)
        self.O_op  = qml.Hermitian(self._o_base_matrix, wires=range(self.n_qubits))
        self.O2_op = qml.Hermitian(self._o2_base_matrix, wires=range(self.n_qubits))
        # Отдельный целевой оператор для предсказания
        self.A_target_op = qml.PauliZ(0) # Или другой фиксированный оператор

        print("QCMLModel Initialized (Target EXCLUDED from Training H):")
        print(f"  Qubits: {self.n_qubits}, Input Features/Operators (K'): {self.k_features}")
        print(f"  VQE Layers: {self.n_layers_vqe}, U Layers: {self.l_u_layers}")
        print(f"  Device: {self.dev.name}")
        print(f"  Prediction Target Operator: {self.A_target_op}")

        # Формы параметров
        self.phi_shape = qml.templates.StronglyEntanglingLayers.shape(
            n_layers=self.n_layers_vqe, n_wires=self.n_qubits
        )
        self.theta_k_shape = qml.templates.StronglyEntanglingLayers.shape(
            n_layers=self.l_u_layers, n_wires=self.n_qubits
        )
        # Thetas для K' операторов (по числу входных признаков)
        self.thetas_shape = (self.k_features,) + self.theta_k_shape

        self._vqe_ansatz = qml.templates.StronglyEntanglingLayers
        self._apply_U = qml.templates.StronglyEntanglingLayers

        # --- QNodes ---
        # QNode для VQE и градиента theta (использует K' операторов)
        @qml.qnode(self.dev)
        def _exp_vals_for_A_generic(phi, theta_k):
            self._vqe_ansatz(weights=phi, wires=range(self.n_qubits))
            self._apply_U(weights=theta_k, wires=range(self.n_qubits))
            return qml.expval(self.O2_op), qml.expval(self.O_op)
        self._exp_vals_for_A_generic = _exp_vals_for_A_generic

        @qml.qnode(self.dev)
        def _get_state_vector(phi):
            self._vqe_ansatz(weights=phi, wires=range(self.n_qubits))
            return qml.state()
        self._get_state_vector = _get_state_vector

        @qml.qnode(self.dev, diff_method="adjoint")
        def _exp_vals_for_A_theta_grad_prep(psi_vector, theta_k):
            qml.StatePrep(psi_vector, wires=range(self.n_qubits))
            self._apply_U(weights=theta_k, wires=range(self.n_qubits))
            return qml.expval(self.O2_op), qml.expval(self.O_op)
        self._exp_vals_for_A_theta_grad_prep = _exp_vals_for_A_theta_grad_prep

        @qml.qnode(self.dev)
        def _predict_target_value(phi):
            self._vqe_ansatz(weights=phi, wires=range(self.n_qubits))
            return qml.expval(self.A_target_op) # Измеряем целевой оператор
        self._predict_target_value = _predict_target_value

    # --- Энергия VQE (использует K' входных признаков/операторов) ---
    def energy_vqe(self, phi, thetas_input, xs_input):
        """
        Расчет энергии VQE только для ВХОДНЫХ признаков/операторов (K' штук).
        """
        E = 0.0
        if len(thetas_input) != len(xs_input) or len(thetas_input) != self.k_features:
             raise ValueError("Mismatch in lengths for energy_vqe (should use k_features)")
        for k_idx in range(self.k_features): # Итерация по K'
            theta_k = thetas_input[k_idx]
            x_k = xs_input[k_idx]
            exp_O2, exp_O = self._exp_vals_for_A_generic(phi, theta_k)
            E += exp_O2 - 2 * x_k * exp_O + x_k ** 2
        return E

    # --- Функция стоимости для градиента по thetas (использует K' опер.)---
    def cost_for_theta_gradient(self, thetas_input, psi_fixed, xs_input):
        """
        Расчет стоимости E_t = <H> для градиента по theta (K' штук).
        Использует ВХОДНЫЕ признаки xs_input (K' штук).
        """
        E_theta = 0.0
        if len(thetas_input) != len(xs_input) or len(thetas_input) != self.k_features:
             raise ValueError("Mismatch in lengths for cost_for_theta_gradient (should use k_features)")

        for k in range(self.k_features): # Итерация по K'
            exp_O2, exp_O = self._exp_vals_for_A_theta_grad_prep(psi_fixed, thetas_input[k])
            E_theta += exp_O2 - 2 * xs_input[k] * exp_O + xs_input[k] ** 2
        return E_theta

    # --- Методы для состояния и предсказания (без изменений) ---
    def get_state_vector(self, phi):
        return self._get_state_vector(phi)

    def predict_target(self, phi):
        return self._predict_target_value(phi)