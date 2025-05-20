# quantum_model.py
import pennylane as qml
from pennylane import numpy as pnp  # Используем numpy от Pennylane для параметров, если нужно
import torch
import torch.nn as nn
import config


class QNN(nn.Module):
    def __init__(self, n_features_for_embedding, n_qubits_for_ansatz, n_layers, embedding_type="Angle", rotation_gate='X', device_name="default.qubit"):
        super().__init__()
        self.n_qubits_for_ansatz = n_qubits_for_ansatz # Например, 5
        self.n_features_for_embedding = n_features_for_embedding # Например, 18
        self.n_layers = n_layers
        self.embedding_type = embedding_type
        self.rotation_gate = rotation_gate

        self.dev = qml.device(device_name, wires=self.n_qubits_for_ansatz)

        # Веса для анзаца (StronglyEntanglingLayers) теперь зависят от n_qubits_for_ansatz
        weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits_for_ansatz)
        initial_weights_np = pnp.random.random(size=weights_shape) * 2 * pnp.pi
        initial_weights = torch.tensor(initial_weights_np, dtype=torch.get_default_dtype()) # Используем default_dtype
        self.q_weights = nn.Parameter(initial_weights)

        @qml.qnode(self.dev, interface='torch', diff_method='adjoint')  # Убедитесь, что diff_method='adjoint'
        def quantum_circuit(inputs, weights):
            if self.embedding_type == "Angle":
                # inputs должны иметь размерность (batch_size, n_qubits_for_ansatz)
                # Если n_features_for_embedding > n_qubits_for_ansatz, это не будет работать без PCA
                # Здесь предполагается, что если Angle, то n_features_for_embedding == n_qubits_for_ansatz
                qml.AngleEmbedding(inputs, wires=range(self.n_qubits_for_ansatz), rotation=self.rotation_gate)
            elif self.embedding_type == "Amplitude":
                # inputs должны иметь размерность (batch_size, n_features_for_embedding)
                # n_features_for_embedding может быть до 2**n_qubits_for_ansatz
                qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits_for_ansatz), normalize=True, pad_with=0.)
            else:
                raise ValueError(f"Unsupported embedding_type: {self.embedding_type}")

            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits_for_ansatz))
            return qml.expval(qml.PauliZ(0))

        self.q_circuit = quantum_circuit
        print(
            f"QNN initialized with embedding: {self.embedding_type}, {self.n_qubits_for_ansatz} ansatz qubits, {self.n_layers} entangling layers.")
        print(f"Quantum weights shape: {self.q_weights.shape}")

    def forward(self, inputs):
        # Для AmplitudeEmbedding, inputs это (batch_size, n_features_for_embedding)
        # Для AngleEmbedding, inputs это (batch_size, n_qubits_for_ansatz) - убедитесь, что это так
        return self.q_circuit(inputs, self.q_weights)
    def forward(self, inputs):
        """
        Проход данных через квантовую схему.

        Args:
            inputs (torch.Tensor): Входной тензор с признаками.
                                   Форма (batch_size, n_features).
                                   n_features должно быть равно n_qubits.
        """
        # QNode ожидает, что inputs и weights будут переданы как отдельные аргументы.
        # PyTorch nn.Module использует self для доступа к параметрам.
        # Для пакетной обработки с PyTorch интерфейсом Pennylane автоматически
        # обрабатывает батчи для inputs, если QNode правильно декорирован.
        # qml.qnn.TorchLayer мог бы быть альтернативой для более глубокой интеграции,
        # но прямое использование QNode тоже эффективно. [5]

        # Pennylane's Torch interface handles batching if inputs is [batch_size, num_features]
        # The q_weights are fixed for the batch during a forward pass.
        # The QNode will execute for each item in the batch with the same q_weights.
        return self.q_circuit(inputs, self.q_weights)


if __name__ == '__main__':
    # Пример использования (для отладки)
    num_features_example = 4  # Должно соответствовать k_features из data_loader

    # Устанавливаем N_QUBITS в config для этого примера, если он не был установлен
    # В реальном сценарии это будет установлено в train.py из k_features
    if not hasattr(config, 'N_QUBITS') or config.N_QUBITS != num_features_example:
        print(f"Temporarily setting config.N_QUBITS to {num_features_example} for quantum_model.py test")
        config.N_QUBITS = num_features_example

    model = QNN(n_qubits=config.N_QUBITS,
                n_layers=config.N_LAYERS,
                rotation_gate=config.ROTATION_GATE_EMBEDDING,
                device_name=config.QUANTUM_DEVICE)

    print("\nQNN Model Structure:")
    print(model)

    # Создание примера входного батча
    # Значения признаков должны быть в диапазоне, ожидаемом AngleEmbedding
    # (обычно [0, 1] после скейлера, которые AngleEmbedding преобразует в углы)
    example_batch_size = 5
    example_input = torch.rand(example_batch_size, config.N_QUBITS, dtype=torch.float32)

    print(f"\nExample input shape: {example_input.shape}")

    # Прогон через модель
    try:
        output = model(example_input)
        print(f"Example output shape: {output.shape}")  # Ожидается (batch_size,)
        print(f"Example output values: {output.data.numpy()}")  # Ожидаются значения в [-1, 1]
    except Exception as e:
        print(f"Error during model forward pass: {e}")