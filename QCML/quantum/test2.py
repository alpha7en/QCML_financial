    import pennylane as qml
    from pennylane import numpy as np

    # --- Настройки ---
    n = 5                # число кубитов
    K = 4                # число признаков (любой K)
    n_layers_vqe = 3     # число слоёв в ansatz-е VQE
    L_U = 1              # число слоёв в U для каждого A_k
    dev = qml.device("lightning.qubit", wires=n)

    # --- 1. Базовый оператор O_base ---
    dim = 2**n
    # Нормируем спектр в диапазон [-1,1]
    vals = np.arange(dim, dtype=float)
    vals = 2 * (vals - vals.mean()) / (vals.max() - vals.min())
    O_base = np.diag(vals)
    O2_base = np.diag(vals**2)

    # Квантовые объекты для измерений
    O_op  = qml.Hermitian(O_base, wires=range(n))
    O2_op = qml.Hermitian(O2_base, wires=range(n))
    I_op  = qml.Identity(wires=0)

    # --- 2. Функция apply_U ---
    def apply_U(theta_k):
        qml.templates.StronglyEntanglingLayers(weights=theta_k, wires=range(n))

    # --- 3. QNode для одного оператора A_k и заданного phi ---
    @qml.qnode(dev)
    def exp_vals_for_A(phi, theta_k):
        # Готовим вариационное состояние |phi>
        qml.templates.StronglyEntanglingLayers(weights=phi, wires=range(n))
        # Применяем U(theta_k)
        apply_U(theta_k)
        # Возвращаем <A_k^2> и <A_k>, т.е. <O2_base> и <O_base> на текущем состоянии
        return qml.expval(O2_op), qml.expval(O_op)

    # --- 4. Энергия VQE: сумма по k от вкладов всех операторов для фиксированных theta ---
    def energy_vqe(phi, thetas, xs):
        E = 0.0
        for k in range(K):
            exp_O2, exp_O = exp_vals_for_A(phi, thetas[k])
            E += exp_O2 - 2 * xs[k] * exp_O + xs[k] ** 2
        return E

    # --- 5. VQE-фаза: оптимизация phi при фиксированных thetas и xs ---
    # Пример входных данных
    xs = np.random.normal(0, 1, size=(K,))  # любой вектор размерности K

    # Инициализируем phi
    shape_phi = qml.templates.StronglyEntanglingLayers.shape(n_layers_vqe, n)
    phi = np.random.normal(0, 0.01, size=shape_phi)

    # Инициализируем thetas для каждого оператора
    thetas = np.random.normal(0, 0.1, size=(K, L_U, n, 3))

    # Оптимизация phi
    opt_phi = qml.AdamOptimizer(0.05)
    max_iter = 400
    for i in range(max_iter):
        phi, energy = opt_phi.step_and_cost(lambda v: energy_vqe(v, thetas, xs), phi)
        if i % 20 == 0:
            print(f"VQE iter {i:>3}: energy = {energy:.6f}")
    phi_opt = phi
    print("Final VQE energy:", energy)

    # --- 6. Извлекаем состояние psi_t для градиентов по thetas ---
    @qml.qnode(dev)
    def psi_state(phi):
        qml.templates.StronglyEntanglingLayers(weights=phi, wires=range(n))
        return qml.state()

    psi = psi_state(phi_opt)

    # --- 7. QNode для оценки одного A_k на ψ и фиксация stateprep ---
    @qml.qnode(dev, diff_method="adjoint", gradient_kwargs={"trainable_params": list(range(K * L_U * n * n_layers_vqe))})
    def exp_vals_for_A_on_psi(theta_k):
        # Готовим найденное psi
        qml.StatePrep(psi, wires=range(n))
        # Применяем U(theta_k)
        apply_U(theta_k)
        # Возвращаем <O2>, <O>
        return qml.expval(O2_op), qml.expval(O_op)

    # --- 8. Энергия H(θ) и её градиенты для каждого k ---
    def energy_total_theta(thetas, xs):
        E = 0.0
        for k in range(K):
            exp_O2, exp_O = exp_vals_for_A_on_psi(thetas[k])
            E += exp_O2 - 2 * xs[k] * exp_O + xs[k] ** 2
        return E

    # Один шаг: вычисляем энергию до, градиент и обновляем thetas
    E_before = energy_total_theta(thetas, xs)
    grad_fun = qml.grad(energy_total_theta)
    grads = grad_fun(thetas, xs)      # форма (K, L_U, n, 3)
    grad_norm = np.linalg.norm(grads[0])
    print("Grad norm over all thetas:", grad_norm)

    # Обновление thetas (градиентный спуск)
    lr = 0.1
    thetas = thetas - lr * grads[0]
    E_after = energy_total_theta(thetas, xs)

    # --- 9. Итоги шага ---
    print(f"E before = {E_before:.6f}, E after = {E_after:.6f}")
