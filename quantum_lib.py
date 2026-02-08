import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score


class QuantumInformationField:
    def __init__(self, n_features, device_name="lightning.qubit"):
        self.n_qubits = min(n_features, 16)
        self.dev = qml.device(device_name, wires=self.n_qubits)
        self.centroids = {}
        self.target_names = None
        self.n_interactions = (self.n_qubits * (self.n_qubits - 1)) // 2
        
        self.qnode_independent = qml.QNode(self._circuit_independent, self.dev)
        self.qnode_hamiltonian = qml.QNode(self._circuit_hamiltonian, self.dev)

    def _circuit_independent(self, x):
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)
        return qml.state()

    def _circuit_hamiltonian(self, x, J_params):
        x_red = x * 0.5
        for i in range(self.n_qubits): qml.RY(x_red[i], wires=i)
        k = 0
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if k < len(J_params):
                    qml.IsingZZ(J_params[k], wires=[i, j])
                    k += 1
        for i in range(self.n_qubits): qml.RY(x_red[i], wires=i)
        return qml.state()

    def get_full_quantum_stats(self, data, J_params=None):
        circ = self.qnode_hamiltonian if J_params is not None else self.qnode_independent
        dim = 2**self.n_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        
        for n, x in enumerate(data, 1):
            res = circ(x, J_params) if J_params is not None else circ(x)
            psi = np.array(res, dtype=complex).flatten()
            rho += (np.outer(psi, np.conj(psi)) - rho) / n
            
        eigvals = np.sort(np.linalg.eigvalsh(rho))[::-1]
        eigvals = np.clip(eigvals, 1e-12, 1.0)
        purity = float(np.real(np.sum(eigvals**2)))
        _, eigvecs = np.linalg.eigh(rho)
        
        return {
            "purity": purity,
            "gap": float(np.log(eigvals[0] / eigvals[1])) if eigvals[1] > 1e-12 else 0,
            "hbar_eff": 1.0 - purity,
            "entropy": float(-np.sum(eigvals * np.log(eigvals))),
            "ground_state": np.array(eigvecs[:, -1], dtype=complex),
            "lambda_0": float(eigvals[0]),
        }

    def train_field(self, X, y, method="gap", restarts=1):
        unique_classes = np.unique(y)
        class_subsets = [X[y == c] for c in unique_classes]
        best_J, best_score = None, float('inf')

        for _ in range(restarts):
            init_J = pnp.random.uniform(-0.5, 0.5, size=self.n_interactions)
            obj = lambda J: -np.mean([self.get_full_quantum_stats(s, J)['gap'] for s in class_subsets])
            res = minimize(obj, init_J, method='Nelder-Mead', options={'maxiter': 50})
            if res.fun < best_score:
                best_score, best_J = res.fun, res.x
        return np.array(best_J)

    def analyze(self, X, y, J_params=None, target_names=None):
        self.target_names = target_names if target_names is not None else [str(i) for i in np.unique(y)]
        unique_classes = np.unique(y)
        
        results = {self.target_names[c]: self.get_full_quantum_stats(X[y == c], J_params) for c in unique_classes}
        for c in unique_classes: 
            self.centroids[c] = results[self.target_names[c]]['ground_state']

        y_pred, y_scores = [], []
        for x in X:
            psi_x = np.array(self.qnode_hamiltonian(x, J_params) if J_params is not None else self.qnode_independent(x), dtype=complex).flatten()
            fids = [float(np.abs(np.vdot(self.centroids[c], psi_x))**2) for c in unique_classes]
            y_pred.append(unique_classes[np.argmax(fids)])
            y_scores.append(fids)
            
        return results, {"accuracy": accuracy_score(y, y_pred)}, np.array(y_pred), np.array(y_scores)

    def print_quantum_stats(self, results):
        df = pd.DataFrame([{**{'Class': n}, **{k: f"{v:.4f}" for k, v in s.items() if k != 'ground_state'}} for n, s in results.items()])
        print("\n" + "="*60 + "\nKWANTOWA ANALIZA SPEKTRALNA POLA (BOZONY)\n" + "="*60)
        print(df.to_string(index=False))

    def print_mismatch_report(self, y_true, y_pred, y_scores):
        mismatches = np.where(y_true != y_pred)[0]
        report = []
        for idx in mismatches:
            scores = y_scores[idx]
            gap = np.sort(scores)[-1] - np.sort(scores)[-2]
            row = {'ID': idx, 'True': self.target_names[y_true[idx]], 'Pred': self.target_names[y_pred[idx]], 'Fid Gap': f"{gap:.4f}"}
            for i, name in enumerate(self.target_names): row[f"Fid {name}"] = f"{scores[i]:.4f}"
            row['Diagnosis'] = "Ambiguity" if gap < 0.05 else "Anomaly"
            report.append(row)
        print("\n--- RAPORT BŁĘDÓW (BOZONOWY) ---")
        print(pd.DataFrame(report).to_string(index=False) if report else "Brak błędów!")


class ZeemanSpectroscopicClassifier:
    """
    Narzędzie diagnostyczne. Używa efektów Zeemana (Z) i Rabiego (X/Y)
    do badania, dlaczego konkretne punkty danych zostały błędnie sklasyfikowane.
    """
    def __init__(self, n_qubits, centroids, optimal_J, device_name="lightning.qubit"):
        self.n_qubits = n_qubits
        self.centroids = centroids
        self.optimal_J = np.array(optimal_J) if optimal_J is not None else None
        self.dev = qml.device(device_name, wires=self.n_qubits)
        self.qnode_spectroscopy = qml.QNode(self._circuit_spectroscopy, self.dev)

    def _circuit_spectroscopy(self, x, J_params, angle, axis='Z'):
        # Initial state embedding
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)
        
        # APLIKACJA POLA ZEWNĘTRZNEGO (Perturbacja)
        for i in range(self.n_qubits):
            if axis == 'Z':   qml.RZ(angle, wires=i) # Efekt Zeemana (Faza)
            elif axis == 'X': qml.RX(angle, wires=i) # Oscylacje Rabiego (Amplituda)
            elif axis == 'Y': qml.RY(angle, wires=i) # Przesunięcie fazowe Y
            
        # Interakcje (Topologia J)
        if J_params is not None:
            k = 0
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    if k < len(J_params):
                        qml.IsingZZ(J_params[k], wires=[i, j])
                        k += 1
        return qml.state()

    def recover_sample(self, x, true_label, steps=40):
        """Skanuje pola rotacji, aby 'naprawić' klasyfikację próbki."""
        # 1. Baseline Check
        psi_0 = self.qnode_spectroscopy(x, self.optimal_J, 0.0, 'Z')
        psi_0 = np.array(psi_0, dtype=complex).flatten()
        
        # 2. Smart Scan
        scan_range = np.linspace(-np.pi, np.pi, steps)
        scan_range = sorted(scan_range, key=abs) 
        axes = ['Z', 'X', 'Y']

        for angle in scan_range:
            for ax in axes:
                psi = self.qnode_spectroscopy(x, self.optimal_J, angle, ax)
                psi_v = np.array(psi, dtype=complex).flatten()
                
                fids = {lab: np.abs(np.vdot(self.centroids[lab], psi_v))**2 for lab in self.centroids}
                if max(fids, key=fids.get) == true_label:
                    return ax, float(angle)
        return "FAILED", None

    def diagnose(self, field_type, angle):
        if field_type == "FAILED": return "CRITICAL", "Event Horizon / Unrecoverable"
        abs_a = abs(angle)
        if abs_a <= 0.15: return "NOISE", "Thermal Instability"
        if abs_a <= 0.60: return "BOUNDARY", "Geometric Ambiguity"
        return "ANOMALY", "Spectral Anomaly"
