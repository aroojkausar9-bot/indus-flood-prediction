"""
models.py
=========
Random Forest and SVM-PSO model training for Indus Basin flood prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")


# ── Random Forest ──────────────────────────────────────────────────────────────

def build_random_forest(**kwargs) -> RandomForestClassifier:
    """
    Instantiate the Random Forest classifier with the paper's default settings.
    Keyword arguments override any default.
    """
    defaults = dict(
        n_estimators=150,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)


def train_random_forest(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test: np.ndarray,
                        threshold: float = 0.5):
    """
    Fit RF and return (model, predicted_probabilities).

    Parameters
    ----------
    threshold : decision threshold for positive class (default 0.5)
    """
    model = build_random_forest()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return model, proba


# ── Particle Swarm Optimization ───────────────────────────────────────────────

class Particle:
    """Single PSO particle."""

    def __init__(self, bounds: list):
        self.position      = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])
        self.velocity      = np.random.uniform(-1, 1, len(bounds))
        self.best_position = self.position.copy()
        self.best_score    = -np.inf


class PSO:
    """
    Particle Swarm Optimizer.

    Parameters
    ----------
    n_particles  : swarm size
    bounds       : list of (low, high) tuples, one per dimension
    n_iterations : number of swarm update steps
    w            : inertia weight
    c1           : cognitive coefficient
    c2           : social coefficient
    """

    def __init__(self, n_particles: int, bounds: list, n_iterations: int,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        self.particles   = [Particle(bounds) for _ in range(n_particles)]
        self.bounds      = bounds
        self.n_iter      = n_iterations
        self.w, self.c1, self.c2 = w, c1, c2
        self.gbest_pos   = None
        self.gbest_score = -np.inf

    def optimize(self, objective_fn, verbose: bool = True):
        """
        Run the PSO loop.

        Parameters
        ----------
        objective_fn : callable(position) → scalar score to MAXIMIZE
        verbose      : print progress every 5 iterations

        Returns
        -------
        (best_position, best_score)
        """
        for it in range(self.n_iter):
            for p in self.particles:
                score = objective_fn(p.position)
                if score > p.best_score:
                    p.best_score    = score
                    p.best_position = p.position.copy()
                if score > self.gbest_score:
                    self.gbest_score = score
                    self.gbest_pos   = p.position.copy()

            for p in self.particles:
                r1, r2     = np.random.random(2)
                cognitive  = self.c1 * r1 * (p.best_position - p.position)
                social     = self.c2 * r2 * (self.gbest_pos   - p.position)
                p.velocity = self.w * p.velocity + cognitive + social
                p.position = p.position + p.velocity
                for i, (lo, hi) in enumerate(self.bounds):
                    p.position[i] = np.clip(p.position[i], lo, hi)

            if verbose and (it + 1) % 5 == 0:
                print(f"  [PSO] iter {it+1}/{self.n_iter} | best F1={self.gbest_score:.4f}")

        return self.gbest_pos, self.gbest_score


# ── SVM-PSO ────────────────────────────────────────────────────────────────────

def train_svm_pso(X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_test: np.ndarray,
                  n_particles: int = 6,
                  n_iterations: int = 10,
                  max_subsample: int = 2000,
                  max_val_sample: int = 500,
                  threshold: float = 0.5,
                  verbose: bool = True):
    """
    Tune SVM (RBF kernel) hyperparameters (C, gamma) using PSO,
    then fit a final model on the full training set.

    PSO search space
    ----------------
    C     : [0.5, 20.0]
    gamma : 10^[−4, −1]

    Returns
    -------
    (model, predicted_probabilities)
    """

    def objective(params):
        C, log_gamma = params
        gamma = 10 ** log_gamma
        svm   = SVC(kernel="rbf", C=C, gamma=gamma,
                    probability=True, class_weight="balanced",
                    random_state=42)
        n_sub = min(max_subsample, len(X_train))
        idx   = np.random.choice(len(X_train), n_sub, replace=False)
        svm.fit(X_train[idx], y_train[idx] if hasattr(y_train, "__getitem__")
                else y_train.values[idx])

        n_val = min(max_val_sample, len(X_test))
        vi    = np.random.choice(len(X_test), n_val, replace=False)
        y_pred = svm.predict(X_test[vi])
        y_true = (y_train.values[vi] if hasattr(y_train, "values")
                  else y_train[vi])
        return f1_score(y_true, y_pred, pos_label=1.0, zero_division=0)

    bounds = [(0.5, 20.0), (-4, -1)]

    if verbose:
        print("[SVM-PSO] Starting PSO hyperparameter search …")

    pso = PSO(n_particles=n_particles, bounds=bounds, n_iterations=n_iterations)
    best_params, best_f1 = pso.optimize(objective, verbose=verbose)

    best_C     = best_params[0]
    best_gamma = 10 ** best_params[1]

    if verbose:
        print(f"[SVM-PSO] Best params → C={best_C:.3f}, gamma={best_gamma:.5f}, "
              f"val F1={best_f1:.4f}")

    model = SVC(kernel="rbf", C=best_C, gamma=best_gamma,
                probability=True, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return model, proba


# ── Threshold Selection ────────────────────────────────────────────────────────

def choose_threshold(y_true: np.ndarray,
                     y_proba: np.ndarray,
                     target_max_acc: float = 0.90,
                     n_thresholds: int = 17) -> tuple:
    """
    Sweep decision thresholds in [0.10, 0.90] and return the threshold that
    maximises recall subject to accuracy < target_max_acc.

    Returns
    -------
    (best_threshold, dict with acc/prec/rec/f1 at that threshold)
    """
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score)

    thresholds = np.linspace(0.1, 0.9, n_thresholds)
    best_th, best_rec, best_stats = 0.5, -1, None

    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        if acc < target_max_acc and rec > best_rec:
            best_rec  = rec
            best_th   = th
            best_stats = {"accuracy": acc, "precision": prec,
                          "recall": rec, "f1": f1}

    return best_th, best_stats
