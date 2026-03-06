import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

EPS = 1e-12

def _normalize_importances(col: np.ndarray) -> np.ndarray:
    col = np.maximum(col, 0.0)
    s = col.sum()
    if s < EPS:
        return np.ones_like(col) / len(col)
    return col / s

def _entropy_base(p: np.ndarray, base: int) -> float:
    p = np.clip(p, EPS, 1.0)
    return -(p * (np.log(p) / np.log(base))).sum()

@dataclass
class DCIResult:
    D: float
    C: float
    I: float
    R: np.ndarray  # (L, K)

def _coerce_labels_to_Z(labels: np.ndarray) -> np.ndarray:
    """
    Accepts:
      - (N,) integer class ids
      - (N,1) integer class ids
      - (N,K) binary/multi-hot or continuous factors
    Returns:
      - Z of shape (N,K)
    """
    labels = np.asarray(labels)
    if labels.ndim == 1:
        return labels.reshape(-1, 1)
    if labels.ndim == 2:
        return labels
    raise ValueError(f"labels must have shape (N,), (N,1) or (N,K), got {labels.shape}")

def compute_dci(
    C: np.ndarray,              # (N, L) codes
    Z: np.ndarray,              # (N, K) factors
    factor_types: list,         # length K: "continuous" or "discrete"
    probe: str = "lasso",       # "lasso" or "rf"
    test_size: float = 0.2,
    random_state: int = 0,
    lasso_alpha: float = 0.002,
    rf_n_estimators: int = 200,
    rf_max_depth: int | None = None,
) -> DCIResult:
    C = np.asarray(C, dtype=float)
    Z = np.asarray(Z)
    N, L = C.shape
    K = Z.shape[1]
    # assert len(factor_types) == K

    C_train, C_test, Z_train, Z_test = train_test_split(
        C, Z, test_size=test_size, random_state=random_state
    )

    R = np.zeros((L, K), dtype=float)
    I_scores = []

    for j in range(K):
        z_tr = Z_train[:, j]
        z_te = Z_test[:, j]
        ftype = factor_types[j].lower()

        if probe == "lasso":
            if ftype == "continuous":
                model = make_pipeline(StandardScaler(), Lasso(alpha=lasso_alpha, max_iter=50_000))
                model.fit(C_train, z_tr.astype(float))
                pred = model.predict(C_test)
                Ij = r2_score(z_te.astype(float), pred)
                coef = model.named_steps["lasso"].coef_
                imp = np.abs(coef)

            elif ftype == "discrete":
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        penalty="l1", solver="saga", multi_class="auto",
                        C=1.0, max_iter=20_000
                    )
                )
                model.fit(C_train, z_tr.astype(int))
                pred = model.predict(C_test)
                Ij = accuracy_score(z_te.astype(int), pred)

                lr = model.named_steps["logisticregression"]
                W = lr.coef_  # (n_classes, L) or (1, L)
                imp = np.mean(np.abs(W), axis=0)
            else:
                raise ValueError(f"Unknown factor type: {factor_types[j]}")

        elif probe == "rf":
            if ftype == "continuous":
                model = RandomForestRegressor(
                    n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                    random_state=random_state, n_jobs=-1
                )
                model.fit(C_train, z_tr.astype(float))
                pred = model.predict(C_test)
                Ij = r2_score(z_te.astype(float), pred)
                imp = model.feature_importances_

            elif ftype == "discrete":
                model = RandomForestClassifier(
                    n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                    random_state=random_state, n_jobs=-1
                )
                model.fit(C_train, z_tr.astype(int))
                pred = model.predict(C_test)
                Ij = accuracy_score(z_te.astype(int), pred)
                imp = model.feature_importances_
            else:
                raise ValueError(f"Unknown factor type: {factor_types[j]}")
        else:
            raise ValueError("probe must be one of: 'lasso', 'rf'")

        R[:, j] = _normalize_importances(imp)
        I_scores.append(Ij)

    # Disentanglement D
    D_i = np.zeros(L, dtype=float)
    r_i = np.zeros(L, dtype=float)
    for i in range(L):
        row = R[i, :]
        s = row.sum()
        P = (row / s) if s >= EPS else (np.ones(K) / K)
        H = _entropy_base(P, base=K)
        D_i[i] = 1.0 - H
        r_i[i] = row.mean()
    D = float(np.sum(r_i * D_i) / (np.sum(r_i) + EPS))

    # Completeness C
    C_j = np.zeros(K, dtype=float)
    for j in range(K):
        P = R[:, j]  # sums to 1
        H = _entropy_base(P, base=L)
        C_j[j] = 1.0 - H
    C_score = float(C_j.mean())

    # Informativeness I
    I = float(np.mean(I_scores))

    return DCIResult(D=D, C=C_score, I=I, R=R)