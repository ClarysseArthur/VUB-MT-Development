import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error

# Based on paper:
#  > Eastwood, Cian, Andrei Liviu Nicolicioiu, Julius von Kügelgen, et al. 2023. “DCI-ES: An Extended Disentanglement Framework with Connections to Identifiability.” arXiv:2210.00364. Preprint, arXiv, February 16. https://doi.org/10.48550/arXiv.2210.00364.

EPS = 1e-12

def _normalize_importances(imp, eps=1e-12):
    imp = np.asarray(imp, dtype=float)
    imp = np.maximum(imp, 0.0)
    s = imp.sum()

    if s <= eps:
        return np.ones_like(imp) / len(imp)

    return imp / s

def compute_dci(C, Z, factor_types, probe="lasso"):
    C = np.asarray(C, dtype=float)
    Z = np.asarray(Z)

    N, L = C.shape
    K = Z.shape[1]

    C_train, C_test, Z_train, Z_test = train_test_split(        # Split into train/test for the probes --> C = codes, Z = factors
        C, Z, test_size=0.2, random_state=0
    )

    R = np.zeros((L, K), dtype=float)                           # R[i,j] = importance of code i for predicting factor j
    I_scores = []                                               # I_scores[j] = informativeness of code for factor j

    for j in range(K):
        z_tr = Z_train[:, j]                                    # factor j for training
        z_te = Z_test[:, j]                                     # factor j for testing
        ftype = factor_types[j].lower()

        #! Lasso                                                ---
        if probe == "lasso":
            if ftype == "continuous":                           # if factor is continuous, use Lasso regression
                model = make_pipeline(
                    StandardScaler(),
                    Lasso(alpha=0.002, max_iter=50_000)
                    )

                model.fit(C_train, z_tr.astype(float))          # Fit training data to Lasso model
                pred = model.predict(C_test)                    # Predict on test data
                Ij = r2_score(z_te.astype(float), pred)
                coef = model.named_steps["lasso"].coef_         # Lasso coefficients => importance of each code for predicting the factor
                imp = np.abs(coef)

            elif ftype == "discrete":                           # if factor is discrete, use Logistic regression with L1 penalty (Lasso)
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        solver="saga",
                        l1_ratio=1.0,                           #! CHANGED
                        C=1.0,
                        max_iter=20_000
                    )
                )
                model.fit(C_train, z_tr.astype(int))            # Fit training data to Logistic regression model
                pred = model.predict(C_test)                    # Predict on test data
                Ij = accuracy_score(z_te.astype(int), pred)     # Accuracy score for discrete factor

                lr = model.named_steps["logisticregression"]
                W = lr.coef_                                    # Logistic regression coefficients => importance of each code for predicting the factor
                imp = np.mean(np.abs(W), axis=0)
            else:
                raise ValueError(f"Unknown factor type: {factor_types[j]}")

        #! Random Forest                                        ---
        elif probe == "rf":
            if ftype == "continuous":
                model = RandomForestRegressor(
                    n_estimators=200, max_depth=50, n_jobs=-1
                )

                model.fit(C_train, z_tr.astype(float))          # Fit the model to the training data
                pred = model.predict(C_test)                    # Predict on the test data
                Ij = r2_score(z_te.astype(float), pred)
                imp = model.feature_importances_                # Random Forest feature importances => importance of each code for predicting the factor

            elif ftype == "discrete":
                model = RandomForestClassifier(
                    n_estimators=200, max_depth=50,
                    random_state=0, n_jobs=-1                   #! CHANGED
                )

                model.fit(C_train, z_tr.astype(int))            # Fit the model to the training data
                pred = model.predict(C_test)                    # Predict on the test data
                Ij = accuracy_score(z_te.astype(int), pred)     # Accuracy score for discrete factor
                imp = model.feature_importances_                # Random Forest feature importances => importance of each code for predicting the factor
            else:
                raise ValueError(f"Unknown factor type: {factor_types[j]}")
        else:
            raise ValueError("probe must be one of: 'lasso', 'rf'")

        R[:, j] = _normalize_importances(imp)                   # Normalize importances to sum to 1 for each factor
        I_scores.append(Ij)                                     # Append informativeness score for factor j

    #! Disentanglement D                                        ---
    D_i = np.zeros(L, dtype=float)
    r_i = np.zeros(L, dtype=float)

    for i in range(L):
        row = R[i, :]                                           # importance of code i for all factors, sums to 1 across factors
        s = row.sum()                                           # should be 1 due to normalization, but we check to avoid division by zero

        if s >= EPS:                                            # If the above s is not zero, we can compute the probability distribution over factors for code i
            P = (row / s)                                       # Normalize to get a probability distribution over factors for code i

        else:
            P = np.ones(K) / K                                  # If s is zero, we assign a uniform distribution over factors for code i

        if K <= 1:                                              #! CHANGED
            H = 0.0                                             #! CHANGED
        else:
            H = entropy(P, base=K)                              #! CHANGED

        D_i[i] = 1.0 - H                                        # Disentanglement score for code i => 1 - entropy, higher is better (max 1 when all importance is on one factor)
        r_i[i] = row.mean()                                     # Importance of code i across all factors, used for weighting the disentanglement scores

    D = float(np.sum(r_i * D_i) / (np.sum(r_i) + EPS))          # Overall disentanglement score D => weighted average of code-wise disentanglement scores, weighted by the importance of each code across all factors

    #! Completeness C                                           ---
    C_j = np.zeros(K, dtype=float)

    for j in range(K):                                          # Loop over factors to compute completeness for each factor
        P = R[:, j]                                             # Importance of all codes for factor j, sums to 1 across codes

        if L <= 1:                                              #! CHANGED
            H = 0.0                                             #! CHANGED
        else:
            H = entropy(P, base=L)                              #! CHANGED

        C_j[j] = 1.0 - H                                        # Completeness score for factor j => 1 - entropy, higher is better (max 1 when all importance is on one code)

    C_score = float(C_j.mean())                                 # Overall completeness score C => average of factor-wise completeness scores, unweighted average across factors

    #! Informativeness I                                        ---
    I = float(np.mean(I_scores))                                # Overall informativeness score I => average of informativeness scores across factors
    
    return {'D': D, 'C': C_score, 'I': I}

def report_dci(res, title="DCI Scores"):
    print(f'--- {title} ---')
    print(f' > Disentanglement: {round(res["D"], 4)}')
    print(f' > Completeness: {round(res["C"], 4)}')
    print(f' > Informativeness: {round(res["I"], 4)}')