import numpy as np

EPS = 1e-12


def _drop_constant_dims(latents: np.ndarray) -> np.ndarray:
    """Drop latent dimensions with zero variance (same as disentanglement_lib)."""
    latents = np.asarray(latents)
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2D (N,L), got {latents.shape}")
    variances = latents.var(axis=0)
    active_mask = variances > 0.0
    return latents[:, active_mask]


def _estimate_entropy(values: np.ndarray, num_bins: int = 100) -> float:
    counts, _ = np.histogram(values, bins=num_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _estimate_mutual_information(latent_dim: np.ndarray, factor: np.ndarray, num_bins: int = 100) -> float:
    hz = _estimate_entropy(latent_dim, num_bins=num_bins)

    uniq, counts = np.unique(factor, return_counts=True)
    probs_v = counts / counts.sum()

    h_z_given_v = 0.0
    for val, p_val in zip(uniq, probs_v):
        mask = factor == val
        z_given_v = latent_dim[mask]
        if len(z_given_v) < 2:
            continue
        h_z_given_v += p_val * _estimate_entropy(z_given_v, num_bins=num_bins)

    mi = hz - h_z_given_v
    return max(float(mi), 0.0)  # clip numerical negatives


def _estimate_factor_entropy(factor: np.ndarray) -> float:
    _, counts = np.unique(factor, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + EPS)))


def scalable_mig_score(gen_factors: np.ndarray, latents: np.ndarray, num_bins: int = 100) -> dict:
    gen_factors = np.asarray(gen_factors)
    latents = np.asarray(latents, dtype=float)

    if gen_factors.ndim != 2:
        raise ValueError(f"gen_factors must be 2D (N,K), got {gen_factors.shape}")
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2D (N,L), got {latents.shape}")
    if gen_factors.shape[0] != latents.shape[0]:
        raise ValueError(
            f"N mismatch: gen_factors N={gen_factors.shape[0]} vs latents N={latents.shape[0]}"
        )
    if not isinstance(num_bins, int) or num_bins < 2:
        raise ValueError("num_bins must be an integer >= 2.")

    N, K = gen_factors.shape
    _, L = latents.shape

    # --- entropy of each factor -------------------------------------------
    factor_entropy = np.array(
        [_estimate_factor_entropy(gen_factors[:, k]) for k in range(K)], dtype=float
    )

    # --- mutual information matrix I(z_j; v_k) ----------------------------
    mi_matrix = np.zeros((L, K), dtype=float)
    for j in range(L):
        for k in range(K):
            mi_matrix[j, k] = _estimate_mutual_information(
                latents[:, j], gen_factors[:, k], num_bins=num_bins
            )

    # --- normalise by factor entropy: m[j,k] = I(z_j; v_k) / H(v_k) ------
    # Avoid divide-by-zero for constant factors (H=0).
    safe_entropy = np.maximum(factor_entropy, EPS)
    mig_matrix = mi_matrix / safe_entropy[None, :]  # (L, K)

    # --- per-factor gap: top-1 minus top-2 normalised MI ------------------
    per_factor = np.zeros(K, dtype=float)
    for k in range(K):
        col = mig_matrix[:, k]

        sorted_vals = np.sort(col)[::-1]
        per_factor[k] = float(sorted_vals[0] - sorted_vals[1])

    avg_score = float(np.mean(per_factor))

    return {
        "MIG": avg_score,
        "MIG_matrix": mig_matrix,
        "per_factor": per_factor,
        "factor_entropy": factor_entropy,
    }


def compute_mig(C: np.ndarray, Z: np.ndarray, num_bins: int = 100) -> dict:
    C_active = _drop_constant_dims(C)
    if C_active.shape[1] == 0:
        return {"MIG": 0.0, "MIG_matrix": np.zeros((0, Z.shape[1]))}

    return scalable_mig_score(Z, C_active, num_bins=num_bins)