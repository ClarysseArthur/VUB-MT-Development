import numpy as np

EPS = 1e-12


def _drop_constant_dims(latents: np.ndarray) -> np.ndarray:
    """Remove latent dimensions with zero variance."""
    latents = np.asarray(latents, dtype=float)
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2D (N, L), got {latents.shape}")

    variances = np.var(latents, axis=0)
    active_mask = variances > 0.0
    return latents[:, active_mask]


def _entropy_discrete(x: np.ndarray) -> float:
    """Entropy H(X) for a discrete 1D variable."""
    x = np.asarray(x).ravel()
    _, counts = np.unique(x, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    probs = np.clip(probs, EPS, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def _mutual_information_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """
    Mutual information I(X;Y) for two discrete 1D variables.
    Uses the definition:
        I(X;Y) = sum_{x,y} p(x,y) log( p(x,y) / (p(x)p(y)) )
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must have same length, got {x.shape[0]} and {y.shape[0]}")

    n = x.shape[0]

    x_vals, x_inv = np.unique(x, return_inverse=True)
    y_vals, y_inv = np.unique(y, return_inverse=True)

    joint = np.zeros((len(x_vals), len(y_vals)), dtype=float)
    np.add.at(joint, (x_inv, y_inv), 1.0)
    joint /= n

    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)

    nz = joint > 0
    mi = np.sum(joint[nz] * np.log(joint[nz] / (px @ py)[nz]))
    return float(max(mi, 0.0))  # clip tiny negative numerical noise


def _discretize_latents(latents: np.ndarray, num_bins: int = 20) -> np.ndarray:
    """
    Discretize each latent dimension independently into integer bins.
    Returns array of shape (N, L) with discrete values in {0, ..., num_bins-1}.
    """
    latents = np.asarray(latents, dtype=float)
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2D (N, L), got {latents.shape}")
    if not isinstance(num_bins, int) or num_bins < 2:
        raise ValueError("num_bins must be an integer >= 2.")

    N, L = latents.shape
    latents_disc = np.zeros((N, L), dtype=int)

    for j in range(L):
        z = latents[:, j]

        z_min = np.min(z)
        z_max = np.max(z)

        if np.isclose(z_min, z_max):
            latents_disc[:, j] = 0
            continue

        # Use fixed bin edges for this latent dimension
        edges = np.linspace(z_min, z_max, num_bins + 1)

        # Digitize into bins 0..num_bins-1
        latents_disc[:, j] = np.digitize(z, edges[1:-1], right=False)

    return latents_disc


def scalable_mig_score(gen_factors: np.ndarray, latents: np.ndarray, num_bins: int = 20) -> dict:
    """
    Compute MIG for discrete generative factors and continuous latent codes.

    Parameters
    ----------
    gen_factors : np.ndarray, shape (N, K)
        Discrete generative factors.
    latents : np.ndarray, shape (N, L)
        Continuous latent codes.
    num_bins : int
        Number of bins used to discretize each latent dimension.

    Returns
    -------
    dict with keys:
        - "MIG": average MIG over valid factors
        - "MIG_matrix": normalized MI matrix of shape (L, K)
        - "MI_matrix": raw mutual information matrix of shape (L, K)
        - "per_factor": per-factor MIG values of shape (K,)
        - "factor_entropy": entropy H(v_k) for each factor
        - "valid_factors": boolean mask for non-constant factors
        - "active_latents": number of non-constant latent dimensions used
    """
    gen_factors = np.asarray(gen_factors)
    latents = np.asarray(latents, dtype=float)

    if gen_factors.ndim != 2:
        raise ValueError(f"gen_factors must be 2D (N, K), got {gen_factors.shape}")
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2D (N, L), got {latents.shape}")
    if gen_factors.shape[0] != latents.shape[0]:
        raise ValueError(
            f"N mismatch: gen_factors N={gen_factors.shape[0]} vs latents N={latents.shape[0]}"
        )

    # Remove constant latent dimensions
    latents_active = _drop_constant_dims(latents)
    if latents_active.shape[1] == 0:
        K = gen_factors.shape[1]
        return {
            "MIG": 0.0,
            "MIG_matrix": np.zeros((0, K), dtype=float),
            "MI_matrix": np.zeros((0, K), dtype=float),
            "per_factor": np.zeros(K, dtype=float),
            "factor_entropy": np.array([_entropy_discrete(gen_factors[:, k]) for k in range(K)]),
            "valid_factors": np.array(
                [_entropy_discrete(gen_factors[:, k]) > EPS for k in range(K)], dtype=bool
            ),
            "active_latents": 0,
        }

    # Discretize latents
    latents_disc = _discretize_latents(latents_active, num_bins=num_bins)

    N, K = gen_factors.shape
    _, L = latents_disc.shape

    # Entropy of each factor
    factor_entropy = np.array(
        [_entropy_discrete(gen_factors[:, k]) for k in range(K)],
        dtype=float
    )
    valid_factors = factor_entropy > EPS

    # Raw MI matrix: I(z_j ; v_k)
    mi_matrix = np.zeros((L, K), dtype=float)
    for j in range(L):
        for k in range(K):
            if valid_factors[k]:
                mi_matrix[j, k] = _mutual_information_discrete(
                    latents_disc[:, j], gen_factors[:, k]
                )
            else:
                mi_matrix[j, k] = 0.0

    # Normalized MI matrix: I(z_j ; v_k) / H(v_k)
    mig_matrix = np.zeros((L, K), dtype=float)
    for k in range(K):
        if valid_factors[k]:
            mig_matrix[:, k] = mi_matrix[:, k] / factor_entropy[k]

    # Per-factor MIG = top1 - top2 of normalized MI
    per_factor = np.zeros(K, dtype=float)
    for k in range(K):
        if not valid_factors[k]:
            per_factor[k] = 0.0
            continue

        col = np.sort(mig_matrix[:, k])[::-1]

        if len(col) == 0:
            per_factor[k] = 0.0
        elif len(col) == 1:
            per_factor[k] = float(col[0])
        else:
            per_factor[k] = float(col[0] - col[1])

    avg_mig = float(np.mean(per_factor[valid_factors])) if np.any(valid_factors) else 0.0

    return {
        "MIG": avg_mig,
        "MIG_matrix": mig_matrix,
        "MI_matrix": mi_matrix,
        "per_factor": per_factor,
        "factor_entropy": factor_entropy,
        "valid_factors": valid_factors,
        "active_latents": L,
    }


def compute_mig(C: np.ndarray, Z: np.ndarray, num_bins: int = 20) -> dict:
    """
    Wrapper:
      C = latent codes, shape (N, L)
      Z = discrete generative factors, shape (N, K)
    """
    return scalable_mig_score(gen_factors=Z, latents=C, num_bins=num_bins)