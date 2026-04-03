import numpy as np

EPS = 1e-12

def _drop_constant_dims(latents: np.ndarray) -> np.ndarray:
    """Drop latent dimensions with zero variance (same as disentanglement_lib)."""
    latents = np.asarray(latents)
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2D (N,L), got {latents.shape}")
    variances = latents.var(axis=0)  # variance per latent dim
    active_mask = variances > 0.0
    return latents[:, active_mask]

def scalable_irs_score(gen_factors: np.ndarray, latents: np.ndarray, diff_quantile: float = 0.99):
    """
    Convenience for IRS computation. -> based on disentanglement_lib's implementation

    Args:
      C: (N,L) latent code matrix.
      Z: (N,K) discrete generative factors.
         Continuous factors should be discretized first.
      diff_quantile: quantile used to approximate maximal deviations.

    Returns:
      Dictionary containing the empirical IRS score and per-latent/per-factor scores.
    """
    gen_factors = np.asarray(gen_factors)
    latents = np.asarray(latents, dtype=float)

    if gen_factors.ndim != 2:
        raise ValueError(f"gen_factors must be 2D (N,K), got {gen_factors.shape}")
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2D (N,L), got {latents.shape}")
    if gen_factors.shape[0] != latents.shape[0]:
        raise ValueError(f"N mismatch: gen_factors N={gen_factors.shape[0]} vs latents N={latents.shape[0]}")
    if not (0.0 < diff_quantile <= 1.0):
        raise ValueError("diff_quantile must be in (0,1].")

    N, K = gen_factors.shape
    _, L = latents.shape

    # Normalizer per latent dim: max |z - mean(z)|
    max_deviations = np.max(np.abs(latents - latents.mean(axis=0)), axis=0)  # (L,)
    max_deviations = np.maximum(max_deviations, EPS)  # avoid divide-by-zero

    cum_deviations = np.zeros((L, K), dtype=float)

    for i in range(K):                                                  # For each generative factor
        uniq = np.unique(gen_factors[:, i])                             # unique values of g_i

        for val in uniq:
            match = (gen_factors[:, i] == val)

            if not np.any(match):
                continue

            e_loc = latents[match].mean(axis=0)                         # E[Z | g_i=val]
            diffs = np.abs(latents[match] - e_loc)                      # |Z - E[Z|g]|
            q = np.percentile(diffs, q=diff_quantile * 100.0, axis=0)
            cum_deviations[:, i] += q

        cum_deviations[:, i] /= max(len(uniq), 1)

    normalized_deviations = cum_deviations / max_deviations[:, None]  # (L,K) EMPIDA / max deviation per latent
    irs_matrix = 1.0 - normalized_deviations                          # higher is better

    disent_scores = irs_matrix.max(axis=1)     # per-latent best factor
    parents = irs_matrix.argmax(axis=1)

    # Weighted average by max_deviations (as in disentanglement_lib)
    avg_score = float(np.average(disent_scores, weights=max_deviations))

    return {
        "IRS": avg_score,
        "IRS_matrix": irs_matrix,
        "parents": parents,
        "disentanglement_scores": disent_scores,
        "max_deviations": max_deviations,
    }

def compute_irs_from_multihot_labels(C: np.ndarray, labels_40: np.ndarray, diff_quantile: float = 0.99):
    """
    Convenience for your setup:
      C: (N,L) code computed WITHOUT feeding the label-view
      labels_40: (N,40) multi-hot {0,1} factors
    """
    Z = np.asarray(labels_40)
    if Z.ndim != 2 or Z.shape[1] != 40:
        raise ValueError(f"Expected labels shape (N,40), got {Z.shape}")
    # Ensure discrete 0/1
    Z = (Z > 0.5).astype(int)

    C_active = _drop_constant_dims(C)
    if C_active.shape[1] == 0:
        return {"IRS": 0.0, "IRS_matrix": np.zeros((0, Z.shape[1]))}

    return scalable_irs_score(Z, C_active, diff_quantile=diff_quantile)

def compute_irs(C: np.ndarray, Z: np.ndarray, diff_quantile: float = 0.99):
    """
    Convenience for your setup:
      C: (N,L) code computed WITHOUT feeding the label-view
      Z: (N,K) factors (can be discrete or continuous)
    """
    C_active = _drop_constant_dims(C)
    if C_active.shape[1] == 0:
        return {"IRS": 0.0, "IRS_matrix": np.zeros((0, Z.shape[1]))}

    return scalable_irs_score(Z, C_active, diff_quantile=diff_quantile)

def report_irs(irs_result: dict, title: str = "IRS"):
    print(f"{title} score: {irs_result['IRS']:.4f}")
    print("Per-latent disentanglement scores:", irs_result["disentanglement_scores"])
    print("Per-latent parents:", irs_result["parents"])
    print("Max deviations per latent:", irs_result["max_deviations"])