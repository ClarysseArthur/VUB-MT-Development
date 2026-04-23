import numpy as np
import sklearn as sk

def _discretize_latents(latents: np.ndarray, num_bins: int = 50):
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


def _calculate_mutual_information_matrix(gen_factors: np.ndarray, latents: np.ndarray) -> np.ndarray:
    N, K = gen_factors.shape                                                                                # Number of samples N and number of generative factors K
    _, L = latents.shape                                                                                    # Number of latent dimensions L

    mi_matrix = np.zeros((L, K), dtype=float)

    for i in range(L):  # For each latent dimension
        for j in range(K):  # For each generative factor
            mi_matrix[i, j] = sk.metrics.mutual_info_score(latents[:, i], gen_factors[:, j])

    return mi_matrix

def _drop_constant_dims(latents: np.ndarray) -> np.ndarray:
    """Remove latent dimensions with zero variance."""
    latents = np.asarray(latents, dtype=float)
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2D (N, L), got {latents.shape}")

    variances = np.var(latents, axis=0)
    active_mask = variances > 0.0
    return latents[:, active_mask]