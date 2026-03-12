def decorrelation_loss(h):
    """
    h: (B, D)
    Penalizes off-diagonal covariance terms.
    """
    h = h - h.mean(dim=0, keepdim=True)
    B = h.size(0)
    cov = (h.T @ h) / max(B - 1, 1)   # (D, D)

    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).mean()

def variance_loss(h, target_var=1.0):
    """
    Encourages each latent dimension to have non-zero, stable variance.
    """
    var = h.var(dim=0, unbiased=False)
    return ((var - target_var) ** 2).mean()

def sparsity_loss(h):
    return h.abs().mean()

def assignment_entropy_loss(weight_matrix, eps=1e-12):
    """
    weight_matrix: (n_factors, h_dim)
    Encourages each factor to mainly depend on one latent dim.
    """
    p = torch.softmax(weight_matrix.abs(), dim=1)
    ent = -(p * torch.log(p + eps)).sum(dim=1)
    return ent.mean()