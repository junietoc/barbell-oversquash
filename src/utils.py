import torch

def layer_condition(jac):
    """κ(J) = σ_max / σ_min ‑‑ usamos SVD completa (pequeño tamaño)."""
    U, S, Vh = torch.linalg.svd(jac)
    return (S.max() / S.min()).item()

def energy(h):
    """E(h) = ½‖h‖₂²  (convención de mec. estadística)."""
    return 0.5 * h.pow(2).sum().item()