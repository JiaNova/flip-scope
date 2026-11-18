"""Historical Tweedie score comparator retained for Adult reproduction."""
from __future__ import annotations
import numpy as np

def tweedie_flip_probability(
    z_eval: np.ndarray | float,
    z_ref: np.ndarray,
    sigma2: float,
    *,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Estimate alert flip risk after correcting selection with Tweedie's formula.

    The marginal density of the deployment's calibration margins is estimated
    with a Gaussian KDE and Silverman's robust bandwidth.  Its log derivative
    is evaluated on a 2001-point grid, then interpolated at ``z_eval``.
    Set ``return_diagnostics`` to also receive the posterior-mean estimates,
    bandwidth, and probability clipping fraction.
    """
    from scipy.special import ndtr

    evaluation = np.atleast_1d(np.asarray(z_eval, dtype=np.float64))
    reference = np.asarray(z_ref, dtype=np.float64)
    variance = float(sigma2)
    if (evaluation.ndim != 1 or reference.ndim != 1 or len(reference) < 2
            or not np.isfinite(evaluation).all() or not np.isfinite(reference).all()
            or not np.isfinite(variance) or variance <= 0):
        raise ValueError("Tweedie inputs need finite one-dimensional margins and positive sigma2")
    sd = float(np.std(reference, ddof=1))
    q25, q75 = np.quantile(reference, [0.25, 0.75])
    robust_sd = float((q75 - q25) / 1.349)
    scale = min(sd, robust_sd)
    bandwidth = 1.06 * scale * len(reference) ** (-0.2)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Silverman bandwidth is zero or invalid for calibration margins")

    lower = float(np.min(evaluation) - 1.0)
    upper = float(np.max(evaluation) + 1.0)
    grid = np.linspace(lower, upper, 2001, dtype=np.float64)
    log_density_derivative = np.empty_like(grid)
    inverse_variance = 1.0 / (bandwidth * bandwidth)
    # The log-sum-exp normalization keeps the derivative stable even at the
    # edges of the registered [min(z_eval)-1, max(z_eval)+1] grid.
    block_size = 64
    for start in range(0, len(grid), block_size):
        block = grid[start:start + block_size]
        delta = block[:, None] - reference[None, :]
        log_weight = -0.5 * np.square(delta / bandwidth)
        log_weight -= np.max(log_weight, axis=1, keepdims=True)
        weight = np.exp(log_weight)
        denominator = np.sum(weight, axis=1)
        numerator = np.sum(weight * (-delta * inverse_variance), axis=1)
        log_density_derivative[start:start + len(block)] = numerator / denominator

    derivative = np.interp(evaluation, grid, log_density_derivative)
    posterior_mean = evaluation + variance * derivative
    raw = ndtr(-np.sign(evaluation) * posterior_mean / np.sqrt(2.0 * variance))
    clipped = np.clip(raw, 0.0, 0.5)
    if not return_diagnostics:
        return clipped
    details: dict[str, object] = {
        "bandwidth": float(bandwidth),
        "grid_size": int(len(grid)),
        "grid_min": lower,
        "grid_max": upper,
        "log_density_derivative": derivative,
        "posterior_mean": posterior_mean,
        "raw_probability": raw,
        "clipped_proportion": float(np.mean(raw != clipped)) if raw.size else 0.0,
    }
    return clipped, details
