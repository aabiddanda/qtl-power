"""Functions to calculate power based on LMM approximations."""
import numpy as np
from gwas import Gwas


class LMM_Quant(Gwas):
    """Initialize an LMM power calculator for a quantitative trait."""

    def __init__(self):
        """Initialize super-class for LMM_Quant."""
        super(LMM_Quant, self).__init__()


class LMM_GxE(Gwas):
    """LMM-based power calculator for GxE effects."""

    def __init__(self):
        """Initialize superclass for LMM_GxE."""
        super(LMM_GxE, self).__init__()

    def ncp_gxe_unrelated(n=1000, delta=0.01, sigma2r=0.5, seed=42):
        """Compute the approximate NCP for a GxE test in the unrelated setting."""
        if seed is not None:
            np.random.seed(seed)
        # Simulate an environmental exposure ...
        d = np.random.uniform(size=n)
        dstd = ((d - np.mean(d)) / np.std(d)) ** 2
        denom = np.sum(1 / sigma2r * dstd)
        ncp = (delta**2) / denom
        return ncp
