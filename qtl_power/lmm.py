"""Functions to calculate power based on LMM approximations."""
import numpy as np

from qtl_power.gwas import Gwas


class Lmm_Quant(Gwas):
    """Initialize an LMM power calculator for a quantitative trait."""

    def __init__(self):
        """Initialize super-class for LMM_Quant."""
        super(Lmm_Quant, self).__init__()


class Lmm_GxE(Gwas):
    """LMM-based power calculator for GxE effects."""

    def __init__(self):
        """Initialize superclass for LMM_GxE."""
        super(Lmm_GxE, self).__init__()

    def ncp_gxe_unrelated(self, n=1000, delta=0.01, sigma2r=0.5, seed=42):
        """Compute the approximate NCP for a GxE test in the unrelated sample.

        Args:
            n (`int`): number of sanples.
            delta (`float`): variance explained by GxE.
            sigma2r (`float`): variance explained by residual effects.
            seed (`int`): random seed for simulated effects.
        Returns:
            ncp (`float`): non-centrality parameter

        """
        if seed is not None:
            np.random.seed(seed)
        # Simulate an environmental exposure ...
        d = np.random.uniform(size=n)
        dstd = ((d - np.mean(d)) / np.std(d)) ** 2
        denom = np.sum(1 / sigma2r * dstd)
        ncp = (delta**2) / denom
        return ncp

    def power_gxe_unrelated(self, alpha=5e-8, **kwargs):
        """Compute the power to detect a GxE standardized effect.

        Args:
            alpha (`float`): p-value threshold for GWAS.
            df (`int`): degrees of freedom.
            n (`int`): number of sanples.
            delta (`float`): variance explained by GxE.
            sigma2r (`float`): variance explained by residual effects.
            seed (`int`): random seed.
        Returns:
            power (`float`): power for association

        """
        ncp = self.ncp_gxe_unrelated(**kwargs)
        power = self.llr_power(ncp=ncp, alpha=alpha)
        return power
