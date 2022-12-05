"""Power calculations for extreme phenotype sampling designs."""

import numpy as np
from scipy.stats import fisher_exact


class ExtremePhenotype:
    """Class defining extreme phenotype designs."""

    def __init__(self):
        """Initialize the class for an extreme phenotype design."""
        pass

    def sim_extreme_pheno(self, n=100, maf=0.01, beta=0.1, seed=42):
        """Simulate an extreme phenotype under an HWE assumption.

        Args:
            n (`int`): total sample size.
            maf (`float`): minor allele frequency of tested variant.
            beta (`float`): effect-size in standard deviations.
            seed (`int`): random seed for simulations.
        Returns:
            allele_count (`np.array`): vector of allele-counts.
            phenotypes (`np.array`): quantitative phenotypes.

        """
        assert seed > 0
        assert n > 0
        assert (maf > 0) & (maf <= 0.5)
        np.random.seed(seed)
        allele_count = np.random.binomial(2, maf, size=n)
        phenotype = np.random.normal(size=n) + beta * allele_count
        return allele_count, phenotype

    def est_power_extreme_pheno(
        self, n=100, maf=0.01, beta=0.1, niter=100, alpha=0.05, q0=0.1, q1=0.1
    ):
        """
        Estimate the power from an extreme-phenotype sampling design.

        Args:
            n (`int`): total sample size.
            maf (`float`): minor allele frequency of tested variant.
            beta (`float`): effect-size in standard deviations.
            niter (`int`): number of simulation iterations.
            alpha (`float`): significance threshold for Fishers Exact Test.
            q0 (`float`): bottom quantile to establish as controls (or low-extremes).
            q1 (`float`): upper quantile to establish as cases (or upper extremes).

        Returns:
            power (`float`): power of extreme sampling design

        """
        assert niter > 0
        assert q0 <= 0.5
        assert q1 <= 0.5
        assert (alpha > 0) and (alpha < 1.0)
        n_reject = 0
        for i in range(niter):
            ac, phenotype = self.sim_extreme_pheno(n=n, maf=maf, beta=beta, seed=i + 1)
            excontrols = phenotype <= np.quantile(phenotype, q0)  # bottom percentiles
            excases = phenotype >= np.quantile(phenotype, 1 - q1)  # top-percentiles
            table = np.array(
                [
                    [np.sum(ac[excases]), 2 * np.sum(excases) - np.sum(ac[excases])],
                    [
                        np.sum(ac[excontrols]),
                        2 * np.sum(excontrols) - np.sum(ac[excontrols]),
                    ],
                ]
            )
            odds_ratio, pval = fisher_exact(table)
            n_reject += pval < alpha
        return n_reject / niter
