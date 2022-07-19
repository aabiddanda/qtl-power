import numpy as np
from scipy.optimize import brentq
from scipy.stats import fisher_exact, ncx2


class GWAS_Power:
    def __init__(self):
        pass

    def llr_power(self, alpha=5e-8, df=1, ncp=1):
        return 1.0 - ncx2.cdf(ncx2.ppf(1.0 - alpha, df, 0), df, ncp)

    def ncp_quant(self, n=100, p=0.1, beta=0.1, r2=1.0):
        ncp = r2 * n * 2 * p * (1.0 - p) * (beta**2)
        return ncp

    def ncp_case_control(self, n=100, p=0.1, beta=0.1, r2=1.0, prop_cases=0.1):
        assert (prop_cases > 0) & (prop_cases < 1.0)
        ncp = r2 * n * 2 * p * (1.0 - p) * prop_cases * (1.0 - prop_cases) * (beta**2)
        return ncp

    def quant_trait_power(self, n=100, p=0.1, beta=0.1, r2=1.0, alpha=5e-8):
        """Power for a quantitative trait association study."""
        ncp = self.ncp_quant(n, p, beta, r2)
        return self.llr_power(alpha, df=1, ncp=ncp)

    def binary_trait_power(
        self, n=100, p=0.1, beta=0.1, r2=1.0, alpha=5e-8, prop_cases=0.1
    ):
        """Power under a case-control study design."""
        ncp = self.ncp_case_control(n, p, beta, r2, prop_cases)
        return self.llr_power(alpha, df=1, ncp=ncp)

    def quant_trait_beta_power(self, n=100, power=0.90, p=0.1, r2=1.0, alpha=5e-8):
        """Determine the effect-size required to detect an association at this MAF."""
        assert (power > 0) & (power < 1)
        f = (
            lambda beta: self.quant_trait_power(n=n, p=p, r2=r2, beta=beta, alpha=alpha)
            - power
        )
        opt_beta = brentq(f, 0.0, 1e2)
        return opt_beta

    def binary_trait_beta_power(
        self, n=100, power=0.90, p=0.1, r2=1.0, alpha=5e-8, prop_cases=0.5
    ):
        assert (power > 0) & (power < 1)
        f = (
            lambda beta: self.binary_trait_power(
                n=n, p=p, r2=r2, beta=beta, alpha=alpha, prop_cases=prop_cases
            )
            - power
        )
        opt_beta = brentq(f, 0.0, 1e2)
        return opt_beta


class ExtremePhenotype_Power:
    def __init__(self):
        pass

    # Extreme Phenotype Study Design
    def sim_extreme_pheno(self, n=100, maf=0.01, beta=0.1, seed=42):
        np.random.seed(seed)
        allele_count = np.random.binomial(2, maf, size=n)
        phenotype = np.random.normal(size=n) + beta * allele_count
        return allele_count, phenotype

    def est_power_extreme_pheno(
        self, n=100, maf=0.01, beta=0.1, niter=100, alpha=0.05, q0=0.1, q1=0.1
    ):
        """Estimate the power from an extreme-phenotype sampling design."""
        assert niter > 0
        n_reject = 0
        for i in range(niter):
            ac, phenotype = self.sim_extreme_pheno(n=n, maf=maf, beta=beta, seed=i)
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


# def power_pqtl_study(n=100, p=0.1, beta=0.1, r2=1.0, alpha=0.05):
#     """Calculating power in a pQTL study."""
#     ncp = n*r2*(1. - (2*p*(1-p)*(beta**2))) / (2*p*(1-p)*(beta**2))
#     ncp_alt = n * r2 * p *(1.-p) * (beta**2)
#     print(p, ncp, ncp_alt)
#     return llr_power(alpha, 1, ncp=ncp)
