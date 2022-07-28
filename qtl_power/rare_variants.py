import numpy as np
from scipy.optimize import brentq
from scipy.stats import gamma, ncx2


class RareVariantBurdenPower:
    """
    Approximation of power for rare-variant burden tests based on results from Derkach et al (2018)
    """

    def __init__(self):
        pass

    def llr_power(self, alpha=5e-8, df=1, ncp=1):
        return 1.0 - ncx2.cdf(ncx2.ppf(1.0 - alpha, df, 0), df, ncp)

    def ncp_burden_test_model1(self, n=100, j=30, jd=10, jp=10, tev=0.1):
        """Approximation of the non-centrality parameter under model S1 from Derkach et al.

        The key assumption in this case is that there is independence between an alleles effect-size and its MAF.
        This is also known in the literature as the alpha=0 model.

        Args:
            - n: total sample size
            - j: total number of variants in the gene
            - jd: number of dises variants in the gene
            - jp: number of protective variants in the gene
            - tev: proportion of variance explained by gene
        """
        assert (tev > 0) & (tev < 1.0)
        ncp = n * ((jd - jp) ** 2) * tev / (j * (jd + jp))
        return ncp

    def power_burden_model1(self, n=100, j=30, jd=10, jp=10, tev=0.1, alpha=1e-6):
        """Estimate the power under a burden model 1."""
        ncp = self.ncp_burden_test_model1(n=n, j=j, jd=jd, jp=jp, tev=tev)
        return self.llr_power(alpha=alpha, ncp=ncp)

    def ncp_burden_test_model2(self, ws, ps, jd=10, jp=10, n=100, tev=0.1):
        """Estimate NCP burden under Model 2."""
        assert ws.size == ps.size
        assert n > 0
        assert jd > 0
        assert jp > 0
        assert (tev > 0) & (tev < 1.0)
        j = ws.size
        sum_weights = np.sum((ws**2) * (ps**2) * (1.0 - ps))
        sum_test = np.sum((ws**2) * ps * (1.0 - ps))
        ncp = n * ((jd - jp) ** 2) * tev * sum_weights / (j * np.sum(ps * sum_test))
        return ncp

    def power_burden_model2(self, ws, ps, jd=10, jp=10, n=100, tev=0.1, alpha=1e-6):
        """Estimate power under burden for model 2."""
        ncp = self.ncp_burden_test_model2(ws, ps, jd=jd, jp=jp, n=n, tev=tev)
        return self.llr_power(alpha=alpha, ncp=ncp)


class RareVariantVCPower:
    """Approximation of power for rare-variant variance component tests based on results from Derkach et al (2018)."""

    def __init__(self):
        pass

    def sim_af_weights(self, j=100, a1=1.0, b1=1.0):
        """Simulating allele frequencies from a gamma distribution.

        Args:
            j (`int`): number of variants
            a1 (`float`): shape parameter of a gamma distribution
            b1 (`float`): scale parameter of a gamma distribution
        """
        assert j > 0
        ps = gamma.rvs(a1, scale=1 / b1, size=j)
        return ps

    def llr_power(self, alpha=1e-6, df=1, ncp=1, ncp0=0):
        """We have to include a null NCP due to the weighted distribution."""
        return 1.0 - ncx2.cdf(ncx2.ppf(1.0 - alpha, df, ncp0), df, ncp)

    def match_cumulants_ncp(self, c1, c2, c3, c4):
        """Match the cumulants to a single non-centrality parameter value."""
        f1 = lambda l: (1.0 + l) - c1
        f2 = lambda l: (2.0 + 4 * l) - c2
        f3 = lambda l: (8.0 + 24 * l) - c3
        f4 = lambda l: (48.0 + 192 * l) - c4
        f_tot = lambda l: f1(l) + f2(l) + f3(l) + f4(l)
        opt_ncp = brentq(f_tot, 0.0, 1e3)
        return opt_ncp

    def ncp_vc_first_order_model1(self, ws, ps, n=100, tev=0.1):
        """Approximation of the non-centrality parameter under model S1 from Derkach et al.

        The key assumption in this case is that there is independence between an alleles effect-size and its MAF.

        This is from Table S1 in Derkach et al.
        """
        assert ws.size == ps.size
        assert n > 0
        assert (tev >= 0) & (tev < 1.0)
        j = ws.size
        lambdas = n * ws * ps * (1.0 - ps)
        c1 = np.sum(lambdas * (1 + 1 / j * tev))
        c2 = np.sum((lambdas**2) * (1 + 2 / j * tev))
        c3 = np.sum((lambdas**3) * (1 + 3 / j * tev))
        c4 = np.sum((lambdas**4) * (1 + 4 / j * tev))
        ncp = self.match_cumulants_ncp(c1, c2, c3, c4)
        return ncp

    def power_vc_first_order_model1(self, ws, ps, n=100, tev=0.1, alpha=1e-6, df=1):
        """Compute the power for detection under model 1.

        Args:
            ws (`np.array`): numpy array of weights per-variant
            ps (`np.array`): numpy array of allele frequencies
            n (`int`): sample size
            tev (`float`): total explained variance by a locus
            alpha (`float`): total significance level for estimation of power

        Returns:
            Predicted power under a chi-squared distribution...
        """
        ncp0 = self.ncp_vc_first_order_model1(ws, ps, n, 0.0)
        ncp = self.ncp_vc_first_order_model1(ws, ps, n, tev)
        return self.llr_power(df=df, ncp0=ncp0, ncp=ncp, alpha=alpha)

    def effect_size_vc_first_order_model1(
        self, ws, ps, n=100, power=0.8, alpha=1e-6, df=1
    ):
        """The total explained variance to discover an effect with this degree of power.

        Args:
            ws (`np.array`): numpy array of weights per-variant
            ps (`np.array`): numpy array of allele frequencies
            n (`int`): sample size (N)
            power (`float`): set level of power to det
            alpha (`float`): level of significance for
            df (`int`): degrees of freedom for test

        Return:
            - opt_tev: the total explained variance by the gene detectable at this power
        """
        assert ws.ndim == 1
        assert ps.ndim == 1
        assert ws.size == ps.size
        assert n > 0
        assert (power > 0) & (power < 1)
        f = (
            lambda tev: self.power_vc_first_order_model1(
                ws, ps, n=n, tev=tev, alpha=alpha, df=df
            )
            - power
        )
        opt_tev = brentq(f, 0.0, 1.0)
        return opt_tev
