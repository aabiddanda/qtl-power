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
        """
        assert (tev > 0) & (tev < 1.0)
        ncp = n * ((jd - jp) ** 2) * tev / (j * (jd + jp))
        return ncp

    def power_burden_model1(self, n=100, j=30, jd=10, jp=10, tev=0.1):
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

    def power_burden_model2(self, ws, ps, jd=10, jp=10, n=100, tev=0.1):
        """Estimate power under burden for model 2."""
        ncp = self.ncp_burden_test_model2(ws, ps, jd=jd, jp=jp, n=n, tev=tev)
        return self.llr_power(alpha=alpha, ncp=ncp)


class RareVariantVCPower:
    """Approximation of power for rare-variant variance component tests based on results from Derkach et al (2018)."""

    def __init__(self):
        pass

    def match_cumulants_ncp(self, c1, c2, c3, c4):
        """Match the cumulants to a single ncp value."""
        f1 = lambda l: (1.0 + l) - c1
        f2 = lambda l: (2.0 + 4 * l) - c2
        f3 = lambda l: (8.0 + 24 * l) - c3
        f4 = lambda l: (48.0 + 192 * l) - c4
        f_tot = lambda l: f1(l) + f2(l) + f3(l) + f4(l)
        opt_ncp = brentq(f_tot, 0.0, 1e3)
        return opt_ncp

    def cumulants_vc_first_order_model1(self, ws, ps, n=100, tev=0.1):
        """Approximation of the non-centrality parameter under model S1 from Derkach et al.

        The key assumption in this case is that there is independence between an alleles effect-size and its MAF.
        This is also known in the literature as the alpha=0 model.
        """
        assert ws.size == ps.size
        assert n > 0
        assert (tev > 0) & (tev < 1.0)
        j = ws.size
        lambdas = n * ws * ps * (1.0 - ps)
        c1 = (np.sum(lambdas) / j) * (tev / j)
        c2 = (np.sum(lambdas**2) / j) * (tev / j)
        c3 = (np.sum(lambdas**3) / j) * (tev / j)
        c4 = (np.sum(lambdas**4) / j) * (tev / j)
        ncp = self.match_cumulants_ncp(c1, c2, c3, c4)
        return ncp

    def cumulants_vc_first_order_model2(self, ws, ps, n=100, tev=0.1):
        """Approximation of the non-centrality parameter under model S2 from Derkach et al."""
        assert ws.size == ps.size
        assert n > 0
        assert (tev > 0) & (tev < 1.0)
        pass

    def sim_af_weights(self, n=100, a1=1.0, b1=1.0):
        """Simulating allele frequencies from a gamma distribution.
        NOTE: this is just under a specific model ...
        """
        ps = gamma.rvs(a1, scale=1 / b1, size=n)
        ws = 1.0 / np.sqrt(ps * (1.0 - ps))
        return ws, ps
