"""Estimating power for rare-variant association methods from PAGEANT."""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta, gamma, ncx2


class RareVariantPower:
    """Power calculator for rare-variant power."""

    def __init__(self):
        """Initialize rare-variant power calculator."""
        pass

    def llr_power(self, alpha=1e-6, df=1, ncp=1, ncp0=0):
        """Power under a non-central chi-squared distribution.

        Args:
            alpha (`float`): p-value threshold for GWAS
            df (`int`): degrees of freedom
            ncp (`float`): non-centrality parameter
            ncp0 (`float`): null non-centrality parameter

        Returns:
            power (`float`): power for association

        """
        assert df > 0
        assert (alpha > 0) & (alpha < 1)
        return 1.0 - ncx2.cdf(ncx2.ppf(1.0 - alpha, df, ncp0), df, ncp)

    def sim_af_weights(
        self, j=100, a1=0.1846, b1=11.1248, n=100, clip=True, seed=42, test="SKAT"
    ):
        """Simulate allele frequencies from a beta distribution.

        Ideally the beta distribution is derived from realized catalogues of variation.
        The current parameters are based on 15k African ancestry individuals.
        For mimicing a much larger set (112k) of Non-Finnish European individuals,
        use the parameters a1=0.14311324240262455, b1=26.97369198989023,

        Args:
            j (`int`): number of variants
            a1 (`float`): shape parameter of a gamma distribution
            b1 (`float`): scale parameter of a gamma distribution
            n (`float`): number of samples
            clip (`boolean`): perform clipping based on the current sample-size.
            seed (`int`): random seed.
            test (`string`): type of test to be performed (SKAT, Calpha, Hotelling)

        Returns:
            ws (`np.array`): array of weights per-variant.
            ps (`np.array`): array of allele frequencies.

        """
        assert j > 0
        assert a1 > 0
        assert b1 > 0
        assert n > 0
        assert seed > 0
        assert test in ["SKAT", "Calpha", "Hotelling"]
        np.random.seed(seed)
        ps = beta.rvs(a1, b1, size=j, random_state=seed)
        if clip:
            ps = np.clip(ps, 1.0 / n, (1 - 1.0 / n))
        if test == "SKAT":
            ws = beta.pdf(ps, 1.0, 1.0) ** 2
        elif test == "Calpha":
            ws = beta.pdf(ps, 1.0, 25.0) ** 2
        else:
            ws = beta.pdf(ps, 0.5, 0.5) ** 2
        return ws, ps

    def sim_var_per_gene(self, a=1.47, b=0.0108, seed=42):
        """Simulate the number of variants per-gene.

        Parameter values are derived from GnomAD Exonic variants on Chromosome 4 from ~15730 AFR ancestry subjects.

        For a Non-Finnish European ancestry setting with larger sample size (~112350), use a=1.44306, b=0.00372.

        Args:
            a (`float`):  shape parameter for a gamma distribution
            b (`float`): scale parameter for a gamma distribution
            seed (`int`): random seed.

        Returns:
            nvar (`int`): number of variants per-gene.

        """
        assert a > 0
        assert b > 0
        assert seed > 0
        nvar = gamma.rvs(a, scale=1 / b, random_state=seed)
        nvar = np.round(nvar).astype("int")
        if nvar <= 1:
            nvar = 1
        return nvar


class RareVariantBurdenPower(RareVariantPower):
    """Approximation of power for rare-variant burden tests based on results from Derkach et al (2018)."""

    def __init__(self):
        """Initialize the power calculator for the burden tests."""
        super(RareVariantBurdenPower, self).__init__()

    def ncp_burden_test_model1(self, n=100, j=30, jd=10, jp=0, tev=0.1):
        """Approximation of the non-centrality parameter under model S1 from Derkach et al.

        The key assumption in this case is that there is independence between an alleles effect-size and its MAF.

        Args:
            n (`int`): total sample size
            j (`int`): total number of variants in the gene
            jd (`int`): number of disease variants in the gene
            jp (`int`): number of protective variants in the gene
            tev (`float`): proportion of variance explained by gene

        Returns:
           ncp (`float`): non-centrality parameter

        """
        assert n > 0
        assert j > 0
        assert (jd + jp) <= j
        assert (tev > 0) & (tev < 1.0)
        ncp = n * ((jd - jp) ** 2) * tev / (j * (jd + jp))
        return ncp

    def power_burden_model1(
        self, n=100, j=30, prop_causal=0.80, prop_risk=0.1, tev=0.1, alpha=1e-6
    ):
        """Estimate the power under a burden model 1.

        Args:
            n (`int`): total sample size.
            j (`int`): total number of variants in the gene.
            prop_causal (`float`): proportion of causal variants.
            prop_risk (`float`): number of protective variants.
            tev (`float`): proportion of variance explained by gene.
            alpha (`float`): p-value threshold for power.

        Returns:
           power (`float`): power for detection under the burden model.

        """
        assert (prop_causal > 0.0) & (prop_causal <= 1.0)
        assert (prop_risk > 0.0) & (prop_risk <= 1.0)
        j_causal = j * prop_causal
        jd = j_causal * prop_risk
        jp = j_causal * (1 - prop_risk)
        ncp = self.ncp_burden_test_model1(n=n, j=j, jd=jd, jp=jp, tev=tev)
        return self.llr_power(alpha=alpha, ncp=ncp)

    def beta_power_burden_model1(
        self, n=100, j=30, prop_causal=0.80, prop_risk=0.5, alpha=1e-6, power=0.80
    ):
        """Estimate the total explained variance by a region for adequate detection at a power threshold.

        Args:
            n (`int`): total sample size.
            j (`int`): total number of variants in the gene.
            prop_causal (`float`): proportion of causal variants.
            prop_risk (`float`): number of protective variants.
            alpha (`float`): p-value threshold for power.
            power (`float`): power for detection under the burden model.

        Returns:
            opt_tev (`float`): TEV required for detection at this rate.

        """
        assert (power > 0) & (power <= 1.0)
        f = (
            lambda t: self.power_burden_model1(
                n=n,
                j=j,
                prop_causal=prop_causal,
                prop_risk=prop_risk,
                tev=t,
                alpha=alpha,
            )
            - power
        )
        try:
            opt_tev = brentq(f, 1e-32, 1.0)
        except (OverflowError, ValueError):
            opt_tev = np.nan
        return opt_tev

    def burden_model1_opt_n(
        self, j=30, tev=0.01, prop_causal=0.80, prop_risk=0.5, alpha=1e-6, power=0.80
    ):
        """Estimate the sample-size required for detection of supplied TEV in a region.

        Args:
            j (`int`): total number of variants in the gene.
            tev (`float`): proportion of variance explained by gene.
            prop_causal (`float`): proportion of causal variants.
            prop_risk (`float`): number of protective variants.
            alpha (`float`): p-value threshold for power.
            power (`float`): power for detection under the burden model.

        Returns:
            opt_tev (`float`): TEV required for detection at this rate.

        """
        assert (power > 0) & (power <= 1.0)
        f = (
            lambda n: self.power_burden_model1(
                n=n,
                j=j,
                prop_causal=prop_causal,
                prop_risk=prop_risk,
                tev=tev,
                alpha=alpha,
            )
            - power
        )
        try:
            opt_n = brentq(f, 1, 1e9)
        except (OverflowError, ValueError):
            opt_n = np.nan
        return opt_n

    def power_burden_model1_real(self, n=100, nreps=10, **kwargs):
        """Estimate power under model 1 with realistic numbers of variants per gene.

        Args:
            n (`int`): number of samples
            nreps (`int`): number of replicates

        Returns:
            est_power (`np.array`): array of power estimates based on realistic number of variants.

        """
        assert n > 0
        assert nreps > 0
        est_power = np.zeros(nreps)
        for i in range(nreps):
            # Actually simulating the number of variants per-gene
            j = self.sim_var_per_gene(seed=(i + 1))
            est_power[i] = self.power_burden_model1(n=n, j=j, **kwargs)
        return est_power

    def ncp_burden_test_model2(self, ws, ps, n=100, jd=10, jp=0, tev=0.1):
        """Estimate the non-centrality parameter for burden under Model 2.

        Args:
            ws (`np.array`): array of weights for alleles.
            ps (`np.array`): array of variant allele frequencies.
            n (`int`): sample-size.
            jd (`int`): number of disease variants.
            jp (`int`): number of protective variants.
            tev (`int`): total explained variance in trait of locus.

        Returns:
            ncp (`float`): non-centrality parameter for chi-squared distribution

        """
        assert ws.size == ps.size
        assert n > 0
        assert jd >= 0
        assert jp >= 0
        assert jd + jp > 0
        assert (tev > 0) & (tev < 1.0)
        j = ws.size
        sum_weights = np.sum((ws**2) * (ps**2) * (1.0 - ps))
        sum_test = np.sum((ws**2) * ps * (1.0 - ps))
        ncp = n * ((jd - jp) ** 2) * tev * sum_weights / (j * np.sum(ps * sum_test))
        return ncp

    def power_burden_model2(
        self, ws, ps, n=100, prop_causal=0.8, prop_risk=0.1, tev=0.1, alpha=1e-6
    ):
        """Estimate power under burden for model 2.

        Args:
            ws (`np.array`): array of weights for alleles.
            ps (`np.array`): array of variant allele frequencies.
            n (`int`): sample-size.
            prop_causal (`float`): proportion of causal variants.
            prop_risk (`float`): number of protective variants.
            tev (`int`): total explained variance in trait of locus.
            alpha (`float`): p-value threshold for power.

        Returns:
            power  (`float`): power under model2 for burden test

        """
        assert ws.ndim == 1
        assert ws.size == ps.size
        assert (prop_causal > 0.0) & (prop_causal <= 1.0)
        assert (prop_risk > 0.0) & (prop_risk <= 1.0)
        j = ws.size
        jc = j * prop_causal
        jd = jc * prop_risk
        jp = jc * (1.0 - prop_risk)
        ncp = self.ncp_burden_test_model2(ws, ps, jd=jd, jp=jp, n=n, tev=tev)
        return self.llr_power(alpha=alpha, ncp=ncp)

    def power_burden_model2_real(self, n=100, nreps=10, test="SKAT", **kwargs):
        """Estimate power under model 2 with realistic numbers of variants per gene.

        Args:
            n (`int`): number of samples.
            nreps (`int`): number of replicates.
            test (`string`): type of weighting scheme for allele frequencies.

        Returns:
            est_power (`np.array`): array of power estimates based on realistic number of variants.

        """
        assert n > 0
        assert nreps > 0
        est_power = np.zeros(nreps)
        for i in range(nreps):
            # Simulating the number of variants per-gene.
            j = self.sim_var_per_gene(seed=(i + 1))
            # Simulating the weights and allele frequencies.
            ws, ps = self.sim_af_weights(j=j, n=n, clip=True, test=test, seed=(i + 1))
            est_power[i] = self.power_burden_model2(ws=ws, ps=ps, n=n, **kwargs)
        return est_power


class RareVariantVCPower(RareVariantPower):
    """Approximation of power for rare-variant variance component tests based on results from Derkach et al (2018)."""

    def __init__(self):
        """Initialize the power calculator for the variance-component tests."""
        super(RareVariantVCPower, self).__init__()

    def match_cumulants_ncp(self, c1, c2, c3, c4):
        """Obtain the degrees of freedom and non-centrality parameter from cumulants.

        Args:
            c1 (`float`): first cumulant of non-central chi-squared dist.
            c2 (`float`): second cumulant of non-central chi-squared dist.
            c3 (`float`): third cumulant of non-central chi-squared dist.
            c4 (`float`): fourth cumulant of non-central chi-squared dist.
        Returns:
            df (`int`): degrees of freedom for test.
            ncp (`float`): non-centrality parameter.

        """
        s1 = c3 / c2 ** (3 / 2)
        s2 = c4 / c2**2
        if (s1**2) > s2:
            a = 1 / (s1 - np.sqrt(s1**2 - s2))
            ncp = s1 * a**3 - a**2
            df = a**2 - 2 * ncp
        else:
            ncp = 0.0
            df = c2**3 / c3**2
        return df, ncp

    def ncp_vc_first_order_model1(self, ws, ps, n=100, tev=0.1):
        """Approximation of the non-centrality parameter under model S1 from Derkach et al.

        The key assumption is independence between an alleles effect-size and its MAF, from Table S1 in Derkach et al.

        Args:
            ws (`np.array`): numpy array of weights per-variant.
            ps (`np.array`): numpy array of allele frequencies.
            n (`int`): sample size.
            tev (`float`): total explained variance by a locus.

        Returns:
           ncp (`float`): non-centrality parameter

        """
        assert ws.size == ps.size
        assert n > 0
        assert (tev >= 0) & (tev <= 1.0)
        j = ws.size
        lambdas = n * ws * ps * (1.0 - ps)
        c1 = np.sum(lambdas * (1 + 1 / j * tev))
        c2 = np.sum((lambdas**2) * (1 + 2 / j * tev))
        c3 = np.sum((lambdas**3) * (1 + 3 / j * tev))
        c4 = np.sum((lambdas**4) * (1 + 4 / j * tev))
        _, ncp = self.match_cumulants_ncp(c1, c2, c3, c4)
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
           power (`float`): estimated power under this variance component model.

        """
        assert ws.ndim == 1
        assert ps.ndim == 1
        assert n > 0
        ncp0 = self.ncp_vc_first_order_model1(ws, ps, n, 0.0)
        ncp = self.ncp_vc_first_order_model1(ws, ps, n, tev)
        return self.llr_power(df=df, ncp0=ncp0, ncp=ncp, alpha=alpha)

    def effect_size_vc_first_order_model1(
        self, ws, ps, n=100, power=0.8, alpha=1e-6, df=1
    ):
        """Estimate the total explained variance to discover an effect with this degree of power.

        Args:
            ws (`np.array`): numpy array of weights per-variant
            ps (`np.array`): numpy array of allele frequencies
            n (`int`): sample size (N)
            power (`float`): set level of power to determine impact.
            alpha (`float`): level of significance for the rare-variant test
            df (`int`): degrees of freedom for test

        Returns:
            opt_tev: the total explained variance by the gene detectable at this power

        """
        assert ws.ndim == 1
        assert ps.ndim == 1
        assert ws.size == ps.size
        assert df > 0
        assert n > 0
        assert (power > 0) & (power < 1)
        f = (
            lambda tev: self.power_vc_first_order_model1(
                ws, ps, n=n, tev=tev, alpha=alpha, df=df
            )
            - power
        )
        try:
            opt_tev = brentq(f, 0.0, 1.0)
        except (OverflowError, ValueError):
            opt_tev = np.nan
        return opt_tev
