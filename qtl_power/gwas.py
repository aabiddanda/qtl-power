"""Functions to calculate power in GWAS designs."""
import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import ncx2


class Gwas:
    """Parent class for GWAS Power calculation."""

    def __init__(self):
        """Initialize base class."""
        pass

    @staticmethod
    def genotype_var(af, var_g):
        """Return genotype variance: var_g if provided, else 2*af*(1-af)."""
        if var_g is not None:
            return var_g
        return 2.0 * af * (1.0 - af)

    def llr_power(self, alpha=5e-8, df=1, ncp=1):
        r"""Power under a non-central chi-squared distribution.

        .. math::

            \text{power} = 1 - F_{\chi^2(df,\,\lambda)}\!\left(q_{1-\alpha}\right)

        where :math:`q_{1-\alpha}` is the :math:`(1-\alpha)` quantile of the
        central :math:`\chi^2(df)` distribution and :math:`\lambda` is the
        non-centrality parameter.

        Args:
            alpha (`float`): p-value threshold for GWAS
            df (`int`): degrees of freedom
            ncp (`float`): non-centrality parameter :math:`\lambda`
        Returns:
            power (`float`): power for association

        """
        try:
            return 1.0 - ncx2.cdf(ncx2.ppf(1.0 - alpha, df, 0), df, ncp)
        except OverflowError:
            return np.nan


class GwasQuant(Gwas):
    r"""Power calculations for a GWAS on a standardised quantitative trait.

    The linear model is :math:`Y_i = \mu + \beta\,g_i + \varepsilon_i` with
    :math:`\varepsilon_i \sim N(0, 1)` and additive genotype
    :math:`g_i \in \{0, 1, 2\}` (HWE).  The score-test NCP is:

    .. math::

        \lambda = r^2 \, N \, \beta^2 \cdot 2\,\text{af}(1 - \text{af})

    where :math:`r^2` is the LD / imputation-accuracy squared correlation
    between the causal variant and the typed tag.

    For CNV predictors pass ``var_g`` = :math:`\text{Var}(C)` to replace
    :math:`2\,\text{af}(1-\text{af})` with the copy-number variance.
    """

    def __init__(self):
        """Initialize a GWAS power calculator for quantitative traits."""
        super(GwasQuant, self).__init__()

    def ncp_quant(self, n=100, af=0.1, beta=0.1, r2=1.0, var_g=None):
        r"""Non-centrality parameter for a quantitative trait GWAS.

        .. math::

            \lambda = r^2 \, N \, \beta^2 \cdot V_g

        where :math:`V_g = 2\,\text{af}(1-\text{af})` for SNPs or
        :math:`V_g = \text{Var}(C)` when ``var_g`` is supplied.

        Args:
            n (`int`): sample-size of unrelated individuals.
            af (`float`): allele frequency of variant (:math:`0 < \text{af} < 1`).
                Ignored when ``var_g`` is provided.
            beta (`float`): per-allele effect size in phenotypic-SD units.
            r2 (`float`): LD :math:`r^2` between causal and tagged variant
                (:math:`0 < r^2 \leq 1`).
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            ncp (`float`): non-centrality parameter :math:`\lambda`.

        """
        assert n > 0
        assert (r2 > 0) & (r2 <= 1.0)
        if var_g is None:
            assert (af > 0.0) and (af < 1.0)
        vg = self.genotype_var(af, var_g)
        return r2 * n * vg * (beta**2)

    def quant_trait_power(self, n=100, af=0.1, beta=0.1, r2=1.0, alpha=5e-8, var_g=None):
        r"""Power for a quantitative trait association study.

        Args:
            n (`int`): sample-size of unrelated individuals.
            af (`float`): allele frequency of variant (:math:`0 < \text{af} < 1`).
                Ignored when ``var_g`` is provided.
            beta (`float`): per-allele effect size in phenotypic-SD units.
            r2 (`float`): LD :math:`r^2` (:math:`0 < r^2 \leq 1`).
            alpha (`float`): p-value threshold for GWAS.
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            power (`float`): power in :math:`[0, 1]`.

        """
        ncp = self.ncp_quant(n, af, beta, r2, var_g=var_g)
        return self.llr_power(alpha, df=1, ncp=ncp)

    def quant_trait_beta_power(self, n=100, power=0.90, af=0.1, r2=1.0, alpha=5e-8, var_g=None):
        r"""Minimum detectable effect size at the target power level.

        Args:
            n (`int`): sample-size of unrelated individuals.
            power (`float`): target power level in :math:`(0, 1)`.
            af (`float`): allele frequency of variant (:math:`0 < \text{af} < 1`).
                Ignored when ``var_g`` is provided.
            r2 (`float`): LD :math:`r^2` (:math:`0 < r^2 \leq 1`).
            alpha (`float`): p-value threshold for GWAS.
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            opt_beta (`float`): minimum detectable :math:`|\beta|`.

        """
        assert (power >= 0) & (power <= 1)
        f = (
            lambda beta: self.quant_trait_power(
                n=n, af=af, r2=r2, beta=beta, alpha=alpha, var_g=var_g
            )
            - power
        )
        try:
            opt_beta = root_scalar(f, bracket=(0.0, 1e3)).root
        except (OverflowError, ValueError):
            opt_beta = np.nan
        return opt_beta

    def quant_trait_opt_n(self, beta=0.1, power=0.90, af=0.1, r2=1.0, alpha=5e-8, var_g=None):
        r"""Minimum sample size to achieve the target power.

        Args:
            beta (`float`): per-allele effect size in phenotypic-SD units.
            power (`float`): target power level in :math:`(0, 1)`.
            af (`float`): allele frequency of variant (:math:`0 < \text{af} < 1`).
                Ignored when ``var_g`` is provided.
            r2 (`float`): LD :math:`r^2` (:math:`0 < r^2 \leq 1`).
            alpha (`float`): p-value threshold for GWAS.
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.

        Returns:
            opt_n (`float`): required :math:`N` (fractional; take
                :math:`\lceil \cdot \rceil` in practice).

        """
        assert (power >= 0) & (power <= 1)
        f = (
            lambda n: self.quant_trait_power(n=n, af=af, r2=r2, beta=beta, alpha=alpha, var_g=var_g)
            - power
        )
        try:
            opt_n = root_scalar(f, bracket=(1e-24, 1e24)).root
        except (OverflowError, ValueError):
            opt_n = np.nan
        return opt_n


class GwasBinary(Gwas):
    r"""GWAS power calculator for a case/control study design.

    Under an additive model the score-test NCP is:

    .. math::

        \lambda = r^2 \, N \, \beta^2 \cdot 2\,\text{af}(1-\text{af}) \cdot K(1-K)

    where :math:`K` is the proportion of cases, :math:`\beta` is the
    per-allele log-OR (small-effect linear approximation), and :math:`r^2` is
    the LD / imputation-accuracy squared correlation.

    For CNV predictors pass ``var_g`` = :math:`\text{Var}(C)` to replace
    :math:`2\,\text{af}(1-\text{af})` with the copy-number variance.
    """

    def __init__(self):
        """Initialize a GWAS power calculator for case/control traits."""
        super(GwasBinary, self).__init__()

    def ncp_binary(self, n=100, af=0.1, beta=0.1, r2=1.0, prop_cases=0.1, var_g=None):
        r"""Non-centrality parameter for a case/control GWAS.

        .. math::

            \lambda = r^2 \, N \, \beta^2 \cdot V_g \cdot K(1-K)

        where :math:`K` = ``prop_cases`` and :math:`V_g = 2\,\text{af}(1-\text{af})`
        for SNPs or :math:`V_g = \text{Var}(C)` when ``var_g`` is supplied.

        Args:
            n (`int`): sample-size of unrelated individuals.
            af (`float`): allele frequency (:math:`0 \leq \text{af} \leq 1`).
                Ignored when ``var_g`` is provided.
            beta (`float`): per-allele effect size (linear approximation to log-OR).
            r2 (`float`): LD :math:`r^2` (:math:`0 \leq r^2 \leq 1`).
            prop_cases (`float`): proportion of samples that are cases
                (:math:`0 < K < 1`).
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            ncp (`float`): non-centrality parameter :math:`\lambda`.

        """
        assert n > 0
        assert (r2 >= 0) & (r2 <= 1.0)
        assert (prop_cases > 0) & (prop_cases < 1.0)
        if var_g is None:
            assert (af >= 0.0) and (af <= 1.0)
        vg = self.genotype_var(af, var_g)
        return r2 * n * vg * prop_cases * (1.0 - prop_cases) * (beta**2)

    def binary_trait_power(
        self, n=100, af=0.1, beta=0.1, r2=1.0, alpha=5e-8, prop_cases=0.1, var_g=None
    ):
        r"""Power under a case-control GWAS study design.

        Args:
            n (`int`): sample-size of unrelated individuals.
            af (`float`): allele frequency (:math:`0 < \text{af} < 1`).
                Ignored when ``var_g`` is provided.
            beta (`float`): per-allele effect size.
            r2 (`float`): LD :math:`r^2` (:math:`0 \leq r^2 \leq 1`).
            alpha (`float`): p-value threshold for detection.
            prop_cases (`float`): proportion of samples that are cases.
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            power (`float`): power in :math:`[0, 1]`.

        """
        ncp = self.ncp_binary(n, af, beta, r2, prop_cases, var_g=var_g)
        return self.llr_power(alpha, df=1, ncp=ncp)

    def binary_trait_beta_power(
        self, n=100, power=0.90, af=0.1, r2=1.0, alpha=5e-8, prop_cases=0.5, var_g=None
    ):
        r"""Minimum detectable effect size under a case-control GWAS study design.

        Args:
            n (`int`): sample-size of unrelated individuals.
            power (`float`): target power level in :math:`(0, 1)`.
            af (`float`): allele frequency (:math:`0 < \text{af} < 1`).
                Ignored when ``var_g`` is provided.
            r2 (`float`): LD :math:`r^2` (:math:`0 \leq r^2 \leq 1`).
            alpha (`float`): p-value threshold for detection.
            prop_cases (`float`): proportion of samples that are cases.
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.

        Returns:
            opt_beta (`float`): minimum detectable :math:`|\beta|`.

        """
        assert n > 0
        if var_g is None:
            assert (af > 0) & (af < 1)
        assert (r2 >= 0.0) & (r2 <= 1.0)
        assert (power > 0) & (power < 1)
        f = (
            lambda beta: self.binary_trait_power(
                n=n, af=af, r2=r2, beta=beta, alpha=alpha, prop_cases=prop_cases, var_g=var_g
            )
            - power
        )
        try:
            opt_beta = root_scalar(f, bracket=(0.0, 1e3)).root
        except (OverflowError, ValueError):
            opt_beta = np.nan
        return opt_beta

    def binary_trait_opt_n(
        self, beta=0.1, power=0.90, af=0.1, r2=1.0, alpha=5e-8, prop_cases=0.5, var_g=None
    ):
        r"""Minimum sample size to achieve the target power.

        Args:
            beta (`float`): per-allele effect size.
            power (`float`): target power level in :math:`(0, 1)`.
            af (`float`): allele frequency (:math:`0 < \text{af} < 1`).
                Ignored when ``var_g`` is provided.
            r2 (`float`): LD :math:`r^2` (:math:`0 \leq r^2 \leq 1`).
            alpha (`float`): p-value threshold for GWAS.
            prop_cases (`float`): proportion of cases in the dataset.
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.

        Returns:
            opt_n (`float`): required :math:`N` (fractional; take
                :math:`\lceil \cdot \rceil` in practice).

        """
        assert (power >= 0) & (power <= 1)
        f = (
            lambda n: self.binary_trait_power(
                n=n, af=af, r2=r2, beta=beta, alpha=alpha, prop_cases=prop_cases, var_g=var_g
            )
            - power
        )
        try:
            opt_n = root_scalar(f, bracket=(1.0, 1e24)).root
        except (OverflowError, ValueError):
            opt_n = np.nan
        return opt_n


class GwasBinaryModel(Gwas):
    """GWAS Power calculations under different encodings of genotypic risk."""

    def __init__(self):
        """Initialize a GWAS power calculator for case/control traits under different genotypic models."""
        super(GwasBinaryModel, self).__init__()

    def ncp_binary_model(
        self,
        n=100,
        af=0.1,
        beta=0.1,
        model="additive",
        prev=0.01,
        alpha=5e-8,
        prop_cases=0.5,
    ):
        """Explore how multiple models affect power in case-control traits."""
        assert (prev > 0) & (prev < 1.0)
        assert n > 0
        assert (af > 0) & (af < 1)
        if model == "additive":
            x = np.array([1.0 + 2 * beta, 1.0 + beta, 1.0])
        elif model == "dominant":
            x = np.array([1.0 + beta, 1.0 + beta, 1.0])
        elif model == "recessive":
            x = np.array([1.0 + beta, 1.0, 1.0])
        else:
            raise ValueError(
                f"Model should be additive|dominant|recessive, not {model}"
            )
        n_cases = n * prop_cases
        n_control = n * (1.0 - prop_cases)
        geno_freq = np.array([af**2, 2 * af * (1.0 - af), (1 - af) ** 2])
        denom = (x * geno_freq).sum()
        aa_prob = x[0] * prev / denom
        ab_prob = x[1] * prev / denom
        case_af = (aa_prob * geno_freq[0] + ab_prob * geno_freq[1] * 0.5) / prev
        control_af = (
            (1.0 - aa_prob) * geno_freq[0] + (1.0 - ab_prob) * geno_freq[1] * 0.5
        ) / (1.0 - prev)
        v_cases = case_af * (1.0 - case_af)
        v_control = control_af * (1.0 - control_af)
        ncp = (case_af - control_af) / (
            np.sqrt((v_cases / n_cases + v_control / n_control) * 0.5)
        )
        return ncp

    def binary_trait_power_model(
        self,
        n=100,
        af=0.1,
        beta=0.1,
        model="additive",
        prev=0.01,
        alpha=5e-8,
        prop_cases=0.5,
    ):
        """Power under a case-control GWAS study design.

        Args:
            n (`int`): sample-size of unrelated individuals.
            af (`float`): allele frequency of variant.
            beta (`float`): effect-size of variant (in terms of relative-risk).
            model (`string`): genetic model for effects (additive, recessive, or dominant).
            prev (`float`): prevalence of the trait in question.
            alpha (`float`): p-value threshold for detection.
            prop_cases (`float`): proportion of samples that are cases.

        Returns:
            power (`float`): power under the model.

        """
        ncp = self.ncp_binary_model(
            n=n,
            af=af,
            beta=beta,
            model=model,
            prev=prev,
            alpha=alpha,
            prop_cases=prop_cases,
        )
        return self.llr_power(alpha, df=1, ncp=ncp)

    def binary_trait_beta_power_model(
        self,
        n=100,
        af=0.1,
        model="additive",
        prev=0.01,
        alpha=5e-8,
        prop_cases=0.5,
        power=0.90,
    ):
        """Threshold effects under a specific power threshold and genetic model.

        Args:
            n (`int`): sample-size of unrelated individuals.
            af (`float`): allele frequency of variant.
            beta (`float`): effect-size of variant (in terms of relative-risk).
            model (`string`): genetic model for effects (additive, recessive, or dominant).
            prev (`float`): prevalence of the trait in question.
            alpha (`float`): p-value threshold for detection.
            prop_cases (`float`): proportion of samples that are cases.
            power (`float`): power under the model.

        Returns:
            opt_beta (`float`): detectable effect-size at the power threshold and model.

        """
        assert (power >= 0) & (power <= 1)
        f = (
            lambda beta: self.binary_trait_power_model(
                n=n,
                af=af,
                beta=beta,
                model=model,
                prev=prev,
                prop_cases=prop_cases,
                alpha=alpha,
            )
            - power
        )
        try:
            opt_beta = root_scalar(f, bracket=(0.0, 1e3)).root
        except (OverflowError, ValueError):
            opt_beta = np.nan
        return opt_beta


class GwasBinomialTrait(Gwas):
    r"""GWAS power calculator for a binomial count trait.

    Each individual contributes :math:`n_i` Bernoulli trials:

    .. math::

        Y_i \sim \text{Binomial}(n_i,\; p_i),\quad
        p_i = \mu + \beta\,(g_i - 2\,\text{af})

    where :math:`g_i \in \{0, 1, 2\}` is the additive genotype under HWE and
    the genotype is mean-centred so that :math:`\mu = E[p_i]` is the
    *population mean* success probability, invariant when sweeping af.

    The score-test non-centrality parameter is:

    .. math::

        \lambda = r^2 \, N \, \beta^2 \cdot 2\,\text{af}(1-\text{af}) \cdot
                  \frac{\bar{n}}{\mu(1-\mu)}

    where :math:`\bar{n}` is the mean number of trials per individual and
    :math:`r^2` is the LD / imputation-accuracy squared correlation between the
    causal variant and the typed/imputed tag (:math:`r^2 = 1` gives the
    perfectly-typed case).

    For CNV predictors pass ``var_g`` = :math:`\text{Var}(C)` to replace
    :math:`2\,\text{af}(1-\text{af})` with the copy-number variance.
    """

    def __init__(self, mu=0.5):
        """Initialize a binomial GWAS power calculator.

        Args:
            mu (`float`): population mean success probability (0 < mu < 1).

        """
        super().__init__()
        if not (0.0 < mu < 1.0):
            raise ValueError("mu must be strictly between 0 and 1.")
        self.mu = mu

    @staticmethod
    def mu_from_p0(p0, af, beta):
        r"""Convert baseline probability :math:`p_0 = \Pr(\text{success} \mid g=0)` to population mean :math:`\mu`.

        .. math::

            \mu = p_0 + 2\,\text{af}\cdot\beta

        Args:
            p0 (`float`): success probability for the :math:`g=0` genotype.
            af (`float`): allele frequency.
            beta (`float`): per-allele change in success probability.
        Returns:
            mu (`float`): population mean success probability.

        """
        return p0 + 2.0 * af * beta

    @staticmethod
    def beta_to_sd_units(beta, mu, n_mean):
        r"""Convert raw probability :math:`\beta` to phenotypic-SD units.

        The per-individual rate :math:`Y_i/n_i` has variance
        :math:`\mu(1-\mu)/\bar{n}`, so:

        .. math::

            \beta_\text{SD} = \frac{\beta}{\sqrt{\mu(1-\mu)/\bar{n}}}

        This is exact under the identity-link model.  Deeper sequencing (larger
        :math:`\bar{n}`) makes the same raw :math:`\beta` appear larger in SD
        units.

        Args:
            beta (`float`): per-allele change in success probability (raw units).
            mu (`float`): population mean success probability.
            n_mean (`float`): mean number of trials per individual :math:`\bar{n}`.
        Returns:
            beta_sd (`float`): effect size in units of phenotypic SD.

        """
        return beta * np.sqrt(n_mean / (mu * (1.0 - mu)))

    @staticmethod
    def sd_units_to_beta(beta_sd, mu, n_mean):
        r"""Convert SD-unit effect size back to raw probability units.

        Inverse of :meth:`beta_to_sd_units`:

        .. math::

            \beta = \beta_\text{SD} \cdot \sqrt{\frac{\mu(1-\mu)}{\bar{n}}}

        Args:
            beta_sd (`float`): effect size in units of phenotypic SD.
            mu (`float`): population mean success probability.
            n_mean (`float`): mean number of trials per individual :math:`\bar{n}`.
        Returns:
            beta (`float`): per-allele change in success probability.

        """
        return beta_sd / np.sqrt(n_mean / (mu * (1.0 - mu)))

    @staticmethod
    def beta_to_log_or(beta, mu):
        r"""Convert raw probability :math:`\beta` to an approximate log-odds ratio.

        First-order delta method on the logit transformation:

        .. math::

            \log\text{OR} \approx \frac{\beta}{\mu(1-\mu)}

        Valid when :math:`\beta` is small relative to :math:`\mu(1-\mu)`.
        Accuracy degrades when :math:`\mu` is near 0 or 1, or when the raw
        :math:`\beta` is large.

        Args:
            beta (`float`): per-allele change in success probability (raw units).
            mu (`float`): population mean success probability.
        Returns:
            log_or (`float`): approximate log-odds ratio per allele.

        """
        return beta / (mu * (1.0 - mu))

    @staticmethod
    def log_or_to_beta(log_or, mu):
        r"""Convert a log-odds ratio to an approximate raw probability :math:`\beta`.

        Inverse of :meth:`beta_to_log_or` (same small-effect approximation applies):

        .. math::

            \beta \approx \log\text{OR} \cdot \mu(1-\mu)

        Args:
            log_or (`float`): log-odds ratio per allele.
            mu (`float`): population mean success probability.
        Returns:
            beta (`float`): approximate per-allele change in success probability.

        """
        return log_or * mu * (1.0 - mu)

    def ncp_binomial(self, n=100, af=0.2, beta=0.05, n_mean=10.0, r2=1.0, var_g=None):
        r"""Non-centrality parameter for the binomial-trait score test.

        .. math::

            \lambda = r^2 \, N \, \beta^2 \cdot V_g \cdot
                      \frac{\bar{n}}{\mu(1-\mu)}

        where :math:`V_g = 2\,\text{af}(1-\text{af})` for SNPs or
        :math:`V_g = \text{Var}(C)` when ``var_g`` is supplied.

        Args:
            n (`int`): number of individuals.
            af (`float`): allele frequency (:math:`0 < \text{af} < 1`).
                Ignored when ``var_g`` is provided.
            beta (`float`): per-allele change in success probability.
            n_mean (`float`): mean number of Binomial trials per individual
                :math:`\bar{n}`.
            r2 (`float`): LD / imputation-accuracy :math:`r^2` between causal
                and typed variant (:math:`0 < r^2 \leq 1`).
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            ncp (`float`): non-centrality parameter :math:`\lambda`.

        """
        assert n > 0
        assert n_mean > 0
        assert 0.0 < r2 <= 1.0
        if var_g is None:
            assert 0.0 < af < 1.0
        vg = self.genotype_var(af, var_g)
        return r2 * n * beta**2 * vg * n_mean / (self.mu * (1.0 - self.mu))

    def ncp_binomial_sd(self, n=100, af=0.2, beta=0.05, n_mean=10.0, n_var=0.0, r2=1.0, var_g=None):
        r"""Standard deviation of the realised NCP due to variable trial counts.

        By the delta method:

        .. math::

            \text{Var}(\lambda) \approx
            \lambda^2 \cdot \frac{\text{Var}(\tilde{g}^2 n_i)}
                                  {\bigl(E[\tilde{g}^2]\,\bar{n}\bigr)^2 \, N}

        where :math:`\tilde{g} = g - 2\,\text{af}`.  Returns 0.0 when
        ``n_var`` = 0 (fixed-trial design).

        The higher-moment calculation requires the full :math:`\{0,1,2\}`
        genotype distribution and is therefore not supported when ``var_g``
        is supplied and ``n_var`` > 0.

        Args:
            n (`int`): number of individuals.
            af (`float`): allele frequency (:math:`0 < \text{af} < 1`).
            beta (`float`): per-allele change in success probability.
            n_mean (`float`): mean trials per individual :math:`\bar{n}`.
            n_var (`float`): variance of trials per individual (:math:`\geq 0`).
            r2 (`float`): LD / imputation-accuracy :math:`r^2`
                (:math:`0 < r^2 \leq 1`).
            var_g (`float`, optional): genotype / copy-number variance.  Supported
                only when ``n_var`` = 0 (fixed-trial design); raises
                ``NotImplementedError`` otherwise.
        Returns:
            sd (`float`): standard deviation of the realised NCP.

        """
        assert n_var >= 0.0
        if n_var == 0.0:
            return 0.0
        if var_g is not None:
            raise NotImplementedError(
                "ncp_binomial_sd with variable trial counts requires the full "
                "genotype distribution (etg4); supply var_g only when n_var=0."
            )
        lam = self.ncp_binomial(n, af, beta, n_mean, r2)
        vg = 2.0 * af * (1.0 - af)
        # E[(g - 2*af)^4] under HWE
        etg4 = (
            (-2 * af) ** 4 * (1 - af) ** 2
            + (1 - 2 * af) ** 4 * 2 * af * (1 - af)
            + (2 * (1 - af)) ** 4 * af**2
        )
        en2 = n_var + n_mean**2
        var_tg2n = etg4 * en2 - (vg * n_mean) ** 2
        cv2_denom = var_tg2n / ((vg * n_mean) ** 2 * n)
        return np.sqrt(lam**2 * cv2_denom)

    def binomial_trait_power(
        self, n=100, af=0.2, beta=0.05, n_mean=10.0, r2=1.0, alpha=5e-8, var_g=None
    ):
        r"""Power to detect the association under the binomial trait model.

        Args:
            n (`int`): number of individuals.
            af (`float`): allele frequency (:math:`0 < \text{af} < 1`).
                Ignored when ``var_g`` is provided.
            beta (`float`): per-allele change in success probability.
            n_mean (`float`): mean trials per individual :math:`\bar{n}`.
            r2 (`float`): LD / imputation-accuracy :math:`r^2`
                (:math:`0 < r^2 \leq 1`).
            alpha (`float`): p-value threshold.
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            power (`float`): power in :math:`[0, 1]`.

        """
        ncp = self.ncp_binomial(n, af, beta, n_mean, r2, var_g=var_g)
        return self.llr_power(alpha=alpha, df=1, ncp=ncp)

    def binomial_trait_power_with_nvar(
        self,
        n=100,
        af=0.2,
        beta=0.05,
        n_mean=10.0,
        n_var=0.0,
        n_sigma=1.0,
        r2=1.0,
        alpha=5e-8,
        var_g=None,
    ):
        r"""Power with :math:`\pm` ``n_sigma`` uncertainty bands from variable trial counts.

        Args:
            n (`int`): number of individuals.
            af (`float`): allele frequency (:math:`0 < \text{af} < 1`).
                Ignored when ``var_g`` is provided.
            beta (`float`): per-allele change in success probability.
            n_mean (`float`): mean trials per individual :math:`\bar{n}`.
            n_var (`float`): variance of trials per individual.
            n_sigma (`float`): number of NCP standard deviations for the bands.
            r2 (`float`): LD / imputation-accuracy :math:`r^2`
                (:math:`0 < r^2 \leq 1`).
            alpha (`float`): p-value threshold.
            var_g (`float`, optional): genotype / copy-number variance.  Supported
                only when ``n_var`` = 0; see :meth:`ncp_binomial_sd`.
        Returns:
            (power_low, power_mid, power_high) (`tuple[float, float, float]`):
                power at :math:`\lambda - \sigma_\lambda`,
                :math:`\lambda`, and
                :math:`\lambda + \sigma_\lambda`.

        """
        lam = self.ncp_binomial(n, af, beta, n_mean, r2, var_g=var_g)
        sd = self.ncp_binomial_sd(n, af, beta, n_mean, n_var, r2, var_g=var_g)
        return (
            self.llr_power(alpha=alpha, df=1, ncp=max(0.0, lam - n_sigma * sd)),
            self.llr_power(alpha=alpha, df=1, ncp=lam),
            self.llr_power(alpha=alpha, df=1, ncp=lam + n_sigma * sd),
        )

    def binomial_trait_opt_n(
        self, af=0.2, beta=0.05, n_mean=10.0, power=0.8, r2=1.0, alpha=5e-8, var_g=None
    ):
        r"""Minimum sample size to achieve the target power.

        Args:
            af (`float`): allele frequency (:math:`0 < \text{af} < 1`).
                Ignored when ``var_g`` is provided.
            beta (`float`): per-allele change in success probability.
            n_mean (`float`): mean trials per individual :math:`\bar{n}`.
            power (`float`): target power level in :math:`(0, 1)`.
            r2 (`float`): LD / imputation-accuracy :math:`r^2`
                (:math:`0 < r^2 \leq 1`).
            alpha (`float`): p-value threshold.
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            opt_n (`float`): required :math:`N` (fractional; take
                :math:`\lceil \cdot \rceil` in practice).

        """
        assert 0.0 < power < 1.0
        f = lambda n: self.binomial_trait_power(n, af, beta, n_mean, r2, alpha, var_g=var_g) - power
        try:
            opt_n = root_scalar(f, bracket=(1.0, 1e10)).root
        except (OverflowError, ValueError):
            opt_n = np.nan
        return opt_n

    def binomial_trait_beta_power(
        self, n=100, af=0.2, n_mean=10.0, power=0.8, r2=1.0, alpha=5e-8, var_g=None
    ):
        r"""Minimum detectable :math:`|\beta|` at the target power level.

        For SNPs (:math:`g \in \{0,1,2\}`), :math:`\beta` is bounded above so
        that :math:`p_i = \mu + \beta(g_i - 2\,\text{af})` stays in
        :math:`(0,1)` for all genotypes:

        .. math::

            \beta_\max = \min\!\left(
                \frac{1-\mu}{2(1-\text{af})},\;
                \frac{\mu}{2\,\text{af}}
            \right)

        When ``var_g`` is supplied the copy-number range is unknown, so the
        solver uses :math:`\beta_\max = \min(\mu,\,1-\mu)` as a conservative
        bound (valid at the population mean; may not hold for extreme copy
        numbers in the tails of the CNV distribution).

        Args:
            n (`int`): number of individuals.
            af (`float`): allele frequency (:math:`0 < \text{af} < 1`).
                Used for :math:`\beta_\max` when ``var_g`` is ``None``.
            n_mean (`float`): mean trials per individual :math:`\bar{n}`.
            power (`float`): target power level in :math:`(0, 1)`.
            r2 (`float`): LD / imputation-accuracy :math:`r^2`
                (:math:`0 < r^2 \leq 1`).
            alpha (`float`): p-value threshold.
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            opt_beta (`float`): minimum detectable :math:`|\beta|`.

        """
        assert 0.0 < power < 1.0
        if var_g is None:
            # Tightest constraint keeping p_i in (0,1) for all genotypes:
            #   g=2 carrier: mu + 2*(1-af)*beta < 1  =>  beta < (1-mu) / (2*(1-af))
            #   g=0 carrier: mu - 2*af*beta     > 0  =>  beta < mu     / (2*af)
            beta_max = (
                min((1.0 - self.mu) / (2.0 * (1.0 - af)), self.mu / (2.0 * af)) * 0.9999
            )
        else:
            # Copy-number range unknown; bound by population-mean constraint only.
            beta_max = min(self.mu, 1.0 - self.mu) * 0.9999
        f = lambda b: self.binomial_trait_power(n, af, b, n_mean, r2, alpha, var_g=var_g) - power
        try:
            opt_beta = root_scalar(f, bracket=(1e-9, beta_max)).root
        except (OverflowError, ValueError):
            opt_beta = np.nan
        return opt_beta

    def power_curve(
        self, sample_sizes, af=0.2, beta=0.05, n_mean=10.0, r2=1.0, alpha=5e-8, var_g=None
    ):
        r"""Power as a function of sample size (vectorised).

        All NCPs are computed in one pass and a single :func:`ncx2.cdf` call
        is made — no Python loop over sample sizes.

        Args:
            sample_sizes (`array-like`): array of :math:`N` values.
            af (`float`): allele frequency (:math:`0 < \text{af} < 1`).
                Ignored when ``var_g`` is provided.
            beta (`float`): per-allele change in success probability.
            n_mean (`float`): mean trials per individual :math:`\bar{n}`.
            r2 (`float`): LD / imputation-accuracy :math:`r^2`
                (:math:`0 < r^2 \leq 1`).
            alpha (`float`): p-value threshold.
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            powers (`np.ndarray`): power at each sample size.

        """
        ns = np.asarray(sample_sizes, dtype=float)
        vg = self.genotype_var(af, var_g)
        ncps = r2 * ns * beta**2 * vg * n_mean / (self.mu * (1.0 - self.mu))
        chi2_crit = ncx2.ppf(1.0 - alpha, df=1, nc=0)
        return 1.0 - ncx2.cdf(chi2_crit, df=1, nc=ncps)


class GwasPoisson(Gwas):
    r"""GWAS power calculator for a Poisson count trait.

    The outcome :math:`Y_i \sim \text{Poisson}(\mu_i)` is linked to the
    additive genotype :math:`g_i \in \{0, 1, 2\}` (HWE) via:

    **Log link** (default, :math:`\beta` is a log-rate-ratio per allele):

    .. math::

        \log(\mu_i) = \log(\mu) + \beta\,(g_i - 2\,\text{af})

    **Identity link** (:math:`\beta` is an absolute rate change per allele):

    .. math::

        \mu_i = \mu + \beta\,(g_i - 2\,\text{af})

    :math:`\mu` is the population mean count at the null.  The genotype is
    mean-centred (:math:`g_i - 2\,\text{af}`) so the NCP is symmetric in af.

    The score-test non-centrality parameter is:

    .. math::

        \lambda = r^2 \, N \, \beta^2 \cdot 2\,\text{af}(1-\text{af}) \cdot
        \begin{cases} \mu & \text{log link} \\ 1/\mu & \text{identity link} \end{cases}

    For CNV predictors pass ``var_g`` = :math:`\text{Var}(C)` to replace
    :math:`2\,\text{af}(1-\text{af})` with the copy-number variance.
    """

    def __init__(self, mu=1.0, link="log"):
        r"""Initialise a Poisson GWAS power calculator.

        Args:
            mu (`float`): population mean count at null (:math:`\mu > 0`).
            link (`str`): ``'log'`` (default) or ``'identity'``.

        """
        super().__init__()
        if not (mu > 0.0):
            raise ValueError("mu must be strictly positive.")
        if link not in ("log", "identity"):
            raise ValueError("link must be 'log' or 'identity'.")
        self.mu = mu
        self.link = link

    @staticmethod
    def beta_to_fold_change(beta):
        r"""Fold change in rate per allele copy (log link only).

        .. math::

            \text{fold change} = e^{\beta}

        Args:
            beta (`float`): log-rate-ratio per allele.
        Returns:
            fold_change (`float`): multiplicative rate ratio per allele.

        """
        return np.exp(beta)

    @staticmethod
    def beta_to_log_rr(beta, mu):
        r"""Convert an identity-link :math:`\beta` to an approximate log-rate-ratio.

        First-order delta method on the log transformation:

        .. math::

            \log\text{RR} \approx \frac{\beta}{\mu}

        Accurate when :math:`\beta \ll \mu`.

        Args:
            beta (`float`): per-allele rate change (identity-link units).
            mu (`float`): population mean count.
        Returns:
            log_rr (`float`): approximate log-rate-ratio per allele.

        """
        return beta / mu

    @staticmethod
    def log_rr_to_beta(log_rr, mu):
        r"""Convert a log-rate-ratio to an approximate identity-link :math:`\beta`.

        Inverse of :meth:`beta_to_log_rr` (same small-effect approximation applies):

        .. math::

            \beta \approx \mu \cdot \log\text{RR}

        Args:
            log_rr (`float`): log-rate-ratio per allele.
            mu (`float`): population mean count.
        Returns:
            beta (`float`): approximate per-allele rate change.

        """
        return log_rr * mu

    def ncp_poisson(self, n=100, af=0.2, beta=0.1, r2=1.0, var_g=None):
        r"""Non-centrality parameter for the Poisson-trait score test.

        .. math::

            \lambda = r^2 \, N \, \beta^2 \cdot V_g \cdot
            \begin{cases} \mu & \text{log link} \\ 1/\mu & \text{identity link} \end{cases}

        where :math:`V_g = 2\,\text{af}(1-\text{af})` for SNPs or
        :math:`V_g = \text{Var}(C)` when ``var_g`` is supplied.

        Args:
            n (`int`): number of individuals.
            af (`float`): allele frequency (:math:`0 < \text{af} < 1`).
                Ignored when ``var_g`` is provided.
            beta (`float`): per-allele log-rate-ratio (log link) or rate change (identity link).
            r2 (`float`): LD / imputation-accuracy :math:`r^2` (:math:`0 < r^2 \leq 1`).
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            ncp (`float`): non-centrality parameter.

        """
        assert n > 0
        assert 0.0 < r2 <= 1.0
        if var_g is None:
            assert 0.0 < af < 1.0
        vg = self.genotype_var(af, var_g)
        if self.link == "log":
            return r2 * n * beta**2 * vg * self.mu
        else:
            return r2 * n * beta**2 * vg / self.mu

    def poisson_trait_power(self, n=100, af=0.2, beta=0.1, r2=1.0, alpha=5e-8, var_g=None):
        r"""Power to detect association under the Poisson trait model.

        Args:
            n (`int`): number of individuals.
            af (`float`): allele frequency.
                Ignored when ``var_g`` is provided.
            beta (`float`): per-allele log-rate-ratio (log link) or rate change (identity link).
            r2 (`float`): LD / imputation-accuracy :math:`r^2` (:math:`0 < r^2 \leq 1`).
            alpha (`float`): p-value threshold.
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            power (`float`): power in :math:`[0, 1]`.

        """
        ncp = self.ncp_poisson(n, af, beta, r2, var_g=var_g)
        return self.llr_power(alpha=alpha, df=1, ncp=ncp)

    def poisson_trait_opt_n(self, af=0.2, beta=0.1, power=0.8, r2=1.0, alpha=5e-8, var_g=None):
        r"""Minimum sample size to achieve target power.

        Args:
            af (`float`): allele frequency.
                Ignored when ``var_g`` is provided.
            beta (`float`): per-allele log-rate-ratio (log link) or rate change (identity link).
            power (`float`): target power level.
            r2 (`float`): LD / imputation-accuracy :math:`r^2` (:math:`0 < r^2 \leq 1`).
            alpha (`float`): p-value threshold.
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            opt_n (`float`): required :math:`N` (fractional; take :math:`\lceil \cdot \rceil` in practice).

        """
        assert 0.0 < power < 1.0
        f = lambda n: self.poisson_trait_power(n, af, beta, r2, alpha, var_g=var_g) - power
        try:
            opt_n = root_scalar(f, bracket=(1.0, 1e10)).root
        except (OverflowError, ValueError):
            opt_n = np.nan
        return opt_n

    def poisson_trait_beta_power(self, n=100, af=0.2, power=0.8, r2=1.0, alpha=5e-8, var_g=None):
        r"""Minimum detectable :math:`|\beta|` at the target power level.

        The solver bracket upper bound is:

        - **Log link**: :math:`\log(100)` (no analytical bound; cap avoids blowup).
        - **Identity link (SNP)**: :math:`\mu / (2\,\text{af})`, the tightest
          constraint keeping all Poisson means positive
          (:math:`\mu_i = \mu + \beta(g_i - 2\,\text{af}) > 0` at :math:`g_i = 0`).
        - **Identity link (CNV)**: :math:`\mu` as a conservative bound when
          ``var_g`` is supplied and the copy-number range is unknown.

        Args:
            n (`int`): number of individuals.
            af (`float`): allele frequency.
                Used for :math:`\beta_\max` (identity link) when ``var_g`` is ``None``.
            power (`float`): target power level.
            r2 (`float`): LD / imputation-accuracy :math:`r^2` (:math:`0 < r^2 \leq 1`).
            alpha (`float`): p-value threshold.
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            opt_beta (`float`): minimum detectable :math:`\beta`.

        """
        assert 0.0 < power < 1.0
        if self.link == "log":
            beta_max = np.log(100)
        elif var_g is None:
            # g=0 genotype: mu - 2*af*beta > 0  =>  beta < mu/(2*af)
            beta_max = self.mu / (2.0 * af) * 0.9999
        else:
            # Copy-number range unknown; bound conservatively by mu.
            beta_max = self.mu * 0.9999
        f = lambda b: self.poisson_trait_power(n, af, b, r2, alpha, var_g=var_g) - power
        try:
            opt_beta = root_scalar(f, bracket=(1e-9, beta_max)).root
        except (OverflowError, ValueError):
            opt_beta = np.nan
        return opt_beta

    def power_curve(self, sample_sizes, af=0.2, beta=0.1, r2=1.0, alpha=5e-8, var_g=None):
        r"""Power as a function of sample size (vectorised).

        All NCPs are computed in one pass and a single :func:`ncx2.cdf` call is
        made — no Python loop over sample sizes.

        Args:
            sample_sizes (`array-like`): array of :math:`N` values.
            af (`float`): allele frequency.
                Ignored when ``var_g`` is provided.
            beta (`float`): per-allele log-rate-ratio (log link) or rate change (identity link).
            r2 (`float`): LD / imputation-accuracy :math:`r^2` (:math:`0 < r^2 \leq 1`).
            alpha (`float`): p-value threshold.
            var_g (`float`, optional): genotype / copy-number variance to use in
                place of :math:`2\,\text{af}(1-\text{af})`.
        Returns:
            powers (`np.ndarray`): power at each sample size.

        """
        ns = np.asarray(sample_sizes, dtype=float)
        vg = self.genotype_var(af, var_g)
        if self.link == "log":
            ncps = r2 * ns * beta**2 * vg * self.mu
        else:
            ncps = r2 * ns * beta**2 * vg / self.mu
        chi2_crit = ncx2.ppf(1.0 - alpha, df=1, nc=0)
        return 1.0 - ncx2.cdf(chi2_crit, df=1, nc=ncps)
