"""Functions to calculate power in GWAS designs."""
import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import ncx2


class Gwas:
    """Parent class for GWAS Power calculation."""

    def __init__(self):
        """Initialize base class."""
        pass

    def llr_power(self, alpha=5e-8, df=1, ncp=1):
        """Power under a non-central chi-squared distribution.

        Args:
            alpha (`float`): p-value threshold for GWAS
            df (`int`): degrees of freedom
            ncp (`float`): non-centrality parameter
        Returns:
            power (`float`): power for association

        """
        try:
            return 1.0 - ncx2.cdf(ncx2.ppf(1.0 - alpha, df, 0), df, ncp)
        except OverflowError:
            return np.nan


class GwasQuant(Gwas):
    """Class for power calculations of a GWAS for a quantitative trait."""

    def __init__(self):
        """Initialize a GWAS power calculator for quantitative traits."""
        super(GwasQuant, self).__init__()

    def ncp_quant(self, n=100, p=0.1, beta=0.1, r2=1.0):
        """Compute the non-centrality parameter for a quantitative trait GWAS.

        Args:
            n (`int`): sample-size of unrelated individuals.
            p (`float`): minor allele frequency of variant.
            beta (`float`): effect-size of variant.
            r2 (`float`): correlation r2 between causal variant and tagging variant.
        Returns:
            ncp (`float`): non-centrality parameter.

        """
        assert n > 0
        assert (p > 0.0) and (p < 1.0)
        assert (r2 > 0) & (r2 <= 1.0)
        ncp = r2 * n * 2 * p * (1.0 - p) * (beta**2)
        return ncp

    def quant_trait_power(self, n=100, p=0.1, beta=0.1, r2=1.0, alpha=5e-8):
        """Power for a quantitative trait association study.

        Args:
            n (`int`): sample-size of unrelated individuals.
            p (`float`): minor allele frequency of variant.
            beta (`float`): effect-size of variant.
            r2 (`float`): correlation r2 between causal variant and tagging variant.
            alpha (`float`): p-value threshold for GWAS
        Returns:
            ncp (`float`): non-centrality parameter.

        """
        ncp = self.ncp_quant(n, p, beta, r2)
        return self.llr_power(alpha, df=1, ncp=ncp)

    def quant_trait_beta_power(self, n=100, power=0.90, p=0.1, r2=1.0, alpha=5e-8):
        """Determine the effect-size required to detect an association at this MAF.

        Args:
            n (`int`): sample-size of unrelated individuals.
            power (`float`): threshold power level.
            p (`float`): minor allele frequency of variant.
            r2 (`float`): correlation r2 between causal variant and tagging variant.
            alpha (`float`): p-value threshold for GWAS
        Returns:
            opt_beta  (`float`): optimal beta for detection at a specific power level

        """
        assert (power >= 0) & (power <= 1)
        f = (
            lambda beta: self.quant_trait_power(n=n, p=p, r2=r2, beta=beta, alpha=alpha)
            - power
        )
        try:
            opt_beta = root_scalar(f, bracket=(0.0, 1e3)).root
        except (OverflowError, ValueError):
            opt_beta = np.nan
        return opt_beta

    def quant_trait_opt_n(self, beta=0.1, power=0.90, p=0.1, r2=1.0, alpha=5e-8):
        """Determine the sample-size required to detect this effect.

        Args:
            beta (`float`): effect-size of the variant.
            power (`float`): threshold power level.
            p (`float`): minor allele frequency of variant.
            r2 (`float`): correlation r2 between causal variant and tagging variant.
            alpha (`float`): p-value threshold for GWAS

        Returns:
            opt_n  (`float`): optimal sample size for detection at this power-level.

        """
        assert (power >= 0) & (power <= 1)
        f = (
            lambda n: self.quant_trait_power(n=n, p=p, r2=r2, beta=beta, alpha=alpha)
            - power
        )
        try:
            opt_n = root_scalar(f, bracket=(1e-24, 1e24)).root
        except (OverflowError, ValueError):
            opt_n = np.nan
        return opt_n


class GwasBinary(Gwas):
    """GWAS Power calculator for Case/Control study design."""

    def __init__(self):
        """Initialize a GWAS power calculator for case/control traits."""
        super(GwasBinary, self).__init__()

    def ncp_binary(self, n=100, p=0.1, beta=0.1, r2=1.0, prop_cases=0.1):
        """Determine the effect-size required to detect an association at this MAF.

        Args:
            n (`int`): sample-size of unrelated individuals.
            p (`float`): minor allele frequency of variant.
            beta (`float`): effect-size of variant.
            r2 (`float`): correlation r2 between causal variant and tagging variant.
            prop_cases (`float`): proportion of samples that are cases.
        Returns:
            ncp  (`float`): non-centrality parameter.

        """
        assert n > 0
        assert (p >= 0.0) and (p <= 1.0)
        assert (r2 >= 0) & (r2 <= 1.0)
        assert (prop_cases > 0) & (prop_cases < 1.0)
        ncp = r2 * n * 2 * p * (1.0 - p) * prop_cases * (1.0 - prop_cases) * (beta**2)
        return ncp

    def binary_trait_power(
        self, n=100, p=0.1, beta=0.1, r2=1.0, alpha=5e-8, prop_cases=0.1
    ):
        """Power under a case-control GWAS study design.

        Args:
            n (`int`): sample-size of unrelated individuals.
            p (`float`): minor allele frequency of variant.
            beta (`float`): effect-size of variant.
            r2 (`float`): correlation r2 between causal variant and tagging variant.
            alpha (`float`): p-value threshold for detection.
            prop_cases (`float`): proportion of samples that are cases.
        Returns:
            ncp  (`float`): non-centrality parameter.

        """
        ncp = self.ncp_binary(n, p, beta, r2, prop_cases)
        return self.llr_power(alpha, df=1, ncp=ncp)

    def binary_trait_beta_power(
        self, n=100, power=0.90, p=0.1, r2=1.0, alpha=5e-8, prop_cases=0.5
    ):
        """Optimal detectable effect-size under a case-control GWAS study design.

        Args:
            n (`int`): sample-size of unrelated individuals.
            power (`float`): .
            beta (`float`): effect-size of variant.
            r2 (`float`): correlation r2 between causal variant and tag variant.
            alpha (`float`): p-value threshold for detection.
            prop_cases (`float`): proportion of samples that are cases.

        Returns:
            ncp  (`float`): non-centrality parameter.

        """
        assert n > 0
        assert (p > 0) & (p < 1)
        assert (r2 >= 0.0) & (r2 <= 1.0)
        assert (power > 0) & (power < 1)
        f = (
            lambda beta: self.binary_trait_power(
                n=n, p=p, r2=r2, beta=beta, alpha=alpha, prop_cases=prop_cases
            )
            - power
        )
        try:
            opt_beta = root_scalar(f, bracket=(0.0, 1e3)).root
        except (OverflowError, ValueError):
            opt_beta = np.nan
        return opt_beta

    def binary_trait_opt_n(
        self, beta=0.1, power=0.90, p=0.1, r2=1.0, alpha=5e-8, prop_cases=0.5
    ):
        """Determine the sample-size required to detect this effect.

        Args:
            beta (`float`): effect-size of the variant.
            power (`float`): threshold power level.
            p (`float`): minor allele frequency of variant.
            r2 (`float`): correlation r2 between causal variant and tagging variant.
            alpha (`float`): p-value threshold for GWAS
            prop_cases (`float`): proportion of cases in the dataset

        Returns:
            opt_n  (`float`): optimal sample size for detection at this power-level.

        """
        assert (power >= 0) & (power <= 1)
        f = (
            lambda n: self.binary_trait_power(
                n=n, p=p, r2=r2, beta=beta, alpha=alpha, prop_cases=prop_cases
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
        p=0.1,
        beta=0.1,
        model="additive",
        prev=0.01,
        alpha=5e-8,
        prop_cases=0.5,
    ):
        """Explore how multiple models affect power in case-control traits."""
        assert (prev > 0) & (prev < 1.0)
        assert n > 0
        assert (p > 0) & (p < 1)
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
        af = np.array([p**2, 2 * p * (1.0 - p), (1 - p) ** 2])
        denom = (x * af).sum()
        aa_prob = x[0] * prev / denom
        ab_prob = x[1] * prev / denom
        case_af = (aa_prob * af[0] + ab_prob * af[1] * 0.5) / prev
        control_af = ((1.0 - aa_prob) * af[0] + (1.0 - ab_prob) * af[1] * 0.5) / (
            1.0 - prev
        )
        v_cases = case_af * (1.0 - case_af)
        v_control = control_af * (1.0 - control_af)
        ncp = (case_af - control_af) / (
            np.sqrt((v_cases / n_cases + v_control / n_control) * 0.5)
        )
        return ncp

    def binary_trait_power_model(
        self,
        n=100,
        p=0.1,
        beta=0.1,
        model="additive",
        prev=0.01,
        alpha=5e-8,
        prop_cases=0.5,
    ):
        """Power under a case-control GWAS study design.

        Args:
            n (`int`): sample-size of unrelated individuals.
            p (`float`): minor allele frequency of variant.
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
            p=p,
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
        p=0.1,
        model="additive",
        prev=0.01,
        alpha=5e-8,
        prop_cases=0.5,
        power=0.90,
    ):
        """Threshold effects under a specific power threshold and genetic model.

        Args:
            n (`int`): sample-size of unrelated individuals.
            p (`float`): minor allele frequency of variant.
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
                p=p,
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
    """GWAS power calculator for a binomial count trait.

    Model: Y_i ~ Binomial(n_i, p_i),  p_i = mu + beta*(g_i - 2*af)

    g_i in {0,1,2} is the additive genotype under HWE.  The genotype is
    mean-centred so the NCP is symmetric in af.  mu is the *population mean*
    success probability E[p_i], which is invariant when sweeping af.

    NCP: lambda = r2 * N * beta^2 * 2*af*(1-af) * n_mean / (mu*(1-mu))

    r2 is the LD / imputation-accuracy correlation between the causal variant
    and the typed/imputed tag; r2=1 recovers the perfectly-typed case.
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
        """Convert baseline probability p0 = Pr(success | g=0) to population mean mu.

        Args:
            p0 (`float`): success probability for the aa genotype.
            af (`float`): allele frequency.
            beta (`float`): per-allele change in success probability.
        Returns:
            mu (`float`): population mean success probability.
        """
        return p0 + 2.0 * af * beta

    @staticmethod
    def beta_to_sd_units(beta, mu, n_mean):
        """Convert raw probability beta to phenotypic-SD units (comparable to quantitative GWAS).

        The per-individual rate Y_i/n_i has variance mu*(1-mu)/n_mean, so:
            beta_sd = beta / sqrt(mu*(1-mu) / n_mean)

        This is exact under the identity-link model and shows that deeper
        sequencing (larger n_mean) makes the same raw beta appear larger in SD units.

        Args:
            beta (`float`): per-allele change in success probability (raw units).
            mu (`float`): population mean success probability.
            n_mean (`float`): mean number of trials per individual.
        Returns:
            beta_sd (`float`): effect size in units of phenotypic SD.
        """
        return beta * np.sqrt(n_mean / (mu * (1.0 - mu)))

    @staticmethod
    def sd_units_to_beta(beta_sd, mu, n_mean):
        """Convert SD-unit effect size back to raw probability units.

        Inverse of beta_to_sd_units.

        Args:
            beta_sd (`float`): effect size in units of phenotypic SD.
            mu (`float`): population mean success probability.
            n_mean (`float`): mean number of trials per individual.
        Returns:
            beta (`float`): per-allele change in success probability.
        """
        return beta_sd / np.sqrt(n_mean / (mu * (1.0 - mu)))

    @staticmethod
    def beta_to_log_or(beta, mu):
        """Convert raw probability beta to an approximate log-odds ratio.

        First-order delta method on the logit transformation:
            log(OR) ≈ beta / (mu*(1-mu))

        Valid when beta is small relative to mu*(1-mu). Accuracy degrades when
        mu is near 0 or 1, or when the raw beta is large.

        Args:
            beta (`float`): per-allele change in success probability (raw units).
            mu (`float`): population mean success probability.
        Returns:
            log_or (`float`): approximate log-odds ratio per allele.
        """
        return beta / (mu * (1.0 - mu))

    @staticmethod
    def log_or_to_beta(log_or, mu):
        """Convert a log-odds ratio to an approximate raw probability beta.

        Inverse of beta_to_log_or (same small-effect approximation applies).

        Args:
            log_or (`float`): log-odds ratio per allele.
            mu (`float`): population mean success probability.
        Returns:
            beta (`float`): approximate per-allele change in success probability.
        """
        return log_or * mu * (1.0 - mu)

    def ncp_binomial(self, n=100, af=0.2, beta=0.05, n_mean=10.0, r2=1.0):
        """Non-centrality parameter for the binomial-trait score test.

        Args:
            n (`int`): number of individuals.
            af (`float`): allele frequency (0 < af < 1).
            beta (`float`): per-allele change in success probability.
            n_mean (`float`): mean number of Binomial trials per individual.
            r2 (`float`): LD / imputation-accuracy r² between causal and typed variant (0 < r2 <= 1).
        Returns:
            ncp (`float`): non-centrality parameter.
        """
        assert n > 0
        assert (0.0 < af < 1.0)
        assert n_mean > 0
        assert (0.0 < r2 <= 1.0)
        var_g = 2.0 * af * (1.0 - af)
        return r2 * n * beta**2 * var_g * n_mean / (self.mu * (1.0 - self.mu))

    def ncp_binomial_sd(self, n=100, af=0.2, beta=0.05, n_mean=10.0, n_var=0.0, r2=1.0):
        """Standard deviation of the realised NCP due to variable trial counts.

        By the delta method the variance of the realised NCP is:
            Var(lambda) = lambda^2 * Var(tg^2 * n) / (E[tg^2 * n])^2 / N

        Returns 0.0 when n_var == 0 (fixed-n design).

        Args:
            n (`int`): number of individuals.
            af (`float`): allele frequency.
            beta (`float`): per-allele change in success probability.
            n_mean (`float`): mean trials per individual.
            n_var (`float`): variance of trials per individual (>= 0).
            r2 (`float`): LD / imputation-accuracy r² (0 < r2 <= 1).
        Returns:
            sd (`float`): standard deviation of the realised NCP.
        """
        assert n_var >= 0.0
        if n_var == 0.0:
            return 0.0
        lam = self.ncp_binomial(n, af, beta, n_mean, r2)
        p, q = af, 1.0 - af
        var_g = 2.0 * p * q
        # E[(g - 2p)^4] under HWE
        etg4 = (-2*p)**4 * q**2 + (1 - 2*p)**4 * 2*p*q + (2*q)**4 * p**2
        en2 = n_var + n_mean**2
        var_tg2n = etg4 * en2 - (var_g * n_mean)**2
        cv2_denom = var_tg2n / ((var_g * n_mean)**2 * n)
        return np.sqrt(lam**2 * cv2_denom)

    def binomial_trait_power(self, n=100, af=0.2, beta=0.05, n_mean=10.0, r2=1.0, alpha=5e-8):
        """Power to detect the association under the binomial trait model.

        Args:
            n (`int`): number of individuals.
            af (`float`): allele frequency.
            beta (`float`): per-allele change in success probability.
            n_mean (`float`): mean trials per individual.
            r2 (`float`): LD / imputation-accuracy r² (0 < r2 <= 1).
            alpha (`float`): p-value threshold.
        Returns:
            power (`float`): power in [0, 1].
        """
        ncp = self.ncp_binomial(n, af, beta, n_mean, r2)
        return self.llr_power(alpha=alpha, df=1, ncp=ncp)

    def binomial_trait_power_with_nvar(
        self, n=100, af=0.2, beta=0.05, n_mean=10.0, n_var=0.0, n_sigma=1.0, r2=1.0, alpha=5e-8
    ):
        """Power with ± n_sigma uncertainty bands from variable trial counts.

        Args:
            n (`int`): number of individuals.
            af (`float`): allele frequency.
            beta (`float`): per-allele change in success probability.
            n_mean (`float`): mean trials per individual.
            n_var (`float`): variance of trials per individual.
            n_sigma (`float`): number of NCP standard deviations for bands.
            r2 (`float`): LD / imputation-accuracy r² (0 < r2 <= 1).
            alpha (`float`): p-value threshold.
        Returns:
            (power_low, power_mid, power_high) (`tuple[float, float, float]`).
        """
        lam = self.ncp_binomial(n, af, beta, n_mean, r2)
        sd = self.ncp_binomial_sd(n, af, beta, n_mean, n_var, r2)
        return (
            self.llr_power(alpha=alpha, df=1, ncp=max(0.0, lam - n_sigma * sd)),
            self.llr_power(alpha=alpha, df=1, ncp=lam),
            self.llr_power(alpha=alpha, df=1, ncp=lam + n_sigma * sd),
        )

    def binomial_trait_opt_n(self, af=0.2, beta=0.05, n_mean=10.0, power=0.8, r2=1.0, alpha=5e-8):
        """Minimum sample size to achieve target power.

        Args:
            af (`float`): allele frequency.
            beta (`float`): per-allele change in success probability.
            n_mean (`float`): mean trials per individual.
            power (`float`): target power level.
            r2 (`float`): LD / imputation-accuracy r² (0 < r2 <= 1).
            alpha (`float`): p-value threshold.
        Returns:
            opt_n (`float`): required N (fractional; take ceil in practice).
        """
        assert (0.0 < power < 1.0)
        f = lambda n: self.binomial_trait_power(n, af, beta, n_mean, r2, alpha) - power
        try:
            opt_n = root_scalar(f, bracket=(1.0, 1e10)).root
        except (OverflowError, ValueError):
            opt_n = np.nan
        return opt_n

    def binomial_trait_beta_power(self, n=100, af=0.2, n_mean=10.0, power=0.8, r2=1.0, alpha=5e-8):
        """Minimum detectable |beta| at the target power level.

        beta is bounded above so that p_i = mu + beta*(g-2*af) stays in (0,1).
        The hard cap is beta_max = (1 - mu) / 2, applied with a small margin.

        Args:
            n (`int`): number of individuals.
            af (`float`): allele frequency.
            n_mean (`float`): mean trials per individual.
            power (`float`): target power level.
            r2 (`float`): LD / imputation-accuracy r² (0 < r2 <= 1).
            alpha (`float`): p-value threshold.
        Returns:
            opt_beta (`float`): minimum detectable beta.
        """
        assert (0.0 < power < 1.0)
        # Tightest constraint keeping p_i in (0,1) for all genotypes:
        #   g=2 carrier: mu + 2*(1-af)*beta < 1  =>  beta < (1-mu) / (2*(1-af))
        #   g=0 carrier: mu - 2*af*beta     > 0  =>  beta < mu     / (2*af)
        beta_max = min((1.0 - self.mu) / (2.0 * (1.0 - af)),
                       self.mu / (2.0 * af)) * 0.9999
        f = lambda b: self.binomial_trait_power(n, af, b, n_mean, r2, alpha) - power
        try:
            opt_beta = root_scalar(f, bracket=(1e-9, beta_max)).root
        except (OverflowError, ValueError):
            opt_beta = np.nan
        return opt_beta

    def power_curve(self, sample_sizes, af=0.2, beta=0.05, n_mean=10.0, r2=1.0, alpha=5e-8):
        """Power as a function of sample size.

        Vectorised: all NCPs are computed in one pass, then a single ncx2.cdf
        call is made — no Python loop over sample sizes.

        Args:
            sample_sizes (`array-like`): array of N values.
            af (`float`): allele frequency.
            beta (`float`): per-allele change in success probability.
            n_mean (`float`): mean trials per individual.
            r2 (`float`): LD / imputation-accuracy r² (0 < r2 <= 1).
            alpha (`float`): p-value threshold.
        Returns:
            powers (`np.ndarray`): power at each sample size.
        """
        ns = np.asarray(sample_sizes, dtype=float)
        var_g = 2.0 * af * (1.0 - af)
        ncps = r2 * ns * beta**2 * var_g * n_mean / (self.mu * (1.0 - self.mu))
        chi2_crit = ncx2.ppf(1.0 - alpha, df=1, nc=0)
        return 1.0 - ncx2.cdf(chi2_crit, df=1, nc=ncps)
