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
