"""Testing module for GWAS power calculations."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qtl_power.gwas import Gwas, GwasBinary, GwasBinaryModel, GwasQuant, GwasBinomialTrait, GwasPoisson


@given(
    a=st.floats(
        min_value=0, max_value=1, exclude_min=True, exclude_max=True, allow_nan=False
    ),
    d=st.integers(min_value=1, max_value=1000),
    ncp=st.floats(min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False),
)
@settings(deadline=None, max_examples=50)
def test_llr_power(a, d, ncp):
    """Test calculation of log-likelihood ratio calculation."""
    obj = Gwas()
    obj.llr_power(alpha=a, df=d, ncp=ncp)


@given(
    n=st.integers(min_value=1, max_value=10000000),
    af=st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True),
    beta=st.floats(
        min_value=-1e3, max_value=1e3, allow_infinity=False, allow_nan=False
    ),
    r2=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
)
@settings(deadline=None, max_examples=50)
def test_ncp_quant(n, af, beta, r2):
    """Test that the non-centrality parameter is calculatable."""
    obj = GwasQuant()
    obj.ncp_quant(n=n, af=af, beta=beta, r2=r2)


@given(
    n=st.integers(min_value=1, max_value=10000000),
    af=st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True),
    beta=st.floats(
        min_value=-1e3, max_value=1e3, allow_infinity=False, allow_nan=False
    ),
    r2=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
    alpha=st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True),
)
@settings(deadline=None, max_examples=50)
def test_quant_trait_power(n, af, beta, r2, alpha):
    """Test the function to obtain power under a quantitative model."""
    obj = GwasQuant()
    power = obj.quant_trait_power(n, af, beta, r2, alpha)
    if ~np.isnan(power):
        assert (power >= 0) & (power <= 1)


@given(
    n=st.integers(min_value=10, max_value=10000000),
    af=st.floats(min_value=1e-4, max_value=0.5),
    power=st.floats(min_value=0.5, max_value=1, exclude_max=True),
    r2=st.floats(min_value=0.5, max_value=1.0),
    alpha=st.floats(exclude_min=True, exclude_max=True, min_value=1e-32, max_value=0.5),
)
@settings(deadline=None, max_examples=50)
def test_quant_trait_beta_power(n, af, power, r2, alpha):
    """Test estimation of optimal beta under a quantitative model."""
    obj = GwasQuant()
    obj.quant_trait_beta_power(n=n, af=af, power=power, r2=r2, alpha=alpha)


@given(
    af=st.floats(min_value=1e-4, max_value=0.5),
    power=st.floats(min_value=0.5, max_value=1, exclude_max=True),
    r2=st.floats(min_value=0.5, max_value=1.0),
    alpha=st.floats(exclude_min=True, exclude_max=True, min_value=1e-32, max_value=0.5),
)
@settings(deadline=None, max_examples=50)
def test_quant_trait_opt_n(af, power, r2, alpha):
    """Test estimation of optimal sample-size under a quantitative model."""
    obj = GwasQuant()
    opt_n = obj.quant_trait_opt_n(af=af, power=power, r2=r2, alpha=alpha)
    if ~np.isnan(opt_n):
        assert opt_n > 0


@given(
    n=st.integers(min_value=1),
    af=st.floats(min_value=0.0, max_value=1.0),
    beta=st.floats(
        min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False
    ),
    r2=st.floats(min_value=0.0, max_value=1.0),
    prop_cases=st.floats(
        min_value=1e-3,
        max_value=0.5,
        allow_infinity=False,
        allow_nan=False,
    ),
)
@settings(deadline=None, max_examples=50)
def test_ncp_binary(n, af, beta, r2, prop_cases):
    """Test NCP generation in a case/control model."""
    obj = GwasBinary()
    obj.ncp_binary(n, af, beta, r2, prop_cases)


@given(
    n=st.integers(min_value=1),
    af=st.floats(min_value=0.0, max_value=1.0),
    beta=st.floats(
        min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False
    ),
    r2=st.floats(min_value=0.0, max_value=1.0),
    alpha=st.floats(exclude_min=True, exclude_max=True, min_value=0.0, max_value=1.0),
    prop_cases=st.floats(
        min_value=1e-3,
        max_value=0.5,
        allow_infinity=False,
        allow_nan=False,
    ),
)
@settings(deadline=None, max_examples=50)
def test_binary_trait_power(n, af, beta, r2, alpha, prop_cases):
    """Test the function to obtain power under a quantitative model."""
    obj = GwasBinary()
    power = obj.binary_trait_power(n, af, beta, r2, alpha, prop_cases)
    assert np.isnan(power) | ((power >= 0) & (power <= 1))


@given(
    n=st.integers(min_value=10),
    af=st.floats(min_value=1e-4, max_value=0.5, exclude_max=True),
    power=st.floats(min_value=0.5, max_value=1, exclude_max=True),
    r2=st.floats(min_value=0.5, max_value=1.0),
    alpha=st.floats(exclude_min=True, exclude_max=True, min_value=1e-32, max_value=0.5),
    prop_cases=st.floats(
        min_value=1e-3,
        max_value=0.5,
        allow_infinity=False,
        allow_nan=False,
    ),
)
@settings(deadline=None, max_examples=50)
def test_binary_trait_beta_power(n, af, power, r2, alpha, prop_cases):
    """Test the function to obtain power under a quantitative model."""
    obj = GwasBinary()
    obj.binary_trait_beta_power(
        n=n, af=af, power=power, r2=r2, alpha=alpha, prop_cases=prop_cases
    )


@given(
    af=st.floats(min_value=1e-4, max_value=0.5),
    power=st.floats(min_value=0.5, max_value=1, exclude_max=True),
    r2=st.floats(min_value=0.5, max_value=1.0),
    alpha=st.floats(exclude_min=True, exclude_max=True, min_value=1e-32, max_value=0.5),
    prop_cases=st.floats(
        min_value=1e-3,
        max_value=0.5,
        allow_infinity=False,
        allow_nan=False,
    ),
)
@settings(deadline=None, max_examples=50)
def test_binary_trait_opt_n(af, power, r2, alpha, prop_cases):
    """Test the function to obtain power under a quantitative model."""
    obj = GwasBinary()
    opt_n = obj.binary_trait_opt_n(
        af=af, power=power, r2=r2, alpha=alpha, prop_cases=prop_cases
    )
    if ~np.isnan(opt_n):
        assert opt_n > 0


@given(
    n=st.integers(min_value=10),
    af=st.floats(min_value=1e-4, max_value=0.5),
    model=st.sampled_from(["additive", "recessive", "dominant"]),
    prev=st.floats(min_value=0, max_value=0.5, exclude_min=True),
    alpha=st.floats(exclude_min=True, exclude_max=True, min_value=1e-32, max_value=0.5),
    prop_cases=st.floats(
        min_value=1e-3,
        max_value=0.5,
        allow_infinity=False,
        allow_nan=False,
    ),
)
@settings(deadline=None, max_examples=50)
def test_ncp_binary_model(n, af, model, prev, alpha, prop_cases):
    """Test NCP generation under different genetic models."""
    obj = GwasBinaryModel()
    obj.ncp_binary_model(
        n=n, af=af, model=model, prev=prev, alpha=alpha, prop_cases=prop_cases
    )


@given(
    n=st.integers(min_value=10),
    af=st.floats(min_value=1e-4, max_value=0.5),
    model=st.sampled_from(["10101", "", "a"]),
    prev=st.floats(min_value=0, max_value=0.5, exclude_min=True),
    alpha=st.floats(exclude_min=True, exclude_max=True, min_value=1e-32, max_value=0.5),
    prop_cases=st.floats(
        min_value=1e-3,
        max_value=0.5,
        allow_infinity=False,
        allow_nan=False,
    ),
)
@settings(deadline=None, max_examples=50)
def test_ncp_binary_model_bad_model(n, af, model, prev, alpha, prop_cases):
    """Test NCP generation under different genetic models."""
    obj = GwasBinaryModel()
    with pytest.raises(ValueError):
        obj.ncp_binary_model(
            n=n, af=af, model=model, prev=prev, alpha=alpha, prop_cases=prop_cases
        )


@given(
    n=st.integers(min_value=10),
    af=st.floats(min_value=1e-4, max_value=0.5, exclude_max=True),
    model=st.sampled_from(["additive", "recessive", "dominant"]),
    prev=st.floats(min_value=0, max_value=0.5, exclude_min=True),
    alpha=st.floats(exclude_min=True, exclude_max=True, min_value=1e-32, max_value=0.5),
    prop_cases=st.floats(
        min_value=1e-3,
        max_value=0.5,
        allow_infinity=False,
        allow_nan=False,
    ),
)
@settings(deadline=None, max_examples=50)
def test_binary_trait_power_model(n, af, model, prev, alpha, prop_cases):
    """Test NCP generation under different genetic models."""
    obj = GwasBinaryModel()
    power = obj.binary_trait_power_model(
        n=n, af=af, model=model, prev=prev, alpha=alpha, prop_cases=prop_cases
    )
    if ~np.isnan(power):
        assert (power >= 0) & (power <= 1)


@given(
    n=st.integers(min_value=10),
    af=st.floats(min_value=1e-4, max_value=0.5, exclude_max=True),
    model=st.sampled_from(["additive", "recessive", "dominant"]),
    prev=st.floats(min_value=1e-4, max_value=0.5, exclude_min=True, exclude_max=True),
    alpha=st.floats(exclude_min=True, exclude_max=True, min_value=1e-32, max_value=0.5),
    power=st.floats(min_value=0.5, max_value=1, exclude_max=True),
    prop_cases=st.floats(
        min_value=1e-3,
        max_value=0.5,
        allow_infinity=False,
        allow_nan=False,
    ),
)
@settings(deadline=None, max_examples=50)
def test_binary_trait_beta_power_model(n, af, model, prev, alpha, prop_cases, power):
    """Test effect-size estimate for different models and power."""
    obj = GwasBinaryModel()
    opt_beta = obj.binary_trait_beta_power_model(
        n=n,
        af=af,
        model=model,
        prev=prev,
        alpha=alpha,
        prop_cases=prop_cases,
        power=power,
    )
    if ~np.isnan(opt_beta):
        assert opt_beta >= 0


# ---------------------------------------------------------------------------
# GwasBinomialTrait
# ---------------------------------------------------------------------------

@given(
    n=st.integers(min_value=1, max_value=10_000_000),
    af=st.floats(min_value=1e-4, max_value=1 - 1e-4),
    beta=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    n_mean=st.floats(min_value=1e-3, max_value=1e4, allow_nan=False, allow_infinity=False),
    mu=st.floats(min_value=1e-4, max_value=1 - 1e-4),
)
@settings(deadline=None, max_examples=50)
def test_ncp_binomial(n, af, beta, n_mean, mu):
    """NCP is non-negative for any valid inputs."""
    obj = GwasBinomialTrait(mu=mu)
    ncp = obj.ncp_binomial(n=n, af=af, beta=beta, n_mean=n_mean)
    assert ncp >= 0


@given(
    n=st.integers(min_value=1, max_value=10_000_000),
    af=st.floats(min_value=1e-4, max_value=1 - 1e-4),
    beta=st.floats(min_value=1e-6, max_value=0.1, allow_nan=False),
    n_mean=st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
    mu=st.floats(min_value=0.1, max_value=0.9),
    alpha=st.floats(min_value=1e-32, max_value=0.5, exclude_min=True, exclude_max=True),
)
@settings(deadline=None, max_examples=50)
def test_binomial_trait_power(n, af, beta, n_mean, mu, alpha):
    """Power is in [0, 1]."""
    obj = GwasBinomialTrait(mu=mu)
    power = obj.binomial_trait_power(n=n, af=af, beta=beta, n_mean=n_mean, alpha=alpha)
    assert np.isnan(power) or (0.0 <= power <= 1.0)


@given(
    af=st.floats(min_value=0.05, max_value=0.45),
    beta=st.floats(min_value=1e-4, max_value=0.05),
    n_mean=st.floats(min_value=1.0, max_value=50.0),
    mu=st.floats(min_value=0.1, max_value=0.9),
    power=st.floats(min_value=0.5, max_value=0.95),
)
@settings(deadline=None, max_examples=50)
def test_binomial_trait_opt_n(af, beta, n_mean, mu, power):
    """Optimal N is positive when finite."""
    obj = GwasBinomialTrait(mu=mu)
    opt_n = obj.binomial_trait_opt_n(af=af, beta=beta, n_mean=n_mean, power=power, alpha=0.05)
    if ~np.isnan(opt_n):
        assert opt_n > 0


@given(
    n=st.integers(min_value=1000, max_value=1_000_000),
    af=st.floats(min_value=0.05, max_value=0.45),
    n_mean=st.floats(min_value=1.0, max_value=50.0),
    mu=st.floats(min_value=0.1, max_value=0.8),
    power=st.floats(min_value=0.5, max_value=0.95),
)
@settings(deadline=None, max_examples=50)
def test_binomial_trait_beta_power(n, af, n_mean, mu, power):
    """Min detectable beta is non-negative when finite."""
    obj = GwasBinomialTrait(mu=mu)
    opt_beta = obj.binomial_trait_beta_power(n=n, af=af, n_mean=n_mean, power=power, alpha=0.05)
    if ~np.isnan(opt_beta):
        assert opt_beta >= 0


def test_ncp_binomial_af_symmetry():
    """NCP must be identical at af and 1-af (mean-centred genotype)."""
    obj = GwasBinomialTrait(mu=0.3)
    for af in [0.1, 0.2, 0.3, 0.4]:
        ncp_af = obj.ncp_binomial(n=5000, af=af, beta=0.05, n_mean=10)
        ncp_comp = obj.ncp_binomial(n=5000, af=1.0 - af, beta=0.05, n_mean=10)
        assert abs(ncp_af - ncp_comp) < 1e-10


def test_binomial_trait_power_known_values():
    """Analytic power matches validation table from the derivation notes (tol 2%)."""
    obj = GwasBinomialTrait(mu=0.3)
    expected = {0.1: 0.1004, 0.2: 0.1408, 0.3: 0.1701, 0.5: 0.1936}
    for af, exp_pwr in expected.items():
        pwr = obj.binomial_trait_power(n=2000, af=af, beta=0.005, n_mean=10, alpha=0.05)
        assert abs(pwr - exp_pwr) < 0.02, f"af={af}: got {pwr:.4f}, expected {exp_pwr:.4f}"


def test_mu_from_p0():
    """mu_from_p0 recovers the correct population mean."""
    p0, af, beta = 0.2, 0.3, 0.05
    mu = GwasBinomialTrait.mu_from_p0(p0, af, beta)
    assert abs(mu - (p0 + 2 * af * beta)) < 1e-12


def test_ncp_binomial_sd_zero_for_fixed_n():
    """SD of NCP is zero for a fixed-n design."""
    obj = GwasBinomialTrait(mu=0.3)
    assert obj.ncp_binomial_sd(n=1000, af=0.2, beta=0.05, n_mean=10, n_var=0.0) == 0.0


def test_binomial_trait_power_with_nvar_ordering():
    """Uncertainty bands are ordered: power_low <= power_mid <= power_high."""
    obj = GwasBinomialTrait(mu=0.3)
    lo, mid, hi = obj.binomial_trait_power_with_nvar(
        n=2000, af=0.2, beta=0.05, n_mean=10, n_var=10, n_sigma=1.0, alpha=0.05
    )
    assert lo <= mid <= hi


def test_beta_sd_units_round_trip():
    """beta -> SD units -> beta round-trips exactly."""
    for mu in [0.2, 0.5, 0.8]:
        for n_mean in [1.0, 10.0, 50.0]:
            beta = 0.03
            beta_sd = GwasBinomialTrait.beta_to_sd_units(beta, mu, n_mean)
            assert abs(GwasBinomialTrait.sd_units_to_beta(beta_sd, mu, n_mean) - beta) < 1e-12


def test_beta_log_or_round_trip():
    """beta -> log-OR -> beta round-trips exactly (approximation is self-consistent)."""
    for mu in [0.2, 0.5, 0.8]:
        beta = 0.01
        log_or = GwasBinomialTrait.beta_to_log_or(beta, mu)
        assert abs(GwasBinomialTrait.log_or_to_beta(log_or, mu) - beta) < 1e-12


def test_beta_to_sd_units_increases_with_n_mean():
    """Deeper sequencing makes the same raw beta larger in SD units."""
    mu = 0.4
    beta = 0.05
    sd_low = GwasBinomialTrait.beta_to_sd_units(beta, mu, n_mean=5.0)
    sd_high = GwasBinomialTrait.beta_to_sd_units(beta, mu, n_mean=20.0)
    assert sd_high > sd_low


def test_gwas_binomial_invalid_mu():
    """Constructor raises ValueError for mu outside (0, 1)."""
    with pytest.raises(ValueError):
        GwasBinomialTrait(mu=0.0)
    with pytest.raises(ValueError):
        GwasBinomialTrait(mu=1.5)


def test_binomial_trait_beta_power_self_consistent():
    """power(opt_beta) should recover the target power (solver round-trip)."""
    obj = GwasBinomialTrait(mu=0.3)
    for af in [0.1, 0.3, 0.5]:
        opt_beta = obj.binomial_trait_beta_power(n=5000, af=af, n_mean=10, power=0.8, alpha=0.05)
        if not np.isnan(opt_beta):
            recovered = obj.binomial_trait_power(n=5000, af=af, beta=opt_beta, n_mean=10, alpha=0.05)
            assert abs(recovered - 0.8) < 1e-4, f"af={af}: power={recovered:.4f}"


def test_binomial_trait_beta_power_mid_af():
    """beta_max bug: at af=0.5, mu=0.3 the old cap (0.35) exceeded the valid range (0.3).
    The solver must return a finite, valid beta."""
    obj = GwasBinomialTrait(mu=0.3)
    opt_beta = obj.binomial_trait_beta_power(n=50000, af=0.5, n_mean=10, power=0.8, alpha=0.05)
    assert not np.isnan(opt_beta)
    assert opt_beta < 0.3  # strict validity bound at af=0.5, mu=0.3


def test_power_curve_matches_scalar():
    """Vectorised power_curve must match per-point binomial_trait_power calls."""
    obj = GwasBinomialTrait(mu=0.3)
    ns = np.array([500, 1000, 2000, 5000, 10000])
    curve = obj.power_curve(ns, af=0.2, beta=0.05, n_mean=10, alpha=0.05)
    scalar = np.array([obj.binomial_trait_power(n, af=0.2, beta=0.05, n_mean=10, alpha=0.05) for n in ns])
    np.testing.assert_allclose(curve, scalar, rtol=1e-10)


def test_power_curve_monotone():
    """Power must be non-decreasing in sample size."""
    obj = GwasBinomialTrait(mu=0.3)
    ns = np.linspace(100, 20000, 50)
    curve = obj.power_curve(ns, af=0.2, beta=0.05, n_mean=10, alpha=0.05)
    assert np.all(np.diff(curve) >= 0)


def test_ncp_binomial_r2_scales_ncp():
    """NCP scales linearly with r2, matching the GwasQuant convention."""
    obj = GwasBinomialTrait(mu=0.3)
    ncp_full = obj.ncp_binomial(n=5000, af=0.2, beta=0.05, n_mean=10, r2=1.0)
    for r2 in [0.25, 0.5, 0.8]:
        assert abs(obj.ncp_binomial(n=5000, af=0.2, beta=0.05, n_mean=10, r2=r2) - r2 * ncp_full) < 1e-10


def test_binomial_trait_power_r2_reduces_power():
    """Imperfect imputation (r2 < 1) strictly reduces power."""
    obj = GwasBinomialTrait(mu=0.3)
    pwr_full = obj.binomial_trait_power(n=2000, af=0.2, beta=0.05, n_mean=10, r2=1.0, alpha=0.05)
    pwr_partial = obj.binomial_trait_power(n=2000, af=0.2, beta=0.05, n_mean=10, r2=0.7, alpha=0.05)
    assert pwr_partial < pwr_full


@given(
    n=st.integers(min_value=100, max_value=1_000_000),
    af=st.floats(min_value=1e-4, max_value=1 - 1e-4),
    beta=st.floats(min_value=1e-6, max_value=0.1, allow_nan=False),
    n_mean=st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
    mu=st.floats(min_value=0.1, max_value=0.9),
    r2=st.floats(min_value=1e-4, max_value=1.0),
)
@settings(deadline=None, max_examples=50)
def test_ncp_binomial_with_r2(n, af, beta, n_mean, mu, r2):
    """NCP with r2 is non-negative and no greater than the r2=1 NCP."""
    obj = GwasBinomialTrait(mu=mu)
    ncp = obj.ncp_binomial(n=n, af=af, beta=beta, n_mean=n_mean, r2=r2)
    ncp_full = obj.ncp_binomial(n=n, af=af, beta=beta, n_mean=n_mean, r2=1.0)
    assert ncp >= 0
    assert ncp <= ncp_full + 1e-10


# ---------------------------------------------------------------------------
# GwasPoisson
# ---------------------------------------------------------------------------

@given(
    n=st.integers(min_value=1, max_value=10_000_000),
    af=st.floats(min_value=1e-4, max_value=1 - 1e-4),
    beta=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    mu=st.floats(min_value=1e-3, max_value=1e4),
    r2=st.floats(min_value=1e-4, max_value=1.0),
    link=st.sampled_from(["log", "identity"]),
)
@settings(deadline=None, max_examples=50)
def test_ncp_poisson(n, af, beta, mu, r2, link):
    """NCP is non-negative for any valid inputs."""
    obj = GwasPoisson(mu=mu, link=link)
    ncp = obj.ncp_poisson(n=n, af=af, beta=beta, r2=r2)
    assert ncp >= 0


@given(
    n=st.integers(min_value=1, max_value=10_000_000),
    af=st.floats(min_value=1e-4, max_value=1 - 1e-4),
    beta=st.floats(min_value=1e-6, max_value=2.0, allow_nan=False),
    mu=st.floats(min_value=1e-2, max_value=1e3),
    r2=st.floats(min_value=1e-4, max_value=1.0),
    alpha=st.floats(min_value=1e-32, max_value=0.5, exclude_min=True, exclude_max=True),
    link=st.sampled_from(["log", "identity"]),
)
@settings(deadline=None, max_examples=50)
def test_poisson_trait_power(n, af, beta, mu, r2, alpha, link):
    """Power is in [0, 1]."""
    obj = GwasPoisson(mu=mu, link=link)
    power = obj.poisson_trait_power(n=n, af=af, beta=beta, r2=r2, alpha=alpha)
    assert np.isnan(power) or (0.0 <= power <= 1.0)


@given(
    af=st.floats(min_value=0.05, max_value=0.45),
    beta=st.floats(min_value=1e-4, max_value=1.0),
    mu=st.floats(min_value=0.1, max_value=100.0),
    power=st.floats(min_value=0.5, max_value=0.95),
    link=st.sampled_from(["log", "identity"]),
)
@settings(deadline=None, max_examples=50)
def test_poisson_trait_opt_n(af, beta, mu, power, link):
    """Optimal N is positive when finite."""
    obj = GwasPoisson(mu=mu, link=link)
    opt_n = obj.poisson_trait_opt_n(af=af, beta=beta, power=power, alpha=0.05)
    if ~np.isnan(opt_n):
        assert opt_n > 0


@given(
    n=st.integers(min_value=1000, max_value=1_000_000),
    af=st.floats(min_value=0.05, max_value=0.45),
    mu=st.floats(min_value=0.1, max_value=100.0),
    power=st.floats(min_value=0.5, max_value=0.95),
    link=st.sampled_from(["log", "identity"]),
)
@settings(deadline=None, max_examples=50)
def test_poisson_trait_beta_power(n, af, mu, power, link):
    """Min detectable beta is non-negative when finite."""
    obj = GwasPoisson(mu=mu, link=link)
    opt_beta = obj.poisson_trait_beta_power(n=n, af=af, power=power, alpha=0.05)
    if ~np.isnan(opt_beta):
        assert opt_beta >= 0


def test_ncp_poisson_af_symmetry():
    """NCP is identical at af and 1-af (mean-centred genotype)."""
    for link in ("log", "identity"):
        obj = GwasPoisson(mu=2.0, link=link)
        for af in [0.1, 0.2, 0.3, 0.4]:
            ncp_af = obj.ncp_poisson(n=5000, af=af, beta=0.1)
            ncp_comp = obj.ncp_poisson(n=5000, af=1.0 - af, beta=0.1)
            assert abs(ncp_af - ncp_comp) < 1e-10, f"link={link}, af={af}"


def test_ncp_poisson_log_link_formula():
    """NCP (log link) matches the analytic formula r2*N*beta^2*var_g*mu."""
    obj = GwasPoisson(mu=3.0, link="log")
    n, af, beta, r2 = 1000, 0.3, 0.2, 0.8
    expected = r2 * n * beta**2 * 2 * af * (1 - af) * obj.mu
    assert abs(obj.ncp_poisson(n=n, af=af, beta=beta, r2=r2) - expected) < 1e-10


def test_ncp_poisson_identity_link_formula():
    """NCP (identity link) matches the analytic formula r2*N*beta^2*var_g/mu."""
    obj = GwasPoisson(mu=3.0, link="identity")
    n, af, beta, r2 = 1000, 0.3, 0.2, 0.8
    expected = r2 * n * beta**2 * 2 * af * (1 - af) / obj.mu
    assert abs(obj.ncp_poisson(n=n, af=af, beta=beta, r2=r2) - expected) < 1e-10


def test_ncp_poisson_r2_scales():
    """NCP scales linearly with r2."""
    for link in ("log", "identity"):
        obj = GwasPoisson(mu=2.0, link=link)
        ncp_full = obj.ncp_poisson(n=5000, af=0.2, beta=0.1, r2=1.0)
        for r2 in [0.25, 0.5, 0.8]:
            assert abs(obj.ncp_poisson(n=5000, af=0.2, beta=0.1, r2=r2) - r2 * ncp_full) < 1e-10


def test_poisson_power_curve_matches_scalar():
    """Vectorised power_curve must match per-point poisson_trait_power calls."""
    for link in ("log", "identity"):
        obj = GwasPoisson(mu=2.0, link=link)
        ns = np.array([500, 1000, 2000, 5000, 10000])
        curve = obj.power_curve(ns, af=0.2, beta=0.1, alpha=0.05)
        scalar = np.array([obj.poisson_trait_power(n, af=0.2, beta=0.1, alpha=0.05) for n in ns])
        np.testing.assert_allclose(curve, scalar, rtol=1e-10)


def test_poisson_power_curve_monotone():
    """Power is non-decreasing in sample size."""
    for link in ("log", "identity"):
        obj = GwasPoisson(mu=2.0, link=link)
        ns = np.linspace(100, 20000, 50)
        curve = obj.power_curve(ns, af=0.2, beta=0.1, alpha=0.05)
        assert np.all(np.diff(curve) >= 0), f"power not monotone for link={link}"


def test_poisson_trait_beta_power_self_consistent():
    """power(opt_beta) recovers the target power."""
    for link in ("log", "identity"):
        obj = GwasPoisson(mu=2.0, link=link)
        for af in [0.1, 0.3, 0.5]:
            opt_beta = obj.poisson_trait_beta_power(n=5000, af=af, power=0.8, alpha=0.05)
            if not np.isnan(opt_beta):
                recovered = obj.poisson_trait_power(n=5000, af=af, beta=opt_beta, alpha=0.05)
                assert abs(recovered - 0.8) < 1e-4, f"link={link}, af={af}: power={recovered:.4f}"


def test_gwas_poisson_invalid_mu():
    """Constructor raises ValueError for mu <= 0."""
    with pytest.raises(ValueError):
        GwasPoisson(mu=0.0)
    with pytest.raises(ValueError):
        GwasPoisson(mu=-1.0)


def test_gwas_poisson_invalid_link():
    """Constructor raises ValueError for unrecognised link."""
    with pytest.raises(ValueError):
        GwasPoisson(mu=1.0, link="logit")


def test_beta_to_fold_change():
    """beta_to_fold_change is exp(beta)."""
    for beta in [0.0, 0.5, 1.0, np.log(2)]:
        assert abs(GwasPoisson.beta_to_fold_change(beta) - np.exp(beta)) < 1e-12


def test_poisson_beta_log_rr_round_trip():
    """beta -> log_rr -> beta round-trips via the delta-method approximation."""
    for mu in [0.5, 2.0, 10.0]:
        beta = 0.05 * mu
        log_rr = GwasPoisson.beta_to_log_rr(beta, mu)
        assert abs(GwasPoisson.log_rr_to_beta(log_rr, mu) - beta) < 1e-12


def test_poisson_r2_reduces_power():
    """Imperfect imputation (r2 < 1) strictly reduces power."""
    for link in ("log", "identity"):
        obj = GwasPoisson(mu=2.0, link=link)
        pwr_full = obj.poisson_trait_power(n=2000, af=0.2, beta=0.1, r2=1.0, alpha=0.05)
        pwr_partial = obj.poisson_trait_power(n=2000, af=0.2, beta=0.1, r2=0.7, alpha=0.05)
        assert pwr_partial < pwr_full, f"link={link}: r2<1 did not reduce power"
