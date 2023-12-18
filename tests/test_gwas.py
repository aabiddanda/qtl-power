"""Testing module for GWAS power calculations."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qtl_power.gwas import Gwas, GwasBinary, GwasBinaryModel, GwasQuant


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
    p=st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True),
    beta=st.floats(
        min_value=-1e3, max_value=1e3, allow_infinity=False, allow_nan=False
    ),
    r2=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
)
@settings(deadline=None, max_examples=50)
def test_ncp_quant(n, p, beta, r2):
    """Test that the non-centrality parameter is calculatable."""
    obj = GwasQuant()
    obj.ncp_quant(n=n, p=p, beta=beta, r2=r2)


@given(
    n=st.integers(min_value=1, max_value=10000000),
    p=st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True),
    beta=st.floats(
        min_value=-1e3, max_value=1e3, allow_infinity=False, allow_nan=False
    ),
    r2=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
    alpha=st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True),
)
@settings(deadline=None, max_examples=50)
def test_quant_trait_power(n, p, beta, r2, alpha):
    """Test the function to obtain power under a quantitative model."""
    obj = GwasQuant()
    power = obj.quant_trait_power(n, p, beta, r2, alpha)
    if ~np.isnan(power):
        assert (power >= 0) & (power <= 1)


@given(
    n=st.integers(min_value=10, max_value=10000000),
    p=st.floats(min_value=1e-4, max_value=0.5),
    power=st.floats(min_value=0.5, max_value=1, exclude_max=True),
    r2=st.floats(min_value=0.5, max_value=1.0),
    alpha=st.floats(exclude_min=True, exclude_max=True, min_value=1e-32, max_value=0.5),
)
@settings(deadline=None, max_examples=50)
def test_quant_trait_beta_power(n, p, power, r2, alpha):
    """Test estimation of optimal beta under a quantitative model."""
    obj = GwasQuant()
    obj.quant_trait_beta_power(n=n, p=p, power=power, r2=r2, alpha=alpha)


@given(
    p=st.floats(min_value=1e-4, max_value=0.5),
    power=st.floats(min_value=0.5, max_value=1, exclude_max=True),
    r2=st.floats(min_value=0.5, max_value=1.0),
    alpha=st.floats(exclude_min=True, exclude_max=True, min_value=1e-32, max_value=0.5),
)
@settings(deadline=None, max_examples=50)
def test_quant_trait_opt_n(p, power, r2, alpha):
    """Test estimation of optimal sample-size under a quantitative model."""
    obj = GwasQuant()
    opt_n = obj.quant_trait_opt_n(p=p, power=power, r2=r2, alpha=alpha)
    if ~np.isnan(opt_n):
        assert opt_n > 0


@given(
    n=st.integers(min_value=1),
    p=st.floats(min_value=0.0, max_value=1.0),
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
def test_ncp_binary(n, p, beta, r2, prop_cases):
    """Test NCP generation in a case/control model."""
    obj = GwasBinary()
    obj.ncp_binary(n, p, beta, r2, prop_cases)


@given(
    n=st.integers(min_value=1),
    p=st.floats(min_value=0.0, max_value=1.0),
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
def test_binary_trait_power(n, p, beta, r2, alpha, prop_cases):
    """Test the function to obtain power under a quantitative model."""
    obj = GwasBinary()
    power = obj.binary_trait_power(n, p, beta, r2, alpha, prop_cases)
    assert np.isnan(power) | ((power >= 0) & (power <= 1))


@given(
    n=st.integers(min_value=10),
    p=st.floats(min_value=1e-4, max_value=0.5, exclude_max=True),
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
def test_binary_trait_beta_power(n, p, power, r2, alpha, prop_cases):
    """Test the function to obtain power under a quantitative model."""
    obj = GwasBinary()
    obj.binary_trait_beta_power(
        n=n, p=p, power=power, r2=r2, alpha=alpha, prop_cases=prop_cases
    )


@given(
    p=st.floats(min_value=1e-4, max_value=0.5),
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
def test_binary_trait_opt_n(p, power, r2, alpha, prop_cases):
    """Test the function to obtain power under a quantitative model."""
    obj = GwasBinary()
    opt_n = obj.binary_trait_opt_n(
        p=p, power=power, r2=r2, alpha=alpha, prop_cases=prop_cases
    )
    if ~np.isnan(opt_n):
        assert opt_n > 0


@given(
    n=st.integers(min_value=10),
    p=st.floats(min_value=1e-4, max_value=0.5),
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
def test_ncp_binary_model(n, p, model, prev, alpha, prop_cases):
    """Test NCP generation under different genetic models."""
    obj = GwasBinaryModel()
    obj.ncp_binary_model(
        n=n, p=p, model=model, prev=prev, alpha=alpha, prop_cases=prop_cases
    )


@given(
    n=st.integers(min_value=10),
    p=st.floats(min_value=1e-4, max_value=0.5),
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
def test_ncp_binary_model_bad_model(n, p, model, prev, alpha, prop_cases):
    """Test NCP generation under different genetic models."""
    obj = GwasBinaryModel()
    with pytest.raises(ValueError):
        obj.ncp_binary_model(
            n=n, p=p, model=model, prev=prev, alpha=alpha, prop_cases=prop_cases
        )


@given(
    n=st.integers(min_value=10),
    p=st.floats(min_value=1e-4, max_value=0.5, exclude_max=True),
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
def test_binary_trait_power_model(n, p, model, prev, alpha, prop_cases):
    """Test NCP generation under different genetic models."""
    obj = GwasBinaryModel()
    power = obj.binary_trait_power_model(
        n=n, p=p, model=model, prev=prev, alpha=alpha, prop_cases=prop_cases
    )
    if ~np.isnan(power):
        assert (power >= 0) & (power <= 1)


@given(
    n=st.integers(min_value=10),
    p=st.floats(min_value=1e-4, max_value=0.5, exclude_max=True),
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
def test_binary_trait_beta_power_model(n, p, model, prev, alpha, prop_cases, power):
    """Test effect-size estimate for different models and power."""
    obj = GwasBinaryModel()
    opt_beta = obj.binary_trait_beta_power_model(
        n=n,
        p=p,
        model=model,
        prev=prev,
        alpha=alpha,
        prop_cases=prop_cases,
        power=power,
    )
    if ~np.isnan(opt_beta):
        assert opt_beta >= 0
