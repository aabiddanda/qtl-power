"""Testing module for GWAS power calculations."""
# import pytest
from hypothesis import given
from hypothesis import strategies as st

from qtl_power.gwas import GWAS, GwasCaseControl, GwasQuant


@given(
    a=st.floats(min_value=0, max_value=1),
    d=st.integers(min_value=1),
    ncp=st.floats(allow_infinity=False, allow_nan=False),
)
def test_llr_power(a, d, ncp):
    """Test calculation of log-likelihood ratio calculation."""
    obj = GWAS()
    obj.llr_power(alpha=a, df=d, ncp=ncp)


@given(
    n=st.integers(min_value=1, allow_infinity=False),
    p=st.floats(min_value=0.0, max_value=1.0),
    beta=st.floats(allow_infinity=False, allow_nan=False),
    r2=st.floats(min_value=0.0, max_value=1.0),
)
def test_ncp_quant(n, p, beta, r2):
    """Test that the non-centrality parameter is calculatable."""
    obj = GwasQuant()
    obj.ncp_quant(n=n, p=p, beta=beta, r2=r2)


@given(
    n=st.integers(min_value=1, allow_infinity=False),
    p=st.floats(min_value=0.0, max_value=1.0),
    beta=st.floats(allow_infinity=False, allow_nan=False),
    r2=st.floats(min_value=0.0, max_value=1.0),
    alpha=st.floats(min_value=0.0, max_value=1.0),
)
def test_quant_trait_power(n, p, beta, r2, alpha):
    """Test the function to obtain power under a quantitative model."""
    obj = GwasQuant()
    power = obj.quant_trait_power(n, p, beta, r2, alpha)
    assert (power >= 0) & (power <= 1)


@given(
    n=st.integers(min_value=1, allow_infinity=False),
    p=st.floats(min_value=0.0, max_value=1.0),
    power=st.floats(min_value=0.0, max_value=1.0),
    r2=st.floats(min_value=0.0, max_value=1.0),
    alpha=st.floats(min_value=0.0, max_value=1.0),
)
def test_quant_trait_beta_power(n, p, power, r2, alpha):
    """Test the function to obtain power under a quantitative model."""
    obj = GwasQuant()
    beta = obj.quant_trait_beta_power(n, p, power, r2, alpha)
    assert beta >= 0


@given(
    n=st.integers(min_value=1, allow_infinity=False),
    p=st.floats(min_value=0.0, max_value=1.0),
    power=st.floats(min_value=0.0, max_value=1.0),
    r2=st.floats(min_value=0.0, max_value=1.0),
    prop_cases=st.floats(min_value=0.0, max_value=1.0),
)
def test_ncp_case_control(n, p, beta, r2, prop_cases):
    """Test NCP generation in a case/control model."""
    obj = GwasCaseControl()
    obj.ncp_case_control(n, p, beta, r2, prop_cases)
