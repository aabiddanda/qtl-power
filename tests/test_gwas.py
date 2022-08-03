"""Testing module for GWAS power calculations."""
from hypothesis import given
from hypothesis import strategies as st

from qtl_power.gwas import GWAS, GwasBinary, GwasQuant


@given(
    a=st.floats(
        min_value=0, max_value=1, exclude_min=True, exclude_max=True, allow_nan=False
    ),
    d=st.integers(min_value=1, max_value=1000),
    ncp=st.floats(min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False),
)
def test_llr_power(a, d, ncp):
    """Test calculation of log-likelihood ratio calculation."""
    obj = GWAS()
    obj.llr_power(alpha=a, df=d, ncp=ncp)


@given(
    n=st.integers(min_value=1),
    p=st.floats(min_value=0.0, max_value=1.0),
    beta=st.floats(
        min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False
    ),
    r2=st.floats(min_value=0.0, max_value=1.0),
)
def test_ncp_quant(n, p, beta, r2):
    """Test that the non-centrality parameter is calculatable."""
    obj = GwasQuant()
    obj.ncp_quant(n=n, p=p, beta=beta, r2=r2)


@given(
    n=st.integers(min_value=1),
    p=st.floats(min_value=0.0, max_value=1.0),
    beta=st.floats(
        min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False
    ),
    r2=st.floats(min_value=0.0, max_value=1.0),
    alpha=st.floats(exclude_min=True, exclude_max=True, min_value=0.0, max_value=1.0),
)
def test_quant_trait_power(n, p, beta, r2, alpha):
    """Test the function to obtain power under a quantitative model."""
    obj = GwasQuant()
    power = obj.quant_trait_power(n, p, beta, r2, alpha)
    assert (power >= 0) & (power <= 1)


@given(
    n=st.integers(min_value=10),
    p=st.floats(min_value=1e-4, max_value=0.5),
    power=st.floats(min_value=0.5, max_value=1, exclude_max=True),
    r2=st.floats(min_value=0.5, max_value=1.0),
    alpha=st.floats(exclude_min=True, exclude_max=True, min_value=1e-32, max_value=0.5),
)
def test_quant_trait_beta_power(n, p, power, r2, alpha):
    """Test the function to obtain power under a quantitative model."""
    obj = GwasQuant()
    obj.quant_trait_beta_power(n=n, p=p, power=power, r2=r2, alpha=alpha)


@given(
    n=st.integers(min_value=1),
    p=st.floats(min_value=0.0, max_value=1.0),
    beta=st.floats(
        min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False
    ),
    r2=st.floats(min_value=0.0, max_value=1.0),
    prop_cases=st.floats(
        min_value=0.0,
        max_value=1.0,
        exclude_min=True,
        exclude_max=True,
        allow_infinity=False,
        allow_nan=False,
    ),
)
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
        min_value=0.0,
        max_value=1.0,
        exclude_min=True,
        exclude_max=True,
        allow_infinity=False,
        allow_nan=False,
    ),
)
def test_binary_trait_power(n, p, beta, r2, alpha, prop_cases):
    """Test the function to obtain power under a quantitative model."""
    obj = GwasBinary()
    power = obj.binary_trait_power(n, p, beta, r2, alpha, prop_cases)
    assert (power >= 0) & (power <= 1)


@given(
    n=st.integers(min_value=10),
    p=st.floats(min_value=1e-4, max_value=0.5),
    power=st.floats(min_value=0.5, max_value=1, exclude_max=True),
    r2=st.floats(min_value=0.5, max_value=1.0),
    alpha=st.floats(exclude_min=True, exclude_max=True, min_value=1e-32, max_value=0.5),
    prop_cases=st.floats(
        min_value=0.0,
        max_value=1.0,
        exclude_min=True,
        exclude_max=True,
        allow_infinity=False,
        allow_nan=False,
    ),
)
def test_binary_trait_beta_power(n, p, power, r2, alpha, prop_cases):
    """Test the function to obtain power under a quantitative model."""
    obj = GwasBinary()
    obj.binary_trait_beta_power(n=n, p=p, power=power, r2=r2, alpha=alpha)
