"""Testing module for GWAS power calculations."""
from hypothesis import given, assume
from hypothesis import strategies as st

from qtl_power.rare_variants import RareVariantBurdenPower, RareVariantPower


@given(
    a=st.floats(
        min_value=0, max_value=1, exclude_min=True, exclude_max=True, allow_nan=False
    ),
    d=st.integers(min_value=1, max_value=1000),
    ncp=st.floats(min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False),
    ncp0=st.floats(
        min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False
    ),
)
def test_llr_power(a, d, ncp, ncp0):
    """Test calculation of log-likelihood ratio calculation."""
    obj = RareVariantPower()
    obj.llr_power(alpha=a, df=d, ncp=ncp, ncp0=ncp0)


@given(
    j=st.integers(min_value=1, max_value=100000),
    a1=st.floats(
        min_value=0,
        max_value=1e6,
        exclude_min=True,
        allow_infinity=False,
        allow_nan=False,
    ),
    b1=st.floats(
        min_value=0,
        max_value=1e6,
        exclude_min=True,
        allow_infinity=False,
        allow_nan=False,
    ),
)
def test_sim_af_weights(j, a1, b1):
    """Test of allele frequency weight calculation."""
    obj = RareVariantPower()
    ws = obj.sim_af_weights(j=j, a1=a1, b1=b1)
    assert ws.size == j


@given(
    n=st.integers(min_value=1),
    j=st.integers(min_value=100, max_value=100000),
    jd=st.integers(min_value=0, max_value=50),
    jp=st.integers(min_value=0, max_value=50),
    tev=st.floats(
        min_value=0,
        max_value=1,
        exclude_min=True,
        exclude_max=True,
        allow_infinity=False,
        allow_nan=False,
    ),
)
def test_ncp_burden_test_model1(n, j, jd, jp, tev):
    """Test of non-centrality parameter under a burden model."""
    assume(jd + jp > 0)
    obj = RareVariantBurdenPower()
    obj.ncp_burden_test_model1(n=n, j=j, jd=jd, jp=jp, tev=tev)
