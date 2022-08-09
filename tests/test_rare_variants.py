"""Testing module for GWAS power calculations."""
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qtl_power.rare_variants import (RareVariantBurdenPower, RareVariantPower,
                                     RareVariantVCPower)


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
    """Test calculation of log-likelihood ratio power."""
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
    ws, ps = obj.sim_af_weights(j=j, a1=a1, b1=b1)
    assert ws.size == j
    assert ps.size == j


@given(
    a=st.floats(
        min_value=0,
        max_value=1e6,
        exclude_min=True,
        allow_infinity=False,
        allow_nan=False,
    ),
    b=st.floats(
        min_value=0,
        max_value=1e6,
        exclude_min=True,
        allow_infinity=False,
        allow_nan=False,
    ),
    seed=st.integers(min_value=1, max_value=1000000),
)
def test_sim_var_per_gene(a, b, seed):
    """Test of simulating variants per-gene."""
    obj = RareVariantPower()
    nvar = obj.sim_var_per_gene(a=a, b=b, seed=seed)
    assert nvar > 0


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


@given(
    n=st.integers(min_value=1, max_value=1000000),
    ws=arrays(dtype=float, shape=100, elements=st.floats(0, 100)),
    ps=arrays(
        dtype=float,
        shape=100,
        elements=st.floats(1e-8, 1 - 1e-8, allow_nan=False, allow_infinity=False),
    ),
    tev=st.floats(
        min_value=0,
        max_value=1,
        exclude_min=True,
        exclude_max=True,
        allow_infinity=False,
        allow_nan=False,
    ),
)
def test_ncp_vc_first_order_model1(ws, ps, n, tev):
    """Test estimation of the NCP in a rare-variant model."""
    assume(ws.sum() > 0)
    obj = RareVariantVCPower()
    obj.ncp_vc_first_order_model1(ws=ws, ps=ps, n=n, tev=tev)
