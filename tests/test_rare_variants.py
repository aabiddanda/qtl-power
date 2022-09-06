"""Testing module for GWAS power calculations."""
import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

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
    """Test calculation of log-likelihood ratio power."""
    obj = RareVariantPower()
    power = obj.llr_power(alpha=a, df=d, ncp=ncp, ncp0=ncp0)
    if ~np.isnan(power):
        assert power >= 0
        assert power <= 1.0


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
    test=st.sampled_from(["SKAT", "Calpha", "Hotelling"]),
)
def test_sim_af_weights(j, a1, b1, test):
    """Test of allele frequency weight calculation."""
    obj = RareVariantPower()
    ws, ps = obj.sim_af_weights(j=j, a1=a1, b1=b1, test=test)
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
        min_value=1e-5,
        max_value=1,
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
    n=st.integers(min_value=1),
    j=st.integers(min_value=100, max_value=100000),
    prop_causal=st.floats(min_value=1e-2, max_value=1.0),
    prop_risk=st.floats(min_value=0.5, max_value=1.0),
    tev=st.floats(
        min_value=1e-5,
        max_value=1,
        exclude_max=True,
        allow_infinity=False,
        allow_nan=False,
    ),
    alpha=st.floats(
        min_value=1e-32, max_value=0.5, allow_infinity=False, allow_nan=False
    ),
)
def test_power_burden_model1(n, j, prop_causal, prop_risk, tev, alpha):
    """Test of power under burden model 1."""
    obj = RareVariantBurdenPower()
    obj.power_burden_model1(
        n=n, j=j, prop_causal=prop_causal, prop_risk=prop_risk, tev=tev, alpha=alpha
    )


@given(n=st.integers(min_value=1), nreps=st.integers(min_value=1, max_value=100))
@settings(deadline=None, max_examples=200)
def test_power_burden_model1_real(n, nreps):
    """Test of power under burden model 1 and real sampling."""
    obj = RareVariantBurdenPower()
    obj.power_burden_model1_real(n=n, nreps=nreps)


@given(
    n=st.integers(min_value=1),
    j=st.integers(min_value=100, max_value=100000),
    prop_causal=st.floats(min_value=1e-2, max_value=1.0),
    prop_risk=st.floats(min_value=0.5, max_value=1.0),
    power=st.floats(min_value=0.5, max_value=1.0, exclude_max=True),
    alpha=st.floats(
        min_value=1e-32, max_value=0.5, allow_infinity=False, allow_nan=False
    ),
)
@settings(deadline=None, max_examples=200)
def test_tev_power_burden_model1(n, j, prop_causal, prop_risk, alpha, power):
    """Test detectable TEV under Burden Model 1."""
    obj = RareVariantBurdenPower()
    tev = obj.tev_power_burden_model1(
        n=n, j=j, prop_causal=prop_causal, prop_risk=prop_risk, alpha=alpha, power=power
    )
    if ~np.isnan(tev):
        assert tev >= 0
        assert tev <= 1


@given(
    j=st.integers(min_value=100, max_value=1000),
    tev=st.floats(
        min_value=1e-4,
        max_value=1,
        exclude_max=True,
        allow_infinity=False,
        allow_nan=False,
    ),
    prop_causal=st.floats(min_value=0.5, max_value=1.0),
    prop_risk=st.floats(min_value=0.5, max_value=1.0),
    power=st.floats(min_value=0.5, max_value=1.0, exclude_max=True),
    alpha=st.floats(
        min_value=1e-32, max_value=5e-2, allow_infinity=False, allow_nan=False
    ),
)
@settings(deadline=None, max_examples=200)
def test_opt_n_burden_model1(j, tev, prop_causal, prop_risk, alpha, power):
    """Test detectable TEV under Burden Model 1."""
    obj = RareVariantBurdenPower()
    opt_n = obj.opt_n_burden_model1(
        j=j,
        tev=tev,
        prop_causal=prop_causal,
        prop_risk=prop_risk,
        alpha=alpha,
        power=power,
    )
    if ~np.isnan(opt_n):
        assert opt_n > 1


@given(
    n=st.integers(min_value=1, max_value=1000000),
    ws=arrays(dtype=float, shape=100, elements=st.floats(0, 100)),
    ps=arrays(
        dtype=float,
        shape=100,
        elements=st.floats(1e-8, 1 - 1e-8, allow_nan=False, allow_infinity=False),
    ),
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
def test_ncp_burden_test_model2(ws, ps, jd, jp, n, tev):
    """Test estimation of NCP in second model for rv burden."""
    assume(jd + jp > 0)
    assume(ws.sum() > 0)
    obj = RareVariantBurdenPower()
    obj.ncp_burden_test_model2(ws=ws, ps=ps, jd=jd, jp=jp, n=n, tev=tev)


@given(
    n=st.integers(min_value=1),
    ws=arrays(dtype=float, shape=100, elements=st.floats(0, 100)),
    ps=arrays(
        dtype=float,
        shape=100,
        elements=st.floats(1e-8, 1 - 1e-8, allow_nan=False, allow_infinity=False),
    ),
    prop_causal=st.floats(min_value=1e-2, max_value=1.0),
    prop_risk=st.floats(min_value=0.5, max_value=1.0),
    tev=st.floats(
        min_value=1e-5,
        max_value=1,
        exclude_min=True,
        exclude_max=True,
        allow_infinity=False,
        allow_nan=False,
    ),
    alpha=st.floats(
        min_value=1e-32, max_value=0.5, allow_infinity=False, allow_nan=False
    ),
)
def test_power_burden_model2(ws, ps, n, prop_causal, prop_risk, tev, alpha):
    """Test of power under burden."""
    obj = RareVariantBurdenPower()
    obj.power_burden_model2(
        ws=ws,
        ps=ps,
        n=n,
        prop_causal=prop_causal,
        prop_risk=prop_risk,
        tev=tev,
        alpha=alpha,
    )


@given(
    n=st.integers(min_value=1),
    nreps=st.integers(min_value=1, max_value=100),
    test=st.sampled_from(["SKAT", "Calpha", "Hotelling"]),
    prop_causal=st.floats(min_value=1e-2, max_value=1.0),
    prop_risk=st.floats(min_value=0.5, max_value=1.0),
    tev=st.floats(
        min_value=1e-5,
        max_value=1,
        exclude_min=True,
        exclude_max=True,
        allow_infinity=False,
        allow_nan=False,
    ),
    alpha=st.floats(
        min_value=1e-32, max_value=0.5, allow_infinity=False, allow_nan=False
    ),
)
@settings(deadline=None, max_examples=200)
def test_power_burden_model2_real(n, nreps, test, prop_causal, prop_risk, tev, alpha):
    """Test of power under burden model 2 and real sampling."""
    obj = RareVariantBurdenPower()
    obj.power_burden_model2_real(
        n=n,
        nreps=nreps,
        test=test,
        prop_causal=prop_causal,
        prop_risk=prop_risk,
        tev=tev,
        alpha=alpha,
    )
