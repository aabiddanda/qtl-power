"""Testing module for Extreme Phenotype power calculations."""
from hypothesis import given, settings
from hypothesis import strategies as st

from qtl_power.extreme_pheno import ExtremePhenotype


@given(
    maf=st.floats(
        min_value=0, max_value=0.5, exclude_min=True, exclude_max=True, allow_nan=False
    ),
    n=st.integers(min_value=1, max_value=1000),
    beta=st.floats(
        min_value=-1e3, max_value=1e3, allow_infinity=False, allow_nan=False
    ),
    seed=st.integers(min_value=1, max_value=100),
)
def test_sim_extreme_pheno(n, maf, beta, seed):
    """Test extreme phenotype association via simulation."""
    extreme_pheno = ExtremePhenotype()
    extreme_pheno.sim_extreme_pheno(n=n, maf=maf, beta=beta, seed=seed)


@given(
    n=st.integers(min_value=1, max_value=1000),
    maf=st.floats(
        min_value=0, max_value=0.5, exclude_min=True, exclude_max=True, allow_nan=False
    ),
    beta=st.floats(
        min_value=-1e3, max_value=1e3, allow_infinity=False, allow_nan=False
    ),
    niter=st.integers(min_value=10, max_value=1000),
    alpha=st.floats(
        min_value=1e-6, max_value=5e-2, allow_infinity=False, allow_nan=False
    ),
    q0=st.floats(min_value=5e-2, max_value=0.5, allow_infinity=False, allow_nan=False),
    q1=st.floats(min_value=5e-2, max_value=0.5, allow_infinity=False, allow_nan=False),
)
@settings(deadline=None, max_examples=200)
def test_est_power_extreme_pheno(n, maf, beta, niter, alpha, q0, q1):
    """Test estimation of power via simulation."""
    extreme_pheno = ExtremePhenotype()
    power = extreme_pheno.est_power_extreme_pheno(
        n=n, maf=maf, beta=beta, niter=niter, alpha=alpha, q0=q0, q1=q1
    )
    assert (power >= 0.0) and (power <= 1.0)
