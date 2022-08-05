"""Testing module for GWAS power calculations."""
from hypothesis import given
from hypothesis import strategies as st

from qtl_power.rare_variants import RareVariantPower


@given(
    a=st.floats(
        min_value=0, max_value=1, exclude_min=True, exclude_max=True, allow_nan=False
    ),
    d=st.integers(min_value=1, max_value=1000),
    ncp=st.floats(min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False),
    ncp0=st.floats(min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False)
)
def test_llr_power(a, d, ncp, ncp0):
    """Test calculation of log-likelihood ratio calculation."""
    obj = RareVariantPower()
    obj.llr_power(alpha=a, df=d, ncp=ncp, ncp0=ncp0)
